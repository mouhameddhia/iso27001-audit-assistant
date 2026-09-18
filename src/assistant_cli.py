"""Full-assistant command line.

    python -m src.assistant_cli analyze "Le contrôle des accès privilégiés n'est pas revu périodiquement." [--lang en]
    python -m src.assistant_cli evaluate-grounding [--details]

    python -m src.assistant_cli session new --title T --client C --scope S \
        --standard "ISO/IEC 27001:2022" [--standard ...] --team "Nom Prénom:Lead Auditor" [--team ...] -o session.json
    python -m src.assistant_cli session add-finding session.json "observation" [--lang fr]
    python -m src.assistant_cli session list session.json
    python -m src.assistant_cli session review session.json <index> --approve|--reject [--severity majeure|mineure] [--edit "..."]
    python -m src.assistant_cli session set-recommendation session.json --recommend|--no-recommend [--decided-by "..."]
    python -m src.assistant_cli session report session.json --format docx|pdf [-o path]

    python -m src.assistant_cli session import-document session.json rapport_precedent.pdf --doc-type previous_report
    python -m src.assistant_cli session confirm-import session.json <doc_id>
    python -m src.assistant_cli session list-documents session.json
    python -m src.assistant_cli session close session.json
"""

import argparse
import logging
import sys
from pathlib import Path

from src.audit_assistant import AuditAssistant
from src.config import PROJECT_ROOT, get_settings
from src.documents.extractors import ExtractionError
from src.documents.ingestion import DocumentImportError, close_session_collection, confirm_import, propose_import, session_collection_name
from src.documents.retrieval import build_document_retriever
from src.evaluation.generation import load_cases, run_grounding_eval
from src.generation.generator import FindingGenerator
from src.generation.validation import GroundingValidator
from src.report.builder import ReportError, build_report
from src.report.docx_renderer import render_docx
from src.report.pdf_renderer import render_pdf
from src.report.session import (
    AuditSession, CertificationDecision, ReviewDecision, SessionError, SessionMetadata, Severity, TeamMember,
    load_session, save_session,
)
from src.retrieval.pipeline import SecureRetrievalPipeline

DEFAULT_GROUNDING_EVAL = PROJECT_ROOT / "evaluation" / "grounding_eval.json"


def cmd_analyze(args: argparse.Namespace) -> None:
    assistant = AuditAssistant.from_settings()
    finding = assistant.analyze(args.observation, language=args.lang)
    print(finding.model_dump_json(indent=2))


def cmd_evaluate_grounding(args: argparse.Namespace) -> None:
    settings = get_settings()
    cases = load_cases(args.cases)
    retrieval = SecureRetrievalPipeline.from_settings(settings)
    generator = FindingGenerator.from_settings(settings)
    validator = GroundingValidator()

    report = run_grounding_eval(cases, retrieval, generator, validator)

    print(f"{report.total} cases | generation model={settings.generation_model}\n")
    print(f"Expectation match rate (cited a reference iff expected): {report.expectation_match_rate:.0%}")
    print(f"Cases where the model tried to cite an unsupported reference (before validation): "
          f"{report.cases_with_raw_hallucination}/{report.total}")
    print(f"Cases with an unsupported reference in the FINAL output (must be 0): "
          f"{report.cases_with_final_unsupported_reference}/{report.total}")
    print(f"Cases with a weakly-backed citation NOT flagged for review (must be 0): "
          f"{report.cases_with_weak_citation_not_flagged}/{report.total}")
    non_conformite_checked = [r for r in report.results if r.case.expect_non_conformite is not None]
    if non_conformite_checked:
        print(f"Compliant observations fabricated into a non-conformité (category F, reported not gated -- "
              f"always forced to human review, see README): "
              f"{report.cases_with_unexpected_non_conformite}/{len(non_conformite_checked)}")
    print("\nBy category:")
    for category, stats in report.by_category().items():
        print(f"  {category}: n={stats['n']} expectation_match_rate={stats['expectation_match_rate']:.0%}")

    if args.details:
        print()
        for r in report.results:
            status = "OK" if r.matched_expectation else "MISMATCH"
            print(f"[{status}] {r.case.id:6} ({r.case.category}) expect_reference={r.case.expect_reference} "
                  f"-> iso_reference={r.validated.iso_reference} evidence_status={r.validated.evidence_status.value} "
                  f"requires_review={r.validated.requires_human_review} confidence={r.validated.confidence}")
            if r.raw_unsupported_references:
                print(f"         model tried to cite (dropped by validation): {r.raw_unsupported_references}")


def _parse_team(entries: list) -> list:
    team = []
    for entry in entries:
        name, _, role = entry.partition(":")
        team.append(TeamMember(name=name.strip(), role=role.strip() or "Auditeur"))
    return team


def cmd_session_new(args: argparse.Namespace) -> None:
    session = AuditSession(metadata=SessionMetadata(
        title=args.title, client_name=args.client, client_aliases=args.alias, scope=args.scope,
        standards=args.standard, audit_team=_parse_team(args.team), reference=args.reference,
        start_date=args.start_date, end_date=args.end_date,
    ))
    save_session(session, args.output)
    print(f"Session created: {args.output} (session_id={session.metadata.session_id})")


def cmd_session_add_finding(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    settings = get_settings()
    # The session's own persistent anonymizer, not a fresh one: each CLI invocation is a separate
    # process, and without this, pseudonym numbering would silently restart on every call.
    anonymizer = session.anonymizer
    assistant = AuditAssistant.from_settings(settings, anonymizer=anonymizer)
    # None when the session has no confirmed imported documents -- behaviour is then unchanged.
    document_retriever = build_document_retriever(session, settings)
    reviewed = session.add_finding(assistant, args.observation, language=args.lang, document_retriever=document_retriever)
    session.sync_anonymizer(anonymizer)
    save_session(session, args.session)
    index = len(session.findings) - 1
    finding_type = reviewed.finding.finding_type.value if reviewed.finding.finding_type else "?"
    print(f"[{index}] {finding_type} | evidence_status={reviewed.finding.evidence_status.value} | "
          f"requires_review={reviewed.finding.requires_human_review} | refs={reviewed.finding.iso_reference}")
    if reviewed.finding.document_sources:
        print(f"  (used client document context: {len(reviewed.finding.document_sources)} chunk(s) -- review required)")
    print(reviewed.finding.finding)


def cmd_session_list(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    if not session.findings:
        print("No findings in this session yet.")
        return
    for i, r in enumerate(session.findings):
        severity = f" [{r.severity.value}]" if r.severity else ""
        finding_type = r.finding.finding_type.value if r.finding.finding_type else "?"
        print(f"[{i}] {r.decision.value}{severity} | {finding_type} | refs={r.finding.iso_reference} | {r.text[:80]}")


def cmd_session_review(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    decision = ReviewDecision.APPROVED if args.approve else ReviewDecision.REJECTED
    severity = Severity(args.severity) if args.severity else None
    try:
        reviewed = session.review(
            args.index, decision, reviewer=args.reviewer, severity=severity, edited_text=args.edit
        )
    except SessionError as exc:
        sys.exit(str(exc))
    save_session(session, args.session)
    print(f"[{args.index}] -> {reviewed.decision.value}")


def cmd_session_set_recommendation(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    decision = CertificationDecision.RECOMMENDS if args.recommend else CertificationDecision.DOES_NOT_RECOMMEND
    session.set_certification_decision(decision, decided_by=args.decided_by or "?")
    save_session(session, args.session)
    print(f"Certification decision recorded: {decision.value} (by {session.certification_decided_by})")


def cmd_session_report(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    try:
        report = build_report(session)
    except ReportError as exc:
        sys.exit(str(exc))

    settings = get_settings()
    output = args.output or settings.report_output_dir / f"{session.metadata.reference or 'rapport'}.{args.format}"
    output = Path(output)
    (render_docx if args.format == "docx" else render_pdf)(report, output)
    print(f"Report written: {output}")


def cmd_session_import_document(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    settings = get_settings()
    try:
        record = propose_import(session, args.path, args.doc_type, settings=settings)
    except (DocumentImportError, ExtractionError) as exc:
        sys.exit(str(exc))
    save_session(session, args.session)
    print(f"Imported (pending review): {record.doc_id}")
    print(f"Review the anonymized content, then run:")
    print(f"  python -m src.assistant_cli session confirm-import {args.session} {record.doc_id}")
    print(f"Sidecar file: {record.sidecar_path}")


def cmd_session_confirm_import(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    settings = get_settings()
    try:
        record = confirm_import(session, args.doc_id, settings=settings)
    except DocumentImportError as exc:
        sys.exit(str(exc))
    save_session(session, args.session)
    print(f"Confirmed: {record.doc_id} ({record.chunk_count} chunks stored)")


def cmd_session_list_documents(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    if not session.imported_documents:
        print("No imported documents in this session yet.")
        return
    for d in session.imported_documents:
        print(f"{d.doc_id} | {d.status.value} | {d.doc_type} | {d.filename} | chunks={d.chunk_count}")


def cmd_session_close(args: argparse.Namespace) -> None:
    session = load_session(args.session)
    settings = get_settings()
    collection = session_collection_name(session, settings)
    close_session_collection(session, settings=settings)
    print(f"Deleted this session's document collection: {collection}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.assistant_cli", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="run the full pipeline on one auditor observation")
    p_analyze.add_argument("observation")
    p_analyze.add_argument("--lang", default="fr", choices=["fr", "en"])
    p_analyze.set_defaults(func=cmd_analyze)

    p_eval = sub.add_parser("evaluate-grounding", help="run the hallucination-resistance evaluation set")
    p_eval.add_argument("--cases", default=DEFAULT_GROUNDING_EVAL)
    p_eval.add_argument("--details", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate_grounding)

    p_session = sub.add_parser("session", help="manage an audit session (multiple findings -> a report)")
    session_sub = p_session.add_subparsers(dest="session_command", required=True)

    p_new = session_sub.add_parser("new", help="create a new session")
    p_new.add_argument("--title", required=True)
    p_new.add_argument("--client", required=True)
    p_new.add_argument("--alias", action="append", default=[],
                        help="other name(s) the client is known by, repeatable")
    p_new.add_argument("--scope", required=True)
    p_new.add_argument("--standard", action="append", default=[], help="repeatable")
    p_new.add_argument("--team", action="append", default=[], help="repeatable, Name:Role")
    p_new.add_argument("--reference")
    p_new.add_argument("--start-date")
    p_new.add_argument("--end-date")
    p_new.add_argument("-o", "--output", required=True, type=Path)
    p_new.set_defaults(func=cmd_session_new)

    p_add = session_sub.add_parser("add-finding", help="run the pipeline on one observation, add it pending review")
    p_add.add_argument("session", type=Path)
    p_add.add_argument("observation")
    p_add.add_argument("--lang", default="fr", choices=["fr", "en"])
    p_add.set_defaults(func=cmd_session_add_finding)

    p_list = session_sub.add_parser("list", help="list a session's findings and their review status")
    p_list.add_argument("session", type=Path)
    p_list.set_defaults(func=cmd_session_list)

    p_review = session_sub.add_parser("review", help="approve or reject one finding")
    p_review.add_argument("session", type=Path)
    p_review.add_argument("index", type=int)
    decision_group = p_review.add_mutually_exclusive_group(required=True)
    decision_group.add_argument("--approve", action="store_true")
    decision_group.add_argument("--reject", action="store_true")
    p_review.add_argument("--severity", choices=["majeure", "mineure"], help="non-conformité findings only")
    p_review.add_argument("--edit", help="replacement finding text")
    p_review.add_argument("--reviewer")
    p_review.set_defaults(func=cmd_session_review)

    p_recommendation = session_sub.add_parser(
        "set-recommendation", help="record the Auditeur Principal's certification recommendation"
    )
    p_recommendation.add_argument("session", type=Path)
    recommendation_group = p_recommendation.add_mutually_exclusive_group(required=True)
    recommendation_group.add_argument("--recommend", action="store_true")
    recommendation_group.add_argument("--no-recommend", dest="recommend", action="store_false")
    p_recommendation.add_argument("--decided-by", help="name of the Auditeur Principal recording this decision")
    p_recommendation.set_defaults(func=cmd_session_set_recommendation)

    p_report = session_sub.add_parser("report", help="render the session's approved findings to a document")
    p_report.add_argument("session", type=Path)
    p_report.add_argument("--format", choices=["docx", "pdf"], required=True)
    p_report.add_argument("-o", "--output", type=Path)
    p_report.set_defaults(func=cmd_session_report)

    p_import = session_sub.add_parser(
        "import-document", help="propose importing a client document (extract, anonymize, write a reviewable sidecar)"
    )
    p_import.add_argument("session", type=Path)
    p_import.add_argument("path", type=Path)
    p_import.add_argument("--doc-type", required=True,
                           help="e.g. previous_report, soa, risk_analysis, procedure, policy")
    p_import.set_defaults(func=cmd_session_import_document)

    p_confirm = session_sub.add_parser("confirm-import", help="chunk/embed/store a reviewed sidecar file")
    p_confirm.add_argument("session", type=Path)
    p_confirm.add_argument("doc_id")
    p_confirm.set_defaults(func=cmd_session_confirm_import)

    p_docs = session_sub.add_parser("list-documents", help="list a session's imported documents")
    p_docs.add_argument("session", type=Path)
    p_docs.set_defaults(func=cmd_session_list_documents)

    p_close = session_sub.add_parser("close", help="delete this session's client-document Qdrant collection")
    p_close.add_argument("session", type=Path)
    p_close.set_defaults(func=cmd_session_close)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
