"""Deterministic session -> Report transformation: the cahier des charges §4.4 section mapping.

No LLM call here. Only `session.approved` findings are used; pending/rejected findings never
reach the output. The conclusion's certification recommendation is rendered from the Auditeur
Principal's own explicit decision (`AuditSession.certification_decision`, set via
`set_certification_decision`) -- never an unresolved bracketed placeholder and never a value the
report invents on the auditor's behalf.
"""

from typing import List, Optional

from src.generation.models import FindingType
from src.report.models import BulletList, Paragraph, Report, ReportSection, Table
from src.report.session import AuditSession, CertificationDecision, ReviewedFinding, Severity

# A professional stand-in for missing metadata -- never a bare "-" or "?", which reads as a
# rendering bug rather than "this auditor-provided field was left blank" (reported live).
NOT_PROVIDED = "Non renseigné"


class ReportError(RuntimeError):
    pass


def _period(start: Optional[str], end: Optional[str]) -> str:
    if not start and not end:
        return NOT_PROVIDED
    return f"{start or NOT_PROVIDED} - {end or NOT_PROVIDED}"


def _finding_blocks(reviewed: ReviewedFinding, index: int, prefix: str) -> List[Paragraph]:
    refs = ", ".join(reviewed.finding.iso_reference) or "aucune référence spécifique"
    header = f"{prefix}-{index:03d}"
    if reviewed.severity:
        header += f" ({reviewed.severity.value.capitalize()})"
    # The internal policy's own constat template (retrievable in the knowledge base, SECTION 04)
    # requires "une description factuelle et neutre de la situation observée" as its own element,
    # distinct from the drafted finding -- the auditor's original wording, not the AI's paraphrase,
    # so a reviewer can always trace the report back to what was actually observed.
    blocks = [
        Paragraph(f"{header} — {refs}", bold=True),
        Paragraph(f"Observation initiale : {reviewed.finding.observation}"),
        Paragraph(reviewed.text),
    ]
    if reviewed.finding.requirement:
        blocks.append(Paragraph(f"Exigence associée : {reviewed.finding.requirement}"))
    if reviewed.finding.justification:
        blocks.append(Paragraph(f"Justification : {reviewed.finding.justification}"))
    if reviewed.finding.risk:
        blocks.append(Paragraph(f"Risque : {reviewed.finding.risk}"))
    if reviewed.finding.recommendation:
        blocks.append(Paragraph(f"Recommandation : {reviewed.finding.recommendation}"))
    return blocks


def _section_for(findings: List[ReviewedFinding], title: str, prefix: str) -> ReportSection:
    section = ReportSection(title=title)
    if not findings:
        section.blocks.append(Paragraph("Aucun élément de ce type dans cette mission."))
        return section
    for i, reviewed in enumerate(findings, start=1):
        section.blocks.extend(_finding_blocks(reviewed, i, prefix))
    return section


def _conclusion_section(session: AuditSession, approved: List[ReviewedFinding]) -> ReportSection:
    non_conformites = [f for f in approved if f.finding.finding_type == FindingType.NON_CONFORMITE]
    majeures = sum(f.severity == Severity.MAJEURE for f in non_conformites)
    mineures = sum(f.severity == Severity.MINEURE for f in non_conformites)
    observations = sum(f.finding.finding_type == FindingType.OBSERVATION for f in approved)
    opportunites = sum(f.finding.finding_type == FindingType.OPPORTUNITE_AMELIORATION for f in approved)
    references = sorted({ref for f in approved for ref in f.finding.iso_reference})

    section = ReportSection(title="Conclusion")
    section.blocks.append(Paragraph(f"L'audit a porté sur le périmètre suivant : {session.metadata.scope}"))
    section.blocks.append(
        Paragraph(f"Référentiel(s) audité(s) : {', '.join(session.metadata.standards) or NOT_PROVIDED}.")
    )
    section.blocks.append(Table(
        headers=["Catégorie", "Nombre"],
        rows=[
            ["Non-conformités majeures", str(majeures)],
            ["Non-conformités mineures", str(mineures)],
            ["Observations", str(observations)],
            ["Opportunités d'amélioration", str(opportunites)],
        ],
    ))
    if references:
        section.blocks.append(Paragraph("Contrôles/clauses référencés dans ce rapport :"))
        section.blocks.append(BulletList(references))

    if session.certification_decision == CertificationDecision.RECOMMENDS:
        decision_clause = "l'équipe d'audit recommande le maintien de la certification dans le périmètre audité"
    elif session.certification_decision == CertificationDecision.DOES_NOT_RECOMMEND:
        decision_clause = "l'équipe d'audit ne recommande pas le maintien de la certification dans le périmètre audité"
    else:
        # Never invented or defaulted: the AI must not produce a certification recommendation the
        # Auditeur Principal has not explicitly given (cahier des charges: validation humaine
        # obligatoire). Rendered as an explicit, professional pending state, not a raw
        # "[recommande / ne recommande pas]" template placeholder left unresolved in a deliverable.
        decision_clause = "la décision de recommandation de certification reste à confirmer par l'Auditeur Principal"
    section.blocks.append(Paragraph(
        "Sous réserve de la mise en œuvre d'actions correctives efficaces sur les non-conformités "
        f"relevées, {decision_clause}. Cette conclusion doit être validée et contresignée par "
        "l'Auditeur Principal avant transmission au client."
    ))
    if session.certification_decision is not None:
        section.blocks.append(Paragraph(
            f"Décision enregistrée par {session.certification_decided_by or NOT_PROVIDED} "
            f"le {session.certification_decided_at or NOT_PROVIDED}."
        ))
    return section


def build_report(session: AuditSession) -> Report:
    approved = session.approved
    if not approved:
        raise ReportError(
            "No approved finding in this session -- nothing to report "
            f"({session.pending_count} pending, {session.rejected_count} rejected)."
        )

    non_conformites = [f for f in approved if f.finding.finding_type == FindingType.NON_CONFORMITE]
    unclassified = [f for f in non_conformites if f.severity is None]
    if unclassified:
        raise ReportError(
            f"{len(unclassified)} approved non-conformité(s) have no severity set; "
            "classify majeure/mineure (session review --severity) before generating the report."
        )

    meta = session.metadata
    info_section = ReportSection(title="Informations générales", blocks=[Table(
        headers=["Champ", "Valeur"],
        rows=[
            ["Client", meta.client_name],
            ["Référence de mission", meta.reference or NOT_PROVIDED],
            ["Référentiel(s)", ", ".join(meta.standards) or NOT_PROVIDED],
            ["Période d'audit", _period(meta.start_date, meta.end_date)],
        ],
    )])
    scope_section = ReportSection(title="Portée", blocks=[Paragraph(meta.scope)])
    team_section = ReportSection(title="Équipe d'audit", blocks=[
        Table(headers=["Nom", "Rôle"], rows=[[m.name, m.role] for m in meta.audit_team])
        if meta.audit_team else Paragraph(f"Équipe d'audit : {NOT_PROVIDED}.")
    ])

    constats = [f for f in approved if f.finding.finding_type == FindingType.CONSTAT]
    observations = [f for f in approved if f.finding.finding_type == FindingType.OBSERVATION]
    opportunites = [f for f in approved if f.finding.finding_type == FindingType.OPPORTUNITE_AMELIORATION]

    return Report(
        title=meta.title,
        subtitle=f"{meta.client_name} — {meta.reference}" if meta.reference else meta.client_name,
        sections=[
            info_section,
            scope_section,
            team_section,
            _section_for(constats, "Constats", "C"),
            _section_for(non_conformites, "Non-conformités", "NC"),
            _section_for(observations, "Observations", "OBS"),
            _section_for(opportunites, "Opportunités d'amélioration", "OPP"),
            _conclusion_section(session, approved),
        ],
    )
