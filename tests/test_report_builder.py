"""build_report(): the deterministic session -> Report transformation (cahier des charges §4.4).
No LLM call is possible here -- there is nothing to mock; every assertion is on real data flow.
"""

import pytest

from src.generation.models import FindingType
from src.report.builder import NOT_PROVIDED, ReportError, build_report
from src.report.models import BulletList, Paragraph, Table
from src.report.session import (
    AuditSession, CertificationDecision, ReviewDecision, ReviewedFinding, SessionMetadata, Severity, TeamMember,
)
from tests.conftest import make_audit_finding

SECTION_TITLES = [
    "Informations générales", "Portée", "Équipe d'audit", "Constats",
    "Non-conformités", "Observations", "Opportunités d'amélioration", "Conclusion",
]


def make_metadata(**overrides) -> SessionMetadata:
    defaults = dict(
        title="Rapport ISO/IEC 27001 - Contoso", client_name="Contoso", scope="Périmètre de test",
        standards=["ISO/IEC 27001:2022"], audit_team=[TeamMember(name="A. Martin", role="Lead Auditor")],
        reference="MISSION-001",
    )
    defaults.update(overrides)
    return SessionMetadata(**defaults)


def approved_session(*findings_and_severities) -> AuditSession:
    """findings_and_severities: (AuditFinding, severity_or_None) pairs, all approved."""
    session = AuditSession(metadata=make_metadata())
    for finding, severity in findings_and_severities:
        session.findings.append(ReviewedFinding(finding=finding))
        session.review(len(session.findings) - 1, ReviewDecision.APPROVED, severity=severity)
    return session


def section(report, title):
    return next(s for s in report.sections if s.title == title)


def flatten_text(section) -> str:
    parts = []
    for block in section.blocks:
        if isinstance(block, Paragraph):
            parts.append(block.text)
        elif isinstance(block, BulletList):
            parts.extend(block.items)
        elif isinstance(block, Table):
            parts.extend(cell for row in block.rows for cell in row)
    return "\n".join(parts)


class TestGuards:
    def test_no_approved_findings_raises(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(ReviewedFinding(finding=make_audit_finding()))  # left pending
        with pytest.raises(ReportError, match="nothing to report"):
            build_report(session)

    def test_unclassified_non_conformite_raises(self):
        session = approved_session((make_audit_finding(finding_type=FindingType.NON_CONFORMITE), None))
        with pytest.raises(ReportError, match="no severity set"):
            build_report(session)

    def test_report_has_every_cahier_des_charges_section_in_order(self):
        session = approved_session((make_audit_finding(text="Un constat."), None))
        report = build_report(session)
        assert [s.title for s in report.sections] == SECTION_TITLES


class TestSectionMapping:
    def test_constat_goes_only_in_constats(self):
        report = build_report(approved_session((make_audit_finding(FindingType.CONSTAT, text="Constat X"), None)))
        assert "Constat X" in flatten_text(section(report, "Constats"))
        for title in ["Non-conformités", "Observations", "Opportunités d'amélioration"]:
            assert "Constat X" not in flatten_text(section(report, title))

    def test_non_conformite_carries_its_severity(self):
        report = build_report(approved_session(
            (make_audit_finding(FindingType.NON_CONFORMITE, text="Accès non revus"), Severity.MAJEURE)
        ))
        text = flatten_text(section(report, "Non-conformités"))
        assert "Majeure" in text and "Accès non revus" in text

    def test_observation_and_opportunite_go_to_their_own_sections(self):
        report = build_report(approved_session(
            (make_audit_finding(FindingType.OBSERVATION, text="Une observation."), None),
            (make_audit_finding(FindingType.OPPORTUNITE_AMELIORATION, text="Une opportunité."), None),
        ))
        assert "Une observation." in flatten_text(section(report, "Observations"))
        assert "Une opportunité." in flatten_text(section(report, "Opportunités d'amélioration"))
        assert "Une observation." not in flatten_text(section(report, "Opportunités d'amélioration"))

    def test_empty_section_says_so_rather_than_being_blank(self):
        report = build_report(approved_session((make_audit_finding(FindingType.CONSTAT), None)))
        assert "Aucun élément" in flatten_text(section(report, "Observations"))

    def test_pending_and_rejected_findings_never_reach_any_section(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(ReviewedFinding(finding=make_audit_finding(text="Approuvé.")))
        session.findings.append(ReviewedFinding(finding=make_audit_finding(text="En attente.")))
        session.findings.append(ReviewedFinding(finding=make_audit_finding(text="Rejeté.")))
        session.review(0, ReviewDecision.APPROVED)
        session.review(2, ReviewDecision.REJECTED)
        # index 1 stays pending

        report = build_report(session)
        full_text = "\n".join(flatten_text(s) for s in report.sections)
        assert "Approuvé." in full_text
        assert "En attente." not in full_text
        assert "Rejeté." not in full_text

    def test_edited_text_is_used_instead_of_the_original(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(ReviewedFinding(finding=make_audit_finding(text="Brouillon IA.")))
        session.review(0, ReviewDecision.APPROVED, edited_text="Texte corrigé par l'auditeur.")

        text = flatten_text(section(build_report(session), "Constats"))
        assert "Texte corrigé par l'auditeur." in text
        assert "Brouillon IA." not in text

    def test_iso_references_appear_in_the_finding_header(self):
        report = build_report(approved_session(
            (make_audit_finding(FindingType.CONSTAT, iso_reference=["ISO/IEC 27001 A.5.18"]), None)
        ))
        assert "ISO/IEC 27001 A.5.18" in flatten_text(section(report, "Constats"))

    def test_the_auditor_s_original_observation_appears_in_the_report(self):
        """Regression test for a real question raised live: the drafted `finding` text is the
        model's paraphrase; the firm's own internal policy (SECTION 04) requires a distinct
        "description factuelle et neutre de la situation observée" element in every constat, and a
        reviewer needs to be able to trace the report back to what the auditor actually wrote, not
        just the AI's rendition of it."""
        session = AuditSession(metadata=make_metadata())
        session.findings.append(ReviewedFinding(finding=make_audit_finding(
            observation="12 utilisateurs disposent d'un accès en écriture non justifié.",
        )))
        session.review(0, ReviewDecision.APPROVED)
        text = flatten_text(section(build_report(session), "Constats"))
        assert "12 utilisateurs disposent d'un accès en écriture non justifié." in text


class TestInformationsGeneralesPorteeEquipe:
    def test_informations_generales_has_client_reference_and_standards(self):
        report = build_report(approved_session((make_audit_finding(), None)))
        text = flatten_text(section(report, "Informations générales"))
        assert "Contoso" in text and "MISSION-001" in text and "ISO/IEC 27001:2022" in text

    def test_portee_shows_the_session_scope(self):
        session = approved_session((make_audit_finding(), None))
        text = flatten_text(section(build_report(session), "Portée"))
        assert session.metadata.scope in text

    def test_equipe_lists_team_members(self):
        report = build_report(approved_session((make_audit_finding(), None)))
        text = flatten_text(section(report, "Équipe d'audit"))
        assert "A. Martin" in text and "Lead Auditor" in text

    def test_missing_team_is_stated_explicitly(self):
        session = AuditSession(metadata=make_metadata(audit_team=[]))
        session.findings.append(ReviewedFinding(finding=make_audit_finding()))
        session.review(0, ReviewDecision.APPROVED)
        text = flatten_text(section(build_report(session), "Équipe d'audit"))
        assert "Non renseigné" in text

    def test_missing_reference_is_a_professional_placeholder_not_a_bare_dash(self):
        """Regression test for a real, live-reported issue: a missing mission reference rendered
        as a bare "-", which reads as a formatting bug rather than "not provided"."""
        session = AuditSession(metadata=make_metadata(reference=None))
        session.findings.append(ReviewedFinding(finding=make_audit_finding()))
        session.review(0, ReviewDecision.APPROVED)
        table = next(b for b in section(build_report(session), "Informations générales").blocks if isinstance(b, Table))
        assert dict(table.rows)["Référence de mission"] == NOT_PROVIDED

    def test_missing_audit_period_is_a_professional_placeholder_not_question_marks(self):
        """Regression test: a missing start/end date used to render as the literal "? - ?"."""
        session = AuditSession(metadata=make_metadata(start_date=None, end_date=None))
        session.findings.append(ReviewedFinding(finding=make_audit_finding()))
        session.review(0, ReviewDecision.APPROVED)
        table = next(b for b in section(build_report(session), "Informations générales").blocks if isinstance(b, Table))
        assert dict(table.rows)["Période d'audit"] == NOT_PROVIDED

    def test_partially_known_audit_period_shows_the_known_side_and_flags_the_rest(self):
        session = AuditSession(metadata=make_metadata(start_date="2026-01-06", end_date=None))
        session.findings.append(ReviewedFinding(finding=make_audit_finding()))
        session.review(0, ReviewDecision.APPROVED)
        table = next(b for b in section(build_report(session), "Informations générales").blocks if isinstance(b, Table))
        period = dict(table.rows)["Période d'audit"]
        assert "2026-01-06" in period and NOT_PROVIDED in period and "?" not in period

    def test_report_title_and_subtitle_come_from_metadata(self):
        session = approved_session((make_audit_finding(), None))
        report = build_report(session)
        assert report.title == session.metadata.title
        assert session.metadata.client_name in report.subtitle
        assert session.metadata.reference in report.subtitle


class TestConclusion:
    def test_counts_match_the_approved_severities_and_types(self):
        session = approved_session(
            (make_audit_finding(FindingType.NON_CONFORMITE), Severity.MAJEURE),
            (make_audit_finding(FindingType.NON_CONFORMITE), Severity.MAJEURE),
            (make_audit_finding(FindingType.NON_CONFORMITE), Severity.MINEURE),
            (make_audit_finding(FindingType.OBSERVATION), None),
            (make_audit_finding(FindingType.OPPORTUNITE_AMELIORATION), None),
        )
        conclusion = section(build_report(session), "Conclusion")
        table = next(b for b in conclusion.blocks if isinstance(b, Table))
        counts = dict(table.rows)
        assert counts["Non-conformités majeures"] == "2"
        assert counts["Non-conformités mineures"] == "1"
        assert counts["Observations"] == "1"
        assert counts["Opportunités d'amélioration"] == "1"

    def test_rejected_and_pending_findings_are_excluded_from_the_counts(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(ReviewedFinding(finding=make_audit_finding(FindingType.NON_CONFORMITE)))
        session.findings.append(ReviewedFinding(finding=make_audit_finding(FindingType.NON_CONFORMITE)))
        session.review(0, ReviewDecision.APPROVED, severity=Severity.MAJEURE)
        session.review(1, ReviewDecision.REJECTED)

        table = next(b for b in section(build_report(session), "Conclusion").blocks if isinstance(b, Table))
        assert dict(table.rows)["Non-conformités majeures"] == "1"

    def test_referenced_controls_are_listed_and_deduplicated(self):
        session = approved_session(
            (make_audit_finding(FindingType.CONSTAT, iso_reference=["ISO/IEC 27001 A.5.18"]), None),
            (make_audit_finding(FindingType.OBSERVATION, iso_reference=["ISO/IEC 27001 A.5.18", "ISO/IEC 27001 A.8.8"]), None),
        )
        text = flatten_text(section(build_report(session), "Conclusion"))
        assert text.count("ISO/IEC 27001 A.5.18") == 1
        assert "ISO/IEC 27001 A.8.8" in text

    def test_unset_certification_decision_is_a_professional_pending_note_never_a_raw_bracket(self):
        """Regression test for a real, live-reported issue: the report used to always contain a raw,
        unresolved "[recommande / ne recommande pas]" template placeholder in a document meant to
        be a deliverable. The AI must never invent this decision -- but it also must never ship a
        template artifact; the correct behaviour is an explicit "pending" statement."""
        text = flatten_text(section(build_report(approved_session((make_audit_finding(), None))), "Conclusion"))
        assert "[recommande / ne recommande pas]" not in text
        assert "reste à confirmer par l'Auditeur Principal" in text

    def test_certification_decision_recommends_is_rendered_when_explicitly_set(self):
        session = approved_session((make_audit_finding(), None))
        session.set_certification_decision(CertificationDecision.RECOMMENDS, decided_by="J. Dupont")
        text = flatten_text(section(build_report(session), "Conclusion"))
        assert "l'équipe d'audit recommande le maintien de la certification" in text
        assert "ne recommande pas" not in text
        assert "J. Dupont" in text

    def test_certification_decision_does_not_recommend_is_rendered_when_explicitly_set(self):
        session = approved_session((make_audit_finding(), None))
        session.set_certification_decision(CertificationDecision.DOES_NOT_RECOMMEND, decided_by="J. Dupont")
        text = flatten_text(section(build_report(session), "Conclusion"))
        assert "l'équipe d'audit ne recommande pas le maintien de la certification" in text

    def test_certification_decision_is_never_defaulted_by_build_report_itself(self):
        """build_report() never sets a decision on the auditor's behalf -- only reads whatever
        AuditSession.certification_decision already is."""
        session = approved_session((make_audit_finding(), None))
        assert session.certification_decision is None
        build_report(session)
        assert session.certification_decision is None
