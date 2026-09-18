"""AuditSession: collects AuditFindings from one engagement, with a human review gate.

`AuditSession.add_finding()` must only call `assistant.analyze()` -- it does not re-implement
retrieval or generation, so a fake AuditAssistant (mirroring test_audit_assistant.py's fakes) is
enough to test the wiring.
"""

import pytest

from src.generation.models import FindingType
from src.report.session import (
    AuditSession, ReviewDecision, ReviewedFinding, SessionError, SessionMetadata, Severity, TeamMember,
    load_session, save_session,
)
from tests.conftest import make_audit_finding


class FakeAuditAssistant:
    def __init__(self, finding):
        self.finding = finding
        self.calls = []

    def analyze(self, observation, language="fr", document_retriever=None):
        self.calls.append((observation, language, document_retriever))
        return self.finding


def make_metadata(**overrides) -> SessionMetadata:
    defaults = dict(
        title="Rapport ISO/IEC 27001 - Contoso", client_name="Contoso", scope="Périmètre de test",
        standards=["ISO/IEC 27001:2022"], audit_team=[TeamMember(name="A. Martin", role="Lead Auditor")],
        reference="MISSION-001",
    )
    defaults.update(overrides)
    return SessionMetadata(**defaults)


class TestAddFinding:
    def test_add_finding_calls_the_assistant_and_appends_pending(self):
        session = AuditSession(metadata=make_metadata())
        assistant = FakeAuditAssistant(make_audit_finding(text="Un constat."))

        reviewed = session.add_finding(assistant, "observation brute", language="en")

        assert assistant.calls == [("observation brute", "en", None)]
        assert len(session.findings) == 1
        assert session.findings[0] is reviewed
        assert reviewed.decision == ReviewDecision.PENDING
        assert reviewed.finding.finding == "Un constat."

    def test_document_retriever_is_forwarded_to_the_assistant(self):
        session = AuditSession(metadata=make_metadata())
        assistant = FakeAuditAssistant(make_audit_finding())
        sentinel_retriever = object()

        session.add_finding(assistant, "obs", document_retriever=sentinel_retriever)

        assert assistant.calls == [("obs", "fr", sentinel_retriever)]

    def test_multiple_findings_accumulate_in_order(self):
        session = AuditSession(metadata=make_metadata())
        assistant = FakeAuditAssistant(make_audit_finding())
        session.add_finding(assistant, "obs 1")
        session.add_finding(assistant, "obs 2")
        assert len(session.findings) == 2


class TestReview:
    def test_approve_sets_decision_reviewer_and_timestamp(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding()))

        reviewed = session.review(0, ReviewDecision.APPROVED, reviewer="A. Martin")

        assert reviewed.decision == ReviewDecision.APPROVED
        assert reviewed.reviewer == "A. Martin"
        assert reviewed.reviewed_at is not None

    def test_reject_excludes_from_approved(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding()))
        session.review(0, ReviewDecision.REJECTED)
        assert session.approved == []
        assert session.rejected_count == 1

    def test_out_of_range_index_raises(self):
        session = AuditSession(metadata=make_metadata())
        with pytest.raises(SessionError, match="No finding at index"):
            session.review(0, ReviewDecision.APPROVED)

    def test_severity_allowed_on_non_conformite(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(finding_type=FindingType.NON_CONFORMITE)))
        reviewed = session.review(0, ReviewDecision.APPROVED, severity=Severity.MAJEURE)
        assert reviewed.severity == Severity.MAJEURE

    @pytest.mark.parametrize("finding_type", [FindingType.CONSTAT, FindingType.OBSERVATION, FindingType.OPPORTUNITE_AMELIORATION])
    def test_severity_rejected_on_other_finding_types(self, finding_type):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(finding_type=finding_type)))
        with pytest.raises(SessionError, match="Severity only applies"):
            session.review(0, ReviewDecision.APPROVED, severity=Severity.MINEURE)

    def test_edited_text_overrides_the_original_in_text_property(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(text="Texte original.")))
        session.review(0, ReviewDecision.APPROVED, edited_text="Texte corrigé par l'auditeur.")
        assert session.findings[0].text == "Texte corrigé par l'auditeur."

    def test_text_property_falls_back_to_original_when_not_edited(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(text="Texte original.")))
        session.review(0, ReviewDecision.APPROVED)
        assert session.findings[0].text == "Texte original."

    def test_re_reviewing_without_severity_does_not_clear_a_previously_set_one(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(finding_type=FindingType.NON_CONFORMITE)))
        session.review(0, ReviewDecision.APPROVED, severity=Severity.MAJEURE)
        session.review(0, ReviewDecision.APPROVED, reviewer="Second reviewer")  # no severity passed this time
        assert session.findings[0].severity == Severity.MAJEURE

    def test_re_reviewing_without_edited_text_does_not_clear_a_previous_edit(self):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(text="Original.")))
        session.review(0, ReviewDecision.APPROVED, edited_text="Corrigé.")
        session.review(0, ReviewDecision.REJECTED)
        assert session.findings[0].edited_text == "Corrigé."


class TestCounts:
    def test_approved_pending_and_rejected_counts(self):
        session = AuditSession(metadata=make_metadata())
        for _ in range(4):
            session.findings.append(_reviewed(make_audit_finding()))
        session.review(0, ReviewDecision.APPROVED)
        session.review(1, ReviewDecision.APPROVED)
        session.review(2, ReviewDecision.REJECTED)
        # index 3 stays pending

        assert len(session.approved) == 2
        assert session.pending_count == 1
        assert session.rejected_count == 1


class TestPersistence:
    def test_save_and_load_round_trip_preserves_everything(self, tmp_path):
        session = AuditSession(metadata=make_metadata())
        session.findings.append(_reviewed(make_audit_finding(
            finding_type=FindingType.NON_CONFORMITE, iso_reference=["ISO/IEC 27001 A.5.18"],
        )))
        session.review(0, ReviewDecision.APPROVED, severity=Severity.MAJEURE, reviewer="A. Martin", edited_text="Corrigé.")

        path = tmp_path / "session.json"
        save_session(session, path)
        reloaded = load_session(path)

        assert reloaded.metadata == session.metadata
        assert reloaded.findings[0].decision == ReviewDecision.APPROVED
        assert reloaded.findings[0].severity == Severity.MAJEURE
        assert reloaded.findings[0].edited_text == "Corrigé."
        assert reloaded.findings[0].finding.iso_reference == ["ISO/IEC 27001 A.5.18"]

    def test_load_of_an_empty_session_has_no_findings(self, tmp_path):
        session = AuditSession(metadata=make_metadata())
        path = tmp_path / "empty.json"
        save_session(session, path)
        assert load_session(path).findings == []


def _reviewed(finding):
    return ReviewedFinding(finding=finding)
