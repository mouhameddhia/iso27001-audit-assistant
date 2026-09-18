"""Session CRUD, visibility (least privilege), review workflow, and report-generation guard
errors. No LLM call anywhere here -- `add_finding` (which does call the real pipeline) is covered
live in tests/test_api/test_e2e.py.
"""

import pytest

from api.models_orm import Role
from tests.test_api.conftest import auth_headers, make_user

pytestmark = pytest.mark.integration


def _create_session(client, headers, **overrides):
    body = dict(
        title="Rapport ISO/IEC 27001 - Contoso", client_name="Contoso", scope="Perimetre de test",
        standards=["ISO/IEC 27001:2022"], audit_team=[{"name": "A. Martin", "role": "Lead Auditor"}],
        reference="MISSION-001",
    )
    body.update(overrides)
    r = client.post("/sessions", headers=headers, json=body)
    assert r.status_code == 201, r.text
    return r.json()


class TestCreateAndList:
    def test_create_session_returns_a_summary(self, client, headers):
        summary = _create_session(client, headers)
        assert summary["title"] == "Rapport ISO/IEC 27001 - Contoso"
        assert summary["pending_count"] == 0 and summary["approved_count"] == 0

    def test_auditeur_only_sees_their_own_sessions(self, client, db_session):
        make_user(db_session, username="alice", role=Role.AUDITEUR)
        make_user(db_session, username="bob", role=Role.AUDITEUR)
        _create_session(client, auth_headers(client, "alice"))
        _create_session(client, auth_headers(client, "bob"))

        r = client.get("/sessions", headers=auth_headers(client, "alice"))
        assert r.status_code == 200 and len(r.json()) == 1

    def test_chef_equipe_sees_every_session(self, client, db_session):
        make_user(db_session, username="alice", role=Role.AUDITEUR)
        make_user(db_session, username="bob", role=Role.AUDITEUR)
        make_user(db_session, username="chef", role=Role.CHEF_EQUIPE)
        _create_session(client, auth_headers(client, "alice"))
        _create_session(client, auth_headers(client, "bob"))

        r = client.get("/sessions", headers=auth_headers(client, "chef"))
        assert r.status_code == 200 and len(r.json()) == 2


class TestGetSession:
    def test_owner_can_view_their_session(self, client, headers):
        summary = _create_session(client, headers)
        r = client.get(f"/sessions/{summary['id']}", headers=headers)
        assert r.status_code == 200
        assert r.json()["metadata"]["client_name"] == "Contoso"

    def test_the_anonymization_mapping_is_never_sent_over_the_wire(self, client, headers):
        """Internal implementation detail with real, deanonymized values in it -- never needed by
        a client and never sent, even to the session's own owner."""
        summary = _create_session(client, headers)
        r = client.get(f"/sessions/{summary['id']}", headers=headers)
        assert "anonymization_mapping" not in r.json()

    def test_another_auditeur_cannot_view_someone_else_s_session(self, client, db_session):
        make_user(db_session, username="alice", role=Role.AUDITEUR)
        make_user(db_session, username="bob", role=Role.AUDITEUR)
        summary = _create_session(client, auth_headers(client, "alice"))
        r = client.get(f"/sessions/{summary['id']}", headers=auth_headers(client, "bob"))
        assert r.status_code == 403

    def test_unknown_session_is_404(self, client, headers):
        r = client.get("/sessions/does-not-exist", headers=headers)
        assert r.status_code == 404


class TestReview:
    def test_reviewing_an_out_of_range_finding_is_a_clean_400(self, client, headers):
        summary = _create_session(client, headers)
        r = client.post(f"/sessions/{summary['id']}/findings/0/review", headers=headers, json={"approve": True})
        assert r.status_code == 400


class TestReport:
    def test_generating_a_report_with_no_approved_findings_is_a_clean_400(self, client, headers):
        summary = _create_session(client, headers)
        r = client.get(f"/sessions/{summary['id']}/report", headers=headers, params={"format": "docx"})
        assert r.status_code == 400

    def test_unknown_format_is_a_clean_400(self, client, headers):
        summary = _create_session(client, headers)
        r = client.get(f"/sessions/{summary['id']}/report", headers=headers, params={"format": "xml"})
        assert r.status_code == 400


class TestClose:
    def test_closing_a_session_marks_it_closed(self, client, headers):
        summary = _create_session(client, headers)
        r = client.post(f"/sessions/{summary['id']}/close", headers=headers)
        assert r.status_code == 200 and r.json()["status"] == "closed"


class TestCertificationDecision:
    def test_auditeur_cannot_set_the_certification_decision(self, client, headers):
        summary = _create_session(client, headers)
        r = client.post(f"/sessions/{summary['id']}/certification-decision", headers=headers, json={"recommends": True})
        assert r.status_code == 403

    def test_chef_equipe_can_set_it_and_it_is_reflected_on_the_session(self, client, db_session):
        make_user(db_session, username="chef", role=Role.CHEF_EQUIPE)
        make_user(db_session, username="alice", role=Role.AUDITEUR)
        chef_headers = auth_headers(client, "chef")
        summary = _create_session(client, auth_headers(client, "alice"))

        r = client.post(
            f"/sessions/{summary['id']}/certification-decision", headers=chef_headers,
            json={"recommends": True, "decided_by": "Chef Auditeur"},
        )
        assert r.status_code == 200
        assert r.json()["certification_decision"] == "recommande"
        assert r.json()["certification_decided_by"] == "Chef Auditeur"

    def test_does_not_recommend_is_stored_distinctly(self, client, db_session):
        make_user(db_session, username="chef", role=Role.CHEF_EQUIPE)
        summary = _create_session(client, auth_headers(client, "chef"))
        r = client.post(
            f"/sessions/{summary['id']}/certification-decision", headers=auth_headers(client, "chef"),
            json={"recommends": False},
        )
        assert r.status_code == 200
        assert r.json()["certification_decision"] == "ne_recommande_pas"
