"""Client document import through the API: upload -> sidecar review -> confirm. No LLM call (the
embedding step is real Ollama + Qdrant, same as the CLI's own live tests) -- these use a small CSV
so the offline-vs-live split matches the rest of the project: import/propose is pure Python, only
`confirm` needs the real embedding model and Qdrant.
"""

import io

import pytest

from tests.conftest import huggingface_reachable, ollama_model_available
from src.config import get_settings

pytestmark = pytest.mark.integration


def _create_session(client, headers):
    body = dict(
        title="Rapport ISO/IEC 27001 - Contoso", client_name="Contoso", client_aliases=["Contoso SA"],
        scope="Perimetre de test", standards=["ISO/IEC 27001:2022"],
    )
    r = client.post("/sessions", headers=headers, json=body)
    assert r.status_code == 201, r.text
    return r.json()


CSV_CONTENT = b"Nom,Responsable\nCompte admin,Marie Dupont\n"


class TestImportAndSidecar:
    def test_import_writes_an_anonymized_sidecar(self, client, headers):
        session = _create_session(client, headers)
        r = client.post(
            f"/sessions/{session['id']}/documents", headers=headers,
            files={"file": ("comptes.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
            data={"doc_type": "previous_report"},
        )
        assert r.status_code == 201, r.text
        record = r.json()
        assert record["status"] == "pending_review"
        doc_id = record["doc_id"]

        r = client.get(f"/sessions/{session['id']}/documents/{doc_id}/sidecar", headers=headers)
        assert r.status_code == 200
        content = r.json()["content"]
        assert "Marie Dupont" not in content
        assert "PERSON_001" in content

    def test_list_documents_reflects_the_import(self, client, headers):
        session = _create_session(client, headers)
        client.post(
            f"/sessions/{session['id']}/documents", headers=headers,
            files={"file": ("comptes.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
            data={"doc_type": "previous_report"},
        )
        r = client.get(f"/sessions/{session['id']}/documents", headers=headers)
        assert r.status_code == 200 and len(r.json()) == 1

    def test_unsupported_file_type_is_a_clean_400(self, client, headers):
        session = _create_session(client, headers)
        r = client.post(
            f"/sessions/{session['id']}/documents", headers=headers,
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"doc_type": "previous_report"},
        )
        assert r.status_code == 400

    def test_sidecar_of_unknown_document_is_404(self, client, headers):
        session = _create_session(client, headers)
        r = client.get(f"/sessions/{session['id']}/documents/doc-nope/sidecar", headers=headers)
        assert r.status_code == 404

    def test_confirming_an_unknown_document_is_a_clean_400(self, client, headers):
        session = _create_session(client, headers)
        r = client.post(f"/sessions/{session['id']}/documents/doc-nope/confirm", headers=headers)
        assert r.status_code == 400

    def test_another_auditeur_cannot_import_into_someone_else_s_session(self, client, db_session):
        from api.models_orm import Role
        from tests.test_api.conftest import auth_headers, make_user

        make_user(db_session, username="alice")
        make_user(db_session, username="bob", role=Role.AUDITEUR)
        session = _create_session(client, auth_headers(client, "alice"))
        r = client.post(
            f"/sessions/{session['id']}/documents", headers=auth_headers(client, "bob"),
            files={"file": ("comptes.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
            data={"doc_type": "previous_report"},
        )
        assert r.status_code == 403


@pytest.mark.e2e
class TestConfirmLive:
    @pytest.fixture(autouse=True)
    def _skip_if_unreachable(self):
        settings = get_settings()
        if not (ollama_model_available(settings) and huggingface_reachable()):
            pytest.skip("Embedding Ollama model or huggingface.co not reachable")

    def test_confirm_chunks_embeds_and_stores(self, client, headers):
        session = _create_session(client, headers)
        r = client.post(
            f"/sessions/{session['id']}/documents", headers=headers,
            files={"file": ("comptes.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
            data={"doc_type": "previous_report"},
        )
        doc_id = r.json()["doc_id"]

        r = client.post(f"/sessions/{session['id']}/documents/{doc_id}/confirm", headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "confirmed"
        assert r.json()["chunk_count"] > 0

        # cleanup: close the session's Qdrant collection
        client.post(f"/sessions/{session['id']}/close", headers=headers)
