"""Session-scoped document import: propose (extract+anonymize, write sidecar) -> human review ->
confirm (re-read the sidecar, chunk, embed, store). All offline: an in-memory Qdrant client and
the deterministic HashingEmbedder from conftest stand in for the real services.
"""

import pytest
from qdrant_client import QdrantClient

from src.documents.ingestion import (
    DocumentImportError, close_session_collection, confirm_import, propose_import, session_chunks,
    session_collection_name,
)
from src.report.session import AuditSession, ImportStatus, SessionMetadata
from src.vectorstore import QdrantStore
from tests.conftest import HashingEmbedder


def make_session(**overrides) -> AuditSession:
    defaults = dict(title="T", client_name="Northwind Traders", scope="S", standards=["ISO/IEC 27001:2022"])
    defaults.update(overrides)
    return AuditSession(metadata=SessionMetadata(**defaults))


def make_store(session: AuditSession) -> QdrantStore:
    return QdrantStore(QdrantClient(location=":memory:"), collection=session_collection_name(session))


@pytest.fixture
def settings(settings, tmp_path):
    return settings.model_copy(update={"document_import_dir": tmp_path / "imports"})


class TestProposeImport:
    def test_writes_an_anonymized_sidecar_and_marks_pending_review(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport_precedent.csv"
        source.write_text("Nom,Responsable\nCompte admin,Marie Dupont\n", encoding="utf-8")

        record = propose_import(session, source, doc_type="previous_report", settings=settings)

        assert record.status == ImportStatus.PENDING_REVIEW
        assert record.doc_id == "doc-rapport-precedent"
        assert session.imported_documents == [record]

        sidecar_text = record.sidecar_path
        from pathlib import Path
        content = Path(sidecar_text).read_text(encoding="utf-8")
        assert "Marie Dupont" not in content
        assert "PERSON_001" in content

    def test_updates_the_session_s_persistent_anonymization_mapping(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Nom,Responsable\nCompte admin,Marie Dupont\n", encoding="utf-8")

        propose_import(session, source, doc_type="previous_report", settings=settings)

        assert "Marie Dupont" in session.anonymization_mapping.values()

    def test_a_client_alias_with_no_nearby_keyword_is_still_caught(self, settings, tmp_path):
        session = make_session(client_aliases=["NWT"])
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nAudit realise pour NWT en 2025.\n", encoding="utf-8")

        record = propose_import(session, source, doc_type="previous_report", settings=settings)

        from pathlib import Path
        content = Path(record.sidecar_path).read_text(encoding="utf-8")
        assert "NWT" not in content

    def test_reimporting_the_same_filename_replaces_the_pending_record(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Nom\nA\n", encoding="utf-8")
        propose_import(session, source, doc_type="previous_report", settings=settings)
        propose_import(session, source, doc_type="previous_report", settings=settings)
        assert len(session.imported_documents) == 1

    def test_empty_document_raises(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "empty.csv"
        source.write_text("", encoding="utf-8")
        with pytest.raises(DocumentImportError):
            propose_import(session, source, doc_type="previous_report", settings=settings)


class TestConfirmImport:
    def test_requires_a_prior_propose(self, settings):
        session = make_session()
        with pytest.raises(DocumentImportError, match="No imported document"):
            confirm_import(session, "doc-nope", settings=settings, embedder=HashingEmbedder(), store=make_store(session))

    def test_confirms_precisely_the_reviewed_sidecar_not_the_original_again(self, settings, tmp_path):
        """The auditor may hand-edit the sidecar during review; confirm must reflect that edit,
        proving it never goes back to re-read the original document."""
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nConstat original non pertinent.\n", encoding="utf-8")
        record = propose_import(session, source, doc_type="previous_report", settings=settings)

        from pathlib import Path
        Path(record.sidecar_path).write_text(
            "Notes: Le pare-feu applicatif accepte encore des connexions TLS 1.0.", encoding="utf-8",
        )

        store = make_store(session)
        embedder = HashingEmbedder()
        confirm_import(session, record.doc_id, settings=settings, embedder=embedder, store=store)

        assert any("TLS 1.0" in t for t in embedder.embedded_texts)
        assert not any("original non pertinent" in t for t in embedder.embedded_texts)

    def test_embeds_and_stores_chunks_and_marks_confirmed(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text(
            "Notes\nLe pare-feu applicatif accepte encore des connexions TLS 1.0 sur l'API exposee.\n",
            encoding="utf-8",
        )
        record = propose_import(session, source, doc_type="previous_report", settings=settings)
        store = make_store(session)

        confirmed = confirm_import(session, record.doc_id, settings=settings, embedder=HashingEmbedder(), store=store)

        assert confirmed.status == ImportStatus.CONFIRMED
        assert confirmed.chunk_count > 0
        assert confirmed.confirmed_at is not None
        assert store.count() == confirmed.chunk_count

    def test_missing_sidecar_raises(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nUn constat.\n", encoding="utf-8")
        record = propose_import(session, source, doc_type="previous_report", settings=settings)
        from pathlib import Path
        Path(record.sidecar_path).unlink()

        with pytest.raises(DocumentImportError):
            confirm_import(session, record.doc_id, settings=settings, embedder=HashingEmbedder(), store=make_store(session))

    def test_reconfirming_an_updated_sidecar_replaces_stale_chunks_not_duplicates(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nPremiere version du constat sur les acces.\n", encoding="utf-8")
        record = propose_import(session, source, doc_type="previous_report", settings=settings)
        store = make_store(session)
        embedder = HashingEmbedder()
        confirm_import(session, record.doc_id, settings=settings, embedder=embedder, store=store)
        first_count = store.count()

        from pathlib import Path
        Path(record.sidecar_path).write_text(
            "Notes\nDeuxieme version, beaucoup plus detaillee, du constat sur les acces privilegies "
            "qui ne sont pas revus periodiquement selon la politique interne du client.",
            encoding="utf-8",
        )
        confirm_import(session, record.doc_id, settings=settings, embedder=embedder, store=store)

        assert store.count(filters={"doc_id": record.doc_id}) == store.count()  # only this doc's points remain


class TestSessionChunksAndIsolation:
    def test_session_chunks_only_include_confirmed_documents(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nUn constat quelconque assez long pour faire un chunk valide.\n", encoding="utf-8")
        propose_import(session, source, doc_type="previous_report", settings=settings)  # never confirmed

        assert session_chunks(session, settings=settings) == []

    def test_confirmed_document_chunks_are_returned(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nUn constat quelconque assez long pour faire un chunk valide.\n", encoding="utf-8")
        record = propose_import(session, source, doc_type="previous_report", settings=settings)
        confirm_import(session, record.doc_id, settings=settings, embedder=HashingEmbedder(), store=make_store(session))

        chunks = session_chunks(session, settings=settings)
        assert len(chunks) == record.chunk_count

    def test_two_sessions_get_different_collection_names(self, settings):
        a, b = make_session(), make_session()
        assert session_collection_name(a, settings) != session_collection_name(b, settings)

    def test_closing_a_session_deletes_its_collection(self, settings, tmp_path):
        session = make_session()
        source = tmp_path / "rapport.csv"
        source.write_text("Notes\nUn constat quelconque assez long pour faire un chunk valide.\n", encoding="utf-8")
        record = propose_import(session, source, doc_type="previous_report", settings=settings)
        store = make_store(session)
        confirm_import(session, record.doc_id, settings=settings, embedder=HashingEmbedder(), store=store)
        assert store.collection_exists()

        close_session_collection(session, settings=settings, store=store)

        assert not store.collection_exists()
