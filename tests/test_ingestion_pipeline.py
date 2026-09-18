"""Full ingestion pipeline without external services: deterministic embedder + in-process Qdrant."""

import pytest

from src.config import Settings
from src.embeddings import EmbeddingError
from src.ingestion.models import RECORD
from src.ingestion.pipeline import IngestionError, discover_documents, embedding_hash, ingest, prepare_documents
from src.vectorstore import CollectionConfigError
from tests.conftest import REAL_KB_DIR, HashingEmbedder, kb_document

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


def kb_settings(raw_dir) -> Settings:
    """Default settings (ignoring any local .env) pointed at `raw_dir`."""
    return Settings(_env_file=None, kb_raw_dir=raw_dir)


def write_doc(folder, name, doc_id, sections):
    (folder / name).write_text(
        kb_document(doc_id, f"ISO/IEC 27001:2022 - {doc_id}", "iso27001 / ISO 27001", sections), encoding="utf-8"
    )


NC = ("NC-2026-001 (Mineure)\nRéférence : A.5.18 - Droits d'accès\n"
      "Énoncé : La revue des accès privilégiés n'est pas réalisée.\nRisque : Droits injustifiés.")


@pytest.fixture
def small_kb(tmp_path):
    write_doc(tmp_path, "a.txt", "doc-a", {
        1: ("INFORMATIONS GÉNÉRALES", "La norme définit les exigences d'un SMSI certifiable."),
        5: ("NON-CONFORMITÉS", f"Exemples :\n\n{NC}"),
    })
    write_doc(tmp_path, "b.txt", "doc-b", {1: ("PORTÉE", "Le périmètre couvre les sauvegardes et la restauration.")})
    return tmp_path


def test_discovery_finds_supported_files_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["a.txt", "sub/b.md", "c.docx", "d.pdf"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert [p.relative_to(tmp_path).as_posix() for p in discover_documents(tmp_path)] == ["a.txt", "sub/b.md"]


def test_missing_or_empty_knowledge_base_folder_is_an_error(tmp_path, memory_store):
    with pytest.raises(IngestionError, match="not found"):
        discover_documents(tmp_path / "missing")
    with pytest.raises(IngestionError, match="No supported documents"):
        ingest(kb_settings(tmp_path), embedder=HashingEmbedder(), store=memory_store)


def test_real_knowledge_base_is_ingested_with_full_payloads(memory_store):
    settings = kb_settings(REAL_KB_DIR)
    embedder = HashingEmbedder()
    prepared = prepare_documents(discover_documents(REAL_KB_DIR), REAL_KB_DIR,
                                 settings.chunk_min_words, settings.chunk_max_words)
    chunks = [c for _, doc_chunks in prepared for c in doc_chunks]

    report = ingest(settings, embedder=embedder, store=memory_store)

    assert len(report.documents) == len(discover_documents(REAL_KB_DIR))
    assert report.chunks_total == len(chunks) == report.points_in_collection == memory_store.count()
    assert report.chunks_embedded == len(chunks) and report.chunks_reused == 0
    assert all(d.chunks == d.points_stored for d in report.documents)
    stored = {hit.id: hit.payload for hit in memory_store.get([c.point_id for c in chunks])}
    assert stored == {c.point_id: {**c.payload(), "embedding_hash": embedding_hash(c, embedder)} for c in chunks}


def test_search_filters_on_unit_type_and_references(small_kb, memory_store):
    embedder = HashingEmbedder()
    ingest(kb_settings(small_kb), embedder=embedder, store=memory_store)
    query = embedder.embed_query("revue des accès privilégiés")

    [record] = memory_store.search(query, top_k=5, filters={"unit_type": RECORD})
    assert record.payload["record_id"] == "NC-2026-001"
    assert record.payload["heading"] == "ISO/IEC 27001:2022 › Non-conformités › Exemples"
    by_reference = memory_store.search(query, top_k=5, filters={"references": "ISO/IEC 27001 A.5.18"})
    assert [h.id for h in by_reference] == [record.id]


def test_reingestion_embeds_nothing_when_nothing_changed(small_kb, memory_store):
    embedder = HashingEmbedder()
    first = ingest(kb_settings(small_kb), embedder=embedder, store=memory_store)
    calls = embedder.calls

    second = ingest(kb_settings(small_kb), embedder=embedder, store=memory_store)

    assert not second.collection_created
    assert (second.chunks_embedded, second.chunks_reused) == (0, first.chunks_total)
    assert embedder.calls == calls
    assert memory_store.count() == first.chunks_total


def test_only_changed_chunks_are_embedded_again(small_kb, memory_store):
    embedder = HashingEmbedder()
    ingest(kb_settings(small_kb), embedder=embedder, store=memory_store)
    write_doc(small_kb, "b.txt", "doc-b", {1: ("PORTÉE", "Le périmètre couvre aussi la journalisation.")})
    embedder.embedded_texts.clear()

    report = ingest(kb_settings(small_kb), embedder=embedder, store=memory_store)

    assert report.chunks_embedded == 1
    assert embedder.embedded_texts == ["ISO/IEC 27001:2022 › Portée\n\nLe périmètre couvre aussi la journalisation."]
    hits = memory_store.search(embedder.embed_query("journalisation"), top_k=1)
    assert "journalisation" in hits[0].payload["text"]


def test_a_different_embedding_model_re_embeds_everything(small_kb, memory_store):
    first = ingest(kb_settings(small_kb), embedder=HashingEmbedder(model="model-a"), store=memory_store)
    second = ingest(kb_settings(small_kb), embedder=HashingEmbedder(model="model-b"), store=memory_store)
    assert second.chunks_embedded == first.chunks_total


def test_changed_and_removed_documents_are_synchronised(small_kb, memory_store):
    long_body = "\n\n".join(f"Paragraphe {i}. " + " ".join([f"mot{i}"] * 150) for i in range(4))
    write_doc(small_kb, "a.txt", "doc-a", {1: ("CONSTATS", long_body)})
    ingest(kb_settings(small_kb), embedder=HashingEmbedder(), store=memory_store)
    assert memory_store.count({"doc_id": "doc-a"}) == 4

    write_doc(small_kb, "a.txt", "doc-a", {1: ("CONSTATS", "Section désormais courte.")})
    (small_kb / "b.txt").unlink()
    report = ingest(kb_settings(small_kb), embedder=HashingEmbedder(), store=memory_store)

    assert memory_store.doc_ids() == ["doc-a"]
    assert memory_store.count() == report.chunks_total == 1


def test_duplicate_document_ids_are_rejected(tmp_path, memory_store):
    write_doc(tmp_path, "a.txt", "doc-same", {1: ("CONSTATS", "Un.")})
    write_doc(tmp_path, "b.txt", "doc-same", {1: ("CONSTATS", "Deux.")})
    with pytest.raises(IngestionError, match="Duplicate doc_id 'doc-same'"):
        ingest(kb_settings(tmp_path), embedder=HashingEmbedder(), store=memory_store)


def test_embedding_failure_leaves_the_collection_untouched(small_kb, memory_store):
    ingest(kb_settings(small_kb), embedder=HashingEmbedder(), store=memory_store)
    before = sorted(h.payload["text"] for h in memory_store.search([1.0] * 128, top_k=100))
    write_doc(small_kb, "b.txt", "doc-b", {1: ("PORTÉE", "Texte modifié qui doit être embarqué à nouveau.")})

    class BrokenEmbedder(HashingEmbedder):
        def embed_documents(self, texts):
            raise EmbeddingError("ollama down")

    with pytest.raises(EmbeddingError):
        ingest(kb_settings(small_kb), embedder=BrokenEmbedder(), store=memory_store)
    assert sorted(h.payload["text"] for h in memory_store.search([1.0] * 128, top_k=100)) == before


def test_changing_embedding_dimension_requires_recreate(small_kb, memory_store):
    ingest(kb_settings(small_kb), embedder=HashingEmbedder(dim=128), store=memory_store)

    with pytest.raises(CollectionConfigError, match="--recreate"):
        ingest(kb_settings(small_kb), embedder=HashingEmbedder(dim=64), store=memory_store)

    report = ingest(kb_settings(small_kb), embedder=HashingEmbedder(dim=64), store=memory_store, recreate=True)
    assert report.vector_size == 64 and memory_store.stats().vector_size == 64
    assert report.chunks_embedded == report.chunks_total
