"""End-to-end ingestion of the real knowledge base with the configured Ollama embedder and Qdrant server.

Writes to the collection `<QDRANT_COLLECTION>_e2e`, never to the working collection. The collection is
kept between runs: ingestion is incremental, so later runs only embed what changed.
"""

import pytest

from src.config import get_settings
from src.embeddings import OllamaEmbedder
from src.evaluation.retrieval import load_benchmark, run_benchmark
from src.ingestion.pipeline import discover_documents, embedding_hash, ingest, prepare_documents
from src.kb import DEFAULT_BENCHMARK
from src.retrieval.semantic import SemanticSearchRetriever
from src.vectorstore import QdrantStore
from tests.conftest import ollama_model_available, qdrant_reachable

pytestmark = pytest.mark.e2e

# Quality gate for the default configuration, set below the measured results (see README).
MIN_HIT_AT_5 = 0.90
MIN_MRR_AT_10 = 0.80


@pytest.fixture(scope="module")
def live():
    settings = get_settings()
    if not qdrant_reachable(settings):
        pytest.skip(f"Qdrant not reachable at {settings.qdrant_url}")
    if not ollama_model_available(settings):
        pytest.skip(f"Ollama model {settings.embedding_model} not available at {settings.embedding_base_url}")
    store = QdrantStore.from_settings(settings, collection=f"{settings.qdrant_collection}_e2e")
    embedder = OllamaEmbedder.from_settings(settings)
    report = ingest(settings, embedder=embedder, store=store)
    prepared = prepare_documents(discover_documents(settings.kb_raw_dir), settings.kb_raw_dir,
                                 settings.chunk_min_words, settings.chunk_max_words)
    return settings, store, embedder, report, prepared


def test_every_document_is_parsed_chunked_embedded_and_stored(live):
    settings, store, embedder, report, prepared = live

    assert len(report.documents) == len(discover_documents(settings.kb_raw_dir)) == len(prepared)
    assert all(d.chunks == d.points_stored > 0 for d in report.documents)
    stats = store.stats()
    assert stats.status == "green" and stats.distance == "Cosine"
    assert stats.vector_size == report.vector_size == len(embedder.embed_query("contrôle d'accès"))
    assert stats.points_count == report.points_in_collection == report.chunks_total
    assert store.doc_ids() == sorted(doc.doc_id for doc, _ in prepared)


def test_every_stored_point_has_the_chunk_payload(live):
    _, store, embedder, _, prepared = live
    chunks = [c for _, doc_chunks in prepared for c in doc_chunks]

    stored = {hit.id: hit.payload for hit in store.get([c.point_id for c in chunks])}

    assert stored == {c.point_id: {**c.payload(), "embedding_hash": embedding_hash(c, embedder)} for c in chunks}


def test_retrieval_benchmark_meets_the_quality_gate(live):
    _, store, embedder, _, _ = live
    report = run_benchmark(SemanticSearchRetriever(store, embedder), load_benchmark(DEFAULT_BENCHMARK))
    assert report.overall.hit_at_5 >= MIN_HIT_AT_5, report.overall
    assert report.overall.mrr_at_10 >= MIN_MRR_AT_10, report.overall


def test_payload_filters_restrict_results(live):
    _, store, embedder, _, _ = live
    query = embedder.embed_query("écart constaté lors de l'audit")

    for unit_type, _ in store.facet("unit_type"):
        hits = store.search(query, top_k=5, filters={"unit_type": unit_type})
        assert hits and all(h.payload["unit_type"] == unit_type for h in hits)

    reference, count = max(store.facet("references"), key=lambda item: item[1])
    hits = store.search(query, top_k=count + 5, filters={"references": reference})
    assert len(hits) == count and all(reference in h.payload["references"] for h in hits)


def test_reingestion_embeds_nothing_new(live):
    settings, store, embedder, first, _ = live
    second = ingest(settings, embedder=embedder, store=store)

    assert not second.collection_created
    assert (second.chunks_embedded, second.chunks_reused) == (0, first.chunks_total)
    assert store.count() == first.points_in_collection
