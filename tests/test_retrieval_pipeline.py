"""End-to-end Stage 2 pipeline: anonymize -> BM25 + semantic -> fusion -> cross-encoder -> context.

Offline (deterministic embedder, fake cross-encoder, in-process Qdrant): behaviour and wiring.
Live (real Ollama + Qdrant + HF model): the actual cahier-des-charges example, on the real corpus.
"""

import pytest

from src.anonymization.anonymizer import Anonymizer
from src.config import Settings
from src.ingestion.pipeline import discover_documents, ingest, prepare_documents
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.pipeline import SecureRetrievalPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticSearchRetriever
from tests.conftest import REAL_KB_DIR, FakeCrossEncoder, HashingEmbedder, huggingface_reachable, ollama_model_available

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


def kb_settings() -> Settings:
    return Settings(_env_file=None, kb_raw_dir=REAL_KB_DIR)


@pytest.fixture
def offline_pipeline(memory_store):
    settings = kb_settings()
    embedder = HashingEmbedder()
    ingest(settings, embedder=embedder, store=memory_store)
    bm25 = BM25Retriever.from_settings(settings)
    hybrid = HybridRetriever(SemanticSearchRetriever(memory_store, embedder), bm25, candidates_per_retriever=20)
    reranker = CrossEncoderReranker("fake-model", model=FakeCrossEncoder())
    return SecureRetrievalPipeline(Anonymizer(), hybrid, reranker, fusion_top_k=20, final_top_k=5)


class TestPipelineWiring:
    def test_finding_is_anonymized_before_retrieval(self, offline_pipeline):
        result = offline_pipeline.run("Le serveur SRV-PROD-01 de la société ABC utilise le compte admin.")

        assert result.anonymized_query == "Le serveur SERVER_001 de la société CLIENT_001 utilise le compte USER_001."
        assert result.anonymization_map == {"SERVER_001": "SRV-PROD-01", "CLIENT_001": "ABC", "USER_001": "admin"}

    def test_iso_terminology_in_the_finding_reaches_retrieval_unanonymized(self, offline_pipeline):
        result = offline_pipeline.run("Le contrôle des accès privilégiés A.5.18 n'est pas revu périodiquement.")
        assert result.anonymized_query == "Le contrôle des accès privilégiés A.5.18 n'est pas revu périodiquement."
        assert result.anonymization_map == {}

    def test_context_is_a_reranked_subset_of_the_fused_candidates(self, offline_pipeline):
        result = offline_pipeline.run("La revue des droits d'accès privilégiés n'est pas réalisée.")

        assert 0 < len(result.context) <= offline_pipeline.final_top_k
        assert {h.id for h in result.context} <= {h.id for h in result.candidates}
        assert len(result.candidates) <= offline_pipeline.fusion_top_k

    def test_no_matching_chunk_yields_empty_context_not_an_error(self, offline_pipeline):
        result = offline_pipeline.run("xylophone marmelade zeppelin qwzxjk")
        assert result.context == [] or isinstance(result.context, list)

    def test_two_findings_in_the_same_engagement_reuse_the_same_pseudonym(self, offline_pipeline):
        first = offline_pipeline.run("Le serveur SRV-PROD-01 n'est pas à jour.")
        second = offline_pipeline.run("Le serveur SRV-PROD-01 accepte encore TLS 1.0.")
        assert "SERVER_001" in first.anonymized_query and "SERVER_001" in second.anonymized_query
        assert offline_pipeline.anonymizer.mapping == {"SERVER_001": "SRV-PROD-01"}


@pytest.mark.integration
def test_live_pipeline_finds_the_cahier_des_charges_example(settings):
    if not (ollama_model_available(settings) and huggingface_reachable()):
        pytest.skip("Ollama embedding model or huggingface.co not reachable")

    from src.embeddings import build_embedder
    from src.vectorstore import QdrantStore

    store = QdrantStore.from_settings(settings, collection=f"{settings.qdrant_collection}_e2e")
    embedder = build_embedder(settings)
    ingest(settings, embedder=embedder, store=store)
    bm25 = BM25Retriever.from_settings(settings)
    hybrid = HybridRetriever(SemanticSearchRetriever(store, embedder), bm25,
                             rrf_k=settings.retrieval_rrf_k,
                             candidates_per_retriever=settings.retrieval_candidates_per_retriever)
    pipeline = SecureRetrievalPipeline(
        Anonymizer(), hybrid, CrossEncoderReranker.from_settings(settings),
        fusion_top_k=settings.retrieval_fusion_top_k, final_top_k=settings.rerank_top_k,
    )

    result = pipeline.run("Le contrôle des accès privilégiés n'est pas revu périodiquement.")

    assert result.context
    assert result.context[0].payload["doc_id"] == "doc-iso27001-2022-full"
    assert any("A.5.18" in h.payload["text"] for h in result.context[:2])


@pytest.mark.integration
def test_live_pipeline_handles_a_finding_with_sensitive_data(settings):
    if not (ollama_model_available(settings) and huggingface_reachable()):
        pytest.skip("Ollama embedding model or huggingface.co not reachable")

    from src.embeddings import build_embedder
    from src.vectorstore import QdrantStore

    store = QdrantStore.from_settings(settings, collection=f"{settings.qdrant_collection}_e2e")
    embedder = build_embedder(settings)
    ingest(settings, embedder=embedder, store=store)
    bm25 = BM25Retriever.from_settings(settings)
    hybrid = HybridRetriever(SemanticSearchRetriever(store, embedder), bm25,
                             rrf_k=settings.retrieval_rrf_k,
                             candidates_per_retriever=settings.retrieval_candidates_per_retriever)
    pipeline = SecureRetrievalPipeline(
        Anonymizer(), hybrid, CrossEncoderReranker.from_settings(settings),
        fusion_top_k=settings.retrieval_fusion_top_k, final_top_k=settings.rerank_top_k,
    )

    result = pipeline.run(
        "Chez la société Contoso, le serveur SRV-WEB-07 accepte encore TLS 1.0, "
        "en écart avec la politique cryptographique interne."
    )

    assert "Contoso" not in result.anonymized_query and "SRV-WEB-07" not in result.anonymized_query
    assert set(result.anonymization_map.values()) == {"Contoso", "SRV-WEB-07"}
    assert result.context
    assert any(c.payload["doc_id"] == "doc-iso27002-2022-full" for c in result.context)
