"""Stage 2 end-to-end pipeline.

    Auditor input -> Anonymization -> BM25 + Semantic -> Hybrid Fusion -> Cross-Encoder -> context

Produces the ranked, safe-to-use context chunks a future generation stage grounds its answer in.
No generation, no deanonymization of a generated answer: both belong to a later stage. `mapping`
is returned so a later stage can still restore sensitive values once it produces final output.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.anonymization.anonymizer import Anonymizer
from src.config import Settings, get_settings
from src.embeddings import Embedder, build_embedder
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticSearchRetriever
from src.vectorstore import QdrantStore, SearchHit


@dataclass
class RerankedRetriever:
    """`Retriever` adapter: fuse a wide candidate set, then rerank it down to `top_k`.

    Lets the reranked pipeline be evaluated by `evaluation.retrieval.run_benchmark` exactly like
    the plain semantic or hybrid retrievers, through the same `.retrieve(query, top_k)` call.
    """

    hybrid: HybridRetriever
    reranker: CrossEncoderReranker
    fusion_top_k: int = 20

    def retrieve(self, query: str, top_k: int) -> List[SearchHit]:
        candidates = self.hybrid.retrieve(query, top_k=max(self.fusion_top_k, top_k))
        return self.reranker.rerank(query, candidates, top_k=top_k)


@dataclass
class RetrievalResult:
    anonymized_query: str
    anonymization_map: Dict[str, str]
    candidates: List[SearchHit]  # fused candidates, before reranking (for inspection)
    context: List[SearchHit]     # final reranked chunks


@dataclass
class SecureRetrievalPipeline:
    anonymizer: Anonymizer
    hybrid: HybridRetriever
    reranker: CrossEncoderReranker
    fusion_top_k: int = 20
    final_top_k: int = 5

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        store: QdrantStore | None = None,
        embedder: Embedder | None = None,
        bm25: BM25Retriever | None = None,
        anonymizer: Anonymizer | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> "SecureRetrievalPipeline":
        settings = settings or get_settings()
        embedder = embedder or build_embedder(settings)
        store = store or QdrantStore.from_settings(settings)
        bm25 = bm25 or BM25Retriever.from_settings(settings)
        hybrid = HybridRetriever(
            SemanticSearchRetriever(store, embedder), bm25,
            rrf_k=settings.retrieval_rrf_k, candidates_per_retriever=settings.retrieval_candidates_per_retriever,
        )
        return cls(
            anonymizer=anonymizer or Anonymizer(),
            hybrid=hybrid,
            reranker=reranker or CrossEncoderReranker.from_settings(settings),
            fusion_top_k=settings.retrieval_fusion_top_k,
            final_top_k=settings.rerank_top_k,
        )

    def run(self, auditor_text: str) -> RetrievalResult:
        anonymized, mapping = self.anonymizer.anonymize(auditor_text)
        candidates = self.hybrid.retrieve(anonymized, top_k=self.fusion_top_k)
        context = self.reranker.rerank(anonymized, candidates, top_k=self.final_top_k)
        return RetrievalResult(anonymized, mapping, candidates, context)
