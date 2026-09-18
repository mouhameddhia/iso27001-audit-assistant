"""
hybrid_retriever.py

Combines the semantic retriever and the BM25 retriever using Reciprocal
Rank Fusion (RRF): a simple, well-known way to merge two ranked lists
without needing to compare their raw scores directly (cosine similarity
and BM25 scores are not on the same scale, so we can't just add them).

RRF score for a chunk = sum over each retriever of  1 / (k + rank),
where `rank` is that chunk's 1-based position in that retriever's
results, and `k` is a small constant (60 is the common default) that
softens the impact of rank 1 vs rank 2.
"""

from typing import List
from src.ingestion.document_loader import Chunk
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """Retrieves with both retrievers, then fuses the two ranked lists."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def index(self, chunks: List[Chunk]) -> None:
        """Build both underlying indexes over the same chunks."""
        self.semantic_retriever.index(chunks)
        self.bm25_retriever.index(chunks)

    def retrieve(self, query: str, top_k: int = 5, candidates_per_retriever: int = 20) -> List[Chunk]:
        """Return the `top_k` chunks after fusing semantic + BM25 rankings."""
        semantic_results = self.semantic_retriever.retrieve(query, top_k=candidates_per_retriever)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidates_per_retriever)

        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}

        for ranked_list in (semantic_results, bm25_results):
            for rank, chunk in enumerate(ranked_list, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunk_by_id[chunk.id] = chunk

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_by_id[cid] for cid in ranked_ids[:top_k]]
