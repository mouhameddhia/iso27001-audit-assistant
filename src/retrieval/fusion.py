"""Hybrid fusion: combine ranked candidate lists from independent retrievers.

Reciprocal Rank Fusion (RRF): a candidate's fused score is the sum, over every retriever that
returned it, of 1 / (k + rank), where `rank` is its 1-based position in that retriever's own
ranked list. A candidate missing from a list simply contributes 0 for it -- no need to compare
BM25 scores and cosine similarities directly, which is the point: their scales are not
comparable (they depend on query length and term rarity vs. embedding geometry), so combining
raw scores would need per-query normalisation to mean anything, while rank position is already
directly comparable across retrievers. `k` (60 by default, the usual choice in the IR
literature) softens the gap between rank 1 and rank 2, so a candidate's position in a second
list still matters even against a strong top-1 elsewhere. Measured against the alternative
(weighted score averaging after min-max normalisation) on this project's benchmark: see the
README.
"""

from typing import Dict, List, Sequence

from src.retrieval.types import SearchHit


def reciprocal_rank_fusion(rankings: Sequence[Sequence[SearchHit]], k: int = 60, top_k: int = 20) -> List[SearchHit]:
    scores: Dict[str, float] = {}
    payloads: Dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(hit.id, hit.payload)  # identical across retrievers (same point id)
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [SearchHit(id=cid, score=scores[cid], payload=payloads[cid]) for cid in ranked_ids[:top_k]]


class HybridRetriever:
    """Retrieves with both retrievers, then fuses the two ranked lists with RRF."""

    def __init__(self, semantic, lexical, rrf_k: int = 60, candidates_per_retriever: int = 20):
        self.semantic = semantic
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidates_per_retriever = candidates_per_retriever

    def retrieve(self, query: str, top_k: int = 20) -> List[SearchHit]:
        semantic_hits = self.semantic.retrieve(query, top_k=self.candidates_per_retriever)
        lexical_hits = self.lexical.retrieve(query, top_k=self.candidates_per_retriever)
        return reciprocal_rank_fusion([semantic_hits, lexical_hits], k=self.rrf_k, top_k=top_k)
