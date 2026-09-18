"""Shared candidate representation and retriever contract for Stage 2.

Every retrieval component (semantic, lexical, fused, reranked) speaks the same currency: a
`SearchHit` (id = Qdrant point id, so results from different retrievers over the same chunks
compare and deduplicate directly) and the `Retriever` protocol below. This is what lets
`fusion.HybridRetriever` and `evaluation.retrieval.run_benchmark` treat any of them
interchangeably.
"""

from typing import List, Protocol

from src.vectorstore import SearchHit

__all__ = ["Retriever", "SearchHit"]


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> List[SearchHit]: ...
