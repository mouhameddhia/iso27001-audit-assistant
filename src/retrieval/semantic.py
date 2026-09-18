"""Stage 2's semantic candidate retriever: a thin `Retriever` adapter over Stage 1's Qdrant store.

No Qdrant logic lives here -- everything is delegated to `QdrantStore.search`, so this module has
exactly one job: embed the query and call it. It exists so semantic retrieval can be composed with
`BM25Retriever` in `fusion.HybridRetriever` and swapped into `evaluation.retrieval.run_benchmark`
through the same `Retriever` interface as every other retrieval component.
"""

from typing import Any, List, Mapping

from src.embeddings import Embedder
from src.vectorstore import QdrantStore, SearchHit


class SemanticSearchRetriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, filters: Mapping[str, Any] | None = None):
        self.store = store
        self.embedder = embedder
        self.filters = filters

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchHit]:
        return self.store.search(self.embedder.embed_query(query), top_k=top_k, filters=self.filters)
