"""Session-scoped retrieval over a session's own imported client documents.

Reuses Stage 2's retrieval components verbatim (`BM25Retriever`, `SemanticSearchRetriever`,
`HybridRetriever`) -- the only difference from the knowledge-base pipeline is *what* they're
pointed at: this session's own chunks and its own Qdrant collection, never `data/raw` or the
shared knowledge-base collection. No cross-encoder rerank here: client-document context is
supplementary background for the model, not a citable source needing the same precision as ISO
evidence (see the "client document context" prompt block in generation/prompts.py).
"""

from typing import List, Optional

from src.config import Settings, get_settings
from src.documents.ingestion import session_collection_name, session_chunks
from src.embeddings import Embedder, build_embedder
from src.report.session import AuditSession
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.semantic import SemanticSearchRetriever
from src.vectorstore import QdrantStore, SearchHit


def build_document_retriever(
    session: AuditSession, settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None, store: Optional[QdrantStore] = None,
) -> Optional[HybridRetriever]:
    """None when the session has no confirmed imported documents -- callers must skip document
    retrieval entirely in that case, so a document-less session's behaviour stays unchanged."""
    if not session.confirmed_documents:
        return None
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    store = store or QdrantStore.from_settings(settings, collection=session_collection_name(session, settings))
    bm25 = BM25Retriever(session_chunks(session, settings))
    semantic = SemanticSearchRetriever(store, embedder)
    return HybridRetriever(
        semantic, bm25, rrf_k=settings.retrieval_rrf_k,
        candidates_per_retriever=settings.retrieval_candidates_per_retriever,
    )


def retrieve_document_evidence(
    session: AuditSession, query: str, top_k: int = 3, settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None, store: Optional[QdrantStore] = None,
) -> List[SearchHit]:
    """The session's imported-document context for one query, or [] when there are none."""
    retriever = build_document_retriever(session, settings, embedder, store)
    if retriever is None:
        return []
    return retriever.retrieve(query, top_k=top_k)
