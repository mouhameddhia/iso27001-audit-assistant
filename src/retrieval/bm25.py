"""Lexical (BM25) retrieval, independent from the Qdrant/semantic retriever.

Complements semantic search on exact terminology: ISO control ids, clause numbers, quoted
identifiers, and other tokens a bi-encoder can blur together. Generic tokenizer, no rules tied to
a specific document -- it treats any letter-or-digit run joined by dots as one identifier token
("A.5.18", "CLD.6.3.1", "9.2", "27001:2022"), so a query for a control id matches it as a whole
instead of scattering across single characters and digits.
"""

import re
from typing import List, Sequence

from rank_bm25 import BM25Okapi

from src.config import Settings
from src.ingestion.models import KnowledgeChunk
from src.vectorstore import SearchHit

# An identifier: letters/digits, optionally with dot- or colon-separated segments (A.5.18, 27001:2022).
# Tried before the plain-word alternative so it wins whenever it applies. Plain words need at least
# 2 characters: a lone letter is never meaningful and mostly comes from an apostrophe splitting a
# contraction ("d'accès" -> "d", "n'a" -> "a"), where it would otherwise match almost every chunk.
_TOKEN_RE = re.compile(r"[a-zà-ÿ]+(?:[.:]\d[\w.:]*)+|\d[\w.:]*(?:[.:]\d[\w.:]*)+|[a-zà-ÿ0-9]{2,}")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 index over a fixed set of chunks, built once at startup."""

    def __init__(self, chunks: Sequence[KnowledgeChunk]):
        self._ids = [chunk.point_id for chunk in chunks]
        self._payloads = [chunk.payload() for chunk in chunks]
        corpus = [tokenize(chunk.embedding_text()) for chunk in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_settings(cls, settings: Settings) -> "BM25Retriever":
        # Chunks are re-derived from data/raw (parsing has no side effects and needs no network),
        # so the lexical index always matches whatever the embedder/Qdrant were given.
        from src.ingestion.pipeline import discover_documents, prepare_documents

        prepared = prepare_documents(
            discover_documents(settings.kb_raw_dir), settings.kb_raw_dir,
            settings.chunk_min_words, settings.chunk_max_words,
        )
        return cls([chunk for _, chunks in prepared for chunk in chunks])

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchHit(id=self._ids[i], score=float(scores[i]), payload=self._payloads[i])
            for i in ranked[:top_k] if scores[i] > 0
        ]
