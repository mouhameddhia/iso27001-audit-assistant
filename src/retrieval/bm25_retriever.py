"""
bm25_retriever.py

Simple lexical (keyword-based) retriever using BM25. Good for exact
terminology matches (ISO clause numbers, control names, acronyms) that
semantic search can sometimes miss.
"""

import re
from typing import List
from rank_bm25 import BM25Okapi

from src.ingestion.document_loader import Chunk


def _tokenize(text: str) -> List[str]:
    """Very simple tokenizer: lowercase words, works fine for FR/EN text."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """In-memory BM25 index over a list of chunks."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def index(self, chunks: List[Chunk]) -> None:
        """Tokenize and index all chunks. Call this once at startup."""
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Return the `top_k` chunks with the highest BM25 score for `query`."""
        if self.bm25 is None or not self.chunks:
            return []

        scores = self.bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self.chunks[i] for i in ranked_indices[:top_k]]
