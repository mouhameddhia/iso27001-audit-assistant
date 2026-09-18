"""
semantic_retriever.py

Simple semantic (meaning-based) retriever:
1. Embed every chunk once, ahead of time, into a vector.
2. Embed the user's query the same way.
3. Return the chunks whose vectors are most similar (cosine similarity)
   to the query vector.

The embedding model is local, served by Ollama (e.g. `nomic-embed-text`),
so no data ever leaves the machine and no external model download is
required beyond `ollama pull nomic-embed-text`.

The embedding function is injected (`embed_fn`) rather than hard-coded,
so tests can use a fake embedder without needing Ollama running.
"""

from typing import Callable, List, Sequence
import numpy as np
import requests

from src.ingestion.document_loader import Chunk


# --- Default embedder: calls a local Ollama server ------------------------

def ollama_embed(
    texts: Sequence[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> List[List[float]]:
    """Embed a list of texts using Ollama's /api/embeddings endpoint.

    Requires `ollama pull nomic-embed-text` (or another embedding model)
    and the Ollama server running locally.
    """
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])
    return embeddings


# --- Retriever --------------------------------------------------------------

class SemanticRetriever:
    """In-memory vector index over a list of chunks."""

    def __init__(self, embed_fn: Callable[[Sequence[str]], List[List[float]]] = ollama_embed):
        self.embed_fn = embed_fn
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape: (n_chunks, dim)

    def index(self, chunks: List[Chunk]) -> None:
        """Embed and store all chunks. Call this once at startup."""
        self.chunks = chunks
        if not chunks:
            self.vectors = np.zeros((0, 0))
            return
        raw_vectors = self.embed_fn([c.text for c in chunks])
        self.vectors = self._normalize(np.array(raw_vectors, dtype=np.float32))

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row so a dot product equals cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for empty vectors
        return matrix / norms

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Return the `top_k` chunks most semantically similar to `query`."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vector = self._normalize(np.array(self.embed_fn([query]), dtype=np.float32))
        similarities = self.vectors @ query_vector[0]  # cosine similarity per chunk

        top_indices = np.argsort(-similarities)[:top_k]
        return [self.chunks[i] for i in top_indices]
