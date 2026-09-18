"""Text embeddings through a local Ollama server (no data leaves the machine)."""

from typing import List, Protocol, Sequence

import requests

from src.config import Settings


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str

    @property
    def fingerprint(self) -> str:
        """Identifies everything that changes document vectors besides the text (model, prefix)."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str,
        batch_size: int = 16,
        timeout: int = 120,
        document_prefix: str = "",
        query_prefix: str = "",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaEmbedder":
        return cls(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
        )

    @property
    def fingerprint(self) -> str:
        return f"ollama:{self.model}:{self.document_prefix}"

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingError(f"Ollama embedding request failed ({self.base_url}, model={self.model}): {exc}") from exc

        vectors = response.json().get("embeddings") or []
        if len(vectors) != len(inputs):
            raise EmbeddingError(f"Expected {len(inputs)} embeddings, got {len(vectors)}")
        if any(not vector for vector in vectors):
            raise EmbeddingError("Ollama returned an empty embedding")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [f"{self.document_prefix}{t}" for t in texts[start:start + self.batch_size]]
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed([f"{self.query_prefix}{text}"])[0]


def build_embedder(settings: Settings) -> Embedder:
    return OllamaEmbedder.from_settings(settings)
