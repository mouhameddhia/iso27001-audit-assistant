"""Cross-encoder reranking of a fused candidate set.

A cross-encoder scores a (query, passage) pair jointly through one transformer pass, unlike the
bi-encoder used for semantic retrieval (Stage 1), which embeds them independently and compares
vectors. Joint scoring is usually more accurate but too slow to run over a whole collection, hence
its place here: only over the small candidate set that fusion already narrowed down.

Model: see the README for the comparison that led to the configured default. In short, the
knowledge base and expected auditor input are largely French, so an English-only cross-encoder
(e.g. the popular `cross-encoder/ms-marco-*` line) was not considered; the model must support
French out of the box.
"""

import os

# transformers tries to import its TensorFlow integration on load; on this environment that fails
# (TensorFlow installed with Keras 3, which transformers does not yet support). Only the PyTorch
# backend is used here, so the TF path is turned off before the first import pulls it in.
os.environ.setdefault("USE_TF", "0")

from typing import List, Protocol, Sequence

from src.config import Settings
from src.vectorstore import SearchHit


class RerankError(RuntimeError):
    pass


class ScoringModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


def _load_model(model_name: str, device: str) -> ScoringModel:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankError("sentence-transformers is required for cross-encoder reranking") from exc
    try:
        return CrossEncoder(model_name, device=device)
    except Exception as exc:
        raise RerankError(f"Failed to load cross-encoder '{model_name}': {exc}") from exc


class CrossEncoderReranker:
    """Reorders a candidate set by (query, chunk) relevance. `model` is injectable for testing."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16, model: ScoringModel | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = model if model is not None else _load_model(model_name, device)

    @classmethod
    def from_settings(cls, settings: Settings) -> "CrossEncoderReranker":
        return cls(settings.rerank_model, device=settings.rerank_device, batch_size=settings.rerank_batch_size)

    @staticmethod
    def _passage(hit: SearchHit) -> str:
        payload = hit.payload
        heading = payload.get("heading")
        text = payload.get("text", "")
        return f"{heading}\n\n{text}" if heading else text

    def rerank(self, query: str, candidates: Sequence[SearchHit], top_k: int) -> List[SearchHit]:
        if not candidates:
            return []
        pairs = [(query, self._passage(hit)) for hit in candidates]
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        except TypeError:
            scores = self.model.predict(pairs)  # a test double may not accept batch_size
        except Exception as exc:
            raise RerankError(f"Cross-encoder scoring failed: {exc}") from exc

        ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
        return [
            SearchHit(id=hit.id, score=float(score), payload=hit.payload, vector=hit.vector)
            for hit, score in ranked[:top_k]
        ]
