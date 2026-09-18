"""The only module that talks to Qdrant. Everything else uses `QdrantStore`."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from qdrant_client import QdrantClient, models

from src.config import Settings


class CollectionConfigError(RuntimeError):
    """The existing collection is incompatible with the vectors being written."""


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: List[float]
    payload: Dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass(frozen=True)
class CollectionStats:
    name: str
    status: str
    points_count: int
    vector_size: int
    distance: str


class QdrantStore:
    # Payload fields used for filtering at retrieval time.
    INDEXED_FIELDS = (
        "doc_id", "doc_type", "standard", "category", "language",
        "section_slug", "unit_type", "record_label", "references",
    )

    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    @classmethod
    def from_settings(cls, settings: Settings, collection: str | None = None) -> "QdrantStore":
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout,
        )
        return cls(client, collection or settings.qdrant_collection)

    # --- connection / collection lifecycle ---------------------------------

    def is_available(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, vector_size: int, distance: models.Distance = models.Distance.COSINE) -> bool:
        """Create the collection if missing. Returns True if it was created.

        Raises CollectionConfigError if it exists with a different vector size or distance,
        which happens when the embedding model is changed without recreating the collection.
        """
        if self.collection_exists():
            stats = self.stats()
            if stats.vector_size != vector_size or stats.distance != distance.value:
                raise CollectionConfigError(
                    f"Collection '{self.collection}' has vectors of size {stats.vector_size} "
                    f"({stats.distance}), expected {vector_size} ({distance.value}). "
                    "Re-run ingestion with --recreate after changing the embedding model."
                )
            return False

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        for field in self.INDEXED_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        return True

    def recreate_collection(self, vector_size: int, distance: models.Distance = models.Distance.COSINE) -> None:
        self.delete_collection()
        self.ensure_collection(vector_size, distance)

    def delete_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection)

    def stats(self) -> CollectionStats:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        return CollectionStats(
            name=self.collection,
            status=str(getattr(info.status, "value", info.status)),
            points_count=self.count(),
            vector_size=vectors.size,
            distance=str(getattr(vectors.distance, "value", vectors.distance)),
        )

    # --- writes ------------------------------------------------------------

    def upsert(self, records: Sequence[VectorRecord], batch_size: int = 64) -> int:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in batch],
                wait=True,
            )
        return len(records)

    def delete_stale_chunks(self, doc_id: str, keep_ids: Sequence[str]) -> None:
        """Remove points of `doc_id` that are not in `keep_ids` (chunks that no longer exist)."""
        self._delete(
            models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))],
                must_not=[models.HasIdCondition(has_id=list(keep_ids))],
            )
        )

    def delete_documents_except(self, doc_ids: Sequence[str]) -> None:
        """Remove points of every document not listed in `doc_ids`."""
        self._delete(
            models.Filter(must_not=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))])
        )

    def _delete(self, selector: models.Filter) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=selector),
            wait=True,
        )

    def facet(self, field: str, limit: int = 10_000) -> List[tuple[str, int]]:
        """Distinct values of an indexed payload field with their point counts, sorted by value."""
        values = self.client.facet(collection_name=self.collection, key=field, limit=limit, exact=True)
        # The server keeps values of deleted points in the payload index with a count of 0.
        return sorted((str(hit.value), hit.count) for hit in values.hits if hit.count > 0)

    def doc_ids(self) -> List[str]:
        return [value for value, _ in self.facet("doc_id")]

    # --- reads -------------------------------------------------------------

    def count(self, filters: Mapping[str, Any] | None = None) -> int:
        return self.client.count(
            collection_name=self.collection,
            count_filter=_build_filter(filters),
            exact=True,
        ).count

    def get(self, ids: Iterable[str], with_vectors: bool = False, batch_size: int = 256) -> List[SearchHit]:
        ids = list(ids)
        hits: List[SearchHit] = []
        for start in range(0, len(ids), batch_size):
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=ids[start:start + batch_size],
                with_payload=True,
                with_vectors=with_vectors,
            )
            hits.extend(
                SearchHit(id=str(p.id), score=1.0, payload=p.payload or {}, vector=p.vector if with_vectors else None)
                for p in points
            )
        return hits

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> List[SearchHit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=_build_filter(filters),
            with_payload=True,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in response.points]


def _build_filter(filters: Mapping[str, Any] | None) -> models.Filter | None:
    """AND of payload conditions. A scalar must be equal; a list/tuple/set matches any of its values.

    On list payload fields (references) a condition matches when any element matches.
    """
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=key,
                match=models.MatchAny(any=list(value)) if isinstance(value, (list, tuple, set, frozenset))
                else models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
    )
