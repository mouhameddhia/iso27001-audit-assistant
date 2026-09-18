"""QdrantStore tests, run against Qdrant's in-process engine and, when reachable, the real server."""

import uuid

import pytest

from src.vectorstore import CollectionConfigError, VectorRecord

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


@pytest.fixture(params=["memory", pytest.param("server", marks=pytest.mark.integration)])
def store(request):
    return request.getfixturevalue(f"{request.param}_store")


def record(doc_id: str, n: int, vector: list[float], **payload) -> VectorRecord:
    chunk_id = f"{doc_id}:s01:c{n:02d}"
    return VectorRecord(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
        vector=vector,
        payload={"chunk_id": chunk_id, "doc_id": doc_id, "doc_type": "iso_standard", "unit_type": "guidance",
                 "references": [], **payload},
    )


RECORDS = [
    record("doc-a", 0, [1.0, 0.0, 0.0, 0.0], text="accès privilégiés", unit_type="record",
           references=["ISO/IEC 27001 A.5.18", "ISO/IEC 27001 A.8.2"]),
    record("doc-a", 1, [0.0, 1.0, 0.0, 0.0], text="sauvegardes", unit_type="example"),
    record("doc-b", 0, [0.0, 0.0, 1.0, 0.0], text="politique interne", doc_type="internal_policy"),
]


def test_collection_is_created_once_with_expected_vector_config(store):
    assert not store.collection_exists()
    assert store.ensure_collection(vector_size=4) is True
    assert store.ensure_collection(vector_size=4) is False

    stats = store.stats()
    assert (stats.vector_size, stats.distance, stats.points_count) == (4, "Cosine", 0)


def test_existing_collection_with_other_vector_size_is_rejected(store):
    store.ensure_collection(vector_size=4)
    with pytest.raises(CollectionConfigError, match="size 4"):
        store.ensure_collection(vector_size=768)


def test_upserted_points_keep_their_payload_and_vector(store):
    store.ensure_collection(vector_size=4)

    assert store.upsert(RECORDS) == 3
    assert store.count() == 3
    assert store.count({"doc_id": "doc-a"}) == 2

    stored = {hit.id: hit for hit in store.get([r.id for r in RECORDS], with_vectors=True)}
    assert {i: hit.payload for i, hit in stored.items()} == {r.id: r.payload for r in RECORDS}
    assert stored[RECORDS[0].id].vector == pytest.approx(RECORDS[0].vector)
    assert store.get([RECORDS[0].id])[0].vector is None


def test_upsert_with_same_ids_overwrites_instead_of_duplicating(store):
    store.ensure_collection(vector_size=4)
    store.upsert(RECORDS)
    store.upsert([VectorRecord(RECORDS[0].id, RECORDS[0].vector, {**RECORDS[0].payload, "text": "v2"})])

    assert store.count() == 3
    assert store.get([RECORDS[0].id])[0].payload["text"] == "v2"


def test_search_ranks_by_similarity_and_applies_filters(store):
    store.ensure_collection(vector_size=4)
    store.upsert(RECORDS)
    query = [0.9, 0.1, 0.0, 0.0]

    hits = store.search(query, top_k=2)
    assert [h.payload["chunk_id"] for h in hits] == ["doc-a:s01:c00", "doc-a:s01:c01"]
    assert hits[0].score > hits[1].score

    def chunk_ids(**filters):
        return [h.payload["chunk_id"] for h in store.search(query, top_k=5, filters=filters)]

    assert chunk_ids(doc_type="internal_policy") == ["doc-b:s01:c00"]
    assert chunk_ids(references="ISO/IEC 27001 A.8.2") == ["doc-a:s01:c00"]
    assert chunk_ids(unit_type=["record", "example"]) == ["doc-a:s01:c00", "doc-a:s01:c01"]
    assert chunk_ids(unit_type=["record", "example"], doc_id="doc-b") == []


def test_facet_counts_distinct_values(store):
    store.ensure_collection(vector_size=4)
    store.upsert(RECORDS)

    assert store.facet("doc_id") == [("doc-a", 2), ("doc-b", 1)]
    assert store.facet("references") == [("ISO/IEC 27001 A.5.18", 1), ("ISO/IEC 27001 A.8.2", 1)]


def test_stale_chunks_and_removed_documents_are_deleted(store):
    store.ensure_collection(vector_size=4)
    store.upsert(RECORDS)

    store.delete_stale_chunks("doc-a", keep_ids=[RECORDS[0].id])
    assert sorted(h.payload["chunk_id"] for h in store.get([r.id for r in RECORDS])) == ["doc-a:s01:c00", "doc-b:s01:c00"]

    store.delete_documents_except(["doc-a"])
    assert store.doc_ids() == ["doc-a"]
    assert store.count() == 1
