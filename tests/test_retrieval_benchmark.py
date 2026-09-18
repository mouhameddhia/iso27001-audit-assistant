import pytest

from src.evaluation.retrieval import (
    Anchor, BenchmarkQuery, QueryResult, aggregate, load_benchmark, missing_anchors, run_benchmark, score_query,
)
from src.ingestion.parser import parse_document
from src.ingestion.pipeline import discover_documents
from src.kb import DEFAULT_BENCHMARK
from src.vectorstore import SearchHit
from tests.conftest import REAL_KB_DIR


def hit(doc_id: str, text: str) -> SearchHit:
    return SearchHit(id=text, score=0.0, payload={"doc_id": doc_id, "text": text, "word_count": len(text.split())})


QUERY = BenchmarkQuery("q1", "finding", "?", (Anchor("doc-a", "revue des accès"), Anchor("doc-a", "NC-2026-014")))


def test_benchmark_file_is_well_formed_and_matches_the_knowledge_base():
    queries = load_benchmark(DEFAULT_BENCHMARK)
    documents = [parse_document(p, REAL_KB_DIR) for p in discover_documents(REAL_KB_DIR)]

    assert len({q.id for q in queries}) == len(queries) > 0
    assert all(q.query.strip() and q.category and q.relevant for q in queries)
    assert missing_anchors(queries, documents) == []


def test_relevance_requires_same_document_and_is_case_and_space_insensitive():
    hits = [hit("doc-b", "la revue des accès"), hit("doc-a", "La REVUE   des accès privilégiés")]
    result = score_query(QUERY, hits, budget_words=500)
    assert result.first_relevant_rank == 2
    assert result.recall_at_5 == 0.5


def test_recall_within_budget_only_counts_chunks_that_fit():
    hits = [hit("doc-a", "revue des accès " + "x " * 300), hit("doc-a", "NC-2026-014 " + "y " * 300)]
    result = score_query(QUERY, hits, budget_words=400)
    assert (result.recall_at_5, result.recall_at_budget) == (1.0, 0.5)


def test_first_chunk_counts_even_if_larger_than_budget():
    result = score_query(QUERY, [hit("doc-a", "revue des accès " + "x " * 900)], budget_words=100)
    assert result.recall_at_budget == 0.5


class FakeRetriever:
    """Any object with `.retrieve(query, top_k)` works with `run_benchmark`, whatever pipeline it wraps."""

    def __init__(self, hits_by_query: dict[str, list[SearchHit]]):
        self.hits_by_query = hits_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        return self.hits_by_query.get(query, [])[:top_k]


def test_run_benchmark_queries_the_retriever_and_aggregates_results():
    queries = [
        BenchmarkQuery("q1", "cat", "revue des accès", (Anchor("doc-a", "revue des accès"),)),
        BenchmarkQuery("q2", "cat", "sauvegardes", (Anchor("doc-b", "sauvegardes"),)),
    ]
    retriever = FakeRetriever({
        "revue des accès": [hit("doc-a", "revue des accès privilégiés")],
        "sauvegardes": [hit("doc-x", "non pertinent"), hit("doc-b", "test des sauvegardes")],
    })

    report = run_benchmark(retriever, queries, top_k=10, budget_words=500)

    assert retriever.calls == [("revue des accès", 10), ("sauvegardes", 10)]
    assert [r.first_relevant_rank for r in report.results] == [1, 2]
    assert report.overall.hit_at_1 == 0.5 and report.overall.hit_at_5 == 1.0


def test_aggregate_metrics():
    results = [
        QueryResult("a", "c", 1, 1.0, 1.0, ()),
        QueryResult("b", "c", 4, 0.5, 0.0, ()),
        QueryResult("c", "c", None, 0.0, 0.0, ()),
        QueryResult("d", "c", 20, 0.0, 0.0, ()),
    ]
    m = aggregate(results)
    assert (m.queries, m.hit_at_1, m.hit_at_5) == (4, 0.25, 0.5)
    assert m.mrr_at_10 == pytest.approx((1 + 0.25) / 4)
    assert m.recall_at_5 == pytest.approx(0.375)
