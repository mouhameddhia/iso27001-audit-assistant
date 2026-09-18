"""Retrieval benchmark: chunking-independent relevance judgments and ranking metrics.

A query lists anchors (doc_id + phrase). A retrieved chunk is relevant when it belongs to the
anchor's document and its text contains the phrase, so the same benchmark compares any chunking
strategy or embedding model.

Metrics per query, averaged over queries:
- hit@1, hit@5   a relevant chunk is ranked first / in the top 5
- mrr@10         1 / rank of the first relevant chunk (0 if not in the top 10)
- recall@5       share of the query's anchors found in the top 5 chunks
- recall@budget  share of anchors found in the ranked chunks that fit in a word budget, which
                 compares chunk sizes fairly: large chunks cover more text but cost more context
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.ingestion.models import ParsedDocument
from src.ingestion.parser import normalize_for_matching
from src.retrieval.types import Retriever, SearchHit


@dataclass(frozen=True)
class Anchor:
    doc_id: str
    contains: str


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    category: str
    query: str
    relevant: Tuple[Anchor, ...]


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    category: str
    first_relevant_rank: int | None
    recall_at_5: float
    recall_at_budget: float
    top_chunk_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Metrics:
    queries: int
    hit_at_1: float
    hit_at_5: float
    mrr_at_10: float
    recall_at_5: float
    recall_at_budget: float


@dataclass(frozen=True)
class BenchmarkReport:
    overall: Metrics
    by_category: Dict[str, Metrics]
    results: List[QueryResult]
    budget_words: int


def load_benchmark(path: Path) -> List[BenchmarkQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        BenchmarkQuery(
            id=item["id"],
            category=item["category"],
            query=item["query"],
            relevant=tuple(Anchor(a["doc_id"], a["contains"]) for a in item["relevant"]),
        )
        for item in data["queries"]
    ]


def missing_anchors(queries: Sequence[BenchmarkQuery], documents: Iterable[ParsedDocument]) -> List[Tuple[str, Anchor]]:
    """Anchors whose phrase does not occur in their document: the benchmark is stale."""
    texts = {
        doc.doc_id: normalize_for_matching(" ".join(block for s in doc.sections for block in s.blocks))
        for doc in documents
    }
    return [
        (q.id, anchor)
        for q in queries
        for anchor in q.relevant
        if normalize_for_matching(anchor.contains) not in texts.get(anchor.doc_id, "")
    ]


def _matched_anchors(hit: SearchHit, anchors: Sequence[Anchor]) -> set[int]:
    text = normalize_for_matching(hit.payload.get("text", ""))
    return {
        i for i, anchor in enumerate(anchors)
        if hit.payload.get("doc_id") == anchor.doc_id and normalize_for_matching(anchor.contains) in text
    }


def score_query(query: BenchmarkQuery, hits: Sequence[SearchHit], budget_words: int) -> QueryResult:
    first_rank = None
    in_top_5: set[int] = set()
    in_budget: set[int] = set()
    used_words = 0
    for rank, hit in enumerate(hits, start=1):
        matched = _matched_anchors(hit, query.relevant)
        if matched and first_rank is None:
            first_rank = rank
        if rank <= 5:
            in_top_5 |= matched
        used_words += int(hit.payload.get("word_count") or len(hit.payload.get("text", "").split()))
        if rank == 1 or used_words <= budget_words:
            in_budget |= matched
    n = len(query.relevant)
    return QueryResult(
        query_id=query.id,
        category=query.category,
        first_relevant_rank=first_rank,
        recall_at_5=len(in_top_5) / n,
        recall_at_budget=len(in_budget) / n,
        top_chunk_ids=tuple(str(h.payload.get("chunk_id", h.id)) for h in hits[:5]),
    )


def aggregate(results: Sequence[QueryResult]) -> Metrics:
    n = len(results)
    ranks = [r.first_relevant_rank for r in results]
    return Metrics(
        queries=n,
        hit_at_1=sum(r == 1 for r in ranks) / n,
        hit_at_5=sum(r is not None and r <= 5 for r in ranks) / n,
        mrr_at_10=sum(1 / r for r in ranks if r is not None and r <= 10) / n,
        recall_at_5=sum(r.recall_at_5 for r in results) / n,
        recall_at_budget=sum(r.recall_at_budget for r in results) / n,
    )


def run_benchmark(
    retriever: Retriever,
    queries: Sequence[BenchmarkQuery],
    top_k: int = 10,
    budget_words: int = 500,
) -> BenchmarkReport:
    """Score `retriever` (semantic-only, hybrid, or hybrid+reranked -- anything with `.retrieve`)."""
    results = [score_query(q, retriever.retrieve(q.query, top_k=top_k), budget_words) for q in queries]
    by_category: Dict[str, List[QueryResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    return BenchmarkReport(
        overall=aggregate(results),
        by_category={category: aggregate(items) for category, items in by_category.items()},
        results=results,
        budget_words=budget_words,
    )
