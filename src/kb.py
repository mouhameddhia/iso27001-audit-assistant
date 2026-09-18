"""Knowledge-base command line.

    python -m src.kb ingest [--recreate] [--no-prune]
    python -m src.kb stats
    python -m src.kb search "revue des droits d'accès privilégiés" [--mode rerank] [--top-k 5]
                            [--doc-type iso_standard] [--unit-type record] [--reference "ISO/IEC 27001 A.5.18"]
    python -m src.kb evaluate [--mode rerank] [--in-memory] [--details]

--mode selects the retrieval pipeline: semantic (Stage 1: Qdrant only), hybrid (+ BM25, RRF-fused),
or rerank (+ cross-encoder reranking of the fused candidates; the default, full Stage 2 pipeline).
"""

import argparse
import logging
import sys
import warnings

from qdrant_client import QdrantClient

from src.anonymization.anonymizer import Anonymizer
from src.config import PROJECT_ROOT, get_settings
from src.embeddings import build_embedder
from src.evaluation.retrieval import load_benchmark, missing_anchors, run_benchmark
from src.ingestion.parser import parse_document
from src.ingestion.pipeline import discover_documents, ingest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.pipeline import RerankedRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticSearchRetriever
from src.vectorstore import QdrantStore

DEFAULT_BENCHMARK = PROJECT_ROOT / "evaluation" / "kb_retrieval_benchmark.json"
MODES = ("semantic", "hybrid", "rerank")


def _store(args: argparse.Namespace) -> QdrantStore:
    store = QdrantStore.from_settings(get_settings(), collection=args.collection)
    if not store.is_available():
        sys.exit(f"Qdrant is not reachable at {get_settings().qdrant_url}")
    return store


def _existing_store(args: argparse.Namespace) -> QdrantStore:
    store = _store(args)
    if not store.collection_exists():
        sys.exit(f"Collection '{store.collection}' does not exist. Run: python -m src.kb ingest")
    return store


def _build_retriever(mode: str, store: QdrantStore, settings):
    """A `Retriever` (semantic / hybrid / hybrid+reranked) built from the CLI's --mode."""
    semantic = SemanticSearchRetriever(store, build_embedder(settings))
    if mode == "semantic":
        return semantic
    hybrid = HybridRetriever(
        semantic, BM25Retriever.from_settings(settings),
        rrf_k=settings.retrieval_rrf_k, candidates_per_retriever=settings.retrieval_candidates_per_retriever,
    )
    if mode == "hybrid":
        return hybrid
    return RerankedRetriever(hybrid, CrossEncoderReranker.from_settings(settings), settings.retrieval_fusion_top_k)


def cmd_ingest(args: argparse.Namespace) -> None:
    report = ingest(store=_store(args), recreate=args.recreate, prune=not args.no_prune)
    print(f"\nCollection      : {report.collection} ({'created' if report.collection_created else 'existing'})")
    print(f"Embedding model : {report.embedding_model} (dim {report.vector_size})")
    print(f"{'source file':42} {'doc_id':32} {'type':16} {'sections':>8} {'chunks':>6} {'stored':>6}")
    for doc in report.documents:
        print(f"{doc.source_file:42} {doc.doc_id:32} {doc.doc_type:16} {doc.sections:>8} {doc.chunks:>6} {doc.points_stored:>6}")
    print(f"Documents: {len(report.documents)} | chunks: {report.chunks_total} "
          f"(embedded {report.chunks_embedded}, unchanged {report.chunks_reused}) | "
          f"points in collection: {report.points_in_collection} | {report.duration_seconds}s")


def cmd_stats(args: argparse.Namespace) -> None:
    store = _existing_store(args)
    stats = store.stats()
    print(f"Collection {stats.name}: status={stats.status} points={stats.points_count} "
          f"vector_size={stats.vector_size} distance={stats.distance}")
    for field in ("doc_id", "unit_type", "section_slug"):
        print(f"\n{field}:")
        for value, count in store.facet(field):
            print(f"  {value:35} {count:>4}")


def cmd_search(args: argparse.Namespace) -> None:
    settings = get_settings()
    store = _existing_store(args)
    filters = {
        key: value
        for key, value in (("doc_type", args.doc_type), ("doc_id", args.doc_id),
                           ("unit_type", args.unit_type), ("references", args.reference))
        if value
    }
    if filters and args.mode != "semantic":
        sys.exit("--doc-type/--doc-id/--unit-type/--reference filters only work with --mode semantic "
                  "(BM25 has no payload filter).")

    if args.anonymize:
        query, mapping = Anonymizer().anonymize(args.query)
        if mapping:
            print(f"anonymized query: {query}  (mapping: {mapping})\n")
    else:
        query = args.query

    retriever = _build_retriever(args.mode, store, settings)
    if args.mode == "semantic":
        retriever.filters = filters
    for rank, hit in enumerate(retriever.retrieve(query, top_k=args.top_k), start=1):
        p = hit.payload
        record = f" {p['record_id']} ({p['record_label']})" if p.get("record_id") else ""
        print(f"\n#{rank} score={hit.score:.4f}  {p['chunk_id']}  [{p['unit_type']}{record}]")
        print(f"   {p['heading']}")
        print(f"   refs: {', '.join(p['references']) or '-'}")
        print("   " + p["text"][:300].replace("\n", " ") + ("..." if len(p["text"]) > 300 else ""))


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = get_settings()
    queries = load_benchmark(args.benchmark)
    documents = [parse_document(p, settings.kb_raw_dir) for p in discover_documents(settings.kb_raw_dir)]
    stale = missing_anchors(queries, documents)
    if stale:
        for query_id, anchor in stale:
            print(f"Anchor not found in {anchor.doc_id}: {anchor.contains!r} ({query_id})")
        sys.exit("The benchmark no longer matches the knowledge base.")

    if args.in_memory:
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        store = QdrantStore(QdrantClient(location=":memory:"), "evaluation")
        ingest(settings, embedder=build_embedder(settings), store=store)
        target = f"in-memory ingestion of {settings.kb_raw_dir}"
    else:
        store = _existing_store(args)
        target = f"collection {store.collection}"

    retriever = _build_retriever(args.mode, store, settings)
    report = run_benchmark(retriever, queries, top_k=10, budget_words=args.budget)
    print(f"{len(queries)} queries | {target} | mode={args.mode} | {settings.embedding_model} | "
          f"chunk words min={settings.chunk_min_words} max={settings.chunk_max_words}\n")
    print(f"{'category':12} {'n':>3} {'hit@1':>6} {'hit@5':>6} {'mrr@10':>7} {'rec@5':>6} {f'rec@{report.budget_words}w':>9}")
    rows = [*sorted(report.by_category.items()), ("all", report.overall)]
    for name, m in rows:
        print(f"{name:12} {m.queries:>3} {m.hit_at_1:>6.2f} {m.hit_at_5:>6.2f} {m.mrr_at_10:>7.2f} "
              f"{m.recall_at_5:>6.2f} {m.recall_at_budget:>9.2f}")
    if args.details:
        print()
        for r in report.results:
            print(f"{r.query_id:9} first relevant rank={r.first_relevant_rank}  top: {', '.join(r.top_chunk_ids[:3])}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.kb", description=__doc__.splitlines()[0])
    parser.add_argument("--collection", help="override QDRANT_COLLECTION")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="synchronise data/raw into Qdrant")
    p_ingest.add_argument("--recreate", action="store_true", help="drop and recreate the collection first")
    p_ingest.add_argument("--no-prune", action="store_true", help="keep points of documents no longer in the KB folder")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("stats", help="show collection statistics").set_defaults(func=cmd_stats)

    p_search = sub.add_parser("search", help="search the knowledge base")
    p_search.add_argument("query")
    p_search.add_argument("--mode", choices=MODES, default="rerank")
    p_search.add_argument("--anonymize", action="store_true", help="anonymize the query before retrieval")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--doc-type", choices=["iso_standard", "internal_policy", "other"])
    p_search.add_argument("--doc-id")
    p_search.add_argument("--unit-type", choices=["guidance", "example", "record", "template"])
    p_search.add_argument("--reference", help='canonical reference, e.g. "ISO/IEC 27001 A.5.18"')
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("evaluate", help="run the retrieval benchmark")
    p_eval.add_argument("--mode", choices=MODES, default="rerank")
    p_eval.add_argument("--in-memory", action="store_true",
                        help="ingest data/raw with the current settings into a temporary in-process Qdrant")
    p_eval.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p_eval.add_argument("--budget", type=int, default=500, help="context budget in words for rec@budget")
    p_eval.add_argument("--details", action="store_true", help="print the rank obtained for each query")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
