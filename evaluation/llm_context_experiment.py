"""Experiment: does an LLM-written chunk context improve retrieval over the structural heading?

"Contextual retrieval": a local LLM reads the whole document and writes one or two sentences that
situate each chunk; that sentence is embedded with the chunk. Compared, on the same chunks, embedder
and benchmark:

    structural   heading + text                  (what the pipeline embeds)
    llm          LLM context + heading + text
    llm_only     LLM context + text

Not part of the pipeline: see README "LLM-assisted chunk context" for the results and the decision.

    python evaluation/llm_context_experiment.py [--llm qwen2.5:7b] [--cache path.json]
"""

import argparse
import json
import sys
import time
import uuid
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from qdrant_client import QdrantClient

from src.config import get_settings
from src.embeddings import build_embedder
from src.evaluation.retrieval import load_benchmark, run_benchmark
from src.ingestion.pipeline import discover_documents, prepare_documents
from src.kb import DEFAULT_BENCHMARK
from src.vectorstore import QdrantStore, VectorRecord

PROMPT = """<document>
{document}
</document>
Voici un extrait de ce document :
<extrait>
{chunk}
</extrait>
Rédige un contexte court (une ou deux phrases, en français) qui situe cet extrait dans le document, \
afin d'améliorer la recherche de l'extrait. Réponds uniquement avec le contexte."""


def generate_contexts(prepared, llm: str, base_url: str, cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    started = time.perf_counter()
    for doc, chunks in prepared:
        document = "\n\n".join(f"{s.title}\n" + "\n\n".join(s.blocks) for s in doc.sections)
        for chunk in chunks:
            if chunk.chunk_id in cache:
                continue
            response = requests.post(f"{base_url}/api/generate", timeout=600, json={
                "model": llm, "stream": False, "prompt": PROMPT.format(document=document, chunk=chunk.text),
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 120},
            })
            response.raise_for_status()
            cache[chunk.chunk_id] = response.json()["response"].strip()
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"contexts ready: {doc.doc_id} ({time.perf_counter() - started:.0f}s)", flush=True)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", default="qwen2.5:7b")
    parser.add_argument("--cache", type=Path, default=Path(__file__).with_name("llm_contexts.json"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Payload indexes have no effect")

    settings = get_settings()
    prepared = prepare_documents(discover_documents(settings.kb_raw_dir), settings.kb_raw_dir,
                                 settings.chunk_min_words, settings.chunk_max_words)
    contexts = generate_contexts(prepared, args.llm, settings.embedding_base_url, args.cache)
    chunks = [chunk for _, doc_chunks in prepared for chunk in doc_chunks]
    embedder = build_embedder(settings)
    queries = load_benchmark(DEFAULT_BENCHMARK)

    variants = {
        "structural": lambda c: c.embedding_text(),
        "llm": lambda c: f"{contexts[c.chunk_id]}\n{c.embedding_text()}",
        "llm_only": lambda c: f"{contexts[c.chunk_id]}\n\n{c.text}",
    }
    print(f"\n{settings.embedding_model}, {len(chunks)} chunks, {len(queries)} queries")
    print(f"{'variant':12} {'hit@1':>6} {'hit@5':>6} {'mrr@10':>7} {'rec@5':>6} {'rec@500w':>9}")
    for name, embedding_text in variants.items():
        vectors = embedder.embed_documents([embedding_text(c) for c in chunks])
        store = QdrantStore(QdrantClient(location=":memory:"), name)
        store.ensure_collection(len(vectors[0]))
        store.upsert([VectorRecord(str(uuid.uuid4()), v, c.payload()) for c, v in zip(chunks, vectors)])
        m = run_benchmark(store, embedder, queries).overall
        print(f"{name:12} {m.hit_at_1:>6.2f} {m.hit_at_5:>6.2f} {m.mrr_at_10:>7.2f} {m.recall_at_5:>6.2f} {m.recall_at_budget:>9.2f}")


if __name__ == "__main__":
    main()
