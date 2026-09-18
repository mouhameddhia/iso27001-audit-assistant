"""
main.py

Simple command-line entrypoint for the ISO/IEC 27001 audit-report assistant
prototype.

What it does:
1. Loads the processed document chunks (run ingestion first if needed).
2. Builds the hybrid retriever (semantic + BM25) and indexes the chunks.
3. Builds the anonymizer and the full RAG pipeline.
4. Reads auditor text from the command line (or stdin) and prints the
   grounded, deanonymized answer, plus the sources used.

Run from the project root:
    python -m src.main "Chez ABC Consulting, le serveur 192.168.1.20 n'est pas correctement protégé." --lang fr
"""

import argparse
import sys
from pathlib import Path

from src.anonymization.anonymizer import Anonymizer
from src.ingestion.document_loader import (
    load_and_chunk_documents,
    load_chunks,
    save_chunks,
)
from src.pipeline.rag_pipeline import RagPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.semantic_retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def build_pipeline(ollama_model: str = "llama3") -> RagPipeline:
    """Load/prepare chunks, build the retrievers and the pipeline."""
    if not PROCESSED_PATH.exists():
        print(f"No processed chunks found, ingesting documents from {RAW_DIR} ...")
        chunks = load_and_chunk_documents(RAW_DIR)
        save_chunks(chunks, PROCESSED_PATH)
    else:
        chunks = load_chunks(PROCESSED_PATH)
    print(f"Loaded {len(chunks)} chunks for retrieval.")

    semantic_retriever = SemanticRetriever()   # uses Ollama embeddings by default
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    hybrid_retriever.index(chunks)

    # NOTE: pass your engagement's known client/user names here so they are
    # anonymized reliably, e.g.:
    #   Anonymizer(known_entities={"CLIENT": ["ABC Consulting"]})
    anonymizer = Anonymizer()

    return RagPipeline(retriever=hybrid_retriever, anonymizer=anonymizer, ollama_model=ollama_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISO/IEC 27001 audit-report writing assistant")
    parser.add_argument("text", nargs="?", help="Auditor's text to improve/reformulate")
    parser.add_argument("--lang", choices=["fr", "en"], default="fr", help="Response language")
    parser.add_argument("--model", default="llama3", help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    user_text = args.text or sys.stdin.read().strip()
    if not user_text:
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    pipeline = build_pipeline(ollama_model=args.model)

    print("\nRunning pipeline: anonymize -> hybrid retrieval -> Ollama -> deanonymize\n")
    result = pipeline.run(user_text, language=args.lang, top_k=args.top_k)

    print("=== Answer ===")
    print(result.answer)

    print("\n=== Sources used ===")
    if not result.sources:
        print("(no chunks retrieved)")
    for chunk in result.sources:
        print(f"- {chunk.source} (chunk {chunk.id})")


if __name__ == "__main__":
    main()
