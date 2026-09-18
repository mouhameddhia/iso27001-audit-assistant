"""
Tests for the retrieval modules. Run with: pytest tests/test_retrieval.py -v

The semantic retriever normally calls Ollama for embeddings. To test it
without needing Ollama running, we inject a small deterministic fake
embedder (`fake_embed`) that turns text into a bag-of-words vector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.document_loader import Chunk
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


VOCAB = [
    "access", "control", "encryption", "password", "server",
    "review", "privileged", "chiffrement", "acces", "audit",
]


def fake_embed(texts):
    """Turn each text into a fixed-size vector counting vocab word hits.
    Good enough to make 'semantically' related fake-test text score higher,
    without needing a real embedding model or network access.
    """
    vectors = []
    for text in texts:
        lower = text.lower()
        vectors.append([lower.count(word) for word in VOCAB])
    return vectors


SAMPLE_CHUNKS = [
    Chunk(id="c1", source="doc1.txt", text="Access control review of privileged accounts."),
    Chunk(id="c2", source="doc1.txt", text="Encryption and password policy for servers."),
    Chunk(id="c3", source="doc2.txt", text="Backup and disaster recovery procedure."),
]


def test_bm25_retrieves_keyword_match():
    retriever = BM25Retriever()
    retriever.index(SAMPLE_CHUNKS)

    results = retriever.retrieve("privileged access accounts", top_k=1)

    assert results[0].id == "c1"


def test_bm25_empty_index_returns_empty():
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.retrieve("anything") == []


def test_semantic_retriever_with_fake_embedder():
    retriever = SemanticRetriever(embed_fn=fake_embed)
    retriever.index(SAMPLE_CHUNKS)

    results = retriever.retrieve("encryption password server", top_k=1)

    assert results[0].id == "c2"


def test_hybrid_retriever_fuses_results():
    semantic = SemanticRetriever(embed_fn=fake_embed)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(semantic, bm25)

    hybrid.index(SAMPLE_CHUNKS)
    results = hybrid.retrieve("privileged access control review", top_k=2)

    result_ids = [c.id for c in results]
    assert "c1" in result_ids          # relevant chunk should surface
    assert "c3" not in result_ids      # irrelevant chunk should be excluded
    assert len(results) == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
