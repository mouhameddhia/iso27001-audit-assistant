"""
End-to-end test of the full pipeline:
  anonymize -> hybrid retrieval -> Ollama generation -> deanonymize

Both Ollama calls (embeddings + chat generation) are mocked so this runs
without Ollama installed. This mirrors exactly the example from the spec:

    Input:  "Chez ABC Consulting, le serveur 192.168.1.20 n'est pas
             correctement protégé."
    Sent to the LLM: "Chez CLIENT_001, le serveur SERVER_001 n'est pas
             correctement protégé."
    Final answer: original client/server values restored.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anonymization.anonymizer import Anonymizer
from src.ingestion.document_loader import load_and_chunk_documents
from src.pipeline.rag_pipeline import RagPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.semantic_retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def fake_ollama_embed(texts, model="nomic-embed-text", base_url="http://localhost:11434"):
    """Deterministic fake embedding: bag-of-words over a small vocabulary."""
    vocab = ["access", "control", "privileged", "encryption", "server", "review"]
    return [[t.lower().count(word) + 1 for word in vocab] for t in texts]


def test_full_pipeline_anonymizes_retrieves_generates_and_restores():
    # 1) Real ingestion, on the actual sample documents in data/raw/
    chunks = load_and_chunk_documents(RAW_DIR)
    assert len(chunks) > 0, "expected at least the placeholder sample docs to be ingested"

    # 2) Build hybrid retriever with a fake (but deterministic) embedder
    semantic = SemanticRetriever(embed_fn=fake_ollama_embed)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(semantic, bm25)
    hybrid.index(chunks)

    # 3) Anonymizer knows this engagement's client name
    anonymizer = Anonymizer(known_entities={"CLIENT": ["ABC Consulting"]})

    pipeline = RagPipeline(retriever=hybrid, anonymizer=anonymizer)

    # 4) Mock only the Ollama /api/chat call used by the generator
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "message": {
            "content": (
                "Chez CLIENT_001, le serveur SERVER_001 ne fait pas l'objet "
                "d'une revue périodique des droits d'accès privilégiés, "
                "contrairement aux exigences de l'ISO/IEC 27001:2022 Annexe A.5.15."
            )
        }
    }

    user_text = "Chez ABC Consulting, le serveur SRV-PROD-01 n'est pas correctement protégé."

    with patch("src.generation.ollama_generator.requests.post", return_value=fake_response) as mock_post:
        result = pipeline.run(user_text, language="fr", top_k=3)

    # --- The LLM must NEVER see the real client/server names ---
    sent_payload = mock_post.call_args.kwargs["json"]
    sent_user_message = sent_payload["messages"][1]["content"]
    assert "ABC Consulting" not in sent_user_message
    assert "SRV-PROD-01" not in sent_user_message
    assert "CLIENT_001" in sent_user_message
    assert "SERVER_001" in sent_user_message

    # --- The final answer must have the real values restored ---
    assert "ABC Consulting" in result.answer
    assert "SRV-PROD-01" in result.answer
    assert "CLIENT_001" not in result.answer
    assert "SERVER_001" not in result.answer

    # --- Retrieved evidence should be returned for grounding/citations ---
    assert len(result.sources) > 0
    assert all(source.source.endswith((".txt", ".md", ".pdf")) for source in result.sources)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
