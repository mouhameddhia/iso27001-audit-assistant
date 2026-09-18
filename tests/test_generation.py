"""
Tests for the generation module. Run with: pytest tests/test_generation.py -v

We mock the HTTP call to Ollama so this test runs without Ollama installed.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.document_loader import Chunk
from src.generation.ollama_generator import generate, _build_user_prompt, SYSTEM_PROMPT


SAMPLE_CHUNKS = [
    Chunk(id="c1", source="iso27001.txt", text="Access control policy must be reviewed regularly."),
]


def test_build_user_prompt_includes_query_and_evidence():
    prompt = _build_user_prompt("CLIENT_001 has weak access control.", SAMPLE_CHUNKS, "en")
    assert "CLIENT_001 has weak access control." in prompt
    assert "Access control policy must be reviewed regularly." in prompt
    assert "iso27001.txt" in prompt


def test_generate_calls_ollama_and_returns_sources():
    fake_response = MagicMock()
    fake_response.json.return_value = {"message": {"content": "Reformulated finding."}}
    fake_response.raise_for_status.return_value = None

    with patch("src.generation.ollama_generator.requests.post", return_value=fake_response) as mock_post:
        result = generate("CLIENT_001 has weak access control.", SAMPLE_CHUNKS, language="en", model="llama3")

    assert result.answer == "Reformulated finding."
    assert result.sources == SAMPLE_CHUNKS

    # check the system prompt (the "never invent facts" instruction) was sent
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["messages"][0]["content"] == SYSTEM_PROMPT
    assert sent_payload["model"] == "llama3"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
