"""FindingGenerator: turns evidence into a structured draft. Never retrieves, never ranks."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.generation.generator import FindingGenerator, GenerationError
from src.generation.models import EvidenceStatus, FindingType
from tests.conftest import huggingface_reachable, make_hit, ollama_generation_model_available

VALID_RESPONSE = {
    "finding": "Le contrôle des accès privilégiés n'est pas revu périodiquement.",
    "finding_type": "non_conformite",
    "iso_reference": ["ISO/IEC 27001 A.5.18"],
    "evidence_status": "supported",
    "requires_human_review": False,
}


def fake_ollama_response(message_content: str):
    response = MagicMock()
    response.json.return_value = {"message": {"content": message_content}}
    response.raise_for_status.return_value = None
    return response


def make_generator(**overrides) -> FindingGenerator:
    params = dict(model="llama3:8b", base_url="http://ollama:11434", temperature=0.1, max_tokens=400, timeout=60)
    return FindingGenerator(**{**params, **overrides})


EVIDENCE = [make_hit("doc-a:s05:c00", "Preuve.", references=["ISO/IEC 27001 A.5.18"])]


class TestEmptyEvidenceShortCircuit:
    def test_no_evidence_never_calls_ollama(self):
        with patch("src.generation.generator.requests.post") as post:
            result = make_generator().generate("Une observation.", [])
        post.assert_not_called()
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.iso_reference == []
        assert result.requires_human_review is True

    def test_no_evidence_result_echoes_the_observation(self):
        result = make_generator().generate("Mon observation exacte.", [])
        assert result.finding == "Mon observation exacte."


class TestRequestConstruction:
    def test_sends_model_temperature_and_schema_from_settings(self):
        with patch("src.generation.generator.requests.post", return_value=fake_ollama_response(json.dumps(VALID_RESPONSE))) as post:
            make_generator(model="llama3:8b", temperature=0.2, max_tokens=123, timeout=45).generate("obs", EVIDENCE)

        call = post.call_args
        assert call.args[0] == "http://ollama:11434/api/chat"
        payload = call.kwargs["json"]
        assert payload["model"] == "llama3:8b"
        assert payload["options"] == {"temperature": 0.2, "num_predict": 123}
        assert payload["format"]["required"] == ["finding", "finding_type", "evidence_status", "requires_human_review"]
        assert call.kwargs["timeout"] == 45

    def test_sends_system_and_user_messages(self):
        with patch("src.generation.generator.requests.post", return_value=fake_ollama_response(json.dumps(VALID_RESPONSE))) as post:
            make_generator().generate("mon observation", EVIDENCE, language="en")

        messages = post.call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system" and "n'invente jamais" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "mon observation" in messages[1]["content"] and "English" in messages[1]["content"]


class TestResponseHandling:
    def test_valid_json_is_parsed_into_generated_finding(self):
        with patch("src.generation.generator.requests.post", return_value=fake_ollama_response(json.dumps(VALID_RESPONSE))):
            result = make_generator().generate("obs", EVIDENCE)
        assert result.finding_type == FindingType.NON_CONFORMITE
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]
        assert result.evidence_status == EvidenceStatus.SUPPORTED

    def test_malformed_json_raises_generation_error(self):
        with patch("src.generation.generator.requests.post", return_value=fake_ollama_response("not json at all")):
            with pytest.raises(GenerationError, match="invalid structured output"):
                make_generator().generate("obs", EVIDENCE)

    def test_json_missing_a_required_field_raises_generation_error(self):
        incomplete = {k: v for k, v in VALID_RESPONSE.items() if k != "evidence_status"}
        with patch("src.generation.generator.requests.post", return_value=fake_ollama_response(json.dumps(incomplete))):
            with pytest.raises(GenerationError, match="invalid structured output"):
                make_generator().generate("obs", EVIDENCE)

    def test_http_failure_is_wrapped_in_generation_error(self):
        with patch("src.generation.generator.requests.post", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(GenerationError, match="model=llama3:8b"):
                make_generator().generate("obs", EVIDENCE)

    def test_http_error_status_is_wrapped_in_generation_error(self):
        response = fake_ollama_response(json.dumps(VALID_RESPONSE))
        response.raise_for_status.side_effect = requests.HTTPError("500 server error")
        with patch("src.generation.generator.requests.post", return_value=response):
            with pytest.raises(GenerationError):
                make_generator().generate("obs", EVIDENCE)

    def test_timeout_is_wrapped_in_generation_error(self):
        with patch("src.generation.generator.requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(GenerationError):
                make_generator().generate("obs", EVIDENCE)


def test_from_settings_reads_generation_config(settings):
    generator = FindingGenerator.from_settings(settings)
    assert generator.model == settings.generation_model
    assert generator.base_url == settings.generation_base_url
    assert generator.temperature == settings.generation_temperature


@pytest.mark.integration
def test_live_model_returns_valid_structured_output(settings):
    if not ollama_generation_model_available(settings):
        pytest.skip(f"Ollama model {settings.generation_model} not available at {settings.generation_base_url}")
    generator = FindingGenerator.from_settings(settings)

    result = generator.generate(
        "Le contrôle des accès privilégiés n'est pas revu périodiquement.",
        [make_hit("doc-iso27001-2022-full:s05:c01",
                  "NC-2026-014 (Mineure)\nRéférence : A.5.18 - Droits d'accès\n"
                  "Énoncé : La revue périodique des droits d'accès privilégiés n'a pas été réalisée.",
                  doc_id="doc-iso27001-2022-full", doc_title="ISO/IEC 27001:2022",
                  section_title="NON-CONFORMITÉS", references=["ISO/IEC 27001 A.5.18"])],
    )

    assert result.finding.strip()
    assert result.evidence_status in EvidenceStatus
    assert isinstance(result.requires_human_review, bool)
