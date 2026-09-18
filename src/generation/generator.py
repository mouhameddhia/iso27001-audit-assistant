"""Grounded finding generation through a local Ollama chat model.

Takes an anonymized observation and already-ranked evidence (from Stage 2) and asks the model for
one `GeneratedFinding`. Never queries Qdrant, never retrieves, never ranks -- it only turns
evidence it is handed into a structured draft. Structural safety, not just prompting: the model's
JSON schema (`GeneratedFinding`) has no field for a chunk id or a confidence score, so it has no
way to invent either; `GroundingValidator` (validation.py) adds both afterwards, deterministically.
"""

import json
from typing import Sequence

import pydantic
import requests

from src.config import Settings
from src.generation.models import EvidenceStatus, FindingType, GeneratedFinding
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.vectorstore import SearchHit

FINDING_SCHEMA = GeneratedFinding.model_json_schema()


class GenerationError(RuntimeError):
    pass


def _no_evidence_finding(observation: str) -> GeneratedFinding:
    """Short-circuit for empty retrieval: no LLM call, so there is nothing to hallucinate from."""
    return GeneratedFinding(
        finding=observation,
        finding_type=FindingType.CONSTAT,
        iso_reference=[],
        evidence_status=EvidenceStatus.INSUFFICIENT,
        requires_human_review=True,
    )


class FindingGenerator:
    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int, timeout: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> "FindingGenerator":
        return cls(
            model=settings.generation_model, base_url=settings.generation_base_url,
            temperature=settings.generation_temperature, max_tokens=settings.generation_max_tokens,
            timeout=settings.generation_timeout,
        )

    def generate(
        self, observation: str, evidence: Sequence[SearchHit], language: str = "fr",
        document_evidence: Sequence[SearchHit] = (),
    ) -> GeneratedFinding:
        if not evidence:
            # A finding still needs real ISO evidence to be drafted at all -- client-document
            # context alone, with no ISO evidence, is never enough (see prompts.py rule 6).
            return _no_evidence_finding(observation)

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": FINDING_SCHEMA,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(observation, evidence, language, document_evidence)},
                    ],
                    "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GenerationError(f"Ollama chat request failed ({self.base_url}, model={self.model}): {exc}") from exc

        content = response.json()["message"]["content"]
        try:
            return GeneratedFinding.model_validate_json(content)
        except (pydantic.ValidationError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Model returned invalid structured output: {exc}") from exc
