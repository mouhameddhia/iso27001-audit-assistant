"""
ollama_generator.py

Calls a local LLM through Ollama to reformulate/improve the auditor's
(anonymized) text, grounded in the retrieved ISO/cybersecurity chunks.

The system prompt strictly instructs the model to:
- only improve wording, never change technical meaning
- use only the provided evidence
- never invent facts, findings, controls, risks, or ISO references
- say so if the evidence is insufficient
- answer in the language requested (French or English)
"""

from dataclasses import dataclass
from typing import List, Literal
import requests

from src.ingestion.document_loader import Chunk

Language = Literal["fr", "en"]

SYSTEM_PROMPT = """You are an ISO/IEC 27001 audit-report writing assistant.
Improve and reformulate the auditor's text without changing its technical meaning.
Use only the provided evidence.
Do not invent facts, findings, controls, risks, or ISO references.
If the evidence is insufficient, say so.
Answer in the language requested by the auditor (French or English)."""


@dataclass
class GenerationResult:
    answer: str
    sources: List[Chunk]  # the chunks actually given to the model, for grounding/citations


def _build_context(chunks: List[Chunk]) -> str:
    """Format retrieved chunks as a numbered evidence block for the prompt."""
    if not chunks:
        return "(no evidence retrieved)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n".join(parts)


def _build_user_prompt(anonymized_query: str, chunks: List[Chunk], language: Language) -> str:
    language_name = "French" if language == "fr" else "English"
    return f"""Auditor's request (respond in {language_name}):
\"\"\"{anonymized_query}\"\"\"

Retrieved evidence (ISO/IEC 27001 and related documents):
{_build_context(chunks)}

Task: Improve/reformulate the auditor's text above using only the evidence
provided. Do not add findings, risks, or ISO clause references that are not
supported by the evidence. If the evidence does not cover something the
auditor mentioned, explicitly say the evidence is insufficient for that
part rather than guessing."""


def generate(
    anonymized_query: str,
    chunks: List[Chunk],
    language: Language = "fr",
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
) -> GenerationResult:
    """Send the anonymized query + retrieved evidence to a local Ollama model.

    Requires Ollama running locally with `model` pulled, e.g.:
        ollama pull llama3
    """
    user_prompt = _build_user_prompt(anonymized_query, chunks, language)

    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"]

    return GenerationResult(answer=answer, sources=chunks)
