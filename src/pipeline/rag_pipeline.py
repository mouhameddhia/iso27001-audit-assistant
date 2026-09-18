"""
rag_pipeline.py

Wires the full pipeline together:

    user input
      -> anonymization
      -> hybrid retrieval (semantic + BM25, fused with RRF)
      -> Ollama generation (grounded on retrieved evidence)
      -> deanonymization (restore original sensitive values)

This module doesn't invent any new logic; it just calls the other
modules in the right order and passes data between them. Keeping it
"dumb" like this makes each step easy to test and swap out on its own.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.anonymization.anonymizer import Anonymizer
from src.generation.ollama_generator import generate, Language, GenerationResult
from src.ingestion.document_loader import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class PipelineResult:
    answer: str                 # final answer, with sensitive info restored
    sources: List[Chunk]        # retrieved chunks used for grounding
    anonymization_map: Dict[str, str]  # kept for debugging/audit trail only


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        anonymizer: Anonymizer = None,
        ollama_model: str = "llama3",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.retriever = retriever
        self.anonymizer = anonymizer or Anonymizer()
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url

    def run(self, user_text: str, language: Language = "fr", top_k: int = 5) -> PipelineResult:
        # 1) Anonymization: strip sensitive info before anything leaves this function
        anonymized_text, mapping = self.anonymizer.anonymize(user_text)

        # 2) Hybrid retrieval: get relevant ISO/cybersecurity evidence chunks
        retrieved_chunks = self.retriever.retrieve(anonymized_text, top_k=top_k)

        # 3) Generation: local LLM reformulates, grounded in the evidence only
        generation_result: GenerationResult = generate(
            anonymized_query=anonymized_text,
            chunks=retrieved_chunks,
            language=language,
            model=self.ollama_model,
            base_url=self.ollama_base_url,
        )

        # 4) Deanonymization: restore the real client/IP/server/user values
        restored_answer = self.anonymizer.deanonymize(generation_result.answer, mapping)

        return PipelineResult(
            answer=restored_answer,
            sources=generation_result.sources,
            anonymization_map=mapping,
        )
