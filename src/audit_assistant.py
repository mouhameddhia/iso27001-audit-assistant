"""High-level entry point for the full system.

    result = AuditAssistant.from_settings().analyze(auditor_input)

Orchestrates, without duplicating the logic of any of them:

    Anonymization -> BM25 + Semantic -> RRF Fusion -> Cross-Encoder -> Generation -> Grounding

Each stage stays independently usable and independently tested; this module only wires them in
order and adds the one thing that only makes sense at this level: restoring the auditor's original
wording in the final result. The LLM never sees real client data -- deanonymization happens after
validation, as a local string substitution, with no AI involved.
"""

from dataclasses import dataclass
from typing import Optional

from src.anonymization.anonymizer import Anonymizer
from src.config import Settings, get_settings
from src.generation.generator import FindingGenerator, GenerationError
from src.generation.models import AuditFinding, EvidenceStatus, FindingType
from src.generation.validation import GroundingValidator
from src.retrieval.pipeline import SecureRetrievalPipeline
from src.retrieval.types import Retriever


def _deanonymize_finding(finding: AuditFinding, anonymizer: Anonymizer) -> AuditFinding:
    """Restores original values using the anonymizer's full *cumulative* mapping, not just the
    placeholders introduced by this one `analyze()` call. Needed once a session anonymizes more
    than the current call's query -- e.g. imported client documents -- so a placeholder the model
    echoes from that earlier context still gets restored here."""
    if not anonymizer.mapping:
        return finding
    restore = lambda text: anonymizer.deanonymize(text) if text else text
    return finding.model_copy(update={
        "observation": restore(finding.observation),
        "finding": restore(finding.finding),
        "requirement": restore(finding.requirement),
        "justification": restore(finding.justification),
        "risk": restore(finding.risk),
        "recommendation": restore(finding.recommendation),
    })


def _generation_failure_finding(observation: str, error: str) -> AuditFinding:
    """Never fabricate a finding when the generator itself failed (Ollama down, bad output, ...)."""
    return AuditFinding(
        observation=observation, finding=observation, finding_type=FindingType.CONSTAT,
        requirement=None, iso_reference=[], justification=None, risk=None, recommendation=None,
        evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True, confidence=0.0,
        supporting_sources=[], validation_notes=[f"Generation failed, no finding was drafted: {error}"],
    )


@dataclass
class AuditAssistant:
    retrieval: SecureRetrievalPipeline
    generator: FindingGenerator
    validator: GroundingValidator

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None, **retrieval_kwargs) -> "AuditAssistant":
        settings = settings or get_settings()
        return cls(
            retrieval=SecureRetrievalPipeline.from_settings(settings, **retrieval_kwargs),
            generator=FindingGenerator.from_settings(settings),
            validator=GroundingValidator(),
        )

    def analyze(
        self, auditor_input: str, language: str = "fr", document_retriever: Optional[Retriever] = None,
    ) -> AuditFinding:
        """`document_retriever` is a session's own imported-document retriever (see
        documents/retrieval.py), or None for a session with no confirmed imported documents --
        the default, which keeps behaviour byte-for-byte what it is without one. When given, it is
        queried with the *same anonymized query* already used for the ISO evidence, so its results
        line up with the same placeholder mapping (both go through the same session anonymizer)."""
        retrieval_result = self.retrieval.run(auditor_input)
        document_evidence = (
            document_retriever.retrieve(retrieval_result.anonymized_query, top_k=3)
            if document_retriever is not None else []
        )
        try:
            generated = self.generator.generate(
                retrieval_result.anonymized_query, retrieval_result.context, language, document_evidence,
            )
        except GenerationError as exc:
            finding = _generation_failure_finding(retrieval_result.anonymized_query, str(exc))
        else:
            finding = self.validator.validate(
                retrieval_result.anonymized_query, generated, retrieval_result.context, document_evidence,
            )
        return _deanonymize_finding(finding, self.retrieval.anonymizer)
