"""Structured audit-finding schema (Pydantic): validated, not parsed from free-text prose.

Two models, deliberately not one:

- `GeneratedFinding` is exactly what the LLM is asked to produce (its JSON schema is sent to
  Ollama's structured-output `format` parameter). It carries no traceability or confidence: the
  model is never asked for a chunk id or a confidence score, so it cannot fabricate either.
- `AuditFinding` is the final, validated result. `GroundingValidator` (see validation.py) builds
  it from a `GeneratedFinding` plus the real retrieved evidence -- `supporting_sources` and
  `confidence` are computed by that deterministic step, never by the model.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"        # the cited ISO reference(s) are backed by the retrieved evidence
    PARTIAL = "partial"            # some support, but incomplete (e.g. a cited reference had to be dropped)
    INSUFFICIENT = "insufficient"  # not enough evidence to ground a specific ISO reference


class FindingType(str, Enum):
    CONSTAT = "constat"
    NON_CONFORMITE = "non_conformite"
    OBSERVATION = "observation"
    OPPORTUNITE_AMELIORATION = "opportunite_amelioration"


class GeneratedFinding(BaseModel):
    """The LLM's structured output. Field order matches the drafting flow: observation -> requirement -> assessment."""

    finding: str = Field(description="The drafted audit finding text, grounded in the evidence.")
    finding_type: FindingType
    requirement: Optional[str] = Field(default=None, description="The normative requirement the evidence describes.")
    iso_reference: List[str] = Field(default_factory=list, description="Canonical references copied verbatim from the evidence.")
    justification: Optional[str] = Field(default=None, description="Why the evidence supports (or does not support) the finding.")
    # The firm's own internal policy (SECTION 05, retrievable in the knowledge base) requires a
    # "Risque" field in every non-conformity write-up, and the cahier des charges' own worked
    # example (§4.1) names "Risque associé" as a distinct output alongside the finding and the ISO
    # reference -- not something to infer from `justification` alone.
    risk: Optional[str] = Field(default=None, description="The risk associated with this finding, grounded in the evidence.")
    recommendation: Optional[str] = None
    evidence_status: EvidenceStatus
    requires_human_review: bool


class SupportingSource(BaseModel):
    """One retrieved chunk that backs the finding, built by the validator from real evidence -- never by the model."""

    chunk_id: str
    doc_id: str
    doc_title: str
    references: List[str] = Field(default_factory=list)
    rerank_score: float


class AuditFinding(BaseModel):
    """Final, validated, traceable result returned by `AuditAssistant.analyze()`."""

    observation: str
    finding: str
    finding_type: Optional[FindingType]
    requirement: Optional[str]
    iso_reference: List[str]
    justification: Optional[str]
    risk: Optional[str]
    recommendation: Optional[str]
    evidence_status: EvidenceStatus
    requires_human_review: bool
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_sources: List[SupportingSource]
    # Imported client-document chunks (if any) shown to the model as background context -- never
    # a source of ISO citations. Always empty for a session with no confirmed imported documents.
    # Populated deterministically by GroundingValidator, like supporting_sources, never by the model.
    document_sources: List[SupportingSource] = Field(default_factory=list)
    validation_notes: List[str] = Field(default_factory=list)
