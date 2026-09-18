"""Grounding / hallucination-resistance evaluation for Stage 3 generation.

Not an LLM-as-judge: every metric here is a deterministic, structural check -- did the model try to
cite something the evidence doesn't contain, did that survive validation, did the system abstain
when it should have. "Relevance" (does the finding read well) is not scored here; there is no
reliable automatic way to judge that without another LLM call standing in for human judgment, which
would just move the trust problem rather than solve it. Read the generated text yourself.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from src.generation.generator import FindingGenerator, GenerationError
from src.generation.models import AuditFinding, EvidenceStatus, FindingType, GeneratedFinding
from src.generation.validation import GroundingValidator, reference_index

_LOW_CONFIDENCE_REVIEW_THRESHOLD = GroundingValidator.LOW_CONFIDENCE_REVIEW_THRESHOLD
from src.retrieval.pipeline import SecureRetrievalPipeline
from src.vectorstore import SearchHit


@dataclass(frozen=True)
class GroundingCase:
    id: str
    category: str
    observation: str
    expect_reference: bool
    note: str
    # None = not checked (a judgment call, like expectation_match_rate for B/C). Set for category F
    # (compliant observations): a live, reproducible failure mode where the model fabricated a
    # non-conformity -- and once, a contradicting "fact" not in the observation -- for an
    # observation that plainly describes a compliant situation.
    expect_non_conformite: bool | None = None


@dataclass(frozen=True)
class CaseResult:
    case: GroundingCase
    evidence: List[SearchHit]
    raw: GeneratedFinding      # exactly what the model returned, before any validation
    validated: AuditFinding    # what the system actually returns

    @property
    def raw_unsupported_references(self) -> List[str]:
        """References the model tried to cite that are absent from the evidence it was given --
        the model's own hallucination rate, before the safety net."""
        by_exact, by_local = reference_index(self.evidence)
        return [
            ref for ref in self.raw.iso_reference
            if ref.strip().lower() not in by_exact and len(by_local.get(ref.strip().lower(), [])) != 1
        ]

    @property
    def final_unsupported_references(self) -> List[str]:
        """Must always be empty: independently re-verifies the validator's guarantee, rather than
        trusting it structurally."""
        by_exact, by_local = reference_index(self.evidence)
        return [
            ref for ref in self.validated.iso_reference
            if ref.strip().lower() not in by_exact and len(by_local.get(ref.strip().lower(), [])) != 1
        ]

    @property
    def matched_expectation(self) -> bool:
        return bool(self.validated.iso_reference) == self.case.expect_reference

    @property
    def matched_non_conformite_expectation(self) -> bool | None:
        if self.case.expect_non_conformite is None:
            return None
        return (self.validated.finding_type == FindingType.NON_CONFORMITE) == self.case.expect_non_conformite


def load_cases(path: Path) -> List[GroundingCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GroundingCase(**item) for item in data["cases"]]


def run_case(
    case: GroundingCase, retrieval: SecureRetrievalPipeline, generator: FindingGenerator, validator: GroundingValidator,
) -> CaseResult:
    retrieval_result = retrieval.run(case.observation)
    try:
        raw = generator.generate(retrieval_result.anonymized_query, retrieval_result.context)
    except GenerationError as exc:
        raw = GeneratedFinding(
            finding=f"(generation failed: {exc})", finding_type="constat", iso_reference=[],
            evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True,
        )
    validated = validator.validate(retrieval_result.anonymized_query, raw, retrieval_result.context)
    return CaseResult(case=case, evidence=retrieval_result.context, raw=raw, validated=validated)


@dataclass(frozen=True)
class GroundingReport:
    results: List[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def expectation_match_rate(self) -> float:
        return sum(r.matched_expectation for r in self.results) / self.total

    @property
    def cases_with_raw_hallucination(self) -> int:
        """How many cases the model tried to cite at least one unsupported reference in, before validation."""
        return sum(bool(r.raw_unsupported_references) for r in self.results)

    @property
    def cases_with_final_unsupported_reference(self) -> int:
        """Must be 0: an unsupported reference reaching the returned AuditFinding is the one
        failure this whole layered design exists to prevent."""
        return sum(bool(r.final_unsupported_references) for r in self.results)

    @property
    def cases_with_weak_citation_not_flagged(self) -> int:
        """Must be 0: a real-but-weakly-backed reference (low reranker confidence) reaching the
        output without requires_human_review=True would defeat the point of computing confidence."""
        return sum(
            bool(r.validated.iso_reference)
            and r.validated.confidence < _LOW_CONFIDENCE_REVIEW_THRESHOLD
            and not r.validated.requires_human_review
            for r in self.results
        )

    @property
    def cases_with_unexpected_non_conformite(self) -> int:
        """Must be 0 for cases where it's checked (category F): a compliant observation must never
        be turned into a fabricated non-conformity, regardless of what evidence was retrieved."""
        return sum(r.matched_non_conformite_expectation is False for r in self.results)

    def by_category(self) -> dict:
        categories = {r.case.category for r in self.results}
        return {
            category: {
                "n": sum(r.case.category == category for r in self.results),
                "expectation_match_rate": sum(
                    r.matched_expectation for r in self.results if r.case.category == category
                ) / sum(r.case.category == category for r in self.results),
            }
            for category in sorted(categories)
        }


def run_grounding_eval(
    cases: Sequence[GroundingCase], retrieval: SecureRetrievalPipeline,
    generator: FindingGenerator, validator: GroundingValidator,
) -> GroundingReport:
    return GroundingReport([run_case(case, retrieval, generator, validator) for case in cases])
