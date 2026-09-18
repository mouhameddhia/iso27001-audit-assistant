"""Runs evaluation/grounding_eval.json (see assistant_cli.py evaluate-grounding for the human-
readable report) and asserts the safety properties this whole layered design exists to guarantee
-- the ones the *architecture* can actually enforce (strip an unsupported citation, force review).
`expectation_match_rate` and category F's classification accuracy are reported but not gated hard:
whether a real-but-weak reference should have been cited, or whether the model correctly classified
a compliant observation, are judgment calls / model-quality questions the validation layer cannot
itself fix -- only detect and flag for a human. See the README for the measured, honest breakdown,
including category F's residual risk (llama3:8b measurably still fabricates a non-conformity for
some compliant observations even after a targeted prompt fix -- always forced to human review, never
silently shipped, but not eliminated at this model's scale)."""

import pytest

from src.config import PROJECT_ROOT, get_settings
from src.evaluation.generation import load_cases, run_grounding_eval
from src.generation.generator import FindingGenerator
from src.generation.validation import GroundingValidator
from src.retrieval.pipeline import SecureRetrievalPipeline
from tests.conftest import huggingface_reachable, ollama_generation_model_available, ollama_model_available

pytestmark = pytest.mark.e2e

GROUNDING_EVAL_PATH = PROJECT_ROOT / "evaluation" / "grounding_eval.json"


@pytest.fixture(scope="module")
def report():
    settings = get_settings()
    if not (ollama_model_available(settings) and ollama_generation_model_available(settings) and huggingface_reachable()):
        pytest.skip("Embedding/generation Ollama model or huggingface.co not reachable")

    cases = load_cases(GROUNDING_EVAL_PATH)
    retrieval = SecureRetrievalPipeline.from_settings(settings)
    generator = FindingGenerator.from_settings(settings)
    return run_grounding_eval(cases, retrieval, generator, GroundingValidator())


def test_grounding_eval_cases_load_and_cover_every_category():
    cases = load_cases(GROUNDING_EVAL_PATH)
    assert len({c.id for c in cases}) == len(cases) > 0
    assert {c.category for c in cases} == {"A", "B", "C", "D", "E", "F"}


def test_no_unsupported_reference_ever_reaches_the_final_output(report):
    assert report.cases_with_final_unsupported_reference == 0, [
        (r.case.id, r.final_unsupported_references) for r in report.results if r.final_unsupported_references
    ]


def test_no_weakly_backed_citation_escapes_review(report):
    assert report.cases_with_weak_citation_not_flagged == 0, [
        (r.case.id, r.validated.confidence) for r in report.results
        if r.validated.iso_reference and not r.validated.requires_human_review
    ]


def test_off_domain_and_no_evidence_categories_abstain(report):
    """D (no useful retrieval): the system should not cite anything. Not gated for B/C, where a
    real-but-weakly-relevant reference is a judgment call, not a clear-cut abstention failure."""
    d_cases = [r for r in report.results if r.case.category == "D"]
    assert d_cases and all(r.matched_expectation for r in d_cases)


def test_fabricated_non_conformite_is_always_forced_to_human_review(report):
    """F: a real, live-reproduced failure mode (2026-09-17) -- the model can turn a plainly
    compliant observation into a fabricated non-conformity, sometimes asserting the opposite of
    what the auditor actually wrote or inventing a specific false detail. Classification accuracy
    itself is not something GroundingValidator can enforce (see `test_category_f_classification_
    accuracy_is_reported`); what it *can* and must guarantee is that such a finding never reaches a
    report without a human explicitly reviewing it -- the cahier des charges' own "validation
    humaine obligatoire" mitigation. Scoped to category F, matching what this actually guarantees:
    a genuinely clean, well-evidenced, non-contradictory finding elsewhere in the set (category A)
    is legitimately allowed to skip forced review -- that is the whole point of confidence-based
    review, not a gap in it."""
    f_cases = [r for r in report.results if r.case.category == "F"]
    assert f_cases and all(r.validated.requires_human_review for r in f_cases), [
        r.case.id for r in f_cases if not r.validated.requires_human_review
    ]


def test_insufficient_evidence_status_is_always_forced_to_human_review(report):
    """Live reconfirmation of a general invariant already unit-tested in isolation
    (`TestInsufficientEvidenceAlwaysForcesReview`, src/generation/validation.py): whatever the
    model's own requires_human_review flag says, a final evidence_status of "insufficient" must
    always mean a human looks at it."""
    insufficient_cases = [r for r in report.results if r.validated.evidence_status.value == "insufficient"]
    assert insufficient_cases and all(r.validated.requires_human_review for r in insufficient_cases), [
        r.case.id for r in insufficient_cases if not r.validated.requires_human_review
    ]


def test_category_f_classification_accuracy_is_reported(report):
    """Not gated hard (see module docstring): reports how often the model correctly avoided
    classifying a compliant observation as a non-conformity, so a regression or improvement is
    visible in the report even though it isn't a pass/fail architectural guarantee."""
    f_cases = [r for r in report.results if r.case.category == "F"]
    assert f_cases
    accuracy = sum(bool(r.matched_non_conformite_expectation) for r in f_cases) / len(f_cases)
    print(f"\nCategory F (compliant observations) classification accuracy: {accuracy:.0%} ({len(f_cases)} cases)")
