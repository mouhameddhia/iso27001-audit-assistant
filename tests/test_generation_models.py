"""The structured-output schema: what the LLM can and cannot express."""

import pytest
from pydantic import ValidationError

from src.generation.models import AuditFinding, EvidenceStatus, FindingType, GeneratedFinding, SupportingSource


def test_generated_finding_requires_no_traceability_fields():
    """The model's schema has no field for a chunk id or a confidence number -- it cannot invent either."""
    schema = GeneratedFinding.model_json_schema()
    assert set(schema["properties"]) == {
        "finding", "finding_type", "requirement", "iso_reference",
        "justification", "risk", "recommendation", "evidence_status", "requires_human_review",
    }


def test_generated_finding_rejects_an_unknown_evidence_status():
    with pytest.raises(ValidationError):
        GeneratedFinding(finding="x", finding_type=FindingType.CONSTAT, evidence_status="very_sure", requires_human_review=False)


def test_generated_finding_rejects_an_unknown_finding_type():
    with pytest.raises(ValidationError):
        GeneratedFinding(finding="x", finding_type="bug_report", evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True)


def test_generated_finding_parses_from_ollama_style_json():
    raw = """{"finding": "x", "finding_type": "non_conformite", "iso_reference": ["ISO/IEC 27001 A.5.18"],
              "evidence_status": "supported", "requires_human_review": false}"""
    parsed = GeneratedFinding.model_validate_json(raw)
    assert parsed.iso_reference == ["ISO/IEC 27001 A.5.18"]
    assert parsed.requirement is None and parsed.justification is None and parsed.recommendation is None
    assert parsed.risk is None


def test_audit_finding_confidence_is_bounded():
    with pytest.raises(ValidationError):
        AuditFinding(
            observation="x", finding="x", finding_type=None, requirement=None, iso_reference=[],
            justification=None, risk=None, recommendation=None, evidence_status=EvidenceStatus.INSUFFICIENT,
            requires_human_review=True, confidence=1.5, supporting_sources=[],
        )


def test_supporting_source_defaults_to_no_references():
    source = SupportingSource(chunk_id="c1", doc_id="d1", doc_title="Doc", rerank_score=0.5)
    assert source.references == []
