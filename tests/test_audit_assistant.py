"""AuditAssistant: orchestration only -- it wires real components together and adds exactly one
thing that only makes sense at this level (restoring the auditor's original wording on output).
No retrieval logic, no generation logic, no validation logic lives here.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pytest

from src.anonymization.anonymizer import Anonymizer
from src.audit_assistant import AuditAssistant
from src.generation.generator import GenerationError
from src.generation.models import EvidenceStatus, FindingType, GeneratedFinding
from src.generation.validation import GroundingValidator
from src.vectorstore import SearchHit
from tests.conftest import huggingface_reachable, make_hit, ollama_generation_model_available, ollama_model_available


@dataclass
class FakeRetrievalResult:
    anonymized_query: str
    anonymization_map: Dict[str, str]
    candidates: List[SearchHit]
    context: List[SearchHit]


@dataclass
class FakeRetrievalPipeline:
    """Stands in for SecureRetrievalPipeline: AuditAssistant must only call `.run()` and read
    `.anonymizer` -- it never touches BM25/semantic/fusion/reranker directly."""

    anonymizer: Anonymizer
    result: FakeRetrievalResult

    def run(self, auditor_text: str) -> FakeRetrievalResult:
        # A real anonymizer.anonymize() call updates self.mapping as a side effect; mirror that so
        # AuditAssistant's use of self.retrieval.anonymizer.mapping (the cumulative map) is exercised
        # the same way it would be against the real SecureRetrievalPipeline.
        self.anonymizer.mapping.update(self.result.anonymization_map)
        return self.result


@dataclass
class FakeGenerator:
    response: GeneratedFinding = None
    error: Exception = None
    calls: list = field(default_factory=list)

    def generate(self, observation, evidence, language="fr", document_evidence=()):
        self.calls.append((observation, evidence, language, document_evidence))
        if self.error:
            raise self.error
        return self.response


EVIDENCE = [make_hit("doc-a:s05:c01", "Preuve.", doc_id="doc-a", references=["ISO/IEC 27001 A.5.18"])]


def make_assistant(anonymization_map=None, generator_response=None, generator_error=None,
                   anonymized_query="obs anonymisée") -> AuditAssistant:
    anonymizer = Anonymizer()
    retrieval = FakeRetrievalPipeline(
        anonymizer=anonymizer,
        result=FakeRetrievalResult(anonymized_query, anonymization_map or {}, EVIDENCE, EVIDENCE),
    )
    generator = FakeGenerator(
        response=generator_response or GeneratedFinding(
            finding="Le constat.", finding_type=FindingType.NON_CONFORMITE,
            iso_reference=["ISO/IEC 27001 A.5.18"], evidence_status=EvidenceStatus.SUPPORTED,
            requires_human_review=False,
        ),
        error=generator_error,
    )
    return AuditAssistant(retrieval=retrieval, generator=generator, validator=GroundingValidator())


class TestOrchestration:
    def test_analyze_passes_the_anonymized_query_and_context_to_the_generator(self):
        assistant = make_assistant(anonymized_query="Le serveur SERVER_001 ...")
        assistant.analyze("Le serveur SRV-01 ...")
        observation, evidence, _, _ = assistant.generator.calls[0]
        assert observation == "Le serveur SERVER_001 ..."
        assert evidence == EVIDENCE

    def test_analyze_forwards_the_requested_language(self):
        assistant = make_assistant()
        assistant.analyze("obs", language="en")
        assert assistant.generator.calls[0][2] == "en"

    def test_result_is_validated_by_the_real_grounding_validator(self):
        assistant = make_assistant()
        result = assistant.analyze("obs")
        assert result.iso_reference == ["ISO/IEC 27001 A.5.18"]  # validator confirmed it against EVIDENCE


class TestDeanonymizationOnOutput:
    def test_original_entities_are_restored_in_the_final_finding(self):
        mapping = {"SERVER_001": "SRV-PROD-01", "CLIENT_001": "Contoso"}
        response = GeneratedFinding(
            finding="Chez CLIENT_001, le serveur SERVER_001 n'est pas conforme.",
            finding_type=FindingType.CONSTAT, iso_reference=[],
            evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True,
        )
        assistant = make_assistant(anonymization_map=mapping, generator_response=response,
                                   anonymized_query="Chez CLIENT_001, le serveur SERVER_001 n'est pas conforme.")

        result = assistant.analyze("Chez Contoso, le serveur SRV-PROD-01 n'est pas conforme.")

        assert "SERVER_001" not in result.finding and "CLIENT_001" not in result.finding
        assert "SRV-PROD-01" in result.finding and "Contoso" in result.finding
        assert "SRV-PROD-01" in result.observation and "Contoso" in result.observation

    def test_no_mapping_leaves_the_finding_unchanged(self):
        assistant = make_assistant(anonymization_map={})
        result = assistant.analyze("Le contrôle des accès n'est pas revu.")
        assert result.finding == "Le constat."

    def test_restores_a_placeholder_from_the_anonymizer_s_cumulative_mapping_not_just_this_call(self):
        """Regression test: a prior call (e.g. anonymizing an imported document) may have added a
        placeholder to the anonymizer's persistent mapping that this call's own local map doesn't
        include. The model can still echo it, and it must still be restored."""
        response = GeneratedFinding(
            finding="Voir le rapport precedent concernant SERVER_002.",
            finding_type=FindingType.CONSTAT, iso_reference=[],
            evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True,
        )
        assistant = make_assistant(anonymization_map={}, generator_response=response)
        assistant.retrieval.anonymizer.mapping["SERVER_002"] = "SRV-LEGACY-09"

        result = assistant.analyze("obs")

        assert "SERVER_002" not in result.finding
        assert "SRV-LEGACY-09" in result.finding

    def test_supporting_sources_are_kb_metadata_not_deanonymized(self):
        """Chunk/doc metadata comes from the knowledge base, never from the auditor's text."""
        mapping = {"SERVER_001": "SRV-PROD-01"}
        assistant = make_assistant(anonymization_map=mapping)
        result = assistant.analyze("obs")
        assert result.supporting_sources[0].chunk_id == "doc-a:s05:c01"
        assert result.supporting_sources[0].doc_id == "doc-a"

    def test_iso_terminology_is_never_touched_by_deanonymization(self):
        mapping = {"SERVER_001": "SRV-PROD-01"}
        response = GeneratedFinding(
            finding="Non-conformité au regard de ISO/IEC 27001 A.5.18, MFA et RBAC absents, cf. RGPD.",
            finding_type=FindingType.NON_CONFORMITE, iso_reference=["ISO/IEC 27001 A.5.18"],
            evidence_status=EvidenceStatus.SUPPORTED, requires_human_review=False,
        )
        assistant = make_assistant(anonymization_map=mapping, generator_response=response)
        result = assistant.analyze("obs")
        for term in ("ISO/IEC 27001 A.5.18", "MFA", "RBAC", "RGPD"):
            assert term in result.finding


class TestGenerationFailureIsSafe:
    def test_generation_error_never_fabricates_a_finding(self):
        assistant = make_assistant(generator_error=GenerationError("Ollama unreachable"))
        result = assistant.analyze("obs")
        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.iso_reference == []
        assert result.requires_human_review is True
        assert result.confidence == 0.0
        assert "Ollama unreachable" in result.validation_notes[0]

    def test_generation_error_still_deanonymizes_the_observation(self):
        mapping = {"SERVER_001": "SRV-PROD-01"}
        assistant = make_assistant(anonymization_map=mapping, generator_error=GenerationError("boom"),
                                   anonymized_query="Le serveur SERVER_001 ...")
        result = assistant.analyze("Le serveur SRV-PROD-01 ...")
        assert "SRV-PROD-01" in result.observation


class TestDocumentEvidence:
    def test_document_retriever_is_queried_with_the_same_anonymized_query_and_result_passed_through(self):
        document_hit = make_hit("doc-client:s01:c00", "Contexte client.", doc_id="doc-client")

        @dataclass
        class FakeDocumentRetriever:
            calls: list = field(default_factory=list)

            def retrieve(self, query, top_k):
                self.calls.append((query, top_k))
                return [document_hit]

        retriever = FakeDocumentRetriever()
        assistant = make_assistant(anonymized_query="Le serveur SERVER_001 ...")

        result = assistant.analyze("Le serveur SRV-01 ...", document_retriever=retriever)

        assert retriever.calls == [("Le serveur SERVER_001 ...", 3)]
        assert result.document_sources[0].chunk_id == "doc-client:s01:c00"
        # requires_human_review forced True by the real GroundingValidator once document evidence is used
        assert result.requires_human_review is True

    def test_no_document_retriever_leaves_document_sources_empty(self):
        assistant = make_assistant()
        result = assistant.analyze("obs")
        assert result.document_sources == []

    def test_a_placeholder_that_only_appeared_via_document_evidence_is_still_restored(self):
        """Regression test for the deanonymization fix: a placeholder introduced while anonymizing
        an imported document (not this call's own query) must still be resolved in the output."""
        document_hit = make_hit("doc-client:s01:c00", "Contexte.", doc_id="doc-client")
        response = GeneratedFinding(
            finding="Voir le rapport precedent concernant SERVER_002.",
            finding_type=FindingType.CONSTAT, iso_reference=[],
            evidence_status=EvidenceStatus.INSUFFICIENT, requires_human_review=True,
        )
        assistant = make_assistant(generator_response=response)
        assistant.retrieval.anonymizer.mapping["SERVER_002"] = "SRV-LEGACY-09"

        @dataclass
        class FakeDocumentRetriever:
            def retrieve(self, query, top_k):
                return [document_hit]

        result = assistant.analyze("obs", document_retriever=FakeDocumentRetriever())

        assert "SERVER_002" not in result.finding
        assert "SRV-LEGACY-09" in result.finding


class TestFromSettings:
    def test_builds_a_working_assistant(self, settings):
        assistant = AuditAssistant.from_settings(settings)
        assert assistant.retrieval is not None and assistant.generator is not None and assistant.validator is not None


@pytest.mark.integration
def test_live_assistant_analyzes_the_cahier_des_charges_example(settings):
    if not (ollama_model_available(settings) and ollama_generation_model_available(settings) and huggingface_reachable()):
        pytest.skip("Embedding/generation Ollama model or huggingface.co not reachable")

    assistant = AuditAssistant.from_settings(settings)
    result = assistant.analyze("Le contrôle des accès privilégiés n'est pas revu périodiquement.")

    assert result.finding.strip()
    assert result.evidence_status in EvidenceStatus
    for ref in result.iso_reference:
        assert any(ref in source.references for source in result.supporting_sources)
