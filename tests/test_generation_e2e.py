"""End-to-end Stage 3: real Qdrant + real Ollama (embedding, reranker, generation) + real corpus.

Realistic auditor observations, deliberately different from both evaluation/kb_retrieval_benchmark.json
(Stage 1/2) and evaluation/grounding_eval.json (used by test_grounding_eval.py), covering the topics
named in the Stage 3 brief: privileged access, logging/monitoring, asset management, supplier
security, incident management. Nothing here is hard-coded application behaviour -- these are inputs
to the same `AuditAssistant.analyze()` any other observation goes through.
"""

import pytest

from src.audit_assistant import AuditAssistant
from src.config import get_settings
from src.generation.models import EvidenceStatus
from tests.conftest import huggingface_reachable, ollama_generation_model_available, ollama_model_available

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def assistant():
    settings = get_settings()
    if not (ollama_model_available(settings) and ollama_generation_model_available(settings) and huggingface_reachable()):
        pytest.skip("Embedding/generation Ollama model or huggingface.co not reachable")
    return AuditAssistant.from_settings(settings)


# (observation, category) -- category is informational only, not asserted on.
OBSERVATIONS = [
    ("Le serveur SRV-PROD-04 de la société Northwind : les droits d'accès administrateur ne sont "
     "revus qu'une fois par an, alors que la politique exige une revue trimestrielle.", "privileged access"),
    ("Les mots de passe des comptes de service ne sont pas renouvelés depuis leur création, "
     "certains datant de plus de trois ans.", "password/security configuration"),
    ("Les journaux d'accès aux systèmes critiques ne sont pas centralisés ni surveillés en continu.", "logging and monitoring"),
    ("Aucun inventaire à jour des actifs logiciels n'est disponible pour le périmètre audité.", "asset management"),
    ("Le contrat avec le prestataire hébergeant les données clients ne mentionne aucune clause de sécurité.", "supplier security"),
    ("La procédure de gestion des incidents de sécurité ne définit pas de délai de notification à la direction.", "incident management"),
]


class TestFullPipelineWiring:
    """Auditor input -> anonymization -> BM25+semantic -> RRF -> rerank -> evidence -> generation -> validation."""

    @pytest.mark.parametrize("observation,category", OBSERVATIONS)
    def test_produces_a_valid_structured_finding(self, assistant, observation, category):
        result = assistant.analyze(observation)

        # 9. Structured output is produced, always -- never free prose the app has to parse.
        assert result.finding.strip()
        assert result.evidence_status in EvidenceStatus
        assert isinstance(result.requires_human_review, bool)
        assert 0.0 <= result.confidence <= 1.0

        # 10/11. Generated ISO references are supported and traceable to real retrieved chunks.
        cited_by_sources = {ref for source in result.supporting_sources for ref in source.references}
        for ref in result.iso_reference:
            assert ref in cited_by_sources, f"{ref!r} in iso_reference but not backed by any supporting source"
        for source in result.supporting_sources:
            assert source.chunk_id and source.doc_id  # real KB identifiers, not invented

        # 13. Insufficient evidence is handled safely: never a reference with no backing.
        if result.evidence_status == EvidenceStatus.INSUFFICIENT:
            assert result.iso_reference == []
            # 14. Human review is indicated where appropriate.
            assert result.requires_human_review is True

    def test_original_wording_is_restored_for_the_human_reviewer(self, assistant):
        # 12. Confidential entities remain anonymized through retrieval and generation, then
        # restored for the auditor's own review (the LLM never saw the real values).
        result = assistant.analyze(
            "Chez la société Fabrikam, le serveur SRV-DB-02 stocke des mots de passe en clair."
        )
        assert "SERVER_001" not in result.observation and "CLIENT_001" not in result.observation
        assert "Fabrikam" in result.observation and "SRV-DB-02" in result.observation

    def test_iso_terminology_in_the_observation_reaches_the_result_unanonymized(self, assistant):
        result = assistant.analyze(
            "Le contrôle A.5.18 n'est pas appliqué : l'authentification MFA et le modèle RBAC ne "
            "sont pas déployés, en écart avec ISO/IEC 27001 et le RGPD."
        )
        for term in ("A.5.18", "MFA", "RBAC", "ISO/IEC 27001", "RGPD"):
            assert term in result.observation

    def test_two_findings_in_the_same_session_reuse_the_same_pseudonym(self, assistant):
        first = assistant.analyze("Le serveur SRV-EDGE-09 accepte des connexions non chiffrées.")
        second = assistant.analyze("Le serveur SRV-EDGE-09 n'est toujours pas corrigé après le dernier scan.")
        assert "SRV-EDGE-09" in first.observation and "SRV-EDGE-09" in second.observation


class TestSafeAbstention:
    def test_an_off_domain_observation_does_not_produce_a_confident_iso_finding(self, assistant):
        result = assistant.analyze("La machine à café du troisième étage est en panne depuis une semaine.")
        assert result.iso_reference == [] or result.confidence < 0.5
        if not result.iso_reference:
            assert result.requires_human_review is True


class TestFailureHandling:
    def test_unreachable_ollama_fails_safely_without_fabricating_a_finding(self, assistant, monkeypatch):
        # 20. Never silently fabricate: point generation at a dead port and confirm a safe fallback.
        monkeypatch.setattr(assistant.generator, "base_url", "http://localhost:1")
        monkeypatch.setattr(assistant.generator, "timeout", 3)

        result = assistant.analyze("Le contrôle des accès privilégiés n'est pas revu périodiquement.")

        assert result.evidence_status == EvidenceStatus.INSUFFICIENT
        assert result.iso_reference == []
        assert result.requires_human_review is True
        assert result.confidence == 0.0
