from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt, format_document_evidence, format_evidence
from tests.conftest import make_hit


def test_format_evidence_lists_chunk_id_source_and_available_references():
    hit = make_hit("doc-a:s05:c00", "Le texte de la preuve.", doc_title="ISO/IEC 27001:2022 - Exigences",
                   section_title="NON-CONFORMITÉS", references=["ISO/IEC 27001 A.5.18"])
    formatted = format_evidence([hit])
    assert "chunk_id=doc-a:s05:c00" in formatted
    assert "ISO/IEC 27001:2022 - Exigences" in formatted and "NON-CONFORMITÉS" in formatted
    assert "ISO/IEC 27001 A.5.18" in formatted
    assert "Le texte de la preuve." in formatted


def test_format_evidence_marks_chunks_with_no_references():
    hit = make_hit("doc-a:s01:c00", "Texte sans référence.", references=[])
    assert "références disponibles : aucune" in format_evidence([hit])


def test_format_evidence_of_empty_list_says_no_evidence():
    assert format_evidence([]) == "(aucune preuve retrouvée)"


def test_user_prompt_includes_observation_and_evidence():
    hit = make_hit("doc-a:s05:c00", "Preuve pertinente.", references=["ISO/IEC 27001 A.5.18"])
    prompt = build_user_prompt("L'accès n'est pas revu.", [hit], language="fr")
    assert "L'accès n'est pas revu." in prompt
    assert "Preuve pertinente." in prompt
    assert "français" in prompt


def test_user_prompt_requests_english_when_asked():
    assert "English" in build_user_prompt("obs", [], language="en")


def test_system_prompt_forbids_inventing_references_and_facts():
    lowered = SYSTEM_PROMPT.lower()
    assert "n'invente jamais" in lowered
    assert "insufficient" in lowered or "insuffisant" in lowered


class TestDocumentEvidenceBlock:
    """The document-context instruction lives entirely in the *user* prompt, conditional on
    `document_evidence`, never in the fixed SYSTEM_PROMPT -- so a document-less call (still the
    overwhelming majority of calls) sees byte-for-byte the same prompt Stage 3 already measured
    and validated. A live comparison (6 trials each way against the grounding eval's off-domain
    case D-01) showed this fixed SYSTEM_PROMPT wording is not the same as adding an always-present
    rule about content that, for that call, was never there."""

    def test_absent_from_the_system_prompt(self):
        assert "contexte documentaire du client" not in SYSTEM_PROMPT.lower()

    def test_absent_from_the_user_prompt_when_no_document_evidence_is_given(self):
        hit = make_hit("doc-a:s05:c00", "Preuve ISO.", references=["ISO/IEC 27001 A.5.18"])
        prompt = build_user_prompt("obs", [hit], language="fr")
        assert "information de fond uniquement" not in prompt.lower()
        assert "contexte documentaire du client" not in prompt.lower()

    def test_present_and_labelled_when_document_evidence_is_given(self):
        iso_hit = make_hit("doc-a:s05:c00", "Preuve ISO.", references=["ISO/IEC 27001 A.5.18"])
        doc_hit = make_hit("doc-client:s01:c00", "Le pare-feu accepte TLS 1.0.", doc_title="Rapport precedent")
        prompt = build_user_prompt("obs", [iso_hit], language="fr", document_evidence=[doc_hit])
        assert "information de fond uniquement" in prompt.lower()
        assert "contexte documentaire du client" in prompt.lower()
        assert "mot pour mot" in prompt.lower()
        assert "Rapport precedent" in prompt
        assert "Le pare-feu accepte TLS 1.0." in prompt

    def test_format_document_evidence_of_empty_list_is_empty_string(self):
        assert format_document_evidence([]) == ""
