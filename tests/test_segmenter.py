import pytest

from src.ingestion.models import EXAMPLE, GUIDANCE, RECORD, TEMPLATE
from src.ingestion.parser import parse_document
from src.ingestion.pipeline import discover_documents
from src.ingestion.segmenter import classify_block, segment_section
from tests.conftest import REAL_KB_DIR

NC_MAJOR = ("NC-2026-015 (Majeure)\nRéférence : A.8.8 - Gestion des vulnérabilités techniques\n"
            "Énoncé : Trois vulnérabilités critiques restent ouvertes.\nRisque : Exploitation possible.")
NC_MINOR = ("NC-2026-014 (Mineure)\nRéférence : A.5.18 - Droits d'accès\n"
            "Énoncé : La revue des accès privilégiés n'est pas réalisée.\nPreuve : Export IAM.")


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (NC_MAJOR, RECORD),
        ('- "Référence A.5.18 (Droits d\'accès) : la revue n\'est pas réalisée."', EXAMPLE),
        ('"L\'audit a permis de vérifier la conformité. L\'équipe [recommande / ne recommande pas] le maintien."', TEMPLATE),
        ("Exemples de constats fréquemment rencontrés (formulation-type) :", "introduction"),
        ("La structure reprend les 93 contrôles. La valeur ajoutée réside dans :", "lead_in"),
        ("Règles d'indépendance :\n- Un auditeur ne peut pas auditer son propre travail.\n- Toute relation est déclarée.", GUIDANCE),
        ("Une non-conformité formalise un écart avéré par rapport à une exigence.", GUIDANCE),
        ("NC-2026-099 (Mineure)\nTexte libre sans champs.", GUIDANCE),
        ('- "Citation"\n- autre élément de liste', GUIDANCE),
    ],
)
def test_blocks_are_classified_by_shape(block, expected):
    assert classify_block(block) == expected


def test_record_identifier_label_and_fields_are_parsed():
    [unit] = segment_section([NC_MAJOR])
    assert (unit.type, unit.record_id, unit.record_label) == (RECORD, "NC-2026-015", "Majeure")
    assert unit.fields == {
        "Référence": "A.8.8 - Gestion des vulnérabilités techniques",
        "Énoncé": "Trois vulnérabilités critiques restent ouvertes.",
        "Risque": "Exploitation possible.",
    }


def test_introduction_is_the_context_of_the_run_of_units_it_introduces():
    blocks = ["Une non-conformité formalise un écart.", "Exemples de non-conformités :", NC_MINOR, NC_MAJOR,
              "Une observation signale un point de vigilance."]

    units = segment_section(blocks)

    assert [(u.type, u.context) for u in units] == [
        (GUIDANCE, None),
        (RECORD, "Exemples de non-conformités"),
        (RECORD, "Exemples de non-conformités"),
        (GUIDANCE, None),
    ]


def test_run_ends_at_a_block_with_its_own_introduction():
    blocks = ["L'audit cloud requiert des compétences spécifiques :",
              "- Modèles de service IaaS, PaaS et SaaS.\n- Isolation multi-tenant.",
              "Composition typique :\n- Auditeur principal.\n- Spécialiste cloud."]
    assert [u.context for u in segment_section(blocks)] == ["L'audit cloud requiert des compétences spécifiques", None]


def test_introduction_applies_at_least_to_the_next_block():
    blocks = ["Méthodologie interne de rédaction d'un constat :",
              "Chaque constat comporte :\n1. Une référence.\n2. Une preuve.",
              "Aucun constat ne repose sur une déclaration verbale."]
    assert [u.context for u in segment_section(blocks)] == ["Méthodologie interne de rédaction d'un constat", None]


def test_multi_sentence_lead_in_is_fused_with_the_block_it_introduces():
    blocks = ["Le guide couvre 93 contrôles. Sa valeur ajoutée réside dans :", "1. Le texte explicatif.\n2. Les attributs."]
    units = segment_section(blocks)
    assert [(u.type, u.text) for u in units] == [(GUIDANCE, "\n\n".join(blocks))]


def test_introduction_closing_a_section_keeps_its_text():
    units = segment_section(["Un paragraphe.", "Exemples :"])
    assert [(u.type, u.text, u.context) for u in units] == [(GUIDANCE, "Un paragraphe.", None), (GUIDANCE, "Exemples :", None)]


def _tokens(text: str) -> list[str]:
    return [token for token in text.split() if token != ":"]


def test_real_knowledge_base_segmentation_is_lossless_and_consistent():
    for path in discover_documents(REAL_KB_DIR):
        for section in parse_document(path, REAL_KB_DIR).sections:
            units = segment_section(section.blocks)
            rebuilt, previous_context = [], None
            for unit in units:
                if unit.context and unit.context != previous_context:
                    rebuilt.append(unit.context)
                rebuilt.append(unit.text)
                previous_context = unit.context

            assert _tokens(" ".join(rebuilt)) == _tokens(" ".join(section.blocks))
            for unit in units:
                assert unit.text.strip()
                assert (unit.type == RECORD) == (unit.record_id is not None)
                if unit.type == RECORD:
                    assert len(unit.fields) >= 2
