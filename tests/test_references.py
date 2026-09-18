"""Reference extraction, on phrasings taken from the knowledge base and generic edge cases."""

import pytest

from src.ingestion.references import Standard, extract_references, find_standard

ISO27001, ISO27002, ISO27017, ISO27018, ISO27701 = (f"ISO/IEC {n}" for n in (27001, 27002, 27017, 27018, 27701))


@pytest.mark.parametrize(
    ("text", "document_standard", "expected"),
    [
        # explicit standard before the reference
        ("Référence : ISO/IEC 27017 CLD.6.3.1 ; ISO/IEC 27001 A.5.23", ISO27017,
         [f"{ISO27017} CLD.6.3.1", f"{ISO27001} A.5.23"]),
        ("Référence : ISO/IEC 27001 A.5.15, guidance ISO/IEC 27002 §5.15", ISO27002,
         [f"{ISO27001} A.5.15", f"{ISO27002} 5.15"]),
        ("Exigence associée : ISO/IEC 27001:2022, Annexe A.5.18 ; Clause 8.1", ISO27001,
         [f"{ISO27001} A.5.18", f"{ISO27001} clause 8.1"]),
        ("lignes directrices actualisées de 27002:2022 §8.24, en particulier", ISO27002, [f"{ISO27002} 8.24"]),
        # an annex "A.x" of another standard is not mistaken for ISO/IEC 27001 Annex A
        ("Référence 27018 §A.10.1 (Notification en cas de demande d'accès)", ISO27018, [f"{ISO27018} A.10.1"]),
        ("Référence : ISO/IEC 27701 Annexe A §7.2.8 ; RGPD Article 30", ISO27701,
         [f"{ISO27701} A.7.2.8", "RGPD art. 30"]),
        ("Référence Annexe B, 8.2.2 (Objectifs liés à la protection de la vie privée)", ISO27701,
         [f"{ISO27701} B.8.2.2"]),
        # conventions: bare Annex A is ISO/IEC 27001, CLD is ISO/IEC 27017, whatever the document
        ("l'inventaire applicatif (A.5.9 du SMSI sous-jacent)", ISO27701, [f"{ISO27001} A.5.9"]),
        ("les contrôles CLD.9.5.1/9.5.2 (isolation et durcissement des VM)", None,
         [f"{ISO27017} CLD.9.5.1", f"{ISO27017} CLD.9.5.2"]),
        # standard written after a clause or an article
        ("conformément à la Clause 10.2 du SMSI sous-jacent.", ISO27701, [f"{ISO27001} clause 10.2"]),
        ("prévue par l'Article 33 du RGPD si le sous-traitant", ISO27701, ["RGPD art. 33"]),
        ("principes du RGPD (articles 5, 6, 32)", ISO27701, ["RGPD art. 5", "RGPD art. 6", "RGPD art. 32"]),
        # document default for numbered controls and clauses
        ("Exemple d'attributs pour le contrôle 8.24 (Utilisation de la cryptographie)", ISO27002, [f"{ISO27002} 8.24"]),
        ("(ex. Clause 5.2.1 - Comprendre les besoins)", ISO27701, [f"{ISO27701} clause 5.2.1"]),
        # ranges
        ("la gouvernance contractuelle (A.5.19-A.5.23)", ISO27017, [f"{ISO27001} A.5.{n}" for n in range(19, 24)]),
        ("pour les contrôles A.5-A.8 d'ISO/IEC 27001", ISO27002, [f"{ISO27001} A.{n}" for n in range(5, 9)]),
        ("complétant les clauses 4-10 d'ISO/IEC 27001", ISO27701, [f"{ISO27001} clause {n}" for n in range(4, 11)]),
        ("exigences des clauses 4 à 10 de la norme", ISO27001, [f"{ISO27001} clause {n}" for n in range(4, 11)]),
        # not references to a standard
        ("la politique de contrôle d'accès (réf. POL-SEC-004, §4.2)", ISO27001, []),
        ("TLS 1.2 ou supérieur, sous 24 mois, selon ISO 19011 ; l'Annexe A constitue une liste", ISO27001, []),
    ],
)
def test_references_are_resolved_to_their_standard(text, document_standard, expected):
    assert extract_references(text, document_standard) == expected


def test_ambiguous_references_are_dropped_when_there_is_no_document_standard():
    assert extract_references("voir §5.3, la Clause 6.1, l'Annexe B, 8.2.2 et l'article 30") == []
    assert extract_references("revue A.5.18 et Clause 9.2 d'ISO/IEC 27001") == [f"{ISO27001} A.5.18", f"{ISO27001} clause 9.2"]


def test_references_are_unique_and_in_order_of_appearance():
    text = "A.8.8 puis A.5.18, puis encore A.8.8 et ISO/IEC 27001 A.5.18"
    assert extract_references(text) == [f"{ISO27001} A.8.8", f"{ISO27001} A.5.18"]


def test_reversed_or_oversized_ranges_keep_only_their_endpoints():
    assert extract_references("A.5.30-A.5.2") == [f"{ISO27001} A.5.30", f"{ISO27001} A.5.2"]
    assert extract_references("clauses 1-99", ISO27001) == [f"{ISO27001} clause 1", f"{ISO27001} clause 99"]


def test_find_standard_reads_the_first_standard_and_its_version():
    assert find_standard("ISO/IEC 27017:2015 - Code de bonnes pratiques") == Standard(ISO27017, "2015")
    assert find_standard("Politique interne - missions ISO 27001 et normes associées") == Standard(ISO27001, None)
    assert find_standard("ISO 19011:2018 Lignes directrices") == Standard("ISO 19011", "2018")
    assert find_standard("Guide d'hygiène informatique") is None
