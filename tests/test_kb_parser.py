import re

import pytest

from src.ingestion.parser import normalize_text, parse_document, parse_text, reflow_block, slugify
from src.ingestion.pipeline import discover_documents
from tests.conftest import REAL_KB_DIR, kb_document

WRAPPED_TITLE_DOC = """================================================================================
DOCUMENT: ISO/IEC 27017:2015 - Code de bonnes pratiques pour les contrôles
de sécurité de l'information pour les services
cloud
================================================================================
ID:             doc-iso27017-full
Category:       iso27017 / ISO 27017
Language:       fr


================================================================================
  SECTION 01 : INFORMATIONS GÉNÉRALES
================================================================================

Premier paragraphe écrit sur
deux lignes.

Second paragraphe.


================================================================================
  SECTION 02 : PORTÉE
================================================================================

La norme s'applique :
  - aux fournisseurs de services cloud,
    quel que soit le modèle ;
  - aux clients de services cloud.
"""


def test_header_metadata_is_extracted_including_wrapped_title():
    doc = parse_text(WRAPPED_TITLE_DOC, "iso27017_full_KB.txt")

    assert doc.structured
    assert doc.doc_id == "doc-iso27017-full"
    assert doc.title == ("ISO/IEC 27017:2015 - Code de bonnes pratiques pour les contrôles "
                         "de sécurité de l'information pour les services cloud")
    assert (doc.category, doc.category_label, doc.doc_type) == ("iso27017", "ISO 27017", "iso_standard")
    assert (doc.standard, doc.standard_version) == ("ISO/IEC 27017", "2015")
    assert doc.language == "fr"
    assert doc.source_file == "iso27017_full_KB.txt"


def test_sections_are_split_with_numbers_titles_and_blocks():
    doc = parse_text(WRAPPED_TITLE_DOC, "x.txt")

    assert [(s.number, s.title) for s in doc.sections] == [(1, "INFORMATIONS GÉNÉRALES"), (2, "PORTÉE")]
    assert doc.sections[0].blocks == ["Premier paragraphe écrit sur deux lignes.", "Second paragraphe."]
    assert doc.sections[1].blocks == [
        "La norme s'applique :\n"
        "- aux fournisseurs de services cloud, quel que soit le modèle ;\n"
        "- aux clients de services cloud."
    ]


def test_reflow_keeps_record_fields_on_their_own_lines():
    block = ("NC-2026-014 (Mineure)\nRéférence : A.5.18 - Droits d'accès\n"
             "Énoncé : La revue périodique n'a pas été\nréalisée dans les délais.\nPreuve : Export IAM.")
    assert reflow_block(block) == (
        "NC-2026-014 (Mineure)\nRéférence : A.5.18 - Droits d'accès\n"
        "Énoncé : La revue périodique n'a pas été réalisée dans les délais.\nPreuve : Export IAM."
    )


def test_reflow_rejoins_compounds_and_ranges_cut_at_line_end_but_not_spaced_separators():
    block = ("isolation multi-\ntenants, contrôles (A.5.19-\nA.5.23), analyse (AIPD/\nDPIA),\n"
             "priorité (élevée / moyenne /\nfaible), (SoA -\nStatement of Applicability)")
    assert reflow_block(block) == (
        "isolation multi-tenants, contrôles (A.5.19-A.5.23), analyse (AIPD/DPIA), "
        "priorité (élevée / moyenne / faible), (SoA - Statement of Applicability)"
    )


def test_normalize_text_unifies_newlines_spaces_apostrophes_and_unicode():
    raw = "﻿contrôle d’accès\r\nA :\tB   \r\n\r\n\r\n\r\nfin  "
    assert normalize_text(raw) == "contrôle d'accès\nA : B\n\nfin"


@pytest.mark.parametrize(
    ("title", "category", "expected"),
    [
        ("ISO/IEC 27018:2019 - Code de bonnes pratiques", "iso27018 / ISO 27018", ("iso_standard", "ISO/IEC 27018", "2019")),
        ("Politique interne du cabinet - missions d'audit ISO/IEC 27001", "politique_interne / Politique interne",
         ("internal_policy", "ISO/IEC 27001", None)),
        ("Guide d'hygiène informatique", "guide / Guide ANSSI", ("other", None, None)),
    ],
)
def test_doc_type_and_standard_are_derived_from_title_and_category(title, category, expected):
    doc = parse_text(kb_document("doc-x", title, category, {1: ("SECTION", "Du texte.")}), "x.txt")
    assert (doc.doc_type, doc.standard, doc.standard_version) == expected


def test_file_without_template_falls_back_to_single_section(tmp_path):
    path = tmp_path / "Note ANSSI hygiène.md"
    path.write_text("# Hygiène\n\nMettre à jour les systèmes.\n\nSauvegarder régulièrement.", encoding="utf-8")

    doc = parse_document(path, tmp_path)

    assert not doc.structured
    assert doc.doc_id == "doc-note-anssi-hygiene"
    assert (doc.doc_type, doc.standard) == ("other", None)
    assert len(doc.sections) == 1
    assert doc.sections[0].blocks == ["# Hygiène", "Mettre à jour les systèmes.", "Sauvegarder régulièrement."]


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"binary")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document(path)


def test_slugify_strips_accents_and_punctuation():
    assert slugify("OPPORTUNITÉS D'AMÉLIORATION") == "opportunites_d_amelioration"


def test_every_knowledge_base_file_parses_into_clean_sections():
    docs = [parse_document(path, REAL_KB_DIR) for path in discover_documents(REAL_KB_DIR)]

    assert docs
    assert len({doc.doc_id for doc in docs}) == len(docs)
    for doc in docs:
        assert doc.title and doc.sections and all(section.blocks for section in doc.sections)
        blocks = [block for section in doc.sections for block in section.blocks]
        assert not any("=====" in block or re.match(r"SECTION \d+ :", block) for block in blocks)
        assert not any(re.search(r"\w[-/] \w", block) for block in blocks), "compound word split by a line wrap"
