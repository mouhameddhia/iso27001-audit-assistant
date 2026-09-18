import json
import uuid

import pytest

from src.ingestion.chunker import chunk_document, group_units, split_to_fit, word_count
from src.ingestion.models import EXAMPLE, EXEMPLAR_TYPES, GUIDANCE, POINT_ID_NAMESPACE, RECORD, TEMPLATE, Unit
from src.ingestion.parser import parse_document, parse_text
from src.ingestion.pipeline import discover_documents
from src.ingestion.segmenter import segment_section
from tests.conftest import REAL_KB_DIR, kb_document


def words(n: int, word: str = "mot") -> str:
    return " ".join([word] * n)


def unit(n_words: int, type_: str = GUIDANCE, context: str | None = None, word: str = "mot") -> Unit:
    return Unit(type=type_, text=words(n_words, word), context=context)


def sizes(groups):
    return [[word_count(u.text) for u in group] for group in groups]


def test_exemplars_are_never_merged_even_when_small():
    units = [unit(20, EXAMPLE), unit(20, EXAMPLE), unit(15, RECORD), unit(10, TEMPLATE), unit(10)]
    assert sizes(group_units(units, min_words=40, max_words=200)) == [[20], [20], [15], [10], [10]]


@pytest.mark.parametrize(
    ("unit_sizes", "max_words", "expected"),
    [
        ([100, 20], 200, [[100, 20]]),          # short rule stays with the text it completes
        ([20, 100], 200, [[20, 100]]),
        ([100, 60], 200, [[100], [60]]),        # two self-contained paragraphs stay apart
        ([150, 30], 160, [[150], [30]]),        # merging would exceed the maximum
        ([10, 10, 10, 100], 200, [[10, 10, 10, 100]]),
    ],
)
def test_short_guidance_is_merged_with_its_neighbour(unit_sizes, max_words, expected):
    assert sizes(group_units([unit(n) for n in unit_sizes], min_words=40, max_words=max_words)) == expected


def test_guidance_under_another_introduction_is_not_merged():
    units = [unit(20, context="Règles d'indépendance"), unit(20, context="Composition de l'équipe")]
    assert len(group_units(units, 40, 200)) == 2


def test_guidance_without_introduction_joins_the_previous_introduction():
    groups = group_units([unit(80, context="Critères de classification"), unit(20)], 40, 200)
    assert sizes(groups) == [[80, 20]]


def test_oversized_list_is_split_between_items():
    text = "Critères :\n" + "\n".join(f"- critère{i} {words(29)}" for i in range(10))

    parts = split_to_fit(text, max_words=100)

    assert all(word_count(p) <= 100 for p in parts) and len(parts) > 1
    assert all(line.startswith(("Critères", "- critère")) for p in parts for line in p.split("\n"))
    assert " ".join(parts).split() == text.split()


def test_oversized_line_is_split_between_sentences_then_words():
    sentence = "Phrase " + words(29) + "."
    parts = split_to_fit(" ".join([sentence] * 10), max_words=100)
    assert [word_count(p) for p in parts] == [90, 90, 90, 30]
    assert all(p.endswith(".") for p in parts)

    long_sentence = words(250)
    assert [word_count(p) for p in split_to_fit(long_sentence, max_words=100)] == [84, 84, 82]


def test_min_words_above_max_words_is_rejected():
    with pytest.raises(ValueError):
        group_units([unit(10)], min_words=300, max_words=200)


NC_SECTION = """Une non-conformité formalise un écart avéré.

Exemples de non-conformités :

NC-2026-015 (Majeure)
Référence : A.8.8 - Gestion des vulnérabilités techniques
Énoncé : Trois vulnérabilités critiques restent ouvertes au-delà du délai.
Risque : Exploitation possible."""


def test_chunks_carry_document_section_unit_and_reference_metadata():
    text = kb_document("doc-iso27001-2022-full", "ISO/IEC 27001:2022 - Exigences", "iso27001 / ISO 27001",
                       {5: ("NON-CONFORMITÉS", NC_SECTION)})
    doc = parse_text(text, "iso27001_2022_full_KB.txt")

    guidance, record = chunk_document(doc, min_words=40, max_words=200)

    assert (guidance.chunk_id, guidance.unit_type, guidance.context) == ("doc-iso27001-2022-full:s05:c00", GUIDANCE, None)
    assert record.chunk_id == "doc-iso27001-2022-full:s05:c01"
    assert (record.unit_type, record.record_id, record.record_label) == (RECORD, "NC-2026-015", "Majeure")
    assert record.fields["Risque"] == "Exploitation possible."
    assert record.heading == "ISO/IEC 27001:2022 › Non-conformités › Exemples de non-conformités"
    assert record.embedding_text() == f"{record.heading}\n\n{record.text}"
    assert record.references == ["ISO/IEC 27001 A.8.8"]
    assert (record.standard, record.standard_version, record.doc_type) == ("ISO/IEC 27001", "2022", "iso_standard")
    assert (record.section_number, record.section_slug, record.chunk_index, record.chunk_count) == (5, "non_conformites", 1, 2)
    payload = record.payload()
    assert json.loads(json.dumps(payload)) == payload
    assert record.point_id == str(uuid.uuid5(POINT_ID_NAMESPACE, record.chunk_id))


@pytest.mark.parametrize(("min_words", "max_words"), [(40, 200), (0, 120), (80, 300)])
def test_real_knowledge_base_chunks_are_bounded_unique_and_lossless(min_words, max_words):
    for path in discover_documents(REAL_KB_DIR):
        doc = parse_document(path, REAL_KB_DIR)
        chunks = chunk_document(doc, min_words, max_words)

        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(0 < c.word_count <= max_words for c in chunks)
        assert chunks == chunk_document(doc, min_words, max_words)  # deterministic
        for section in doc.sections:
            units = segment_section(section.blocks)
            section_chunks = [c for c in chunks if c.section_number == section.number]
            assert " ".join(c.text for c in section_chunks).split() == " ".join(u.text for u in units).split()
            exemplar_units = [u for u in units if u.type in EXEMPLAR_TYPES]
            exemplar_chunks = [c for c in section_chunks if c.unit_type in EXEMPLAR_TYPES]
            if all(word_count(u.text) <= max_words for u in exemplar_units):
                assert len(exemplar_chunks) == len(exemplar_units)
