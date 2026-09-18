"""Chunk assembly: semantic units -> retrieval chunks with metadata.

1. Units come from `segmenter.segment_section`; a chunk never crosses a section boundary.
2. Exemplars (one example, one record, one template) are always a chunk of their own: packing
   several examples together blends distinct topics into one vector.
3. Guidance units shorter than `min_words` are merged with the adjacent guidance unit of the same
   context, up to `max_words`, so a one-sentence rule stays next to the criteria it completes.
4. A unit longer than `max_words` is split at line boundaries (list items, record fields), then
   between sentences, and only as a last resort by words.
5. No overlap. Each chunk carries a heading "document › section › introduction", embedded with the
   text, and the standard-qualified references cited in its text and introduction.
"""

import math
from typing import List

from src.ingestion.models import GUIDANCE, KnowledgeChunk, ParsedDocument, Section, Unit
from src.ingestion.parser import slugify
from src.ingestion.references import extract_references
from src.ingestion.segmenter import segment_section, split_sentences


def word_count(text: str) -> int:
    return len(text.split())


def _pack(pieces: List[str], separator: str, max_words: int) -> List[str]:
    packed: List[str] = []
    current: List[str] = []
    for piece in pieces:
        if current and word_count(separator.join(current + [piece])) > max_words:
            packed.append(separator.join(current))
            current = []
        current.append(piece)
    if current:
        packed.append(separator.join(current))
    return packed


def _split_words_evenly(text: str, max_words: int) -> List[str]:
    words = text.split()
    size = math.ceil(len(words) / math.ceil(len(words) / max_words))
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def split_to_fit(text: str, max_words: int) -> List[str]:
    if word_count(text) <= max_words:
        return [text]
    parts: List[str] = []
    for by_lines in _pack(text.split("\n"), "\n", max_words):
        if word_count(by_lines) <= max_words:
            parts.append(by_lines)
            continue
        for by_sentences in _pack(split_sentences(by_lines), " ", max_words):
            fits = word_count(by_sentences) <= max_words
            parts.extend([by_sentences] if fits else _split_words_evenly(by_sentences, max_words))
    return [part for part in parts if part.strip()]


def _can_merge(group: List[Unit], unit: Unit, min_words: int, max_words: int) -> bool:
    if unit.type != GUIDANCE or group[-1].type != GUIDANCE:
        return False
    if unit.context not in (None, group[0].context):
        return False
    group_words = sum(word_count(u.text) for u in group)
    unit_words = word_count(unit.text)
    return min(group_words, unit_words) < min_words and group_words + unit_words <= max_words


def group_units(units: List[Unit], min_words: int, max_words: int) -> List[List[Unit]]:
    if min_words > max_words:
        raise ValueError("min_words must be <= max_words")
    groups: List[List[Unit]] = []
    for unit in units:
        for text in split_to_fit(unit.text, max_words):
            part = Unit(unit.type, text, unit.context, unit.record_id, unit.record_label, unit.fields)
            if groups and _can_merge(groups[-1], part, min_words, max_words):
                groups[-1].append(part)
            else:
                groups.append([part])
    return groups


def _short_title(doc: ParsedDocument) -> str:
    return doc.title.split(" - ")[0].strip() or doc.doc_id


def _display_title(title: str) -> str:
    return title[:1] + title[1:].lower() if title.isupper() else title


def chunk_section(doc: ParsedDocument, section: Section, min_words: int, max_words: int) -> List[KnowledgeChunk]:
    groups = group_units(segment_section(section.blocks), min_words, max_words)
    chunks = []
    for index, group in enumerate(groups):
        first = group[0]
        text = "\n\n".join(unit.text for unit in group)
        heading = " › ".join(filter(None, [_short_title(doc), _display_title(section.title), first.context]))
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{doc.doc_id}:s{section.number:02d}:c{index:02d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                standard=doc.standard,
                standard_version=doc.standard_version,
                category=doc.category,
                category_label=doc.category_label,
                language=doc.language,
                source_file=doc.source_file,
                section_number=section.number,
                section_title=section.title,
                section_slug=slugify(section.title),
                chunk_index=index,
                chunk_count=len(groups),
                unit_type=first.type,
                context=first.context,
                heading=heading,
                text=text,
                word_count=word_count(text),
                references=extract_references(f"{first.context or ''}\n{text}", doc.standard),
                record_id=first.record_id,
                record_label=first.record_label,
                fields=dict(first.fields),
            )
        )
    return chunks


def chunk_document(doc: ParsedDocument, min_words: int, max_words: int) -> List[KnowledgeChunk]:
    return [chunk for section in doc.sections for chunk in chunk_section(doc, section, min_words, max_words)]
