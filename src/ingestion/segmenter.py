"""Segmentation of a section into semantic units, from the shape of its blocks.

Block shapes (no document-specific rules):
- record:        an identifier line ("NC-2026-014 (Mineure)") followed by >= 2 "Label : value" lines
- example:       a single list item whose content is a quotation (- "Référence A.8.8 ...")
- template:      a quotation containing [placeholders]
- introduction:  a single sentence ending with ":" ("Exemples de constats :"); it is not a unit,
                 it becomes the `context` of the units it introduces
- lead-in:       a multi-sentence paragraph ending with ":"; fused with the block it introduces
- guidance:      any other paragraph or list

An introduction applies to the run of blocks that follows it. The run ends at the next plain
paragraph or at a block that has its own introduction line (a list starting with "... :").
An introduction always applies to at least the block right after it.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.ingestion.models import EXAMPLE, GUIDANCE, RECORD, TEMPLATE, Unit

_ITEM_RE = re.compile(r"^(?:[-•*]|\d+[.)])\s+")
_FIELD_RE = re.compile(r"^([A-ZÀ-Ý][\w' -]{0,40}?)\s+:\s+(.+)$")
_RECORD_ID_RE = re.compile(r"^([A-Z]{2,}(?:-[A-Z0-9]+)+)\s*(?:\(([^)]+)\))?$")
_PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+(?=[\"«(A-ZÀ-Ý0-9-])")
_QUOTES = ('"', "«")

_INTRODUCTION = "introduction"
_LEAD_IN = "lead_in"


def _is_list(lines: List[str]) -> bool:
    return any(_ITEM_RE.match(line) for line in lines[1:]) or bool(_ITEM_RE.match(lines[0]))


def _parse_record(lines: List[str]) -> Optional[Tuple[str, Optional[str], Dict[str, str]]]:
    header = _RECORD_ID_RE.match(lines[0])
    if not header:
        return None
    fields = {m.group(1): m.group(2) for m in map(_FIELD_RE.match, lines[1:]) if m}
    return (header.group(1), header.group(2), fields) if len(fields) >= 2 else None


def classify_block(block: str) -> str:
    lines = block.split("\n")
    if _parse_record(lines):
        return RECORD
    if len(lines) == 1 and _ITEM_RE.match(block) and _ITEM_RE.sub("", block, count=1).startswith(_QUOTES):
        return EXAMPLE
    if block.startswith(_QUOTES) and _PLACEHOLDER_RE.search(block):
        return TEMPLATE
    if len(lines) == 1 and block.endswith(":"):
        return _LEAD_IN if _SENTENCE_BOUNDARY_RE.search(block) else _INTRODUCTION
    return GUIDANCE


def _ends_run(block_type: str, lines: List[str]) -> bool:
    has_own_introduction = len(lines) > 1 and lines[0].endswith(":")
    plain_paragraph = block_type == GUIDANCE and not _is_list(lines)
    return has_own_introduction or plain_paragraph


def introduction_text(block: str) -> str:
    return block.rstrip(": ").strip()


def segment_section(blocks: List[str]) -> List[Unit]:
    units: List[Unit] = []
    context: Optional[str] = None
    context_applied = False
    lead_in: Optional[str] = None

    for block in blocks:
        block_type = classify_block(block)
        if block_type == _INTRODUCTION:
            context, context_applied = introduction_text(block), False
            continue
        if block_type == _LEAD_IN:
            lead_in = f"{lead_in}\n\n{block}" if lead_in else block
            continue

        lines = block.split("\n")
        if context and context_applied and _ends_run(block_type, lines):
            context = None

        record = _parse_record(lines) if block_type == RECORD else None
        units.append(
            Unit(
                type=block_type,
                text=f"{lead_in}\n\n{block}" if lead_in else block,
                context=context,
                record_id=record[0] if record else None,
                record_label=record[1] if record else None,
                fields=record[2] if record else {},
            )
        )
        lead_in = None
        context_applied = context is not None

    if lead_in:
        units.append(Unit(type=GUIDANCE, text=lead_in, context=context))
    elif context and not context_applied:  # introduction closing the section: keep its text
        units.append(Unit(type=GUIDANCE, text=f"{context} :"))
    return units


def split_sentences(text: str) -> List[str]:
    """Sentences of `text` with their original separators removed; words are preserved."""
    return [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s]
