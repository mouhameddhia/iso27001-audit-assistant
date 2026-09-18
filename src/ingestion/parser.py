"""Parsing and cleaning of knowledge-base source files.

The KB files share a template:

    ==========
    DOCUMENT: <title, may wrap over several lines>
    ==========
    ID:        doc-...
    Category:  <key> / <label>
    Language:  fr
    ==========
      SECTION 01 : <TITLE>
    ==========
    <body: blocks separated by blank lines>

Files that do not follow the template are still ingested as a single
unstructured section, so new sources never require code changes.
"""

import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List

from src.ingestion.models import ParsedDocument, Section
from src.ingestion.references import find_standard


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


# Add a reader here to support a new file format.
LOADERS: Dict[str, Callable[[Path], str]] = {
    ".txt": _read_text,
    ".md": _read_text,
}
SUPPORTED_EXTENSIONS = frozenset(LOADERS)

_TITLE_RE = re.compile(r"^DOCUMENT:[ ]*(.+?)\n=+[ ]*$", re.M | re.S)
_SECTION_RE = re.compile(r"^=+\n[ ]*SECTION[ ]+(\d+)[ ]*:[ ]*(.+?)[ ]*\n=+[ ]*$", re.M)
_BLOCK_SPLIT_RE = re.compile(r"\n[ ]*\n")
# A line starting a list item or a "Label : value" record keeps its own line when reflowing.
_LINE_START_RE = re.compile(r"^(?:[-•*][ ]|\d+[.)][ ]|[A-ZÀ-Ý][\w'’ -]{0,40}[ ]:[ ])")
_WRAPPED_COMPOUND_RE = re.compile(r"\w[-/]$")


def _field_re(name: str) -> re.Pattern:
    return re.compile(rf"^{name}:[ ]*(.+?)[ ]*$", re.M)


def normalize_text(text: str) -> str:
    """Unicode NFC, unified newlines and spaces, no trailing whitespace, no runs of blank lines."""
    text = unicodedata.normalize("NFC", text).replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("’", "'")
    text = re.sub(r"[\t   ]", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Case-, apostrophe- and whitespace-insensitive form used to compare phrases."""
    return " ".join(text.replace("’", "'").lower().split())


def reflow_block(block: str) -> str:
    """Join hard-wrapped lines while keeping list items and record fields on separate lines.

    A line ending with a hyphen or slash glued to a word ("multi-", "A.5.19-", "AIPD/") was cut
    inside a compound, a range or a pair, so the next line is joined without a space. A spaced
    separator ("SoA -", "élevée / moyenne /") is joined with a space.
    """
    lines: List[str] = []
    for raw in block.split("\n"):
        line = re.sub(r" {2,}", " ", raw.strip())
        if not line or set(line) == {"="}:
            continue
        if not lines or _LINE_START_RE.match(line):
            lines.append(line)
        elif _WRAPPED_COMPOUND_RE.search(lines[-1]):
            lines[-1] = f"{lines[-1]}{line}"
        else:
            lines[-1] = f"{lines[-1]} {line}"
    return "\n".join(lines)


def split_blocks(body: str) -> List[str]:
    blocks = (reflow_block(b) for b in _BLOCK_SPLIT_RE.split(body))
    return [b for b in blocks if b]


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")


def _split_category(raw: str) -> tuple[str, str]:
    if "/" in raw:
        key, label = (part.strip() for part in raw.split("/", 1))
        return slugify(key), label
    return slugify(raw), raw.strip()


def _doc_type(title: str, category: str) -> str:
    if re.match(r"ISO(?:/IEC)?\s*\d", title):
        return "iso_standard"
    if re.search(r"politique|policy", f"{category} {title}", re.I):
        return "internal_policy"
    return "other"


def fallback_doc_id(source_file: str) -> str:
    """The doc_id an unstructured file (no recognised `DOCUMENT:`/`ID:` header) is given. Exposed
    so code that needs a document's id *before* parsing it (e.g. to name a file derived from it)
    stays consistent with what `parse_text` itself will compute."""
    return f"doc-{slugify(Path(source_file).stem).replace('_', '-')}"


def parse_text(text: str, source_file: str) -> ParsedDocument:
    text = normalize_text(text)
    title_match = _TITLE_RE.search(text)
    id_match = _field_re("ID").search(text)

    if not (title_match and id_match):
        stem = Path(source_file).stem
        standard = find_standard(stem)
        return ParsedDocument(
            doc_id=fallback_doc_id(source_file),
            title=stem,
            category="uncategorized",
            category_label="Uncategorized",
            doc_type=_doc_type(stem, ""),
            standard=standard.name if standard else None,
            standard_version=standard.version if standard else None,
            language="und",
            source_file=source_file,
            structured=False,
            sections=[Section(number=0, title="Document", blocks=split_blocks(text))],
        )

    category_match = _field_re("Category").search(text)
    language_match = _field_re("Language").search(text)
    category, category_label = _split_category(category_match.group(1) if category_match else "")

    section_matches = list(_SECTION_RE.finditer(text))
    sections = []
    for i, match in enumerate(section_matches):
        end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
        sections.append(
            Section(
                number=int(match.group(1)),
                title=match.group(2).strip(),
                blocks=split_blocks(text[match.end():end]),
            )
        )
    if not sections:
        header_end = max(m.end() for m in (id_match, category_match, language_match) if m)
        sections = [Section(number=0, title="Document", blocks=split_blocks(text[header_end:]))]

    title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    standard = find_standard(title)
    return ParsedDocument(
        doc_id=id_match.group(1),
        title=title,
        category=category or "uncategorized",
        category_label=category_label or "Uncategorized",
        doc_type=_doc_type(title, category),
        standard=standard.name if standard else None,
        standard_version=standard.version if standard else None,
        language=language_match.group(1) if language_match else "und",
        source_file=source_file,
        structured=True,
        sections=[s for s in sections if s.blocks],
    )


def parse_document(path: Path, root: Path | None = None) -> ParsedDocument:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.name}")
    source_file = path.relative_to(root).as_posix() if root else path.name
    return parse_text(loader(path), source_file)
