"""Standard-aware extraction of normative references (ISO controls and clauses, RGPD articles).

Each reference is resolved to the standard it belongs to and returned in a canonical form:

    ISO/IEC 27001 A.5.18       Annex A control (or theme: ISO/IEC 27001 A.8)
    ISO/IEC 27001 clause 9.2   management-system clause
    ISO/IEC 27002 8.24         numbered control/section of a code of practice
    ISO/IEC 27017 CLD.6.3.1    cloud control
    ISO/IEC 27701 B.8.2.2      annex control of another standard
    RGPD art. 33

Resolution, from strongest to weakest evidence:
1. a standard written right before the reference ("ISO/IEC 27002 §5.15", "27018 §A.10.1",
   "RGPD Article 30") or right after a clause/article ("clauses 4-10 d'ISO/IEC 27001",
   "Clause 10.2 du SMSI", "Article 33 du RGPD");
2. a numbering scheme owned by one standard: "CLD.x" is ISO/IEC 27017, and a bare "A.x.y" is
   ISO/IEC 27001 Annex A, the audit convention;
3. `default_standard`, the standard of the document being processed.
A reference that stays ambiguous is dropped rather than guessed, and section numbers of client
documents ("POL-SEC-004, §4.2") are ignored. Ranges are expanded ("A.5.19-A.5.23", "clauses 4 à 10").
"""

import re
from typing import Iterator, List, NamedTuple, Optional

ISO_27001 = "ISO/IEC 27001"
ISO_27017 = "ISO/IEC 27017"
GDPR = "RGPD"
_CLIENT_DOCUMENT = "client-document"
_MAX_RANGE = 30

_NUM = r"\d{1,2}(?:\.\d{1,2}){0,3}"

_STANDARD_RE = re.compile(r"\bISO(?:/IEC)?\s*(\d{4,5})(?::(\d{4}))?\b|(?<![\w.:-])(27\d{3})(?::\d{4})?\b|\b(RGPD|GDPR)\b")
_CLIENT_DOCUMENT_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z]{2,})+-\d+\b")
_QUALIFIER_GAP_RE = re.compile(r"[\s,(]{0,4}")
_FOLLOWING_STANDARD_RE = re.compile(
    r"\s*(?:d'|de l'|du |de la (?:norme )?)(?:ISO(?:/IEC)?\s*(\d{4,5})|(SMSI)|(RGPD|GDPR))"
)

_PATTERNS = {
    "annex_numbered": re.compile(rf"\bAnnexe\s+([A-F])\s*[,§]\s*§?\s*(\d{{1,2}}(?:\.\d{{1,2}}){{1,3}})"),
    "annex": re.compile(rf"(?:\bAnnexe\s+)?(§\s*)?(?<![\w.])([AB])\.({_NUM})(?:-(?:[AB]\.)?({_NUM}))?(?!\w)"),
    "cloud": re.compile(rf"\bCLD\.({_NUM})(?:/({_NUM}))?"),
    "section": re.compile(r"(?:§\s*|\b[Cc]ontrôle\s+)(\d{1,2}(?:\.\d{1,2}){1,3})"),
    "clause": re.compile(rf"\b[Cc]lauses?\s+({_NUM})(?:\s*(?:-|à)\s*(\d{{1,2}}))?"),
    "article": re.compile(r"\b[Aa]rticles?\s+(\d{1,3}(?:\s*,\s*\d{1,3})*)"),
}


class Standard(NamedTuple):
    name: str
    version: Optional[str]


def standard_name(number: str) -> str:
    return f"ISO/IEC {number}" if number.startswith(("27", "29")) else f"ISO {number}"


def find_standard(text: str) -> Optional[Standard]:
    """First ISO standard written in `text`, e.g. a document title."""
    match = re.search(r"\bISO(?:/IEC)?\s*(\d{4,5})(?::(\d{4}))?", text)
    return Standard(standard_name(match.group(1)), match.group(2)) if match else None


def _expand(first: str, last: Optional[str]) -> List[str]:
    if not last:
        return [first]
    head, _, start = first.rpartition(".")
    last_head, _, end = last.rpartition(".")
    if head == last_head and int(start) <= int(end) and int(end) - int(start) <= _MAX_RANGE:
        prefix = f"{head}." if head else ""
        return [f"{prefix}{n}" for n in range(int(start), int(end) + 1)]
    return [first, last]


class _Qualifier(NamedTuple):
    end: int
    standard: str


def _qualifiers(text: str) -> List[_Qualifier]:
    found = [
        _Qualifier(m.end(), GDPR if m.group(4) else standard_name(m.group(1) or m.group(3)))
        for m in _STANDARD_RE.finditer(text)
    ]
    found += [_Qualifier(m.end(), _CLIENT_DOCUMENT) for m in _CLIENT_DOCUMENT_RE.finditer(text)]
    return sorted(found)


def _preceding(text: str, start: int, qualifiers: List[_Qualifier]) -> Optional[str]:
    for qualifier in reversed(qualifiers):
        if qualifier.end <= start:
            gap_is_short = _QUALIFIER_GAP_RE.fullmatch(text, qualifier.end, start) is not None
            return qualifier.standard if gap_is_short else None
    return None


def _following(text: str, end: int) -> Optional[str]:
    match = _FOLLOWING_STANDARD_RE.match(text, end)
    if not match:
        return None
    if match.group(1):
        return standard_name(match.group(1))
    return ISO_27001 if match.group(2) else GDPR


def _resolve(kind: str, match: re.Match, before: Optional[str], after: Optional[str], default: Optional[str]) -> Iterator[str]:
    if before == _CLIENT_DOCUMENT:
        return
    iso_before = before if before not in (None, GDPR) else None

    if kind == "cloud":
        yield from (f"{ISO_27017} CLD.{n}" for n in filter(None, match.groups()))
    elif kind == "annex":
        has_section_sign, letter, first, last = match.groups()
        conventional = ISO_27001 if letter == "A" and not has_section_sign else default
        standard = iso_before or conventional
        if standard:
            yield from (f"{standard} {letter}.{n}" for n in _expand(first, last))
    elif kind == "annex_numbered":
        standard = iso_before or default
        if standard:
            yield f"{standard} {match.group(1)}.{match.group(2)}"
    elif kind == "section":
        standard = iso_before or default
        if standard:
            yield f"{standard} {match.group(1)}"
    elif kind == "clause":
        standard = iso_before or (after if after != GDPR else None) or default
        if standard:
            yield from (f"{standard} clause {n}" for n in _expand(*match.groups()))
    elif kind == "article" and GDPR in (before, after):
        yield from (f"{GDPR} art. {n.strip()}" for n in match.group(1).split(","))


def extract_references(text: str, default_standard: Optional[str] = None) -> List[str]:
    """Canonical references cited in `text`, in order of appearance, without duplicates."""
    qualifiers = _qualifiers(text)
    candidates = sorted(
        (m.start(), -m.end(), kind, m) for kind, pattern in _PATTERNS.items() for m in pattern.finditer(text)
    )
    references: List[str] = []
    covered_until = -1
    for start, negative_end, kind, match in candidates:
        if start < covered_until:  # inside a longer reference already read
            continue
        covered_until = -negative_end
        before = _preceding(text, start, qualifiers)
        after = _following(text, match.end())
        references.extend(_resolve(kind, match, before, after, default_standard))
    return list(dict.fromkeys(references))
