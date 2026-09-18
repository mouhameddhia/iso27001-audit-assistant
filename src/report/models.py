"""Format-agnostic report representation.

One `Report` is built once (`builder.py`, deterministic, no LLM) and rendered independently by
`docx_renderer.py` and `pdf_renderer.py`. Renderers only walk these block types; neither knows
anything about audit findings, sessions, or the cahier des charges section list -- that mapping
lives entirely in `builder.py`. This is what keeps the two renderers from duplicating section logic.
"""

from dataclasses import dataclass, field
from typing import List, Union


@dataclass(frozen=True)
class Paragraph:
    text: str
    bold: bool = False  # a sub-heading within a section (e.g. one finding's identifier line)


@dataclass(frozen=True)
class BulletList:
    items: List[str]


@dataclass(frozen=True)
class Table:
    headers: List[str]
    rows: List[List[str]]


Block = Union[Paragraph, BulletList, Table]


@dataclass
class ReportSection:
    title: str
    blocks: List[Block] = field(default_factory=list)


@dataclass
class Report:
    title: str
    subtitle: str  # e.g. client name + reference, shown under the title
    sections: List[ReportSection] = field(default_factory=list)
