import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Fixed namespace so the same chunk_id always maps to the same Qdrant point id.
POINT_ID_NAMESPACE = uuid.UUID("6f1c2a3e-6b1d-4f7e-9a51-27001aa00001")

# Unit types. Exemplars are complete, reusable examples of audit-report content.
GUIDANCE = "guidance"    # explanatory text, requirements, rules, lists of criteria
EXAMPLE = "example"      # one quoted example statement (finding, observation, improvement)
RECORD = "record"        # one identified record with labelled fields (e.g. a non-conformity sheet)
TEMPLATE = "template"    # a quoted model text with [placeholders]
EXEMPLAR_TYPES = frozenset({EXAMPLE, RECORD, TEMPLATE})


@dataclass
class Section:
    number: int
    title: str
    blocks: List[str]  # blank-line separated blocks, hard-wrapped lines already joined


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    category: str               # machine key, e.g. "iso27001"
    category_label: str         # human label, e.g. "ISO 27001"
    doc_type: str               # "iso_standard" | "internal_policy" | "other"
    standard: Optional[str]     # main standard, e.g. "ISO/IEC 27017" (also for a policy about it)
    standard_version: Optional[str]
    language: str
    source_file: str            # path relative to the knowledge-base root
    structured: bool            # False when the file had no recognised header/sections
    sections: List[Section] = field(default_factory=list)


@dataclass
class Unit:
    """A self-contained piece of a section: one example, one record, one paragraph or list."""

    type: str
    text: str
    context: Optional[str] = None       # introduction governing the unit, e.g. "Exemples de constats"
    record_id: Optional[str] = None
    record_label: Optional[str] = None  # qualifier written after the id, e.g. "Mineure"
    fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    standard: Optional[str]
    standard_version: Optional[str]
    category: str
    category_label: str
    language: str
    source_file: str
    section_number: int
    section_title: str
    section_slug: str
    chunk_index: int            # position within its section
    chunk_count: int            # number of chunks in its section
    unit_type: str
    context: Optional[str]
    heading: str                # "document › section › context", embedded with the text and used for citation
    text: str
    word_count: int
    references: List[str]       # canonical, standard-qualified references (see references.py)
    record_id: Optional[str] = None
    record_label: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(POINT_ID_NAMESPACE, self.chunk_id))

    def embedding_text(self) -> str:
        return f"{self.heading}\n\n{self.text}"

    def payload(self) -> Dict[str, Any]:
        return asdict(self)
