"""Audit session: collects AuditFindings from one engagement, with a human review gate.

One session = one JSON file (`save_session`/`load_session`). `AuditSession.add_finding()` calls
`AuditAssistant.analyze()` the same way `assistant_cli.py`'s `analyze` command does -- nothing
here re-implements retrieval or generation. Findings start `pending`; only `approved` findings
reach a report (see builder.py). Severity (majeure/mineure) is a reviewer decision, not something
`AuditFinding` itself carries: the firm's internal policy classifies it by judgment (systemic
failure, certification impact) that the retrieved evidence alone doesn't establish.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.anonymization.anonymizer import Anonymizer
from src.audit_assistant import AuditAssistant
from src.generation.models import AuditFinding, FindingType


class ReviewDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Severity(str, Enum):
    MAJEURE = "majeure"
    MINEURE = "mineure"


class CertificationDecision(str, Enum):
    RECOMMENDS = "recommande"
    DOES_NOT_RECOMMEND = "ne_recommande_pas"


class SessionError(RuntimeError):
    pass


class ImportStatus(str, Enum):
    PENDING_REVIEW = "pending_review"  # sidecar written, not yet embedded/stored
    CONFIRMED = "confirmed"


class TeamMember(BaseModel):
    name: str
    role: str


class SessionMetadata(BaseModel):
    # Real identifier, assigned once at creation -- distinct from `reference` (free text,
    # interpolated into output filenames), used to derive a safe, collision-free Qdrant
    # collection name for this session's imported client documents.
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    client_name: str
    # Other names the client is known by (subsidiary, trading name, informal short form) -- seeds
    # the anonymizer's `known_entities` alongside `client_name`, so a bare repeated proper noun
    # with no nearby trigger keyword is still caught (see src/anonymization/anonymizer.py).
    client_aliases: List[str] = Field(default_factory=list)
    scope: str
    standards: List[str] = Field(default_factory=list)
    audit_team: List[TeamMember] = Field(default_factory=list)
    reference: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ReviewedFinding(BaseModel):
    finding: AuditFinding
    decision: ReviewDecision = ReviewDecision.PENDING
    severity: Optional[Severity] = None
    edited_text: Optional[str] = None
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None

    @property
    def text(self) -> str:
        """The finding text a report should use: the reviewer's edit if any, else the original."""
        return self.edited_text or self.finding.finding


class ImportedDocument(BaseModel):
    """Tracks one client document through the two-step import gate (see documents/ingestion.py):
    `propose_import` writes an anonymized sidecar and sets `PENDING_REVIEW`; `confirm_import`
    re-reads that exact sidecar, embeds it, and sets `CONFIRMED`. Re-importing a file of the same
    name replaces this record (same `doc_id`) instead of adding a duplicate."""

    doc_id: str
    filename: str
    doc_type: str
    sidecar_path: str
    status: ImportStatus = ImportStatus.PENDING_REVIEW
    chunk_count: int = 0
    imported_at: Optional[str] = None
    confirmed_at: Optional[str] = None


class AuditSession(BaseModel):
    metadata: SessionMetadata
    findings: List[ReviewedFinding] = Field(default_factory=list)
    imported_documents: List[ImportedDocument] = Field(default_factory=list)
    # Cumulative placeholder -> original mapping for this session's whole audit context (auditor
    # observations *and* any imported client documents), persisted across process boundaries so a
    # fresh CLI invocation resumes it instead of starting a new, disconnected set of placeholders.
    anonymization_mapping: Dict[str, str] = Field(default_factory=dict)
    # The Auditeur Principal's own certification recommendation -- never inferred, drafted or
    # defaulted by the AI (cahier des charges: "validation humaine obligatoire"; the internal
    # policy's cross-signature rule). Unset until explicitly recorded; the same "a human decision,
    # not an LLM field" pattern already used for `severity` on a non-conformité.
    certification_decision: Optional[CertificationDecision] = None
    certification_decided_by: Optional[str] = None
    certification_decided_at: Optional[str] = None

    @property
    def confirmed_documents(self) -> List[ImportedDocument]:
        return [d for d in self.imported_documents if d.status == ImportStatus.CONFIRMED]

    @property
    def qdrant_collection_suffix(self) -> str:
        """Suffix appended to `settings.qdrant_collection` for this session's own, isolated
        collection of imported-document chunks -- never the shared knowledge-base collection."""
        return f"_client_{self.metadata.session_id}"

    @property
    def anonymizer(self) -> Anonymizer:
        """A fresh `Anonymizer` resuming this session's cumulative mapping, seeded with the
        client's known name(s) so a bare repeated proper noun is anonymized even without a nearby
        trigger keyword. Call `sync_anonymizer()` after using it so its new placeholders (if any)
        are kept, since this property builds a new instance each time it is read."""
        known_entities = {"CLIENT": [self.metadata.client_name, *self.metadata.client_aliases]}
        return Anonymizer.restore(self.anonymization_mapping, known_entities=known_entities)

    def sync_anonymizer(self, anonymizer: Anonymizer) -> None:
        """Persists an anonymizer's cumulative mapping back onto the session, so `save_session`
        keeps whatever new placeholders it assigned. Call after any operation that anonymizes
        text through `self.anonymizer` (finding generation, document import)."""
        self.anonymization_mapping = dict(anonymizer.mapping)

    def add_finding(
        self, assistant: AuditAssistant, observation: str, language: str = "fr", document_retriever=None,
    ) -> ReviewedFinding:
        """`document_retriever`: this session's own imported-document retriever (see
        `src.documents.retrieval.build_document_retriever`), or None -- not imported here to keep
        this module independent of the document-import feature, matching how `assistant` itself
        is built by the caller rather than by this method."""
        reviewed = ReviewedFinding(finding=assistant.analyze(observation, language=language, document_retriever=document_retriever))
        self.findings.append(reviewed)
        return reviewed

    def review(
        self, index: int, decision: ReviewDecision, reviewer: Optional[str] = None,
        severity: Optional[Severity] = None, edited_text: Optional[str] = None,
    ) -> ReviewedFinding:
        """Sets the decision; other fields are only updated when explicitly given, so re-reviewing
        (e.g. to fix a typo) never silently clears a previously set severity or edit."""
        if not 0 <= index < len(self.findings):
            raise SessionError(f"No finding at index {index} (session has {len(self.findings)})")
        reviewed = self.findings[index]
        if severity is not None and reviewed.finding.finding_type != FindingType.NON_CONFORMITE:
            raise SessionError(
                f"Severity only applies to non-conformité findings, not {reviewed.finding.finding_type.value}"
            )
        reviewed.decision = decision
        if severity is not None:
            reviewed.severity = severity
        if edited_text is not None:
            reviewed.edited_text = edited_text
        if reviewer is not None:
            reviewed.reviewer = reviewer
        reviewed.reviewed_at = datetime.now(timezone.utc).isoformat()
        return reviewed

    def set_certification_decision(self, decision: CertificationDecision, decided_by: str) -> None:
        self.certification_decision = decision
        self.certification_decided_by = decided_by
        self.certification_decided_at = datetime.now(timezone.utc).isoformat()

    @property
    def approved(self) -> List[ReviewedFinding]:
        return [f for f in self.findings if f.decision == ReviewDecision.APPROVED]

    @property
    def pending_count(self) -> int:
        return sum(f.decision == ReviewDecision.PENDING for f in self.findings)

    @property
    def rejected_count(self) -> int:
        return sum(f.decision == ReviewDecision.REJECTED for f in self.findings)


def save_session(session: AuditSession, path: Path) -> None:
    Path(path).write_text(session.model_dump_json(indent=2), encoding="utf-8")


def load_session(path: Path) -> AuditSession:
    return AuditSession.model_validate_json(Path(path).read_text(encoding="utf-8"))
