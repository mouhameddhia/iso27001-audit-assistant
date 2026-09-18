"""Session management: create/list/get, add/review findings, generate reports.

Every operation calls straight into the existing, already-validated `AuditSession`/
`AuditAssistant`/`build_report` logic (src/) -- this module only adds persistence (via
api/repository.py), authorization and HTTP translation on top. No retrieval, generation,
anonymization or report-building logic is duplicated here.
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session as DbSession

from api import repository
from api.audit_log import log
from api.auth import get_current_user, get_db_session, require_role
from api.dependencies import get_record_and_session
from api.models_orm import AuditSessionRecord, Role, User
from api.schemas import (
    AddFindingRequest, CreateSessionRequest, ReviewFindingRequest, SessionSummary, SetCertificationDecisionRequest,
)
from src.audit_assistant import AuditAssistant
from src.config import get_settings
from src.documents.retrieval import build_document_retriever
from src.generation.models import AuditFinding
from src.report.builder import ReportError, build_report
from src.report.docx_renderer import render_docx
from src.report.pdf_renderer import render_pdf
from src.report.session import (
    AuditSession, CertificationDecision, ReviewDecision, ReviewedFinding, SessionError, SessionMetadata, Severity,
    TeamMember,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _summarize(record: AuditSessionRecord, session: AuditSession) -> SessionSummary:
    return SessionSummary(
        id=record.id, title=session.metadata.title, client_name=session.metadata.client_name,
        reference=session.metadata.reference, status=record.status, owner_id=record.owner_id,
        pending_count=session.pending_count, approved_count=len(session.approved),
        rejected_count=session.rejected_count, imported_documents=len(session.confirmed_documents),
    )


@router.post("", response_model=SessionSummary, status_code=status.HTTP_201_CREATED)
def create_session(
    body: CreateSessionRequest, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> SessionSummary:
    session = AuditSession(metadata=SessionMetadata(
        title=body.title, client_name=body.client_name, client_aliases=body.client_aliases,
        scope=body.scope, standards=body.standards,
        audit_team=[TeamMember(name=m.name, role=m.role) for m in body.audit_team],
        reference=body.reference, start_date=body.start_date, end_date=body.end_date,
    ))
    record = repository.create_session_record(db, user.id, session)
    log(db, "session_created", user_id=user.id, target_type="session", target_id=record.id)
    return _summarize(record, session)


@router.get("", response_model=List[SessionSummary])
def list_sessions(db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user)) -> List[SessionSummary]:
    owner_id = user.id if user.role == Role.AUDITEUR.value else None
    records = repository.list_sessions(db, owner_id=owner_id)
    return [_summarize(r, AuditSession.model_validate(r.data)) for r in records]


@router.get("/{session_id}", response_model=AuditSession, response_model_exclude={"anonymization_mapping"})
def get_session(session_id: str, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user)) -> AuditSession:
    _, session = get_record_and_session(db, session_id, user)
    log(db, "session_viewed", user_id=user.id, target_type="session", target_id=session_id)
    return session


@router.post("/{session_id}/findings", response_model=AuditFinding, status_code=status.HTTP_201_CREATED)
def add_finding(
    session_id: str, body: AddFindingRequest,
    db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> AuditFinding:
    _, session = get_record_and_session(db, session_id, user)
    settings = get_settings()
    # The session's own persistent anonymizer, resumed from its saved mapping -- not a fresh one,
    # so pseudonym numbering stays consistent across every call for this session (see the
    # Document Import stage's P0-2 fix, which this API layer must not regress).
    anonymizer = session.anonymizer
    assistant = AuditAssistant.from_settings(settings, anonymizer=anonymizer)
    document_retriever = build_document_retriever(session, settings)  # None if no confirmed documents
    reviewed = session.add_finding(
        assistant, body.observation, language=body.language, document_retriever=document_retriever,
    )
    session.sync_anonymizer(anonymizer)
    repository.save_session(db, session)
    log(db, "finding_added", user_id=user.id, target_type="session", target_id=session_id,
        details={"finding_index": len(session.findings) - 1})
    return reviewed.finding


@router.get("/{session_id}/findings", response_model=List[ReviewedFinding])
def list_findings(session_id: str, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user)) -> List[ReviewedFinding]:
    _, session = get_record_and_session(db, session_id, user)
    return session.findings


@router.post("/{session_id}/findings/{index}/review", response_model=ReviewedFinding)
def review_finding(
    session_id: str, index: int, body: ReviewFindingRequest,
    db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> ReviewedFinding:
    _, session = get_record_and_session(db, session_id, user)
    decision = ReviewDecision.APPROVED if body.approve else ReviewDecision.REJECTED
    severity = Severity(body.severity) if body.severity else None
    try:
        reviewed = session.review(
            index, decision, reviewer=body.reviewer or user.full_name,
            severity=severity, edited_text=body.edited_text,
        )
    except SessionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    repository.save_session(db, session)
    log(db, "finding_reviewed", user_id=user.id, target_type="session", target_id=session_id,
        details={"finding_index": index, "decision": decision.value})
    return reviewed


@router.post("/{session_id}/certification-decision", response_model=AuditSession, response_model_exclude={"anonymization_mapping"})
def set_certification_decision(
    session_id: str, body: SetCertificationDecisionRequest, db: DbSession = Depends(get_db_session),
    user: User = Depends(require_role(Role.CHEF_EQUIPE, Role.ADMINISTRATEUR)),
) -> AuditSession:
    """The Auditeur Principal's own certification recommendation -- never inferred by the AI
    (cahier des charges: validation humaine obligatoire). Restricted to chef d'équipe/administrateur,
    the closest match in this role model to "Auditeur Principal"; a plain auditeur can draft and
    review findings but does not sign off on the certification call."""
    _, session = get_record_and_session(db, session_id, user)
    decision = CertificationDecision.RECOMMENDS if body.recommends else CertificationDecision.DOES_NOT_RECOMMEND
    session.set_certification_decision(decision, decided_by=body.decided_by or user.full_name)
    repository.save_session(db, session)
    log(db, "certification_decision_set", user_id=user.id, target_type="session", target_id=session_id,
        details={"decision": decision.value})
    return session


@router.get("/{session_id}/report")
def get_report(
    session_id: str, format: str = "docx",
    db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> Response:
    if format not in ("docx", "pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "format must be 'docx' or 'pdf'")
    _, session = get_record_and_session(db, session_id, user)
    try:
        report = build_report(session)
    except ReportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"rapport.{format}"
        (render_docx if format == "docx" else render_pdf)(report, out_path)
        content = out_path.read_bytes()

    log(db, "report_downloaded", user_id=user.id, target_type="session", target_id=session_id, details={"format": format})
    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if format == "docx" else "application/pdf"
    )
    filename = f"rapport_{session.metadata.reference or session_id}.{format}"
    return Response(content=content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/{session_id}/close", response_model=SessionSummary)
def close_session(session_id: str, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user)) -> SessionSummary:
    record, session = get_record_and_session(db, session_id, user)
    from src.documents.ingestion import close_session_collection
    if session.confirmed_documents:
        close_session_collection(session, settings=get_settings())
    repository.close_session_record(db, session_id)
    db.refresh(record)
    log(db, "session_closed", user_id=user.id, target_type="session", target_id=session_id)
    return _summarize(record, session)
