"""Shared FastAPI dependencies used by more than one router."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session as DbSession

from api import repository
from api.models_orm import AuditSessionRecord, Role, User
from api.repository import SessionNotFoundError
from src.report.session import AuditSession


def get_record_and_session(db: DbSession, session_id: str, user: User) -> tuple[AuditSessionRecord, AuditSession]:
    """Loads a session and enforces least-privilege visibility (cahier des charges §5.3): an
    auditeur only sees their own sessions; chef d'équipe/administrateur see every session."""
    try:
        record = repository.get_record(db, session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if user.role == Role.AUDITEUR.value and record.owner_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your session")
    return record, AuditSession.model_validate(record.data)
