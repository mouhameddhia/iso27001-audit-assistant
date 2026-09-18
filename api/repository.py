"""Loads/saves an `AuditSession` to/from Postgres. The only place the JSON <-> ORM conversion
happens; every router works with the real `AuditSession` object and its already-validated methods
(`add_finding`, `review`, `.approved`, ...) exactly like the CLI does -- none of that logic is
duplicated or reimplemented here, only its storage backend changes from a JSON file to a JSONB
column.
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models_orm import AuditSessionRecord
from src.report.session import AuditSession


class SessionNotFoundError(RuntimeError):
    pass


def create_session_record(db: Session, owner_id: str, session: AuditSession) -> AuditSessionRecord:
    record = AuditSessionRecord(
        id=session.metadata.session_id, owner_id=owner_id, title=session.metadata.title,
        client_name=session.metadata.client_name, reference=session.metadata.reference,
        data=session.model_dump(mode="json"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_record(db: Session, session_id: str) -> AuditSessionRecord:
    record = db.get(AuditSessionRecord, session_id)
    if record is None:
        raise SessionNotFoundError(session_id)
    return record


def load_session(db: Session, session_id: str) -> AuditSession:
    return AuditSession.model_validate(get_record(db, session_id).data)


def save_session(db: Session, session: AuditSession) -> None:
    record = get_record(db, session.metadata.session_id)
    record.title = session.metadata.title
    record.client_name = session.metadata.client_name
    record.reference = session.metadata.reference
    record.data = session.model_dump(mode="json")
    db.commit()


def list_sessions(db: Session, owner_id: Optional[str] = None) -> List[AuditSessionRecord]:
    """`owner_id=None` lists every session (chef d'équipe / administrateur oversight); a real
    owner id restricts to that auditor's own sessions (least privilege, cahier des charges §5.3)."""
    stmt = select(AuditSessionRecord).order_by(AuditSessionRecord.updated_at.desc())
    if owner_id is not None:
        stmt = stmt.where(AuditSessionRecord.owner_id == owner_id)
    return list(db.scalars(stmt))


def close_session_record(db: Session, session_id: str) -> None:
    get_record(db, session_id).status = "closed"
    db.commit()
