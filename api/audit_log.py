"""Journalisation (cahier des charges §5.4): connexions, consultations, téléchargements,
modifications. An append-only table -- nothing here ever updates or deletes a row; retention
(>= 12 months) is a database-retention-policy concern for deployment, not application code.
"""

from typing import Any, Optional

from sqlalchemy.orm import Session

from api.models_orm import AuditLogEntry


def log(
    db: Session, action: str, user_id: Optional[str] = None,
    target_type: Optional[str] = None, target_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    db.add(AuditLogEntry(
        user_id=user_id, action=action, target_type=target_type, target_id=target_id, details=details,
    ))
    db.commit()
