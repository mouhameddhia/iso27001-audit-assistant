"""ORM tables. Three, deliberately: users, audit sessions (one JSONB blob each, see db.py), and an
append-only audit log for the cahier des charges' journalisation requirement (§5.4: connexions,
consultations, téléchargements, modifications, conserved >= 12 months -- nothing here ever deletes
a log row)."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from sqlalchemy import ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


class Role(str, Enum):
    """Cahier des charges §5.3: three roles, least-privilege. Kept as a plain string column (not a
    DB-level enum type) so adding a role later is a one-line change, not a migration."""

    AUDITEUR = "auditeur"
    CHEF_EQUIPE = "chef_equipe"
    ADMINISTRATEUR = "administrateur"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(30), default=Role.AUDITEUR.value)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AuditSessionRecord(Base):
    __tablename__ = "audit_sessions"

    # == AuditSession.metadata.session_id, so the API's URL path and the domain object agree.
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    # Denormalised out of `data` purely for fast listing/filtering without deserialising every row;
    # `data` (below) is the single source of truth and is what gets loaded into an AuditSession.
    title: Mapped[str] = mapped_column(String(300))
    client_name: Mapped[str] = mapped_column(String(300))
    reference: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="open")  # open | closed
    data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    target_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    target_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
