"""SQLAlchemy engine/session -- the only module that talks to Postgres directly.

Persists what `src/report/session.py`'s `save_session`/`load_session` used to write as a JSON
file: `AuditSessionRecord.data` holds the exact same `AuditSession.model_dump()` output, in a
JSONB column instead of a file. That business logic (`AuditSession.add_finding`, `.review`,
`.approved`, ...) is not touched -- only where it is read from and written to changes.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=make_engine(database_url), autoflush=False, autocommit=False, future=True)
