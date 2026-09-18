"""FastAPI application: wires the routers, the database session dependency, CORS for the React
dev server, and a one-time bootstrap of the first administrateur account (cahier des charges
§5.3's roles need at least one account able to create the rest).

    uvicorn api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import get_db_session, hash_password
from api.db import Base, make_session_factory
from api.models_orm import Role, User
from api.routers import auth, documents, sessions, users
from src.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
SessionLocal = make_session_factory(settings.database_url)


DEV_SEED_USERS = [
    ("admin", "admin123", "Administrateur", Role.ADMINISTRATEUR.value),
    ("auditor", "auditor123", "Auditeur", Role.AUDITEUR.value),
]


def _bootstrap_admin() -> None:
    """Creates the first accounts if the users table is empty. Development-phase convenience with
    fixed, easy-to-remember dev credentials (matches this repo's existing dev_local_only_change_me
    pattern in .env) -- a real deployment behind SSO (cahier des charges §5.3) would not need this,
    and these fixed passwords must never be used outside a local dev machine."""
    db = SessionLocal()
    try:
        if db.query(User).first() is not None:
            return
        for username, password, full_name, role in DEV_SEED_USERS:
            db.add(User(
                username=username, hashed_password=hash_password(password),
                full_name=full_name, role=role,
            ))
        db.commit()
        logger.warning(
            "Bootstrapped dev accounts with fixed passwords (local dev only, never use in "
            "production): %s",
            ", ".join(f"{u}/{p}" for u, p, _, _ in DEV_SEED_USERS),
        )
    finally:
        db.close()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    Base.metadata.create_all(bind=SessionLocal().get_bind())
    _bootstrap_admin()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ISO/IEC 27001 Audit Assistant API", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # React (Vite) dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_db_session_impl():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db_session] = get_db_session_impl

    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(sessions.router)
    app.include_router(documents.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
