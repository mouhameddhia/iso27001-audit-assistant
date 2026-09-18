"""API test fixtures: a real Postgres test database (a separate database on the project's own
dedicated container/port, never the production `iso27001_audit` database), reset per test for
isolation. A TestClient wired to it via the same `app.dependency_overrides` mechanism `api.main`
uses for the real database -- no mocking of the DB layer.
"""

import pytest
from fastapi.testclient import TestClient

from api.auth import get_db_session, hash_password
from api.db import Base, make_engine, make_session_factory
from api.main import create_app
from api.models_orm import Role, User
from src.config import get_settings

TEST_DATABASE_URL = get_settings().database_url.rsplit("/", 1)[0] + "/iso27001_audit_test"


def postgres_test_reachable() -> bool:
    try:
        make_engine(TEST_DATABASE_URL).connect().close()
        return True
    except Exception:
        return False


@pytest.fixture
def db_session():
    if not postgres_test_reachable():
        pytest.skip(f"Test Postgres database not reachable at {TEST_DATABASE_URL}")
    engine = make_engine(TEST_DATABASE_URL)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    SessionLocal = make_session_factory(TEST_DATABASE_URL)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def app(db_session):
    application = create_app()

    def _override():
        yield db_session

    application.dependency_overrides[get_db_session] = _override
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def make_user(db_session, username="jdupont", role: Role = Role.AUDITEUR, password="test-password-123") -> User:
    user = User(username=username, hashed_password=hash_password(password), full_name="J. Dupont", role=role.value)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def auth_headers(client, username: str, password: str = "test-password-123") -> dict:
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def headers(client, db_session):
    """A default logged-in auditeur ("jdupont"), for tests that don't care who's calling."""
    make_user(db_session, username="jdupont")
    return auth_headers(client, "jdupont")
