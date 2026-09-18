"""Login, current-user identity, and user management (administrateur-only, cahier des charges
§5.3 least privilege). No LLM/Qdrant call anywhere in this file -- pure DB + auth.
"""

import pytest

from api.models_orm import Role
from tests.test_api.conftest import auth_headers, make_user

pytestmark = pytest.mark.integration


def test_login_succeeds_with_correct_credentials(client, db_session):
    make_user(db_session, username="jdupont")
    r = client.post("/auth/login", json={"username": "jdupont", "password": "test-password-123"})
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "jdupont" and body["role"] == "auditeur"
    assert body["access_token"]


def test_login_fails_with_wrong_password(client, db_session):
    make_user(db_session, username="jdupont")
    r = client.post("/auth/login", json={"username": "jdupont", "password": "wrong"})
    assert r.status_code == 401


def test_login_fails_for_unknown_user(client, db_session):
    r = client.post("/auth/login", json={"username": "nobody", "password": "x"})
    assert r.status_code == 401


def test_login_fails_for_inactive_user(client, db_session):
    user = make_user(db_session, username="jdupont")
    user.is_active = False
    db_session.commit()
    r = client.post("/auth/login", json={"username": "jdupont", "password": "test-password-123"})
    assert r.status_code == 401


def test_me_requires_a_valid_token(client, db_session):
    make_user(db_session, username="jdupont")
    headers = auth_headers(client, "jdupont")
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == "jdupont"


def test_me_rejects_a_missing_token(client):
    assert client.get("/auth/me").status_code == 401


def test_me_rejects_a_garbage_token(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


class TestUserManagement:
    def test_administrateur_can_create_a_user(self, client, db_session):
        make_user(db_session, username="admin", role=Role.ADMINISTRATEUR)
        headers = auth_headers(client, "admin")
        r = client.post("/users", headers=headers, json={
            "username": "new_auditeur", "password": "another-password", "full_name": "New Auditeur", "role": "auditeur",
        })
        assert r.status_code == 201
        assert r.json()["username"] == "new_auditeur"
        # the new user can actually log in
        r = client.post("/auth/login", json={"username": "new_auditeur", "password": "another-password"})
        assert r.status_code == 200

    def test_auditeur_cannot_create_a_user(self, client, db_session):
        make_user(db_session, username="jdupont", role=Role.AUDITEUR)
        headers = auth_headers(client, "jdupont")
        r = client.post("/users", headers=headers, json={
            "username": "x", "password": "y", "full_name": "X", "role": "auditeur",
        })
        assert r.status_code == 403

    def test_duplicate_username_is_rejected(self, client, db_session):
        make_user(db_session, username="admin", role=Role.ADMINISTRATEUR)
        headers = auth_headers(client, "admin")
        client.post("/users", headers=headers, json={
            "username": "dup", "password": "p", "full_name": "D", "role": "auditeur",
        })
        r = client.post("/users", headers=headers, json={
            "username": "dup", "password": "p2", "full_name": "D2", "role": "auditeur",
        })
        assert r.status_code == 409

    def test_unknown_role_is_rejected(self, client, db_session):
        make_user(db_session, username="admin", role=Role.ADMINISTRATEUR)
        headers = auth_headers(client, "admin")
        r = client.post("/users", headers=headers, json={
            "username": "x", "password": "y", "full_name": "X", "role": "superuser",
        })
        assert r.status_code == 422

    def test_chef_equipe_can_list_users_but_not_create(self, client, db_session):
        make_user(db_session, username="chef", role=Role.CHEF_EQUIPE)
        headers = auth_headers(client, "chef")
        assert client.get("/users", headers=headers).status_code == 200
        r = client.post("/users", headers=headers, json={
            "username": "x", "password": "y", "full_name": "X", "role": "auditeur",
        })
        assert r.status_code == 403

    def test_auditeur_cannot_list_users(self, client, db_session):
        make_user(db_session, username="jdupont", role=Role.AUDITEUR)
        headers = auth_headers(client, "jdupont")
        assert client.get("/users", headers=headers).status_code == 403
