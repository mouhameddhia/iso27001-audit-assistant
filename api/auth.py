"""Local username/password + JWT authentication and role-based access control.

Cahier des charges §5.3 names SSO/Azure AD/MFA as the target; this is a development-phase stand-in
behind the same shape a real identity provider would need anyway (a signed token carrying a user
id and role, validated by a FastAPI dependency) -- swapping the token issuer later does not change
`get_current_user`'s contract or any router's `require_role(...)` checks.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from api.models_orm import Role, User
from src.config import Settings, get_settings

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user: User, settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id, "username": user.username, "role": user.role,
        "iat": now, "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Optional[Settings] = None) -> dict:
    settings = settings or get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc


def get_db_session():
    """Overridden in api/main.py (app.dependency_overrides) once the real sessionmaker exists --
    kept as a plain marker dependency here so this module has no import-time dependency on it."""
    raise NotImplementedError("get_db_session must be overridden by the app")


def get_current_user(token: str = Depends(_oauth2_scheme), db: Session = Depends(get_db_session)) -> User:
    payload = decode_access_token(token)
    user = db.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user


def require_role(*roles: Role):
    """Least-privilege dependency factory: `Depends(require_role(Role.ADMINISTRATEUR))`."""
    allowed = {r.value for r in roles}

    def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires one of: {', '.join(sorted(allowed))}")
        return user

    return _check
