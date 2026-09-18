"""Login and current-user identity. User creation lives in routers/users.py (administrateur-only)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.audit_log import log
from api.auth import create_access_token, get_current_user, get_db_session, verify_password
from api.models_orm import User
from api.schemas import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db_session)) -> TokenResponse:
    user = db.query(User).filter(User.username == body.username).first()
    if user is None or not user.is_active or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    log(db, "login", user_id=user.id)
    return TokenResponse(
        access_token=create_access_token(user), username=user.username, role=user.role, full_name=user.full_name,
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
