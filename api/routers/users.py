"""User management -- administrateur only (cahier des charges §2 "Gestion des utilisateurs" /
"Gestion des droits d'accès", §5.3 least privilege)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.audit_log import log
from api.auth import get_current_user, get_db_session, hash_password, require_role
from api.models_orm import Role, User
from api.schemas import CreateUserRequest, UserOut

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest, db: Session = Depends(get_db_session),
    admin: User = Depends(require_role(Role.ADMINISTRATEUR)),
) -> User:
    if db.query(User).filter(User.username == body.username).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Username '{body.username}' already exists")
    try:
        role = Role(body.role)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown role '{body.role}'")
    user = User(
        username=body.username, hashed_password=hash_password(body.password),
        full_name=body.full_name, role=role.value,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log(db, "user_created", user_id=admin.id, target_type="user", target_id=user.id, details={"role": role.value})
    return user


@router.get("", response_model=List[UserOut])
def list_users(
    db: Session = Depends(get_db_session),
    _: User = Depends(require_role(Role.ADMINISTRATEUR, Role.CHEF_EQUIPE)),
) -> List[User]:
    return db.query(User).order_by(User.username).all()
