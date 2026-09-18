"""API request bodies. Response models reuse the existing, already-validated domain Pydantic
models (`AuditSession`, `AuditFinding`, `ImportedDocument`, ...) directly -- no parallel schema to
keep in sync. Only what the client sends in gets a schema of its own here.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    full_name: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    full_name: str
    role: str = "auditeur"


class UserOut(BaseModel):
    id: str
    username: str
    full_name: str
    role: str
    is_active: bool


class TeamMemberIn(BaseModel):
    name: str
    role: str


class CreateSessionRequest(BaseModel):
    title: str
    client_name: str
    client_aliases: List[str] = Field(default_factory=list)
    scope: str
    standards: List[str] = Field(default_factory=list)
    audit_team: List[TeamMemberIn] = Field(default_factory=list)
    reference: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SessionSummary(BaseModel):
    id: str
    title: str
    client_name: str
    reference: Optional[str]
    status: str
    owner_id: str
    pending_count: int
    approved_count: int
    rejected_count: int
    imported_documents: int


class AddFindingRequest(BaseModel):
    observation: str
    language: str = "fr"


class ReviewFindingRequest(BaseModel):
    approve: bool
    severity: Optional[str] = None
    edited_text: Optional[str] = None
    reviewer: Optional[str] = None


class SetCertificationDecisionRequest(BaseModel):
    recommends: bool
    decided_by: Optional[str] = None
