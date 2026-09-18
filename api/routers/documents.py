"""Client document import: propose (upload, extract, anonymize, write a reviewable sidecar) ->
human review (see the sidecar's exact anonymized content) -> confirm (chunk, embed, store).

Wraps `src/documents/ingestion.py` exactly as the CLI does -- no extraction, anonymization or
chunking logic lives here. `import_document` writes the upload to a throwaway temp file, under
its own original filename (needed so the tracked `doc_id`/`filename` match what the auditor
actually uploaded), and lets `propose_import` do the rest.
"""

import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session as DbSession

from api import repository
from api.audit_log import log
from api.auth import get_current_user, get_db_session
from api.dependencies import get_record_and_session
from api.models_orm import User
from src.config import get_settings
from src.documents.extractors import ExtractionError
from src.documents.ingestion import DocumentImportError, confirm_import, propose_import
from src.report.session import ImportedDocument

router = APIRouter(prefix="/sessions/{session_id}/documents", tags=["documents"])


@router.post("", response_model=ImportedDocument, status_code=status.HTTP_201_CREATED)
async def import_document(
    session_id: str, doc_type: str = Form(...), file: UploadFile = File(...),
    db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> ImportedDocument:
    _, session = get_record_and_session(db, session_id, user)
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    with tempfile.TemporaryDirectory() as tmp:
        # Path(...).name strips any client-supplied directory components -- the file only ever
        # exists under this fresh temp directory, never combined with untrusted path segments.
        tmp_path = Path(tmp) / Path(file.filename).name
        tmp_path.write_bytes(await file.read())
        try:
            record = propose_import(session, tmp_path, doc_type, settings=get_settings())
        except (DocumentImportError, ExtractionError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    repository.save_session(db, session)
    log(db, "document_imported", user_id=user.id, target_type="session", target_id=session_id,
        details={"doc_id": record.doc_id, "filename": record.filename})
    return record


@router.get("", response_model=List[ImportedDocument])
def list_documents(
    session_id: str, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> List[ImportedDocument]:
    _, session = get_record_and_session(db, session_id, user)
    return session.imported_documents


@router.get("/{doc_id}/sidecar")
def get_sidecar(
    session_id: str, doc_id: str, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> dict:
    """The exact anonymized text a human must review before confirming -- the whole point of the
    two-step gate is that this is what gets embedded, not the original document."""
    _, session = get_record_and_session(db, session_id, user)
    record = next((d for d in session.imported_documents if d.doc_id == doc_id), None)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found in this session")
    sidecar_path = Path(record.sidecar_path)
    if not sidecar_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sidecar file missing on disk")
    log(db, "document_sidecar_viewed", user_id=user.id, target_type="session", target_id=session_id,
        details={"doc_id": doc_id})
    return {"doc_id": doc_id, "status": record.status.value, "content": sidecar_path.read_text(encoding="utf-8")}


@router.post("/{doc_id}/confirm", response_model=ImportedDocument)
def confirm_document(
    session_id: str, doc_id: str, db: DbSession = Depends(get_db_session), user: User = Depends(get_current_user),
) -> ImportedDocument:
    _, session = get_record_and_session(db, session_id, user)
    try:
        record = confirm_import(session, doc_id, settings=get_settings())
    except DocumentImportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    repository.save_session(db, session)
    log(db, "document_confirmed", user_id=user.id, target_type="session", target_id=session_id,
        details={"doc_id": doc_id, "chunk_count": record.chunk_count})
    return record
