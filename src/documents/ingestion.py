"""Session-scoped ingestion of client documents (previous audit reports, SoA, risk analyses,
procedures, policies -- cahier des charges §4.3), as an explicit two-step human gate:

    propose_import()  extract -> anonymize -> write a reviewable sidecar file -> STOP
    (human reviews the sidecar file's content)
    confirm_import()  re-read *that exact sidecar* -> parse -> chunk -> embed -> store

Nothing is embedded or stored until `confirm_import` is called on a sidecar the auditor has
actually looked at -- the sidecar is what gets embedded, not the original file, so what the
auditor reviewed is exactly what reaches the model. Storage is a Qdrant collection scoped to this
one session only (`AuditSession.qdrant_collection_suffix`), never the shared knowledge-base
collection: a client's documents must never be retrievable from another client's session.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import Settings, get_settings
from src.documents.extractors import extract_text
from src.embeddings import Embedder, build_embedder
from src.ingestion.chunker import chunk_document
from src.ingestion.models import KnowledgeChunk
from src.ingestion.parser import fallback_doc_id, parse_text
from src.report.session import AuditSession, ImportedDocument, ImportStatus
from src.vectorstore import QdrantStore, VectorRecord


class DocumentImportError(RuntimeError):
    pass


def session_collection_name(session: AuditSession, settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    return f"{settings.qdrant_collection}{session.qdrant_collection_suffix}"


def _sidecar_path(session: AuditSession, doc_id: str, settings: Settings) -> Path:
    return settings.document_import_dir / session.metadata.session_id / f"{doc_id}.anonymized.txt"


def propose_import(
    session: AuditSession, path: Path, doc_type: str, settings: Optional[Settings] = None,
) -> ImportedDocument:
    """Extracts and anonymizes `path`, writes the result to a sidecar file for human review, and
    records it as `PENDING_REVIEW`. Does not touch Qdrant. Mutates `session` in place (including
    its `anonymization_mapping`) but does not save it -- the caller persists it (see the CLI).
    """
    settings = settings or get_settings()
    path = Path(path)
    doc_id = fallback_doc_id(path.name)

    text = extract_text(path)
    if not text.strip():
        raise DocumentImportError(f"No text content extracted from '{path.name}'")

    anonymizer = session.anonymizer
    anonymized_text, _ = anonymizer.anonymize(text)
    session.sync_anonymizer(anonymizer)

    sidecar_path = _sidecar_path(session, doc_id, settings)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(anonymized_text, encoding="utf-8")

    record = ImportedDocument(
        doc_id=doc_id, filename=path.name, doc_type=doc_type, sidecar_path=str(sidecar_path),
        status=ImportStatus.PENDING_REVIEW, imported_at=datetime.now(timezone.utc).isoformat(),
    )
    session.imported_documents = [d for d in session.imported_documents if d.doc_id != doc_id] + [record]
    return record


def _find_document(session: AuditSession, doc_id: str) -> ImportedDocument:
    for record in session.imported_documents:
        if record.doc_id == doc_id:
            return record
    raise DocumentImportError(f"No imported document with doc_id '{doc_id}' in this session")


def session_chunks(session: AuditSession, settings: Optional[Settings] = None) -> List[KnowledgeChunk]:
    """Re-derives every confirmed document's chunks from its sidecar file (same "re-read the
    source of truth" pattern as `BM25Retriever.from_settings` over the knowledge base): nothing
    beyond the sidecar files and the session record needs to be kept in sync. Uses the same
    chunking parameters as `confirm_import`, so a chunk's point_id here matches the one already
    stored in Qdrant -- needed for BM25 and semantic hits of the same chunk to fuse correctly.
    """
    settings = settings or get_settings()
    chunks: List[KnowledgeChunk] = []
    for record in session.confirmed_documents:
        sidecar_path = Path(record.sidecar_path)
        if not sidecar_path.exists():
            raise DocumentImportError(f"Sidecar file missing for confirmed document '{record.doc_id}': {sidecar_path}")
        text = sidecar_path.read_text(encoding="utf-8")
        parsed = parse_text(text, source_file=record.filename)
        chunks.extend(chunk_document(parsed, settings.chunk_min_words, settings.chunk_max_words))
    return chunks


def confirm_import(
    session: AuditSession, doc_id: str, settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None, store: Optional[QdrantStore] = None,
) -> ImportedDocument:
    """Re-reads the exact sidecar file `propose_import` wrote (never the original document again),
    chunks and embeds it, and upserts it into this session's own Qdrant collection. Re-confirming
    an updated sidecar for the same doc_id replaces its old chunks (Stage 1's `delete_stale_chunks`
    pattern), it does not duplicate them. Mutates `session` in place; the caller persists it.
    """
    settings = settings or get_settings()
    record = _find_document(session, doc_id)

    sidecar_path = Path(record.sidecar_path)
    if not sidecar_path.exists():
        raise DocumentImportError(f"Sidecar file not found (was it moved or deleted?): {sidecar_path}")
    text = sidecar_path.read_text(encoding="utf-8")

    parsed = parse_text(text, source_file=record.filename)
    chunks = chunk_document(parsed, settings.chunk_min_words, settings.chunk_max_words)
    if not chunks:
        raise DocumentImportError(f"No chunkable content in the reviewed sidecar for '{record.filename}'")

    embedder = embedder or build_embedder(settings)
    store = store or QdrantStore.from_settings(settings, collection=session_collection_name(session, settings))

    vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
    if len(vectors) != len(chunks):
        raise DocumentImportError(f"Expected {len(chunks)} embeddings, got {len(vectors)}")
    vector_size = len(vectors[0])
    store.ensure_collection(vector_size)

    records = [VectorRecord(c.point_id, v, c.payload()) for c, v in zip(chunks, vectors)]
    store.upsert(records)
    store.delete_stale_chunks(parsed.doc_id, [r.id for r in records])

    record.status = ImportStatus.CONFIRMED
    record.chunk_count = len(chunks)
    record.confirmed_at = datetime.now(timezone.utc).isoformat()
    return record


def close_session_collection(
    session: AuditSession, settings: Optional[Settings] = None, store: Optional[QdrantStore] = None,
) -> None:
    """Deletes this session's Qdrant collection. Irreversible -- call only once the session's
    report has been generated/exported and the imported documents are no longer needed."""
    settings = settings or get_settings()
    store = store or QdrantStore.from_settings(settings, collection=session_collection_name(session, settings))
    store.delete_collection()
