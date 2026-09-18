"""Knowledge-base ingestion: discover -> parse/clean -> segment/chunk -> metadata -> embed -> Qdrant."""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import Settings, get_settings
from src.embeddings import Embedder, build_embedder
from src.ingestion.chunker import chunk_document
from src.ingestion.models import KnowledgeChunk, ParsedDocument
from src.ingestion.parser import SUPPORTED_EXTENSIONS, parse_document
from src.vectorstore import QdrantStore, VectorRecord

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    pass


@dataclass
class DocumentReport:
    source_file: str
    doc_id: str
    doc_type: str
    structured: bool
    sections: int
    chunks: int
    points_stored: int = 0


@dataclass
class IngestionReport:
    collection: str
    embedding_model: str
    vector_size: int = 0
    collection_created: bool = False
    documents: List[DocumentReport] = field(default_factory=list)
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    points_in_collection: int = 0
    duration_seconds: float = 0.0


def discover_documents(raw_dir: Path) -> List[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise IngestionError(f"Knowledge-base directory not found: {raw_dir}")
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def prepare_documents(
    paths: List[Path], root: Path, min_words: int, max_words: int
) -> List[Tuple[ParsedDocument, List[KnowledgeChunk]]]:
    prepared = []
    seen: Dict[str, str] = {}
    for path in paths:
        doc = parse_document(path, root)
        if doc.doc_id in seen:
            raise IngestionError(f"Duplicate doc_id '{doc.doc_id}' in {doc.source_file} and {seen[doc.doc_id]}")
        seen[doc.doc_id] = doc.source_file
        chunks = chunk_document(doc, min_words, max_words)
        if not chunks:
            logger.warning("Skipping %s: no text content", doc.source_file)
            continue
        prepared.append((doc, chunks))
    return prepared


def embedding_hash(chunk: KnowledgeChunk, embedder: Embedder) -> str:
    """Changes whenever the stored vector of the chunk would change."""
    return hashlib.sha256(f"{embedder.fingerprint}\n{chunk.embedding_text()}".encode("utf-8")).hexdigest()


def _vectors(
    chunks: List[KnowledgeChunk], hashes: Dict[str, str], embedder: Embedder, store: QdrantStore, recreate: bool
) -> Tuple[Dict[str, List[float]], int]:
    """Vector per chunk_id, reusing stored vectors whose embedding hash is unchanged."""
    stored = {}
    if not recreate and store.collection_exists():
        stored = {hit.id: hit for hit in store.get([c.point_id for c in chunks], with_vectors=True)}

    vectors: Dict[str, List[float]] = {}
    to_embed: List[KnowledgeChunk] = []
    for chunk in chunks:
        hit = stored.get(chunk.point_id)
        if hit is not None and hit.vector and hit.payload.get("embedding_hash") == hashes[chunk.chunk_id]:
            vectors[chunk.chunk_id] = hit.vector
        else:
            to_embed.append(chunk)

    if to_embed:
        logger.info("Embedding %d chunks with %s (%d unchanged)", len(to_embed), embedder.model, len(vectors))
        embedded = embedder.embed_documents([chunk.embedding_text() for chunk in to_embed])
        if len(embedded) != len(to_embed):
            raise IngestionError(f"Expected {len(to_embed)} embeddings, got {len(embedded)}")
        vectors.update(zip((chunk.chunk_id for chunk in to_embed), embedded))
    return vectors, len(to_embed)


def ingest(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
    recreate: bool = False,
    prune: bool = True,
) -> IngestionReport:
    """Synchronise the Qdrant collection with every supported file of `settings.kb_raw_dir`.

    Idempotent and incremental: point ids derive from chunk ids, only chunks whose embedding input
    or embedding model changed are embedded, chunks that disappeared from a document are removed,
    and with `prune` documents no longer present are removed. All embeddings are computed before
    the first write, so an embedding failure leaves the collection untouched.
    """
    started = time.perf_counter()
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    store = store or QdrantStore.from_settings(settings)
    report = IngestionReport(collection=store.collection, embedding_model=embedder.model)

    paths = discover_documents(settings.kb_raw_dir)
    if not paths:
        raise IngestionError(f"No supported documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) in {settings.kb_raw_dir}")
    logger.info("Discovered %d documents in %s", len(paths), settings.kb_raw_dir)

    prepared = prepare_documents(paths, settings.kb_raw_dir, settings.chunk_min_words, settings.chunk_max_words)
    chunks = [chunk for _, doc_chunks in prepared for chunk in doc_chunks]
    if not chunks:
        raise IngestionError(f"No text content in the documents of {settings.kb_raw_dir}")
    report.chunks_total = len(chunks)
    logger.info("Created %d chunks from %d documents", len(chunks), len(prepared))

    hashes = {chunk.chunk_id: embedding_hash(chunk, embedder) for chunk in chunks}
    vectors, report.chunks_embedded = _vectors(chunks, hashes, embedder, store, recreate)
    report.chunks_reused = len(chunks) - report.chunks_embedded
    vector_sizes = {len(vector) for vector in vectors.values()}
    if len(vector_sizes) != 1:
        raise IngestionError(f"Inconsistent embedding sizes: {sorted(vector_sizes)}")
    report.vector_size = vector_sizes.pop()

    if recreate:
        store.recreate_collection(report.vector_size)
        report.collection_created = True
    else:
        report.collection_created = store.ensure_collection(report.vector_size)

    for doc, doc_chunks in prepared:
        records = [
            VectorRecord(c.point_id, vectors[c.chunk_id], {**c.payload(), "embedding_hash": hashes[c.chunk_id]})
            for c in doc_chunks
        ]
        store.upsert(records)
        store.delete_stale_chunks(doc.doc_id, [r.id for r in records])
        stored = store.count({"doc_id": doc.doc_id})
        report.documents.append(
            DocumentReport(
                source_file=doc.source_file,
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                structured=doc.structured,
                sections=len(doc.sections),
                chunks=len(doc_chunks),
                points_stored=stored,
            )
        )
        if stored != len(doc_chunks):
            raise IngestionError(f"{doc.doc_id}: {len(doc_chunks)} chunks written but {stored} points stored")
        logger.info("Stored %s: %d chunks", doc.doc_id, stored)

    if prune:
        store.delete_documents_except([doc.doc_id for doc, _ in prepared])

    report.points_in_collection = store.count()
    if prune and report.points_in_collection != report.chunks_total:
        raise IngestionError(
            f"Collection has {report.points_in_collection} points, expected {report.chunks_total}"
        )
    report.duration_seconds = round(time.perf_counter() - started, 2)
    return report
