import hashlib
import math
import re
import unicodedata
import uuid
from typing import List, Sequence

import pytest
import requests
from qdrant_client import QdrantClient

from src.config import PROJECT_ROOT, Settings, get_settings
from src.generation.models import AuditFinding, EvidenceStatus, FindingType, SupportingSource
from src.ingestion.models import GUIDANCE, KnowledgeChunk
from src.vectorstore import QdrantStore, SearchHit

REAL_KB_DIR = PROJECT_ROOT / "data" / "raw"


class HashingEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors. No network."""

    def __init__(self, dim: int = 128, model: str = "fake-hashing-embedder"):
        self.dim = dim
        self.model = model
        self.calls = 0
        self.embedded_texts: List[str] = []

    @property
    def fingerprint(self) -> str:
        return f"{self.model}:{self.dim}"

    def _vector(self, text: str) -> List[float]:
        folded = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]{3,}", folded):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        self.embedded_texts.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class FakeCrossEncoder:
    """Scores a (query, passage) pair by shared-word overlap. Deterministic, no model download."""

    def __init__(self):
        self.calls: List[list] = []
        self.batch_sizes: List[int] = []

    def predict(self, pairs, batch_size=None):
        self.calls.append(list(pairs))
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


def make_chunk(chunk_id: str, text: str, **overrides) -> KnowledgeChunk:
    """A KnowledgeChunk with every required field defaulted, for tests that only care about a few."""
    defaults = dict(
        chunk_id=chunk_id, doc_id=chunk_id.split(":")[0], doc_title="Test Document",
        doc_type="iso_standard", standard="ISO/IEC 27001", standard_version="2022",
        category="iso27001", category_label="ISO 27001", language="fr", source_file="test.txt",
        section_number=1, section_title="TEST", section_slug="test", chunk_index=0, chunk_count=1,
        unit_type=GUIDANCE, context=None, heading="Test Document › Test", text=text,
        word_count=len(text.split()), references=[],
    )
    defaults.update(overrides)
    return KnowledgeChunk(**defaults)


def make_hit(chunk_id: str, text: str, score: float = 1.0, **overrides) -> SearchHit:
    """A SearchHit wrapping a make_chunk() payload, for tests that need retrieval-shaped evidence."""
    chunk = make_chunk(chunk_id, text, **overrides)
    return SearchHit(id=chunk.point_id, score=score, payload=chunk.payload())


def make_audit_finding(
    finding_type: FindingType = FindingType.CONSTAT,
    iso_reference: list | None = None,
    text: str = "Le constat.",
    evidence_status: EvidenceStatus = EvidenceStatus.SUPPORTED,
    requires_human_review: bool = False,
    confidence: float = 0.8,
    **overrides,
) -> AuditFinding:
    """An AuditFinding with every required field defaulted, for report/session tests that only
    care about a few (finding_type, iso_reference, text)."""
    iso_reference = iso_reference or []
    defaults = dict(
        observation="obs", finding=text, finding_type=finding_type, requirement="Exigence.",
        iso_reference=iso_reference, justification="Justification.", risk="Risque.", recommendation="Recommandation.",
        evidence_status=evidence_status, requires_human_review=requires_human_review, confidence=confidence,
        supporting_sources=[
            SupportingSource(chunk_id="c1", doc_id="d1", doc_title="Doc", references=iso_reference, rerank_score=1.0)
        ] if iso_reference else [],
    )
    defaults.update(overrides)
    return AuditFinding(**defaults)


def qdrant_reachable(settings: Settings) -> bool:
    try:
        headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
        return requests.get(f"{settings.qdrant_url}/collections", headers=headers, timeout=5).ok
    except requests.RequestException:
        return False


def _ollama_model_available(base_url: str, model: str) -> bool:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        names = [m["name"] for m in response.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return False
    return any(name == model or name.startswith(f"{model.split(':')[0]}:") for name in names)


def ollama_model_available(settings: Settings) -> bool:
    return _ollama_model_available(settings.embedding_base_url, settings.embedding_model)


def ollama_generation_model_available(settings: Settings) -> bool:
    return _ollama_model_available(settings.generation_base_url, settings.generation_model)


def huggingface_reachable() -> bool:
    try:
        return requests.head("https://huggingface.co", timeout=5).ok
    except requests.RequestException:
        return False


def kb_document(doc_id: str, title: str, category: str, sections: dict[int, tuple[str, str]]) -> str:
    """Render a document in the knowledge-base file template."""
    bar = "=" * 80
    parts = [bar, f"DOCUMENT: {title}", bar, f"ID:             {doc_id}", f"Category:       {category}", "Language:       fr", bar, ""]
    for number, (section_title, body) in sections.items():
        parts += ["", bar, f"  SECTION {number:02d} : {section_title}", bar, "", body, ""]
    return "\n".join(parts)


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def memory_store() -> QdrantStore:
    return QdrantStore(QdrantClient(location=":memory:"), collection="test_kb")


@pytest.fixture
def server_store(settings):
    if not qdrant_reachable(settings):
        pytest.skip(f"Qdrant not reachable at {settings.qdrant_url}")
    store = QdrantStore.from_settings(settings, collection=f"pytest_{uuid.uuid4().hex[:10]}")
    yield store
    store.delete_collection()
