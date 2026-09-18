"""Central configuration, read from environment variables and an optional `.env` file.

Every value has a portable default; override per machine in `.env` (see `.env.example`).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Knowledge base sources
    kb_raw_dir: Path = Path("data/raw")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "iso27001_kb"
    qdrant_timeout: int = 60

    # Embeddings (served by a local Ollama instance)
    embedding_base_url: str = "http://localhost:11434"
    # bge-m3: multilingual (French corpus, French or English auditor input); see README benchmark.
    embedding_model: str = "bge-m3"
    embedding_batch_size: int = Field(default=16, ge=1)
    embedding_timeout: int = 120
    # Task prefixes for models trained with them, e.g. "search_document: " / "search_query: " for nomic-embed-text.
    embedding_document_prefix: str = ""
    embedding_query_prefix: str = ""

    # Chunking (in words), see src/ingestion/chunker.py: guidance units shorter than
    # CHUNK_MIN_WORDS are merged with a neighbour; units longer than CHUNK_MAX_WORDS are split.
    chunk_min_words: int = Field(default=40, ge=0)
    chunk_max_words: int = Field(default=200, ge=20)

    # Stage 2 retrieval: BM25 + semantic, fused with RRF, then reranked.
    retrieval_candidates_per_retriever: int = Field(default=20, ge=1)  # taken from each of BM25 and semantic
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_fusion_top_k: int = Field(default=20, ge=1)  # candidates handed to the reranker

    # multilingual (mMARCO, incl. French); see README for the comparison against BAAI/bge-reranker-v2-m3.
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_device: str = "cpu"
    rerank_batch_size: int = Field(default=16, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)  # final context size returned by the pipeline

    # Stage 3 generation (a local Ollama chat model); see README for the model comparison.
    generation_base_url: str = "http://localhost:11434"
    generation_model: str = "llama3:8b"
    generation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=600, ge=1)
    generation_timeout: int = Field(default=120, ge=1)
    generation_language: str = "fr"

    # Stage 4 report generation
    report_output_dir: Path = Path("data/reports")

    # Client document import: per-session sidecar files (anonymized text, reviewed before
    # embedding) and the prefix for each session's own Qdrant collection.
    document_import_dir: Path = Path("data/imports")

    # Development phase: persistent storage (replaces the JSON-file session store) and API auth.
    # A dedicated Postgres container/port (5433), never the machine's other Postgres instance(s).
    database_url: str = "postgresql+psycopg2://audit_app:dev_local_only_change_me@localhost:5433/iso27001_audit"
    # Local username/password + JWT for this development pass; structured to be replaced by
    # SSO/Azure AD/MFA (cahier des charges §5.3) without changing the role model or API surface.
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=480, ge=1)  # one working day

    @field_validator("kb_raw_dir", "report_output_dir", "document_import_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("qdrant_api_key")
    @classmethod
    def _empty_key_is_none(cls, value: str | None) -> str | None:
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
