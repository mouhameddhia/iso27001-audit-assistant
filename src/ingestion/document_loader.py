"""
document_loader.py

Turns the raw scraped documents (data/raw/) into clean, chunked text
(data/processed/chunks.json) that the retrievers can index.

Pipeline: read files -> clean text -> split into chunks -> save as JSON.

Supported input formats: .txt, .md (read directly), .pdf (best-effort, only
if `pypdf` is installed -- we don't force a heavy dependency for the
prototype). Add more loaders here later if you ingest .docx/.xlsx sources.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    """A single retrievable piece of text."""
    id: str
    source: str  # original filename, for traceability / citations later
    text: str


def _read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Reading PDFs requires `pypdf`. Install it with: "
            "pip install pypdf --break-system-packages"
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


LOADERS = {
    ".txt": _read_txt_or_md,
    ".md": _read_txt_or_md,
    ".pdf": _read_pdf,
}


def clean_text(text: str) -> str:
    """Normalize whitespace so chunking works on tidy text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)          # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse 3+ blank lines to 1
    return text.strip()


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    """Split text into overlapping chunks, sized in words.

    Word-count chunking is simple and good enough for a prototype: it
    keeps chunks a readable, roughly-fixed size without needing a
    tokenizer. `overlap` repeats a few words at chunk boundaries so a
    sentence split across two chunks doesn't lose context entirely.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


def load_and_chunk_documents(raw_dir: Path) -> List[Chunk]:
    """Read every supported file in `raw_dir`, clean it, and chunk it."""
    raw_dir = Path(raw_dir)
    chunks: List[Chunk] = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue

        raw_text = LOADERS[path.suffix.lower()](path)
        text = clean_text(raw_text)
        if not text:
            continue

        for i, chunk_str in enumerate(chunk_text(text)):
            chunks.append(
                Chunk(id=f"{path.stem}_{i:04d}", source=path.name, text=chunk_str)
            )

    return chunks


def save_chunks(chunks: List[Chunk], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def load_chunks(input_path: Path) -> List[Chunk]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]


if __name__ == "__main__":
    # Simple CLI: python -m src.ingestion.document_loader
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed" / "chunks.json"

    chunks = load_and_chunk_documents(raw_dir)
    save_chunks(chunks, processed_path)
    print(f"Loaded {len(chunks)} chunks from {raw_dir} -> {processed_path}")
