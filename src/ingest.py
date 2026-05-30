"""Ingestion pipeline: PDF -> text -> chunks -> embeddings -> vector store.

This is where STEPS 1, 2, and 3 are wired together. After ``ingest_pdf``
returns, your local Chroma collection contains one vector per chunk and
is ready to be queried by :mod:`src.retrieve`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .chunking import Chunk, chunk_text
from .config import Settings
from .embeddings import get_embedding_provider
from .vectorstore import get_vector_store


def extract_pdf_text(pdf_path: Path) -> str:
    """Pull plain text out of a PDF using pypdf.

    pypdf does a reasonable job on text-based PDFs. Scanned PDFs (images
    of text) would need an OCR step like Tesseract first — out of scope
    for this demo.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        # extract_text can return None on pages with no extractable text
        # (e.g. an image-only page); skip those gracefully.
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _batched(items: list, batch_size: int) -> Iterable[list]:
    """Yield successive ``batch_size`` slices of ``items``.

    OpenAI accepts large batches, but we cap them so a single failure
    doesn't lose 10k embeddings of work.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ingest_pdf(
    pdf_path: Path,
    *,
    settings: Settings,
    batch_size: int = 128,
) -> int:
    """Run the full ingest pipeline for one PDF.

    Returns the number of chunks written to the vector store.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- STEP 1: chunk ---
    raw_text = extract_pdf_text(pdf_path)
    chunks: list[Chunk] = chunk_text(
        raw_text,
        source=pdf_path.name,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise RuntimeError(
            f"No extractable text found in {pdf_path.name}. "
            "Is it a scanned/image-only PDF?"
        )

    # --- STEP 2: embed ---
    embedder = get_embedding_provider(settings)

    # --- STEP 3: store ---
    store = get_vector_store(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )

    total_written = 0
    for batch in _batched(chunks, batch_size):
        texts = [c.text for c in batch]
        vectors = embedder.embed(texts)
        ids = [f"{c.source}::chunk-{c.index}" for c in batch]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.index,
                "n_tokens": c.n_tokens,
            }
            for c in batch
        ]
        store.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        total_written += len(batch)

    return total_written
