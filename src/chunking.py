"""STEP 1 of the RAG pipeline: chunking.

Why chunk at all?
-----------------
LLMs have a fixed context window, embeddings are computed per chunk, and
retrieval works best when each chunk is roughly one self-contained idea.
If we embed an entire 200-page PDF as one vector, every query gets the
same blurry answer. If we embed each individual sentence, we lose the
surrounding context that makes the sentence meaningful. Paragraph-sized
chunks (~500 tokens) are the sweet spot for most documents.

Why overlap?
------------
A naive non-overlapping split can cut an idea in half between chunk N
and chunk N+1, and neither chunk will then rank well for a query about
that idea. A small overlap (~10% of chunk size) keeps the boundary
region present in both chunks so retrieval stays robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


# We use the same tokenizer family as the embedding model so our token
# counts line up with what OpenAI bills/limits. cl100k_base covers
# text-embedding-3-* and gpt-4o-*.
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text plus the metadata we need to cite it later."""

    text: str
    # 1-indexed chunk number within the source document.
    index: int
    # Token count of this chunk (handy for debugging / cost estimation).
    n_tokens: int
    # Source document identifier (usually the PDF filename).
    source: str


def count_tokens(text: str) -> int:
    """Count tokens using the same encoding the embedding model uses."""
    return len(_ENCODER.encode(text))


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split ``text`` into overlapping token-bounded chunks.

    Parameters
    ----------
    text:
        The full document text (already extracted from PDF/HTML/etc).
    source:
        A short identifier for the source document, stored on each chunk
        so we can cite it in the final answer.
    chunk_size:
        Target chunk length in tokens.
    overlap:
        Number of tokens shared between consecutive chunks.

    Returns
    -------
    A list of :class:`Chunk`, in document order. Empty input yields ``[]``.

    Notes
    -----
    We chunk on token boundaries rather than character/word boundaries so
    that each chunk fits cleanly under the embedding model's per-input
    token limit and our cost accounting is exact.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        # If overlap >= chunk_size the loop below never advances.
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    token_ids = _ENCODER.encode(text)
    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 1
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        window = token_ids[start:end]
        chunk_str = _ENCODER.decode(window).strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    text=chunk_str,
                    index=idx,
                    n_tokens=len(window),
                    source=source,
                )
            )
            idx += 1
        if end == len(token_ids):
            break
        start += step

    return chunks
