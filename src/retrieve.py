"""STEP 4 of the RAG pipeline: retrieval.

Given a user question, we:
  1. Embed the question with the SAME model used at ingest time.
  2. Ask the vector store for the top-k most similar chunks.
  3. Return those chunks so :mod:`src.generate` can build a prompt.

We do not call the LLM here — keeping retrieval and generation in
separate modules makes it trivial to evaluate retrieval quality on its
own (e.g. "did we surface the right chunk for this question?").
"""

from __future__ import annotations

from .config import Settings
from .embeddings import get_embedding_provider
from .vectorstore import Retrieved, get_vector_store


def retrieve(question: str, *, settings: Settings) -> list[Retrieved]:
    """Return the top-k chunks most relevant to ``question``."""
    if not question.strip():
        return []

    embedder = get_embedding_provider(settings)
    store = get_vector_store(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )

    if store.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `ingest` on a PDF first, e.g.\n"
            "  python -m src.cli ingest data/sample.pdf"
        )

    [query_vector] = embedder.embed([question])
    return store.query(embedding=query_vector, top_k=settings.top_k)
