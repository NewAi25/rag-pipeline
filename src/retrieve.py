"""STEP 4 of the RAG pipeline: retrieval.

Two modes, selected by ``settings.retrieval_mode``:

* ``vector`` (default) — embed the question, pull the top-k most-similar
  chunks from Chroma. Cheap, robust to paraphrase, weaker on rare
  keywords / identifiers.
* ``hybrid`` — combine vector similarity with BM25 keyword search via
  Reciprocal Rank Fusion. See :mod:`src.hybrid`.

We do not call the LLM here — keeping retrieval and generation in
separate modules makes it trivial to evaluate retrieval quality on its
own (e.g. "did we surface the right chunk for this question?").
"""

from __future__ import annotations

from .config import Settings
from .embeddings import get_embedding_provider
from .vectorstore import Retrieved, get_vector_store


def _vector_retrieve(question: str, *, settings: Settings) -> list[Retrieved]:
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


def retrieve(question: str, *, settings: Settings) -> list[Retrieved]:
    """Return the top-k chunks most relevant to ``question``."""
    if not question.strip():
        return []
    if settings.retrieval_mode == "hybrid":
        # Local import keeps `rank_bm25` from being required when only
        # the default vector path is used (e.g. in trimmed-down installs).
        from .hybrid import hybrid_retrieve
        return hybrid_retrieve(question, settings=settings)
    return _vector_retrieve(question, settings=settings)
