"""Hybrid retrieval: BM25 + dense vectors, fused via Reciprocal Rank Fusion.

Why hybrid?
-----------
Pure dense (vector) retrieval is great at matching *meaning* but can miss
rare keywords, identifiers, and proper nouns (model names, error codes,
section numbers) — the embedding model smooths them into similar-looking
neighbours. Pure sparse (BM25) is the opposite: it nails exact lexical
matches but is blind to paraphrase. Combining the two via Reciprocal
Rank Fusion (RRF) gives you both — and is the simplest competitive
fusion in the literature (Cormack et al., 2009).

RRF in one line:
    score(doc) = sum over each retriever of  1 / (k + rank_of_doc_in_that_retriever)

A doc that ranks #1 in BM25 and #5 in vector gets a higher fused score
than one that ranks #3 in only vector. The constant ``k`` (default 60)
damps the contribution of top-1 hits so a single retriever can't dominate.

We intentionally do NOT persist the BM25 index. It's rebuilt per query
from whatever is currently in the vector store, which keeps the two
retrievers in sync and avoids a second on-disk artefact to manage. For
collections up to a few thousand chunks this is well under a millisecond.
"""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from .config import Settings
from .embeddings import get_embedding_provider
from .vectorstore import Retrieved, VectorStore, get_vector_store


# Cheap word tokenizer — lowercase, split on non-word characters. Good
# enough for English prose; would be wrong for Chinese/Japanese (you'd
# want a real tokenizer) but matches the rest of this demo's scope.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked ID lists into one, highest score first.

    ``ranked_lists`` is a list of lists of doc IDs in rank order
    (best first). Each ID's fused score is the sum, across all input
    lists, of ``1 / (k + rank)``.
    """
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """Combines vector similarity with BM25 keyword search via RRF.

    Built lazily: the BM25 index is constructed the first time ``query``
    is called, by reading every document out of the vector store. For
    small/medium corpora (< ~10k chunks) that's fast enough to do at
    query time and avoids cache-invalidation bugs.
    """

    def __init__(self, *, store: VectorStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._embedder = get_embedding_provider(settings)
        self._all: list[Retrieved] | None = None
        self._bm25: BM25Okapi | None = None

    def _ensure_indexed(self) -> None:
        if self._bm25 is not None:
            return
        docs = self._store.all_documents()
        if not docs:
            raise RuntimeError(
                "The vector store is empty. Run `ingest` on a PDF first."
            )
        self._all = docs
        # rank_bm25 needs pre-tokenized documents.
        tokenized = [_tokenize(d.text) for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, question: str, *, top_k: int) -> list[Retrieved]:
        """Return the top-k fused results for ``question``."""
        self._ensure_indexed()
        assert self._all is not None and self._bm25 is not None

        # Pull a wider candidate pool from each retriever than we'll
        # ultimately return — RRF benefits from seeing more of the tail.
        pool = max(top_k * 4, 20)

        # --- Vector side ---
        [qv] = self._embedder.embed([question])
        vector_hits = self._store.query(embedding=qv, top_k=pool)
        vector_ranking = [r.id for r in vector_hits]

        # --- BM25 side ---
        bm25_scores = self._bm25.get_scores(_tokenize(question))
        # argsort descending
        bm25_order = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:pool]
        bm25_ranking = [self._all[i].id for i in bm25_order]

        # --- Fuse ---
        fused = _reciprocal_rank_fusion([vector_ranking, bm25_ranking])
        by_id = {r.id: r for r in self._all}
        # Vector hits carry real distances; prefer those when available.
        by_id.update({r.id: r for r in vector_hits})

        out: list[Retrieved] = []
        for doc_id, score in fused[:top_k]:
            base = by_id.get(doc_id)
            if base is None:
                continue
            # Store the fused score in `distance` (lower-is-better convention
            # is broken here — we negate so callers can still sort the same
            # way and the UI's "distance" label remains comparable).
            out.append(
                Retrieved(
                    id=base.id,
                    text=base.text,
                    source=base.source,
                    chunk_index=base.chunk_index,
                    distance=-score,
                )
            )
        return out


def hybrid_retrieve(question: str, *, settings: Settings) -> list[Retrieved]:
    """One-shot helper: build a retriever and run a single query."""
    store = get_vector_store(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    return HybridRetriever(store=store, settings=settings).query(
        question, top_k=settings.top_k
    )
