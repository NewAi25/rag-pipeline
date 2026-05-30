"""STEP 3 of the RAG pipeline: the vector database.

We wrap Chroma behind a tiny interface (:class:`VectorStore`) so the rest
of the codebase doesn't depend on any specific vendor. Swapping in
Pinecone, pgvector, Weaviate, Qdrant, etc. is a matter of writing a new
class with the same three methods (``add``, ``query``, ``count``).

Chroma is the default because it runs locally with zero setup — no API
keys, no servers, no cloud account. The vectors live in a single folder
on disk (``chroma_db/`` by default), which is gitignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import chromadb
from chromadb.config import Settings as ChromaClientSettings

# Chromadb ships with a posthog client whose `capture()` signature drifts
# between releases; when the bundled chromadb and posthog versions don't
# match, every Chroma call logs a noisy "Failed to send telemetry event"
# even when telemetry is disabled. The setting is still honored — only the
# error logging is silenced.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class Retrieved:
    """One result row returned from a similarity search."""

    id: str
    text: str
    source: str
    chunk_index: int
    # Lower distance = more similar. Chroma returns squared L2 by default;
    # you only need the relative ordering, not the absolute value.
    distance: float


class VectorStore(Protocol):
    """Minimal interface every vector backend must satisfy."""

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None: ...

    def query(
        self, *, embedding: Sequence[float], top_k: int
    ) -> list[Retrieved]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def all_documents(self) -> list[Retrieved]: ...


class ChromaVectorStore:
    """Chroma-backed implementation of :class:`VectorStore`.

    A single ``PersistentClient`` writes to ``persist_dir``. Re-running
    ingest is idempotent for the same ID — re-ingesting an unchanged
    document will just overwrite the existing rows.
    """

    def __init__(self, *, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaClientSettings(anonymized_telemetry=False),
        )
        # We bring our own embeddings, so we explicitly pass None as the
        # embedding function — otherwise Chroma tries to download one.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        # upsert (not add) so re-ingesting the same doc updates in place
        # rather than raising on duplicate IDs.
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def query(
        self, *, embedding: Sequence[float], top_k: int
    ) -> list[Retrieved]:
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns lists-of-lists (one entry per query). We only
        # ever send one query at a time, so index [0] is what we want.
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[Retrieved] = []
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(
                Retrieved(
                    id=id_,
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=float(dist),
                )
            )
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop and re-create the collection so all vectors are wiped.

        Cheaper and faster than deleting the persist directory by hand —
        leaves Chroma's on-disk format intact for subsequent ingests.
        """
        name = self._collection.name
        self._client.delete_collection(name=name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def all_documents(self) -> list[Retrieved]:
        """Return every chunk in the collection (no embeddings, no scores).

        Used by the hybrid retriever to build a BM25 index over the same
        corpus. ``distance`` is set to 0.0 since these aren't query hits.
        """
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids", []) or []
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        out: list[Retrieved] = []
        for id_, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            out.append(
                Retrieved(
                    id=id_,
                    text=doc or "",
                    source=str(meta.get("source", "unknown")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=0.0,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Stub: how you'd swap in Pinecone or pgvector
# ---------------------------------------------------------------------------
# To use a different vector DB, write a class with the same three methods
# (`add`, `query`, `count`) and update `get_vector_store()` below to
# return it. The rest of the pipeline doesn't care which backend is in use.
#
# Example sketch for Pinecone:
#
#   class PineconeVectorStore:
#       def __init__(self, *, api_key, index_name):
#           from pinecone import Pinecone
#           self._index = Pinecone(api_key=api_key).Index(index_name)
#       def add(self, *, ids, embeddings, documents, metadatas):
#           vectors = [
#               {"id": i, "values": list(e), "metadata": {**m, "text": d}}
#               for i, e, d, m in zip(ids, embeddings, documents, metadatas)
#           ]
#           self._index.upsert(vectors=vectors)
#       def query(self, *, embedding, top_k):
#           res = self._index.query(
#               vector=list(embedding), top_k=top_k, include_metadata=True,
#           )
#           return [
#               Retrieved(
#                   id=m.id,
#                   text=m.metadata["text"],
#                   source=m.metadata.get("source", "unknown"),
#                   chunk_index=int(m.metadata.get("chunk_index", -1)),
#                   distance=1.0 - float(m.score),  # cosine -> distance
#               )
#               for m in res.matches
#           ]
#       def count(self):
#           return int(self._index.describe_index_stats().total_vector_count)
#
# Example sketch for pgvector (using psycopg + a `chunks` table with
# columns id text, source text, chunk_index int, document text,
# embedding vector(1536)):
#
#   class PgVectorStore:
#       def __init__(self, *, dsn): ...
#       def add(...):  # INSERT ... ON CONFLICT (id) DO UPDATE
#       def query(...):  # SELECT ... ORDER BY embedding <=> %s LIMIT %s
#       def count(...):  # SELECT count(*) FROM chunks
#
# ---------------------------------------------------------------------------


def get_vector_store(*, persist_dir: Path, collection_name: str) -> VectorStore:
    """Factory: return the configured vector store. Default = Chroma."""
    return ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
