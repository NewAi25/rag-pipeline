"""STEP 2 of the RAG pipeline: embeddings.

This module wraps the embedding provider behind a tiny interface so the
rest of the pipeline never imports the OpenAI SDK directly. To swap in
a different provider (Cohere, Voyage, a local sentence-transformers
model, etc.) you only need to implement :class:`EmbeddingProvider` and
update :func:`get_embedding_provider`.

Both OpenAI and Google Gemini are supported out of the box because
Gemini exposes an OpenAI-compatible endpoint — we just point the same
:class:`openai.OpenAI` client at a different ``base_url`` and key.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from openai import OpenAI

from ._retry import with_retries
from .config import Settings, require_api_key


class EmbeddingProvider(Protocol):
    """Minimal interface every embedding backend must satisfy."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts and return one vector per input."""
        ...


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for any OpenAI-wire-format endpoint.

    Works with vanilla OpenAI (``base_url=None``) and Gemini's
    OpenAI-compatible endpoint (``base_url`` pointing at
    ``https://generativelanguage.googleapis.com/v1beta/openai/``).

    We batch all inputs into a single API call where possible — the
    endpoint accepts a list and returns vectors in the same order, so
    batching is dramatically cheaper than one call per chunk.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        # Passing ``base_url=None`` lets the SDK use its built-in default
        # (api.openai.com); any other value (e.g. Gemini's) overrides it.
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # The API rejects empty strings; replace with a single space so the
        # output length still matches the input length.
        cleaned = [t if t.strip() else " " for t in texts]

        def _call():
            return self._client.embeddings.create(
                model=self.model,
                input=list(cleaned),
            )

        # Free tiers (especially Gemini's) throttle aggressively — retry on 429.
        response = with_retries(_call)
        # The API returns items in the same order as the input.
        return [item.embedding for item in response.data]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory: return the configured embedding provider.

    To plug in a non-OpenAI-compatible backend, branch here on
    ``settings.provider`` and return an alternate implementation.
    """
    api_key = require_api_key(settings)
    return OpenAICompatibleEmbeddingProvider(
        api_key=api_key,
        model=settings.embed_model,
        base_url=settings.base_url,
    )
