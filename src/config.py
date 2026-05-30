"""Central configuration for the RAG demo.

All tunable knobs (provider, model names, chunk size, top_k, storage paths)
live here. Values are sourced from environment variables (loaded from a local
`.env` file if present), with sensible defaults so the project runs out of
the box.

Two LLM providers are supported, selected via the ``PROVIDER`` env var:

* ``gemini`` (default) — Google's free tier via the **OpenAI-compatible**
  endpoint at https://generativelanguage.googleapis.com/v1beta/openai/.
  Use ``GEMINI_API_KEY`` (free key from https://aistudio.google.com/apikey).
* ``openai`` — the regular OpenAI API. Use ``OPENAI_API_KEY``.

Because both providers speak the same wire format, the rest of the codebase
just constructs an :class:`openai.OpenAI` client with a configurable
``base_url`` and ``api_key`` and stays provider-agnostic.

Why a dedicated config module?

* Single place to tweak behavior without grep-hunting through the codebase.
* Makes it obvious which values are user-tunable.
* Lets tests override settings cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from the project root if it exists. Safe to call multiple times.
load_dotenv()


# Gemini's OpenAI-compatible base URL. Documented at
# https://ai.google.dev/gemini-api/docs/openai
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Provider-specific defaults. Both providers expose chat + embedding models
# behind the OpenAI wire format, so the rest of the pipeline doesn't care
# which one is selected.
_PROVIDER_DEFAULTS = {
    "gemini": {
        "chat_model": "gemini-2.5-flash",
        # gemini-embedding-001 is the current stable embedding model on the
        # OpenAI-compatible endpoint; the older `text-embedding-004` has been
        # retired.
        "embed_model": "gemini-embedding-001",
        "base_url": GEMINI_BASE_URL,
    },
    "openai": {
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        # ``None`` means: let the OpenAI SDK use its built-in default.
        "base_url": None,
    },
}


def _get_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on missing/invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    """Read a string env var, treating empty/whitespace as missing."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of runtime settings.

    Frozen so we can pass it around without anyone mutating shared state.
    """

    # --- Provider selection ---
    provider: str          # "gemini" or "openai"
    api_key: str | None    # Whichever provider's key is in use
    base_url: str | None   # None for vanilla OpenAI; URL for Gemini

    # --- Models ---
    chat_model: str
    embed_model: str

    # --- Chunking ---
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    # --- Retrieval ---
    top_k: int
    retrieval_mode: str  # "vector" (default) or "hybrid"

    # --- Storage ---
    chroma_dir: Path
    chroma_collection: str


def load_settings() -> Settings:
    """Build a Settings object from current env vars + provider defaults."""
    provider = _get_str("PROVIDER", "gemini").lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise RuntimeError(
            f"Unknown PROVIDER={provider!r}. "
            f"Set PROVIDER to one of: {sorted(_PROVIDER_DEFAULTS)}."
        )
    defaults = _PROVIDER_DEFAULTS[provider]

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    # ``CHAT_MODEL`` / ``EMBED_MODEL`` (provider-agnostic) take precedence;
    # otherwise we fall back to the per-provider defaults above.
    chat_model = _get_str("CHAT_MODEL", defaults["chat_model"])
    embed_model = _get_str("EMBED_MODEL", defaults["embed_model"])

    retrieval_mode = _get_str("RETRIEVAL_MODE", "vector").lower()
    if retrieval_mode not in {"vector", "hybrid"}:
        raise RuntimeError(
            f"Unknown RETRIEVAL_MODE={retrieval_mode!r}. "
            "Set RETRIEVAL_MODE to 'vector' or 'hybrid'."
        )

    return Settings(
        provider=provider,
        api_key=api_key,
        base_url=defaults["base_url"],
        chat_model=chat_model,
        embed_model=embed_model,
        chunk_size_tokens=_get_int("CHUNK_SIZE_TOKENS", 500),
        chunk_overlap_tokens=_get_int("CHUNK_OVERLAP_TOKENS", 50),
        top_k=_get_int("TOP_K", 4),
        retrieval_mode=retrieval_mode,
        chroma_dir=Path(os.getenv("CHROMA_DIR", "./chroma_db")).resolve(),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "rag_demo"),
    )


def require_api_key(settings: Settings) -> str:
    """Return the configured provider's API key, or raise a friendly error.

    Centralizing this means every entry point gives the same helpful message
    instead of a cryptic 401 from deep inside the OpenAI SDK.
    """
    if settings.api_key:
        return settings.api_key

    if settings.provider == "gemini":
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  1. Get a FREE key at https://aistudio.google.com/apikey\n"
            "  2. Copy the template:  cp .env.example .env\n"
            "  3. Edit .env and set GEMINI_API_KEY=...\n"
            "  4. Re-run the command."
        )
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "  1. Get a key at https://platform.openai.com/api-keys\n"
        "  2. Copy the template:  cp .env.example .env\n"
        "  3. Edit .env and set PROVIDER=openai and OPENAI_API_KEY=sk-...\n"
        "  4. Re-run the command."
    )


# Back-compat alias so older imports keep working if anything still uses it.
require_openai_key = require_api_key
