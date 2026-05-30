"""Generation: turn the retrieved chunks + question into a grounded answer.

The system prompt is the heart of any RAG app. We do two important things:

  1. We instruct the model to answer ONLY from the provided context.
     This is what keeps the model from hallucinating things that aren't
     in the PDF.
  2. We tell it to say "I don't know" when the context doesn't cover the
     question, rather than guessing. This is what makes the system
     trustworthy enough to actually use.

The same code path works for OpenAI and Gemini because we point the
:class:`openai.OpenAI` client at a configurable ``base_url``; Gemini
speaks the same wire format as OpenAI at
``https://generativelanguage.googleapis.com/v1beta/openai/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from ._retry import with_retries
from .config import Settings, require_api_key
from .vectorstore import Retrieved


SYSTEM_PROMPT = """You are a careful, concise assistant answering questions \
about a specific document.

Rules you MUST follow:
1. Use ONLY the information in the "Context" section to answer. Do not \
use outside knowledge, even if you think you know the answer.
2. If the context does not contain the answer, reply exactly: \
"I don't know based on the provided document."
3. Quote or paraphrase the relevant passage. Cite the chunk you used \
with its [source#chunk] tag (e.g. [handbook.pdf#chunk-7]).
4. Keep the answer short — a sentence or two unless the user asks for \
more detail."""


@dataclass(frozen=True)
class Answer:
    """Final answer plus the chunks that produced it (for transparency)."""

    text: str
    sources: list[Retrieved]


def _format_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks into a labelled block the model can cite."""
    parts = []
    for c in chunks:
        tag = f"[{c.source}#chunk-{c.chunk_index}]"
        parts.append(f"{tag}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[Retrieved],
    *,
    settings: Settings,
) -> Answer:
    """Build the RAG prompt and call the chat model."""
    api_key = require_api_key(settings)
    # ``base_url=None`` means "use the SDK default" (OpenAI); any other value
    # (e.g. Gemini's OpenAI-compatible URL) overrides it.
    client = OpenAI(api_key=api_key, base_url=settings.base_url)

    if not chunks:
        # No retrieval hits at all — short-circuit instead of paying for
        # a call that's guaranteed to return "I don't know".
        return Answer(
            text="I don't know based on the provided document.",
            sources=[],
        )

    user_prompt = (
        f"Context:\n{_format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    def _call():
        return client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # Low temperature -> the model sticks closer to the context and
            # is less likely to embellish.
            temperature=0.1,
        )

    # Free tiers (especially Gemini's) throttle aggressively — retry on 429.
    completion = with_retries(_call)
    text = (completion.choices[0].message.content or "").strip()
    return Answer(text=text, sources=chunks)
