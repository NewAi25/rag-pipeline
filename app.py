"""Streamlit chat UI.

Two operating modes, selected by the ``DEMO_MODE`` env var:

* **Default (full):** sidebar lets you upload+ingest your own PDF.
  This is what runs locally and inside ``docker compose up ui``.

* **Demo (``DEMO_MODE=1``):** read-only public demo. No upload. The
  configured dataset is auto-ingested at startup if the vector store
  is empty, a per-session question cap protects the shared free-tier
  Gemini key, and a banner explains the limits to visitors. This is
  what runs on Hugging Face Spaces / Streamlit Cloud.

This file deliberately stays separate from the core pipeline: nothing
in ``src/`` imports Streamlit, so the CLI keeps working even if
Streamlit isn't installed.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.config import load_settings
from src.generate import generate_answer
from src.ingest import ingest_pdf
from src.retrieve import retrieve
from src.vectorstore import get_vector_store


# ---------------------------------------------------------------------------
# Demo-mode configuration
# ---------------------------------------------------------------------------

DEMO_MODE = os.getenv("DEMO_MODE", "0").lower() in {"1", "true", "yes"}
DEMO_PDF = Path(os.getenv("DEMO_PDF", "data/nist_ai_rmf_1.0.pdf"))
DEMO_QUESTION_CAP = int(os.getenv("DEMO_QUESTION_CAP", "20"))
DEMO_TITLE = os.getenv(
    "DEMO_TITLE",
    "📚 RAG Pipeline — live demo (NIST AI Risk Management Framework)",
)


st.set_page_config(
    page_title="RAG Pipeline Demo" if DEMO_MODE else "RAG demo",
    page_icon="📚",
)


@st.cache_resource(show_spinner="Preparing the index (one-time, ~30 s)...")
def _bootstrap_demo() -> int:
    """In demo mode, ingest the bundled PDF once if the collection is empty.

    Cached at the resource level so concurrent visitors share one ingest.
    """
    settings = load_settings()
    store = get_vector_store(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    if store.count() > 0:
        return store.count()
    if not DEMO_PDF.exists():
        raise FileNotFoundError(
            f"Demo PDF not found at {DEMO_PDF}. "
            "Run `python scripts/get_dataset.py` first, or ship the PDF in the image."
        )
    n = ingest_pdf(DEMO_PDF, settings=settings)
    return n


settings = load_settings()

if DEMO_MODE:
    st.title(DEMO_TITLE)
    n_chunks = _bootstrap_demo()
    st.info(
        "🔓 **Read-only public demo.** This Space runs on a shared free-tier "
        "Gemini API key with strict per-minute rate limits. To prevent abuse, "
        f"you can ask up to **{DEMO_QUESTION_CAP} questions per session**, and "
        "PDF upload is disabled. Want to point this at your own documents? "
        "Clone the repo — the README has setup steps.",
        icon="ℹ️",
    )
    st.caption(
        f"Dataset: **{DEMO_PDF.name}** ({n_chunks} chunks indexed) · "
        f"Retrieval mode: **{settings.retrieval_mode}** · "
        f"Model: **{settings.chat_model}**"
    )
else:
    st.title("📚 RAG demo")
    st.caption(
        "Ask questions about a PDF. Only the most relevant chunks are sent to "
        "the LLM — not the whole document."
    )

    # --- Sidebar: ingest a new PDF (only in non-demo mode) ---
    with st.sidebar:
        st.header("Ingest a PDF")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded is not None and st.button("Ingest"):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            try:
                renamed = tmp_path.with_name(uploaded.name)
                tmp_path.rename(renamed)
                with st.spinner("Chunking + embedding..."):
                    n = ingest_pdf(renamed, settings=settings)
                st.success(f"Indexed {n} chunks from {uploaded.name}")
            except Exception as e:  # noqa: BLE001 — surface anything to the user
                st.error(str(e))

        store = get_vector_store(
            persist_dir=settings.chroma_dir,
            collection_name=settings.chroma_collection,
        )
        st.metric("Chunks in store", store.count())


# --- Main: chat ---
if "history" not in st.session_state:
    st.session_state.history = []
if "n_questions" not in st.session_state:
    st.session_state.n_questions = 0

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Show remaining quota in demo mode (above the input).
if DEMO_MODE:
    remaining = max(0, DEMO_QUESTION_CAP - st.session_state.n_questions)
    if remaining == 0:
        st.warning(
            f"Per-session question cap of {DEMO_QUESTION_CAP} reached. "
            "Refresh the page to start a new session.",
            icon="⛔",
        )
    else:
        st.caption(f"Questions remaining this session: **{remaining}**")

if question := st.chat_input("Ask a question about the indexed PDF..."):
    if DEMO_MODE and st.session_state.n_questions >= DEMO_QUESTION_CAP:
        st.warning(
            f"Per-session question cap of {DEMO_QUESTION_CAP} reached. "
            "Refresh to start a new session.",
            icon="⛔",
        )
        st.stop()

    st.session_state.n_questions += 1
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            chunks = retrieve(question, settings=settings)
            answer = generate_answer(question, chunks, settings=settings)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()

        st.markdown(answer.text)
        if answer.sources:
            with st.expander(f"Sources ({len(answer.sources)})"):
                for c in answer.sources:
                    st.markdown(
                        f"**[{c.source}#chunk-{c.chunk_index}]** "
                        f"_(distance={c.distance:.3f})_"
                    )
                    st.write(c.text)
        st.session_state.history.append(("assistant", answer.text))
