"""Optional Streamlit chat UI.

This file is intentionally separate from the core pipeline: nothing in
`src/` imports Streamlit, so the CLI keeps working even if Streamlit
isn't installed.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.config import load_settings
from src.generate import generate_answer
from src.ingest import ingest_pdf
from src.retrieve import retrieve
from src.vectorstore import get_vector_store


st.set_page_config(page_title="RAG demo", page_icon="📚")
st.title("📚 RAG demo")
st.caption(
    "Ask questions about a PDF. Only the most relevant chunks are sent to "
    "the LLM — not the whole document."
)

settings = load_settings()

# --- Sidebar: ingest a new PDF ---
with st.sidebar:
    st.header("Ingest a PDF")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest"):
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        try:
            # Rename so the chunk metadata uses the original filename, not
            # the random tempfile name.
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

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if question := st.chat_input("Ask a question about the indexed PDF..."):
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
