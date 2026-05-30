# RAG Pipeline

> **Ask grounded, cited questions about your own PDFs — without paying to re-send the whole document on every query.**

A small, beginner-friendly **Retrieval-Augmented Generation (RAG)** pipeline you can clone, point at any PDF (a 200-page company handbook, a stack of research papers, a contract, your meeting notes), and start asking questions in under five minutes. Runs locally in Docker. Uses Google Gemini's **free** API by default — switch to OpenAI or fully-local Ollama with a one-line config change.

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-black.svg)](https://peps.python.org/pep-0008/)

---

## Table of contents

1. [The problem RAG solves](#the-problem-rag-solves)
2. [How it works](#how-it-works)
3. [Use cases](#use-cases)
4. [Prerequisites](#prerequisites)
5. [Get a free Gemini key](#get-a-free-gemini-key)
6. [Setup](#setup)
7. [Quickstart (with the bundled sample)](#quickstart-with-the-bundled-sample)
8. [Use your own document](#use-your-own-document)
9. [Optional: Streamlit chat UI](#optional-streamlit-chat-ui)
10. [Configuration](#configuration)
11. [Swap the components](#swap-the-components)
12. [Troubleshooting](#troubleshooting)
13. [Project structure](#project-structure)
14. [Future scope](#future-scope)
15. [Contributing](#contributing)
16. [License](#license)

---

## The problem RAG solves

Imagine a 400-page company handbook. You want an AI that can answer any question about it. The obvious approach is to paste the entire handbook into the model's prompt on every query. That works… for a while.

Then three problems hit you at once:

- **Cost.** Every question re-sends all 200 pages through the LLM. Your bill scales with `pages × queries`, not with the actual signal you need.
- **Speed.** Long prompts mean slow responses. Users notice.
- **Accuracy (the "lost-in-the-middle" effect).** When the real answer is buried halfway down a huge wall of text, LLMs tend to skim past it. Multiple studies — most famously [Liu et al., 2023](https://arxiv.org/abs/2307.03172) — show that recall is best at the *start* and *end* of long prompts, and noticeably worse in the middle.

**RAG flips the approach: don't make the model read everything — help it find only what it needs.**

Instead of stuffing the whole document into the prompt, we **embed** the document once into a vector index, then at query time **retrieve** only the handful of paragraphs that actually match. The LLM reads three paragraphs instead of 200 pages. Cheaper. Faster. More accurate. And grounded in your real data — with citations.

---

## How it works

Four classic RAG steps, each living in its own clearly-named file:

| Step | File | What it does |
|------|------|--------------|
| 1. Chunking  | [src/chunking.py](src/chunking.py)    | Splits the document into overlapping paragraph-sized pieces, on token boundaries so chunks fit cleanly under model limits. |
| 2. Embedding | [src/embeddings.py](src/embeddings.py) | Turns each chunk (and later, each question) into a vector — a numerical fingerprint of its *meaning*. |
| 3. Vector DB | [src/vectorstore.py](src/vectorstore.py) | Stores vectors on disk and runs sub-millisecond similarity search. Defaults to Chroma; Pinecone / pgvector stubs included. |
| 4. Retrieval | [src/retrieve.py](src/retrieve.py)    | Embeds the question and pulls the top-k most-similar chunks. |
| Generation   | [src/generate.py](src/generate.py)    | Builds a "use only this context, cite your sources, say *I don't know* otherwise" prompt and calls the LLM. |
| Orchestration| [src/ingest.py](src/ingest.py), [src/cli.py](src/cli.py) | Wires steps 1–3 together for ingest; exposes `ingest`, `ask`, and `clear` on the CLI. |

### Architecture

```
                       INGEST (run once per document)
   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
   │   PDF    │──▶│ Chunking │──▶│ Embedding │──▶│  Vector DB   │
   └──────────┘   └──────────┘   └───────────┘   └──────────────┘
                                                         ▲
                       ASK (run per question)            │ similarity
   ┌──────────┐   ┌───────────┐   ┌───────────┐          │ search
   │ Question │──▶│ Embedding │──▶│ Retrieval │──────────┘
   └──────────┘   └───────────┘   └─────┬─────┘
                                        │ top-k chunks
                                        ▼
                                 ┌──────────────┐   ┌──────────────────┐
                                 │   LLM (RAG)  │──▶│  Grounded answer │
                                 │              │   │  + citations     │
                                 └──────────────┘   └──────────────────┘
```

---

## Use cases

A RAG pipeline like this is the foundation under most "chat with your X" products. Drop your own corpus into `data/` and you have:

- **Internal company handbook / HR Q&A** — answer "How many vacation days do I get?" with the exact policy paragraph cited.
- **Customer support over product docs** — let users ask natural-language questions about your manual or knowledge base, with answers traceable to the source page.
- **Legal & contract search** — surface the exact clause that addresses indemnity, termination, or renewal terms, instead of grepping a 90-page PDF.
- **Research-paper Q&A** — ingest a folder of papers, ask "Which of these uses retrieval-augmented training?" and get a cited answer.
- **Study notes & textbooks** — turn your lecture PDFs into a tutor that quotes the relevant page back at you.
- **Technical documentation assistants** — RFCs, internal architecture docs, runbooks — ask "What's the rollback procedure?" and get the runbook step, cited.
- **Policy & compliance lookup** — quickly find which paragraph of which policy governs a given action, with the source visible.

---

## Prerequisites

You need **one** of the following:

- **Docker** (recommended) — works identically on Windows / macOS / Linux, no Python install required. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux).
- **Python 3.12** — if you'd rather use a local virtualenv (3.13/3.14 may not yet have wheels for every dependency; 3.12 is the supported floor).

Plus an API key for whichever provider you pick (default: free Gemini).

---

## Get a free Gemini key

1. Open [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Sign in with any Google account → **Create API key**.
3. Copy the key. It looks like `AQ...` / `AI...`.

**Privacy caveat.** Anything you send to a hosted LLM (Gemini, OpenAI, Anthropic, etc.) leaves your machine. Do not feed Personally Identifiable Information, secrets, or regulated data into the free tier. For genuinely private corpora, see [Swap the components → Ollama](#swap-the-components) below to run fully locally.

> Prefer OpenAI's paid API? Use `PROVIDER=openai` + `OPENAI_API_KEY=sk-...` in `.env`. The pipeline is provider-agnostic — Gemini's OpenAI-compatible endpoint means we use the same SDK for both.

---

## Setup

```bash
# 1. Clone
git clone <this-repo-url> rag-pipeline
cd rag-pipeline

# 2. Configure
cp .env.example .env
# then open .env and set GEMINI_API_KEY=...

# 3. Build the Docker image (one-time, ~2 min on first run)
docker compose build
```

That's it. You now have a pinned Python 3.12 environment with all dependencies installed — the same on every machine.

---

## Quickstart (with the bundled sample)

A tiny `data/sample.pdf` is included so you can confirm the pipeline works end-to-end before you wire in your own documents.

**Ingest it:**

```bash
docker compose run --rm rag python -m src.cli ingest data/sample.pdf
```

You'll see something like:

```
Ingesting data/sample.pdf ...
  + 3 chunks
Done. Wrote 3 chunks to /app/chroma_db (collection: rag_demo).
```

**Ask a question:**

```bash
docker compose run --rm rag python -m src.cli ask "What is the refund policy?"
```

Example output:

```
Answer:
The refund policy allows full refunds within 30 days of purchase
for items in original condition. [sample.pdf#chunk-2]

Sources used:
  - [sample.pdf#chunk-2] (distance=0.214)  Refund policy: customers may return any item in...
  - [sample.pdf#chunk-1] (distance=0.481)  Welcome to ACME Co. This handbook covers our...
```

The answer is grounded in the document, the chunks that produced it are shown, and the prompt told the model to say *"I don't know based on the provided document"* when the answer isn't there — so you can trust the output.

### Shorter, with `make`

```bash
make build
make ingest PDF=data/sample.pdf
make ask Q="What is the refund policy?"
make ui      # optional Streamlit UI
make clear   # wipe the collection
make clean   # delete the whole chroma_db/ folder
```

On Windows without `make` installed, use the equivalent `docker compose ...` commands above.

---

## Use your own document

The bundled sample is just there to prove the install works. The real workflow is to point this at your own PDFs.

**Step 1 — Put the PDF in `data/`.**

```bash
cp /path/to/your-document.pdf data/
```

`data/` is mounted into the container, so anything you drop there is immediately visible inside.

**Step 2 — (Optional) clear the previous index.**

If you've already ingested another document and want to start fresh:

```bash
docker compose run --rm rag python -m src.cli clear --yes
# or, to also delete the folder on disk:
make clean
```

You don't *have* to clear — re-ingesting is idempotent (chunks are upserted by ID), and you can keep multiple documents in the same collection if you want cross-document Q&A. Clear when you specifically want one corpus at a time.

**Step 3 — Ingest.** You can pass a single PDF or a folder of PDFs:

```bash
# Single file:
docker compose run --rm rag python -m src.cli ingest data/your-document.pdf

# Whole folder (recursively picks up every *.pdf):
docker compose run --rm rag python -m src.cli ingest data/
```

**Step 4 — Ask.**

```bash
docker compose run --rm rag python -m src.cli ask "Your question here"
```

### Tips for better answers

- **Be specific.** "What is the policy on remote work?" beats "remote?". Vector similarity rewards meaningful phrasing.
- **Text PDFs, not scanned images.** This pipeline uses `pypdf` for text extraction. A scanned PDF (image-of-text) yields no text and ingest will error out — run it through OCR (e.g. `ocrmypdf in.pdf out.pdf`) first.
- **Tune `TOP_K`.** The default is `4` chunks per query. If answers feel incomplete, try `TOP_K=8` in `.env` for more context. Too high and you'll burn tokens; too low and you may miss the relevant passage.
- **Re-ingest after changing the embedding model.** Different embedding models produce different-dimensional vectors. Mixing them in one collection breaks similarity search. Run `clear` first if you switch.

---

## Optional: Streamlit chat UI

If you prefer a browser instead of the CLI:

```bash
docker compose up ui
```

Open [http://localhost:8501](http://localhost:8501). Upload a PDF in the sidebar, ingest it with one click, and chat with it in the main pane. Source chunks are shown in an expandable panel beneath each answer.

---

## Configuration

All settings live in `.env` (copied from `.env.example`).

| Variable               | Default                | Description                                                                                            |
|------------------------|------------------------|--------------------------------------------------------------------------------------------------------|
| `PROVIDER`             | `gemini`               | Which LLM provider to use: `gemini` (free) or `openai` (paid).                                          |
| `GEMINI_API_KEY`       | *(required if gemini)* | Free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).                        |
| `OPENAI_API_KEY`       | *(required if openai)* | Key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).                          |
| `CHAT_MODEL`           | *(provider default)*   | Override the chat model. Gemini default: `gemini-2.5-flash`. OpenAI default: `gpt-4o-mini`.            |
| `EMBED_MODEL`          | *(provider default)*   | Override the embedding model. Gemini default: **`gemini-embedding-001`**. OpenAI default: `text-embedding-3-small`. |
| `CHUNK_SIZE_TOKENS`    | `500`                  | Target chunk length in tokens. ~500 ≈ a paragraph.                                                     |
| `CHUNK_OVERLAP_TOKENS` | `50`                   | Tokens shared between consecutive chunks. Prevents ideas from getting cut in half at the boundary.     |
| `TOP_K`                | `4`                    | How many chunks to retrieve per question.                                                              |
| `CHROMA_DIR`           | `./chroma_db`          | Where the local vector store lives on disk (auto-created on first ingest).                             |
| `CHROMA_COLLECTION`    | `rag_demo`             | Collection name inside Chroma.                                                                         |
| `ANONYMIZED_TELEMETRY` | `False`                | Silence Chroma's anonymous usage pings (also avoids noisy posthog log lines).                          |

> **Embedding-model warning.** The Gemini default **must be `gemini-embedding-001`**. The older `text-embedding-004` has been retired and returns 404. **If you change `EMBED_MODEL`, you must `clear` and re-ingest** — vectors from different models live in different geometric spaces and aren't comparable.

---

## Swap the components

The provider and store are isolated behind tiny wrappers, so swapping is one file each:

### Different LLM / embedding provider

- **OpenAI (paid, very accurate).** Set `PROVIDER=openai` + `OPENAI_API_KEY=sk-...` in `.env`. Done — same code path, different `base_url`.
- **Fully local with Ollama (private, no rate limits, no bill).** Run `ollama serve`, point the OpenAI client at `http://localhost:11434/v1`. Use `nomic-embed-text` for embeddings and `llama3.1` (or similar) for chat. Edit `src/embeddings.py` and `src/generate.py` to use the Ollama `base_url`. Nothing leaves your machine.
- **Anything else.** Implement the `EmbeddingProvider` protocol in [src/embeddings.py](src/embeddings.py) — one method, `embed(texts) -> list[list[float]]`.

### Different vector database

[src/vectorstore.py](src/vectorstore.py) defines a small `VectorStore` protocol (`add`, `query`, `count`, `reset`) and ships with commented stubs for:

- **Pinecone** — managed, serverless, scales to billions of vectors.
- **Qdrant** — open source, Rust-based, very fast; runs in Docker or as a managed service.
- **pgvector** — Postgres extension; great when your team already runs Postgres and wants vectors alongside relational data.
- **Weaviate / Milvus / LanceDB** — drop-in if you prefer one of these.

Implement the same four methods in a new class, return it from `get_vector_store()`, and the rest of the pipeline doesn't change.

---

## Troubleshooting

| Symptom                                                                                          | Cause                                                                                                                 | Fix                                                                                                                              |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `404 ... text-embedding-004 is not found`                                                        | The older Gemini embedding model has been retired.                                                                    | Make sure `EMBED_MODEL=gemini-embedding-001` (or leave it blank to take the default), then `clear` + re-ingest.                  |
| `GEMINI_API_KEY is not set.`                                                                     | No `.env` file, or the key line is empty.                                                                             | `cp .env.example .env`, edit, paste the key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).               |
| `Cannot connect to the Docker daemon` / `error during connect`                                   | Docker Desktop / Docker Engine isn't running.                                                                          | Start Docker Desktop (or `sudo systemctl start docker` on Linux), then retry.                                                    |
| Lots of `Failed to send telemetry event` log lines                                               | Chroma's bundled posthog client and `chromadb` version drift.                                                          | Already silenced: `ANONYMIZED_TELEMETRY=False` is in `.env`, plus the posthog logger is muted in code. Make sure your `.env` has the line. |
| `Number of requested results 4 is greater than number of elements in index N`                    | You're querying before enough chunks are indexed.                                                                      | Either ingest more, or lower `TOP_K` in `.env`. The notice is harmless — Chroma still returns what it has.                       |
| `429 RESOURCE_EXHAUSTED`                                                                          | Gemini's free-tier per-minute or per-day cap.                                                                          | The pipeline auto-retries with exponential backoff (1s → 2s → 4s → 8s). If it persists, wait or switch to `PROVIDER=openai`.     |
| `No extractable text found in <file>.pdf`                                                        | The PDF is a scanned image, not real text.                                                                             | OCR it first: `ocrmypdf in.pdf out.pdf` (then ingest `out.pdf`).                                                                 |
| Answer is `I don't know based on the provided document.` for a question you *know* is in the PDF | Either the chunk wasn't retrieved (`TOP_K` too low / question phrased differently from the doc), or the chunk has poor semantic match. | Try `TOP_K=8`, rephrase the question to use words closer to the document, or shrink `CHUNK_SIZE_TOKENS` for finer-grained matches. |

---

## Project structure

```
rag-pipeline/
├── Dockerfile              # Pinned Python 3.12 environment
├── docker-compose.yml      # Two services: `rag` (CLI) and `ui` (Streamlit)
├── Makefile                # Convenience wrappers around docker compose
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Copy to .env and fill in your key
├── .dockerignore           # Keep secrets and the local DB out of the image
├── .gitignore              # Keep secrets and caches out of git
├── LICENSE                 # MIT
├── README.md               # You are here
├── data/
│   └── sample.pdf          # A tiny bundled doc so the demo works out of the box
├── scripts/
│   └── make_sample_pdf.py  # Regenerates data/sample.pdf
├── src/
│   ├── __init__.py
│   ├── config.py           # Loads settings from env vars + provider defaults
│   ├── chunking.py         # STEP 1 — token-aware overlapping chunker
│   ├── embeddings.py       # STEP 2 — OpenAI-compatible embedding client
│   ├── vectorstore.py      # STEP 3 — Chroma wrapper + Pinecone/pgvector stubs
│   ├── retrieve.py         # STEP 4 — similarity search
│   ├── generate.py         # Grounded-answer prompt + LLM call
│   ├── ingest.py           # Ties steps 1–3 together for one PDF
│   ├── _retry.py           # Exponential-backoff helper for 429s
│   └── cli.py              # `ingest` / `ask` / `clear` Typer commands
├── app.py                  # Optional Streamlit chat UI
└── tests/
    └── test_chunking.py    # Unit tests for the chunker
```

---

## Future scope

This repo is intentionally small — a teaching pipeline you can read in an afternoon. If you want to grow it into something production-grade, here are the natural next steps, grouped by what they improve.

### Document support
- **More file types** — DOCX, HTML, Markdown, Confluence pages. Drop in `python-docx`, `markdown-it-py`, `BeautifulSoup`, etc., behind a `DocumentLoader` interface.
- **OCR for scanned PDFs** — wire in Tesseract via `ocrmypdf` or `unstructured.io` so image-of-text PDFs aren't dead-ends.
- **URL ingest** — accept `https://...` and crawl the page (plus optional same-domain links).
- **Multi-document Q&A** — already works (just ingest more), but the citation format could be extended to make cross-document answers more legible.

### Retrieval quality
- **Semantic chunking** — chunk on paragraph / heading boundaries instead of fixed token windows.
- **Hybrid retrieval (BM25 + vectors)** — keyword and semantic search combined catches both exact terms and paraphrases. Easy with `rank-bm25` or via a vector DB that supports it natively.
- **Reranking** — pass the top-N candidates through a cross-encoder (e.g. `bge-reranker`) for a much sharper top-K.
- **Query rewriting** — ask the LLM to expand vague queries before retrieval; or run HyDE-style "imagine the answer, embed that, then search".

### Answer quality
- **Page-level / span-level citations** — track PDF page numbers and highlight the exact span the answer came from.
- **Streaming responses** — stream tokens as they're generated for snappier UX in the Streamlit UI.
- **Conversational memory** — keep the last N turns in the prompt so follow-up questions ("and what about for new hires?") work.
- **Stronger "I don't know" handling** — use [self-RAG](https://arxiv.org/abs/2310.11511)-style critique tokens to refuse when retrieval confidence is low.

### Production readiness
- **Managed vector DBs** — swap Chroma for Pinecone, Qdrant Cloud, Weaviate, or pgvector for scale and durability.
- **Embedding cache** — never re-embed an unchanged chunk; key by content hash.
- **Evaluation harness** — adopt [RAGAS](https://github.com/explodinggradients/ragas) or [TruLens](https://github.com/truera/trulens) to score faithfulness, context relevance, and answer correctness on a fixed eval set.
- **Observability** — log every retrieval (query, top-k IDs, distances, latency) so you can debug bad answers later. OpenTelemetry + a tracing UI is a clean fit.
- **Auth + browser upload** — wrap the Streamlit UI in SSO and let non-technical users drop PDFs in via the browser.
- **Agentic RAG** — let the model decide when to retrieve, run multiple retrieval steps, or call tools (e.g. SQL) when the answer isn't in the docs.

---

## Contributing

Issues and pull requests welcome. For non-trivial changes, please open an issue first so we can discuss the approach before you spend time coding.

To run the test suite locally:

```bash
docker compose run --rm rag python -m pytest -q
# or, without Docker:
pip install -r requirements.txt && pytest -q
```

---

## License

MIT — see [LICENSE](LICENSE). Use it for anything, including commercial work; attribution appreciated but not required.
