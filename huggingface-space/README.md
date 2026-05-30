---
title: RAG Pipeline Demo
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Ask grounded, cited questions about NIST's AI Risk Management Framework — powered by hybrid (BM25 + vector) retrieval.
---

# RAG Pipeline — live demo

Read-only public demo of the open-source RAG pipeline at
[github.com/NewAi25/rag-pipeline](https://github.com/NewAi25/rag-pipeline).

Ask grounded, cited questions about the **NIST AI Risk Management Framework
1.0** (a public-domain US Gov publication, ~48 pages). Answers come with
the exact source chunks that produced them.

## Limits

- Runs on a **shared free-tier Gemini API key** — per-minute rate limits apply.
- **20 questions per session**; refresh the page for a new session.
- PDF upload is disabled here. Clone the repo to run it on your own documents.

## How it works

Four steps: Chunking → Embedding → Vector DB (Chroma) → Retrieval (vector
or hybrid BM25 + vector via Reciprocal Rank Fusion). See the
[GitHub README](https://github.com/NewAi25/rag-pipeline) for the full
architecture diagram, evaluation results, and configuration options.
