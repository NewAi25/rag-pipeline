# Evaluation results

**Document:** `data/nist_ai_rmf_1.0.pdf`
**Source:** [https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
**License:** U.S. Government work, public domain (17 U.S.C. § 105)
**Eval set:** 12 labeled questions in [`eval/dataset.json`](dataset.json), written by reading the document directly.
**Top-k:** 4 (the default).
**Date run:** 2026-05-30.

---

## Aggregate metrics (vector vs. hybrid)

| Metric | vector | hybrid | Δ |
| --- | --- | --- | --- |
| Hit@k                       | 91.7%       | **100.0%**  | +8.3 pp |
| MRR                         | 0.854       | **0.944**   | +0.090 |
| Context precision           | 39.6%       | **45.8%**   | +6.2 pp |
| Context recall              | 91.7%       | **100.0%**  | +8.3 pp |
| Faithfulness (LLM judge)    | not run     | not run     | — |
| Answer relevance (LLM judge) | not run     | not run     | — |

> **Why no LLM-judge numbers?** Running the full both-modes-with-judge eval
> consumes 24 chat-API calls for generation plus 24 for judging — well beyond
> Gemini's free-tier daily ceiling on `gemini-2.5-flash` (20 requests/day at
> the time of writing). The deterministic metrics above use only embedding
> calls (different, much-higher quota) and are the load-bearing measure for
> this *retrieval-quality* comparison. Re-run with judge metrics later via:
>
> ```bash
> docker compose run --rm rag python eval/run_eval.py --modes vector hybrid
> ```

---

## Per-question Hit@k

| Question ID | Question | vector | hybrid |
| --- | --- | --- | --- |
| q01-functions                | What are the four functions of the AI RMF Core?                                  | hit  | hit  |
| q02-publication-date         | When was the AI RMF 1.0 published?                                               | hit  | hit  |
| q03-playbook-email           | Where can comments on the AI RMF Playbook be sent?                               | hit  | hit  |
| q04-trustworthy-characteristics | List the characteristics of trustworthy AI systems described in the framework. | hit  | hit  |
| q05-voluntary                | Is the AI RMF mandatory or voluntary?                                            | hit  | hit  |
| q06-govern-cross-cutting     | How does the GOVERN function relate to the other AI RMF functions?               | hit  | hit  |
| q07-socio-technical          | Why are AI systems described as socio-technical?                                 | hit  | hit  |
| q08-review-year              | By what year does NIST expect to formally review the AI RMF?                     | hit  | hit  |
| q09-ai-actor-definition      | Who is defined as an AI actor in the AI RMF?                                     | hit  | hit  |
| q10-versioning               | How does the AI RMF version-number itself?                                       | hit  | hit  |
| q11-document-id              | What is the official NIST publication number of the AI Risk Management Framework? | hit  | hit  |
| q12-bias-section             | What does section 3.7 of the framework cover?                                    | **miss** | **hit** |

---

## Interpretation

Hybrid retrieval wins on this corpus, but not by a huge margin — because the
dataset is small (12 questions on a single 48-page document) and most
questions are semantic enough that the embedding model finds the right chunk
on its own. The one flip (`q12-bias-section`) is also the most informative:

- The question asks about **"section 3.7"** and the ground-truth snippet is the
  literal heading **"Harmful Bias Managed"**.
- Pure vector retrieval doesn't surface that exact heading in its top-4 — it
  pulls semantically-related discussions of fairness from elsewhere in the
  document instead.
- BM25 matches the rare token `3.7` (and the unusual phrase "Harmful Bias
  Managed") cleanly. Once RRF blends BM25's high rank for that chunk with
  vector's softer support, the correct chunk lands in the fused top-4.

This is the classic case for hybrid retrieval: **rare keywords, identifiers,
and exact phrasings get washed out by dense embeddings**. Adding a cheap
keyword channel recovers them at near-zero cost.

The Context precision lift (+6.2 pp) is real but modest — at top-4, two of
the four chunks are typically the right neighbourhood and the others are
adjacent context. Tighter eval snippets or a larger eval set would expose
more of hybrid's win.

### Where hybrid did NOT help

On the 11 questions both modes hit, hybrid sometimes returned the relevant
chunk at a different rank than vector did, but never *lost* a hit. The
MRR delta (+0.09) reflects a few questions where the relevant chunk moved up
from rank-2 to rank-1 under hybrid — incremental but real.

---

## How to reproduce

```bash
# Get the dataset (one-time):
docker compose run --rm rag python scripts/get_dataset.py
docker compose run --rm rag python -m src.cli clear --yes
docker compose run --rm rag python -m src.cli ingest data/nist_ai_rmf_1.0.pdf

# Run the full eval (deterministic metrics only — no chat quota used):
docker compose run --rm -v "$(pwd)/eval:/app/eval" rag \
    python eval/run_eval.py --modes vector hybrid --no-judge

# With LLM-judge metrics too (needs >40 chat calls — paid Gemini or wait for daily quota reset):
docker compose run --rm -v "$(pwd)/eval:/app/eval" rag \
    python eval/run_eval.py --modes vector hybrid
```

Metric definitions and implementation are at the top of
[`eval/run_eval.py`](run_eval.py).
