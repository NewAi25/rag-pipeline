"""Run the evaluation harness over the labeled dataset.

What this computes
------------------
Deterministic (no LLM, fast, free):

* Hit@k         — 1 if any retrieved chunk contains any labeled snippet.
* MRR           — 1 / (rank of first chunk containing any labeled snippet).
* Context precision — fraction of retrieved chunks containing >=1 snippet.
* Context recall    — fraction of labeled snippets that appear in some retrieved chunk.

LLM-judged (Gemini via the existing OpenAI-compatible client):

* Faithfulness    — 0-5, is the answer supported by the retrieved context?
* Answer relevance — 0-5, does the answer actually address the question?

Both judge scores come from a single chat call per question (a JSON
response with two fields), which halves chat-API traffic vs. one call
per metric — useful on Gemini's free tier.

Usage
-----
    # Default: evaluate current RETRIEVAL_MODE only
    python eval/run_eval.py

    # Evaluate both modes and produce a before/after comparison:
    python eval/run_eval.py --modes vector hybrid

    # Skip the LLM judge (deterministic metrics only):
    python eval/run_eval.py --no-judge

    # Custom dataset / output paths:
    python eval/run_eval.py --dataset eval/dataset.json --out eval/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make `src` importable when running as a top-level script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from src._retry import with_retries  # noqa: E402
from src.config import Settings, load_settings, require_api_key  # noqa: E402
from src.generate import generate_answer  # noqa: E402
from src.retrieve import retrieve  # noqa: E402
from src.vectorstore import Retrieved  # noqa: E402


JUDGE_SYSTEM = """You are a strict evaluator of RAG (Retrieval-Augmented Generation) outputs.
Score each axis on an integer 0-5 scale (0 = terrible, 5 = perfect).
Return ONLY a single JSON object, no prose, with this exact shape:
{"faithfulness": <int 0-5>, "answer_relevance": <int 0-5>, "rationale": "<one short sentence>"}
Definitions:
- faithfulness:    Is every factual claim in the ANSWER directly supported by the CONTEXT?
                   5 = fully supported, 0 = entirely fabricated.
- answer_relevance: Does the ANSWER actually address the QUESTION?
                    5 = directly answers it, 0 = off-topic or refuses without cause."""


def _judge_prompt(question: str, context: str, answer: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return the JSON object now."
    )


def _safe_parse_judge(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of ``raw``, even if the model added prose."""
    raw = raw.strip()
    # Trim ```json fences if the model used them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    # Find the first {...} block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"faithfulness": None, "answer_relevance": None, "rationale": raw[:120]}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"faithfulness": None, "answer_relevance": None, "rationale": raw[:120]}


def llm_judge(
    *,
    client: OpenAI,
    settings: Settings,
    question: str,
    context: str,
    answer: str,
) -> dict[str, Any]:
    """One chat call that returns both faithfulness and answer_relevance."""

    def _call():
        return client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _judge_prompt(question, context, answer)},
            ],
            temperature=0.0,
        )

    completion = with_retries(_call)
    raw = (completion.choices[0].message.content or "").strip()
    return _safe_parse_judge(raw)


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------


def _contains_any_snippet(text: str, snippets: list[str]) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in snippets)


def deterministic_metrics(
    retrieved: list[Retrieved], snippets: list[str]
) -> dict[str, float]:
    if not retrieved:
        return {"hit": 0.0, "rr": 0.0, "ctx_precision": 0.0, "ctx_recall": 0.0}

    hit = 0.0
    rr = 0.0
    hit_chunks = 0
    for i, r in enumerate(retrieved, start=1):
        if _contains_any_snippet(r.text, snippets):
            hit_chunks += 1
            if rr == 0.0:
                rr = 1.0 / i
                hit = 1.0
    ctx_precision = hit_chunks / len(retrieved)

    # Recall = fraction of distinct snippets covered by the union of chunks.
    covered = 0
    joined = " ".join(r.text for r in retrieved).lower()
    for s in snippets:
        if s.lower() in joined:
            covered += 1
    ctx_recall = covered / len(snippets) if snippets else 0.0

    return {
        "hit": hit,
        "rr": rr,
        "ctx_precision": ctx_precision,
        "ctx_recall": ctx_recall,
    }


# ---------------------------------------------------------------------------
# Per-mode evaluation
# ---------------------------------------------------------------------------


@dataclass
class ItemResult:
    id: str
    question: str
    answer: str
    hit: float
    rr: float
    ctx_precision: float
    ctx_recall: float
    faithfulness: float | None
    answer_relevance: float | None
    rationale: str


@dataclass
class ModeReport:
    mode: str
    items: list[ItemResult]

    @property
    def n(self) -> int:
        return len(self.items)

    def mean(self, attr: str) -> float | None:
        vals = [getattr(i, attr) for i in self.items if getattr(i, attr) is not None]
        return statistics.mean(vals) if vals else None


def evaluate_mode(
    *,
    mode: str,
    items: list[dict[str, Any]],
    base_settings: Settings,
    judge: bool,
    judge_sleep: float,
    gen_sleep: float,
) -> ModeReport:
    # Build a Settings with the requested retrieval mode without mutating
    # the frozen dataclass.
    settings = Settings(
        provider=base_settings.provider,
        api_key=base_settings.api_key,
        base_url=base_settings.base_url,
        chat_model=base_settings.chat_model,
        embed_model=base_settings.embed_model,
        chunk_size_tokens=base_settings.chunk_size_tokens,
        chunk_overlap_tokens=base_settings.chunk_overlap_tokens,
        top_k=base_settings.top_k,
        retrieval_mode=mode,
        chroma_dir=base_settings.chroma_dir,
        chroma_collection=base_settings.chroma_collection,
    )

    judge_client = None
    if judge:
        api_key = require_api_key(settings)
        judge_client = OpenAI(api_key=api_key, base_url=settings.base_url)

    out: list[ItemResult] = []
    for item in items:
        q = item["question"]
        snippets = item.get("relevant_snippets", [])
        print(f"  [{mode}] {item['id']}: {q[:80]}")

        retrieved = retrieve(q, settings=settings)
        det = deterministic_metrics(retrieved, snippets)

        # Skip generation entirely when not judging — the deterministic
        # metrics don't need the answer, and generation burns chat quota
        # we can't afford on Gemini's free tier (20 req/day on Flash).
        if judge:
            ans_obj = generate_answer(q, retrieved, settings=settings)
            answer = ans_obj.text
            if gen_sleep:
                time.sleep(gen_sleep)
        else:
            answer = "(generation skipped: --no-judge)"

        faith = relevance = None
        rationale = ""
        if judge and judge_client is not None:
            context = "\n\n---\n\n".join(r.text for r in retrieved)
            verdict = llm_judge(
                client=judge_client,
                settings=settings,
                question=q,
                context=context,
                answer=answer,
            )
            try:
                faith = float(verdict.get("faithfulness")) if verdict.get("faithfulness") is not None else None
                relevance = float(verdict.get("answer_relevance")) if verdict.get("answer_relevance") is not None else None
            except (TypeError, ValueError):
                pass
            rationale = str(verdict.get("rationale", ""))[:160]
            # Gentle pacing for free-tier rate limits.
            if judge_sleep:
                time.sleep(judge_sleep)

        out.append(
            ItemResult(
                id=item["id"],
                question=q,
                answer=answer,
                hit=det["hit"],
                rr=det["rr"],
                ctx_precision=det["ctx_precision"],
                ctx_recall=det["ctx_recall"],
                faithfulness=faith,
                answer_relevance=relevance,
                rationale=rationale,
            )
        )

    return ModeReport(mode=mode, items=out)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt5(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f} / 5"


def render_summary(reports: list[ModeReport]) -> str:
    headers = ["Metric"] + [r.mode for r in reports]
    rows = [
        ("Hit@k",                      [_fmt_pct(r.mean("hit"))           for r in reports]),
        ("MRR",                        [_fmt(r.mean("rr"))                for r in reports]),
        ("Context precision",          [_fmt_pct(r.mean("ctx_precision")) for r in reports]),
        ("Context recall",             [_fmt_pct(r.mean("ctx_recall"))    for r in reports]),
        ("Faithfulness (LLM judge)",   [_fmt5(r.mean("faithfulness"))     for r in reports]),
        ("Answer relevance (LLM judge)", [_fmt5(r.mean("answer_relevance")) for r in reports]),
    ]
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [
        "| " + " | ".join(headers) + " |",
        sep,
    ]
    for label, cells in rows:
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def render_per_item(reports: list[ModeReport]) -> str:
    """Per-question Hit@k by mode, so the reader can see WHICH questions changed."""
    by_id: dict[str, dict[str, float]] = {}
    for r in reports:
        for it in r.items:
            by_id.setdefault(it.id, {})[r.mode] = it.hit

    headers = ["Question ID"] + [r.mode + " Hit" for r in reports]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for qid, by_mode in by_id.items():
        cells = [qid] + [("hit" if by_mode.get(r.mode) else "miss") for r in reports]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results(reports: list[ModeReport], out_path: Path, dataset_meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = render_summary(reports)
    per_item = render_per_item(reports)
    payload = (
        "# Evaluation results\n\n"
        f"Document: `{dataset_meta.get('source_document', '?')}`  \n"
        f"Source:   {dataset_meta.get('source_url', '?')}  \n"
        f"License:  {dataset_meta.get('license', '?')}  \n"
        f"Items in eval set: **{reports[0].n}** (if multiple modes compared, "
        "the same items run under each).\n\n"
        "## Aggregate metrics\n\n"
        f"{summary}\n\n"
        "## Per-question Hit@k by mode\n\n"
        f"{per_item}\n\n"
        "## How to reproduce\n\n"
        "```bash\n"
        "# Inside Docker (recommended — same env that ships):\n"
        "docker compose run --rm rag python eval/run_eval.py --modes vector hybrid\n"
        "```\n\n"
        "Definitions and caveats are documented in `eval/run_eval.py`.\n"
    )
    out_path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/RESULTS.md"))
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Retrieval modes to evaluate (e.g. --modes vector hybrid). "
        "Default: whichever single mode is currently configured.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-judged metrics (faithfulness, answer relevance). "
        "Useful when you're rate-limited or just iterating on retrieval.",
    )
    parser.add_argument(
        "--judge-sleep",
        type=float,
        default=float(os.getenv("EVAL_JUDGE_SLEEP", "13.0")),
        help="Seconds to sleep after each judge call (free-tier pacing). Default 13s "
        "keeps total chat RPM under Gemini Flash's free-tier 5/min cap.",
    )
    parser.add_argument(
        "--gen-sleep",
        type=float,
        default=float(os.getenv("EVAL_GEN_SLEEP", "13.0")),
        help="Seconds to sleep after each answer-generation call. Default 13s.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    items = raw["items"]
    meta = raw.get("_meta", {})

    base = load_settings()
    require_api_key(base)  # fail fast if no key

    modes = args.modes or [base.retrieval_mode]
    judge = not args.no_judge

    print(f"Evaluating {len(items)} items across modes={modes}, judge={judge}")
    reports: list[ModeReport] = []
    for mode in modes:
        print(f"\n=== MODE: {mode} ===")
        reports.append(
            evaluate_mode(
                mode=mode,
                items=items,
                base_settings=base,
                judge=judge,
                judge_sleep=args.judge_sleep,
                gen_sleep=args.gen_sleep,
            )
        )

    summary = render_summary(reports)
    print("\nAggregate metrics")
    print("=================")
    print(summary)

    write_results(reports, args.out, meta)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
