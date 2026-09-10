"""Benchmark harness: VeriRAG vs vanilla RAG, plus ablations.

Run:
    python -m evaluation.evaluate                  # both systems
    python -m evaluation.evaluate --ablation       # add component ablations
    python -m evaluation.evaluate --provider groq  # with a real LLM

Ablations answer the question a reviewer will actually ask -- "which part is
doing the work?" -- by disabling one component at a time:

    no_rerank        skip cross-encoder reranking
    no_adjudicator   detect conflicts but always take the top-ranked claim
    recency_only     adjudicate purely by date (the naive policy)

`recency_only` is the important one. The corpus contains a marketing blog that
is the newest document and factually wrong, so this ablation is expected to
lose on the contradictory split. Demonstrating a predicted failure is a
stronger result than only reporting wins.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from agents.adjudicator import Adjudication  # noqa: E402
from agents.llm import LLMClient  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate,
    aggregate_by_category,
    score_item,
)
from pipeline.strong_baseline import StrongBaseline  # noqa: E402
from pipeline.vanilla_rag import VanillaRAG  # noqa: E402
from pipeline.verirag import VeriRAG  # noqa: E402

# Known-wrong values planted in the corpus. If one appears in an answer, the
# system was misled by a distractor.
DISTRACTOR_VALUES = ["35 days", "90 days", "18 days", "14 days", "5 days"]


def load_benchmark(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["questions"]


def _parse_date_safe(value: str):
    from datetime import datetime

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime((value or "").strip(), fmt).date()
        except ValueError:
            continue
    return None


def recency_only_adjudicate(
    evidence, conflicted_ids, relations=None
) -> Adjudication:
    """Naive adjudicator: newest conflicting claim wins, full stop.

    Only the *selection rule* is ablated. Rejection/disclosure semantics remain
    the same as the full system so the comparison does not accidentally change
    two components at once.
    """
    candidates = [e for e in evidence if e.claim_id in conflicted_ids]
    if not candidates:
        return Adjudication(None, [], 0.0, "Nothing to adjudicate.", "none", "ablation")

    dated = sorted(
        candidates,
        key=lambda e: (_parse_date_safe(e.date) or _parse_date_safe("1900-01-01")),
        reverse=True,
    )
    winner = dated[0]
    rejected = []
    if relations:
        for rel in relations:
            if rel.is_conflict and winner.claim_id in rel.claim_ids:
                rejected.extend(i for i in rel.claim_ids if i != winner.claim_id)
        rejected = sorted(set(rejected))
    if not rejected:
        rejected = [e.claim_id for e in candidates if e.claim_id != winner.claim_id]

    return Adjudication(
        winner.claim_id,
        rejected,
        0.6,
        f"Selected the most recent source ({winner.source}, {winner.date}).",
        "recency_only",
        "ablation",
    )


def backend_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count which backend actually produced each extracted claim.

    `rules` means the offline control ran by design. `rules-fallback` means a
    real provider was configured and FAILED, so the deterministic path ran
    instead. Those two look identical in the score table and mean opposite
    things, which is exactly how a run with a dead API key can report a
    perfect result.
    """
    mix: dict[str, int] = {}
    for row in rows:
        for backend, count in (row.get("_backends") or {}).items():
            mix[backend] = mix.get(backend, 0) + count
    return mix


def run_system(system, questions, label: str) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(questions, 1):
        try:
            result = system.run_dict(item["question"])
        except Exception as exc:  # noqa: BLE001
            result = {"answer": f"[error: {exc}]", "evidence": [], "citations": []}
        row = score_item(item, result, DISTRACTOR_VALUES)
        row["system"] = label
        counts: dict[str, int] = {}
        for ev in result.get("evidence", []) or []:
            b = ev.get("extraction_backend")
            if b:
                counts[b] = counts.get(b, 0) + 1
        row["_backends"] = counts
        rows.append(row)
        mark = "ok " if row["answer_correct"] else "MISS"
        print(f"  [{i:2}/{len(questions)}] {mark} {item['id']:12} {item['question'][:52]}")
    return rows


class AblatedVeriRAG:
    """Wraps VeriRAG and disables one component."""

    def __init__(self, base: VeriRAG, mode: str):
        self.base = base
        self.mode = mode

    def run_dict(self, question: str) -> dict[str, Any]:
        from agents.conflict_detector import has_real_conflict

        vr = self.base
        # Rebuild the pipeline manually so a stage can be skipped.
        from pipeline.verirag import VeriRAGState

        state = VeriRAGState(question=question)
        calls_before = vr.client.call_count
        started = time.time()

        t0 = time.time()
        state.retrieved_chunks = vr.retriever.retrieve(question)
        state.stage_timings["retrieve"] = time.time() - t0

        t0 = time.time()
        if self.mode == "no_rerank":
            from rag.reranker import RerankedChunk

            state.reranked_chunks = [
                RerankedChunk(
                    chunk=c.chunk,
                    rerank_score=c.score,
                    retrieval_score=c.score,
                    retrieval_source=c.retrieval_source,
                )
                for c in state.retrieved_chunks[: config.RERANK_TOP_K]
            ]
        else:
            state.reranked_chunks = vr.reranker.rerank(
                question, state.retrieved_chunks, config.RERANK_TOP_K
            )
        state.stage_timings["rerank"] = time.time() - t0

        t0 = time.time()
        state.evidence = vr.analyst.analyse(question, state.reranked_chunks)
        state.stage_timings["extract_evidence"] = time.time() - t0

        t0 = time.time()
        state.conflicts = vr.detector.detect(question, state.evidence)
        state.stage_timings["detect_conflict"] = time.time() - t0

        if has_real_conflict(state.conflicts):
            conflicted = set()
            for rel in state.conflicts:
                if rel.is_conflict:
                    conflicted.update(rel.claim_ids)

            if self.mode == "no_adjudicator":
                state.adjudication = None
                state.adjudicator_invoked = False
            elif self.mode == "recency_only":
                t0 = time.time()
                state.adjudication = recency_only_adjudicate(
                    state.evidence, conflicted, state.conflicts
                )
                state.adjudicator_invoked = True
                state.stage_timings["adjudicate"] = time.time() - t0
            else:
                t0 = time.time()
                state.adjudication = vr.adjudicator.adjudicate(
                    question, state.evidence, state.conflicts
                )
                state.adjudicator_invoked = True
                state.stage_timings["adjudicate"] = time.time() - t0

        t0 = time.time()
        result = vr.generator.generate(
            question, state.evidence, state.conflicts, state.adjudication
        )
        state.stage_timings["generate"] = time.time() - t0
        state.answer = result.answer
        state.citations = result.citations
        state.confidence = result.confidence
        state.latency = time.time() - started
        state.llm_calls = vr.client.call_count - calls_before
        return state.to_dict()


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if value is None:
        return "  n/a"
    if isinstance(value, float):
        return f"{value:5.3f}"
    return str(value)


def print_summary(results: dict[str, list[dict]]) -> None:
    metrics = [
        ("answer_accuracy", "Answer accuracy"),
        ("answer_accuracy_ci", "  95% CI (bootstrap)"),
        ("abstention_recall", "Abstention recall"),
        ("false_abstention_rate", "False-abstention rate"),
        ("conflict_accuracy", "Conflict-label accuracy"),
        ("conflict_precision", "Conflict precision"),
        ("conflict_recall", "Conflict recall"),
        ("primary_citation_accuracy", "Primary gold-source rate"),
        ("citation_accuracy", "Gold-citation hit rate"),
        ("retrieval_hit_rate", "Retrieval hit rate"),
        ("hallucination_rate", "Distractor-value rate"),
        ("mean_llm_calls", "Mean LLM calls/query"),
        ("mean_latency", "Mean latency (s)"),
    ]

    systems = list(results.keys())
    aggregates = {name: aggregate(rows) for name, rows in results.items()}

    width = 26  # noqa: PLR2004
    print("\n" + "=" * (width + 13 * len(systems)))
    print("OVERALL".ljust(width) + "".join(s[:12].rjust(13) for s in systems))
    print("=" * (width + 13 * len(systems)))
    for key, label in metrics:
        line = label.ljust(width)
        for name in systems:
            line += _fmt(aggregates[name].get(key)).rjust(13)
        print(line)

    print("\n" + "=" * (width + 13 * len(systems)))
    print("ANSWER ACCURACY BY CATEGORY".ljust(width) + "".join(s[:12].rjust(13) for s in systems))
    print("=" * (width + 13 * len(systems)))

    by_cat = {name: aggregate_by_category(rows) for name, rows in results.items()}
    categories = sorted({c for cats in by_cat.values() for c in cats})
    for cat in categories:
        line = f"  {cat}".ljust(width)
        for name in systems:
            line += _fmt(by_cat[name].get(cat, {}).get("answer_accuracy")).rjust(13)
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VeriRAG benchmark.")
    parser.add_argument("--provider", default=None, help="LLM provider override")
    parser.add_argument("--ablation", action="store_true", help="run ablations")
    parser.add_argument(
        "--strong-baseline",
        action="store_true",
        help="also run the single-call conflict-aware LLM baseline "
             "(requires a real provider)",
    )
    parser.add_argument("--limit", type=int, default=None, help="limit questions")
    args = parser.parse_args()

    if args.provider:
        config.LLM_PROVIDER = args.provider.lower()

    questions = load_benchmark(config.BENCHMARK_PATH)
    if args.limit:
        questions = questions[: args.limit]

    client = LLMClient(config.LLM_PROVIDER)
    print(f"Provider: {client.provider} | model: {client.model}")

    verirag = VeriRAG(client=client)
    print(
        f"Corpus: {len(verirag.documents)} docs -> {len(verirag.chunks)} chunks | "
        f"embedder={verirag.embedder.backend} | reranker={verirag.reranker.backend}"
    )
    if verirag.embedder.is_degraded or verirag.reranker.is_degraded:
        print(
            "\n  WARNING: transformer backends unavailable "
            f"(embedder={verirag.embedder.backend}, "
            f"reranker={verirag.reranker.backend}).\n"
            "  This is a separate backend condition, not a lower bound. "
            "Results are not\n"
            "  comparable to a run with sentence-transformers installed; the "
            "condition is\n"
            "  recorded in summary.csv and should be reported with any number.\n"
        )

    vanilla = VanillaRAG(verirag.retriever, client)

    results: dict[str, list[dict]] = {}

    print(f"\n--- vanilla_rag ({len(questions)} questions) ---")
    results["vanilla_rag"] = run_system(vanilla, questions, "vanilla_rag")

    print(f"\n--- verirag ({len(questions)} questions) ---")
    results["verirag"] = run_system(verirag, questions, "verirag")

    if args.strong_baseline:
        if client.is_offline:
            print(
                "\n[!] --strong-baseline needs a real provider "
                "(e.g. --provider groq). Skipping it."
            )
        else:
            print(f"\n--- strong_llm_baseline ({len(questions)} questions) ---")
            strong = StrongBaseline(verirag.retriever, verirag.reranker, client)
            results["strong_llm"] = run_system(
                strong, questions, "strong_llm"
            )

    if args.ablation:
        for mode in ("no_rerank", "no_adjudicator", "recency_only"):
            print(f"\n--- ablation: {mode} ---")
            results[mode] = run_system(
                AblatedVeriRAG(verirag, mode), questions, mode
            )

    all_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for rows in results.values()
        for row in rows
    ]
    write_csv(config.RESULTS_PATH, all_rows)

    # Stamp the backend condition onto every summary row. Retrieval results
    # are backend-dependent -- the deterministic BM25 fallback this project
    # used to ship ranked chunks differently from rank_bm25 and silently moved
    # an ablation number -- so a results file that does not say which stack
    # produced it is not reproducible. This makes the condition part of the
    # artifact instead of a line in a terminal nobody kept.
    condition = {
        "provider": client.provider,
        "model": client.model,
        "embedder_backend": verirag.embedder.backend,
        "reranker_backend": verirag.reranker.backend,
        "provider_errors": client.error_count,
    }
    summary_rows = []
    for name, rows in results.items():
        agg = aggregate(rows)
        mix = backend_mix(rows)
        total = sum(mix.values()) or 1
        agg["extraction_backend_mix"] = " ".join(
            f"{b}={c / total:.0%}" for b, c in sorted(mix.items())
        )
        summary_rows.append(
            {
                "system": name,
                **{k: v for k, v in agg.items() if k != "system"},
                **condition,
            }
        )
    write_csv(config.RESULTS_SUMMARY_PATH, summary_rows)

    print("\n" + "=" * 76)
    print("BACKEND USAGE  (what actually produced the claims)")
    print("=" * 76)
    degraded = []
    for name, rows in results.items():
        mix = backend_mix(rows)
        total = sum(mix.values())
        if not total:
            continue
        parts = ", ".join(
            f"{b} {c / total:.0%}" for b, c in sorted(mix.items())
        )
        print(f"  {name:16} {parts}")
        fb = mix.get("rules-fallback", 0)
        if fb:
            degraded.append((name, fb / total))

    if client.error_count:
        print(f"\n  provider errors: {client.error_count}")
        print(f"  first error    : {client.last_error}")

    if degraded:
        worst = max(rate for _, rate in degraded)
        print("\n  WARNING: provider fallback detected. These are not clean "
              "LLM results.")
        for name, rate in degraded:
            print(f"    {name:16} {rate:.0%} of claims came from deterministic rules")
        if worst >= 0.5:
            print(
                "\n  Over half the agent stages never reached the model, so this "
                "run\n  measured the rule-based control rather than the provider. "
                "Do not\n  report these numbers as a provider result."
            )

    print_summary(results)
    print(f"Per-question results -> {config.RESULTS_PATH}")
    print(f"Summary              -> {config.RESULTS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
