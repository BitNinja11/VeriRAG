"""Evaluation metrics.

Scoring is string-based rather than using an LLM judge. On a 32-item benchmark
an LLM judge introduces a second source of error that is harder to audit than
the thing being measured, and the expected answers are short factual values
where numeric-aware matching is reliable.

Metrics:
  answer_correct      the expected value appears in the answer
  abstained           the system declined to answer
  abstention_correct  it abstained exactly when it should have (decision accuracy)
  conflict_correct    conflict flag matches the expected label
  citation_correct    at least one cited source is a gold source (hit rate)
  primary_citation_correct  the first/primary cited source is a gold source
  hallucinated_value  a wrong known-distractor value appears in the answer
"""

from __future__ import annotations

import re
from typing import Any

ABSTAIN_MARKERS = (
    "do not contain enough information",
    "does not contain enough information",
    "not contain enough information",
    "cannot answer",
    "insufficient evidence",
    "unable to answer",
    "no information",
    "does not answer",
    "not answered by",
    "cannot be determined",
    "could not be resolved",
    "not specified in",
    "do not specify",
    "does not specify",
)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_abstention(answer: str) -> bool:
    low = normalise(answer)
    return any(marker in low for marker in ABSTAIN_MARKERS)


# Phrases that mark the transition from the asserted answer to the disclosure
# of what was rejected. Everything after one of these is commentary about
# other sources, not the system's own answer.
DISCLOSURE_MARKERS = (
    "sources disagreed",
    "source disagreed",
    "an older",
    "however,",
    "but the older",
    "conflicting source",
    "other sources",
    "was rejected",
    "the system selected",
    "this is corroborated",
)


def primary_assertion(answer: str) -> str:
    """The part of the answer that states the system's own conclusion.

    A conflict-aware answer reads: "<answer>. Sources disagreed: X says 18,
    Y says 24. Selected X because...". A naive substring check over the WHOLE
    string finds the correct value inside the list of REJECTED claims and
    marks a wrong answer correct. That is exactly how the recency-only
    ablation scored identically to the full system while actually asserting
    the marketing blog's 35 days. Correctness must be judged on what the
    system asserted, not on every number it mentioned.
    """
    low = normalise(answer)
    cut = len(low)
    for marker in DISCLOSURE_MARKERS:
        idx = low.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return low[:cut].strip() or low


_QTY_IN_ANSWER = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:[a-z]+\s+){0,3}?"
    r"(day|days|hour|hours|character|characters|month|months|week|weeks|year|years)\b",
    re.IGNORECASE,
)



_RANGE_IN_ANSWER = re.compile(
    r"(?:between\s+)?(\d+(?:\.\d+)?)\s*"
    r"(?:to|and|[-–—])\s*(\d+(?:\.\d+)?)\s+"
    r"(?:[a-z]+\s+){0,2}?"
    r"(day|days|hour|hours|month|months|week|weeks|year|years)\b",
    re.IGNORECASE,
)


def _ranges(text: str) -> set[tuple[float, float, str]]:
    """All numeric ranges as (low, high, singular-unit)."""
    return {
        (float(m.group(1)), float(m.group(2)), m.group(3).lower().rstrip("s"))
        for m in _RANGE_IN_ANSWER.finditer(text)
    }

def _quantities(text: str) -> set[tuple[float, str]]:
    """All (number, singular-unit) pairs in a span."""
    return {
        (float(m.group(1)), m.group(2).lower().rstrip("s"))
        for m in _QTY_IN_ANSWER.finditer(text)
    }


def answer_is_correct(answer: str, expected_any: list[str]) -> bool:
    """True if any accepted surface form appears in the primary assertion.

    Matching is numeric-aware, not purely literal. Policy prose puts modifiers
    between the number and its unit -- "10 unused leave days", "90 consecutive
    days" -- so a raw substring test for "10 days" fails on a correct answer.
    Four benchmark items were being scored as misses for this reason alone,
    which understated the system by 12 points. Quantities are compared as
    (number, unit) pairs; non-numeric expectations fall back to substring.
    """
    if not expected_any:
        return False

    primary = primary_assertion(answer)
    answer_qty = _quantities(primary)
    answer_ranges = _ranges(primary)

    for expected in expected_any:
        if not expected:
            continue
        norm = normalise(expected)
        if norm in primary:
            return True

        # A range is one value, not two interchangeable endpoints. Without
        # this check, the old 5-to-7-day refund policy can receive credit for
        # a 3-to-5-day gold answer merely because both ranges contain "5".
        expected_ranges = _ranges(norm)
        if expected_ranges:
            if expected_ranges & answer_ranges:
                return True
            continue

        expected_qty = _quantities(norm)
        if expected_qty and expected_qty & answer_qty:
            return True

        bare = set(re.findall(r"\d+(?:\.\d+)?", norm))
        if bare and not re.search(r"(?:to|and|[-–—])", norm):
            answer_numbers = {
                str(int(n)) if n.is_integer() else str(n) for n, _ in answer_qty
            }
            if bare <= answer_numbers:
                return True

    return False


def contains_distractor_value(
    answer: str, distractor_values: list[str]
) -> bool:
    low = normalise(answer)
    return any(normalise(v) in low for v in distractor_values if v)


def misled_by_distractor(
    answer: str, distractor_values: list[str], answer_correct: bool
) -> bool:
    """True only if a wrong value appears AND the answer is wrong.

    A conflict-aware system is supposed to name the rejected value: "the FAQ
    says 18 days, but the 2026 policy supersedes it". A naive
    substring check counts that transparency as a hallucination and punishes
    the system for the exact behaviour it was built to exhibit -- which is why
    an earlier version of this metric scored VeriRAG *worse* than the ablation
    that hid the conflict entirely. Requiring the answer to also be wrong
    separates "was misled" from "explained what it rejected".
    """
    if answer_correct:
        return False
    return contains_distractor_value(answer, distractor_values)


def citation_is_correct(
    citations: list[str], evidence: list[dict], gold_sources: list[str]
) -> bool | None:
    """True if any cited claim came from a gold document.

    Returns None when the question has no gold source (unanswerable items),
    so those are excluded from the citation average rather than counted as
    failures.
    """
    if not gold_sources:
        return None
    if not citations:
        return False

    cited_titles = {normalise(c) for c in citations}
    for ev in evidence:
        label = normalise(f"[{ev.get('claim_id')}] {ev.get('source', '')}")
        title = normalise(ev.get("source", ""))
        if any(title and title in c for c in cited_titles) or label in cited_titles:
            doc_id = str(ev.get("chunk_id", "")).split("::")[0]
            if doc_id in gold_sources:
                return True
    return False


def primary_citation_is_correct(
    citations: list[str], evidence: list[dict], gold_sources: list[str]
) -> bool | None:
    """Whether the answer's primary citation points to a gold document.

    VeriRAG intentionally cites rejected sources when explaining a conflict, so
    requiring *every* citation to be gold would punish transparent disclosure.
    The first citation is the source supporting the primary assertion and is a
    stronger correctness test than the old "at least one gold citation" metric.
    """
    if not gold_sources:
        return None
    if not citations:
        return False

    primary = normalise(citations[0])
    for ev in evidence:
        label = normalise(f"[{ev.get('claim_id')}] {ev.get('source', '')}")
        title = normalise(ev.get("source", ""))
        if (title and title in primary) or label == primary:
            doc_id = str(ev.get("chunk_id", "")).split("::")[0]
            return doc_id in gold_sources
    return False


def retrieval_hit(evidence: list[dict], gold_sources: list[str]) -> bool | None:
    """Did retrieval surface at least one gold document at all?"""
    if not gold_sources:
        return None
    retrieved_docs = {
        str(ev.get("chunk_id", "")).split("::")[0] for ev in evidence
    }
    return bool(retrieved_docs & set(gold_sources))


def score_item(
    item: dict[str, Any],
    result: dict[str, Any],
    distractor_values: list[str] | None = None,
) -> dict[str, Any]:
    """Score one benchmark question against one system's output."""
    answer = result.get("answer", "")
    expected_any = item.get("expected_any") or item.get("expected_answer_any") or []
    expect_abstain = bool(item.get("expect_abstain", False))
    gold_sources = item.get("gold_sources") or []

    abstained = is_abstention(answer)

    if expect_abstain:
        correct = abstained
    else:
        correct = answer_is_correct(answer, expected_any) and not abstained

    evidence = result.get("evidence", []) or []

    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "question": item.get("question"),
        "answer": answer,
        "answer_correct": bool(correct),
        "expect_abstain": expect_abstain,
        "abstained": abstained,
        "abstention_correct": (abstained == expect_abstain),
        "expected_conflict": bool(item.get("expected_conflict", False)),
        "detected_conflict": bool(result.get("conflict_detected", False)),
        "conflict_correct": (
            bool(result.get("conflict_detected", False))
            == bool(item.get("expected_conflict", False))
        ),
        "citation_correct": citation_is_correct(
            result.get("citations", []) or [], evidence, gold_sources
        ),
        "primary_citation_correct": primary_citation_is_correct(
            result.get("citations", []) or [], evidence, gold_sources
        ),
        "retrieval_hit": retrieval_hit(evidence, gold_sources),
        "hallucinated_value": misled_by_distractor(
            answer, distractor_values or [], bool(correct)
        ),
        "adjudicator_invoked": bool(result.get("adjudicator_invoked", False)),
        "llm_calls": int(result.get("llm_calls", 0) or 0),
        "latency": float(result.get("latency", 0.0) or 0.0),
    }


def bootstrap_ci(
    values: list, confidence: float = 0.95, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for a mean.

    Exists because n=32 is small and a bare "1.000" hides that. Resampling the
    per-item scores with replacement and reading the 2.5th/97.5th percentiles
    of the resampled means says how much of the headline number is the system
    and how much is the sample size. On 32 items a perfect score still only
    supports a lower bound near 0.89, and stating that is more honest -- and
    more persuasive -- than the point estimate alone.

    Deterministic: the seed is fixed so a rerun reproduces the same interval.
    """
    clean = [
        1.0 if v is True else (0.0 if v is False else float(v))
        for v in values
        if v is not None
    ]
    if not clean:
        return None
    if len(set(clean)) == 1:
        # Degenerate sample: every resample is identical, so the bootstrap
        # interval collapses to the point. Fall back to a Wilson-style bound
        # for the all-0/all-1 proportion case, which is the honest answer.
        import math

        n, p_hat = len(clean), clean[0]
        if p_hat in (0.0, 1.0):
            z = 1.959963985
            denom = 1 + z * z / n
            centre = (p_hat + z * z / (2 * n)) / denom
            half = (
                z
                * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
                / denom
            )
            return (max(0.0, centre - half), min(1.0, centre + half))
        return (p_hat, p_hat)

    import random

    rng = random.Random(seed)
    n = len(clean)
    means = sorted(
        sum(rng.choices(clean, k=n)) / n for _ in range(n_boot)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return (lo, hi)


def _fmt_ci(ci: tuple[float, float] | None) -> str | None:
    if ci is None:
        return None
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(1 if v is True else (0 if v is False else float(v)) for v in clean) / len(
        clean
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item scores into headline numbers."""
    if not rows:
        return {}

    abstain_items = [r for r in rows if r.get("expect_abstain")]
    answerable_items = [r for r in rows if not r.get("expect_abstain")]
    conflict_items = [r for r in rows if r.get("expected_conflict")]
    detected_items = [r for r in rows if r.get("detected_conflict")]

    true_positive_conflicts = sum(
        1 for r in rows if r.get("expected_conflict") and r.get("detected_conflict")
    )

    return {
        "n": len(rows),
        "answer_accuracy": _mean([r["answer_correct"] for r in rows]),
        "answer_accuracy_ci": _fmt_ci(
            bootstrap_ci([r["answer_correct"] for r in rows])
        ),
        "abstention_recall": _mean([r["abstained"] for r in abstain_items]),
        "false_abstention_rate": _mean([r["abstained"] for r in answerable_items]),
        "conflict_accuracy": _mean([r["conflict_correct"] for r in rows]),
        "conflict_recall": (
            true_positive_conflicts / len(conflict_items) if conflict_items else None
        ),
        "conflict_precision": (
            true_positive_conflicts / len(detected_items) if detected_items else None
        ),
        "citation_accuracy": _mean([r["citation_correct"] for r in rows]),
        "primary_citation_accuracy": _mean(
            [r["primary_citation_correct"] for r in rows]
        ),
        "retrieval_hit_rate": _mean([r["retrieval_hit"] for r in rows]),
        "hallucination_rate": _mean([r["hallucinated_value"] for r in rows]),
        "adjudicator_invocation_rate": _mean(
            [r["adjudicator_invoked"] for r in rows]
        ),
        "mean_llm_calls": _mean([r["llm_calls"] for r in rows]),
        "mean_latency": _mean([r["latency"] for r in rows]),
    }


def aggregate_by_category(rows: list[dict[str, Any]]) -> dict[str, dict]:
    categories: dict[str, list] = {}
    for row in rows:
        categories.setdefault(row["category"], []).append(row)
    return {cat: aggregate(items) for cat, items in sorted(categories.items())}
