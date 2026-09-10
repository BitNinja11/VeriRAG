"""Sparse keyword retrieval via BM25.

BM25 is what catches exact tokens a dense encoder blurs away: policy codes,
dates, and numbers like "24 days".

`rank_bm25` is a hard dependency rather than an optional one. An earlier
version shipped a hand-written BM25-Okapi fallback advertised as equivalent.
It was not: on this corpus it ranks chunks differently (chunks 2 and 9 swap
for the annual-leave query), and that difference propagates through fusion and
reranking far enough to change a published ablation result. A fallback that
silently alters your numbers is worse than a missing dependency, so it is gone.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    backend = "rank_bm25"

    def __init__(self, texts: list[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "rank_bm25 is required for sparse retrieval. "
                "Install it with: pip install rank-bm25"
            ) from exc

        self.tokenized = [tokenize(t) for t in texts]
        self._model = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = list(self._model.get_scores(tokens))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [(int(i), float(s)) for i, s in ranked if s > 0]
