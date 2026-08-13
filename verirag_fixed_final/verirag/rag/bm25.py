"""Sparse keyword retrieval via BM25.

Uses rank_bm25 when installed; otherwise falls back to a compact BM25-Okapi
implementation so the hybrid retriever always has a sparse arm. BM25 is what
catches exact tokens a dense encoder blurs away: policy codes, dates, numbers
like "24 days", and rare terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _FallbackBM25:
    """BM25-Okapi. Standard parameterisation: k1=1.5, b=0.75."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_len = (
            sum(self.doc_lens) / self.n_docs if self.n_docs else 0.0
        )
        self.term_freqs = [Counter(doc) for doc in corpus]

        doc_freq: Counter[str] = Counter()
        for doc in corpus:
            doc_freq.update(set(doc))

        # Standard BM25 IDF with +0.5 smoothing, floored at a small positive
        # value so very common terms cannot contribute negative score.
        self.idf = {
            term: max(
                1e-6,
                math.log(
                    (self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0
                ),
            )
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for idx in range(self.n_docs):
            freqs = self.term_freqs[idx]
            length = self.doc_lens[idx]
            norm = self.k1 * (
                1 - self.b + self.b * length / (self.avg_len or 1.0)
            )
            total = 0.0
            for term in query:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + norm)
            scores[idx] = total
        return scores


class BM25Index:
    def __init__(self, texts: list[str]):
        self.tokenized = [tokenize(t) for t in texts]
        try:
            from rank_bm25 import BM25Okapi

            self._model = BM25Okapi(self.tokenized)
            self.backend = "rank_bm25"
        except Exception:
            self._model = _FallbackBM25(self.tokenized)
            self.backend = "builtin"

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = list(self._model.get_scores(tokens))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [(int(i), float(s)) for i, s in ranked if s > 0]
