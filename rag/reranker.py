"""Cross-encoder reranking.

The retriever is a bi-encoder: query and chunk are embedded independently, so
their representations never interact. A cross-encoder feeds (query, chunk) as
a single joined input, letting attention run across both. That is far more
accurate per pair and far more expensive, which is exactly why it runs on ~10
candidates rather than the whole corpus.

Fallback (no torch / no network): a lexical relevance score combining query
term coverage and IDF-lite weighting. Clearly weaker; kept so the pipeline
never hard-fails.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .bm25 import tokenize
from .hybrid_retriever import RetrievedChunk


_QUERY_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "do", "does", "is", "are", "what",
    "how", "many", "much", "in", "on", "at", "be", "with", "that", "this",
    "it", "which", "who", "when", "long", "can", "may", "must", "after",
    "before", "within",
}


@dataclass
class RerankedChunk:
    chunk: object
    rerank_score: float
    retrieval_score: float
    retrieval_source: str

    @property
    def text(self) -> str:
        return self.chunk.text


class Reranker:
    def __init__(self, model_name: str, prefer_cross_encoder: bool = True):
        self.model_name = model_name
        self.backend = "lexical"
        self._model = None

        if prefer_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name)
                self.backend = "cross-encoder"
            except Exception:
                self._model = None

    @property
    def is_degraded(self) -> bool:
        return self.backend == "lexical"

    def _lexical_scores(
        self, question: str, texts: list[str]
    ) -> list[float]:
        q_tokens = [
            t for t in tokenize(question)
            if len(t) > 2 and t not in _QUERY_STOPWORDS
        ]
        if not q_tokens:
            return [0.0] * len(texts)

        tokenized = [tokenize(t) for t in texts]
        doc_freq: Counter[str] = Counter()
        for doc in tokenized:
            doc_freq.update(set(doc))
        n_docs = max(1, len(texts))

        scores = []
        for doc in tokenized:
            counts = Counter(doc)
            total = 0.0
            matched = 0
            for term in set(q_tokens):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                matched += 1
                idf = math.log(1 + n_docs / (1 + doc_freq.get(term, 0)))
                total += idf * (1 + math.log(tf))
            coverage = matched / len(set(q_tokens))
            length_norm = 1.0 / (1.0 + math.log(1 + len(doc) / 100))
            scores.append(total * length_norm * (0.5 + 0.5 * coverage))

        hi = max(scores) if scores else 0.0
        if hi <= 0:
            return [0.0] * len(texts)
        return [s / hi for s in scores]

    def rerank(
        self, question: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RerankedChunk]:
        if not candidates:
            return []

        texts = [c.chunk.text for c in candidates]

        if self.backend == "cross-encoder":
            pairs = [(question, t) for t in texts]
            raw = self._model.predict(pairs)
            scores = [float(s) for s in raw]
        else:
            scores = self._lexical_scores(question, texts)

        reranked = [
            RerankedChunk(
                chunk=cand.chunk,
                rerank_score=score,
                retrieval_score=cand.score,
                retrieval_source=cand.retrieval_source,
            )
            for cand, score in zip(candidates, scores)
        ]

        reranked.sort(key=lambda r: -r.rerank_score)
        return reranked[:top_k]
