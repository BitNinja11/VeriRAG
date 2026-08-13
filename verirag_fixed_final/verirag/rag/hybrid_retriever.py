"""Hybrid retrieval: dense + BM25, fused into a single ranking.

Two fusion strategies are implemented so the ablation study can compare them:

  rrf       Reciprocal Rank Fusion. Uses only ranks, so the two score
            distributions never need to be made commensurable. Robust default.

  weighted  Min-max normalise each score list, then linearly combine. Simple
            and interpretable, but sensitive to score distribution shape --
            one outlier compresses everything else toward zero.

Both are heuristics. Neither is "the correct" fusion method.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index
from .chunker import Chunk
from .embeddings import EmbeddingBackend
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float = 0.0
    bm25_score: float = 0.0

    @property
    def retrieval_source(self) -> str:
        if self.dense_rank is not None and self.bm25_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        return "bm25"


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingBackend,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        hybrid_top_k: int = 10,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_top_k = hybrid_top_k
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        texts = [c.text for c in chunks]
        vectors = embedder.fit(texts)
        self.vector_store = VectorStore(vectors.shape[1])
        self.vector_store.add(vectors)
        self.bm25 = BM25Index(texts)

    def dense_search(self, question: str, top_k: int) -> list[tuple[int, float]]:
        query_vec = self.embedder.encode([question])[0]
        return self.vector_store.search(query_vec, top_k)

    def bm25_search(self, question: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(question, top_k)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.hybrid_top_k
        dense_hits = self.dense_search(question, self.dense_top_k)
        bm25_hits = self.bm25_search(question, self.bm25_top_k)

        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits)}
        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_hits)}
        dense_raw = dict(dense_hits)
        bm25_raw = dict(bm25_hits)

        candidates = set(dense_rank) | set(bm25_rank)
        if not candidates:
            return []

        if self.fusion == "weighted":
            ordered = sorted(candidates)
            d_norm = dict(
                zip(ordered, _minmax([dense_raw.get(i, 0.0) for i in ordered]))
            )
            b_norm = dict(
                zip(ordered, _minmax([bm25_raw.get(i, 0.0) for i in ordered]))
            )
            fused = {
                i: self.dense_weight * d_norm[i] + self.bm25_weight * b_norm[i]
                for i in ordered
            }
        else:
            fused = {}
            for idx in candidates:
                score = 0.0
                if idx in dense_rank:
                    score += 1.0 / (self.rrf_k + dense_rank[idx] + 1)
                if idx in bm25_rank:
                    score += 1.0 / (self.rrf_k + bm25_rank[idx] + 1)
                fused[idx] = score

        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        return [
            RetrievedChunk(
                chunk=self.chunks[idx],
                score=float(score),
                dense_rank=dense_rank.get(idx),
                bm25_rank=bm25_rank.get(idx),
                dense_score=float(dense_raw.get(idx, 0.0)),
                bm25_score=float(bm25_raw.get(idx, 0.0)),
            )
            for idx, score in ranked
        ]

    def dense_only(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Dense-only retrieval path used by the vanilla RAG baseline."""
        return [
            RetrievedChunk(
                chunk=self.chunks[idx],
                score=float(score),
                dense_rank=rank,
                dense_score=float(score),
            )
            for rank, (idx, score) in enumerate(self.dense_search(question, top_k))
        ]
