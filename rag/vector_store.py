"""Dense vector index.

Uses FAISS IndexFlatIP when available. Vectors are L2-normalised upstream, so
inner product is cosine similarity. Falls back to a plain numpy matmul, which
is identical in result and perfectly adequate at this corpus size — FAISS
matters at 10^6 vectors, not 10^2.
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.backend = "numpy"
        self._index = None
        self._matrix: np.ndarray | None = None

        try:
            import faiss

            self._index = faiss.IndexFlatIP(dimension)
            self.backend = "faiss"
        except Exception:
            self._index = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        if self.backend == "faiss":
            self._index.add(vectors)
        else:
            self._matrix = vectors

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return (index, score) pairs sorted by descending similarity."""
        query = np.ascontiguousarray(query.astype(np.float32).reshape(1, -1))

        if self.backend == "faiss":
            scores, indices = self._index.search(query, top_k)
            return [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]

        if self._matrix is None:
            return []

        scores = (self._matrix @ query[0]).astype(float)
        top_k = min(top_k, len(scores))
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]

    @property
    def size(self) -> int:
        if self.backend == "faiss":
            return int(self._index.ntotal)
        return 0 if self._matrix is None else int(self._matrix.shape[0])
