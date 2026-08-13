"""Dense embedding backend.

Primary path: sentence-transformers (a real bi-encoder).
Fallback path: TF-IDF + truncated SVD, i.e. classical LSA.

The fallback exists so the pipeline runs on a machine with no GPU, no torch,
and no network access to Hugging Face. It is genuinely worse at the semantic
matching that motivates dense retrieval in the first place ("holidays" vs
"paid leave"), and the README says so. It is a portability guarantee, not a
claim of equivalence.
"""

from __future__ import annotations

import numpy as np


class EmbeddingBackend:
    """Wraps whichever embedding implementation is available."""

    def __init__(self, model_name: str, prefer_transformer: bool = True):
        self.model_name = model_name
        self.backend = "none"
        self._model = None
        self._vectorizer = None
        self._svd = None
        self._dim = 0

        if prefer_transformer:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
                self._dim = int(
                    self._model.get_sentence_embedding_dimension()
                )
            except Exception:
                self._model = None

        if self._model is None:
            self.backend = "tfidf-svd"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_degraded(self) -> bool:
        return self.backend == "tfidf-svd"

    def fit(self, texts: list[str]) -> np.ndarray:
        """Fit (if needed) and return embeddings for the corpus."""
        if self.backend == "sentence-transformers":
            vectors = self._model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return self._normalise(np.asarray(vectors, dtype=np.float32))

        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        sparse = self._vectorizer.fit_transform(texts)

        # SVD needs n_components < n_features and (for a useful reduction)
        # more than one available dimension. Tiny corpora can violate that;
        # in that case the dense TF-IDF matrix itself is a perfectly valid
        # cosine-search representation and avoids a hard failure.
        max_components = min(128, sparse.shape[0] - 1, sparse.shape[1] - 1)
        if max_components >= 1:
            self._svd = TruncatedSVD(
                n_components=max_components, random_state=42
            )
            dense = self._svd.fit_transform(sparse)
        else:
            self._svd = None
            dense = sparse.toarray()
        self._dim = dense.shape[1]
        return self._normalise(np.asarray(dense, dtype=np.float32))

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed new text using the already-fitted backend."""
        if self.backend == "sentence-transformers":
            vectors = self._model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return self._normalise(np.asarray(vectors, dtype=np.float32))

        if self._vectorizer is None:
            raise RuntimeError("EmbeddingBackend.fit() must be called first.")

        sparse = self._vectorizer.transform(texts)
        dense = self._svd.transform(sparse) if self._svd is not None else sparse.toarray()
        return self._normalise(np.asarray(dense, dtype=np.float32))

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise rows so inner product equals cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
