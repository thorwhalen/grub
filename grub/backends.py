"""Search backends: the engines that score documents against a query.

Every backend implements the same tiny interface:

* ``index(docs)`` -- build an index over a list of text documents.
* ``scores(query)`` -- return a relevance score per document.
* ``query(query, n)`` -- return the top-``n`` ``(indices, scores)``.

Three backends ship with grub:

* :class:`TfidfBackend` -- lexical TF-IDF + cosine similarity (default,
  no extra dependencies).
* :class:`SemanticBackend` -- dense embeddings; understands meaning, not
  just words. Needs ``sentence-transformers`` (``pip install grub[semantic]``)
  or any embedding function you supply.
* :class:`HybridBackend` -- a weighted blend of the two.

Pick a backend by name with :func:`make_backend`.
"""

from __future__ import annotations

import numpy as np

from grub.base import camelcase_and_underscore_tokenizer
from grub.util import clamp_n

__all__ = [
    "Backend",
    "TfidfBackend",
    "SemanticBackend",
    "HybridBackend",
    "make_backend",
    "DFLT_SEMANTIC_MODEL",
]

#: Default sentence-transformers model for :class:`SemanticBackend`.
DFLT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"


class Backend:
    """Base backend: turns per-document ``scores`` into a ranked ``query``."""

    method = "base"

    def index(self, docs):  # pragma: no cover - abstract
        raise NotImplementedError

    def scores(self, query):  # pragma: no cover - abstract
        raise NotImplementedError

    def query(self, query, n):
        """Top-``n`` ``(indices, scores)`` for ``query``, best first."""
        scores = np.asarray(self.scores(query), dtype="float64")
        n = clamp_n(n, len(scores))
        # primary key: descending score; tie-break: ascending original index
        order = np.lexsort((np.arange(len(scores)), -scores))
        idx = order[:n]
        return idx, scores[idx]


class TfidfBackend(Backend):
    """Lexical search via TF-IDF vectors and cosine similarity.

    Fast, dependency-light, and excellent when the query shares actual
    words with the documents -- the right default for code and docs.
    """

    method = "tfidf"

    def __init__(self, *, tokenizer=camelcase_and_underscore_tokenizer, **tfidf_kwargs):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tokenizer = tokenizer
        self._tfidf_kwargs = tfidf_kwargs
        if tokenizer is not None:
            self.vectorizer = TfidfVectorizer(
                tokenizer=tokenizer, token_pattern=None, **tfidf_kwargs
            )
        else:
            self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self._matrix = None

    def index(self, docs):
        self._matrix = self.vectorizer.fit_transform(list(docs))
        return self

    def scores(self, query):
        if self._matrix is None:
            raise RuntimeError("TfidfBackend.index() must be called before scores().")
        query_vec = self.vectorizer.transform([str(query)])
        # tf-idf rows are L2-normalised, so the dot product is the cosine.
        return np.asarray(self._matrix.dot(query_vec.T).todense()).ravel()


class SemanticBackend(Backend):
    """Dense-embedding search: matches meaning, not just shared words.

    Supply your own ``embed`` callable (``list[str] -> 2D array``) to use
    any provider, or leave it ``None`` to lazily load a local
    ``sentence-transformers`` model.
    """

    method = "semantic"

    def __init__(self, *, embed=None, model: str = DFLT_SEMANTIC_MODEL):
        self._embed = embed
        self.model_name = model
        self._model = None
        self._matrix = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - env dependent
                raise ImportError(
                    "Semantic search needs sentence-transformers. Install it "
                    "with:  pip install 'grub[semantic]'  -- or pass your own "
                    "embed=... function to grub()."
                ) from error
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts):
        """Embed ``texts`` into an L2-normalised matrix."""
        texts = [str(t) for t in texts]
        if self._embed is not None:
            vectors = np.asarray(self._embed(texts), dtype="float32")
        else:
            vectors = np.asarray(
                self._get_model().encode(texts, convert_to_numpy=True),
                dtype="float32",
            )
        return _l2_normalize(vectors)

    def index(self, docs):
        self._matrix = self.embed(list(docs))
        return self

    def scores(self, query):
        if self._matrix is None:
            raise RuntimeError(
                "SemanticBackend.index() must be called before scores()."
            )
        query_vec = self.embed([query])[0]
        return self._matrix @ query_vec

    def __getstate__(self):
        # The (large, sometimes unpicklable) model is rebuilt lazily on load.
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class HybridBackend(Backend):
    """A weighted blend of lexical (TF-IDF) and semantic search.

    ``alpha`` is the lexical weight: ``1.0`` is pure TF-IDF, ``0.0`` is
    pure semantic, ``0.5`` (default) gives an even mix.
    """

    method = "hybrid"

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        tokenizer=camelcase_and_underscore_tokenizer,
        embed=None,
        model: str = DFLT_SEMANTIC_MODEL,
    ):
        self.alpha = float(alpha)
        self.lexical = TfidfBackend(tokenizer=tokenizer)
        self.semantic = SemanticBackend(embed=embed, model=model)

    def index(self, docs):
        docs = list(docs)
        self.lexical.index(docs)
        self.semantic.index(docs)
        return self

    def scores(self, query):
        lexical = _minmax(self.lexical.scores(query))
        semantic = _minmax(self.semantic.scores(query))
        return self.alpha * lexical + (1.0 - self.alpha) * semantic


def make_backend(
    method: str = "tfidf",
    *,
    tokenizer=camelcase_and_underscore_tokenizer,
    embed=None,
    model: str = DFLT_SEMANTIC_MODEL,
    alpha: float = 0.5,
) -> Backend:
    """Build a backend by name: ``'tfidf'``, ``'semantic'``, or ``'hybrid'``.

    >>> make_backend("tfidf").method
    'tfidf'
    """
    name = str(method).lower()
    if name in {"tfidf", "lexical", "keyword"}:
        return TfidfBackend(tokenizer=tokenizer)
    if name in {"semantic", "embedding", "embeddings", "dense"}:
        return SemanticBackend(embed=embed, model=model)
    if name == "hybrid":
        return HybridBackend(alpha=alpha, tokenizer=tokenizer, embed=embed, model=model)
    raise ValueError(
        f"Unknown search method {method!r}. Use 'tfidf', 'semantic', or 'hybrid'."
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows left untouched)."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype="float32"))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale scores into ``[0, 1]`` so backends can be blended fairly."""
    scores = np.asarray(scores, dtype="float64")
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
