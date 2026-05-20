"""The :class:`Searcher` -- grub's unified, modern search object.

A :class:`Searcher` wraps three things behind one friendly interface:

* a **store** (any source, via :func:`grub.sources.to_store`),
* a **backend** (TF-IDF, semantic, or hybrid, via :mod:`grub.backends`),
* **results** enriched with scores and snippets (:mod:`grub.results`).

You rarely build one by hand -- :func:`grub.grub` is the front door --
but the class is here when you want full control, persistence, or to
plug in your own backend.
"""

from __future__ import annotations

import pickle

from grub.backends import DFLT_SEMANTIC_MODEL, Backend, make_backend
from grub.base import camelcase_and_underscore_tokenizer
from grub.results import SearchResult, SearchResults
from grub.sources import to_store
from grub.util import best_snippet

__all__ = ["Searcher"]


class Searcher:
    """Index any source and search it.

    >>> docs = {
    ...     'python': 'Python is a popular high level programming language.',
    ...     'guitar': 'A guitar is a musical instrument with six strings.',
    ...     'coffee': 'Coffee is a hot drink brewed from roasted beans.',
    ... }
    >>> searcher = Searcher(docs)
    >>> searcher
    <Searcher: tfidf index over 3 documents>
    >>> results = searcher('a musical instrument')
    >>> results.keys[0]
    'guitar'
    >>> searcher('hot drink brewed from beans').keys[0]
    'coffee'

    Each result carries a score and a snippet explaining the match:

    >>> hit = results[0]
    >>> hit.key
    'guitar'
    >>> hit.score > 0
    True
    >>> 'musical instrument' in hit.snippet
    True
    """

    def __init__(
        self,
        source,
        *,
        method: str = "tfidf",
        n_results: int = 10,
        tokenizer=camelcase_and_underscore_tokenizer,
        embed=None,
        model: str = DFLT_SEMANTIC_MODEL,
        alpha: float = 0.5,
        chunk=None,
        snippets: bool = True,
        backend: Backend | None = None,
        **source_kwargs,
    ):
        """Index ``source`` so it can be searched.

        :param source: anything :func:`grub.sources.to_store` understands --
            a folder, glob, module, URL, dict, or list of strings.
        :param method: ``'tfidf'`` (lexical), ``'semantic'`` (embeddings),
            or ``'hybrid'`` (a blend).
        :param n_results: default number of results per search.
        :param tokenizer: splits text into tokens (lexical methods).
        :param embed: optional ``list[str] -> matrix`` embedding function;
            lets ``'semantic'`` use any provider you like.
        :param model: sentence-transformers model name (semantic method).
        :param alpha: lexical weight for the hybrid method (``0``-``1``).
        :param chunk: split long documents into windows -- an int size or
            a ``(size, overlap)`` pair.
        :param snippets: attach match-explaining snippets to results.
        :param backend: a ready-made backend, overriding ``method``.
        :param source_kwargs: forwarded to :func:`grub.sources.to_store`
            (e.g. ``recursive=False``, ``extensions=['.py']``).
        """
        self.store = to_store(source, chunk=chunk, **source_kwargs)
        self.method = method
        self.n_results = int(n_results)
        self.tokenizer = tokenizer
        self.snippets = snippets
        self.backend = backend or make_backend(
            method, tokenizer=tokenizer, embed=embed, model=model, alpha=alpha
        )
        self._keys: list | None = None
        self._fitted = False

    # -- building the index -------------------------------------------------

    def fit(self) -> "Searcher":
        """Build the search index. Called automatically on first search."""
        self._keys = list(self.store)
        docs = [str(self.store[k]) for k in self._keys]
        self.backend.index(docs)
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether the index has been built."""
        return self._fitted

    # -- searching ----------------------------------------------------------

    def search(self, query: str, n: int | None = None) -> SearchResults:
        """Return the :class:`~grub.results.SearchResults` for ``query``."""
        if not self._fitted:
            self.fit()
        n = self.n_results if n is None else int(n)
        indices, scores = self.backend.query(query, n)
        results = [
            SearchResult(
                key=self._keys[i],
                score=float(score),
                snippet=self._snippet(self._keys[i], query),
            )
            for i, score in zip(indices, scores)
        ]
        return SearchResults(results, query=query)

    def __call__(self, query: str, n: int | None = None) -> SearchResults:
        """Alias for :meth:`search` -- a searcher *is* callable."""
        return self.search(query, n)

    def _snippet(self, key, query) -> str:
        if not self.snippets:
            return ""
        try:
            return best_snippet(self.store[key], query, tokenizer=self.tokenizer)
        except Exception:  # noqa: BLE001 - a snippet is never worth a crash
            return ""

    # -- store delegation ---------------------------------------------------

    def __getitem__(self, key):
        """The original text of a document, by key."""
        return self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, key) -> bool:
        return key in self.store

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> str:
        """Pickle this searcher (index included) to ``path``.

        :returns: the path written to.
        """
        if not self._fitted:
            self.fit()
        with open(path, "wb") as stream:
            pickle.dump(self, stream)
        return path

    @classmethod
    def load(cls, path: str) -> "Searcher":
        """Load a searcher previously written by :meth:`save`."""
        with open(path, "rb") as stream:
            searcher = pickle.load(stream)
        if not isinstance(searcher, cls):
            raise TypeError(f"{path!r} does not contain a Searcher.")
        return searcher

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            size = len(self.store)
        except TypeError:  # pragma: no cover - exotic lazy stores
            size = "?"
        return f"<Searcher: {self.method} index over {size} documents>"
