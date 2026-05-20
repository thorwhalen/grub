"""The one-line front door to grub: :func:`grub` (and its alias :func:`search`).

``grub(source)`` builds a reusable :class:`~grub.searcher.Searcher`;
``grub(source, query)`` searches in a single call. That is the whole
API most people ever need.
"""

from __future__ import annotations

from grub.results import SearchResults
from grub.searcher import Searcher

__all__ = ["grub", "search"]


def grub(
    source, query: str | None = None, *, method: str = "tfidf", n: int = 10, **kwargs
):
    """Search *anything*, in one line.

    Point ``grub`` at a source -- a folder, a glob, a Python module, a
    URL, a dict, or a list of strings -- and it builds a search engine.

    Called with just a ``source`` it returns a reusable
    :class:`~grub.searcher.Searcher`:

    >>> docs = {
    ...     'python': 'Python is a popular programming language.',
    ...     'guitar': 'A guitar is a musical instrument with strings.',
    ...     'coffee': 'Coffee is a hot drink brewed from beans.',
    ... }
    >>> searcher = grub(docs)
    >>> searcher('a musical instrument').keys[0]
    'guitar'

    Called with a ``query`` too, it searches immediately and returns the
    :class:`~grub.results.SearchResults`:

    >>> grub(docs, 'hot drink brewed from beans').keys[0]
    'coffee'

    :param source: anything :func:`grub.sources.to_store` understands.
        An existing :class:`~grub.searcher.Searcher` is reused as-is.
    :param query: if given, search for it and return results; otherwise
        return the searcher.
    :param method: ``'tfidf'``, ``'semantic'``, or ``'hybrid'``.
    :param n: number of results.
    :param kwargs: forwarded to :class:`~grub.searcher.Searcher`
        (e.g. ``chunk=1500``, ``extensions=['.py']``, ``embed=...``).
    """
    if isinstance(source, Searcher):
        searcher = source
    else:
        searcher = Searcher(source, method=method, n_results=n, **kwargs)
    if query is None:
        return searcher
    return searcher.search(query, n=n)


def search(
    source, query: str, *, method: str = "tfidf", n: int = 10, **kwargs
) -> SearchResults:
    """Search ``source`` for ``query`` -- a query-first alias of :func:`grub`.

    >>> search(['apple pie', 'car engine', 'apple orchard'], 'fruit apple').keys[0]
    '0'
    """
    return grub(source, query, method=method, n=n, **kwargs)
