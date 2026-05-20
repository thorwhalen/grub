"""Rich search results: keys, scores, and snippets that explain a match.

Where the legacy :class:`grub.SearchStore` returns a bare array of keys,
the modern :class:`grub.Searcher` returns a :class:`SearchResults` -- a
list of :class:`SearchResult` records that also know their score and a
snippet showing *why* the document matched.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SearchResult", "SearchResults"]


@dataclass(frozen=True)
class SearchResult:
    """A single hit.

    :param key: the document's key in the store.
    :param score: relevance in ``[0, 1]`` (higher is better).
    :param snippet: a short excerpt showing why the document matched.
    """

    key: str
    score: float
    snippet: str = ""

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.key}"


class SearchResults(list):
    """An ordered list of :class:`SearchResult`, best match first.

    It *is* a list, so it indexes, slices, and iterates as you'd expect::

        >>> results = SearchResults(
        ...     [SearchResult("guitar", 0.9, "a string instrument"),
        ...      SearchResult("piano", 0.4, "a keyboard instrument")],
        ...     query="instrument",
        ... )
        >>> len(results)
        2
        >>> results[0].key
        'guitar'
        >>> results.keys
        ['guitar', 'piano']
        >>> [round(s, 1) for s in results.scores]
        [0.9, 0.4]
        >>> results.as_dict()['guitar']
        0.9
    """

    def __init__(self, results=(), *, query: str | None = None):
        super().__init__(results)
        self.query = query

    @property
    def keys(self) -> list:
        """Just the keys, ranked best-first."""
        return [r.key for r in self]

    @property
    def scores(self) -> list:
        """Just the scores, ranked best-first."""
        return [r.score for r in self]

    @property
    def snippets(self) -> list:
        """Just the snippets, ranked best-first."""
        return [r.snippet for r in self]

    def as_dict(self) -> dict:
        """A ``{key: score}`` mapping of the results."""
        return {r.key: r.score for r in self}

    def show(self, *, snippets: bool = True) -> str:
        """A human-readable, multi-line rendering of the results."""
        if not self:
            return f"No results for {self.query!r}."
        lines = [f"{len(self)} results for {self.query!r}:"]
        for rank, r in enumerate(self, 1):
            lines.append(f"  {rank:>2}. [{r.score:.3f}] {r.key}")
            if snippets and r.snippet:
                lines.append(f"      {r.snippet}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if not self:
            return f"SearchResults(query={self.query!r}, [])"
        head = f"SearchResults for {self.query!r} -- {len(self)} hits:"
        body = "\n".join(
            f"  {rank:>2}. [{r.score:.3f}] {r.key}" for rank, r in enumerate(self, 1)
        )
        return f"{head}\n{body}"
