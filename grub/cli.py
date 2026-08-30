"""Command-line interface for grub: ``grub SOURCE [QUERY ...]``.

Search a folder, glob, module, or URL straight from the shell::

    grub ./docs "how do I configure logging"
    grub ./src --extensions .py --snippets "retry with backoff"
    grub https://example.com            # no query -> interactive prompt

Run ``grub --help`` for the full option list.
"""

from __future__ import annotations

import sys

import cw

from grub.searcher import Searcher

__all__ = ["run", "main"]


def run(
    source: str,
    *query: str,
    method: str = "tfidf",
    n: int = 10,
    snippets: bool = False,
    semantic: bool = False,
    hybrid: bool = False,
    chunk: int = 0,
    extensions: str = "",
):
    """Search ``source``; with no query, drop into an interactive prompt.

    :param source: folder, glob, ``dir/{}.ext`` format, module path, or URL.
    :param query: the search query (omit for an interactive session).
    :param method: ``tfidf``, ``semantic``, or ``hybrid``.
    :param n: number of results to show.
    :param snippets: also show the matching snippet for each result.
    :param semantic: shortcut for ``--method semantic``.
    :param hybrid: shortcut for ``--method hybrid``.
    :param chunk: split documents into windows of this many characters.
    :param extensions: comma-separated file extensions to restrict to,
        e.g. ``.py,.md``.
    """
    if semantic:
        method = "semantic"
    if hybrid:
        method = "hybrid"
    kwargs = {}
    if chunk:
        kwargs["chunk"] = chunk
    if extensions:
        kwargs["extensions"] = [e.strip() for e in extensions.split(",") if e.strip()]

    try:
        searcher = Searcher(source, method=method, n_results=n, **kwargs)
    except Exception as error:  # noqa: BLE001 - surface a clean CLI message
        print(f"grub: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if query:
        _print_results(searcher, " ".join(query), snippets=snippets)
    else:
        _repl(searcher, snippets=snippets)


def _print_results(searcher: Searcher, query: str, *, snippets: bool) -> None:
    results = searcher.search(query)
    if not results:
        print(f"No results for {query!r}.")
    else:
        print(results.show(snippets=snippets))


def _repl(searcher: Searcher, *, snippets: bool) -> None:
    print(f"{searcher}  --  type a query (blank line or Ctrl-D to quit)")
    while True:
        try:
            query = input("grub> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            return
        _print_results(searcher, query, snippets=snippets)


def main(argv=None) -> int:
    """Entry point for the ``grub`` console script; returns its exit code.

    ``argv`` defaults to ``sys.argv[1:]``. :func:`cw.dispatch` takes it positionally and
    reproduces the grammar this CLI has always had -- pinned by ``misc/cli_golden_py*.json``
    and asserted by ``tests/test_cli_parity.py``.
    """
    return cw.dispatch(run, argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
