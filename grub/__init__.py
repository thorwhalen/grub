"""grub -- a ridiculously simple search engine factory.

Point grub at *anything* -- a folder, a Python package, a website, a
dict, a list of strings -- and get back a search engine in one line::

    from grub import grub

    search = grub('./my_docs')              # build a searcher
    results = search('how do I configure logging')

    for hit in results:
        print(hit.score, hit.key, hit.snippet)

Or search in a single call::

    grub('./my_docs', 'how do I configure logging')

Three search methods are available: ``'tfidf'`` (lexical, the default),
``'semantic'`` (embeddings -- ``pip install grub[semantic]``), and
``'hybrid'`` (a blend of both).
"""

from grub.base import (
    SearchStore,
    CodeSearcher,
    TfidfKnnSearcher,
    TextFilesSearcher,
    grubber,
    camelcase_and_underscore_tokenizer,
)
from grub.facade import grub, search
from grub.searcher import Searcher
from grub.results import SearchResult, SearchResults
from grub.sources import to_store, chunk_text, chunk_store, html_to_text
from grub.backends import (
    make_backend,
    TfidfBackend,
    SemanticBackend,
    HybridBackend,
)
from grub.pycode import search_documented_attributes

__all__ = [
    # modern API
    "grub",
    "search",
    "Searcher",
    "SearchResult",
    "SearchResults",
    "to_store",
    "chunk_text",
    "chunk_store",
    "html_to_text",
    "make_backend",
    "TfidfBackend",
    "SemanticBackend",
    "HybridBackend",
    # legacy API (kept for backward compatibility)
    "SearchStore",
    "CodeSearcher",
    "TextFilesSearcher",
    "TfidfKnnSearcher",
    "grubber",
    "camelcase_and_underscore_tokenizer",
    "search_documented_attributes",
]
