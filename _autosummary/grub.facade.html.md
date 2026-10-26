# grub.facade

The one-line front door to grub: [`grub()`](grub.html.md#module-grub) (and its alias [`search()`](#grub.facade.search)).

`grub(source)` builds a reusable [`Searcher`](grub.searcher.html.md#grub.searcher.Searcher);
`grub(source, query)` searches in a single call. That is the whole
API most people ever need.

### Functions

| [`grub`](#grub.facade.grub)(source[, query, method, n])       | Search *anything*, in one line.                                                                                            |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| [`search`](#grub.facade.search)(source, query, \*[, method, n]) | Search `source` for `query` -- a query-first alias of [`grub()`](grub.html.md#module-grub). |

### grub.facade.grub(source, query=None, , method='tfidf', n=10, \*\*kwargs)

Search *anything*, in one line.

Point `grub` at a source – a folder, a glob, a Python module, a
URL, a dict, or a list of strings – and it builds a search engine.

Called with just a `source` it returns a reusable
[`Searcher`](grub.searcher.html.md#grub.searcher.Searcher):

```pycon
>>> docs = {
...     'python': 'Python is a popular programming language.',
...     'guitar': 'A guitar is a musical instrument with strings.',
...     'coffee': 'Coffee is a hot drink brewed from beans.',
... }
>>> searcher = grub(docs)
>>> searcher('a musical instrument').keys[0]
'guitar'
```

Called with a `query` too, it searches immediately and returns the
[`SearchResults`](grub.results.html.md#grub.results.SearchResults):

```pycon
>>> grub(docs, 'hot drink brewed from beans').keys[0]
'coffee'
```

* **Parameters:**
  * **source** – anything [`grub.sources.to_store()`](grub.sources.html.md#grub.sources.to_store) understands.
    An existing [`Searcher`](grub.searcher.html.md#grub.searcher.Searcher) is reused as-is.
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – if given, search for it and return results; otherwise
    return the searcher.
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `'tfidf'`, `'semantic'`, or `'hybrid'`.
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of results.
  * **kwargs** – forwarded to [`Searcher`](grub.searcher.html.md#grub.searcher.Searcher)
    (e.g. `chunk=1500`, `extensions=['.py']`, `embed=...`).

### grub.facade.search(source, query, , method='tfidf', n=10, \*\*kwargs)

Search `source` for `query` – a query-first alias of [`grub()`](grub.html.md#module-grub).

```pycon
>>> search(['apple pie', 'car engine', 'apple orchard'], 'fruit apple').keys[0]
'0'
```

* **Return type:**
  [`SearchResults`](grub.results.html.md#grub.results.SearchResults)
