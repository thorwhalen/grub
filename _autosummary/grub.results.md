# grub.results

Rich search results: keys, scores, and snippets that explain a match.

Where the legacy [`grub.SearchStore`](grub.md#grub.SearchStore) returns a bare array of keys,
the modern [`grub.Searcher`](grub.md#grub.Searcher) returns a [`SearchResults`](#grub.results.SearchResults) – a
list of [`SearchResult`](#grub.results.SearchResult) records that also know their score and a
snippet showing *why* the document matched.

### Classes

| [`SearchResult`](#grub.results.SearchResult)(key, score[, snippet])   | A single hit.                                                                                       |
|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`SearchResults`](#grub.results.SearchResults)([results, query])       | An ordered list of [`SearchResult`](#grub.results.SearchResult), best match first. |

### *class* grub.results.SearchResult(key, score, snippet='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A single hit.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the document’s key in the store.
  * **score** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – relevance in `[0, 1]` (higher is better).
  * **snippet** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – a short excerpt showing why the document matched.

### *class* grub.results.SearchResults(results=(), , query=None)

Bases: [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

An ordered list of [`SearchResult`](#grub.results.SearchResult), best match first.

It *is* a list, so it indexes, slices, and iterates as you’d expect:

```default
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
```

#### as_dict()

A `{key: score}` mapping of the results.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### *property* keys *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)*

Just the keys, ranked best-first.

#### *property* scores *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)*

Just the scores, ranked best-first.

#### show(, snippets=True)

A human-readable, multi-line rendering of the results.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### *property* snippets *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)*

Just the snippets, ranked best-first.
