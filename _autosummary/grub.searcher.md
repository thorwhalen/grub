# grub.searcher

The [`Searcher`](#grub.searcher.Searcher) – grub’s unified, modern search object.

A [`Searcher`](#grub.searcher.Searcher) wraps three things behind one friendly interface:

* a **store** (any source, via [`grub.sources.to_store()`](grub.sources.md#grub.sources.to_store)),
* a **backend** (TF-IDF, semantic, or hybrid, via [`grub.backends`](grub.backends.md#module-grub.backends)),
* **results** enriched with scores and snippets ([`grub.results`](grub.results.md#module-grub.results)).

You rarely build one by hand – [`grub.grub()`](grub.md#grub.grub) is the front door –
but the class is here when you want full control, persistence, or to
plug in your own backend.

### Classes

| [`Searcher`](#grub.searcher.Searcher)(source, \*[, method, n_results, ...])   | Index any source and search it.   |
|---------------------------------------------------------------------------------------------------|-----------------------------------|

### *class* grub.searcher.Searcher(source, \*, method='tfidf', n_results=10, tokenizer=<function camelcase_and_underscore_tokenizer>, embed=None, model='all-MiniLM-L6-v2', alpha=0.5, chunk=None, snippets=True, backend=None, \*\*source_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Index any source and search it.

```pycon
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
```

Each result carries a score and a snippet explaining the match:

```pycon
>>> hit = results[0]
>>> hit.key
'guitar'
>>> hit.score > 0
True
>>> 'musical instrument' in hit.snippet
True
```

#### fit()

Build the search index. Called automatically on first search.

* **Return type:**
  [`Searcher`](#grub.searcher.Searcher)

#### *property* is_fitted *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the index has been built.

#### *classmethod* load(path)

Load a searcher previously written by [`save()`](#grub.searcher.Searcher.save).

* **Return type:**
  [`Searcher`](#grub.searcher.Searcher)

#### save(path)

Pickle this searcher (index included) to `path`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  the path written to.

#### search(query, n=None)

Return the [`SearchResults`](grub.results.md#grub.results.SearchResults) for `query`.

* **Return type:**
  [`SearchResults`](grub.results.md#grub.results.SearchResults)
