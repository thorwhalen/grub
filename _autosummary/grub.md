# grub

grub – a ridiculously simple search engine factory.

Point grub at *anything* – a folder, a Python package, a website, a
dict, a list of strings – and get back a search engine in one line:

```default
from grub import grub

search = grub('./my_docs')              # build a searcher
results = search('how do I configure logging')

for hit in results:
    print(hit.score, hit.key, hit.snippet)
```

Or search in a single call:

```default
grub('./my_docs', 'how do I configure logging')
```

Three search methods are available: `'tfidf'` (lexical, the default),
`'semantic'` (embeddings – `pip install grub[semantic]`), and
`'hybrid'` (a blend of both).

### Functions

| [`grub`](#grub.grub)(source[, query, method, n])                   | Search *anything*, in one line.                                                                                |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [`search`](#grub.search)(source, query, \*[, method, n])             | Search `source` for `query` -- a query-first alias of [`grub()`](#module-grub). |
| [`to_store`](#grub.to_store)(source, \*[, recursive, extensions, ...]) | Coerce `source` into a `Mapping[str, str]` ready to be searched.                                               |
| [`chunk_text`](#grub.chunk_text)(text, \*[, size, overlap])              | Split `text` into overlapping character windows.                                                               |
| [`chunk_store`](#grub.chunk_store)(store, \*[, size, overlap])            | Chunk every value of `store`; keys become `"{key}#{n}"`.                                                       |
| [`html_to_text`](#grub.html_to_text)(html)                                 | Strip HTML tags, returning visible text only.                                                                  |
| [`make_backend`](#grub.make_backend)([method, tokenizer, embed, ...])      | Build a backend by name: `'tfidf'`, `'semantic'`, or `'hybrid'`.                                               |
| [`grubber`](#grub.grubber)(search_store[, n])                         | Computes a TF-IDF model and a KNN model for a search store.                                                    |
| [`camelcase_and_underscore_tokenizer`](#grub.camelcase_and_underscore_tokenizer)(string)         |                                                                                                                |
| [`search_documented_attributes`](#grub.search_documented_attributes)(obj[, ...])           | Search the documented attributes of a python object                                                            |

### Classes

| [`Searcher`](#grub.Searcher)(source, \*[, method, n_results, ...])    | Index any source and search it.                                                                     |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`SearchResult`](#grub.SearchResult)(key, score[, snippet])               | A single hit.                                                                                       |
| [`SearchResults`](#grub.SearchResults)([results, query])                   | An ordered list of [`SearchResult`](#grub.SearchResult), best match first. |
| [`TfidfBackend`](#grub.TfidfBackend)(\*[, tokenizer])                     | Lexical search via TF-IDF vectors and cosine similarity.                                            |
| [`SemanticBackend`](#grub.SemanticBackend)(\*[, embed, model])               | Dense-embedding search: matches meaning, not just shared words.                                     |
| [`HybridBackend`](#grub.HybridBackend)(\*[, alpha, tokenizer, embed, ...]) | A weighted blend of lexical (TF-IDF) and semantic search.                                           |
| [`SearchStore`](#grub.SearchStore)(store[, n_neighbors, tokenizer])      | Build a search index for anything (that is given a mapping interface with string values).           |
| [`CodeSearcher`](#grub.CodeSearcher)([search_store, tfidf, knn])          | Text searcher with fitters                                                                          |
| [`TextFilesSearcher`](#grub.TextFilesSearcher)([search_store, tfidf, knn])     | Fittable Text searcher with fitters                                                                 |
| [`TfidfKnnSearcher`](#grub.TfidfKnnSearcher)([search_store, tfidf, knn])      |                                                                                                     |

### *class* grub.CodeSearcher(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`CodeSearcherBase`](grub.base.md#grub.base.CodeSearcherBase)

Text searcher with fitters

### *class* grub.HybridBackend(\*, alpha=0.5, tokenizer=<function camelcase_and_underscore_tokenizer>, embed=None, model='all-MiniLM-L6-v2')

Bases: [`Backend`](grub.backends.md#grub.backends.Backend)

A weighted blend of lexical (TF-IDF) and semantic search.

`alpha` is the lexical weight: `1.0` is pure TF-IDF, `0.0` is
pure semantic, `0.5` (default) gives an even mix.

### *class* grub.SearchResult(key, score, snippet='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A single hit.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the document’s key in the store.
  * **score** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – relevance in `[0, 1]` (higher is better).
  * **snippet** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – a short excerpt showing why the document matched.

### *class* grub.SearchResults(results=(), , query=None)

Bases: [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

An ordered list of [`SearchResult`](#grub.SearchResult), best match first.

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

### *class* grub.SearchStore(store, n_neighbors=10, tokenizer=<function camelcase_and_underscore_tokenizer>)

Bases: [`StoreMixin`](grub.base.md#grub.base.StoreMixin)

Build a search index for anything (that is given a mapping interface with string values).

A store is anything with a `collections.Mapping` interface.
Typically, a store’s backend comes from local files or data-base wrapped into a mapping
(see `py2store` for tools to do that!).
For testing purposes though, we’ll use a `dict` here:

```pycon
>>> store = {
... "Nelson Mandela": "The greatest glory in living lies not in never falling, but in rising every time we fall.",
... "Walt Disney": "The way to get started is to quit talking and begin doing.",
... "Steve Jobs": "Your time is limited, so don't waste it living someone else's life."
... }
>>> search = SearchStore(store, n_neighbors=2)  # our store is small, so need to restrict our result size to less
>>> list(search('living'))
['Steve Jobs', 'Nelson Mandela']
```

A `SearchStore` instance is picklable.

```pycon
>>> import pickle
>>> unserialized_search = pickle.loads(pickle.dumps(search))
>>> list(unserialized_search('living'))
['Steve Jobs', 'Nelson Mandela']
```

### *class* grub.Searcher(source, \*, method='tfidf', n_results=10, tokenizer=<function camelcase_and_underscore_tokenizer>, embed=None, model='all-MiniLM-L6-v2', alpha=0.5, chunk=None, snippets=True, backend=None, \*\*source_kwargs)

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
  [`Searcher`](grub.searcher.md#grub.searcher.Searcher)

#### *property* is_fitted *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the index has been built.

#### *classmethod* load(path)

Load a searcher previously written by [`save()`](#grub.Searcher.save).

* **Return type:**
  [`Searcher`](grub.searcher.md#grub.searcher.Searcher)

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

### *class* grub.SemanticBackend(, embed=None, model='all-MiniLM-L6-v2')

Bases: [`Backend`](grub.backends.md#grub.backends.Backend)

Dense-embedding search: matches meaning, not just shared words.

Supply your own `embed` callable (`list[str] -> 2D array`) to use
any provider, or leave it `None` to lazily load a local
`sentence-transformers` model.

#### embed(texts)

Embed `texts` into an L2-normalised matrix.

### *class* grub.TextFilesSearcher(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`TextFilesSearcherBase`](grub.base.md#grub.base.TextFilesSearcherBase)

Fittable Text searcher with fitters

### *class* grub.TfidfBackend(\*, tokenizer=<function camelcase_and_underscore_tokenizer>, \*\*tfidf_kwargs)

Bases: [`Backend`](grub.backends.md#grub.backends.Backend)

Lexical search via TF-IDF vectors and cosine similarity.

Fast, dependency-light, and excellent when the query shares actual
words with the documents – the right default for code and docs.

### *class* grub.TfidfKnnSearcher(search_store=<factory>, tfidf=<factory>, knn=<factory>)

Bases: [`SearchStoreMixin`](grub.base.md#grub.base.SearchStoreMixin), `TfidfKnnFitMixin`

#### fvs(error_callback=None)

The feature vectors for the search_store

### grub.camelcase_and_underscore_tokenizer(string)

```pycon
>>> print(*camelcase_and_underscore_tokenizer('thereIsSomething i_wanted toShow'))
there is something i wanted to show
```

### grub.chunk_store(store, , size=1500, overlap=200)

Chunk every value of `store`; keys become `"{key}#{n}"`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> sorted(chunk_store({"doc": "abcdef"}, size=3, overlap=0))
['doc#0', 'doc#1']
```

### grub.chunk_text(text, , size=1500, overlap=200)

Split `text` into overlapping character windows.

Long documents search better in pieces: a hit then points at a
passage instead of a whole file.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> chunk_text("abcdefghij", size=4, overlap=1)
['abcd', 'defg', 'ghij', 'j']
>>> chunk_text("short", size=100)
['short']
```

### grub.grub(source, query=None, , method='tfidf', n=10, \*\*kwargs)

Search *anything*, in one line.

Point `grub` at a source – a folder, a glob, a Python module, a
URL, a dict, or a list of strings – and it builds a search engine.

Called with just a `source` it returns a reusable
[`Searcher`](grub.searcher.md#grub.searcher.Searcher):

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
[`SearchResults`](grub.results.md#grub.results.SearchResults):

```pycon
>>> grub(docs, 'hot drink brewed from beans').keys[0]
'coffee'
```

* **Parameters:**
  * **source** – anything [`grub.sources.to_store()`](grub.sources.md#grub.sources.to_store) understands.
    An existing [`Searcher`](grub.searcher.md#grub.searcher.Searcher) is reused as-is.
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – if given, search for it and return results; otherwise
    return the searcher.
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `'tfidf'`, `'semantic'`, or `'hybrid'`.
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of results.
  * **kwargs** – forwarded to [`Searcher`](grub.searcher.md#grub.searcher.Searcher)
    (e.g. `chunk=1500`, `extensions=['.py']`, `embed=...`).

### grub.grubber(search_store, n=10)

Computes a TF-IDF model and a KNN model for a search store.

### grub.html_to_text(html)

Strip HTML tags, returning visible text only.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> html_to_text("<h1>Title</h1><p>Hello <b>world</b></p>")
'Title Hello world'
>>> html_to_text("<style>.x{}</style><p>visible</p>")
'visible'
```

### grub.make_backend(method='tfidf', \*, tokenizer=<function camelcase_and_underscore_tokenizer>, embed=None, model='all-MiniLM-L6-v2', alpha=0.5)

Build a backend by name: `'tfidf'`, `'semantic'`, or `'hybrid'`.

* **Return type:**
  [`Backend`](grub.backends.md#grub.backends.Backend)

```pycon
>>> make_backend("tfidf").method
'tfidf'
```

### grub.search(source, query, , method='tfidf', n=10, \*\*kwargs)

Search `source` for `query` – a query-first alias of [`grub()`](#module-grub).

* **Return type:**
  [`SearchResults`](grub.results.md#grub.results.SearchResults)

```pycon
>>> search(['apple pie', 'car engine', 'apple orchard'], 'fruit apple').keys[0]
'0'
```

### grub.search_documented_attributes(obj, obj_to_attrs=<function ddir>, max_results=10)

Search the documented attributes of a python object

* **Parameters:**
  * **obj** – Any python object
  * **obj_to_attrs** – The function that gives you attribute names of an object.
* **Returns:**
  A SearchStore instance to search attributes via their docs

```pycon
>>> from inspect import getmodule
>>> containing_module = getmodule(search_documented_attributes)  # the module of this function
>>> search_module = search_documented_attributes(containing_module, max_results=3)
>>> list(search_module('documented attributes'))  # if you get an error here, it's probably just be that the docs changed
['search_documented_attributes', 'ddir', 'inspect']
>>> list(search_module('documented objects'))  # if you get an error here, it's probably just be that the docs changed
['search_documented_attributes', 'DocTestFinder', 'doctest_finder']
```

### grub.to_store(source, , recursive=True, extensions=None, chunk=None, encoding='utf-8', max_bytes=1000000)

Coerce `source` into a `Mapping[str, str]` ready to be searched.

`source` may be:

* a `Mapping` – used as-is
* a folder path – every (decodable) text file under it
* a glob pattern – every matching file
* a `"path/{}.ext"` format – a lazy `py2store` text store
* a Python module/package – its `.py` source files
* a URL (or list of URLs) – fetched and stripped to text
* a list of strings – enumerated into `{"0": ..., "1": ...}`
* an iterable of `(key, value)` pairs – turned into a dict

```pycon
>>> store = to_store({"a": "first doc", "b": "second doc"})
>>> dict(store)
{'a': 'first doc', 'b': 'second doc'}
>>> sorted(to_store(["red", "green", "blue"]))
['0', '1', '2']
```

* **Parameters:**
  * **recursive** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – walk sub-directories (folder sources only).
  * **extensions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – keep only files with these extensions, e.g. `[".py"]`.
  * **chunk** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – split values into overlapping windows – an int size or a
    `(size, overlap)` pair. Handy for long documents.
  * **encoding** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – text encoding for files.
  * **max_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – skip files larger than this (folder/glob sources).
* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

### Modules

| [`backends`](grub.backends.md#module-grub.backends)   | Search backends: the engines that score documents against a query.                                                                                                  |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`base`](grub.base.md#module-grub.base)           | Base functionality                                                                                                                                                  |
| [`cli`](grub.cli.md#module-grub.cli)             | Command-line interface for grub: `grub SOURCE [QUERY ...]`.                                                                                                         |
| [`facade`](grub.facade.md#module-grub.facade)       | The one-line front door to grub: [`grub()`](#module-grub) (and its alias [`search()`](#grub.search)). |
| [`pycode`](grub.pycode.md#module-grub.pycode)       | Searching python code                                                                                                                                               |
| [`results`](grub.results.md#module-grub.results)     | Rich search results: keys, scores, and snippets that explain a match.                                                                                               |
| [`searcher`](grub.searcher.md#module-grub.searcher)   | The [`Searcher`](#grub.Searcher) -- grub's unified, modern search object.                                                              |
| [`sources`](grub.sources.md#module-grub.sources)     | Source adapters: turn *anything* into a searchable store.                                                                                                           |
| [`util`](grub.util.md#module-grub.util)           | Small shared utilities for grub: tokenizing, clamping, snippet extraction.                                                                                          |
