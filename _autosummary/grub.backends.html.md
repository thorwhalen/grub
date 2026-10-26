# grub.backends

Search backends: the engines that score documents against a query.

Every backend implements the same tiny interface:

* `index(docs)` – build an index over a list of text documents.
* `scores(query)` – return a relevance score per document.
* `query(query, n)` – return the top-`n` `(indices, scores)`.

Three backends ship with grub:

* [`TfidfBackend`](#grub.backends.TfidfBackend) – lexical TF-IDF + cosine similarity (default,
  no extra dependencies).
* [`SemanticBackend`](#grub.backends.SemanticBackend) – dense embeddings; understands meaning, not
  just words. Needs `sentence-transformers` (`pip install grub[semantic]`)
  or any embedding function you supply.
* [`HybridBackend`](#grub.backends.HybridBackend) – a weighted blend of the two.

Pick a backend by name with [`make_backend()`](#grub.backends.make_backend).

### Module Attributes

| [`DFLT_SEMANTIC_MODEL`](#grub.backends.DFLT_SEMANTIC_MODEL)   | Default sentence-transformers model for [`SemanticBackend`](#grub.backends.SemanticBackend).   |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|

### Functions

| [`make_backend`](#grub.backends.make_backend)([method, tokenizer, embed, ...])   | Build a backend by name: `'tfidf'`, `'semantic'`, or `'hybrid'`.   |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|

### Classes

| [`Backend`](#grub.backends.Backend)()                                         | Base backend: turns per-document `scores` into a ranked `query`.   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`TfidfBackend`](#grub.backends.TfidfBackend)(\*[, tokenizer])                     | Lexical search via TF-IDF vectors and cosine similarity.           |
| [`SemanticBackend`](#grub.backends.SemanticBackend)(\*[, embed, model])               | Dense-embedding search: matches meaning, not just shared words.    |
| [`HybridBackend`](#grub.backends.HybridBackend)(\*[, alpha, tokenizer, embed, ...]) | A weighted blend of lexical (TF-IDF) and semantic search.          |

### *class* grub.backends.Backend

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base backend: turns per-document `scores` into a ranked `query`.

#### query(query, n)

Top-`n` `(indices, scores)` for `query`, best first.

### grub.backends.DFLT_SEMANTIC_MODEL *= 'all-MiniLM-L6-v2'*

Default sentence-transformers model for [`SemanticBackend`](#grub.backends.SemanticBackend).

### *class* grub.backends.HybridBackend(\*, alpha=0.5, tokenizer=<function camelcase_and_underscore_tokenizer>, embed=None, model='all-MiniLM-L6-v2')

Bases: [`Backend`](#grub.backends.Backend)

A weighted blend of lexical (TF-IDF) and semantic search.

`alpha` is the lexical weight: `1.0` is pure TF-IDF, `0.0` is
pure semantic, `0.5` (default) gives an even mix.

### *class* grub.backends.SemanticBackend(, embed=None, model='all-MiniLM-L6-v2')

Bases: [`Backend`](#grub.backends.Backend)

Dense-embedding search: matches meaning, not just shared words.

Supply your own `embed` callable (`list[str] -> 2D array`) to use
any provider, or leave it `None` to lazily load a local
`sentence-transformers` model.

#### embed(texts)

Embed `texts` into an L2-normalised matrix.

### *class* grub.backends.TfidfBackend(\*, tokenizer=<function camelcase_and_underscore_tokenizer>, \*\*tfidf_kwargs)

Bases: [`Backend`](#grub.backends.Backend)

Lexical search via TF-IDF vectors and cosine similarity.

Fast, dependency-light, and excellent when the query shares actual
words with the documents – the right default for code and docs.

### grub.backends.make_backend(method='tfidf', \*, tokenizer=<function camelcase_and_underscore_tokenizer>, embed=None, model='all-MiniLM-L6-v2', alpha=0.5)

Build a backend by name: `'tfidf'`, `'semantic'`, or `'hybrid'`.

```pycon
>>> make_backend("tfidf").method
'tfidf'
```

* **Return type:**
  [`Backend`](#grub.backends.Backend)
