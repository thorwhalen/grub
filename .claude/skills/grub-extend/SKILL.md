---
name: grub-extend
description: Use when extending or customizing grub — plugging in a custom embedding function or provider (OpenAI, Cohere, Voyage, a local model), writing a new search backend, adding a source adapter, or tuning the tokenizer / TF-IDF parameters. Triggers on "use OpenAI embeddings with grub", "custom embeddings for grub", "add a backend to grub", "grub custom source", "make grub search <some new thing>".
---

# grub-extend — customize and extend grub

grub is built from small, swappable parts. You can replace any of them
without touching the rest.

```
source  --to_store-->  store  --backend-->  scores  -->  SearchResults
```

## Plug in any embedding provider (most common)

`method='semantic'` accepts an `embed` callable so you are **not** tied to
`sentence-transformers`. An embedder is any function
`list[str] -> 2D array` (one row per text). grub L2-normalizes the rows
for you.

**OpenAI embeddings:**
```python
from openai import OpenAI
from grub import grub

client = OpenAI()


def embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


search = grub("./docs", method="semantic", embed=embed)
search("how do I roll back a migration")
```

The same shape works for Cohere, Voyage, a local `sentence-transformers`
model you configured yourself, or a cached/batched wrapper. Keep `embed`
a **module-level function** (not a lambda/closure) if you want the
searcher to `save()`/`load()` cleanly.

## Write a custom backend

A backend implements two methods; `Backend.query()` is inherited.

```python
import numpy as np
from grub.backends import Backend
from grub import Searcher


class KeywordCountBackend(Backend):
    method = "keyword-count"

    def index(self, docs):
        self.docs = [d.lower() for d in docs]
        return self

    def scores(self, query):  # one score per document
        q = query.lower().split()
        return np.array([sum(d.count(w) for w in q) for d in self.docs], dtype=float)


search = Searcher(my_store, backend=KeywordCountBackend())
```

Passing `backend=` overrides `method=`. Built-in backends live in
`grub/backends.py`: `TfidfBackend`, `SemanticBackend`, `HybridBackend`,
and the `make_backend(name, ...)` factory.

## Tune the lexical (TF-IDF) backend

```python
from grub import grub
from grub.util import simple_tokenizer

# A plain word tokenizer instead of the default camelCase-aware one:
grub("./docs", tokenizer=simple_tokenizer)

# Or build the backend yourself to reach sklearn's TfidfVectorizer knobs:
from grub.backends import TfidfBackend
from grub import Searcher

backend = TfidfBackend(
    tokenizer=simple_tokenizer, ngram_range=(1, 2), min_df=2, sublinear_tf=True
)
Searcher(my_store, backend=backend)
```

The default tokenizer (`camelcase_and_underscore_tokenizer`) splits
`fooBar` and `foo_bar` into `foo bar` — ideal for searching source code.

## Add a new source adapter

Everything searchable becomes a `Mapping[str, str]` via
`grub.sources.to_store`. To support a new kind of source, write a function
that returns such a mapping and pass its result straight to `grub()`:

```python
def from_sqlite(db_path, table, key_col, text_col):
    import sqlite3

    con = sqlite3.connect(db_path)
    rows = con.execute(f"SELECT {key_col}, {text_col} FROM {table}")
    return {str(k): str(v) for k, v in rows}


grub(from_sqlite("app.db", "articles", "id", "body"), "search query")
```

Existing adapters in `grub/sources.py` worth reusing: `from_dir`,
`from_files`, `from_urls`, `from_strings`, `from_module`, `chunk_store`,
`html_to_text`.

## Tune the hybrid blend

```python
grub("./docs", method="hybrid", alpha=0.7, embed=my_embed)
# alpha = lexical weight: 1.0 pure TF-IDF, 0.0 pure semantic, 0.5 default
```

## Customize snippets

`grub.util.best_snippet(text, query, *, width=, tokenizer=)` picks the
most relevant line of a document. Use it directly, or set `snippets=False`
on `grub()`/`Searcher` to skip snippet extraction entirely when you only
need keys and scores.

## Where things live

| File | Responsibility |
|---|---|
| `grub/facade.py` | `grub()` / `search()` — the entry point |
| `grub/searcher.py` | `Searcher` — store + backend + results |
| `grub/sources.py` | turn anything into a `Mapping[str, str]` |
| `grub/backends.py` | TF-IDF / semantic / hybrid scoring engines |
| `grub/results.py` | `SearchResult`, `SearchResults` |
| `grub/util.py` | tokenizing, snippets, helpers |
| `grub/base.py` | legacy `SearchStore` API (kept for compatibility) |
