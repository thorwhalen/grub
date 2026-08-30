---
name: grub-search
description: Use when searching a body of text with grub — pointing it at a folder, a codebase, a Python module/package, a set of docs, a website, or a list of strings, then querying it. Triggers on "search my docs/code/notes", "find where X is mentioned", "build a search index", "which file talks about Y", "semantic search over these files", "grub search", or running the grub CLI.
---

# grub-search — search anything with grub

`grub` is a search-engine factory. Point it at a source, get back a
searcher, ask questions. One import, one line.

```python
from grub import grub
```

## The one function you need: `grub()`

```python
grub(source)  # -> a reusable Searcher
grub(source, query)  # -> SearchResults, searched immediately
```

`source` can be any of:

| Source | Example |
|---|---|
| a folder | `grub('./docs')` |
| a glob | `grub('src/**/*.py')` |
| a `dir/{}.ext` format | `grub('notes/{}.md')` |
| a Python module/package | `import requests; grub(requests)` |
| a URL (HTML is stripped to text) | `grub('https://example.com/guide')` |
| a list of URLs | `grub(['https://a.com', 'https://b.com'])` |
| a dict `{key: text}` | `grub({'intro': '...', 'faq': '...'})` |
| a list of strings | `grub(['first doc', 'second doc'])` |

Folder walks skip binary files and noise dirs (`.git`, `node_modules`,
`__pycache__`, ...) automatically.

## Searching and reading results

```python
search = grub("./docs")
results = search("how do I configure logging")

for hit in results:
    print(hit.score, hit.key, hit.snippet)

results.keys  # ['logging.md', 'setup.md', ...]  ranked best-first
results.scores  # [0.71, 0.33, ...]
results.as_dict()  # {'logging.md': 0.71, ...}
print(results.show())  # pretty multi-line rendering
print(search["logging.md"])  # the full original text of a hit
```

Each `SearchResult` has `.key`, `.score` (0-1, higher is better), and
`.snippet` — the line that best explains *why* the document matched.

## Choosing a search method

```python
grub(source, query, method="tfidf")  # lexical — shared words (default)
grub(source, query, method="semantic")  # embeddings — shared *meaning*
grub(source, query, method="hybrid")  # a blend of both
```

- **tfidf** — fast, no extra install, best when the query uses the same
  words as the documents (great for code).
- **semantic** — finds matches that share *meaning* even with no words in
  common ("car" finds "automobile"). Needs `pip install 'grub[semantic]'`
  (downloads a small sentence-transformers model on first use), **or**
  pass your own `embed=` function — see the `grub-extend` skill.
- **hybrid** — `method='hybrid'`, optional `alpha=` (lexical weight, 0-1).

## Useful options (keyword args to `grub()`)

```python
grub("./src", extensions=[".py"])  # only these file types
grub("./book.txt", chunk=1500)  # split long docs into 1500-char windows
grub("./docs", n=20)  # number of results
grub("./docs", recursive=False)  # don't descend into sub-folders
grub("./docs", snippets=False)  # skip snippet extraction (faster)
```

`chunk=` is important for **long documents**: it splits each document so a
hit points at the relevant passage (key becomes `"file.md#3"`) instead of
a whole file.

## Save and reuse an index

Building the index is the slow part — do it once, reuse it.

```python
from grub import Searcher

grub("./big_codebase").save("code.grub")  # build + persist
search = Searcher.load("code.grub")  # instant reload
search("database connection retry")
```

## Command line

Installing grub also installs a `grub` command (or use `python -m grub`):

```bash
grub ./docs "how do I configure logging"
grub ./src --extensions .py --snippets "retry with backoff"
grub ./book.txt --chunk 1500 "the protagonist's motivation"
grub https://example.com/guide --semantic "getting started"
grub ./docs                       # no query -> interactive prompt
```

Flags: `--method`, `-n`, `--snippets`, `--semantic`, `--hybrid`,
`--chunk`, `--extensions`. Run `grub --help` for details.

## Recipes

**Search a codebase for a concept:**
```python
grub("./myproject", extensions=[".py"], query="where is auth handled")
```

**Find which doc answers a question:**
```python
grub("./docs", "can I use this offline").keys[0]
```

**Search a long PDF/book (after extracting its text):**
```python
grub({"book": long_text}, chunk=2000)("what happens in chapter 3")
```

**Search live web pages:**
```python
grub(["https://site.com/a", "https://site.com/b"], "pricing tiers")
```

## Gotchas

- The first search on a new `Searcher` builds the index lazily — expect a
  small one-time cost, then searches are fast.
- `method='semantic'` without `sentence-transformers` installed and
  without an `embed=` function raises a clear `ImportError`.
- Scores are comparable *within* one result set, not across methods.
- For very large corpora, `save()`/`load()` the index instead of
  rebuilding it every run.
