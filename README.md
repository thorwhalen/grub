# grub

**A ridiculously simple search engine factory.**

Point `grub` at *anything* — a folder, a codebase, a Python package, a
website, a pile of notes — and get back a working search engine in one
line. No servers, no indexes to babysit, no configuration.

```bash
pip install grub
```

```python
from grub import grub

grub('./my_notes', 'where did I write about retirement savings')
```

---

## The AI-first way (start here)

You probably shouldn't be calling `grub` yourself at all.

grub ships with **agent skills** — instruction files that teach an AI
coding agent (Claude Code, Cursor, and friends) how to drive grub on your
behalf. The skills live in [`.claude/skills/`](.claude/skills/):

| Skill | What it lets the agent do |
|---|---|
| **`grub-search`** | Build a search index over a folder, codebase, module, website, or list of strings, and answer questions against it. |
| **`grub-extend`** | Wire grub up to custom embedding providers (OpenAI, Cohere, …), new backends, or new data sources. |

With the skills in place, you stop writing code and start *asking*:

> "Search my `./docs` folder and tell me which file explains the
> deployment process."

> "Index this codebase and find where rate limiting is implemented."

> "Use semantic search over my meeting notes to find anything about the
> Q3 budget."

> "Search these three documentation URLs and summarize what they say
> about authentication."

The agent reads the skill, picks the right source type, the right search
method (lexical, semantic, or hybrid), chunks long documents when it
helps, and hands you the answer. You never see a `SearchStore`
constructor. You never tune a vectorizer. You describe the outcome you
want, in English, and it happens.

### Why AI-first?

Because the interface to software is changing, and grub is built for the
change.

For decades, "using a tool" meant *learning the tool* — its API, its
flags, its mental model — and then translating your intent into its
vocabulary. That translation tax was unavoidable. It is not anymore.

An AI agent is a universal adapter between human intent and machine
capability. It already knows grub's vocabulary; you don't have to. So the
job of a well-designed library is no longer "expose a clever API to
humans" — it's **"expose powerful, composable capabilities, and ship the
knowledge an agent needs to wield them."** That knowledge is the skill
files.

grub leans all the way into this:

- **The skills are the primary interface.** They are documentation an
  agent *executes*, not documentation a human *reads and then forgets*.
- **The Python API is the substrate.** It stays clean, small, and
  honest — because an agent calling it deserves the same good design a
  human would.
- **You operate at the level of intent.** "Find the doc about X" instead
  of "instantiate, configure, fit, query, parse."

The future of tooling is not humans memorizing more APIs. It's humans
stating goals and agents composing capabilities. grub is a small tool, so
it's a small example — but the shape is the same all the way up.

---

## For the dinosaurs who want to operate with code directly 🦖

No judgment. Sometimes you *are* the agent, and a REPL is the fastest
path. The Python API is built to be a pleasure to use directly.

### One function does it all

```python
from grub import grub

search = grub('./docs')                     # build a searcher
results = grub('./docs', 'how to deploy')    # ...or search in one call
```

`grub()` figures out what you handed it:

```python
grub('./docs')                       # a folder of files
grub('src/**/*.py')                  # a glob
grub(some_module)                    # a Python package's source
grub('https://example.com/guide')    # a web page (HTML stripped to text)
grub({'intro': '...', 'faq': '...'}) # a dict of documents
grub(['first doc', 'second doc'])    # a list of strings
```

### Results that explain themselves

```python
results = grub('./docs', 'configure logging')

for hit in results:
    print(hit.score, hit.key, hit.snippet)

results.keys        # ['logging.md', 'setup.md', ...]  best-first
results.scores      # [0.71, 0.33, ...]
print(results.show())            # a tidy ranked rendering
print(search['logging.md'])      # the full original text of a hit
```

Every hit carries a **score** and a **snippet** — the line that shows you
*why* it matched.

### Three ways to search

```python
grub(src, query, method='tfidf')     # lexical: shared words (default, fast)
grub(src, query, method='semantic')  # embeddings: shared *meaning*
grub(src, query, method='hybrid')    # a blend of both
```

Semantic search finds "automobile" when you searched "car". It needs
embeddings — either `pip install 'grub[semantic]'` (a local
sentence-transformers model) or your own provider:

```python
grub('./docs', method='semantic', embed=my_openai_embedding_function)
```

### Long documents, chunking, and persistence

```python
grub('./book.txt', chunk=1500)       # split into passages, not whole files
grub('./src', extensions=['.py'])    # filter what gets indexed

from grub import Searcher
grub('./big_codebase').save('code.grub')   # build once
Searcher.load('code.grub')                 # reload instantly
```

### From the command line

```bash
grub ./docs "how do I configure logging"
grub ./src --extensions .py --snippets "retry with backoff"
grub https://example.com/guide --semantic "getting started"
grub ./docs                                  # interactive prompt
```

### The legacy API still works

The original `SearchStore` and friends are unchanged and still exported,
so existing code keeps running:

```python
from grub import SearchStore

import sklearn, os
search = SearchStore(os.path.dirname(sklearn.__file__) + '/{}.py')
search('how to calibrate the estimates of my classifier')
```

---

## How it works

grub is a thin, honest pipeline:

```
source ──to_store──▶ store ──backend──▶ scores ──▶ SearchResults
```

1. **`to_store`** turns any source into a `Mapping[str, str]`.
2. A **backend** (TF-IDF, embeddings, or a hybrid) scores every document
   against your query.
3. Results come back ranked, scored, and annotated with snippets.

Every stage is swappable — see the `grub-extend` skill or
[`grub/backends.py`](grub/backends.py). That's the whole trick: simple
things stay simple, powerful things stay possible.

## Install

```bash
pip install grub               # core (TF-IDF / lexical search)
pip install 'grub[semantic]'   # adds local embedding-based search
```

## License

Apache-2.0
