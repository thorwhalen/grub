# grub.sources

Source adapters: turn *anything* into a searchable store.

A “store” here is just a `Mapping` whose values are text. Everything
grub can search – a folder, a glob, a Python package, a website, a list
of strings, a plain dict – is funnelled through [`to_store()`](#grub.sources.to_store), which
detects the kind of source and returns a uniform mapping.

The single entry point is [`to_store()`](#grub.sources.to_store). The `from_*` helpers are
there when you want to be explicit.

### Functions

| [`to_store`](#grub.sources.to_store)(source, \*[, recursive, extensions, ...])   | Coerce `source` into a `Mapping[str, str]` ready to be searched.     |
|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`from_dir`](#grub.sources.from_dir)(directory, \*[, recursive, ...])            | Every decodable text file under `directory`, keyed by relative path. |
| [`from_files`](#grub.sources.from_files)(paths, \*[, encoding])                    | Read the given files into a `{name: content}` dict.                  |
| [`from_module`](#grub.sources.from_module)(module)                                  | The `.py` source files of a Python module or package.                |
| [`from_strings`](#grub.sources.from_strings)(strings)                                | Enumerate strings into a `{index: string}` dict.                     |
| [`from_urls`](#grub.sources.from_urls)(urls, \*[, encoding, timeout])             | Fetch URLs and return `{url: text}` with HTML stripped to text.      |
| [`chunk_text`](#grub.sources.chunk_text)(text, \*[, size, overlap])                | Split `text` into overlapping character windows.                     |
| [`chunk_store`](#grub.sources.chunk_store)(store, \*[, size, overlap])              | Chunk every value of `store`; keys become `"{key}#{n}"`.             |
| [`html_to_text`](#grub.sources.html_to_text)(html)                                   | Strip HTML tags, returning visible text only.                        |

### grub.sources.chunk_store(store, , size=1500, overlap=200)

Chunk every value of `store`; keys become `"{key}#{n}"`.

```pycon
>>> sorted(chunk_store({"doc": "abcdef"}, size=3, overlap=0))
['doc#0', 'doc#1']
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.sources.chunk_text(text, , size=1500, overlap=200)

Split `text` into overlapping character windows.

Long documents search better in pieces: a hit then points at a
passage instead of a whole file.

```pycon
>>> chunk_text("abcdefghij", size=4, overlap=1)
['abcd', 'defg', 'ghij', 'j']
>>> chunk_text("short", size=100)
['short']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### grub.sources.from_dir(directory, , recursive=True, extensions=None, encoding='utf-8', max_bytes=1000000, skip_dirs=frozenset({'.eggs', '.git', '.hg', '.idea', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.svn', '.venv', '.vscode', '_\_pycache_\_', 'build', 'dist', 'node_modules', 'venv'}))

Every decodable text file under `directory`, keyed by relative path.

Binary files, oversized files, and noise directories (`.git`,
`__pycache__`, `node_modules`, …) are skipped automatically.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.sources.from_files(paths, , encoding='utf-8')

Read the given files into a `{name: content}` dict.

Keys are file names; if names collide the relative path is used.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.sources.from_module(module)

The `.py` source files of a Python module or package.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

### grub.sources.from_strings(strings)

Enumerate strings into a `{index: string}` dict.

```pycon
>>> from_strings(["alpha", "beta"])
{'0': 'alpha', '1': 'beta'}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.sources.from_urls(urls, , encoding='utf-8', timeout=30.0)

Fetch URLs and return `{url: text}` with HTML stripped to text.

Uses only the standard library. Unreachable URLs are skipped.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.sources.html_to_text(html)

Strip HTML tags, returning visible text only.

```pycon
>>> html_to_text("<h1>Title</h1><p>Hello <b>world</b></p>")
'Title Hello world'
>>> html_to_text("<style>.x{}</style><p>visible</p>")
'visible'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### grub.sources.to_store(source, , recursive=True, extensions=None, chunk=None, encoding='utf-8', max_bytes=1000000)

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
