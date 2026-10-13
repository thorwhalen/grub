# grub.cli

Command-line interface for grub: `grub SOURCE [QUERY ...]`.

Search a folder, glob, module, or URL straight from the shell:

```default
grub ./docs "how do I configure logging"
grub ./src --extensions .py --snippets "retry with backoff"
grub https://example.com            # no query -> interactive prompt
```

Run `grub --help` for the full option list.

### Functions

| [`run`](#grub.cli.run)(source, \*query[, method, n, snippets, ...])   | Search `source`; with no query, drop into an interactive prompt.   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`main`](#grub.cli.main)([argv])                                       | Entry point for the `grub` console script; returns its exit code.  |

### grub.cli.main(argv=None)

Entry point for the `grub` console script; returns its exit code.

`argv` defaults to `sys.argv[1:]`. `cw.dispatch()` takes it positionally and
reproduces the grammar this CLI has always had – pinned by `misc/cli_golden_py*.json`
and asserted by `tests/test_cli_parity.py`.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### grub.cli.run(source, \*query, method='tfidf', n=10, snippets=False, semantic=False, hybrid=False, chunk=0, extensions='')

Search `source`; with no query, drop into an interactive prompt.

* **Parameters:**
  * **source** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – folder, glob, `dir/{}.ext` format, module path, or URL.
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the search query (omit for an interactive session).
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `tfidf`, `semantic`, or `hybrid`.
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of results to show.
  * **snippets** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – also show the matching snippet for each result.
  * **semantic** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – shortcut for `--method semantic`.
  * **hybrid** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – shortcut for `--method hybrid`.
  * **chunk** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – split documents into windows of this many characters.
  * **extensions** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – comma-separated file extensions to restrict to,
    e.g. `.py,.md`.
