r"""Source adapters: turn *anything* into a searchable store.

A "store" here is just a ``Mapping`` whose values are text. Everything
grub can search -- a folder, a glob, a Python package, a website, a list
of strings, a plain dict -- is funnelled through :func:`to_store`, which
detects the kind of source and returns a uniform mapping.

The single entry point is :func:`to_store`. The ``from_*`` helpers are
there when you want to be explicit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Iterable

from grub.base import get_py_files_store
from grub.util import is_url

__all__ = [
    "to_store",
    "from_dir",
    "from_files",
    "from_module",
    "from_strings",
    "from_urls",
    "chunk_text",
    "chunk_store",
    "html_to_text",
]

#: Directory names skipped when walking a folder.
DFLT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".eggs",
    }
)

#: Files larger than this (bytes) are skipped by directory walks.
DFLT_MAX_BYTES = 1_000_000

_GLOB_CHARS = set("*?[")


def to_store(
    source,
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    chunk: int | tuple[int, int] | None = None,
    encoding: str = "utf-8",
    max_bytes: int = DFLT_MAX_BYTES,
) -> Mapping:
    """Coerce ``source`` into a ``Mapping[str, str]`` ready to be searched.

    ``source`` may be:

    * a ``Mapping`` -- used as-is
    * a folder path -- every (decodable) text file under it
    * a glob pattern -- every matching file
    * a ``"path/{}.ext"`` format -- a lazy ``py2store`` text store
    * a Python module/package -- its ``.py`` source files
    * a URL (or list of URLs) -- fetched and stripped to text
    * a list of strings -- enumerated into ``{"0": ..., "1": ...}``
    * an iterable of ``(key, value)`` pairs -- turned into a dict

    >>> store = to_store({"a": "first doc", "b": "second doc"})
    >>> dict(store)
    {'a': 'first doc', 'b': 'second doc'}
    >>> sorted(to_store(["red", "green", "blue"]))
    ['0', '1', '2']

    :param recursive: walk sub-directories (folder sources only).
    :param extensions: keep only files with these extensions, e.g. ``[".py"]``.
    :param chunk: split values into overlapping windows -- an int size or a
        ``(size, overlap)`` pair. Handy for long documents.
    :param encoding: text encoding for files.
    :param max_bytes: skip files larger than this (folder/glob sources).
    """
    store = _route(
        source,
        recursive=recursive,
        extensions=extensions,
        encoding=encoding,
        max_bytes=max_bytes,
    )
    if chunk is not None:
        size, overlap = _chunk_params(chunk)
        store = chunk_store(store, size=size, overlap=overlap)
    return store


def _route(source, *, recursive, extensions, encoding, max_bytes):
    """Dispatch ``source`` to the right adapter (no chunking)."""
    if isinstance(source, Mapping):
        return source
    if isinstance(source, ModuleType):
        return from_module(source)
    if isinstance(source, (str, Path)):
        return _route_path(
            source,
            recursive=recursive,
            extensions=extensions,
            encoding=encoding,
            max_bytes=max_bytes,
        )
    if isinstance(source, Iterable):
        return _route_iterable(source)
    raise TypeError(
        f"Don't know how to search a {type(source).__name__}. Give grub a "
        f"folder, glob, module, URL, dict, or list of strings."
    )


def _route_path(source, *, recursive, extensions, encoding, max_bytes):
    """Resolve a string/Path source."""
    text = str(source)
    if is_url(text):
        return from_urls([text], encoding=encoding)
    if "{}" in text:
        from py2store import LocalTextStore

        return LocalTextStore(text)
    path = Path(text).expanduser()
    if path.is_dir():
        return from_dir(
            path,
            recursive=recursive,
            extensions=extensions,
            encoding=encoding,
            max_bytes=max_bytes,
        )
    if path.is_file():
        return from_files([path], encoding=encoding)
    if _GLOB_CHARS & set(text):
        return _from_glob(
            text, extensions=extensions, encoding=encoding, max_bytes=max_bytes
        )
    raise FileNotFoundError(
        f"No such file, folder, or glob: {text!r}. Pass an existing path, a "
        f"glob like 'docs/*.md', a 'dir/{{}}.py' format, a module, or a dict."
    )


def _route_iterable(source):
    """Resolve a list/tuple/generator source."""
    items = list(source)
    if not items:
        return {}
    if all(isinstance(it, str) for it in items):
        if all(is_url(it) for it in items):
            return from_urls(items)
        return from_strings(items)
    if all(isinstance(it, (tuple, list)) and len(it) == 2 for it in items):
        return {str(k): str(v) for k, v in items}
    raise TypeError(
        "An iterable source must be all strings or all (key, value) pairs."
    )


# -- explicit adapters ------------------------------------------------------


def from_strings(strings: Iterable[str]) -> dict:
    """Enumerate strings into a ``{index: string}`` dict.

    >>> from_strings(["alpha", "beta"])
    {'0': 'alpha', '1': 'beta'}
    """
    return {str(i): str(s) for i, s in enumerate(strings)}


def from_module(module: ModuleType) -> Mapping:
    """The ``.py`` source files of a Python module or package."""
    return get_py_files_store(module)


def from_files(paths: Iterable[os.PathLike | str], *, encoding: str = "utf-8") -> dict:
    """Read the given files into a ``{name: content}`` dict.

    Keys are file names; if names collide the relative path is used.
    """
    paths = [Path(p) for p in paths]
    names = [p.name for p in paths]
    use_full = len(set(names)) != len(names)
    out: dict[str, str] = {}
    for path in paths:
        key = str(path) if use_full else path.name
        try:
            out[key] = path.read_text(encoding=encoding, errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
    return out


def from_dir(
    directory: os.PathLike | str,
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    encoding: str = "utf-8",
    max_bytes: int = DFLT_MAX_BYTES,
    skip_dirs: Iterable[str] = DFLT_SKIP_DIRS,
) -> dict:
    """Every decodable text file under ``directory``, keyed by relative path.

    Binary files, oversized files, and noise directories (``.git``,
    ``__pycache__``, ``node_modules``, ...) are skipped automatically.
    """
    root = Path(directory).expanduser()
    exts = _normalize_extensions(extensions)
    skip = set(skip_dirs)
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in skip and not d.startswith(".")
        ]
        if not recursive:
            dirnames[:] = []
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if exts is not None and path.suffix.lower() not in exts:
                continue
            text = _read_text_file(path, encoding=encoding, max_bytes=max_bytes)
            if text is not None:
                out[path.relative_to(root).as_posix()] = text
    return out


def from_urls(
    urls: Iterable[str], *, encoding: str = "utf-8", timeout: float = 30.0
) -> dict:
    """Fetch URLs and return ``{url: text}`` with HTML stripped to text.

    Uses only the standard library. Unreachable URLs are skipped.
    """
    from urllib.request import Request, urlopen

    out: dict[str, str] = {}
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "grub-search"})
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            out[url] = html_to_text(raw.decode(encoding, errors="replace"))
        except Exception:  # noqa: BLE001 -- a bad URL shouldn't kill the batch
            continue
    return out


# -- chunking ---------------------------------------------------------------


def chunk_text(text: str, *, size: int = 1500, overlap: int = 200) -> list[str]:
    """Split ``text`` into overlapping character windows.

    Long documents search better in pieces: a hit then points at a
    passage instead of a whole file.

    >>> chunk_text("abcdefghij", size=4, overlap=1)
    ['abcd', 'defg', 'ghij', 'j']
    >>> chunk_text("short", size=100)
    ['short']
    """
    text = str(text)
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_store(store: Mapping, *, size: int = 1500, overlap: int = 200) -> dict:
    """Chunk every value of ``store``; keys become ``"{key}#{n}"``.

    >>> sorted(chunk_store({"doc": "abcdef"}, size=3, overlap=0))
    ['doc#0', 'doc#1']
    """
    out: dict[str, str] = {}
    for key in store:
        chunks = chunk_text(str(store[key]), size=size, overlap=overlap)
        if len(chunks) == 1:
            out[str(key)] = chunks[0]
        else:
            for i, chunk in enumerate(chunks):
                out[f"{key}#{i}"] = chunk
    return out


# -- html -------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, dropping script/style content."""

    _skip = {"script", "style", "head", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._skip and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if self._depth == 0 and data.strip():
            self._parts.append(data.strip())

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def html_to_text(html: str) -> str:
    """Strip HTML tags, returning visible text only.

    >>> html_to_text("<h1>Title</h1><p>Hello <b>world</b></p>")
    'Title Hello world'
    >>> html_to_text("<style>.x{}</style><p>visible</p>")
    'visible'
    """
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text


# -- internals --------------------------------------------------------------


def _normalize_extensions(extensions):
    if extensions is None:
        return None
    return {
        ("." + e.lstrip(".")).lower() for e in extensions
    }


def _read_text_file(path: Path, *, encoding: str, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding=encoding, errors="strict")
    except (UnicodeDecodeError, OSError, ValueError):
        return None


def _from_glob(pattern, *, extensions, encoding, max_bytes):
    from glob import glob

    exts = _normalize_extensions(extensions)
    paths = [Path(p) for p in sorted(glob(pattern, recursive=True))]
    paths = [p for p in paths if p.is_file()]
    if exts is not None:
        paths = [p for p in paths if p.suffix.lower() in exts]
    out: dict[str, str] = {}
    for path in paths:
        text = _read_text_file(path, encoding=encoding, max_bytes=max_bytes)
        if text is not None:
            out[path.as_posix()] = text
    return out


def _chunk_params(chunk) -> tuple[int, int]:
    if isinstance(chunk, bool):
        raise TypeError("chunk must be an int size or a (size, overlap) pair")
    if isinstance(chunk, int):
        return chunk, max(0, chunk // 8)
    if isinstance(chunk, (tuple, list)) and len(chunk) == 2:
        return int(chunk[0]), int(chunk[1])
    raise TypeError("chunk must be an int size or a (size, overlap) pair")
