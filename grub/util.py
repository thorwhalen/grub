r"""Small shared utilities for grub: tokenizing, clamping, snippet extraction.

These are low-level helpers used across grub's modules. They have no
heavy dependencies and are safe to import anywhere.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

__all__ = [
    "simple_tokenizer",
    "clamp_n",
    "best_snippet",
    "is_url",
    "truncate",
]

_word_p = re.compile(r"\w+")


def simple_tokenizer(text: str) -> list[str]:
    """Lower-cased word tokens of a string.

    >>> simple_tokenizer("Hello, World! Version 2")
    ['hello', 'world', 'version', '2']
    """
    return _word_p.findall(str(text).lower())


def clamp_n(n: int, size: int) -> int:
    """Clamp a requested result count to ``[1, size]``.

    Search backends choke when asked for more neighbours than they have
    documents; this keeps requests in range.

    >>> clamp_n(10, 3)
    3
    >>> clamp_n(5, 100)
    5
    >>> clamp_n(0, 100)
    1
    """
    return max(1, min(int(n), max(1, int(size))))


def truncate(text: str, width: int = 240) -> str:
    """Collapse whitespace and truncate ``text`` to at most ``width`` chars.

    >>> truncate("  lots   of   space  ", width=80)
    'lots of space'
    >>> truncate("abcdefghij", width=5)
    'abcd…'
    """
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def best_snippet(
    text: str,
    query: str,
    *,
    width: int = 240,
    tokenizer: Callable[[str], Sequence[str]] = simple_tokenizer,
) -> str:
    r"""Return a short excerpt of ``text`` that best matches ``query``.

    The excerpt is the single line sharing the most tokens with the
    query -- a cheap way to show *why* a document matched.

    >>> doc = "Intro line.\nThe quick brown fox jumps over.\nTrailing words."
    >>> best_snippet(doc, "quick brown fox")
    'The quick brown fox jumps over.'

    With no overlap at all it falls back to the start of the text:

    >>> best_snippet("alpha beta gamma", "nothing matches here", width=80)
    'alpha beta gamma'
    """
    text = str(text)
    q_tokens = set(tokenizer(query))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not q_tokens or not lines:
        return truncate(text, width)

    def line_score(line: str) -> int:
        return sum(1 for tok in tokenizer(line) if tok in q_tokens)

    best = max(lines, key=line_score)
    if line_score(best) == 0:
        return truncate(text, width)
    if len(best) <= width:
        return truncate(best, width)
    return _center_on_match(best, q_tokens, tokenizer, width)


def _center_on_match(
    line: str,
    q_tokens: set[str],
    tokenizer: Callable[[str], Sequence[str]],
    width: int,
) -> str:
    """Window ``line`` around the first query-token hit."""
    lowered = line.lower()
    hit = next((lowered.find(tok) for tok in q_tokens if tok in lowered), -1)
    if hit < 0:
        return truncate(line, width)
    start = max(0, hit - width // 3)
    window = line[start : start + width]
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(line) else ""
    return prefix + " ".join(window.split()) + suffix


def is_url(value: object) -> bool:
    """Whether ``value`` looks like a fetchable URL.

    >>> is_url("https://example.com")
    True
    >>> is_url("./local/path")
    False
    """
    return (
        isinstance(value, str)
        and "://" in value
        and value.split("://", 1)[0] in {"http", "https", "ftp", "file"}
    )
