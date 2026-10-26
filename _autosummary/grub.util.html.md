# grub.util

Small shared utilities for grub: tokenizing, clamping, snippet extraction.

These are low-level helpers used across grub’s modules. They have no
heavy dependencies and are safe to import anywhere.

### Functions

| [`simple_tokenizer`](#grub.util.simple_tokenizer)(text)                            | Lower-cased word tokens of a string.                              |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`clamp_n`](#grub.util.clamp_n)(n, size)                                  | Clamp a requested result count to `[1, size]`.                    |
| [`best_snippet`](#grub.util.best_snippet)(text, query, \*[, width, tokenizer]) | Return a short excerpt of `text` that best matches `query`.       |
| [`is_url`](#grub.util.is_url)(value)                                     | Whether `value` looks like a fetchable URL.                       |
| [`truncate`](#grub.util.truncate)(text[, width])                           | Collapse whitespace and truncate `text` to at most `width` chars. |

### grub.util.best_snippet(text, query, \*, width=240, tokenizer=<function simple_tokenizer>)

Return a short excerpt of `text` that best matches `query`.

The excerpt is the single line sharing the most tokens with the
query – a cheap way to show *why* a document matched.

```pycon
>>> doc = "Intro line.\nThe quick brown fox jumps over.\nTrailing words."
>>> best_snippet(doc, "quick brown fox")
'The quick brown fox jumps over.'
```

With no overlap at all it falls back to the start of the text:

```pycon
>>> best_snippet("alpha beta gamma", "nothing matches here", width=80)
'alpha beta gamma'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### grub.util.clamp_n(n, size)

Clamp a requested result count to `[1, size]`.

Search backends choke when asked for more neighbours than they have
documents; this keeps requests in range.

```pycon
>>> clamp_n(10, 3)
3
>>> clamp_n(5, 100)
5
>>> clamp_n(0, 100)
1
```

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### grub.util.is_url(value)

Whether `value` looks like a fetchable URL.

```pycon
>>> is_url("https://example.com")
True
>>> is_url("./local/path")
False
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### grub.util.simple_tokenizer(text)

Lower-cased word tokens of a string.

```pycon
>>> simple_tokenizer("Hello, World! Version 2")
['hello', 'world', 'version', '2']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### grub.util.truncate(text, width=240)

Collapse whitespace and truncate `text` to at most `width` chars.

```pycon
>>> truncate("  lots   of   space  ", width=80)
'lots of space'
>>> truncate("abcdefghij", width=5)
'abcd…'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
