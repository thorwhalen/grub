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

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

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

### grub.util.clamp_n(n, size)

Clamp a requested result count to `[1, size]`.

Search backends choke when asked for more neighbours than they have
documents; this keeps requests in range.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> clamp_n(10, 3)
3
>>> clamp_n(5, 100)
5
>>> clamp_n(0, 100)
1
```

### grub.util.is_url(value)

Whether `value` looks like a fetchable URL.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> is_url("https://example.com")
True
>>> is_url("./local/path")
False
```

### grub.util.simple_tokenizer(text)

Lower-cased word tokens of a string.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> simple_tokenizer("Hello, World! Version 2")
['hello', 'world', 'version', '2']
```

### grub.util.truncate(text, width=240)

Collapse whitespace and truncate `text` to at most `width` chars.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> truncate("  lots   of   space  ", width=80)
'lots of space'
>>> truncate("abcdefghij", width=5)
'abcd…'
```
