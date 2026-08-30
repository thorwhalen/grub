# `misc/` — the CLI parity harness

`grub`'s command line is **pinned**. `cli_cases.txt` is a corpus of argv vectors;
`cli_golden_py<major><minor>.json` records what each one does — exit code, stdout,
stderr, the normalised `usage:` line, and the full `--help` body. `tests/test_cli_parity.py`
replays the golden matching the running interpreter on every test run.

The goldens were recorded from the **argh-based** CLI, before the migration to
[`cw`](https://pypi.org/project/cw/) and before any source edit. That is what makes them
evidence rather than decoration: they describe a command line nobody had yet had the
chance to change.

## When the parity test goes red

**Read the diff before touching anything.** It tells you which of three things happened.

1. **You changed the CLI on purpose** — added a command, renamed a flag, reworded a
   message. Re-record (below) and put the diff in the pull request.
2. **A dependency moved.** The only byte in the golden that a dependency can
   legitimately move is a `[0.437]` similarity score, which comes from scikit-learn's
   `TfidfVectorizer`. A diff confined to the bracketed numbers is a scikit-learn change;
   re-record. A diff anywhere else is not.
3. **You changed the CLI by accident.** This is what the test is for. Fix the code, not
   the golden.

## Re-recording

One golden per Python version in `[tool.wads.ci.testing]`, each recorded on its own
interpreter, from the repo root:

```bash
python3.12 -m cw.testing characterize grub --cases misc/cli_cases.txt \
    -o misc/cli_golden_py312.json
```

`argparse` is part of the standard library and rewrites its own text between CPython
versions, which is why one golden cannot cover the matrix. Adding a version to CI means
recording a golden for it; the test fails loudly, with instructions, rather than skipping
when one is missing.

## Adding a case

Anything a user can type belongs here. Four rules, all learned the hard way:

- **No case may print an absolute path.** The golden is committed and has to replay on
  someone else's machine. In particular, no case whose failure escapes as an uncaught
  traceback — the traceback names every frame's file.
- **No case may write to the filesystem or reach the network.**
- **No case may need an optional extra**, so the corpus replays on a bare install.
- **Prefer output that a dependency upgrade cannot move.** `-n 1` on a search keeps the
  assertion to a single score rather than four.
