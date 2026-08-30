"""Assert grub's command line has not moved since it was recorded.

``misc/cli_golden_py3XX.json`` were recorded from the **argh-based** CLI, before the
migration to ``cw`` and before any source edit; ``misc/cli_cases.txt`` is the corpus they
were recorded from, and ``misc/README.md`` explains how to read a red result and how to
re-record. This test replays them against whatever dispatcher is installed now, so neither
a refactor here nor a new ``cw`` release can change what a shell sees without a test going
red.

Every case's exit code, stdout, stderr and normalised ``usage:`` line is asserted, and so is
the full ``--help`` body (``strict_help``). Nothing is advisory.

One golden per CPython minor version, because ``argparse`` is stdlib and rewrites its own
text between versions: 3.12 stopped listing ``nargs='*'`` positionals among "the following
arguments are required", which is the only difference between the 3.10 and 3.12 recordings
and predates this repo's migration. An unrecorded version fails loudly rather than skipping
-- a parity test that quietly does nothing is worse than no parity test.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "misc"


def _golden_for_this_python() -> Path:
    """The golden recorded by the CPython running this test."""
    major, minor = sys.version_info[:2]
    path = GOLDEN_DIR / f"cli_golden_py{major}{minor}.json"
    if path.exists():
        return path
    recorded = sorted(p.name for p in GOLDEN_DIR.glob("cli_golden_py*.json"))
    pytest.fail(
        f"no CLI golden recorded for Python {major}.{minor} (have: {', '.join(recorded)}). "
        f"argparse's own wording differs between CPython versions, so each version in CI "
        f"needs its own recording. See misc/README.md."
    )


def _console_script() -> str:
    """The installed ``grub`` executable, however this environment lays it out."""
    found = shutil.which("grub")
    if found:
        return found
    # A venv whose bin/ is not on PATH -- common under `uv run` and tox.
    for name in ("grub", "grub.exe"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    pytest.fail(
        "the `grub` console script is not installed, so the CLI cannot be checked. "
        "Install the package (`pip install -e .`) before running this test."
    )


def test_cli_surface_is_unchanged():
    """Every recorded argv still produces the same exit code, stdout and stderr."""
    cw_testing = pytest.importorskip("cw.testing")
    cw_testing.assert_replay(
        _golden_for_this_python(),
        prog=[_console_script()],
        cwd=str(REPO_ROOT),
        strict_help=True,
    )
