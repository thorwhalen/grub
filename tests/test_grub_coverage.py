"""Extra coverage: results rendering, snippet windowing, CLI paths, pycode helpers."""

from grub import Searcher
from grub.cli import main
from grub.pycode import call_and_return_none_on_exception, non_doctest_lines
from grub.results import SearchResult, SearchResults
from grub.util import best_snippet

DOCS = {
    "guitar": "A guitar is a musical instrument with six strings.",
    "coffee": "Coffee is a hot drink brewed from roasted beans.",
}


def test_results_show_and_repr():
    r = SearchResults([SearchResult("a", 0.5, "snip")], query="q")
    assert "1 results for 'q'" in r.show()
    assert "snip" in r.show()
    assert "snip" not in r.show(snippets=False)
    assert r.snippets == ["snip"]
    assert "SearchResults for 'q' -- 1 hits" in repr(r)
    assert str(r[0]) == "[0.500] a"


def test_empty_results():
    r = SearchResults(query="zzz")
    assert r.show() == "No results for 'zzz'."
    assert repr(r) == "SearchResults(query='zzz', [])"


def test_best_snippet_windows_long_line():
    long_line = "filler " * 40 + "needle " + "filler " * 40
    out = best_snippet(long_line, "needle", width=60)
    assert "needle" in out
    assert out.startswith("…") and out.endswith("…")


def test_non_doctest_lines_and_safe_call():
    doc = "Intro\n>>> x = 1\nstill code\n\nAfter"
    assert list(non_doctest_lines(doc)) == ["Intro", "After"]
    assert call_and_return_none_on_exception(int, "nope") is None
    assert call_and_return_none_on_exception(int, "3") == 3


def test_cli_query_and_no_results(tmp_path, capsys):
    (tmp_path / "a.txt").write_text(DOCS["guitar"])
    (tmp_path / "b.txt").write_text(DOCS["coffee"])
    assert main([str(tmp_path), "guitar"]) == 0
    out = capsys.readouterr().out
    assert "a.txt" in out


def test_cli_repl(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.txt").write_text(DOCS["guitar"])
    answers = iter(["guitar", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    main([str(tmp_path)])
    assert "grub>" not in capsys.readouterr().err
