"""Tests for grub's modern search API."""

import numpy as np
import pytest

import grub
from grub import (
    Searcher,
    SearchResult,
    SearchResults,
    chunk_store,
    chunk_text,
    html_to_text,
    to_store,
)
from grub.cli import main, run
from grub.util import best_snippet, clamp_n, simple_tokenizer

# A small, well-separated corpus: each doc is about a different topic.
DOCS = {
    "python": "Python is a popular high level programming language for general use.",
    "guitar": "A guitar is a musical instrument with six strings played by hand.",
    "coffee": "Coffee is a hot drink brewed from roasted ground coffee beans.",
    "garden": "A garden grows vegetables flowers and herbs in rich healthy soil.",
}

# A picklable embedding function so the semantic backend can be exercised in
# CI without the heavy optional sentence-transformers dependency.
_VOCAB = (
    "python programming language guitar musical instrument strings "
    "coffee drink beans garden vegetables flowers soil"
).split()


def toy_embed(texts):
    """Bag-of-words embedding over a fixed vocabulary (module-level => picklable)."""
    rows = []
    for text in texts:
        words = set(text.lower().split())
        rows.append([1.0 if w in words else 0.0 for w in _VOCAB])
    return np.array(rows, dtype="float32")


# -- the facade -------------------------------------------------------------


def test_grub_with_query_returns_results():
    results = grub.grub(DOCS, "a musical instrument with strings")
    assert isinstance(results, SearchResults)
    assert results.keys[0] == "guitar"


def test_grub_without_query_returns_searcher():
    searcher = grub.grub(DOCS)
    assert isinstance(searcher, Searcher)
    assert searcher("hot drink brewed from beans").keys[0] == "coffee"


def test_grub_reuses_an_existing_searcher():
    searcher = grub.grub(DOCS)
    assert grub.grub(searcher) is searcher


def test_search_alias():
    results = grub.search(DOCS, "growing vegetables and flowers")
    assert results.keys[0] == "garden"


# -- source adapters --------------------------------------------------------


def test_source_list_of_strings():
    searcher = grub.grub(["apple pie recipe", "car engine repair", "apple orchard"])
    assert set(searcher) == {"0", "1", "2"}
    assert searcher("juicy apple fruit").keys[0] in {"0", "2"}


def test_source_key_value_pairs():
    store = to_store([("k1", "hello world"), ("k2", "goodbye world")])
    assert store == {"k1": "hello world", "k2": "goodbye world"}


def test_source_directory(tmp_path):
    (tmp_path / "logging.md").write_text("configure the log level and handlers")
    (tmp_path / "database.md").write_text("connection pooling and retry strategy")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\xff")
    searcher = grub.grub(str(tmp_path))
    assert "logging.md" in searcher
    assert "binary.bin" not in searcher  # undecodable files are skipped
    assert searcher("how do I set the log level").keys[0] == "logging.md"


def test_source_directory_extension_filter(tmp_path):
    (tmp_path / "a.py").write_text("import os")
    (tmp_path / "b.txt").write_text("plain text")
    store = to_store(str(tmp_path), extensions=[".py"])
    assert set(store) == {"a.py"}


def test_source_glob(tmp_path):
    (tmp_path / "one.md").write_text("first markdown document")
    (tmp_path / "two.md").write_text("second markdown document")
    (tmp_path / "skip.txt").write_text("not markdown")
    store = to_store(str(tmp_path / "*.md"))
    assert len(store) == 2


def test_source_path_format(tmp_path):
    (tmp_path / "alpha.txt").write_text("alpha content")
    (tmp_path / "beta.txt").write_text("beta content")
    store = to_store(str(tmp_path / "{}.txt"))
    assert set(store) == {"alpha.txt", "beta.txt"}


def test_source_module():
    searcher = grub.grub(grub)
    assert len(searcher) > 0
    hit = searcher("turn anything into a searchable store").keys[0]
    assert hit.endswith(".py")


def test_bad_source_raises():
    with pytest.raises(FileNotFoundError):
        to_store("/definitely/not/a/real/path/xyz")
    with pytest.raises(TypeError):
        to_store(12345)


# -- results ----------------------------------------------------------------


def test_search_result_fields():
    results = grub.grub(DOCS, "musical instrument")
    top = results[0]
    assert isinstance(top, SearchResult)
    assert top.key == "guitar"
    assert 0 < top.score <= 1
    assert "instrument" in top.snippet


def test_search_results_helpers():
    results = grub.grub(DOCS, "musical instrument", n=4)
    assert results.keys[0] == "guitar"
    assert len(results.scores) == len(results)
    assert results.as_dict()["guitar"] == results[0].score
    assert "guitar" in results.show()
    assert "guitar" in repr(results)


def test_n_results_is_respected_and_clamped():
    assert len(grub.grub(DOCS, "anything", n=2)) == 2
    # Asking for more than the corpus holds should not error.
    assert len(grub.grub(DOCS, "anything", n=999)) == len(DOCS)


# -- backends ---------------------------------------------------------------


def test_tfidf_is_the_default():
    searcher = grub.grub(DOCS)
    assert searcher.backend.method == "tfidf"


def test_semantic_with_custom_embed():
    searcher = grub.grub(DOCS, method="semantic", embed=toy_embed)
    assert searcher.backend.method == "semantic"
    assert searcher("musical instrument strings").keys[0] == "guitar"


def test_hybrid_with_custom_embed():
    searcher = grub.grub(DOCS, method="hybrid", embed=toy_embed)
    assert searcher.backend.method == "hybrid"
    assert searcher("coffee beans drink").keys[0] == "coffee"


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        grub.grub(DOCS, method="quantum")


@pytest.mark.parametrize("method", ["tfidf", "semantic", "hybrid"])
def test_every_method_ranks_the_obvious_match_first(method):
    searcher = grub.grub(DOCS, method=method, embed=toy_embed)
    assert searcher("vegetables flowers soil garden").keys[0] == "garden"


# -- chunking ---------------------------------------------------------------


def test_chunk_text():
    assert chunk_text("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij", "j"]
    assert chunk_text("short", size=100) == ["short"]


def test_chunk_store_splits_long_values():
    store = chunk_store({"big": "abcdef", "small": "xy"}, size=3, overlap=0)
    assert set(store) == {"big#0", "big#1", "small"}


def test_searcher_with_chunking():
    long_doc = "padding " * 50 + "the secret keyword is platypus " + "padding " * 50
    searcher = grub.grub({"doc": long_doc}, chunk=120)
    assert len(searcher) > 1  # the document was split
    assert searcher("platypus").keys[0].startswith("doc#")


# -- persistence ------------------------------------------------------------


def test_save_and_load(tmp_path):
    path = str(tmp_path / "index.grub")
    grub.grub(DOCS).save(path)
    loaded = Searcher.load(path)
    assert isinstance(loaded, Searcher)
    assert loaded("hot drink brewed from beans").keys[0] == "coffee"


def test_save_and_load_semantic(tmp_path):
    path = str(tmp_path / "sem.grub")
    grub.grub(DOCS, method="semantic", embed=toy_embed).save(path)
    loaded = Searcher.load(path)
    assert loaded("musical instrument strings").keys[0] == "guitar"


# -- utilities --------------------------------------------------------------


def test_simple_tokenizer():
    assert simple_tokenizer("Hello, World!") == ["hello", "world"]


def test_clamp_n():
    assert clamp_n(10, 3) == 3
    assert clamp_n(0, 100) == 1
    assert clamp_n(5, 100) == 5


def test_best_snippet():
    doc = "Intro paragraph.\nThe quick brown fox jumps.\nUnrelated tail."
    assert best_snippet(doc, "quick brown fox") == "The quick brown fox jumps."


def test_html_to_text():
    assert html_to_text("<h1>Hi</h1><p>there</p>") == "Hi there"
    assert html_to_text("<style>.a{}</style><p>only this</p>") == "only this"


# -- cli --------------------------------------------------------------------


def test_cli_run_prints_results(tmp_path, capsys):
    (tmp_path / "doc.md").write_text("configure the log level here")
    run(str(tmp_path), "configure log level")
    out = capsys.readouterr().out
    assert "doc.md" in out


def test_cli_run_with_snippets(tmp_path, capsys):
    (tmp_path / "doc.md").write_text("the snippet keyword aardvark appears here")
    run(str(tmp_path), "aardvark", snippets=True)
    assert "aardvark" in capsys.readouterr().out


def test_cli_main_entry_point(tmp_path, capsys):
    (tmp_path / "doc.md").write_text("hello from the cli main entry point")
    assert main([str(tmp_path), "cli", "entry", "point"]) == 0
    assert "doc.md" in capsys.readouterr().out


def test_cli_bad_source_exits_nonzero():
    with pytest.raises(SystemExit):
        run("/no/such/path/at/all", "query")


# -- backward compatibility -------------------------------------------------


def test_legacy_search_store_still_works():
    from grub import SearchStore

    store = {
        "cats": "the cat sat quietly on the soft warm mat",
        "dogs": "the dog ran fast across the wide green field",
    }
    search = SearchStore(store, n_neighbors=2)
    assert list(search("cat"))[0] == "cats"


def test_legacy_grubber_still_works():
    from grub import grubber

    store = {f"d{i}": f"document number {i} about topic {i}" for i in range(5)}
    searcher = grubber(store, n=3)
    assert len(searcher("topic 2")) == 3
