"""Okapi BM25: the tokenizer, IDF, average length and zero scores."""

from __future__ import annotations

import math

import pytest

from localmelo.melo.memory.bm25 import K1, B, BM25Index, idf, tokenize


def _okapi(frequency: int, length: int, avg_length: float, weight: float) -> float:
    """One query term's share of a document's score, computed by hand."""
    norm = K1 * (1 - B + B * length / avg_length)
    return weight * frequency * (K1 + 1) / (frequency + norm)


def _ids(ranking: list[tuple[str, float]]) -> list[str]:
    return [doc_id for doc_id, _score in ranking]


# ── Tokenizer ───────────────────────────────────────────────────────────────


class TestTokenize:
    def test_ascii_words_and_digits_are_lowercased(self) -> None:
        assert tokenize("Read FILE v2, then 42!") == [
            "read",
            "file",
            "v2",
            "then",
            "42",
        ]

    def test_an_identifier_also_gives_its_words(self) -> None:
        assert tokenize("file_write") == ["file_write", "file", "write"]
        assert tokenize("transpose_test.py") == [
            "transpose_test",
            "transpose",
            "test",
            "py",
        ]
        assert tokenize("mcp__notes__search") == [
            "mcp__notes__search",
            "mcp",
            "notes",
            "search",
        ]

    def test_leading_and_trailing_underscores_are_dropped(self) -> None:
        assert tokenize("__init__ _private ___") == ["init", "private"]

    def test_a_cjk_run_gives_its_characters_then_its_bigrams(self) -> None:
        assert tokenize("读取文件") == ["读", "取", "文", "件", "读取", "取文", "文件"]

    def test_a_single_cjk_character_is_one_term(self) -> None:
        assert tokenize("读") == ["读"]

    def test_cjk_punctuation_separates_runs(self) -> None:
        assert tokenize("天气，明天。") == ["天", "气", "天气", "明", "天", "明天"]

    def test_hiragana_katakana_and_hangul_form_runs(self) -> None:
        assert tokenize("よむ") == ["よ", "む", "よむ"]
        assert tokenize("ファイル") == ["フ", "ァ", "イ", "ル", "ファ", "ァイ", "イル"]
        assert tokenize("파일") == ["파", "일", "파일"]
        # The katakana middle dot is punctuation, not part of a run.
        assert tokenize("ト・ム") == ["ト", "ム"]

    def test_mixed_text(self) -> None:
        assert tokenize("用file_write写入文件") == [
            "用",
            "file_write",
            "file",
            "write",
            "写",
            "入",
            "文",
            "件",
            "写入",
            "入文",
            "文件",
        ]
        assert tokenize("Python脚本v3") == ["python", "脚", "本", "脚本", "v3"]

    def test_other_characters_separate_terms(self) -> None:
        assert tokenize("café naïve → x") == ["caf", "na", "ve", "x"]
        assert tokenize("  ,.; -- ") == []
        assert tokenize("") == []


# ── IDF ─────────────────────────────────────────────────────────────────────


class TestIdf:
    def test_okapi_formula(self) -> None:
        assert idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))
        assert idf(10, 10) == pytest.approx(math.log(1 + 0.5 / 10.5))
        assert idf(1, 1) == pytest.approx(math.log(1 + 0.5 / 1.5))

    def test_lower_for_a_term_in_more_documents_and_always_above_zero(self) -> None:
        values = [idf(10, n) for n in range(11)]
        assert values == sorted(values, reverse=True)
        assert len(set(values)) == len(values)
        assert min(values) > 0

    def test_common_words_do_not_decide_the_ranking(self) -> None:
        """A term in every document weighs far less than a rare one.

        The nine documents repeat the query's common words; the tenth
        holds its one rare word once. Without IDF, the common words would
        rank the nine first.
        """
        index = BM25Index()
        for i in range(9):
            index.add(f"common{i}", "the tool and the helper of the team and the rest")
        index.add("subject", "flip rows into columns: transpose")

        ranking = index.search("the and the and transpose", top_k=10)
        assert _ids(ranking)[0] == "subject"
        assert len(ranking) == 10

        # A word in nine of the ten documents gets less than a tenth of the
        # rare word's IDF: a document that holds it four times scores less
        # than a fifth of one that holds the rare word once.
        assert idf(10, 9) < idf(10, 1) / 10
        [(_, common)] = index.search("the", top_k=1)
        [(_, rare)] = index.search("transpose", top_k=1)
        assert common < rare / 5

    def test_idf_counts_the_indexed_documents(self) -> None:
        index = BM25Index()
        index.add("a", "alpha beta")
        index.add("b", "alpha")
        index.add("c", "gamma")
        scores = dict(index.search("alpha", top_k=3))
        avg_length = (2 + 1 + 1) / 3
        assert scores == {
            "a": pytest.approx(_okapi(1, 2, avg_length, idf(3, 2))),
            "b": pytest.approx(_okapi(1, 1, avg_length, idf(3, 2))),
        }


# ── Average length ──────────────────────────────────────────────────────────


class TestAverageLength:
    def test_lengths_are_normalized_by_the_average_of_the_documents(self) -> None:
        index = BM25Index()
        index.add("short", "alpha beta")
        index.add("long", "alpha " + "pad " * 9)
        scores = dict(index.search("alpha", top_k=2))
        avg_length = (2 + 10) / 2
        assert scores == {
            "short": pytest.approx(_okapi(1, 2, avg_length, idf(2, 2))),
            "long": pytest.approx(_okapi(1, 10, avg_length, idf(2, 2))),
        }
        assert scores["short"] > scores["long"]

    def test_the_average_follows_the_documents_indexed(self) -> None:
        """No fixed length: a new document changes the others' scores."""
        index = BM25Index()
        index.add("a", "alpha beta gamma")
        [(_, alone)] = index.search("alpha", top_k=1)
        index.add("b", "delta " * 30)
        [(_, beside_long)] = index.search("alpha", top_k=1)
        assert alone == pytest.approx(_okapi(1, 3, 3.0, idf(1, 1)))
        assert beside_long == pytest.approx(_okapi(1, 3, 16.5, idf(2, 1)))

    def test_an_identifier_counts_with_its_words(self) -> None:
        index = BM25Index()
        index.add("tool", "file_write")  # file_write, file, write: 3 terms
        index.add("other", "x")
        scores = dict(index.search("write", top_k=2))
        assert scores == {"tool": pytest.approx(_okapi(1, 3, 2.0, idf(2, 1)))}


# ── Zero scores and search ──────────────────────────────────────────────────


class TestZeroScores:
    def test_a_document_without_a_query_term_is_never_returned(self) -> None:
        index = BM25Index()
        index.add("a", "alpha")
        index.add("b", "beta")
        assert _ids(index.search("alpha gamma", top_k=5)) == ["a"]
        assert index.search("gamma", top_k=5) == []

    def test_a_query_without_terms_matches_nothing(self) -> None:
        index = BM25Index()
        index.add("a", "alpha")
        assert index.search("", top_k=5) == []
        assert index.search("... ,", top_k=5) == []

    def test_an_empty_document_is_never_returned(self) -> None:
        index = BM25Index()
        index.add("empty", "")
        assert index.search("alpha", top_k=5) == []
        index.add("a", "alpha")
        assert _ids(index.search("alpha", top_k=5)) == ["a"]

    def test_an_empty_index_matches_nothing(self) -> None:
        assert BM25Index().search("alpha", top_k=5) == []


class TestSearch:
    def test_best_first_up_to_top_k(self) -> None:
        index = BM25Index()
        index.add("once", "alpha beta gamma delta")
        index.add("twice", "alpha alpha beta gamma")
        index.add("none", "beta gamma delta epsilon")
        assert _ids(index.search("alpha", top_k=5)) == ["twice", "once"]
        assert _ids(index.search("alpha", top_k=1)) == ["twice"]
        assert index.search("alpha", top_k=0) == []

    def test_equal_scores_keep_the_order_documents_were_first_added(self) -> None:
        index = BM25Index()
        index.add("b", "same text")
        index.add("a", "same text")
        index.add("b", "same text")  # replaced in place
        assert _ids(index.search("text", top_k=5)) == ["b", "a"]

    def test_add_replaces_and_remove_drops(self) -> None:
        index = BM25Index()
        index.add("a", "alpha")
        index.add("a", "beta")
        assert index.search("alpha", top_k=5) == []
        assert _ids(index.search("beta", top_k=5)) == ["a"]
        index.remove("a")
        index.remove("unknown")
        assert len(index) == 0
        assert index.search("beta", top_k=5) == []

    def test_each_occurrence_of_a_query_term_counts(self) -> None:
        index = BM25Index()
        index.add("a", "alpha beta")
        index.add("b", "gamma")
        [(_, once)] = index.search("alpha", top_k=1)
        [(_, twice)] = index.search("alpha alpha", top_k=1)
        assert twice == pytest.approx(2 * once)

    def test_excluded_documents_take_no_part(self) -> None:
        index = BM25Index()
        index.add("builtin", "alpha alpha alpha")
        index.add("x", "alpha beta")
        index.add("y", "gamma")
        ranking = index.search("alpha", top_k=5, exclude={"builtin"})
        # Ranked among x and y only: N = 2 and the average length is 1.5.
        assert ranking == [("x", pytest.approx(_okapi(1, 2, 1.5, idf(2, 1))))]
        assert _ids(index.search("alpha", top_k=5)) == ["builtin", "x"]

    def test_exclude_must_not_be_a_str(self) -> None:
        index = BM25Index()
        index.add("a", "alpha")
        with pytest.raises(TypeError):
            index.search("alpha", top_k=5, exclude="a")
