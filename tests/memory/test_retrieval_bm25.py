"""The ``bm25`` retriever: IDF, the floor, kinds, the relative cut, caching.

Where a test checks scores, its docs all have the same number of terms and
name each query term at most once. Then every term's share of a score is
its IDF, ``ln(1 + (N - n + 0.5) / (n + 0.5))``, and each expected score is
a sum of IDFs, derived by hand in the comments.
"""

from __future__ import annotations

import math

import pytest

from localmelo.melo.memory.bm25 import tokenize
from localmelo.melo.memory.retrieval import Doc, Hit
from localmelo.melo.memory.retrieval.bm25 import (
    FLOOR_STOPWORDS,
    BM25Retriever,
    floor_units,
)


def _padded(words: str, length: int, tag: str) -> str:
    """*words* and filler words of their own, *length* terms in all."""
    terms = words.split()
    text = " ".join(terms + [f"{tag}{i}" for i in range(length - len(terms))])
    assert len(tokenize(text)) == length
    return text


def _idf(n_docs: int, doc_freq: int) -> float:
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def _ids(hits: list[Hit]) -> list[str]:
    return [hit.doc_id for hit in hits]


def _thought(doc_id: str, text: str, **meta: object) -> Doc:
    return Doc(doc_id, "thought", text, meta=meta)


# ── IDF ─────────────────────────────────────────────────────────────────────


async def test_the_doc_with_the_rarer_matching_term_ranks_first() -> None:
    # N = 4 docs of 4 terms: parser is in 2 of them, rare in 1, common in 3.
    docs = [
        Doc("d2", "note", "parser common pa0 pa1"),
        Doc("d1", "note", "parser rare pb0 pb1"),
        Doc("b1", "note", "common pc0 pc1 pc2"),
        Doc("b2", "note", "common pd0 pd1 pd2"),
    ]
    result = await BM25Retriever().search("parser rare common", docs, k=5)

    # b1 and b2 match one unit of the query: under the floor.
    assert _ids(result.hits) == ["d1", "d2"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    # d1: idf(4, 2) + idf(4, 1) = ln 2 + ln(10/3) = 1.89712
    # d2: idf(4, 2) + idf(4, 3) = ln 2 + ln(10/7) = 1.04982
    assert result.hits[0].score == pytest.approx(math.log(2) + math.log(10 / 3))
    assert result.hits[1].score == pytest.approx(math.log(2) + math.log(10 / 7))
    assert (result.retriever, result.fallback, result.usage) == ("bm25", "", {})


# ── Chinese ─────────────────────────────────────────────────────────────────


class TestChinese:
    # Both docs have 7 terms: 4 characters and 3 bigrams. They share every
    # character with the query; only "bigrams" shares its bigrams.
    DOCS = [Doc("chars", "note", "件文取读"), Doc("bigrams", "note", "读取文件")]

    async def test_shared_bigrams_beat_shared_characters(self) -> None:
        retriever = BM25Retriever(floor_terms=0, relative_cut=0)
        result = await retriever.search("读取文件", self.DOCS, k=5)
        assert _ids(result.hits) == ["bigrams", "chars"]
        # N = 2: each character is in both docs, idf(2, 2) = ln 1.2; each
        # bigram in one, idf(2, 1) = ln 2.
        assert result.hits[0].score == pytest.approx(
            4 * math.log(1.2) + 3 * math.log(2)
        )
        assert result.hits[1].score == pytest.approx(4 * math.log(1.2))

    async def test_the_floor_counts_bigrams_only(self) -> None:
        result = await BM25Retriever().search("读取文件", self.DOCS, k=5)
        assert _ids(result.hits) == ["bigrams"]

    async def test_single_characters_are_not_units(self) -> None:
        assert floor_units("读 取 文 件") == frozenset()
        assert floor_units("读取文件") == frozenset({"读取", "取文", "文件"})
        assert (await BM25Retriever().search("读 取 文 件", self.DOCS, k=5)).hits == []
        unfloored = BM25Retriever(floor_terms=0)
        assert (await unfloored.search("读 取 文 件", self.DOCS, k=5)).hits


# ── The floor ───────────────────────────────────────────────────────────────


class TestFloorUnits:
    def test_paths_and_file_names_are_blanked(self) -> None:
        assert floor_units("read list_ops.py") == frozenset()
        assert floor_units("run tests/test_list_ops.py and ./run.sh") == frozenset()
        assert floor_units(
            "Fix the foldr argument order in list_ops.py and ~/notes"
        ) == frozenset({"fix", "foldr", "argument", "order"})
        # 读取list_ops.py文件: the CJK characters end the file name.
        assert floor_units("读取list_ops.py文件") == frozenset({"读取", "文件"})

    def test_an_identifier_is_one_unit(self) -> None:
        assert floor_units("parse_config") == frozenset({"parse_config"})
        assert floor_units("__init__ _private_ x _") == frozenset({"init", "private"})
        # A dotted name that is not a file name is not blanked: its words
        # are units.
        assert floor_units("self.items functools.reduce") == frozenset(
            {"self", "items", "functools", "reduce"}
        )

    def test_stopwords_are_not_units(self) -> None:
        assert floor_units("the plan for the parser") == frozenset({"parser"})
        assert floor_units("read wrote ran observed outcome ok error") == frozenset()
        assert floor_units("我们然后读取") == frozenset({"们然", "后读", "读取"})
        assert floor_units("我们 然后 读取") == frozenset({"读取"})
        assert {"the", "plan", "draft", "然后", "因为"} <= FLOOR_STOPWORDS

    def test_units_are_lowercase_and_distinct(self) -> None:
        assert floor_units("Parser PARSER parser") == frozenset({"parser"})


class TestFloor:
    async def test_path_tokens_alone_give_no_hit(self) -> None:
        docs = [
            Doc("n1", "note", "read list_ops.py: 120 chars, sha 1a2b3c4d"),
            Doc("n2", "note", "wrote list_ops.py: 300 chars"),
        ]
        assert (await BM25Retriever().search("read list_ops.py", docs, k=5)).hits == []
        # floor_terms=0 turns the floor off: the parts of list_ops.py match.
        unfloored = await BM25Retriever(floor_terms=0).search(
            "read list_ops.py", docs, k=5
        )
        assert set(_ids(unfloored.hits)) == {"n1", "n2"}

    async def test_two_content_units_pass(self) -> None:
        docs = [
            Doc("both", "note", "foldr takes the accumulator last"),
            Doc("one", "note", "foldr is a fold from the right"),
        ]
        result = await BM25Retriever().search("foldr accumulator", docs, k=5)
        assert _ids(result.hits) == ["both"]

    async def test_an_identifier_counts_once(self) -> None:
        docs = [
            Doc("parts", "note", "parse the config with the loader"),
            Doc("whole", "note", "parse_config calls the loader"),
            Doc("alone", "note", "parse_config is called twice"),
        ]
        # Units: parse_config and loader. "parts" holds parse and config,
        # the identifier's parts, but not the identifier.
        result = await BM25Retriever().search("parse_config loader", docs, k=5)
        assert _ids(result.hits) == ["whole"]
        # One unit, though the identifier gives the tokenizer three terms.
        assert (await BM25Retriever().search("parse_config", docs, k=5)).hits == []

    async def test_stopwords_do_not_count(self) -> None:
        docs = [Doc("n1", "note", "the plan for the parser")]
        assert (
            await BM25Retriever().search("the plan for the parser", docs, k=5)
        ).hits == []
        assert (
            await BM25Retriever(floor_terms=0).search(
                "the plan for the parser", docs, k=5
            )
        ).hits

    async def test_a_query_with_one_unit_gives_no_hit(self) -> None:
        docs = [Doc("n1", "note", "the parser splits tokens")]
        assert (await BM25Retriever().search("parser", docs, k=5)).hits == []
        assert (await BM25Retriever().search("parser parser", docs, k=5)).hits == []
        found = await BM25Retriever(floor_terms=1).search("parser", docs, k=5)
        assert _ids(found.hits) == ["n1"]

    async def test_floor_terms_raises_the_bar(self) -> None:
        docs = [
            Doc("two", "note", "parser splits"),
            Doc("three", "note", "parser splits tokens"),
        ]
        result = await BM25Retriever(floor_terms=3, relative_cut=0).search(
            "parser splits tokens", docs, k=5
        )
        assert _ids(result.hits) == ["three"]


# ── Kinds and the relative cut ──────────────────────────────────────────────


class TestRelativeCut:
    QUERY = "alpha bravo charlie delta echo foxtrot golf hotel"
    # N = 9 docs of 8 terms. alpha and bravo are each in 5 docs, the other
    # six query words in 2: idf(9, 5) = ln(20/11), idf(9, 2) = ln 4.
    DOCS = [
        Doc("n1", "note", _padded(QUERY, 8, "pa")),
        Doc("n2", "note", _padded("alpha bravo", 8, "pb")),
        _thought(
            "t1", _padded("charlie delta echo foxtrot", 8, "pc"), task_id="t", step=1
        ),
        _thought("t2", _padded("golf hotel", 8, "pd"), task_id="t", step=2),
        _thought("t3", _padded("alpha bravo", 8, "pe"), task_id="t", step=3),
        # One unit each: under the floor, but they count in IDF.
        Doc("b1", "note", _padded("alpha", 8, "pf")),
        Doc("b2", "note", _padded("bravo", 8, "pg")),
        Doc("b3", "note", _padded("alpha", 8, "ph")),
        Doc("b4", "note", _padded("bravo", 8, "pi")),
    ]
    COMMON = math.log(20 / 11)
    RARE = math.log(4)
    SCORES = {
        "n1": 2 * COMMON + 6 * RARE,  # 9.51344
        "n2": 2 * COMMON,  # 1.19567: 0.126 of n1
        "t1": 4 * RARE,  # 5.54518
        "t2": 2 * RARE,  # 2.77259: 0.5 of t1
        "t3": 2 * COMMON,  # 1.19567: 0.216 of t1, 0.126 of n1
    }

    async def _ids_and_scores(self, relative_cut: float) -> list[tuple[str, float]]:
        retriever = BM25Retriever(relative_cut=relative_cut)
        result = await retriever.search(self.QUERY, self.DOCS, k=10)
        for hit in result.hits:
            assert hit.score == pytest.approx(self.SCORES[hit.doc_id])
        return [(hit.doc_id, hit.rank) for hit in result.hits]

    async def test_no_cut(self) -> None:
        assert await self._ids_and_scores(0) == [
            ("n1", 1),
            ("n2", 2),
            ("t1", 3),
            ("t2", 4),
            ("t3", 5),
        ]

    async def test_the_default_cut_is_per_group(self) -> None:
        # n2 is under 0.2 of n1. t3 is under 0.2 of n1 too, but chunks are
        # cut against their own top, t1, and t3 has 0.216 of it.
        assert await self._ids_and_scores(0.2) == [
            ("n1", 1),
            ("t1", 2),
            ("t2", 3),
            ("t3", 4),
        ]

    async def test_a_higher_cut_drops_chunks_against_their_top(self) -> None:
        assert await self._ids_and_scores(0.25) == [("n1", 1), ("t1", 2), ("t2", 3)]
        assert await self._ids_and_scores(0.6) == [("n1", 1), ("t1", 2)]


class TestKinds:
    async def test_a_chunk_ten_times_stronger_ranks_below_a_note(self) -> None:
        query = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        # N = 10 docs of 10 terms. alpha and bravo are each in 5 docs, the
        # chunk's eight words in 1: idf(10, 5) = ln 2, idf(10, 1) = ln(22/3).
        docs = [
            Doc("n1", "note", _padded("alpha bravo", 10, "pa")),
            _thought(
                "t1",
                _padded("charlie delta echo foxtrot golf hotel india juliet", 10, "pb"),
                task_id="t",
                step=1,
            ),
        ] + [
            Doc(f"b{i}", "note", _padded("alpha" if i % 2 else "bravo", 10, f"p{i}x"))
            for i in range(1, 9)
        ]
        result = await BM25Retriever().search(query, docs, k=5)
        assert _ids(result.hits) == ["n1", "t1"]
        note, chunk = result.hits
        assert note.score == pytest.approx(2 * math.log(2))  # 1.38629
        assert chunk.score == pytest.approx(8 * math.log(22 / 3))  # 15.93944
        assert chunk.score > 10 * note.score

    async def test_every_other_kind_ranks_above_chunks(self) -> None:
        docs = [
            _thought("t1", "foldr accumulator order", task_id="t", step=1),
            Doc("d1", "draft", "foldr accumulator " + " ".join(["x"] * 20)),
            Doc("r1", "reflection", "foldr accumulator " + " ".join(["y"] * 30)),
            Doc("g1", "digest", "foldr accumulator " + " ".join(["z"] * 40)),
            Doc("m1", "long", "foldr accumulator " + " ".join(["w"] * 50)),
        ]
        result = await BM25Retriever(relative_cut=0).search(
            "foldr accumulator order", docs, k=10
        )
        assert _ids(result.hits) == ["d1", "r1", "g1", "m1", "t1"]
        scores = [hit.score for hit in result.hits]
        assert scores[-1] > max(scores[:-1])
        assert scores[:-1] == sorted(scores[:-1], reverse=True)

    async def test_one_chunk_per_step(self) -> None:
        query = "foldr accumulator order"
        docs = [
            # c2 comes first, but c1 scores higher: c1 is the step's chunk.
            _thought(
                "c2", "foldr accumulator order and more words", task_id="t", step=2
            ),
            _thought("c1", "foldr accumulator order", task_id="t", step=2),
            _thought("c3", "foldr accumulator", task_id="t", step=3),
            _thought("c4", "foldr accumulator order again", task_id="u", step=2),
            # Without an integer step, a chunk is never merged.
            _thought("c5", "foldr accumulator here", task_id="t"),
            _thought("c6", "foldr accumulator there", task_id="t"),
            _thought("c7", "foldr accumulator text", task_id="t", step="2"),
            _thought("c8", "foldr accumulator flag", task_id="t", step=True),
            # A note of the same step is never merged with a chunk.
            Doc(
                "n1",
                "note",
                "foldr accumulator order plus words",
                meta={"task_id": "t", "step": 2},
            ),
        ]
        result = await BM25Retriever(relative_cut=0).search(query, docs, k=20)
        ids = _ids(result.hits)
        assert ids[0] == "n1"
        assert sorted(ids[1:]) == ["c1", "c3", "c4", "c5", "c6", "c7", "c8"]
        assert ids.index("c1") < ids.index("c3")


# ── Candidates and caching ──────────────────────────────────────────────────


class TestCandidates:
    QUERY = "foldr accumulator order"
    A = Doc("a", "note", "foldr accumulator order reversed")
    B = Doc("b", "note", "foldr accumulator only")
    C = Doc("c", "note", "foldr accumulator order order order")

    async def test_excluded_docs_take_no_part_and_never_return(self) -> None:
        retriever = BM25Retriever(relative_cut=0)
        excluded = await retriever.search(
            self.QUERY, [self.A, self.B, self.C], k=5, exclude={"c"}
        )
        fresh = await BM25Retriever(relative_cut=0).search(
            self.QUERY, [self.A, self.B], k=5
        )
        assert excluded.hits == fresh.hits
        assert "c" not in _ids(excluded.hits)
        # c would have changed the scores: it holds the query's terms.
        everything = await BM25Retriever(relative_cut=0).search(
            self.QUERY, [self.A, self.B, self.C], k=5
        )
        assert {hit.doc_id: hit.score for hit in everything.hits}["a"] != (
            {hit.doc_id: hit.score for hit in fresh.hits}["a"]
        )

    async def test_docs_indexed_earlier_but_absent_now_take_no_part(self) -> None:
        retriever = BM25Retriever(relative_cut=0)
        await retriever.search(self.QUERY, [self.A, self.B, self.C], k=5)
        later = await retriever.search(self.QUERY, [self.A, self.B], k=5)
        fresh = await BM25Retriever(relative_cut=0).search(
            self.QUERY, [self.A, self.B], k=5
        )
        assert later.hits == fresh.hits

    async def test_ties_keep_the_order_of_docs(self) -> None:
        x = Doc("x", "note", "foldr accumulator order")
        y = Doc("y", "note", "foldr accumulator order")
        retriever = BM25Retriever()
        first = await retriever.search(self.QUERY, [x, y], k=5)
        assert _ids(first.hits) == ["x", "y"]
        # The index added x first; this search's docs put y first.
        second = await retriever.search(self.QUERY, [y, x], k=5)
        assert _ids(second.hits) == ["y", "x"]
        assert second.hits[0].score == second.hits[1].score

    async def test_a_changed_text_is_indexed_again(self) -> None:
        retriever = BM25Retriever()
        old = Doc("a", "note", "foldr accumulator order")
        new = Doc("a", "note", "parser splits tokens")
        assert _ids((await retriever.search(self.QUERY, [old], k=5)).hits) == ["a"]
        assert _ids((await retriever.search("parser tokens", [new], k=5)).hits) == ["a"]
        assert (await retriever.search(self.QUERY, [new], k=5)).hits == []

    async def test_k_cuts_the_ranking(self) -> None:
        docs = [
            Doc(f"n{i}", "note", "foldr accumulator " + " ".join(["w"] * i))
            for i in range(1, 6)
        ]
        result = await BM25Retriever().search(self.QUERY, docs, k=2)
        assert _ids(result.hits) == ["n1", "n2"]
        assert [hit.rank for hit in result.hits] == [1, 2]
