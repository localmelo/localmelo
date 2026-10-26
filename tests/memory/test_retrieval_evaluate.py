"""evaluate(): metrics, latency, call order, token means, fallbacks, errors.

A scripted retriever answers each call from a script and moves a fake
clock, so every expected value here was worked out by hand.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet

import pytest

from localmelo.melo.memory.retrieval import (
    Doc,
    Hit,
    LabelledQuery,
    QueryResult,
    RetrievalResult,
    evaluate,
)
from localmelo.melo.memory.retrieval.bm25 import BM25Retriever
from localmelo.melo.memory.retrieval.evaluate import nearest_rank

# What a scripted call returns: hit ids, fallback and usage.
Answer = tuple[list[str], str, dict[str, int]]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Scripted:
    """A retriever that answers ``answer(query, i)`` for its i-th call.

    It records each call, writes ``search <query>`` to *log*, and moves
    *clock* by ``seconds[i]``.
    """

    def __init__(
        self,
        answer: Callable[[str, int], Answer] | None = None,
        *,
        clock: FakeClock | None = None,
        seconds: Sequence[float] = (),
        log: list[str] | None = None,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self.answer = answer or (lambda query, i: ([], "", {}))
        self.clock = clock
        self.seconds = seconds
        self.log = log
        self.calls: list[tuple[str, list[str], int, frozenset[str]]] = []

    async def search(
        self,
        query: str,
        docs: Sequence[Doc],
        *,
        k: int,
        exclude: AbstractSet[str] = frozenset(),
    ) -> RetrievalResult:
        i = len(self.calls)
        self.calls.append((query, [doc.doc_id for doc in docs], k, frozenset(exclude)))
        if self.log is not None:
            self.log.append(f"search {query}")
        if self.clock is not None:
            self.clock.now += self.seconds[i]
        ids, fallback, usage = self.answer(query, i)
        return RetrievalResult(
            hits=[Hit(doc_id, 1 / rank, rank) for rank, doc_id in enumerate(ids, 1)],
            retriever="scripted",
            ms=0,
            fallback=fallback,
            usage=usage,
        )


def _hits(table: dict[str, list[str]]) -> Callable[[str, int], Answer]:
    return lambda query, i: (table[query], "", {})


CORPUS = [Doc(doc_id, "note", f"text {doc_id}") for doc_id in "abcdefghixy"]


# ── Metrics ─────────────────────────────────────────────────────────────────


async def test_exact_metrics() -> None:
    queries = [
        LabelledQuery("q1", "first", ("a",)),
        LabelledQuery("q2", "second", ("d", "e")),
        LabelledQuery("q3", "third", ("f",)),
    ]
    retriever = Scripted(
        _hits(
            {
                "first": ["a", "b", "c"],  # rr 1, recall@1 1, recall@3 1
                "second": ["x", "d", "y"],  # rr 1/2, recall@1 0, recall@3 1/2
                "third": ["g", "h", "i"],  # rr 0, recall@1 0, recall@3 0
            }
        )
    )
    report = await evaluate(
        retriever, corpus=CORPUS, queries=queries, ks=(1, 3), repeats=0
    )
    assert [(q.qid, q.recall, q.rr) for q in report.queries] == [
        ("q1", {1: 1.0, 3: 1.0}, 1.0),
        ("q2", {1: 0.0, 3: 0.5}, 0.5),
        ("q3", {1: 0.0, 3: 0.0}, 0.0),
    ]
    assert report.recall == {1: 0.3333, 3: 0.5}
    assert report.mrr == 0.5
    assert (report.n, report.calls) == (3, 3)
    # Each call asks for max(ks) hits.
    assert {k for _query, _docs, k, _exclude in retriever.calls} == {3}


async def test_a_query_is_the_mean_of_its_calls() -> None:
    # Cold [a], then warm [b, a] and [b, c]: recall@1 is 1, 0, 0; recall@2
    # 1, 1, 0; rr 1, 1/2, 0.
    script = [["a"], ["b", "a"], ["b", "c"]]
    retriever = Scripted(lambda query, i: (script[i], "", {}))
    report = await evaluate(
        retriever,
        corpus=CORPUS,
        queries=[LabelledQuery("q", "query", ("a",))],
        ks=(1, 2),
        repeats=2,
    )
    [result] = report.queries
    assert result.recall == {1: 0.3333, 2: 0.6667}
    assert result.rr == 0.5
    assert report.recall == {1: 0.3333, 2: 0.6667}
    assert report.mrr == 0.5
    # The query line shows the cold call's hits.
    assert [hit.doc_id for hit in result.hits] == ["a"]


async def test_recall_counts_each_relevant_id_once() -> None:
    retriever = Scripted(_hits({"query": ["a", "b", "c"]}))
    report = await evaluate(
        retriever,
        corpus=CORPUS,
        queries=[LabelledQuery("q", "query", ("a", "d", "a"))],
        ks=(1, 3),
        repeats=0,
    )
    assert report.recall == {1: 0.5, 3: 0.5}


class TestNearestRank:
    @pytest.mark.parametrize(
        ("values", "p50", "p95"),
        [
            ([], None, None),
            ([7], 7, 7),
            ([3, 1, 2], 2, 3),
            ([10, 20], 10, 20),
            ([5, 1, 4, 2, 3], 3, 5),
            (list(range(20, 0, -1)), 10, 19),
            (list(range(1, 101)), 50, 95),
        ],
    )
    def test_percentiles(
        self, values: list[int], p50: int | None, p95: int | None
    ) -> None:
        assert nearest_rank(values, 50) == p50
        assert nearest_rank(values, 95) == p95


# ── Calls and latency ───────────────────────────────────────────────────────


async def test_cold_warm_and_interleaved_calls_with_a_fake_clock() -> None:
    clock = FakeClock()
    log: list[str] = []
    made: list[Scripted] = []
    # Seconds per call, cold first, then 2 warm and 2 interleaved calls; the
    # values are exact in binary, so the milliseconds are too.
    seconds = {
        "one": [0.25, 0.125, 0.25, 0.5, 0.75],
        "two": [0.5, 0.375, 0.5, 1.0, 0.75],
    }

    def factory() -> Scripted:
        query = ("one", "two")[len(made)]
        retriever = Scripted(clock=clock, seconds=seconds[query], log=log)
        made.append(retriever)
        return retriever

    async def interleave() -> None:
        log.append("interleave")
        clock.now += 60.0  # never timed

    def on_query(result: QueryResult) -> None:
        log.append(f"done {result.qid}")

    queries = [
        LabelledQuery("q1", "one", ("a",)),
        LabelledQuery("q2", "two", ("b",)),
    ]
    report = await evaluate(
        factory,
        corpus=CORPUS,
        queries=queries,
        repeats=2,
        interleave=interleave,
        on_query=on_query,
        clock=clock,
    )

    # A fresh retriever per query, each called 1 + 2 + 2 times.
    assert [len(retriever.calls) for retriever in made] == [5, 5]
    assert log == [
        "search one",
        "search one",
        "search one",
        "interleave",
        "search one",
        "interleave",
        "search one",
        "done q1",
        "search two",
        "search two",
        "search two",
        "interleave",
        "search two",
        "interleave",
        "search two",
        "done q2",
    ]
    q1, q2 = report.queries
    assert (q1.ms_cold, q1.ms_warm, q1.ms_interleaved) == (250, [125, 250], [500, 750])
    assert (q2.ms_cold, q2.ms_warm, q2.ms_interleaved) == (500, [375, 500], [1000, 750])
    # Cold [250, 500]; warm [125, 250, 375, 500]; interleaved [500, 750,
    # 750, 1000]. p50 is the ceil(0.5 n)-th smallest, p95 the ceil(0.95 n)-th.
    assert report.ms_cold_p50 == 250
    assert (report.ms_warm_p50, report.ms_warm_p95) == (250, 500)
    assert (report.ms_interleaved_p50, report.ms_interleaved_p95) == (750, 1000)
    assert (report.n, report.calls) == (2, 10)


async def test_a_shared_retriever_serves_every_query() -> None:
    retriever = Scripted()
    queries = [LabelledQuery(f"q{i}", f"query {i}", ("a",)) for i in range(3)]
    report = await evaluate(retriever, corpus=CORPUS, queries=queries, repeats=1)
    assert len(retriever.calls) == 6
    assert report.retriever == "scripted"
    # Without interleave, interleaved figures do not apply.
    assert report.queries[0].ms_interleaved == []
    assert (report.ms_interleaved_p50, report.ms_interleaved_p95) == (None, None)


async def test_a_class_is_a_factory() -> None:
    corpus = [
        Doc("t:n1", "note", "foldr argument order reversed"),
        Doc("t:n2", "note", "append works"),
    ]
    report = await evaluate(
        BM25Retriever,
        corpus=corpus,
        queries=[LabelledQuery("q", "foldr argument order", ("t:n1",))],
        repeats=1,
    )
    assert report.retriever == "bm25"
    assert report.recall == {1: 1.0, 3: 1.0, 5: 1.0}
    assert report.dim is None
    assert report.model is None


async def test_repeats_0_makes_only_the_cold_call() -> None:
    retriever = Scripted()
    report = await evaluate(
        retriever,
        corpus=CORPUS,
        queries=[LabelledQuery("q", "query", ("a",))],
        repeats=0,
        interleave=lambda: _never(),
    )
    assert len(retriever.calls) == 1
    assert (report.ms_warm_p50, report.ms_warm_p95) == (None, None)
    assert (report.ms_interleaved_p50, report.ms_interleaved_p95) == (None, None)


async def _never() -> None:
    raise AssertionError("interleave must not run with repeats 0")


async def test_an_exception_from_interleave_propagates() -> None:
    async def broken() -> None:
        raise RuntimeError("interleave failed")

    with pytest.raises(RuntimeError, match="interleave failed"):
        await evaluate(
            Scripted(),
            corpus=CORPUS,
            queries=[LabelledQuery("q", "query", ("a",))],
            repeats=1,
            interleave=broken,
        )


# ── Usage and fallbacks ─────────────────────────────────────────────────────


async def test_token_means_and_usage() -> None:
    usages = [
        {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        {"prompt_tokens": 100, "cached_tokens": 64, "completion_tokens": 6},
        {"prompt_tokens": 101, "cached_tokens": 0, "completion_tokens": 7},
    ]

    async def interleave() -> None:
        pass

    retriever = Scripted(lambda query, i: ([], "", usages[i]))
    report = await evaluate(
        retriever,
        corpus=CORPUS,
        queries=[LabelledQuery("q", "query", ("a",))],
        repeats=1,
        interleave=interleave,
    )
    # (100 + 100 + 101) / 3 = 100.33; cached over the two calls that report
    # it, (64 + 0) / 2; (5 + 6 + 7) / 3.
    assert report.prompt_tokens_mean == 100.3
    assert report.cached_tokens_mean == 32.0
    assert report.completion_tokens_mean == 6.0
    assert report.queries[0].usage == {
        "prompt_tokens": 301,
        "completion_tokens": 18,
        "total_tokens": 105,
        "cached_tokens": 64,
    }


async def test_no_usage_gives_no_token_means() -> None:
    report = await evaluate(
        Scripted(),
        corpus=CORPUS,
        queries=[LabelledQuery("q", "query", ("a",))],
    )
    assert (
        report.prompt_tokens_mean,
        report.cached_tokens_mean,
        report.completion_tokens_mean,
    ) == (None, None, None)
    assert report.queries[0].usage == {}


async def test_fallbacks_by_reason() -> None:
    # Four calls per query; "other" is no fallback reason and is not counted.
    reasons = {
        "one": ["timeout", "parse", "", "other"],
        "two": ["provider_error", "provider_error", "timeout", ""],
    }
    report = await evaluate(
        lambda: Scripted(lambda query, i: ([], reasons[query][i], {})),
        corpus=CORPUS,
        queries=[
            LabelledQuery("q1", "one", ("a",)),
            LabelledQuery("q2", "two", ("a",)),
        ],
        repeats=3,
    )
    assert [q.fallbacks for q in report.queries] == [
        {"timeout": 1, "parse": 1, "provider_error": 0},
        {"timeout": 1, "parse": 0, "provider_error": 2},
    ]
    assert report.fallbacks == {"timeout": 2, "parse": 1, "provider_error": 2}


# ── Candidates ──────────────────────────────────────────────────────────────

SCOPED = [
    Doc("t-en:n1", "note", "one", meta={"task_id": "t-en", "step": 1}),
    Doc("t-en:n2", "note", "two", meta={"task_id": "t-en", "step": 3}),
    Doc("t-en:d3", "draft", "three", meta={"task_id": "t-en", "step": 5}),
    Doc("t-en:x", "note", "no step", meta={"task_id": "t-en"}),
    Doc("t-zh:n1", "note", "一", meta={"task_id": "t-zh", "step": 1}),
    Doc("free", "note", "no meta"),
]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            LabelledQuery("a", "q", ("t-en:n1",), task_id="t-en", before_step=4),
            ["t-en:n1", "t-en:n2", "t-en:x"],
        ),
        (
            LabelledQuery("b", "q", ("free",)),
            ["t-en:n1", "t-en:n2", "t-en:d3", "t-en:x", "t-zh:n1", "free"],
        ),
        (LabelledQuery("c", "q", ("t-zh:n1",), task_id="t-zh"), ["t-zh:n1"]),
        (
            LabelledQuery("d", "q", ("free",), before_step=2),
            ["t-en:n1", "t-en:x", "t-zh:n1", "free"],
        ),
        (
            LabelledQuery("e", "q", ("t-en:n1",), task_id="t-en", before_step=1),
            ["t-en:x"],
        ),
    ],
)
def test_candidates(query: LabelledQuery, expected: list[str]) -> None:
    assert [doc.doc_id for doc in query.candidates(SCOPED)] == expected


async def test_a_search_gets_the_scoped_docs_and_the_exclusions() -> None:
    retriever = Scripted()
    query = LabelledQuery(
        "a",
        "the query",
        ("t-en:n1",),
        task_id="t-en",
        before_step=4,
        exclude_ids=frozenset({"t-en:n2", "t-zh:n1"}),
    )
    report = await evaluate(retriever, corpus=SCOPED, queries=[query], repeats=0)
    [(text, doc_ids, k, exclude)] = retriever.calls
    assert (text, doc_ids, k) == ("the query", ["t-en:n1", "t-en:n2", "t-en:x"], 5)
    assert exclude == {"t-en:n2", "t-zh:n1"}
    # Three candidates; t-en:n2 is excluded, and t-zh:n1 is not among them.
    assert report.queries[0].candidates == 2


# ── Errors ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"ks": ()}, "ks is empty"),
        ({"ks": (1, 0)}, "every k must be an integer >= 1, got 0"),
        ({"ks": (3, -1)}, "got -1"),
        ({"repeats": -1}, "repeats must be an integer >= 0, got -1"),
        ({"queries": []}, "no queries"),
        (
            {
                "queries": [
                    LabelledQuery("ok", "q", ("t-en:n1",), task_id="t-en"),
                    LabelledQuery("bad", "q", ("t-zh:n1",), task_id="t-en"),
                ]
            },
            "query 'bad': relevant id 't-zh:n1' is not a candidate",
        ),
        (
            {
                "queries": [
                    LabelledQuery(
                        "late", "q", ("t-en:d3",), task_id="t-en", before_step=5
                    )
                ]
            },
            "query 'late': relevant id 't-en:d3'",
        ),
        (
            {"queries": [LabelledQuery("gone", "q", ("t-en:n9",))]},
            "query 'gone': relevant id 't-en:n9'",
        ),
        (
            {"queries": [LabelledQuery("none", "q", ())]},
            "query 'none' has no relevant ids",
        ),
    ],
)
async def test_value_errors_come_before_any_search(
    options: dict[str, object], message: str
) -> None:
    retriever = Scripted()
    arguments: dict[str, object] = {
        "corpus": SCOPED,
        "queries": [LabelledQuery("q", "q", ("free",))],
        **options,
    }
    with pytest.raises(ValueError, match=message):
        await evaluate(retriever, **arguments)  # type: ignore[arg-type]
    assert retriever.calls == []


# ── Output lines ────────────────────────────────────────────────────────────


QUERY_KEYS = [
    "type",
    "qid",
    "source",
    "candidates",
    "relevant",
    "hits",
    "recall@1",
    "recall@3",
    "rr",
    "ms_cold",
    "ms_warm",
    "ms_interleaved",
    "fallbacks",
    "usage",
]
SUMMARY_KEYS = [
    "type",
    "retriever",
    "model",
    "dim",
    "n",
    "calls",
    "recall@1",
    "recall@3",
    "mrr",
    "ms_cold_p50",
    "ms_warm_p50",
    "ms_warm_p95",
    "ms_interleaved_p50",
    "ms_interleaved_p95",
    "prompt_tokens_mean",
    "cached_tokens_mean",
    "completion_tokens_mean",
    "fallbacks",
]


async def test_the_output_lines() -> None:
    clock = FakeClock()
    retriever = Scripted(
        lambda query, i: (["x", "a", "b"], "", {"prompt_tokens": 10}),
        clock=clock,
        seconds=[0.25, 0.125, 0.125, 0.5, 0.5],
    )
    retriever.dim = 16  # type: ignore[attr-defined]

    async def interleave() -> None:
        pass

    report = await evaluate(
        retriever,
        corpus=CORPUS,
        queries=[LabelledQuery("q1", "query", ("a", "c"), source="gold")],
        ks=(1, 3),
        repeats=2,
        interleave=interleave,
        model="qwen-test",
        clock=clock,
    )
    line = report.queries[0].to_line(report.ks)
    assert list(line) == QUERY_KEYS
    assert line == {
        "type": "query",
        "qid": "q1",
        "source": "gold",
        "candidates": 11,
        "relevant": ["a", "c"],
        "hits": [
            {"doc_id": "x", "score": 1.0, "rank": 1},
            {"doc_id": "a", "score": 0.5, "rank": 2},
            {"doc_id": "b", "score": 0.3333, "rank": 3},
        ],
        "recall@1": 0.0,
        "recall@3": 0.5,
        "rr": 0.5,
        "ms_cold": 250,
        "ms_warm": [125, 125],
        "ms_interleaved": [500, 500],
        "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0},
        "usage": {"prompt_tokens": 50},
    }
    summary = report.summary_line()
    assert list(summary) == SUMMARY_KEYS
    assert summary == {
        "type": "summary",
        "retriever": "scripted",
        "model": "qwen-test",
        "dim": 16,
        "n": 1,
        "calls": 5,
        "recall@1": 0.0,
        "recall@3": 0.5,
        "mrr": 0.5,
        "ms_cold_p50": 250,
        "ms_warm_p50": 125,
        "ms_warm_p95": 125,
        "ms_interleaved_p50": 500,
        "ms_interleaved_p95": 500,
        "prompt_tokens_mean": 10.0,
        "cached_tokens_mean": None,
        "completion_tokens_mean": None,
        "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0},
    }
    json.dumps(line, ensure_ascii=False)
    json.dumps(summary, ensure_ascii=False)


async def test_figures_that_do_not_apply_are_null() -> None:
    retriever = Scripted()
    retriever.dim = True  # type: ignore[attr-defined]
    report = await evaluate(
        retriever,
        corpus=CORPUS,
        queries=[LabelledQuery("q1", "query", ("a",))],
        ks=(5, 1),
        repeats=0,
    )
    line = report.queries[0].to_line(report.ks)
    assert [key for key in line if key.startswith("recall@")] == [
        "recall@5",
        "recall@1",
    ]
    assert (line["ms_warm"], line["ms_interleaved"]) == (None, None)
    summary = report.summary_line()
    assert summary["dim"] is None  # a bool is not a dimension
    assert summary["model"] is None
    for key in (
        "ms_warm_p50",
        "ms_warm_p95",
        "ms_interleaved_p50",
        "ms_interleaved_p95",
        "prompt_tokens_mean",
        "cached_tokens_mean",
        "completion_tokens_mean",
    ):
        assert summary[key] is None
    assert isinstance(summary["ms_cold_p50"], int)
