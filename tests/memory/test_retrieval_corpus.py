"""Corpus, labelled queries, raw calls, their documents and a run's history.db.

The fixtures in ``fixtures/retrieval`` are synthetic, written by hand in
English and Chinese. Every expected value here was derived by hand from the
rules of the R3 spec (Requirements 13 to 15) before the code ran; lengths
and sha256 prefixes were computed with ``wc`` and ``shasum``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.retrieval import (
    CorpusError,
    Doc,
    LabelledQuery,
    RawCall,
    doc_to_dict,
    docs_from_calls,
    load_corpus,
    load_queries,
    load_raw_calls,
)
from localmelo.melo.memory.retrieval.corpus import (
    NoCallsTableError,
    load_history_calls,
)
from localmelo.melo.schema import ToolCall, ToolResult
from localmelo.tests.memory.history_fixture import write_history_sync

FIXTURES = Path(__file__).parent / "fixtures" / "retrieval"


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").split("\n")


def _jsonl(*rows: object) -> list[str]:
    return [json.dumps(row, ensure_ascii=False) for row in rows]


def _meta(task_id: str, step: int, seq: int, created_at: float, **extra: Any) -> dict:
    return {
        "task_id": task_id,
        "step": step,
        "attempt": 1,
        "seq": seq,
        "created_at": created_at,
        **extra,
    }


def _heuristic(kind: str = "step", remember: bool = False) -> dict[str, Any]:
    return {
        "source": "heuristic",
        "note_kind": kind,
        "remember": remember,
        "remember_by": "heuristic",
    }


# The documents of fixtures/retrieval/raw_calls.jsonl, derived by hand: t-en
# first (its first line comes first), each task's calls by seq although the
# file has t-en seq 3 before seq 2; nothing from the retrieve call.
RAW_CALL_DOCS: list[dict[str, Any]] = [
    {
        "doc_id": "t-en:n0",
        "kind": "note",
        "title": (
            "Let me read list_ops.py to see which functions are stubs. "
            "→ read list_ops.py"
        ),
        "text": (
            "plan: Let me read list_ops.py to see which functions are stubs. "
            "→ read list_ops.py\n"
            "observed: read list_ops.py: 35 chars, sha 78cb9bac\n"
            "outcome: ok"
        ),
        "meta": _meta("t-en", 1, 0, 1761200000.0, **_heuristic()),
    },
    {
        "doc_id": "t-en:t0.1",
        "kind": "thought",
        "title": "The task asks for list operations without built-ins.",
        "text": (
            "The task asks for list operations without built-ins. "
            "Let me read list_ops.py to see which functions are stubs."
        ),
        "meta": _meta("t-en", 1, 0, 1761200000.0, chunk=1),
    },
    {
        "doc_id": "t-en:n1",
        "kind": "note",
        "title": "Let me check the tests before writing it. → read tests/test_list_ops.py",
        "text": (
            "plan: Let me check the tests before writing it. "
            "→ read tests/test_list_ops.py\n"
            "observed: read tests/test_list_ops.py: 57 chars, sha 2960b8d3\n"
            "outcome: ok"
        ),
        "meta": _meta("t-en", 2, 1, 1761200010.0, **_heuristic()),
    },
    {
        "doc_id": "t-en:d1",
        "kind": "draft",
        "title": "list_ops.py, 4 lines: def append(list1, list2):",
        "text": (
            "def append(list1, list2):\n"
            "    result = list1[:]\n"
            "    result.extend(list2)\n"
            "    return result"
        ),
        # Superseded by the draft of seq 2, for the same file.
        "meta": _meta(
            "t-en",
            2,
            1,
            1761200010.0,
            target="list_ops.py",
            status="superseded",
            lines=4,
            blocks=1,
            cut=False,
        ),
    },
    {
        "doc_id": "t-en:t1.1",
        "kind": "thought",
        "title": "append should keep both lists unchanged.",
        "text": (
            "append should keep both lists unchanged. A first version for "
            "list_ops.py:\n\nLet me check the tests before writing it."
        ),
        "meta": _meta("t-en", 2, 1, 1761200010.0, chunk=1),
    },
    {
        "doc_id": "t-en:n2",
        "kind": "note",
        "title": "write append to list_ops.py with concatenation",
        "text": (
            "plan: write append to list_ops.py with concatenation\n"
            "found: test_append expects a new list\n"
            "decided: use list1 + list2\n"
            "observed: wrote list_ops.py: 51 chars\n"
            "outcome: ok"
        ),
        "meta": _meta(
            "t-en",
            3,
            2,
            1761200020.0,
            source="model",
            note_kind="step",
            remember=True,
            remember_by="model",
        ),
    },
    {
        "doc_id": "t-en:d2",
        "kind": "draft",
        "title": "list_ops.py, 2 lines: def append(list1, list2):",
        "text": "def append(list1, list2):\n    return list1 + list2",
        # Its own call wrote both lines (string arguments, decoded).
        "meta": _meta(
            "t-en",
            3,
            2,
            1761200020.0,
            target="list_ops.py",
            status="written",
            lines=2,
            blocks=1,
            cut=False,
        ),
    },
    {
        "doc_id": "t-en:t2.1",
        "kind": "thought",
        "title": "The test wants a new list, so concatenation is enough.",
        "text": (
            "The test wants a new list, so concatenation is enough. The model "
            "must not emit &lt;tool_call&gt; here. Final version for list_ops.py:"
        ),
        "meta": _meta("t-en", 3, 2, 1761200020.0, chunk=1),
    },
    {
        # The reply holds a reflection (recommended_action "continue") after
        # two sentences, which the plan takes: both are intent sentences
        # with an action word or a file name. The JSON line has no marker.
        "doc_id": "t-en:r3",
        "kind": "reflection",
        "title": (
            "append now passes its test. Next I need to implement reverse in "
            "list_ops.py."
        ),
        "text": (
            "plan: append now passes its test. Next I need to implement reverse "
            "in list_ops.py."
        ),
        "meta": _meta("t-en", 3, 3, 1761200030.0, **_heuristic("reflection")),
    },
    {
        "doc_id": "t-en:t3.1",
        "kind": "thought",
        "title": "One function is done and one is left.",
        "text": "One function is done and one is left.",
        "meta": _meta("t-en", 3, 3, 1761200030.0, chunk=1),
    },
    {
        "doc_id": "t-zh:n0",
        "kind": "note",
        "title": (
            "我先读取 tests/test_list_ops.py，看看测试要求什么。 "
            "→ read tests/test_list_ops.py"
        ),
        "text": (
            "plan: 我先读取 tests/test_list_ops.py，看看测试要求什么。 "
            "→ read tests/test_list_ops.py\n"
            "observed: read tests/test_list_ops.py: 57 chars, sha af3a0c60\n"
            "outcome: ok"
        ),
        "meta": _meta("t-zh", 1, 0, 1761200005.0, **_heuristic()),
    },
    {
        "doc_id": "t-zh:t0.1",
        "kind": "thought",
        "title": "我先读取 tests/test_list_ops.py，看看测试要求什么。",
        "text": "我先读取 tests/test_list_ops.py，看看测试要求什么。",
        "meta": _meta("t-zh", 1, 0, 1761200005.0, chunk=1),
    },
    {
        "doc_id": "t-zh:n1",
        "kind": "note",
        "title": "接下来在 list_ops.py 里实现 reverse： → write list_ops.py",
        "text": (
            "plan: 接下来在 list_ops.py 里实现 reverse： → write list_ops.py\n"
            "observed: wrote list_ops.py: 43 chars\n"
            "outcome: ok"
        ),
        "meta": _meta("t-zh", 2, 1, 1761200015.0, **_heuristic()),
    },
    {
        "doc_id": "t-zh:d1",
        "kind": "draft",
        "title": "list_ops.py, 5 lines: def reverse(items):",
        "text": (
            "def reverse(items):\n"
            "    result = []\n"
            "    for item in items:\n"
            "        result.insert(0, item)\n"
            "    return result"
        ),
        # The write holds 1 of its 5 lines: 0.2, under 0.6.
        "meta": _meta(
            "t-zh",
            2,
            1,
            1761200015.0,
            target="list_ops.py",
            status="pending",
            lines=5,
            blocks=1,
            cut=False,
        ),
    },
    {
        "doc_id": "t-zh:t1.1",
        "kind": "thought",
        "title": "接下来在 list_ops.py 里实现 reverse：",
        "text": "接下来在 list_ops.py 里实现 reverse：",
        "meta": _meta("t-zh", 2, 1, 1761200015.0, chunk=1),
    },
    {
        "doc_id": "t-zh:n2",
        "kind": "note",
        "title": "现在运行测试。 → run python -m pytest -q",
        "text": (
            "plan: 现在运行测试。 → run python -m pytest -q\n"
            "observed: ran `python -m pytest -q`: 5 passed in 0.08s\n"
            "outcome: 5 passed in 0.08s"
        ),
        "meta": _meta("t-zh", 3, 2, 1761200025.0, **_heuristic()),
    },
    {
        "doc_id": "t-zh:t2.1",
        "kind": "thought",
        "title": "现在运行测试。",
        "text": "现在运行测试。",
        "meta": _meta("t-zh", 3, 2, 1761200025.0, chunk=1),
    },
    {
        "doc_id": "t-zh:n3",
        "kind": "note",
        "title": "因为切片会复制列表，所以空列表也能处理。",
        "text": "found: 因为切片会复制列表，所以空列表也能处理。",
        "meta": _meta(
            "t-zh", 4, 3, 1761200035.0, **_heuristic("answer", remember=True)
        ),
    },
    {
        "doc_id": "t-zh:t3.1",
        "kind": "thought",
        "title": "因为切片会复制列表，所以空列表也能处理。",
        "text": "因为切片会复制列表，所以空列表也能处理。",
        "meta": _meta("t-zh", 4, 3, 1761200035.0, chunk=1),
    },
]


# ── Corpus lines ────────────────────────────────────────────────────────────


def test_the_fixture_corpus_loads() -> None:
    docs = load_corpus(_lines("corpus.jsonl"))
    by_task = [doc.meta["task_id"] for doc in docs]
    assert (by_task.count("t-en"), by_task.count("t-zh")) == (8, 6)
    assert docs[0] == Doc(
        doc_id="t-en:n0",
        kind="note",
        title=(
            "Let me read list_ops.py to see which functions are stubs. "
            "→ read list_ops.py"
        ),
        text=(
            "plan: Let me read list_ops.py to see which functions are stubs. "
            "→ read list_ops.py\nobserved: read list_ops.py: 35 chars, sha 78cb9bac\n"
            "outcome: ok"
        ),
        meta={"task_id": "t-en", "step": 1, "source": "heuristic"},
    )
    assert {doc.kind for doc in docs} == {"note", "draft", "thought"}


def test_a_corpus_line_keeps_its_defaults_and_ignores_unknown_keys() -> None:
    [doc] = load_corpus(
        ["", *_jsonl({"doc_id": "x:n1", "kind": "long", "text": "t", "rank": 3}), " "]
    )
    assert doc == Doc(doc_id="x:n1", kind="long", text="t", title="", meta={})


def test_doc_to_dict_writes_the_keys_in_order_and_reads_back() -> None:
    docs = load_corpus(_lines("corpus.jsonl"))
    lines = [json.dumps(doc_to_dict(doc), ensure_ascii=False) for doc in docs]
    assert list(doc_to_dict(docs[0])) == ["doc_id", "kind", "title", "text", "meta"]
    assert load_corpus(lines) == docs


@pytest.mark.parametrize(
    ("rows", "line", "message"),
    [
        (["not json"], 1, "not JSON: Expecting value at column 1"),
        (["[1, 2]"], 1, "not a JSON object"),
        (_jsonl({"kind": "note", "text": "t"}), 1, "doc_id is missing"),
        (
            _jsonl({"doc_id": "", "kind": "note", "text": "t"}),
            1,
            "doc_id must be a non-empty string without whitespace",
        ),
        (
            _jsonl({"doc_id": "a b", "kind": "note", "text": "t"}),
            1,
            "doc_id must be a non-empty string without whitespace",
        ),
        (
            _jsonl({"doc_id": 7, "kind": "note", "text": "t"}),
            1,
            "doc_id must be a non-empty string without whitespace",
        ),
        (
            # Blank lines count: the repeat is on line 4.
            [*_jsonl({"doc_id": "a", "kind": "note", "text": "t"}), "", "  "]
            + _jsonl({"doc_id": "a", "kind": "draft", "text": "u"}),
            4,
            "doc_id 'a' is repeated (first on line 1)",
        ),
        (_jsonl({"doc_id": "a", "text": "t"}), 1, "kind is missing"),
        (
            _jsonl({"doc_id": "a", "kind": "memo", "text": "t"}),
            1,
            "kind must be one of note, draft, thought, digest, reflection, long, "
            "got 'memo'",
        ),
        (_jsonl({"doc_id": "a", "kind": "note"}), 1, "text is missing"),
        (
            _jsonl({"doc_id": "a", "kind": "note", "text": ""}),
            1,
            "text must be a non-empty string",
        ),
        (
            _jsonl({"doc_id": "a", "kind": "note", "text": 5}),
            1,
            "text must be a non-empty string",
        ),
        (
            _jsonl({"doc_id": "a", "kind": "note", "text": "t", "title": 5}),
            1,
            "title must be a string",
        ),
        (
            _jsonl({"doc_id": "a", "kind": "note", "text": "t", "meta": []}),
            1,
            "meta must be an object",
        ),
        (
            _jsonl(
                {"doc_id": "a", "kind": "note", "text": "t", "meta": {"task_id": 1}}
            ),
            1,
            "meta.task_id must be a string",
        ),
        (
            _jsonl({"doc_id": "a", "kind": "note", "text": "t", "meta": {"step": -1}}),
            1,
            "meta.step must be an integer >= 0",
        ),
        (
            _jsonl(
                {"doc_id": "a", "kind": "note", "text": "t", "meta": {"step": True}}
            ),
            1,
            "meta.step must be an integer >= 0",
        ),
        (
            _jsonl({"doc_id": "a", "kind": "note", "text": "t", "meta": {"step": 1.5}}),
            1,
            "meta.step must be an integer >= 0",
        ),
    ],
)
def test_corpus_errors_name_the_line(rows: list[str], line: int, message: str) -> None:
    with pytest.raises(CorpusError) as caught:
        load_corpus(rows)
    assert (caught.value.line, str(caught.value)) == (line, message)


# ── Labelled queries ────────────────────────────────────────────────────────


def test_short_ids_resolve_against_the_scope() -> None:
    corpus = load_corpus(_lines("corpus.jsonl"))
    queries = load_queries(_lines("queries.jsonl"), corpus)
    assert queries == [
        LabelledQuery(
            qid="en-append",
            query="how does append build a new list from list1 and list2",
            relevant=("t-en:d2", "t-en:n2"),
            grades={"t-en:d2": 2, "t-en:n2": 1},
            task_id="t-en",
            source="gold",
        ),
        LabelledQuery(
            qid="zh-reverse",
            query="reverse 函数 切片 实现",
            relevant=("t-zh:d1", "t-zh:n1"),
            task_id="t-zh",
            source="gold",
        ),
        LabelledQuery(
            qid="en-foldr",
            query="foldr accumulator order from the right",
            relevant=("t-en:t3.1", "t-en:n3"),
            source="gold",
        ),
        LabelledQuery(
            qid="en-append-excluded",
            query="append concatenation list1 list2",
            relevant=("t-en:n2",),
            task_id="t-en",
            exclude_ids=frozenset({"t-en:d2"}),
            source="gold",
        ),
        LabelledQuery(
            qid="en-tests-before-3",
            query="which tests does append need",
            relevant=("t-en:n1",),
            grades={"t-en:n1": 1},
            task_id="t-en",
            before_step=3,
            source="gold",
        ),
        LabelledQuery(
            qid="zh-empty",
            query="空列表 切片 处理",
            relevant=("t-zh:n2", "t-zh:t2.1"),
            grades={"t-zh:n2": 2},
            task_id="t-zh",
            before_step=4,
            source="silver",
        ),
    ]
    # The scope gives the candidates: 4 docs of t-en before step 3.
    assert [doc.doc_id for doc in queries[4].candidates(corpus)] == [
        "t-en:n0",
        "t-en:t0.1",
        "t-en:n1",
        "t-en:d1",
    ]


def test_a_query_line_keeps_its_defaults_and_counts_a_repeated_id_once() -> None:
    corpus = [Doc("t:n1", "note", "a"), Doc("t:n2", "note", "b")]
    [query] = load_queries(
        _jsonl({"qid": "q", "query": "a", "relevant": ["t:n1", "t:n1"], "x": 1}),
        corpus,
    )
    assert query == LabelledQuery(qid="q", query="a", relevant=("t:n1",))


# Docs of two tasks, for the query errors.
SCOPED = [
    Doc("t:n1", "note", "a", meta={"task_id": "t", "step": 1}),
    Doc("t:n2", "note", "b", meta={"task_id": "t", "step": 2}),
    Doc("u:n1", "note", "c", meta={"task_id": "u", "step": 1}),
]


def _query(**fields: Any) -> dict[str, Any]:
    return {"qid": "q", "query": "a b", "relevant": ["t:n1"], **fields}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"query": "a", "relevant": ["t:n1"]}, "qid is missing"),
        (_query(qid=""), "qid must be a non-empty string"),
        (_query(qid=3), "qid must be a non-empty string"),
        ({"qid": "q", "relevant": ["t:n1"]}, "query is missing"),
        (_query(query="  "), "query must be a non-blank string"),
        (_query(query=3), "query must be a non-blank string"),
        ({"qid": "q", "query": "a"}, "relevant is missing"),
        (_query(relevant=[]), "relevant must be a non-empty list of ids"),
        (_query(relevant="t:n1"), "relevant must be a non-empty list of ids"),
        (_query(relevant=[1]), "relevant must be a non-empty list of ids"),
        (_query(relevant=["t n1"]), "relevant must be a non-empty list of ids"),
        (_query(scope=[]), "scope must be an object"),
        (
            _query(scope={"task_id": "t", "before": 2}),
            "scope has an unknown key 'before'; it takes task_id, before_step and "
            "exclude_ids",
        ),
        (_query(scope={"task_id": ""}), "scope.task_id must be a non-empty string"),
        (_query(scope={"task_id": 3}), "scope.task_id must be a non-empty string"),
        (
            _query(scope={"task_id": "t", "before_step": 0}),
            "scope.before_step must be an integer >= 1",
        ),
        (
            _query(scope={"task_id": "t", "before_step": True}),
            "scope.before_step must be an integer >= 1",
        ),
        (_query(scope={"before_step": 2}), "scope.before_step needs scope.task_id"),
        (
            _query(scope={"task_id": "t", "exclude_ids": "n2"}),
            "scope.exclude_ids must be a list of ids",
        ),
        (
            _query(scope={"task_id": "t", "exclude_ids": [2]}),
            "scope.exclude_ids must be a list of ids",
        ),
        (_query(grades=[]), "grades must be an object"),
        (
            _query(grades={"t:n2": 1}),
            "grades names 't:n2', which is not a relevant id",
        ),
        (_query(grades={"t:n1": 0}), "grades['t:n1'] must be an integer >= 1"),
        (_query(grades={"t:n1": True}), "grades['t:n1'] must be an integer >= 1"),
        (_query(source=3), "source must be a string"),
        # A short id needs scope.task_id, in each of the three places.
        (
            _query(relevant=["n1"]),
            "relevant id 'n1' has no task prefix, and the query has no "
            "scope.task_id",
        ),
        (
            _query(scope={"exclude_ids": ["n2"]}),
            "exclude_ids id 'n2' has no task prefix, and the query has no "
            "scope.task_id",
        ),
        (
            _query(grades={"n1": 1}),
            "grades id 'n1' has no task prefix, and the query has no scope.task_id",
        ),
        # Every relevant id is in the corpus and a candidate of its query.
        (_query(relevant=["t:n9"]), "relevant id 't:n9' is not in the corpus"),
        (
            _query(scope={"task_id": "t"}, relevant=["u:n1"]),
            "relevant id 'u:n1' is not a candidate of the query: its task is 'u', "
            "not 't'",
        ),
        (
            _query(scope={"task_id": "t", "before_step": 2}, relevant=["n1", "n2"]),
            "relevant id 't:n2' is not a candidate of the query: its step 2 is not "
            "before 2",
        ),
    ],
)
def test_query_errors_name_the_line(row: dict[str, Any], message: str) -> None:
    lines = ["", *_jsonl(row)]
    with pytest.raises(CorpusError) as caught:
        load_queries(lines, SCOPED)
    assert (caught.value.line, str(caught.value)) == (2, message)


def test_a_repeated_qid_names_both_lines() -> None:
    lines = _jsonl(_query(), _query(relevant=["t:n2"]))
    with pytest.raises(CorpusError) as caught:
        load_queries(lines, SCOPED)
    assert (caught.value.line, str(caught.value)) == (
        2,
        "qid 'q' is repeated (first on line 1)",
    )


# ── Raw-call lines ──────────────────────────────────────────────────────────


def test_a_minimal_raw_call_takes_every_default() -> None:
    [call] = load_raw_calls(_jsonl({"task_id": "t", "seq": 0}))
    assert call == RawCall(
        task_id="t",
        seq=0,
        kind="plan",
        step=0,
        attempt=1,
        content="",
        thinking="",
        tool_calls=(),
        tool_results=(),
        outcome="empty",
        timestamp=0.0,
    )


@pytest.mark.parametrize(
    ("fields", "outcome"),
    [
        ({"tool_calls": [{"tool_name": "file_read"}]}, "tools"),
        ({"content": "Done."}, "answer"),
        ({"content": " \n "}, "empty"),
        ({"content": "<note>plan: x</note>"}, "answer"),
        ({"kind": "reflect", "content": "Continue."}, "aux"),
        ({"kind": "retrieve"}, "aux"),
        ({"kind": "estimate"}, "aux"),
        # A given outcome is kept.
        ({"content": "Done.", "outcome": "empty"}, "empty"),
        ({"kind": "reflect", "outcome": "answer"}, "answer"),
    ],
)
def test_the_outcome_is_derived_when_absent(
    fields: dict[str, Any], outcome: str
) -> None:
    [call] = load_raw_calls(_jsonl({"task_id": "t", "seq": 0, **fields}))
    assert call.outcome == outcome


def test_null_step_and_attempt_read_as_the_defaults() -> None:
    [call] = load_raw_calls(
        _jsonl({"task_id": "t", "seq": 4, "step": None, "attempt": None})
    )
    assert (call.step, call.attempt) == (0, 1)


def test_calls_and_results_take_the_steps_shapes() -> None:
    row = {
        "task_id": "t",
        "seq": 1,
        "tool_calls": [
            {"tool_name": "file_read", "arguments": {"path": "a.py"}, "id": "c1"},
            {
                "tool_name": "file_write",
                "arguments": '{"path": "b.py", "content": "x"}',
            },
            {"tool_name": "shell_exec", "arguments": "ls -la"},
            {"tool_name": "shell_exec", "arguments": "[1, 2]"},
            {"tool_name": "shell_exec", "arguments": ""},
            {"tool_name": "mail_search"},
        ],
        "tool_results": [
            {"output": "x = 1\n", "duration_ms": 3},
            {"tool_name": "file_write", "error": "Permission denied", "images": []},
        ],
    }
    [call] = load_raw_calls(_jsonl(row))
    assert call.tool_calls == (
        ToolCall("file_read", {"path": "a.py"}),
        ToolCall("file_write", {"path": "b.py", "content": "x"}),
        ToolCall("shell_exec", {"_raw_arguments": "ls -la"}),
        ToolCall("shell_exec", {"_raw_arguments": "[1, 2]"}),
        ToolCall("shell_exec", {"_raw_arguments": ""}),
        ToolCall("mail_search", {}),
    )
    assert call.tool_calls[0].id == "c1"
    # A result's tool_name defaults to its call's.
    assert call.tool_results == (
        ToolResult("file_read", output="x = 1\n", duration_ms=3.0),
        ToolResult("file_write", error="Permission denied"),
    )
    assert call.outcome == "tools"


def _call(**fields: Any) -> dict[str, Any]:
    return {"task_id": "t", "seq": 0, **fields}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"seq": 0}, "task_id is missing"),
        (
            _call(task_id=""),
            "task_id must be a non-empty string without whitespace",
        ),
        (
            _call(task_id="t en"),
            "task_id must be a non-empty string without whitespace",
        ),
        (_call(task_id=3), "task_id must be a non-empty string without whitespace"),
        ({"task_id": "t"}, "seq is missing"),
        (_call(seq=-1), "seq must be an integer >= 0"),
        (_call(seq="1"), "seq must be an integer >= 0"),
        (_call(seq=True), "seq must be an integer >= 0"),
        (
            _call(kind="chat"),
            "kind must be one of plan, reflect, retrieve, estimate, got 'chat'",
        ),
        (_call(step=-1), "step must be an integer >= 0"),
        (_call(step="2"), "step must be an integer >= 0"),
        (_call(attempt=-1), "attempt must be an integer >= 0"),
        (_call(content=3), "content must be a string"),
        (_call(content=None), "content must be a string"),
        (_call(thinking=["a"]), "thinking must be a string"),
        (_call(tool_calls={}), "tool_calls must be a list"),
        (_call(tool_calls=[3]), "tool_calls[0] must be an object"),
        (_call(tool_calls=[{"arguments": {}}]), "tool_calls[0].tool_name is missing"),
        (
            _call(tool_calls=[{"tool_name": 3}]),
            "tool_calls[0].tool_name must be a string",
        ),
        (
            _call(tool_calls=[{"tool_name": "a", "id": 3}]),
            "tool_calls[0].id must be a string",
        ),
        (
            _call(tool_calls=[{"tool_name": "a"}, {"tool_name": "b", "arguments": 3}]),
            "tool_calls[1].arguments must be an object or a string",
        ),
        (
            _call(tool_calls=[{"tool_name": "a", "arguments": None}]),
            "tool_calls[0].arguments must be an object or a string",
        ),
        (_call(tool_results={}), "tool_results must be a list"),
        (
            _call(tool_calls=[{"tool_name": "a"}], tool_results=[{}, {}]),
            "tool_results has 2 items, more than the 1 tool_calls",
        ),
        (
            _call(tool_calls=[{"tool_name": "a"}], tool_results=["ok"]),
            "tool_results[0] must be an object",
        ),
        (
            _call(tool_calls=[{"tool_name": "a"}], tool_results=[{"output": 3}]),
            "tool_results[0].output must be a string",
        ),
        (
            _call(tool_calls=[{"tool_name": "a"}], tool_results=[{"error": None}]),
            "tool_results[0].error must be a string",
        ),
        (
            _call(tool_calls=[{"tool_name": "a"}], tool_results=[{"tool_name": 1}]),
            "tool_results[0].tool_name must be a string",
        ),
        (
            _call(tool_calls=[{"tool_name": "a"}], tool_results=[{"duration_ms": "1"}]),
            "tool_results[0].duration_ms must be a finite number",
        ),
        (
            _call(outcome="done"),
            "outcome must be one of tools, answer, empty, aux, got 'done'",
        ),
        (
            _call(outcome=None),
            "outcome must be one of tools, answer, empty, aux, got None",
        ),
        (
            _call(outcome="aux"),
            "a plan call's outcome is tools, answer or empty, not aux",
        ),
        (_call(timestamp="now"), "timestamp must be a finite number"),
        (_call(timestamp=True), "timestamp must be a finite number"),
    ],
)
def test_raw_call_errors_name_the_line(row: dict[str, Any], message: str) -> None:
    lines = [*_jsonl({"task_id": "other", "seq": 0}), *_jsonl(row)]
    with pytest.raises(CorpusError) as caught:
        load_raw_calls(lines)
    assert (caught.value.line, str(caught.value)) == (2, message)


def test_a_non_finite_timestamp_is_refused() -> None:
    # Python's json reads Infinity and NaN, which JSON itself has no form for.
    for text in ("Infinity", "NaN", "1e999"):
        with pytest.raises(CorpusError) as caught:
            load_raw_calls([f'{{"task_id": "t", "seq": 0, "timestamp": {text}}}'])
        assert str(caught.value) == "timestamp must be a finite number"


def test_a_repeated_seq_of_a_task_names_both_lines() -> None:
    lines = [
        *_jsonl(_call(), {"task_id": "u", "seq": 0}),
        "",
        *_jsonl(_call(kind="reflect")),
    ]
    with pytest.raises(CorpusError) as caught:
        load_raw_calls(lines)
    assert (caught.value.line, str(caught.value)) == (
        4,
        "seq 0 of task 't' is repeated (first on line 1)",
    )


# ── Documents from calls ────────────────────────────────────────────────────


def test_docs_from_the_fixture_raw_calls() -> None:
    docs = docs_from_calls(load_raw_calls(_lines("raw_calls.jsonl")))
    assert [doc_to_dict(doc) for doc in docs] == RAW_CALL_DOCS


def test_the_fixture_docs_in_short() -> None:
    calls = load_raw_calls(_lines("raw_calls.jsonl"))
    docs = {doc.doc_id: doc for doc in docs_from_calls(calls)}
    # The model's note against the heuristic's.
    assert docs["t-en:n2"].meta["source"] == "model"
    assert docs["t-en:n0"].meta["source"] == "heuristic"
    # Drafts: superseded, written, pending.
    assert [docs[i].meta["status"] for i in ("t-en:d1", "t-en:d2", "t-zh:d1")] == [
        "superseded",
        "written",
        "pending",
    ]
    # The reflection, and no document from the retrieve call (seq 4).
    assert docs["t-en:r3"].kind == "reflection"
    assert not [
        doc
        for doc in docs.values()
        if (doc.meta["task_id"], doc.meta["seq"]) == ("t-en", 4)
    ]
    # Sanitized text.
    assert "<tool_call>" not in docs["t-en:t2.1"].text
    assert "&lt;tool_call&gt;" in docs["t-en:t2.1"].text


def test_a_document_whose_text_is_empty_is_left_out() -> None:
    # No note fields, no code block, and a chunk the sanitizer empties.
    assert docs_from_calls([RawCall("t", 0, thinking="<think>")]) == []
    assert docs_from_calls([RawCall("t", 0, kind="reflect")]) == []


REFLECT_THINKING = "Maybe the fix is to append. Therefore I must edit list_ops.py."


@pytest.mark.parametrize(
    "content",
    [
        "Maybe the fix is to append. Therefore I must edit list_ops.py.",
        '{"summary": "append is done."}',
        '{"recommended_action": "retry", "summary": "append is done."}',
        '{"recommended_action": "Done"}',
        '[{"recommended_action": "done"}]',
    ],
)
def test_a_reflect_reply_without_a_reflection_gives_only_its_chunks(
    content: str,
) -> None:
    # The note would not be empty: the thinking has a hedge and a decision.
    call = RawCall("t", 1, kind="reflect", content=content, thinking=REFLECT_THINKING)
    assert [doc.doc_id for doc in docs_from_calls([call])] == ["t:t1.1"]


def test_a_reflect_reply_with_a_reflection_gives_it() -> None:
    content = '{"recommended_action": "done", "summary": "append is done."}'
    call = RawCall("t", 1, kind="reflect", content=content, thinking=REFLECT_THINKING)
    docs = docs_from_calls([call])
    assert [(doc.doc_id, doc.kind) for doc in docs] == [
        ("t:r1", "reflection"),
        ("t:t1.1", "thought"),
    ]


def _reflect_replies() -> list[str]:
    calls = load_raw_calls(_lines("raw_calls.jsonl"))
    return [call.content for call in calls if call.kind == "reflect"]


# A number the decoder cannot convert: more digits than int() takes.
HUGE_NUMBER = "1" * 4_301

REFLECTION_EDGE_CASES = [
    "",
    "append now passes its test.",
    '{"recommended_action": "continue"}',
    'Done. {"recommended_action": "stop", "summary": "s"} trailing prose',
    # A tool call written out as text, then the reflection.
    '{"tool_name": "file_read", "arguments": {"path": "a.py"}}'
    '{"recommended_action": "done"}',
    '{"tool_name": "file_read", "arguments": {"path": "a.py"}} then prose',
    # Nested inside another object, as the whole reply and within prose.
    '{"wrap": {"recommended_action": "done"}}',
    'See {"wrap": {"recommended_action": "done"}} here.',
    '{"recommended_action": 1}',
    '{"recommended_action": null}',
    '{"recommended_action": true}',
    '{"recommended_action": ["done"]}',
    '{"recommended_action": "retry"}',
    '{"recommended_action": "Done"}',
    # JSON that is not an object.
    '["done"]',
    '[{"recommended_action": "done"}]',
    '"done"',
    "42",
    # A number the decoder cannot convert, as the whole reply, inside the
    # reflection, and before it.
    HUGE_NUMBER,
    '{"recommended_action": "done", "n": ' + HUGE_NUMBER + "}",
    '{"n": ' + HUGE_NUMBER + '} {"recommended_action": "done"}',
    # Fenced JSON, with and without a language tag.
    '```json\n{"recommended_action": "decompose"}\n```',
    '```\n{"recommended_action": "continue"}\n```',
    # A brace that starts no object, then the reflection.
    'a { b {"recommended_action": "done"}',
    # An unclosed object; and a brace in a string, which the decoder reads
    # as a key that runs past the reflection's own brace.
    '{"recommended_action": "done"',
    'Use "{" carefully. {"recommended_action": "stop"}',
]


@pytest.mark.parametrize("reply", [*_reflect_replies(), *REFLECTION_EDGE_CASES])
def test_the_reflection_test_agrees_with_the_agents(reply: str) -> None:
    # corpus.py keeps a copy of the agent's test, since melo/memory does not
    # import melo.agent (R3 spec, Requirement 13); tests may.
    from localmelo.melo.agent.agent import _find_reflection as agent_test
    from localmelo.melo.memory.retrieval.corpus import _find_reflection

    assert _find_reflection(reply) == agent_test(reply)


def test_the_reflection_test_on_the_edge_cases() -> None:
    from localmelo.melo.memory.retrieval.corpus import _find_reflection

    held = [reply for reply in REFLECTION_EDGE_CASES if _find_reflection(reply)]
    assert held == [
        '{"recommended_action": "continue"}',
        'Done. {"recommended_action": "stop", "summary": "s"} trailing prose',
        '{"tool_name": "file_read", "arguments": {"path": "a.py"}}'
        '{"recommended_action": "done"}',
        '```json\n{"recommended_action": "decompose"}\n```',
        '```\n{"recommended_action": "continue"}\n```',
        'a { b {"recommended_action": "done"}',
    ]
    assert [_find_reflection(reply) is not None for reply in _reflect_replies()] == [
        True
    ]


def test_retrieve_and_estimate_calls_give_nothing() -> None:
    calls = [
        RawCall("t", 0, kind="retrieve", thinking="Let me read a.py now."),
        RawCall("t", 1, kind="estimate", content="Let me read a.py now."),
    ]
    assert docs_from_calls(calls) == []


def _read(path: str, error: str = "") -> tuple[ToolCall, ToolResult]:
    return ToolCall("file_read", {"path": path}), ToolResult(
        "file_read", output="x", error=error
    )


def _write(path: str, content: str, error: str = "") -> tuple[ToolCall, ToolResult]:
    return ToolCall("file_write", {"path": path, "content": content}), ToolResult(
        "file_write", output="ok", error=error
    )


def _plan(seq: int, thinking: str, *pairs: tuple[ToolCall, ToolResult]) -> RawCall:
    return RawCall(
        "t",
        seq,
        thinking=thinking,
        tool_calls=tuple(call for call, _ in pairs),
        tool_results=tuple(result for _, result in pairs),
    )


DRAFT = "Here it is:\n```python\nx = 1\ny = 2\n```"


def _drafts(calls: list[RawCall]) -> dict[str, tuple[str, str]]:
    return {
        doc.doc_id: (doc.meta["target"], doc.meta["status"])
        for doc in docs_from_calls(calls)
        if doc.kind == "draft"
    }


def test_the_target_falls_back_to_the_latest_read_or_write_of_an_earlier_call() -> None:
    calls = [
        _plan(0, "Let me look.", _read("src/app.py"), _read("src/lib.py")),
        # A failed read sets nothing.
        _plan(1, "Again.", _read("src/cli.py", error="No such file")),
        # This call's own read comes too late for its draft.
        _plan(2, DRAFT, _read("src/db.py")),
    ]
    assert _drafts(calls) == {"t:d2": ("src/lib.py", "pending")}


def test_a_later_write_marks_a_pending_draft_written() -> None:
    calls = [
        _plan(0, "Write it to app.py:\n```python\nx = 1\ny = 2\n```"),
        # A failed write, and one that holds too little, write nothing.
        _plan(1, "Trying.", _write("app.py", "x = 1\ny = 2\n", error="Disk full")),
        _plan(2, "Again.", _write("./app.py", "x = 1\n")),
        _plan(3, "Now.", _write("/work/app.py", "x = 1\ny = 2\n")),
    ]
    assert _drafts(calls) == {"t:d0": ("app.py", "written")}
    assert _drafts(calls[:3]) == {"t:d0": ("app.py", "pending")}


def test_a_new_draft_supersedes_only_pending_drafts_for_the_same_file() -> None:
    calls = [
        _plan(0, "For app.py:\n```python\nold = 1\n```"),
        _plan(1, "For lib.py:\n```python\nb = 1\n```", _write("lib.py", "b = 1\n")),
        _plan(2, "For app.py:\n```python\nnew = 1\n```"),
        _plan(3, "For lib.py:\n```python\nb = 2\n```"),
        # No file named: recent_path, the write of seq 1.
        _plan(4, "No file here:\n```python\nz = 1\n```"),
        _plan(5, "None here either:\n```python\nw = 1\n```"),
    ]
    assert _drafts(calls) == {
        "t:d0": ("app.py", "superseded"),
        # Written stays written, though three later drafts name lib.py.
        "t:d1": ("lib.py", "written"),
        "t:d2": ("app.py", "pending"),
        "t:d3": ("lib.py", "superseded"),
        "t:d4": ("lib.py", "superseded"),
        "t:d5": ("lib.py", "pending"),
    }


def test_a_draft_comes_from_the_content_but_not_from_its_note_block() -> None:
    call = RawCall(
        "t",
        0,
        content=(
            "For app.py:\n```python\nkept = 1\n```\n"
            "<note>\nplan: x\n```python\ndropped = 1\n```\n</note>"
        ),
    )
    [_note, draft] = docs_from_calls([call])
    assert (draft.doc_id, draft.text, draft.title) == (
        "t:d0",
        "kept = 1",
        "app.py, 1 lines: kept = 1",
    )


def test_a_draft_without_a_target_is_titled_no_target() -> None:
    # "Here it is:" has no marker word, so the reply gives no note.
    [draft, _chunk] = docs_from_calls([_plan(0, DRAFT)])
    assert (draft.title, draft.meta["target"]) == ("no target, 2 lines: x = 1", "")


def test_two_calls_of_a_task_with_one_seq_are_refused() -> None:
    with pytest.raises(CorpusError, match="task 't' has two calls with seq 0"):
        docs_from_calls([RawCall("t", 0), RawCall("u", 0), RawCall("t", 0)])


def test_tasks_go_by_first_appearance_and_calls_by_seq() -> None:
    calls = [
        RawCall("b", 2, content="<note>plan: second of b</note>"),
        RawCall("a", 0, content="<note>plan: only one of a</note>"),
        RawCall("b", 1, content="<note>plan: first of b</note>"),
    ]
    assert [doc.doc_id for doc in docs_from_calls(calls)] == ["b:n1", "b:n2", "a:n0"]


# ── A run's history.db ──────────────────────────────────────────────────────


def _raw_calls() -> list[RawCall]:
    return load_raw_calls(_lines("raw_calls.jsonl"))


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _execute(database: Path, statement: str, *parameters: object) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


def test_the_history_exports_the_docs_of_the_same_calls(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls())
    before = _snapshot(tmp_path)
    assert list(before) == ["history.db"]

    calls, skipped = load_history_calls(database)
    # t-en's first timestamp is the earlier one; calls by seq.
    expected = sorted(_raw_calls(), key=lambda call: (call.task_id, call.seq))
    assert (calls, skipped) == (expected, 0)
    assert [doc_to_dict(doc) for doc in docs_from_calls(calls)] == RAW_CALL_DOCS
    # Read through immutable=1: no -wal or -shm file, not a byte changed.
    assert _snapshot(tmp_path) == before


def test_history_tasks_go_by_their_first_timestamp(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls())
    _execute(
        database,
        "UPDATE llm_calls SET timestamp = 1.0 WHERE task_id = 't-zh' AND seq = 3",
    )
    calls, _ = load_history_calls(database)
    assert [(call.task_id, call.seq) for call in calls] == [
        ("t-zh", 0),
        ("t-zh", 1),
        ("t-zh", 2),
        ("t-zh", 3),
        ("t-en", 0),
        ("t-en", 1),
        ("t-en", 2),
        ("t-en", 3),
        ("t-en", 4),
    ]


def test_a_row_without_step_ids_keeps_its_calls_without_results(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls())
    _execute(
        database,
        "UPDATE llm_calls SET step_ids_json = NULL, step = NULL, attempt = NULL "
        "WHERE task_id = 't-en' AND seq = 0",
    )
    calls, skipped = load_history_calls(database)
    [call] = [call for call in calls if (call.task_id, call.seq) == ("t-en", 0)]
    assert call.tool_calls == (ToolCall("file_read", {"path": "list_ops.py"}),)
    assert (call.tool_results, call.step, call.attempt, skipped) == ((), 0, 1, 0)
    # Its note then has no observed line and no outcome.
    note = docs_from_calls([call])[0]
    assert note.text == (
        "plan: Let me read list_ops.py to see which functions are stubs. "
        "→ read list_ops.py"
    )


THREE_READS = RawCall(
    "t",
    0,
    content="Reading three files.",
    tool_calls=(
        ToolCall("file_read", {"path": "a.py"}, id="c1"),
        ToolCall("file_read", {"path": "b.py"}, id="c2"),
        ToolCall("file_read", {"path": "c.py"}, id="c3"),
    ),
    tool_results=(
        ToolResult("file_read", output="a"),
        ToolResult("file_read", output="b"),
        ToolResult("file_read", output="c"),
    ),
    outcome="tools",
    timestamp=5.0,
)


def _set_step_ids(database: Path, step_ids: list[str]) -> None:
    _execute(
        database,
        "UPDATE llm_calls SET step_ids_json = ? WHERE task_id = 't' AND seq = 0",
        json.dumps(step_ids),
    )


def test_linked_rows_pair_with_calls_by_id(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    ids = write_history_sync(database, [THREE_READS])[("t", 0)]
    # Listed out of order: each call still takes the row with its id.
    _set_step_ids(database, [ids[2], ids[0], ids[1]])
    [call], skipped = load_history_calls(database)
    assert ([result.output for result in call.tool_results], skipped) == (
        ["a", "b", "c"],
        0,
    )


def test_a_dangling_step_id_cuts_the_results_there(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    ids = write_history_sync(database, [THREE_READS])[("t", 0)]
    _set_step_ids(database, [ids[0], "no-such-step", ids[2]])
    [call], skipped = load_history_calls(database)
    # b.py has no row, so c.py's result, after it, is left out too.
    assert len(call.tool_calls) == 3
    assert ([result.output for result in call.tool_results], skipped) == (["a"], 1)


def test_a_linked_row_of_another_call_is_skipped(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    other = RawCall(
        "t",
        1,
        tool_calls=(ToolCall("file_read", {"path": "d.py"}, id="c9"),),
        tool_results=(ToolResult("file_read", output="d"),),
        outcome="tools",
        timestamp=6.0,
    )
    ids = write_history_sync(database, [THREE_READS, other])
    _set_step_ids(database, [*ids[("t", 0)][:2], ids[("t", 1)][0]])
    calls, skipped = load_history_calls(database)
    assert [result.output for result in calls[0].tool_results] == ["a", "b"]
    assert skipped == 1


def test_a_row_only_in_the_wal_file_is_read(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls())
    writer = sqlite3.connect(database)
    try:
        writer.execute(
            "INSERT INTO llm_calls (task_id, seq, kind, step, attempt, outcome, "
            "content, thinking, timestamp) VALUES ('t-new', 0, 'plan', 1, 1, "
            "'answer', 'All done.', 'Every test passes now.', 1761200100.0)"
        )
        writer.commit()
        # The writer stays open: the row is in the -wal file only, so a read
        # that ignores that file (immutable) misses it.
        assert Path(f"{database}-wal").exists()
        uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
        reader = sqlite3.connect(uri, uri=True)
        try:
            missed = reader.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE task_id = 't-new'"
            ).fetchone()
        finally:
            reader.close()
        assert missed == (0,)
        calls, _ = load_history_calls(database)
        assert calls[-1] == RawCall(
            "t-new",
            0,
            step=1,
            content="All done.",
            thinking="Every test passes now.",
            outcome="answer",
            timestamp=1761200100.0,
        )
        assert len(calls) == len(_raw_calls()) + 1
    finally:
        writer.close()


def test_a_history_without_llm_calls_is_refused(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls(), llm_calls=False)
    with pytest.raises(NoCallsTableError):
        load_history_calls(database)


def test_a_missing_column_is_named(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls(), llm_calls=False)
    _execute(
        database,
        "CREATE TABLE llm_calls (task_id TEXT, seq INTEGER, kind TEXT, outcome TEXT, "
        "content TEXT, tool_calls_json TEXT, timestamp REAL)",
    )
    with pytest.raises(CorpusError) as caught:
        load_history_calls(database)
    assert str(caught.value) == (
        "llm_calls has no column step, attempt, thinking, step_ids_json"
    )


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "UPDATE llm_calls SET kind = 'chat' WHERE task_id = 't-en' AND seq = 1",
            "llm_calls row t-en/1: kind must be one of plan, reflect, retrieve, "
            "estimate, got 'chat'",
        ),
        (
            "UPDATE llm_calls SET tool_calls_json = '[{' WHERE task_id = 't-zh' "
            "AND seq = 2",
            "llm_calls row t-zh/2: tool_calls_json is not JSON",
        ),
        (
            "UPDATE llm_calls SET step_ids_json = '{\"a\": 1}' WHERE task_id = "
            "'t-zh' AND seq = 2",
            "llm_calls row t-zh/2: step_ids_json must be a JSON list of step ids",
        ),
        (
            "UPDATE steps SET tool_result_json = 'oops' WHERE task_id = 't-zh'",
            "llm_calls row t-zh/0: steps row {step_id}: tool_result_json is not JSON",
        ),
    ],
)
def test_a_row_that_breaks_the_rules_is_named(
    tmp_path: Path, statement: str, message: str
) -> None:
    database = tmp_path / "history.db"
    ids = write_history_sync(database, _raw_calls())
    _execute(database, statement)
    with pytest.raises(CorpusError) as caught:
        load_history_calls(database)
    assert str(caught.value) == message.format(step_id=ids[("t-zh", 0)][0])


def test_a_file_that_is_no_database_is_refused(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    database.write_bytes(b"SQLite format 3\x00" + b"\x01" * 200)
    with pytest.raises(CorpusError, match="cannot read the database"):
        load_history_calls(database)
