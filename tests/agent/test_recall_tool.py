"""R5's recall tool: registration, offering, records and the steps row."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from localmelo.melo.agent.agent import RECALL_RUN
from localmelo.melo.schema import Message, ToolCall

from .test_recall_loop import (
    FakeLLM,
    _agent,
    _answer,
    _call,
    _entry,
    _options,
)


def _step_calls(path: Path) -> list[dict[str, object]]:
    connection = sqlite3.connect(path)
    try:
        return [
            json.loads(call)
            for (call,) in connection.execute(
                "SELECT tool_call_json FROM steps ORDER BY seq"
            )
            if call is not None
        ]
    finally:
        connection.close()


def _rows(path: Path) -> list[tuple[str, str | None]]:
    connection = sqlite3.connect(path)
    try:
        return [
            (kind, recall)
            for kind, recall in connection.execute(
                "SELECT kind, recall_json FROM llm_calls ORDER BY seq"
            )
        ]
    finally:
        connection.close()


async def test_a_tool_recall_records_on_the_next_planning_row(
    tmp_path: Path,
) -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    llm = FakeLLM(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="recall",
                    arguments={"query": "the answer 42 tests"},
                    id="r1",
                )
            ],
        ),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            max_steps=2,
            memory_dir=str(tmp_path / "memory"),
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    rows = _rows(tmp_path / "memory" / "history.db")
    carrying = [
        json.loads(recall)
        for kind, recall in rows
        if kind == "plan" and recall is not None
    ]
    assert carrying
    assert carrying[0][0]["trigger"] == "tool"
    assert carrying[0][0]["hits"][0]["alias"] == "n1"


async def test_the_tool_adds_a_steps_row_only_when_it_is_called(
    tmp_path: Path,
) -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    unused = FakeLLM(_answer())
    agent = _agent(
        unused,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            max_steps=1,
            memory_dir=str(tmp_path / "unused"),
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.n_tool == 0
    unused_calls = _step_calls(tmp_path / "unused" / "history.db")
    assert all(call["tool_name"] != "recall" for call in unused_calls)

    used = FakeLLM(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="recall",
                    arguments={"query": "the answer 42 tests"},
                    id="r1",
                )
            ],
        ),
        _answer(),
    )
    agent = _agent(
        used,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            max_steps=2,
            memory_dir=str(tmp_path / "used"),
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.n_tool == 1
    assert result.steps == 2
    used_calls = _step_calls(tmp_path / "used" / "history.db")
    assert sum(call["tool_name"] == "recall" for call in used_calls) == 1


async def test_the_recall_tool_reaches_a_thought_only_after_its_turn_left() -> None:
    from localmelo.melo.memory.short import WorkingMemory

    thought = "echo parser build failure"
    llm = FakeLLM(
        _call("plan: echo parser build failure", thinking=thought),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="recall",
                    arguments={"query": "echo parser build failure"},
                    id="r1",
                )
            ],
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="recall",
                    arguments={"query": "echo parser build failure"},
                    id="r2",
                )
            ],
        ),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            recall_kinds=frozenset({"note", "draft", "thought"}),
            max_steps=4,
        ),
    )
    # The query and the first turn fit; the second turn evicts the first.
    agent.hippo.working = WorkingMemory(max_len=3)
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert [e.alias for e in agent.notes() if e.kind == "thought"] == ["t1"]
    # The first recall call still sees turn 1, so it refuses t1; the second
    # runs after eviction and returns it.
    assert "[t1]" not in "\n".join(m.content for m in llm.requests[2])
    assert "[t1]" in "\n".join(m.content for m in llm.requests[3])


async def test_the_tools_k_is_clamped_to_ten() -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    agent = _agent(
        FakeLLM(),
        _options(reasoning="compress", retriever="bm25", recall_at=frozenset({"tool"})),
    )
    assert agent.load_notes([note]) == 1
    from localmelo.melo.agent.agent import _RunState

    run = _RunState(task_id="t")
    token = RECALL_RUN.set(run)
    try:
        block = await agent._recall_tool("the answer 42 tests", 99)
    finally:
        RECALL_RUN.reset(token)
        await agent.close()
    assert block
    assert run.pending_recall_records[0].k == 10


async def test_file_read_and_recall_are_offered_together() -> None:
    llm = FakeLLM(_answer())
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            builtins=frozenset({"file_read", "recall"}),
            max_steps=1,
        ),
    )
    try:
        await agent.run_task("q")
    finally:
        await agent.close()
    offered = set(llm.offers[0])
    assert {"file_read", "recall"} <= offered


def test_a_recall_builtin_without_the_tool_trigger_is_refused() -> None:
    with pytest.raises(ValueError, match="recall needs --recall-at tool"):
        _agent(
            FakeLLM(),
            _options(
                reasoning="compress",
                retriever="bm25",
                builtins=frozenset({"recall"}),
            ),
        )
