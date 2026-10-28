"""R5's record: history's recall_json, the event and the JSON line."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.notes import RECALL_START
from localmelo.melo.schema import Message, ToolDef

from .test_recall_loop import (
    FakeLLM,
    _agent,
    _answer,
    _call,
    _entry,
    _options,
)


def _recall_rows(path: Path) -> list[tuple[str, int, str | None]]:
    connection = sqlite3.connect(path)
    try:
        return [
            (kind, seq, recall)
            for kind, seq, recall in connection.execute(
                "SELECT kind, seq, recall_json FROM llm_calls ORDER BY seq"
            )
        ]
    finally:
        connection.close()


async def test_a_block_is_recorded_once_on_the_request_that_first_carries_it(
    tmp_path: Path,
) -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    llm = FakeLLM(
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "a"},
        ),
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "b"},
        ),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            max_steps=3,
            memory_dir=str(tmp_path / "memory"),
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    rows = _recall_rows(tmp_path / "memory" / "history.db")
    with_recall = [row for row in rows if row[2] is not None]
    # Each new step block is recorded on the request that first carries it,
    # and only there; a later request carries it as sticky window text.
    assert [seq for _kind, seq, _recall in with_recall] == [1]
    for kind, _seq, recall_json in with_recall:
        assert kind == "plan"
        records = json.loads(recall_json or "[]")
        assert len(records) == 1
        assert records[0]["trigger"] == "step"
        assert records[0]["hits"]
        assert "text" in records[0]["hits"][0]
        # The latency is recorded, but the prompt header never shows it.
        assert isinstance(records[0]["ms"], int)


async def test_the_event_and_the_compact_log_match_the_record(
    tmp_path: Path,
) -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    llm = FakeLLM(
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "a"},
        ),
        _answer(),
    )
    events: list[Any] = []
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            max_steps=2,
            memory_dir=str(tmp_path / "memory"),
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q", on_event=events.append)
    finally:
        await agent.close()
    recalls = [event for event in events if event.type == "recall"]
    assert len(recalls) == 1
    data = recalls[0].data
    assert data["trigger"] == "step"
    assert data["retriever"] == "bm25"
    assert data["hits"][0]["alias"] == "n1"
    assert "text" not in data["hits"][0]
    assert isinstance(data["ms"], int)
    assert all(
        "(retriever bm25)" in message.content and " ms)" not in message.content
        for request in llm.requests
        for message in request
        if RECALL_START in message.content
    )
    assert result.memory_stats is not None
    compact = result.memory_stats["recall_log"][0]
    assert compact["trigger"] == data["trigger"]
    assert compact["hits"] == data["hits"]
    assert compact["ms"] == data["ms"]
    assert result.memory_stats["recall_ms"] == data["ms"]
    assert "text" not in compact["hits"][0]
    assert RECALL_START == "[localmelo recall:"


class RetrievalLLM(BaseLLMProvider):
    """Planning replies from a script; one JSON answer to a retrieval call."""

    def __init__(self, *replies: Message) -> None:
        self.replies = list(replies)
        self.retrievals: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if tools is None:
            self.retrievals.append(messages)
            return Message(role="assistant", content='{"ids": ["n1"]}')
        return self.replies.pop(0)

    async def close(self) -> None:
        pass


async def test_a_retrieve_row_has_no_recall_json(tmp_path: Path) -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    llm = RetrievalLLM(
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "a"},
        ),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="llm",
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
    assert llm.retrievals
    rows = _recall_rows(tmp_path / "memory" / "history.db")
    retrieve = [row for row in rows if row[0] == "retrieve"]
    assert len(retrieve) == 1
    assert retrieve[0][2] is None
    assert result.memory_stats is not None
    assert result.memory_stats["aux_retrieve_calls"] == 1


async def test_legacy_rows_keep_recall_json_null(tmp_path: Path) -> None:
    llm = FakeLLM(_answer())
    agent = _agent(
        llm,
        _options(max_steps=1, memory_dir=str(tmp_path / "memory")),
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    rows = _recall_rows(tmp_path / "memory" / "history.db")
    assert rows
    assert all(recall is None for _kind, _seq, recall in rows)


async def test_a_block_built_but_never_sent_is_not_counted(
    tmp_path: Path,
) -> None:
    note = _entry("n1", "plan: the answer is 42 and tests pass")
    llm = FakeLLM(
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "a"},
        )
    )
    events: list[Any] = []
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            max_steps=1,
            memory_dir=str(tmp_path / "memory"),
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q", on_event=events.append)
    finally:
        await agent.close()
    assert result.status == "budget_exhausted"
    assert result.memory_stats is not None
    assert result.memory_stats["recalls"] == 0
    assert result.memory_stats["recall_log"] == []
    assert [event for event in events if event.type == "recall"] == []
    rows = _recall_rows(tmp_path / "memory" / "history.db")
    assert all(recall is None for _kind, _seq, recall in rows)
