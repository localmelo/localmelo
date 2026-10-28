"""Secrets are redacted where memory stores them, not where the model reads them.

History's ``tasks.query``, ``steps.thought``, ``tool_call_json`` and
``tool_result_json``, long-term memory's stored text and the reflection
texts promoted into it carry no secret that :func:`redact_secrets`
recognizes; the working memory and the requests the model receives keep
the original text.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.redaction import REDACTED
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.executor import ToolOutput
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import (
    Message,
    ReflectionEntry,
    StepRecord,
    TaskRecord,
    ToolCall,
    ToolDef,
    ToolResult,
)

# Examples from tests/checker/test_redaction.py's table: a one-time code
# next to its keyword, and an "sk-" key.
OTP = "Your verification code is 482913."
CODE = "482913"
KEY = "sk-" + "aB3" * 10


class FakeEmbedding(BaseEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


async def _echo(text: str = "", key: str = "") -> ToolOutput:
    return ToolOutput(text=f"{text} {key}")


def _rows(db: Path, query: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


class TestStoreStep:
    async def test_the_in_memory_history_gets_the_redacted_step(self) -> None:
        hippo = Hippo()
        task = TaskRecord(query=f"forward this; {OTP}", result=KEY)
        await hippo.save_task(task)
        step = StepRecord(
            thought=f"I will forward it; key {KEY}",
            tool_call=ToolCall("echo", {"text": OTP, "key": KEY}),
            tool_result=ToolResult("echo", output=f"{OTP} {KEY}"),
        )

        summary = await hippo.store_step(task.task_id, step)

        stored = await hippo.get_task(task.task_id)
        assert stored is not None
        assert CODE not in stored.query
        assert KEY not in stored.result
        stored_step = stored.steps[0]
        assert stored_step.tool_call is not None
        assert stored_step.tool_result is not None
        assert CODE not in stored_step.thought
        assert KEY not in json.dumps(stored_step.tool_call.arguments)
        assert CODE not in stored_step.tool_result.output
        assert KEY not in stored_step.tool_result.output
        assert REDACTED in stored_step.thought
        # The summary that goes to long-term memory is redacted too.
        assert CODE not in summary
        assert KEY not in summary
        # The caller's own task and step are untouched: they are what the
        # model and the run still read.
        assert CODE in task.query
        assert task.result == KEY
        assert KEY in step.thought
        assert step.tool_result is not None
        assert OTP in step.tool_result.output

    async def test_the_sqlite_rows_are_redacted(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        history = SqliteHistory(db)
        hippo = Hippo(history=history)
        try:
            task = TaskRecord(query=OTP, result=f"the key is {KEY}")
            await hippo.save_task(task)
            await hippo.store_step(
                task.task_id,
                StepRecord(
                    thought=f"key {KEY}",
                    tool_call=ToolCall("echo", {"text": OTP, "key": KEY}),
                    tool_result=ToolResult("echo", output=OTP, error=f"key {KEY}"),
                ),
            )
        finally:
            await history.aclose()

        [(query, result)] = _rows(db, "SELECT query, result FROM tasks")
        [(thought, tc_json, tr_json)] = _rows(
            db, "SELECT thought, tool_call_json, tool_result_json FROM steps"
        )
        for text in (query, result, thought, tc_json, tr_json):
            assert CODE not in text
            assert KEY not in text
            assert REDACTED in text


class TestLongTermWrites:
    async def test_memorize_stores_the_redacted_text(self, tmp_path: Path) -> None:
        db = tmp_path / "long.db"
        long = SqliteLongTerm(db)
        hippo = Hippo(embedding=FakeEmbedding(), long=long)
        try:
            await hippo.memorize(f"the code was {OTP}", to_working=False)
        finally:
            await long.aclose()

        [(text,)] = _rows(db, "SELECT text FROM long_term")
        assert CODE not in text
        assert REDACTED in text

    async def test_working_memory_keeps_the_original(self) -> None:
        hippo = Hippo(embedding=FakeEmbedding())
        await hippo.memorize(f"the code was {OTP}")

        assert hippo.working.get_window()[-1].content == f"the code was {OTP}"
        [(text, _score, _meta)] = await hippo.long.search([1.0, 0.0, 0.0, 0.0])
        assert CODE not in text

    async def test_promoted_reflections_store_the_redacted_text(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "long.db"
        long = SqliteLongTerm(db)
        hippo = Hippo(embedding=FakeEmbedding(), long=long)
        try:
            hippo.working.add_reflection(
                ReflectionEntry(attempt_id=0, summary=f"the OTP was {CODE}")
            )
            await hippo.promote_reflections("task-1")
        finally:
            await long.aclose()

        [(text,)] = _rows(db, "SELECT text FROM long_term")
        assert CODE not in text
        assert REDACTED in text


class _LeakyCaller(BaseLLMProvider):
    """Calls a tool whose arguments and result carry secrets, then answers."""

    def __init__(self) -> None:
        self.replies = [
            Message(
                role="assistant",
                tool_call=ToolCall("echo", {"text": OTP, "key": KEY}),
            ),
            Message(role="assistant", content="done"),
        ]
        self.calls: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append(list(messages))
        return self.replies.pop(0)

    async def close(self) -> None:
        pass


async def test_a_run_redacts_the_stored_copy_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)
    db = tmp_path / "history.db"
    llm = _LeakyCaller()
    agent = Agent(llm=llm, options=AgentOptions(attempt_policy="none", max_steps=2))
    history = SqliteHistory(db)
    agent.hippo.history = history
    agent.hippo.register_tool(
        ToolDef(name="echo", description="echo text", parameters={"type": "object"})
    )
    agent.executor.register("echo", _echo)

    result = await agent.run_task(f"echo it back: {OTP}")
    await agent.close()

    assert result.status == "completed"
    # The model's next request carried the originals, tool result included.
    sent = llm.calls[1]
    tool_texts = [m.content for m in sent if m.role == "tool"]
    assert tool_texts == [f"{OTP} {KEY}"]
    # What the run stored is redacted instead.
    [(query,)] = _rows(db, "SELECT query FROM tasks")
    [(thought, tc_json, tr_json)] = _rows(
        db, "SELECT thought, tool_call_json, tool_result_json FROM steps"
    )
    for text in (query, thought, tc_json, tr_json):
        assert CODE not in text
        assert KEY not in text
