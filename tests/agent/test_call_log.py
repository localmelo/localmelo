"""The per-run recorder of model calls, against a fake history (R1)."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from localmelo.melo.agent._call_log import CallHistory, CallLog
from localmelo.melo.memory.history import History
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import LLMCallRecord, Message, ToolCall

LOGGER = "localmelo.melo.agent._call_log"
KEY = "sk-" + "a" * 24


@dataclass
class Counters:
    steps: int = 0
    attempts: int = 0


class FakeHistory:
    """Keeps rows and links; fails or blocks the writes it is told to."""

    def __init__(self) -> None:
        self.rows: list[LLMCallRecord] = []
        self.links: list[tuple[str, int, list[str]]] = []
        self.fail_records: set[int] = set()
        self.fail_links = False
        self.record_error: BaseException | None = None

    async def add_llm_call(self, record: LLMCallRecord) -> None:
        if self.record_error is not None:
            raise self.record_error
        if record.seq in self.fail_records:
            raise sqlite3.OperationalError("database is locked")
        self.rows.append(record)

    async def link_llm_call_steps(
        self, task_id: str, seq: int, step_ids: Sequence[str]
    ) -> None:
        if self.fail_links:
            raise sqlite3.OperationalError("database is locked")
        self.links.append((task_id, seq, list(step_ids)))


def _reply(
    content: str = "", *, thinking: str = "", calls: int = 0, **kwargs: Any
) -> Message:
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        tool_calls=[
            ToolCall(tool_name="shell_exec", arguments={"command": "ls"})
            for _ in range(calls)
        ],
        **kwargs,
    )


async def _record(log: CallLog, kind: str, response: Message, **kwargs: Any) -> None:
    fields: dict[str, Any] = {
        "reasoning_key": "reasoning_content",
        "usage": None,
        "duration_s": 0.5,
        "timestamp": 100.0,
    }
    fields.update(kwargs)
    await log.record(kind, response, **fields)


def test_only_a_history_with_both_writers_is_a_call_history() -> None:
    assert isinstance(SqliteHistory(":memory:"), CallHistory)
    assert isinstance(FakeHistory(), CallHistory)
    assert not isinstance(History(), CallHistory)


async def test_each_response_is_one_row() -> None:
    history, run = FakeHistory(), Counters(steps=1, attempts=1)
    log = CallLog(history, "t1", run)
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 8}
    await _record(
        log, "plan", _reply("Listing.", thinking="I list it.", calls=2), usage=usage
    )
    run.steps = 2
    await _record(log, "plan", _reply("  "))
    run.steps = 3
    await _record(log, "plan", _reply("Done.", thinking="It is done."))
    run.attempts = 2
    await _record(log, "reflect", _reply('{"recommended_action": "stop"}'))
    await _record(log, "estimate", _reply("3", thinking="About three."))

    rows = history.rows
    assert [r.seq for r in rows] == [0, 1, 2, 3, 4]
    assert [r.kind for r in rows] == ["plan", "plan", "plan", "reflect", "estimate"]
    assert [r.outcome for r in rows] == ["tools", "empty", "answer", "aux", "aux"]
    assert [(r.step, r.attempt) for r in rows] == [
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 2),
    ]
    # The key only with thinking.
    assert [r.reasoning_key for r in rows] == [
        "reasoning_content",
        "",
        "reasoning_content",
        "",
        "reasoning_content",
    ]
    assert rows[0].usage == usage
    assert rows[0].usage is not usage
    assert rows[1].usage is None
    assert len(rows[0].tool_calls) == 2
    assert isinstance(rows[0].tool_calls, tuple)
    assert (rows[0].duration_s, rows[0].timestamp, rows[0].task_id) == (
        0.5,
        100.0,
        "t1",
    )


async def test_content_and_thinking_are_redacted() -> None:
    history = FakeHistory()
    log = CallLog(history, "t1", Counters(1, 1))
    response = _reply(
        "Your verification code is 482913.", thinking=f"The key is {KEY}."
    )
    await _record(log, "plan", response)
    row = history.rows[0]
    assert row.content == "Your verification code is [redacted]."
    assert row.thinking == "The key is [redacted]."
    assert row.redactions == 2
    # The response itself keeps the text.
    assert response.thinking == f"The key is {KEY}."


async def test_failures_are_logged_once_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    history = FakeHistory()
    history.fail_records = {0, 2}
    log = CallLog(history, "t1", Counters(1, 1))
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    await _record(log, "plan", _reply("a", calls=1))
    await _record(log, "plan", _reply("b", calls=1))
    await _record(log, "plan", _reply("c", calls=1))
    history.fail_links = True
    await log.link_steps(["s1"])

    # The seq advances for every response, also when its write fails.
    assert [r.seq for r in history.rows] == [1]
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (
            logging.WARNING,
            "Could not record model call 0 of task t1 in history: "
            "OperationalError: database is locked",
        ),
        (
            logging.DEBUG,
            "Could not record model call 2 of task t1 in history: "
            "OperationalError: database is locked",
        ),
        (
            logging.DEBUG,
            "Could not link model call 2 of task t1 to its steps in history: "
            "OperationalError: database is locked",
        ),
    ]


async def test_a_failed_link_alone_is_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    history = FakeHistory()
    history.fail_links = True
    log = CallLog(history, "t1", Counters(1, 1))
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    await _record(log, "plan", _reply(calls=1))
    await log.link_steps(["s1"])
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (
            logging.WARNING,
            "Could not link model call 0 of task t1 to its steps in history: "
            "OperationalError: database is locked",
        )
    ]


async def test_links_go_to_the_latest_planning_row_only() -> None:
    history = FakeHistory()
    log = CallLog(history, "t1", Counters(1, 1))
    # Before any planning row, and for an empty list, nothing happens.
    await log.link_steps(["s0"])
    await _record(log, "plan", _reply(calls=1))
    await log.link_steps([])
    await log.link_steps(["s1"])
    await _record(log, "reflect", _reply("{}"))
    await log.link_steps(["s1", "s2"])
    # A planning row whose write failed is still the latest: its ids are
    # never moved onto the earlier row.
    history.fail_records = {2}
    await _record(log, "plan", _reply(calls=1))
    await log.link_steps(["s3"])
    assert history.links == [
        ("t1", 0, ["s1"]),
        ("t1", 0, ["s1", "s2"]),
        ("t1", 2, ["s3"]),
    ]


async def test_a_cancellation_propagates() -> None:
    history = FakeHistory()
    history.record_error = asyncio.CancelledError()
    log = CallLog(history, "t1", Counters(1, 1))
    with pytest.raises(asyncio.CancelledError):
        await _record(log, "plan", _reply("a"))
    history.record_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        await _record(log, "plan", _reply("a"))
