"""An Agent with a memory directory and the llm_calls table of history (R1).

Every memory directory is under ``tmp_path``; no test touches
``~/.cache/localmelo``. The environment is set with ``monkeypatch``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent import agent as agent_module
from localmelo.melo.agent._call_log import CallLog
from localmelo.melo.agent._chat import CALL_RECORDER, Chat
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.memory.history import History
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import (
    LLMCallRecord,
    Message,
    RunResult,
    StepRecord,
    TaskRecord,
    ToolCall,
    ToolDef,
)
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.agent.test_cancellation import Recorder, check_events

SQLITE_HELPERS_LOGGER = "localmelo.melo.memory._sqlite"


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request."""

    def __init__(self, replies: list[Message] | None = None) -> None:
        self.replies: deque[Message] = deque(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            return Message(role="assistant", content="(no reply queued)")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


# ── The memory directory's mode ─────────────────────────────────────────────


async def test_the_bench_directory_is_created_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bench sets these and never passes through _make_memory_dir.
    memory_dir = tmp_path / "run" / "mem"
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(memory_dir))
    agent = Agent(llm=FakeLLM())
    await agent.close()
    assert _mode(memory_dir) == 0o700


async def test_an_options_directory_is_created_0700(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    agent = Agent(llm=FakeLLM(), options=AgentOptions(memory_dir=str(memory_dir)))
    await agent.close()
    assert _mode(memory_dir) == 0o700


async def test_the_default_directory_is_tightened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    default = tmp_path / "default"
    default.mkdir()
    default.chmod(0o755)
    monkeypatch.setattr(agent_module, "DEFAULT_MEMORY_DIR", str(default))
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    caplog.set_level(logging.DEBUG, logger=SQLITE_HELPERS_LOGGER)
    agent = Agent(llm=FakeLLM())
    await agent.close()
    assert _mode(default) == 0o700
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, f"Restricted {default} to mode 0700")
    ]


@pytest.mark.parametrize("named_by", ["environment", "options"])
async def test_a_chosen_directory_keeps_its_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named_by: str
) -> None:
    # Also when it names the default directory.
    default = tmp_path / "default"
    default.mkdir()
    default.chmod(0o755)
    monkeypatch.setattr(agent_module, "DEFAULT_MEMORY_DIR", str(default))
    options = None
    if named_by == "environment":
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(default))
    else:
        options = AgentOptions(memory_dir=str(default))
    agent = Agent(llm=FakeLLM(), options=options)
    await agent.close()
    assert _mode(default) == 0o755
    assert os.path.isdir(default)


# ── Retention through the Agent ─────────────────────────────────────────────


async def _write_old_call(path: Path, age_days: float) -> None:
    """One llm_calls row with thinking, *age_days* old, by a separate history."""
    history = SqliteHistory(path)
    try:
        conn = await history._ensure_ready()
        await conn.execute(
            """INSERT INTO llm_calls (task_id, seq, kind, step, attempt, outcome,
               content, thinking, reasoning_key, timestamp)
               VALUES ('old', 0, 'plan', 1, 1, 'answer', 'Paris.', 'Recall it.',
                       'reasoning_content', ?)""",
            (time.time() - age_days * 86400,),
        )
        await conn.commit()
    finally:
        await history.aclose()


def _old_thinking(path: Path) -> str:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            "SELECT thinking FROM llm_calls WHERE task_id = 'old'"
        ).fetchone()
    return str(row[0])


@pytest.mark.parametrize(("value", "kept"), [("7", ""), (None, "Recall it.")])
async def test_the_retention_variable_reaches_the_agents_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None, kept: str
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    await _write_old_call(memory_dir / "history.db", age_days=8)
    if value is not None:
        monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", value)
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(memory_dir))
    agent = Agent(llm=FakeLLM([Message(role="assistant", content="Paris.")]))
    try:
        result = await agent.run_task("What is the capital of France?")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert _old_thinking(memory_dir / "history.db") == kept


async def test_a_bad_retention_value_raises_before_the_directory_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_dir = tmp_path / "memory"
    monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "abc")
    with pytest.raises(ValueError, match="LOCALMELO_THINKING_RETENTION_DAYS"):
        Agent(llm=FakeLLM(), options=AgentOptions(memory_dir=str(memory_dir)))
    assert not memory_dir.exists()
    # Without persistence the variable is not read.
    agent = Agent(llm=FakeLLM())
    await agent.close()


# ── Requests outside a run ──────────────────────────────────────────────────


async def test_estimate_steps_with_a_recorder_set_by_hand(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    chat = Chat(FakeLLM([Message(role="assistant", content="3", thinking="About 3.")]))
    token = CALL_RECORDER.set(CallLog(history, "t1", _Counters()))
    try:
        assert await chat.estimate_steps("List the files.") == 3
    finally:
        CALL_RECORDER.reset(token)
        await history.aclose()
    with contextlib.closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            "SELECT task_id, seq, kind, outcome, content, thinking, step, attempt "
            "FROM llm_calls"
        ).fetchall()
    assert rows == [("t1", 0, "estimate", "aux", "3", "About 3.", 0, 0)]


class _Counters:
    steps = 0
    attempts = 0


# ── Recording runs ──────────────────────────────────────────────────────────

CALL_LOG_LOGGER = "localmelo.melo.agent._call_log"
KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"


class ScriptedLLM(BaseLLMProvider):
    """Answers with each scripted reply in turn, then waits until cancelled.

    A scripted exception is raised instead. ``recorders`` keeps
    CALL_RECORDER as each request saw it, and ``requests`` each request.
    """

    def __init__(self, replies: list[Message | BaseException]) -> None:
        self.replies: deque[Message | BaseException] = deque(replies)
        self.recorders: list[object] = []
        self.requests: list[Any] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.recorders.append(CALL_RECORDER.get())
        self.requests.append(
            (
                None if tools is None else [tool.name for tool in tools],
                [
                    (
                        m.role,
                        m.content,
                        m.tool_call_id,
                        [(c.tool_name, c.arguments, c.id) for c in m.tool_calls],
                    )
                    for m in messages
                ],
            )
        )
        if not self.replies:
            await asyncio.Event().wait()
        reply = self.replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self) -> None:
        pass


class Tools:
    """echo, and gate, which waits at ``at_gate`` until ``go`` is set."""

    def __init__(self) -> None:
        self.at_gate = asyncio.Event()
        self.go = asyncio.Event()

    async def echo(self, text: str = "") -> str:
        return text

    async def gate(self) -> str:
        self.at_gate.set()
        await self.go.wait()
        return "through"


TOOL_DEFS = [
    ToolDef(
        name="echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    ),
    ToolDef(
        name="gate",
        description="Wait until told to go on",
        parameters={"type": "object", "properties": {}},
    ),
]


async def _no_wait(delay: float) -> None:
    pass


def _agent(
    llm: BaseLLMProvider,
    memory_dir: Path | None,
    *,
    tools: Tools | None = None,
    **options: Any,
) -> Agent:
    """An agent with echo and gate offered on every step, and no retry waits."""
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    if memory_dir is not None:
        options["memory_dir"] = str(memory_dir)
    agent = Agent(llm=llm, options=AgentOptions(**options))
    tools = tools or Tools()
    for definition in TOOL_DEFS:
        agent.hippo.register_tool(definition)
        agent.executor.register(definition.name, getattr(tools, definition.name))
    agent.chat.sleep = _no_wait
    return agent


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _calls(
    *calls: tuple[str, dict[str, Any], str],
    content: str = "",
    thinking: str = "",
    usage: dict[str, int] | None = None,
) -> Message:
    """A reply with tool calls, each ``(name, arguments, id)``."""
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        usage=usage,
        tool_calls=[ToolCall(tool_name=n, arguments=a, id=i) for n, a, i in calls],
    )


def _answer(
    text: str, *, thinking: str = "", usage: dict[str, int] | None = None
) -> Message:
    return Message(role="assistant", content=text, thinking=thinking, usage=usage)


def _reflection(action: str, *, usage: dict[str, int] | None = None) -> Message:
    decision = {
        "summary": "Echoed three times.",
        "failed_hypotheses": [],
        "useful_evidence": [],
        "recommended_avoids": [],
        "next_promising_directions": ["Answer now."],
        "progress_made": True,
        "task_still_feasible": True,
        "new_information_gained": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "recommended_action": action,
        "rationale": "The echoes are done.",
        "best_effort_result": "",
    }
    return _answer(json.dumps(decision), usage=usage)


def _busy() -> ProviderError:
    return ProviderError("HTTP 503: busy", status_code=503, retryable=True)


def _call_rows(path: Path, task_id: str) -> list[dict[str, Any]]:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM llm_calls WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def _step_ids(path: Path, task_id: str) -> list[str]:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            "SELECT step_id FROM steps WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()
    return [row[0] for row in rows]


def _joined(path: Path, task_id: str) -> dict[int, list[str]]:
    """Each call's steps rows, through json_each, as the docs join them."""
    with contextlib.closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            """SELECT c.seq, s.step_id
               FROM llm_calls c, json_each(c.step_ids_json) j
               JOIN steps s ON s.step_id = j.value
               WHERE c.task_id = ? ORDER BY c.seq, j.key""",
            (task_id,),
        ).fetchall()
    joined: dict[int, list[str]] = {}
    for seq, step_id in rows:
        joined.setdefault(seq, []).append(step_id)
    return joined


def _comparable(result: RunResult) -> dict[str, Any]:
    data = result.to_dict()
    for name in ("task_id", "duration_s"):
        del data[name]
    data["config"].pop("memory_dir")
    return data


async def test_each_response_is_one_row(tmp_path: Path) -> None:
    replies: list[Message | BaseException] = [
        _calls(
            ("echo", {"text": "a"}, "call_1"),
            ("echo", {"text": "b"}, "call_2"),
            content="Echoing twice.",
            thinking="Two echoes first.",
            usage=_usage(100, 10),
        ),
        _busy(),
        _answer("", usage=_usage(110, 0)),
        _calls(
            ("echo", {"text": "c"}, "call_3"),
            thinking="One more.",
            usage=_usage(120, 5),
        ),
        # The first reflection reply is a tool call written as text (#31).
        _answer(
            '{"tool_name": "echo", "arguments": {"text": "d"}}', usage=_usage(300, 20)
        ),
        _reflection("continue", usage=_usage(310, 30)),
        _answer("Done: a b c.", thinking="All echoed.", usage=_usage(400, 8)),
    ]
    script = [reply for reply in replies if isinstance(reply, Message)]
    agent = _agent(ScriptedLLM(replies), tmp_path, steps_per_attempt=3)
    try:
        result = await agent.run_task("Echo a, b and c.")
    finally:
        await agent.close()
    assert (result.status, result.n_llm, result.llm_errors) == ("completed", 6, 1)

    db = tmp_path / "history.db"
    rows = _call_rows(db, result.task_id)
    assert len(rows) == result.n_llm
    assert [r["seq"] for r in rows] == list(range(6))
    assert [(r["kind"], r["outcome"], r["step"], r["attempt"]) for r in rows] == [
        ("plan", "tools", 1, 1),
        ("plan", "empty", 2, 1),
        ("plan", "tools", 3, 1),
        ("reflect", "aux", 3, 1),
        ("reflect", "aux", 3, 1),
        ("plan", "answer", 4, 2),
    ]
    steps = _step_ids(db, result.task_id)
    assert len(steps) == 3
    assert _joined(db, result.task_id) == {0: steps[:2], 2: steps[2:]}
    assert [r["step_ids_json"] for r in rows] == [
        json.dumps(steps[:2]),
        None,
        json.dumps(steps[2:]),
        None,
        None,
        None,
    ]
    assert [r["content"] for r in rows] == [m.content for m in script]
    assert [r["thinking"] for r in rows] == [m.thinking for m in script]
    assert [json.loads(r["usage_json"]) for r in rows] == [m.usage for m in script]
    assert [json.loads(r["tool_calls_json"] or "[]") for r in rows][0] == [
        {"tool_name": "echo", "arguments": {"text": "a"}, "id": "call_1"},
        {"tool_name": "echo", "arguments": {"text": "b"}, "id": "call_2"},
    ]
    # A FakeLLM reply with thinking has no reasoning key.
    assert {r["reasoning_key"] for r in rows} == {""}
    assert all(r["duration_s"] >= 0 and r["timestamp"] > 0 for r in rows)


async def test_rows_are_redacted_while_the_run_keeps_the_text(tmp_path: Path) -> None:
    code = "Your verification code is 482913."
    agent = _agent(
        ScriptedLLM(
            [
                _calls(
                    ("echo", {"text": "a"}, "call_1"),
                    content=code,
                    thinking=f"The key is {KEY}.",
                ),
                _answer(code),
            ]
        ),
        tmp_path,
    )
    try:
        result = await agent.run_task("Echo a.")
        window = agent.hippo.working.get_window()
        steps = await agent.hippo.history.get_steps(result.task_id)
    finally:
        await agent.close()
    rows = _call_rows(tmp_path / "history.db", result.task_id)
    assert (rows[0]["content"], rows[0]["thinking"], rows[0]["redactions"]) == (
        "Your verification code is [redacted].",
        "The key is [redacted].",
        2,
    )
    assert (rows[1]["content"], rows[1]["redactions"]) == (
        "Your verification code is [redacted].",
        1,
    )
    [call] = [m for m in window if m.tool_calls]
    assert (call.content, call.thinking) == (code, f"The key is {KEY}.")
    # The run keeps the text; the step history keeps is redacted too.
    assert steps[0].thought == "Your verification code is [redacted]."
    assert result.result == code


def _two_step_script() -> list[Message | BaseException]:
    return [
        _calls(("echo", {"text": "a"}, "call_1"), content="Echo a."),
        _answer("a"),
    ]


async def test_without_persistence_no_recorder_is_set(tmp_path: Path) -> None:
    plain = ScriptedLLM(_two_step_script())
    agent = _agent(plain, None)
    try:
        await agent.run_task("Echo a.")
    finally:
        await agent.close()
    persistent = ScriptedLLM(_two_step_script())
    agent = _agent(persistent, tmp_path)
    try:
        await agent.run_task("Echo a.")
    finally:
        await agent.close()
    assert plain.recorders == [None, None]
    assert all(isinstance(r, CallLog) for r in persistent.recorders)
    assert plain.requests == persistent.requests
    # The recorder is reset once the run ends.
    assert CALL_RECORDER.get() is None


async def test_a_history_set_after_the_agent_records_too(tmp_path: Path) -> None:
    # As the web workspace sets each chat's history.
    agent = _agent(ScriptedLLM(_two_step_script()), None)
    await agent.hippo.aclose()
    agent.hippo.history = SqliteHistory(tmp_path / "history.db")
    try:
        result = await agent.run_task("Echo a.")
    finally:
        await agent.close()
    rows = _call_rows(tmp_path / "history.db", result.task_id)
    assert [(r["kind"], r["outcome"]) for r in rows] == [
        ("plan", "tools"),
        ("plan", "answer"),
    ]


async def _run(memory_dir: Path, **patches: Any) -> tuple[RunResult, SqliteHistory]:
    agent = _agent(
        ScriptedLLM(
            [
                _calls(("echo", {"text": "a"}, "call_1")),
                _calls(("echo", {"text": "b"}, "call_2")),
                _answer("a b"),
            ]
        ),
        memory_dir,
    )
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    for name, value in patches.items():
        setattr(history, name, value)
    try:
        result = await agent.run_task("Echo a, then b.")
    finally:
        await agent.close()
    return result, history


async def _refuse(*args: Any, **kwargs: Any) -> None:
    raise sqlite3.OperationalError("database is locked")


async def test_a_failed_write_changes_nothing_of_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    expected, _ = await _run(tmp_path / "plain")
    caplog.set_level(logging.DEBUG, logger=CALL_LOG_LOGGER)
    result, _ = await _run(tmp_path / "failing", add_llm_call=_refuse)
    assert _comparable(result) == _comparable(expected)
    levels = [r.levelno for r in caplog.records if r.name == CALL_LOG_LOGGER]
    assert levels == [logging.WARNING] + [logging.DEBUG] * 2
    assert caplog.records[0].getMessage() == (
        f"Could not record model call 0 of task {result.task_id} in history: "
        "OperationalError: database is locked"
    )
    assert _call_rows(tmp_path / "failing" / "history.db", result.task_id) == []


async def test_a_failed_link_changes_nothing_of_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    expected, _ = await _run(tmp_path / "plain")
    caplog.set_level(logging.DEBUG, logger=CALL_LOG_LOGGER)
    result, _ = await _run(tmp_path / "failing", link_llm_call_steps=_refuse)
    assert _comparable(result) == _comparable(expected)
    records = [r for r in caplog.records if r.name == CALL_LOG_LOGGER]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.WARNING,
            f"Could not link model call 0 of task {result.task_id} to its steps "
            "in history: OperationalError: database is locked",
        ),
        (
            logging.DEBUG,
            f"Could not link model call 1 of task {result.task_id} to its steps "
            "in history: OperationalError: database is locked",
        ),
    ]


async def test_a_link_after_a_failed_record_logs_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = _agent(ScriptedLLM(_two_step_script()), tmp_path)
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    add_llm_call = history.add_llm_call

    async def refuse_the_first(record: LLMCallRecord) -> None:
        if record.seq == 0:
            raise sqlite3.OperationalError("database is locked")
        await add_llm_call(record)

    history.add_llm_call = refuse_the_first  # type: ignore[method-assign]
    history.link_llm_call_steps = _refuse  # type: ignore[method-assign]
    caplog.set_level(logging.DEBUG, logger=CALL_LOG_LOGGER)
    try:
        result = await agent.run_task("Echo a.")
    finally:
        await agent.close()
    assert result.status == "completed"
    task = result.task_id
    records = [r for r in caplog.records if r.name == CALL_LOG_LOGGER]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.WARNING,
            f"Could not record model call 0 of task {task} in history: "
            "OperationalError: database is locked",
        ),
        (
            logging.DEBUG,
            f"Could not link model call 0 of task {task} to its steps in history: "
            "OperationalError: database is locked",
        ),
    ]


async def test_a_failed_planning_row_keeps_its_steps_off_other_rows(
    tmp_path: Path,
) -> None:
    agent = _agent(
        ScriptedLLM(
            [
                _calls(("echo", {"text": "a"}, "call_1")),
                _calls(("echo", {"text": "b"}, "call_2")),
                _calls(("echo", {"text": "c"}, "call_3")),
                _answer("a b c"),
            ]
        ),
        tmp_path,
    )
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    add_llm_call = history.add_llm_call

    async def refuse_the_second(record: LLMCallRecord) -> None:
        if record.seq == 1:
            raise sqlite3.OperationalError("database is locked")
        await add_llm_call(record)

    history.add_llm_call = refuse_the_second  # type: ignore[method-assign]
    try:
        result = await agent.run_task("Echo a, b and c.")
    finally:
        await agent.close()
    db = tmp_path / "history.db"
    steps = _step_ids(db, result.task_id)
    rows = _call_rows(db, result.task_id)
    assert [r["seq"] for r in rows] == [0, 2, 3]
    assert _joined(db, result.task_id) == {0: [steps[0]], 2: [steps[2]]}


async def test_a_cancellation_while_recording_propagates(tmp_path: Path) -> None:
    agent = _agent(ScriptedLLM([_answer("a", usage=_usage(10, 1))]), tmp_path)
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    blocked = asyncio.Event()

    async def block(record: LLMCallRecord) -> None:
        blocked.set()
        await asyncio.Event().wait()

    history.add_llm_call = block  # type: ignore[method-assign]
    try:
        running = asyncio.create_task(agent.run_task("Say a."))
        await asyncio.wait_for(blocked.wait(), 5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await agent.close()
    # Counted, though no event or row carries the reply.
    assert (agent.chat.stats.n_llm, agent.chat.stats.llm_errors) == (1, 0)
    with contextlib.closing(sqlite3.connect(tmp_path / "history.db")) as db:
        assert db.execute("SELECT status FROM tasks").fetchall() == [("crashed",)]


async def test_a_stop_while_recording_ends_the_run_cancelled(tmp_path: Path) -> None:
    agent = _agent(ScriptedLLM([_answer("a", usage=_usage(10, 1))]), tmp_path)
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    blocked = asyncio.Event()

    async def block(record: LLMCallRecord) -> None:
        blocked.set()
        await asyncio.Event().wait()

    history.add_llm_call = block  # type: ignore[method-assign]
    recorder = Recorder()
    try:
        running = asyncio.create_task(agent.run_task("Say a.", on_event=recorder))
        await asyncio.wait_for(blocked.wait(), 5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await agent.close()
    result = check_events(recorder)
    assert result.status == "cancelled"
    assert result.usage == _usage(10, 1)
    assert recorder.of("assistant.message") == []


async def test_the_bench_queries_read_the_run(tmp_path: Path) -> None:
    agent = _agent(
        ScriptedLLM(
            [
                _calls(
                    ("echo", {"text": "a"}, "call_1"), ("echo", {"text": "b"}, "call_2")
                ),
                _answer("a b"),
            ]
        ),
        tmp_path,
    )
    try:
        result = await agent.run_task("Echo a and b.")
    finally:
        await agent.close()
    uri = f"file:{tmp_path / 'history.db'}?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
        count = db.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        status = db.execute(
            "SELECT status FROM tasks ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    assert (count, status) == (result.n_tool, result.status) == (2, "completed")


# ── The finalizer of a cancelled run links the rows it wrote ────────────────


def _three_calls() -> Message:
    return _calls(
        ("echo", {"text": "a"}, "call_1"),
        ("gate", {}, "call_2"),
        ("echo", {"text": "c"}, "call_3"),
        content="Three calls.",
    )


def _log_calls(
    history: SqliteHistory, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, Any]]:
    """Wrap save_task and link_llm_call_steps to log their calls in order."""
    calls: list[tuple[str, Any]] = []
    save_task, link = history.save_task, history.link_llm_call_steps

    async def logged_save(task: TaskRecord) -> None:
        calls.append(("save_task", task.status))
        await save_task(task)

    async def logged_link(task_id: str, seq: int, step_ids: Sequence[str]) -> None:
        calls.append(("link", (seq, list(step_ids))))
        await link(task_id, seq, step_ids)

    monkeypatch.setattr(history, "save_task", logged_save)
    monkeypatch.setattr(history, "link_llm_call_steps", logged_link)
    return calls


async def test_the_finalizer_links_its_rows_after_the_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, recorder = Tools(), Recorder()
    agent = _agent(ScriptedLLM([_three_calls()]), tmp_path, tools=tools)
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    calls = _log_calls(history, monkeypatch)
    try:
        running = asyncio.create_task(agent.run_task("Go.", on_event=recorder))
        await asyncio.wait_for(tools.at_gate.wait(), 5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await agent.close()
    result = check_events(recorder)
    assert result.status == "cancelled"
    db = tmp_path / "history.db"
    steps = _step_ids(db, result.task_id)
    assert len(steps) == 3
    assert _joined(db, result.task_id) == {0: steps}
    assert calls == [
        ("save_task", "running"),
        ("save_task", "cancelled"),
        ("link", (0, steps)),
    ]


async def test_the_finalizer_links_only_the_rows_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, recorder = Tools(), Recorder()
    agent = _agent(ScriptedLLM([_three_calls()]), tmp_path, tools=tools)
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    calls = _log_calls(history, monkeypatch)
    add_step = history.add_step

    async def refuse_the_third(task_id: str, step: StepRecord) -> None:
        if step.tool_call is not None and step.tool_call.id == "call_3":
            raise sqlite3.OperationalError("database is locked")
        await add_step(task_id, step)

    monkeypatch.setattr(history, "add_step", refuse_the_third)
    try:
        running = asyncio.create_task(agent.run_task("Go.", on_event=recorder))
        await asyncio.wait_for(tools.at_gate.wait(), 5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await agent.close()
    result = check_events(recorder)
    db = tmp_path / "history.db"
    steps = _step_ids(db, result.task_id)
    assert len(steps) == 2
    assert _joined(db, result.task_id) == {0: steps}
    assert calls[-1] == ("link", (0, steps))


async def test_a_stop_during_the_link_links_the_step_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    reply = _calls(
        ("echo", {"text": "a"}, "call_1"),
        ("echo", {"text": "b"}, "call_2"),
        content="Two calls.",
    )
    agent = _agent(ScriptedLLM([reply]), tmp_path)
    history = agent.hippo.history
    assert isinstance(history, SqliteHistory)
    link = history.link_llm_call_steps
    linking = asyncio.Event()
    links: list[list[str]] = []

    async def block_the_first(task_id: str, seq: int, step_ids: Sequence[str]) -> None:
        links.append(list(step_ids))
        if len(links) == 1:
            linking.set()
            await asyncio.Event().wait()
        await link(task_id, seq, step_ids)

    monkeypatch.setattr(history, "link_llm_call_steps", block_the_first)
    try:
        running = asyncio.create_task(agent.run_task("Go.", on_event=recorder))
        await asyncio.wait_for(linking.wait(), 5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await agent.close()
    result = check_events(recorder)
    assert result.status == "cancelled"
    db = tmp_path / "history.db"
    steps = _step_ids(db, result.task_id)
    # No new steps row: both were written before the link.
    assert len(steps) == 2
    assert links == [steps, steps]
    assert _joined(db, result.task_id) == {0: steps}


# ── The real providers: reasoning key and full usage ────────────────────────


def _openai_body(message: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", **message}}], "usage": usage}


async def _provider_rows(
    tmp_path: Path, llm: BaseLLMProvider
) -> tuple[RunResult, list[dict[str, Any]]]:
    agent = Agent(llm=llm, options=AgentOptions(memory_dir=str(tmp_path)))
    try:
        result = await agent.run_task("Say hi.")
    finally:
        await agent.close()
    return result, _call_rows(tmp_path / "history.db", result.task_id)


DETAILED_USAGE = {
    **_usage(120, 40),
    "prompt_tokens_details": {"cached_tokens": 96},
    "completion_tokens_details": {"reasoning_tokens": 28},
}


@pytest.mark.parametrize(
    ("message", "key"),
    [
        ({"content": "hi", "reasoning_content": "Greet."}, "reasoning_content"),
        ({"content": "hi", "reasoning": "Greet."}, "reasoning"),
        ({"content": "hi"}, ""),
    ],
)
async def test_openai_rows_keep_the_key_and_the_full_usage(
    tmp_path: Path, message: dict[str, Any], key: str
) -> None:
    body = _openai_body(message, DETAILED_USAGE)
    llm = OpenAICompatLLM(
        base_url="http://llm.test/v1",
        model="m",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    result, [row] = await _provider_rows(tmp_path, llm)
    assert row["reasoning_key"] == key
    assert json.loads(row["usage_json"]) == {
        **_usage(120, 40),
        "cached_tokens": 96,
        "reasoning_tokens": 28,
    }
    # The RunResult keeps the three keys.
    assert result.usage == _usage(120, 40)


async def test_ollama_rows_keep_the_key_and_three_usage_keys(tmp_path: Path) -> None:
    body = {
        "message": {"role": "assistant", "content": "hi", "thinking": "Greet."},
        "prompt_eval_count": 120,
        "eval_count": 40,
    }
    llm = OllamaNativeChat(
        base_url="http://ollama.test",
        model="m",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    result, [row] = await _provider_rows(tmp_path, llm)
    assert row["reasoning_key"] == "thinking"
    assert row["content"] == "<think>\nGreet.\n</think>\n\nhi"
    assert json.loads(row["usage_json"]) == _usage(120, 40) == result.usage


async def test_a_fake_reply_with_thinking_has_no_key(tmp_path: Path) -> None:
    llm = ScriptedLLM([_answer("hi", thinking="Greet.")])
    _, [row] = await _provider_rows(tmp_path, llm)
    assert (row["reasoning_key"], row["thinking"]) == ("", "Greet.")


# ── Failures that are not the run's ─────────────────────────────────────────


class CancellingCallHistory(History):
    """In-memory history that keeps model calls, but whose writers fail with a
    CancelledError of their own, while nothing cancels the run."""

    async def add_llm_call(self, record: LLMCallRecord) -> None:
        raise asyncio.CancelledError("the history failed")

    async def link_llm_call_steps(
        self, task_id: str, seq: int, step_ids: Sequence[str]
    ) -> None:
        raise asyncio.CancelledError("the history failed")


async def test_a_history_failing_with_a_cancelled_error_is_no_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _agent(ScriptedLLM(_two_step_script()), None)
    agent.hippo.history = CancellingCallHistory()
    caplog.set_level(logging.DEBUG, logger=CALL_LOG_LOGGER)
    try:
        result = await agent.run_task("Echo a.")
    finally:
        await agent.close()
    assert (result.status, result.result) == ("completed", "a")
    levels = [r.levelno for r in caplog.records if r.name == CALL_LOG_LOGGER]
    # Two rows and one link failed: the first at WARNING.
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]


async def test_a_file_system_that_refuses_modes_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)
    monkeypatch.setattr(os, "fchmod", refuse)
    agent = _agent(ScriptedLLM(_two_step_script()), tmp_path / "memory")
    try:
        result = await agent.run_task("Echo a.")
    finally:
        await agent.close()
    assert result.status == "completed"
    rows = _call_rows(tmp_path / "memory" / "history.db", result.task_id)
    assert len(rows) == 2
