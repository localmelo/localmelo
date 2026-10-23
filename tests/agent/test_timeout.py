"""AgentOptions.timeout_s ends a run with status timeout — no real model."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import Message, TaskRecord, ToolCall, ToolDef


class ScriptedLLM(BaseLLMProvider):
    """Pops queued replies; raises queued exceptions; blocks when empty.

    ``blocked`` is set once a request finds the queue empty. A queued
    float is a delay in seconds before the next reply.
    """

    def __init__(self, replies: list[Message | BaseException | float]) -> None:
        self.replies: deque[Message | BaseException | float] = deque(replies)
        self.requests = 0
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests += 1
        while self.replies:
            reply = self.replies.popleft()
            if isinstance(reply, float):
                await asyncio.sleep(reply)
                continue
            if isinstance(reply, BaseException):
                raise reply
            return reply
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def close(self) -> None:
        pass


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _agent(llm: BaseLLMProvider, **options: Any) -> Agent:
    """An agent without built-in tools, with the given options."""
    return Agent(llm=llm, options=AgentOptions(builtins=frozenset(), **options))


def _only_task(agent: Agent) -> TaskRecord:
    tasks = list(agent.hippo.history._tasks.values())
    assert len(tasks) == 1
    return tasks[0]


class TestTimeout:
    @pytest.mark.asyncio
    async def test_a_run_still_going_ends_as_timeout(self) -> None:
        llm = ScriptedLLM([])  # the first request never returns
        agent = _agent(llm, timeout_s=0.05)
        result = await agent.run_task("q")
        assert (result.status, result.result, result.error) == (
            "timeout",
            "",
            "Timed out after 0.05 s",
        )
        # The planning call was made; the request it was waiting for was
        # abandoned and counts as failed.
        assert (result.steps, result.attempts) == (1, 1)
        assert (result.n_llm, result.llm_errors) == (0, 1)
        assert result.config["timeout_s"] == 0.05
        assert 0.04 <= result.duration_s < 5.0
        task = _only_task(agent)
        assert (task.status, task.result) == ("timeout", "Timed out after 0.05 s")

    @pytest.mark.asyncio
    async def test_run_shows_the_status_and_the_error(self) -> None:
        agent = _agent(ScriptedLLM([]), timeout_s=1e-3)
        assert await agent.run("q") == "[timeout] Timed out after 0.001 s"

    @pytest.mark.asyncio
    async def test_a_run_that_finishes_in_time_is_not_affected(self) -> None:
        llm = ScriptedLLM([0.02, _answer("42")])
        result = await _agent(llm, timeout_s=30).run_task("q")
        assert (result.status, result.result) == ("completed", "42")

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self) -> None:
        llm = ScriptedLLM([0.05, _answer("42")])
        agent = _agent(llm)
        assert agent.options.timeout_s is None
        result = await agent.run_task("q")
        assert (result.status, result.result) == ("completed", "42")

    @pytest.mark.asyncio
    async def test_the_timeout_cuts_the_wait_before_a_retry_short(self) -> None:
        busy = ProviderError("HTTP 503", status_code=503, retryable=True)
        llm = ScriptedLLM([busy, _answer("never")])
        agent = _agent(llm, timeout_s=0.2)  # the real 1 s backoff
        started = time.perf_counter()
        result = await agent.run_task("q")
        assert time.perf_counter() - started < 0.9
        assert result.status == "timeout"
        assert (llm.requests, result.n_llm, result.llm_errors) == (1, 0, 1)

    @pytest.mark.asyncio
    async def test_a_timeout_error_from_anywhere_else_is_a_crash(self) -> None:
        llm = ScriptedLLM([TimeoutError("read timed out")])
        result = await _agent(llm, timeout_s=30).run_task("q")
        assert (result.status, result.error) == (
            "crashed",
            "TimeoutError: read timed out",
        )

    @pytest.mark.asyncio
    async def test_an_outside_cancellation_still_propagates(self) -> None:
        llm = ScriptedLLM([])
        agent = _agent(llm, timeout_s=30)
        running = asyncio.create_task(agent.run_task("q"))
        await llm.blocked.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert _only_task(agent).status == "crashed"

    @pytest.mark.asyncio
    async def test_the_agent_runs_again_after_a_timeout(self) -> None:
        llm = ScriptedLLM([5.0, _answer("second")])
        agent = _agent(llm, timeout_s=0.05)
        first = await agent.run_task("q1")
        second = await agent.run_task("q2")
        assert first.status == "timeout"
        assert (second.status, second.result) == ("completed", "second")

    @pytest.mark.asyncio
    async def test_the_task_row_is_saved_to_sqlite(self, tmp_path: Path) -> None:
        agent = _agent(ScriptedLLM([]), timeout_s=0.05, memory_dir=str(tmp_path))
        result = await agent.run_task("q")
        await agent.close()
        history = SqliteHistory(tmp_path / "history.db")
        try:
            task = await history.get_task(result.task_id)
        finally:
            await history.aclose()
        assert task is not None
        assert (task.status, task.result) == ("timeout", "Timed out after 0.05 s")


# ── A shell command the timeout interrupts ──────────────────────────────────


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


@pytest.fixture
def pid_file(tmp_path: Path) -> Iterator[Path]:
    """Where the command writes its PIDs; leftovers are killed afterwards."""
    path = tmp_path / "pids"
    yield path
    if path.exists():
        for pid in path.read_text().split():
            with contextlib.suppress(OSError):
                os.kill(int(pid), signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_timeout_kills_the_shell_command_it_interrupts(pid_file: Path) -> None:
    command = f'sleep 60 & echo $$ $! > "{pid_file}"; wait'
    call = Message(
        role="assistant",
        tool_call=ToolCall(tool_name="shell_exec", arguments={"command": command}),
    )
    agent = Agent(
        llm=ScriptedLLM([call]),
        options=AgentOptions(
            builtins=frozenset({"shell_exec"}), tool_exposure="all", timeout_s=1.0
        ),
    )
    result = await agent.run_task("wait in the shell")
    assert result.status == "timeout"
    # The call was handed to the executor before the timeout cut it short.
    assert (result.steps, result.n_tool) == (1, 1)
    shell, child = (int(pid) for pid in pid_file.read_text().split())
    assert await _gone(shell)
    assert await _gone(child)


@pytest.mark.asyncio
async def test_a_timeout_cannot_interrupt_python_exec() -> None:
    # A documented limit: python_exec runs its code on the event loop, so
    # the deadline takes effect only once the code returns.
    code = "import time\ntime.sleep(0.3)"
    call = Message(
        role="assistant",
        tool_call=ToolCall(tool_name="python_exec", arguments={"code": code}),
    )
    agent = Agent(
        llm=ScriptedLLM([call]),
        options=AgentOptions(
            builtins=frozenset({"python_exec"}), tool_exposure="all", timeout_s=0.05
        ),
    )
    result = await agent.run_task("run python")
    assert (result.status, result.n_tool) == ("timeout", 1)
    assert result.duration_s >= 0.3
