"""Cancelling a run that a caller follows (Agent.run_task with a sink) — no real model.

A recording sink makes each run interactive. The fakes block on an
``asyncio.Event``, so each test cancels a run where it means to: in a tool
call, in planning, in a history write, in a save. Nothing touches the
network, and the only processes are the shell commands of
``test_cancel_kills_a_shell_command_and_its_group``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.agent.agent as agent_mod
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.agent import CANCELLED_ERROR
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import ToolOutput
from localmelo.melo.memory.history import History
from localmelo.melo.schema import (
    CALL_INTERRUPTED,
    CALL_NOT_STARTED,
    ImagePart,
    Message,
    RunResult,
    StepRecord,
    TaskRecord,
    ToolCall,
    ToolDef,
)

# ── Fakes ───────────────────────────────────────────────────────────────────


class ScriptedLLM(BaseLLMProvider):
    """Answers with each scripted reply in turn, then waits until cancelled.

    ``requests`` keeps the messages of every request; ``blocked`` is set
    once a request waits for a reply that is not scripted.
    """

    def __init__(self, replies: Iterable[Message] = ()) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        if self.replies:
            return self.replies.popleft()
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def close(self) -> None:
        pass


class CountingEmbedding(BaseEmbeddingProvider):
    """Embeds every text alike and counts its requests."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


class GatedHistory(History):
    """In-memory history whose writes can wait at a gate, as a locked file would.

    ``save_gate(task)`` and ``step_gate(step)`` say whether a save_task or
    an add_step waits until ``release`` is set; ``holding`` is set while
    one waits. With ``after_store``, a held add_step stores its row first
    and waits after. ``saves`` records the status and result of each save
    as it was made, and ``rows`` each stored step. A repeated ``step_id``
    raises ``sqlite3.IntegrityError``, as ``SqliteHistory``'s primary key
    does.
    """

    def __init__(
        self,
        *,
        save_gate: Callable[[TaskRecord], bool] = lambda task: False,
        step_gate: Callable[[StepRecord], bool] = lambda step: False,
        after_store: bool = False,
        on_step: Callable[[], None] = lambda: None,
    ) -> None:
        super().__init__()
        self.save_gate = save_gate
        self.step_gate = step_gate
        self.after_store = after_store
        self.on_step = on_step
        self.release = asyncio.Event()
        self.holding = asyncio.Event()
        self.saves: list[tuple[str, str]] = []
        self.rows: list[StepRecord] = []

    async def save_task(self, task: TaskRecord) -> None:
        if self.save_gate(task):
            await self._hold()
        self.saves.append((task.status, task.result))
        await super().save_task(task)

    async def add_step(self, task_id: str, step: StepRecord) -> None:
        if step.step_id in {row.step_id for row in self.rows}:
            raise sqlite3.IntegrityError("UNIQUE constraint failed: steps.step_id")
        self.on_step()
        held = self.step_gate(step)
        if held and not self.after_store:
            await self._hold()
        self.rows.append(step)
        await super().add_step(task_id, step)
        if held and self.after_store:
            await self._hold()

    async def _hold(self) -> None:
        self.holding.set()
        await self.release.wait()


class ExitProvider(BaseToolProvider):
    """Provides no tools; with *close_gate*, close() waits for it. Counts kills."""

    def __init__(self, close_gate: asyncio.Event | None = None) -> None:
        self.close_gate = close_gate
        self.closes = 0
        self.kills = 0

    async def start(self) -> list[ProvidedTool]:
        return []

    async def close(self) -> None:
        self.closes += 1
        if self.close_gate is not None:
            await self.close_gate.wait()

    def kill(self) -> None:
        self.kills += 1


class Recorder:
    """A sink that keeps every event."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    @property
    def types(self) -> list[str]:
        return [event.type for event in self.events]

    def of(self, type_: str) -> list[AgentEvent]:
        return [event for event in self.events if event.type == type_]


PNG = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n a small photo", "image/png")


class Tools:
    """The custom tools; ``called`` names each call that started to run."""

    def __init__(self) -> None:
        self.called: list[str] = []
        self.hanging = asyncio.Event()  # hang() is waiting
        self.at_gate = asyncio.Event()  # gate() is waiting for go
        self.go = asyncio.Event()

    async def echo(self, text: str = "") -> str:
        self.called.append("echo")
        return text

    async def photo(self) -> ToolOutput:
        self.called.append("photo")
        return ToolOutput(text="a photo", images=[PNG])

    async def hang(self) -> str:
        self.called.append("hang")
        self.hanging.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def gate(self) -> str:
        self.called.append("gate")
        self.at_gate.set()
        await self.go.wait()
        return "through"

    async def spurious(self) -> str:
        """Raises CancelledError while nothing cancels the run."""
        self.called.append("spurious")
        raise asyncio.CancelledError()


def _tool(name: str, description: str, **properties: str) -> ToolDef:
    return ToolDef(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in properties.items()},
        },
    )


TOOL_DEFS = {
    "echo": _tool("echo", "Echo text back", text="string"),
    "photo": _tool("photo", "Take a photo"),
    "hang": _tool("hang", "Never return"),
    "gate": _tool("gate", "Wait until told to go on"),
    "spurious": _tool("spurious", "Fail with a CancelledError"),
}


async def _no_wait(delay: float) -> None:
    pass


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _agent(
    llm: BaseLLMProvider,
    *,
    tools: Tools | None = None,
    history: History | None = None,
    embedding: BaseEmbeddingProvider | None = None,
    providers: Sequence[BaseToolProvider] = (),
    **options: Any,
) -> Agent:
    """An agent with the custom tools, offered on every step, and no waits."""
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    agent = Agent(
        llm=llm,
        embedding=embedding,
        options=AgentOptions(**options),
        tool_providers=providers,
    )
    if history is not None:
        agent.hippo.history = history
    tools = tools or Tools()
    for name, definition in TOOL_DEFS.items():
        agent.hippo.register_tool(definition)
        agent.executor.register(name, getattr(tools, name))
    agent.chat.sleep = _no_wait
    return agent


# ── Replies ─────────────────────────────────────────────────────────────────


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _calls(*calls: tuple[str, dict[str, Any], str], content: str = "") -> Message:
    """A reply with tool calls, each ``(name, arguments, id)``."""
    return Message(
        role="assistant",
        content=content,
        tool_calls=[ToolCall(tool_name=n, arguments=a, id=i) for n, a, i in calls],
    )


def _reflection(action: str) -> Message:
    decision = {
        "summary": "Echoed once.",
        "progress_made": True,
        "task_still_feasible": True,
        "new_information_gained": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "recommended_action": action,
        "rationale": "",
        "best_effort_result": "",
    }
    return _answer(json.dumps(decision))


# ── Checks ──────────────────────────────────────────────────────────────────


async def _until(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), 5.0)


async def _stop(running: asyncio.Task[Any], message: str | None = None) -> None:
    """Cancel *running*, as a caller's Stop does, and see the cancellation out."""
    running.cancel(message)
    with pytest.raises(asyncio.CancelledError):
        await running


def check_events(recorder: Recorder) -> RunResult:
    """E-1, E-2, E-4 and E-5 for one run; returns run.finished's RunResult."""
    types = recorder.types
    assert types[0] == "run.started" and types[-1] == "run.finished"
    assert types.count("run.started") == types.count("run.finished") == 1
    assert [event.seq for event in recorder.events] == list(range(1, len(types) + 1))
    open_call: str | None = None
    for event in recorder.events:
        if event.type == "tool.started":
            assert open_call is None
            open_call = event.data["call_id"]
        elif event.type == "tool.finished":
            assert event.data["call_id"] == open_call
            open_call = None
        elif event.type in ("step.started", "run.finished"):
            assert open_call is None
    result = RunResult(**recorder.events[-1].data["result"])
    assert types.count("step.started") == result.steps
    assert types.count("tool.started") == result.n_tool
    assert {event.run_id for event in recorder.events} == {result.task_id}
    return result


def assert_protocol_valid(messages: list[Message]) -> None:
    """Every tool call is answered once, right after it; no orphan results."""
    pending: set[str] = set()
    for msg in messages:
        if msg.role == "tool":
            assert msg.tool_call_id in pending, f"orphan result {msg.tool_call_id!r}"
            pending.remove(msg.tool_call_id)
            continue
        assert not pending, f"unanswered tool calls {pending}"
        if msg.role == "assistant" and msg.tool_calls:
            pending = {tc.id for tc in msg.tool_calls}
    assert not pending, f"unanswered tool calls {pending}"


def _window(agent: Agent) -> list[tuple[str, str]]:
    return [(msg.role, msg.content) for msg in agent.hippo.working.get_window()]


def _promoted(agent: Agent) -> list[str]:
    return [
        entry.text
        for entry in agent.hippo.long._entries
        if entry.metadata.get("type") == "reflection"
    ]


def _row_calls(rows: Iterable[StepRecord]) -> list[str]:
    return [row.tool_call.id for row in rows if row.tool_call is not None]


# ── A Stop ──────────────────────────────────────────────────────────────────


class TestStop:
    @pytest.mark.asyncio
    async def test_cancel_during_tool(self) -> None:
        tools, history, recorder = Tools(), GatedHistory(), Recorder()
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools, history=history
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        await _stop(running)

        [finished] = recorder.of("tool.finished")
        assert (finished.data["status"], finished.data["error"]) == (
            "cancelled",
            CALL_INTERRUPTED,
        )
        result = check_events(recorder)
        assert (result.status, result.error, result.result) == (
            "cancelled",
            "Cancelled by the caller",
            "",
        )
        assert CANCELLED_ERROR == "Cancelled by the caller"
        assert (result.steps, result.n_tool) == (1, 1)
        assert history.saves == [
            ("running", ""),
            ("cancelled", "Cancelled by the caller"),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message", "error"),
        [("Stopped", "Stopped"), ("", "Cancelled by the caller")],
    )
    async def test_cancel_reason_becomes_the_error(
        self, message: str, error: str
    ) -> None:
        tools, history, recorder = Tools(), GatedHistory(), Recorder()
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools, history=history
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        await _stop(running, message)

        result = check_events(recorder)
        assert (result.status, result.error) == ("cancelled", error)
        assert history.saves[-1] == ("cancelled", error)

    @pytest.mark.asyncio
    async def test_cancel_closes_the_partial_turn(self) -> None:
        tools, history = Tools(), GatedHistory()
        reply = _calls(
            ("photo", {}, "call_1"),
            ("hang", {}, "call_2"),
            ("echo", {"text": "c"}, "call_3"),
            content="Three calls.",
        )
        agent = _agent(ScriptedLLM([reply]), tools=tools, history=history)
        running = asyncio.create_task(agent.run_task("go", on_event=Recorder()))
        await _until(tools.hanging)
        await _stop(running)

        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        user, assistant, photo, hang, echo = window
        assert (user.role, user.content) == ("user", "go")
        assert [call.id for call in assistant.tool_calls] == [
            "call_1",
            "call_2",
            "call_3",
        ]
        assert assistant.content == "Three calls."
        assert photo.content.startswith("a photo\n[1 image: sha256:")
        assert hang.content == f"Error: {CALL_INTERRUPTED}"
        assert echo.content == f"Error: {CALL_NOT_STARTED}"
        # The third call never ran.
        assert tools.called == ["photo", "hang"]
        # Only the reference of the photo stays.
        [image] = photo.images
        assert (image.data, image.sha256) == ("", PNG.sha256)

        # History has one row per call, the text on the first.
        assert _row_calls(history.rows) == ["call_1", "call_2", "call_3"]
        assert [row.thought for row in history.rows] == ["Three calls.", "", ""]
        results = [row.tool_result for row in history.rows]
        assert [r.error if r else None for r in results] == [
            "",
            CALL_INTERRUPTED,
            CALL_NOT_STARTED,
        ]
        assert results[0] is not None
        assert [(i.data, i.sha256) for i in results[0].images] == [("", PNG.sha256)]

    @pytest.mark.asyncio
    async def test_cancel_during_history_write(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The first add_step stores its row, then blocks: the cancellation
        # finds it unfinished, and the finalizer writes the row again.
        history = GatedHistory(step_gate=lambda step: True, after_store=True)
        reply = _calls(
            ("echo", {"text": "a"}, "call_1"), ("echo", {"text": "b"}, "call_2")
        )
        agent = _agent(ScriptedLLM([reply]), history=history)
        running = asyncio.create_task(agent.run_task("echo", on_event=Recorder()))
        await _until(history.holding)
        history.step_gate = lambda step: False
        with caplog.at_level(logging.DEBUG, logger=agent_mod.__name__):
            await _stop(running)

        # Each call ends with one row.
        assert _row_calls(history.rows) == ["call_1", "call_2"]
        assert history.saves[-1] == ("cancelled", "Cancelled by the caller")
        assert "was saved before the cancellation" in caplog.text
        # Both calls had returned: the closed step holds their real results.
        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        assert [msg.content for msg in window if msg.role == "tool"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_cancel_with_the_history_locked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Another connection holds history.db's write lock while the run is
        # cancelled in add_step, at its BEGIN IMMEDIATE.
        tools, recorder = Tools(), Recorder()
        reply = _calls(("gate", {}, "call_1"), ("echo", {"text": "b"}, "call_2"))
        agent = _agent(ScriptedLLM([reply]), tools=tools, memory_dir=str(tmp_path))
        history = agent.hippo.history
        writing = asyncio.Event()
        add_step = history.add_step

        async def add_step_and_tell(task_id: str, step: StepRecord) -> None:
            writing.set()
            await add_step(task_id, step)

        monkeypatch.setattr(history, "add_step", add_step_and_tell)
        db_path = tmp_path / "history.db"
        try:
            running = asyncio.create_task(agent.run_task("go", on_event=recorder))
            await _until(tools.at_gate)  # the task row reads "running" by now
            with contextlib.closing(
                sqlite3.connect(db_path, isolation_level=None)
            ) as other:
                other.execute("BEGIN IMMEDIATE")
                tools.go.set()
                await _until(writing)
                await asyncio.sleep(0.2)  # add_step waits for the lock
                running.cancel()
                await asyncio.sleep(0.05)
                other.execute("ROLLBACK")
                with pytest.raises(asyncio.CancelledError):
                    await running
        finally:
            await agent.close()

        result = check_events(recorder)
        assert result.status == "cancelled"
        with contextlib.closing(sqlite3.connect(db_path)) as db:
            rows = db.execute(
                "SELECT tool_call_json FROM steps WHERE task_id = ? ORDER BY seq",
                (result.task_id,),
            ).fetchall()
            tasks = db.execute("SELECT status, result FROM tasks").fetchall()
        # Each call once, and the row cancelled.
        assert [json.loads(row[0])["id"] for row in rows] == ["call_1", "call_2"]
        assert tasks == [("cancelled", "Cancelled by the caller")]

    @pytest.mark.asyncio
    async def test_cancel_during_planning(self) -> None:
        llm, recorder = ScriptedLLM(), Recorder()
        agent = _agent(llm)
        running = asyncio.create_task(agent.run_task("q", on_event=recorder))
        await _until(llm.blocked)
        await _stop(running)

        result = check_events(recorder)
        assert recorder.types == ["run.started", "step.started", "run.finished"]
        assert (result.status, result.steps, result.n_llm, result.llm_errors) == (
            "cancelled",
            1,
            0,
            1,
        )
        assert _window(agent) == [("user", "q")]

    @pytest.mark.asyncio
    async def test_cancel_before_the_first_save(self) -> None:
        # The first save of the task row waits, as on a locked file.
        history = GatedHistory(save_gate=lambda task: task.status == "running")
        llm, recorder = ScriptedLLM(), Recorder()
        agent = _agent(llm, history=history)
        running = asyncio.create_task(agent.run_task("q", on_event=recorder))
        await _until(history.holding)
        await _stop(running)

        result = check_events(recorder)
        assert (result.status, result.steps) == ("cancelled", 0)
        # The query is in working memory, so the next run's roles alternate.
        assert _window(agent) == [("user", "q")]
        assert history.saves == [("cancelled", "Cancelled by the caller")]
        assert llm.requests == []

    @pytest.mark.asyncio
    async def test_cancel_skips_promotion(self) -> None:
        embedding = CountingEmbedding()
        llm, recorder = ScriptedLLM([_calls(("echo", {"text": "a"}, "c1"))]), Recorder()
        llm.replies.append(_reflection("continue"))
        agent = _agent(llm, embedding=embedding, steps_per_attempt=1, max_attempts=2)
        running = asyncio.create_task(agent.run_task("echo a", on_event=recorder))
        await _until(llm.blocked)  # attempt 2 is planning
        assert agent.hippo.working.get_reflections()
        embeds, requests = embedding.calls, len(llm.requests)
        await _stop(running)

        assert check_events(recorder).status == "cancelled"
        # Cleared, not promoted: no network call follows the cancellation.
        assert agent.hippo.working.get_reflections() == []
        assert _promoted(agent) == []
        assert (embedding.calls, len(llm.requests)) == (embeds, requests)

    @pytest.mark.asyncio
    async def test_cancel_kills_a_shell_command_and_its_group(
        self, tmp_path: Path
    ) -> None:
        pid_file = tmp_path / "pids"
        command = f'sleep 60 & echo $$ $! > "{pid_file}"; wait'
        reply = _calls(("shell_exec", {"command": command}, "call_1"))
        agent = _agent(ScriptedLLM([reply]), builtins=frozenset({"shell_exec"}))
        running = asyncio.create_task(agent.run_task("run it", on_event=Recorder()))
        try:
            shell = await _shell_pid(pid_file)
            await _stop(running)
            # No process of the killed group survives.
            assert await _group_gone(shell)
        finally:
            for pid in pid_file.read_text().split() if pid_file.exists() else []:
                with contextlib.suppress(OSError):
                    os.kill(int(pid), signal.SIGKILL)
        assert agent.hippo.working.get_window()[-1].content == (
            f"Error: {CALL_INTERRUPTED}"
        )


async def _shell_pid(pid_file: Path) -> int:
    """The PID of the shell, once the command has written it."""
    for _ in range(500):
        text = pid_file.read_text() if pid_file.exists() else ""
        if text.endswith("\n"):
            return int(text.split()[0])
        await asyncio.sleep(0.01)
    raise AssertionError("the command did not write its PIDs")


async def _group_gone(pgid: int, timeout_s: float = 5.0) -> bool:
    """Whether process group *pgid* is gone within *timeout_s*.

    ``PermissionError`` means not gone yet: macOS raises it for zombies
    that are not reaped yet.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        await asyncio.sleep(0.02)
    return False


# ── Runs without a sink or policy, timeouts and late cancellations ──────────


class TestOtherCancellations:
    @pytest.mark.asyncio
    async def test_cancel_without_sink_or_policy(self) -> None:
        # Main's behavior, B6's exit included: crashed, and no closed step.
        script = [
            _calls(("echo", {"text": "a"}, "call_1")),
            _reflection("continue"),
            _calls(("hang", {}, "call_2")),
        ]
        for exit_reason in (None, "Cancelled by SIGTERM"):
            tools, history = Tools(), GatedHistory()
            agent = _agent(
                ScriptedLLM(script),
                tools=tools,
                history=history,
                embedding=CountingEmbedding(),
                steps_per_attempt=1,
                max_attempts=2,
            )
            running = asyncio.create_task(agent.run_task("echo a"))
            await _until(tools.hanging)
            if exit_reason is not None:
                agent.begin_exit(exit_reason, within_s=5.0)
            await _stop(running)

            if exit_reason is None:
                assert history.saves[-1] == ("crashed", "CancelledError")
                assert len(_promoted(agent)) == 1
            else:
                # B6: the exit's reason, and no promotion.
                assert history.saves[-1] == ("crashed", exit_reason)
                assert _promoted(agent) == []
            assert agent.hippo.working.get_reflections() == []
            # No partial turn for the interrupted step, and no row for it.
            assert _window(agent) == [
                ("user", "echo a"),
                ("assistant", ""),
                ("tool", "a"),
            ]
            assert _row_calls(history.rows) == ["call_1"]

    @pytest.mark.asyncio
    async def test_spurious_cancelled_error_is_a_crash(self) -> None:
        # A tool raises CancelledError while nothing cancels the run: that
        # is the tool's failure, and it propagates, as on main.
        history, recorder = GatedHistory(), Recorder()
        agent = _agent(
            ScriptedLLM([_calls(("spurious", {}, "call_1"))]), history=history
        )
        with pytest.raises(asyncio.CancelledError):
            await agent.run_task("q", on_event=recorder)

        result = check_events(recorder)
        assert (result.status, result.error) == ("crashed", "CancelledError")
        [finished] = recorder.of("tool.finished")
        assert (finished.data["status"], finished.data["error"]) == (
            "crashed",
            "CancelledError",
        )
        assert history.saves[-1] == ("crashed", "CancelledError")
        assert _window(agent) == [("user", "q")]

    @pytest.mark.asyncio
    async def test_timeout_still_returns_timeout(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_calls(("hang", {}, "call_1"))]), timeout_s=0.2)
        result = await agent.run_task("wait", on_event=recorder)

        assert (result.status, result.error) == ("timeout", "Timed out after 0.2 s")
        assert check_events(recorder) == result
        [finished] = recorder.of("tool.finished")
        assert finished.data["status"] == "cancelled"
        # No partial turn, as in Phase 1a.
        assert _window(agent) == [("user", "wait")]

    @pytest.mark.asyncio
    async def test_external_cancel_with_timeout_set(self) -> None:
        tools, recorder = Tools(), Recorder()
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools, timeout_s=30
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        await _stop(running)

        result = check_events(recorder)
        assert (result.status, result.error) == ("cancelled", "Cancelled by the caller")
        assert_protocol_valid(agent.hippo.working.get_window())

    @pytest.mark.asyncio
    async def test_late_cancel_keeps_the_status(self) -> None:
        # The run has completed; the cancellation lands in its final save.
        history = GatedHistory(save_gate=lambda task: task.status == "completed")
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_answer("done")]), history=history)
        running = asyncio.create_task(agent.run_task("q", on_event=recorder))
        await _until(history.holding)
        await _stop(running)

        result = check_events(recorder)
        assert (result.status, result.result, result.error) == ("completed", "done", "")
        assert _window(agent) == [("user", "q")]

    @pytest.mark.asyncio
    async def test_next_run_after_cancel_is_valid(self) -> None:
        tools = Tools()
        reply = _calls(
            ("echo", {"text": "a"}, "call_1"),
            ("hang", {}, "call_2"),
            ("echo", {"text": "c"}, "call_3"),
        )
        llm = ScriptedLLM([reply])
        agent = _agent(llm, tools=tools)
        running = asyncio.create_task(agent.run_task("first", on_event=Recorder()))
        await _until(tools.hanging)
        await _stop(running)

        llm.replies.append(_answer("done"))
        second = await agent.run_task("second", on_event=Recorder())
        assert (second.status, second.result) == ("completed", "done")
        request = llm.requests[-1]
        assert_protocol_valid(request)
        assert [msg.content for msg in request if msg.role == "tool"] == [
            "a",
            f"Error: {CALL_INTERRUPTED}",
            f"Error: {CALL_NOT_STARTED}",
        ]
        assert [msg.content for msg in request if msg.role == "user"] == [
            "first",
            "second",
        ]

    @pytest.mark.asyncio
    async def test_run_propagates_cancellation(self) -> None:
        # run_task re-raises the cancellation, so the scopes that cancel it
        # work: asyncio.timeout raises its TimeoutError ...
        recorder = Recorder()
        agent = _agent(ScriptedLLM())
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await agent.run_task("q", on_event=recorder)
        assert check_events(recorder).status == "cancelled"

        # ... and a TaskGroup whose other task fails ends with that failure.
        async def fail() -> None:
            await asyncio.sleep(0.05)
            raise RuntimeError("sibling failed")

        recorder = Recorder()
        with pytest.raises(ExceptionGroup) as raised:
            async with asyncio.TaskGroup() as group:
                group.create_task(agent.run_task("q", on_event=recorder))
                group.create_task(fail())
        assert [str(exc) for exc in raised.value.exceptions] == ["sibling failed"]
        assert check_events(recorder).status == "cancelled"


# ── The finalizer ───────────────────────────────────────────────────────────


async def _until_finished(recorder: Recorder) -> None:
    """Wait for run.finished, which a finalizer may send after run_task raised."""
    for _ in range(500):
        if "run.finished" in recorder.types:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("run.finished did not come")


class TestFinalizer:
    @pytest.mark.asyncio
    async def test_second_cancel_still_finalizes(self) -> None:
        tools, recorder = Tools(), Recorder()
        history = GatedHistory(save_gate=lambda task: task.status == "cancelled")
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools, history=history
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        running.cancel()
        await _until(history.holding)  # the finalizer saves the task row

        # A second cancellation ends run_task at once ...
        started = time.monotonic()
        await _stop(running)
        assert time.monotonic() - started < 1.0
        assert "run.finished" not in recorder.types
        # ... and close() waits for the finalizer.
        closing = asyncio.create_task(agent.close())
        await asyncio.sleep(0.1)
        assert not closing.done()
        history.release.set()
        await asyncio.wait_for(closing, 5.0)

        assert recorder.types.count("run.finished") == 1
        assert check_events(recorder).status == "cancelled"
        assert history.saves[-1] == ("cancelled", "Cancelled by the caller")

    @pytest.mark.asyncio
    async def test_finalizer_timeout(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(agent_mod, "FINALIZE_TIMEOUT_S", 0.2)
        tools, recorder = Tools(), Recorder()
        history = GatedHistory(save_gate=lambda task: task.status == "cancelled")
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools, history=history
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        with caplog.at_level(logging.ERROR, logger=agent_mod.__name__):
            await _stop(running)

        result = check_events(recorder)
        assert result.status == "cancelled"
        [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert record.getMessage() == (
            f"Finalizing cancelled task {result.task_id} timed out; its records "
            "may be incomplete"
        )
        # The row could not be saved in time; the step's row was.
        assert history.saves == [("running", "")]
        assert _row_calls(history.rows) == ["call_1"]


# ── An exit (Agent.begin_exit) ──────────────────────────────────────────────


class TestExit:
    @pytest.mark.asyncio
    async def test_cancel_after_begin_exit(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(agent_mod, "EXIT_SAVE_RESERVE_S", 0.1)
        hang = asyncio.Event()
        provider = ExitProvider(close_gate=hang)
        embedding = CountingEmbedding()
        tools, recorder = Tools(), Recorder()
        history = GatedHistory(step_gate=lambda step: True)  # add_step hangs
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]),
            tools=tools,
            history=history,
            embedding=embedding,
            providers=[provider],
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        embeds = embedding.calls

        started = time.monotonic()
        agent.begin_exit("Cancelled by SIGTERM", within_s=0.5)
        with caplog.at_level(logging.ERROR, logger=agent_mod.__name__):
            await _stop(running)
        elapsed = time.monotonic() - started

        # One run.finished, with the exit's reason, within the exit's time
        # (the slack is test_exit.py's, for a loaded machine).
        assert recorder.types.count("run.finished") == 1
        result = check_events(recorder)
        assert (result.status, result.error) == ("cancelled", "Cancelled by SIGTERM")
        assert 0.3 <= elapsed < 1.0
        # The provider's close() hung, so it was killed; the hanging history
        # write was cut short, but the row was saved.
        assert (provider.closes, provider.kills) == (1, 1)
        assert "timed out; its records may be incomplete" in caplog.text
        assert history.saves[-1] == ("cancelled", "Cancelled by SIGTERM")
        # No network call: no embedding request, no promotion.
        assert embedding.calls == embeds
        hang.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_an_exit_after_a_stop_bounds_what_is_left(self) -> None:
        tools, recorder = Tools(), Recorder()
        history = GatedHistory(save_gate=lambda task: task.status == "cancelled")
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools, history=history
        )
        # The process must exit while the finalizer writes the step's row.
        history.on_step = lambda: agent.begin_exit("Cancelled by SIGTERM", within_s=0.5)
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)

        started = time.monotonic()
        await _stop(running, "Stopped")
        elapsed = time.monotonic() - started

        # The save that hangs is bounded by the exit, not by FINALIZE_TIMEOUT_S.
        result = check_events(recorder)
        assert (result.status, result.error) == ("cancelled", "Stopped")
        assert 0.4 <= elapsed < 1.0
        assert history.saves == [("running", "")]

    @pytest.mark.asyncio
    async def test_close_keeps_to_the_exit_deadline(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hang = asyncio.Event()
        provider = ExitProvider(close_gate=hang)
        tools, recorder = Tools(), Recorder()
        history = GatedHistory(step_gate=lambda step: True)  # add_step hangs
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"))]),
            tools=tools,
            history=history,
            providers=[provider],
        )
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        await _until(tools.hanging)
        running.cancel()
        await _until(history.holding)  # the finalizer is stuck in add_step
        await _stop(running)

        started = time.monotonic()
        agent.begin_exit("Cancelled by SIGTERM", within_s=0.5)
        with caplog.at_level(logging.ERROR, logger=agent_mod.__name__):
            await asyncio.wait_for(agent.close(), 2.0)
        elapsed = time.monotonic() - started

        assert 0.4 <= elapsed < 1.0
        task_id = recorder.events[0].run_id
        assert [
            r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
        ] == [
            f"Cancelled task {task_id} is still being finalized; closing without "
            "waiting for it"
        ]
        assert provider.kills == 1

        # Let the finalizer end.
        history.release.set()
        hang.set()
        await _until_finished(recorder)
        assert check_events(recorder).status == "cancelled"
