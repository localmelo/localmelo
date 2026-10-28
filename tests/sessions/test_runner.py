"""SessionRunner: turns, their stored events, causes, keys, approvals, checkpoints.

Each model is scripted and in-process; each turn runs on an Agent with a
temporary workspace, in-memory history and a temporary sessions file.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.support.sessions.runner as runner_mod
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.schema import INTERRUPT_LABEL, RunResult
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.support.sessions.runner import (
    CLI_FINISH_RETRY_DELAYS_S,
    SINK_TIMEOUT_S,
    TAKEOVER_REASON,
    SessionRunner,
)
from localmelo.support.sessions.store import (
    LEASE_S,
    NewSession,
    SessionBusyError,
    SessionStore,
    StoredEvent,
    TurnInfo,
)
from localmelo.support.sessions.submissions import (
    SubmissionConflict,
    SubmissionRepeated,
    SubmissionWithdrawn,
)
from localmelo.support.sessions.window import decode_window

from .helpers import (
    Clock,
    ScriptedLLM,
    answer,
    check_invariants,
    rows,
    started_agent,
    tool_call,
)

SECRET = "sk-" + "f00dfeed12" * 3


class Setup:
    """A store, an Agent on a scripted model, and a runner of one new session."""

    def __init__(
        self,
        path: Path,
        store: SessionStore,
        llm: ScriptedLLM,
        agent: Agent,
        *,
        progress: bool = False,
    ):
        self.path = path
        self.store = store
        self.llm = llm
        self.agent = agent
        self.events: list[StoredEvent] = []
        self.runner = SessionRunner(
            agent,
            store,
            None,
            new_session=NewSession(),
            on_event=self.events.append,
            progress=progress,
        )

    async def aclose(self) -> None:
        await self.runner.close()
        await self.agent.close()
        await self.store.aclose()


async def setup(
    tmp_path: Path,
    replies: list[Any] | None = None,
    *,
    writer: str = "gateway",
    clock: Clock | None = None,
    progress: bool = False,
) -> Setup:
    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer=writer, clock=clock or Clock())  # type: ignore[arg-type]
    llm = ScriptedLLM(replies or [])
    agent = await started_agent(tmp_path / "ws", llm)
    return Setup(path, store, llm, agent, progress=progress)


def types_of(events: list[StoredEvent]) -> list[str]:
    return [e.event["type"] for e in events]


async def stored_events(s: Setup) -> list[StoredEvent]:
    assert s.runner.session_id is not None
    return await s.store.list_events(s.runner.session_id)


async def last_turn(s: Setup) -> TurnInfo:
    assert s.runner.session_id is not None
    return (await s.store.list_turns(s.runner.session_id))[-1]


# ── Checks ──────────────────────────────────────────────────────────────────


async def test_the_runner_refuses_agents_and_arguments_it_cannot_use(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    llm = ScriptedLLM()
    no_answers = Agent(llm=llm, options=AgentOptions(workspace_root=str(tmp_path)))
    no_root = Agent(llm=llm, options=AgentOptions(record_answers=True))
    good = await started_agent(tmp_path / "ws", llm)
    with pytest.raises(ValueError, match=r"AgentOptions\(record_answers=True\)"):
        SessionRunner(no_answers, store, None, new_session=NewSession())
    with pytest.raises(ValueError, match="needs an Agent built with a workspace_root"):
        SessionRunner(no_root, store, None, new_session=NewSession())
    with pytest.raises(ValueError, match="resume_mode"):
        SessionRunner(good, store, None, new_session=NewSession(), resume_mode="all")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SessionRunner(good, store, "abc", new_session=NewSession())
    with pytest.raises(ValueError):
        SessionRunner(good, store, None)
    runner = SessionRunner(good, store, None, new_session=NewSession())
    with pytest.raises(ValueError, match="record_answers"):
        runner.replace_agent(no_answers)
    with pytest.raises(ValueError, match="workspace_root"):
        runner.replace_agent(no_root)
    for agent in (no_answers, no_root, good):
        await agent.close()
    await store.aclose()


async def test_replace_agent_is_refused_during_a_turn_and_keeps_the_policy(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path)
    await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    other_llm = ScriptedLLM([answer("from the new agent")])
    other = await started_agent(tmp_path / "ws2", other_llm)
    with pytest.raises(SessionBusyError, match="is already running a turn"):
        s.runner.replace_agent(other)
    s.runner.cancel()
    await s.runner.wait()
    s.runner.replace_agent(other)
    # The runner stays: the next turn runs on the new Agent, in the same
    # session (observe_stores_the_request_and_the_answer checks the policy).
    result = await s.runner.run_turn("next")
    assert result.result == "from the new agent"
    assert [t.turn for t in await s.store.list_turns(s.runner.session_id)] == [1, 2]  # type: ignore[arg-type]
    await other.close()
    await s.aclose()


# ── Storage ─────────────────────────────────────────────────────────────────


async def test_a_turns_events_are_stored_in_order_with_its_outcome(
    tmp_path: Path,
) -> None:
    s = await setup(
        tmp_path, [tool_call("file_read", {"path": "notes.txt"}), answer("Buy milk.")]
    )
    result = await s.runner.run_turn("What do my notes say?")
    assert result.status == "completed" and result.result == "Buy milk."
    events = await stored_events(s)
    assert types_of(events) == [
        "turn.started",
        "session.title",
        "run.started",
        "step.started",
        "assistant.message",
        "tool.started",
        "tool.finished",
        "step.started",
        "assistant.message",
        "run.finished",
        "turn.finished",
    ]
    turn = await last_turn(s)
    assert turn.run_id == result.task_id
    assert (turn.status, turn.cause, turn.answer) == ("completed", "run", "Buy milk.")
    assert turn.result == result.to_dict()
    # on_event got every event of the run and the committed turn.finished,
    # each with its durable id; the store's turn.started reaches its listeners.
    assert [e.id for e in s.events] == [e.id for e in events[2:]]
    assert all(e.id is not None for e in s.events)
    check_invariants(s.path, "gateway")
    await s.aclose()


async def test_a_slow_on_event_is_cut_off_and_a_raising_one_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert SINK_TIMEOUT_S == 5.0
    monkeypatch.setattr(runner_mod, "SINK_TIMEOUT_S", 0.05)
    s = await setup(tmp_path, [answer("ok")])
    calls: list[str] = []

    async def slow(event: StoredEvent) -> None:
        calls.append(event.event["type"])
        if event.event["type"] == "run.started":
            await asyncio.sleep(10)
        if event.event["type"] == "step.started":
            raise RuntimeError("a broken callback")

    s.runner._on_event = slow
    caplog.set_level(logging.WARNING, logger="localmelo.support.sessions.runner")
    loop = asyncio.get_running_loop()
    began = loop.time()
    result = await asyncio.wait_for(s.runner.run_turn("hi"), timeout=5)
    assert loop.time() - began < 1.0  # cut at SINK_TIMEOUT_S, not after 10 s
    assert result.status == "completed"
    assert "run.finished" in calls and calls[-1] == "turn.finished"
    messages = [r.getMessage() for r in caplog.records]
    assert any("on_event took more than" in m for m in messages)
    assert any("on_event failed" in m for m in messages)
    await s.aclose()


async def test_a_secret_is_redacted_in_every_stored_copy_but_kept_in_the_result(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path, [answer(f"Your key is {SECRET}.")])
    result = await s.runner.run_turn("Make me a key")
    assert result.result == f"Your key is {SECRET}."
    redacted = "Your key is [redacted]."
    turn = await last_turn(s)
    assert turn.answer == redacted
    assert turn.result is not None and turn.result["result"] == redacted
    finished = [e for e in await stored_events(s) if e.event["type"] == "run.finished"]
    assert finished[0].event["data"]["result"]["result"] == redacted
    forwarded = [e for e in s.events if e.event["type"] == "run.finished"]
    assert forwarded[0].event["data"]["result"]["result"] == redacted
    window = await s.store.get_window(turn.session_id)
    assert window is not None and SECRET not in window.data
    assert decode_window(window.data)[-1].content == redacted  # type: ignore[index]
    assert SECRET not in Path(s.path).read_bytes().decode("utf-8", "replace")
    await s.aclose()


# ── Progress (UR-N) ─────────────────────────────────────────────────────────


async def test_a_runner_built_with_progress_stores_and_forwards_run_progress(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path, [answer("done")], progress=True)
    result = await s.runner.run_turn("hello")
    assert result.status == "completed"
    stored = types_of(await stored_events(s))
    # The first run.progress follows run.started (melo/agent/agent.py).
    assert stored[stored.index("run.started") + 1] == "run.progress"
    assert stored.count("run.progress") >= 2
    # on_event got every run.progress of the turn, once committed.
    assert types_of(s.events).count("run.progress") == stored.count("run.progress")
    check_invariants(s.path, "gateway")
    await s.aclose()


async def test_a_runner_without_progress_reports_none(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("done")])
    result = await s.runner.run_turn("hello")
    assert result.status == "completed"
    assert "run.progress" not in types_of(await stored_events(s))
    assert "run.progress" not in types_of(s.events)
    await s.aclose()


@pytest.mark.parametrize("progress", [True, False])
async def test_the_progress_flag_survives_replace_agent(
    tmp_path: Path, progress: bool
) -> None:
    s = await setup(tmp_path, progress=progress)
    other_llm = ScriptedLLM([answer("from the new agent")])
    other = await started_agent(tmp_path / "ws2", other_llm)
    s.runner.replace_agent(other)
    result = await s.runner.run_turn("hello")
    assert result.result == "from the new agent"
    stored = types_of(await stored_events(s))
    assert ("run.progress" in stored) is progress
    await other.close()
    await s.aclose()


async def test_run_turn_passes_routine_only_when_given(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("one"), answer("two")])
    original = s.agent.run_task
    with mock.patch.object(
        s.agent, "run_task", autospec=True, wraps=original
    ) as run_task:
        await s.runner.run_turn("hello")
        call = run_task.call_args
        assert call.args == ("hello",)
        assert "routine" not in call.kwargs
        assert call.kwargs["progress"] is False

        await s.runner.run_turn("hello again", routine="daily mail")
        call = run_task.call_args
        assert call.kwargs["routine"] == "daily mail"
    await s.aclose()


async def test_start_turn_passes_routine_only_when_given(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("one"), answer("two")], progress=True)
    original = s.agent.run_task
    with mock.patch.object(
        s.agent, "run_task", autospec=True, wraps=original
    ) as run_task:
        await s.runner.start_turn("hello")
        await s.runner.wait()
        call = run_task.call_args
        assert call.args == ("hello",)
        assert "routine" not in call.kwargs
        assert call.kwargs["progress"] is True

        await s.runner.start_turn("hello again", routine="daily mail")
        await s.runner.wait()
        call = run_task.call_args
        assert call.kwargs["routine"] == "daily mail"
    await s.aclose()


# ── Conversation ────────────────────────────────────────────────────────────


async def test_the_next_request_alternates_user_answer_user(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("Paris."), answer("About 2 million.")])
    await s.runner.run_turn("Capital of France?")
    await s.runner.run_turn("How many people live there?")
    second = s.llm.requests[-1]
    turns = [(m.role, m.content) for m in second if m.role in ("user", "assistant")]
    assert turns == [
        ("user", "Capital of France?"),
        ("assistant", "Paris."),
        ("user", "How many people live there?"),
    ]
    await s.aclose()


async def test_a_think_block_stays_in_the_row_and_leaves_the_window(
    tmp_path: Path,
) -> None:
    reply = "<think>\nThe user wants a city.\n</think>\n\nParis."
    s = await setup(tmp_path, [answer(reply)])
    result = await s.runner.run_turn("Capital of France?")
    assert result.result == reply
    turn = await last_turn(s)
    assert turn.answer == reply  # until retention clears it
    live = [(m.role, m.content) for m in s.agent.conversation()]
    assert live == [("user", "Capital of France?"), ("assistant", "Paris.")]
    window = await s.store.get_window(turn.session_id)
    assert window is not None
    decoded = decode_window(window.data)
    assert decoded is not None and decoded[-1].content == "Paris."
    for mode in ("pairs", "window"):
        other = await started_agent(tmp_path / "ws2", ScriptedLLM())
        resumed = SessionRunner(other, s.store, turn.session_id, resume_mode=mode)  # type: ignore[arg-type]
        assert await resumed.resume() == 1
        assert resumed.last_resume is not None
        assert resumed.last_resume.mode == mode
        assert [(m.role, m.content) for m in other.conversation()] == live
        await other.close()
    await s.aclose()


# ── Titles ──────────────────────────────────────────────────────────────────


async def test_the_typed_text_is_the_title(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("ok"), answer("ok")])
    composed = "Summarize this\n\n<file name='a.txt'>\nlong\n</file>"
    await s.runner.run_turn(composed, title="Summarize this")
    assert s.runner.session_id is not None
    info = await s.store.get_session(s.runner.session_id)
    assert info is not None and info.title == "Summarize this"
    other = SessionRunner(s.agent, s.store, None, new_session=NewSession())
    turn = await other.start_turn("x\n<file/>", title="Typed")
    await other.wait()
    info = await s.store.get_session(turn.session_id)
    assert info is not None and info.title == "Typed"
    assert rows(s.path, "SELECT typed_text FROM turns ORDER BY started_at") == [
        ("Summarize this",),
        ("Typed",),
    ]
    await other.close()
    await s.aclose()


# ── Causes ──────────────────────────────────────────────────────────────────


async def test_cancel_records_user_stop(tmp_path: Path) -> None:
    s = await setup(tmp_path)
    task = asyncio.create_task(s.runner.run_turn("wait"))
    await s.llm.blocked.wait()
    assert s.runner.cancel("Stopped by you", cause="user_stop")
    result = await task
    assert (result.status, result.error) == ("cancelled", "Stopped by you")
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "user_stop")
    # The window is closed and protocol-valid, and ends in INTERRUPT_LABEL.
    window = s.agent.conversation()
    assert window[-1].content == INTERRUPT_LABEL
    roles = [m.role for m in window]
    assert roles == ["user", "assistant"]  # the query, then the label
    check_invariants(s.path, "gateway")
    await s.aclose()


async def test_close_records_closed(tmp_path: Path) -> None:
    s = await setup(tmp_path)
    await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    await s.runner.close()
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "closed")
    assert turn.result is not None and turn.result["error"] == "Session closed"
    with pytest.raises(RuntimeError, match="closed"):
        await s.runner.start_turn("again")
    await s.agent.close()
    await s.store.aclose()


async def test_a_shutdown_after_begin_exit_records_shutdown_within_the_bound(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path)
    await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    loop = asyncio.get_running_loop()
    began = loop.time()
    s.agent.begin_exit("Session closed", within_s=3.0)
    await s.runner.close(reason="Session closed", cause="shutdown")
    assert loop.time() - began < 3.0
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "shutdown")
    assert turn.result is not None and turn.result["error"] == "Session closed"
    assert await s.store.get_window(turn.session_id) is not None
    await s.agent.close()
    await s.store.aclose()


async def test_a_stop_after_the_run_ended_keeps_its_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path, [answer("done")])
    stopped: list[bool] = []
    original = SessionStore.finish_turn

    async def stop_then_finish(
        self: SessionStore, *args: Any, **kwargs: Any
    ) -> TurnInfo:
        stopped.append(s.runner.cancel())  # the run has ended: nothing to stop
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "finish_turn", stop_then_finish)
    result = await s.runner.run_turn("hi")
    assert result.status == "completed" and stopped == [False]
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("completed", "run")
    await s.aclose()


async def test_a_crashed_run_has_cause_run(tmp_path: Path) -> None:
    class Broken(ScriptedLLM):
        async def chat(self, messages: Any, tools: Any = None) -> Any:
            raise RuntimeError("a bug in the model client")

    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    agent = await started_agent(tmp_path / "ws", Broken())
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    result = await runner.run_turn("hi")
    assert result.status == "crashed"
    turn = (await store.list_turns(runner.session_id))[-1]  # type: ignore[arg-type]
    assert (turn.status, turn.cause) == ("crashed", "run")
    await agent.close()
    await store.aclose()


async def test_cancel_takes_only_stopping_causes(tmp_path: Path) -> None:
    s = await setup(tmp_path)
    for cause in ("crash", "run", "bogus"):
        with pytest.raises(ValueError, match="cause must be one of"):
            s.runner.cancel(cause=cause)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await s.runner.close(cause="crash")  # type: ignore[arg-type]
    assert not s.runner.cancel()  # nothing runs
    await s.aclose()


async def test_an_outside_cancellation_records_shutdown_and_propagates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    s = await setup(tmp_path)
    task = asyncio.create_task(s.runner.run_turn("wait"))
    await s.llm.blocked.wait()
    caplog.set_level(logging.WARNING, logger="localmelo.support.sessions.runner")
    task.cancel()  # not through the runner: asyncio.run's shutdown, a TaskGroup
    with pytest.raises(asyncio.CancelledError):
        await task
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "shutdown")
    assert any(
        "cancelled from outside its runner" in r.getMessage() for r in caplog.records
    )
    assert not s.runner.busy
    await s.aclose()


async def test_run_turn_returns_the_cancelled_result_and_is_no_longer_cancelling(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path)
    after: list[int] = []

    async def turn() -> RunResult:
        result = await s.runner.run_turn("wait")
        current = asyncio.current_task()
        assert current is not None
        after.append(current.cancelling())
        return result

    task = asyncio.create_task(turn())
    await s.llm.blocked.wait()
    s.runner.cancel()
    result = await task
    assert result.status == "cancelled" and after == [0]
    # The very RunResult of run.finished.
    finished = [e for e in await stored_events(s) if e.event["type"] == "run.finished"]
    assert result.to_dict() == finished[-1].event["data"]["result"]
    await s.aclose()


async def test_a_stop_then_the_callers_timeout_gives_the_caller_its_timeout_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    gate = asyncio.Event()
    original = Agent._save_final_state

    async def slow_save(self: Agent, task: Any) -> None:
        await gate.wait()
        await original(self, task)

    monkeypatch.setattr(Agent, "_save_final_state", slow_save)
    scopes: list[asyncio.Timeout] = []
    after: list[int] = []

    async def caller() -> None:
        try:
            async with asyncio.timeout(None) as scope:
                scopes.append(scope)
                await s.runner.run_turn("wait")
        except TimeoutError:
            current = asyncio.current_task()
            assert current is not None
            after.append(current.cancelling())
            raise

    task = asyncio.create_task(caller())
    await s.llm.blocked.wait()
    assert s.runner.cancel("Stopped by you")
    await asyncio.sleep(0.05)  # the finalizer now waits in its save
    scopes[0].reschedule(asyncio.get_running_loop().time())
    await asyncio.sleep(0.1)  # the caller's deadline expires meanwhile
    gate.set()
    with pytest.raises(TimeoutError):
        await task
    assert after == [0]
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "user_stop")
    await s.aclose()


async def test_a_callers_timeout_while_step_7_stores_the_end_raises_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    reached, gate = asyncio.Event(), asyncio.Event()
    original = SessionStore.finish_turn

    async def slow_finish(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        reached.set()
        await gate.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "finish_turn", slow_finish)
    scopes: list[asyncio.Timeout] = []
    after: list[int] = []

    async def caller() -> None:
        try:
            async with asyncio.timeout(None) as scope:
                scopes.append(scope)
                await s.runner.run_turn("wait")
        except TimeoutError:
            current = asyncio.current_task()
            assert current is not None
            after.append(current.cancelling())
            raise

    task = asyncio.create_task(caller())
    await s.llm.blocked.wait()
    assert s.runner.cancel("Stopped by you")
    await reached.wait()  # step 7 waits in finish_turn
    scopes[0].reschedule(asyncio.get_running_loop().time())
    with pytest.raises(TimeoutError):
        await task  # at once, while the end is still being stored
    assert after == [0] and s.runner.busy
    gate.set()
    result = await s.runner.wait()  # step 7 went on alone
    assert result is not None and result.status == "cancelled"
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "user_stop")
    await s.aclose()


async def test_a_stop_right_after_begin_turn_returns_a_cancelled_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path, [answer("the second answer")])
    original = SessionRunner._begin_turn

    async def begin_then_stop(self: SessionRunner, *args: Any) -> Any:
        turn = await original(self, *args)
        if turn.turn.turn == 1:  # committed; run_turn's caller has not resumed
            assert self.cancel("Stopped at once")
        return turn

    monkeypatch.setattr(SessionRunner, "_begin_turn", begin_then_stop)
    after: list[int] = []

    async def caller() -> tuple[RunResult, RunResult]:
        first = await s.runner.run_turn("first")
        current = asyncio.current_task()
        assert current is not None
        after.append(current.cancelling())
        second = await s.runner.run_turn("second")
        after.append(current.cancelling())
        return first, second

    first, second = await asyncio.create_task(caller())
    assert (first.status, first.error) == ("cancelled", "Stopped at once")
    assert second.status == "completed" and after == [0, 0]
    turns = await s.store.list_turns(s.runner.session_id)  # type: ignore[arg-type]
    assert [(t.status, t.cause) for t in turns] == [
        ("cancelled", "user_stop"),
        ("completed", "run"),
    ]
    assert len(s.llm.requests) == 1  # the first turn never ran
    check_invariants(s.path, "gateway")
    await s.aclose()


async def test_close_while_run_turn_begins_returns_a_cancelled_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    gate = asyncio.Event()
    original = SessionStore.begin_turn

    async def slow_begin(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        await gate.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "begin_turn", slow_begin)
    after: list[int] = []

    async def caller() -> RunResult:
        result = await s.runner.run_turn("never runs")
        current = asyncio.current_task()
        assert current is not None
        after.append(current.cancelling())
        return result

    running = asyncio.create_task(caller())
    await asyncio.sleep(0.05)
    closing = asyncio.create_task(s.runner.close())
    await asyncio.sleep(0.05)
    gate.set()
    result = await running  # nothing outside the runner cancelled this task
    await closing
    assert (result.status, result.error) == ("cancelled", "Session closed")
    assert after == [0]
    stored = await last_turn(s)
    assert (stored.status, stored.cause) == ("cancelled", "closed")
    assert s.llm.requests == []
    await s.agent.close()
    await s.store.aclose()


async def test_a_cancelled_error_from_the_model_client_is_a_crash(
    tmp_path: Path,
) -> None:
    class Spurious(ScriptedLLM):
        async def chat(self, messages: Any, tools: Any = None) -> Any:
            raise asyncio.CancelledError  # while nothing cancels the task

    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    agent = await started_agent(tmp_path / "ws", Spurious())
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    after: list[int] = []

    async def caller() -> RunResult:
        result = await runner.run_turn("hi")
        current = asyncio.current_task()
        assert current is not None
        after.append(current.cancelling())
        return result

    result = await asyncio.create_task(caller())
    assert result.status == "crashed" and after == [0]
    await runner.start_turn("again")
    waited = await runner.wait()
    assert waited is not None and waited.status == "crashed"
    turns = await store.list_turns(runner.session_id)  # type: ignore[arg-type]
    assert [(t.status, t.cause) for t in turns] == [("crashed", "run")] * 2
    await agent.close()
    await store.aclose()


# ── Cancellation ────────────────────────────────────────────────────────────


async def test_start_turn_during_a_turn_is_busy_and_wait_returns_the_result(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path)
    first = await s.runner.start_turn("wait")
    assert s.runner.busy and s.runner.current_turn == first
    with pytest.raises(SessionBusyError, match="is already running a turn"):
        await s.runner.start_turn("second")
    with pytest.raises(SessionBusyError):
        await s.runner.run_turn("second")
    await s.llm.blocked.wait()
    assert s.runner.cancel()
    result = await s.runner.wait()
    assert result is not None and result.status == "cancelled"
    assert not s.runner.busy and s.runner.current_turn is None
    await s.aclose()


async def test_a_double_cancel_still_records_the_finalizers_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    gate = asyncio.Event()
    original = Agent._save_final_state

    async def slow_save(self: Agent, task: Any) -> None:
        await gate.wait()
        await original(self, task)

    monkeypatch.setattr(Agent, "_save_final_state", slow_save)
    await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    assert s.runner.cancel()
    await asyncio.sleep(0.05)
    assert s.runner.cancel()  # a second Stop while the finalizer saves
    await asyncio.sleep(0.05)
    gate.set()
    result = await s.runner.wait()
    assert result is not None and result.status == "cancelled"
    turn = await last_turn(s)
    assert turn.status == "cancelled" and turn.result is not None
    assert turn.result["task_id"] == result.task_id
    await s.aclose()


async def test_a_cancel_by_turn_id_never_stops_another_turn(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("one")])
    first = await s.runner.start_turn("first")
    await s.runner.wait()
    second = await s.runner.start_turn("second")
    await s.llm.blocked.wait()
    assert not s.runner.cancel(turn_id="not-this-turn")
    assert not s.runner.cancel(turn_id=first.id)  # a late Stop of the first turn
    assert s.runner.busy
    assert s.runner.cancel(turn_id=second.id)
    result = await s.runner.wait()
    assert result is not None and result.status == "cancelled"
    await s.aclose()


async def test_resume_right_after_a_cancel(tmp_path: Path) -> None:
    s = await setup(tmp_path)
    task = asyncio.create_task(s.runner.run_turn("wait"))
    await s.llm.blocked.wait()
    s.runner.cancel()
    with pytest.raises(SessionBusyError):  # the turn still ends: never RuntimeError
        await s.runner.resume()
    await task
    assert await s.runner.resume() == 1
    assert s.agent.conversation()[-1].content == INTERRUPT_LABEL
    await s.aclose()


async def test_a_finalizer_that_outlives_the_wait_leaves_the_runner_finishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "FINALIZE_TIMEOUT_S", 0.1)  # step 7.1 waits 1.1 s
    s = await setup(tmp_path, [])
    gate = asyncio.Event()
    original = Agent._finalize

    async def slow_finalize(self: Agent, *args: Any) -> None:
        await gate.wait()
        await original(self, *args)

    monkeypatch.setattr(Agent, "_finalize", slow_finalize)
    await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    s.runner.cancel()
    await asyncio.sleep(0.05)
    s.runner.cancel()  # run_task stops waiting for its finalizer
    await s.runner.wait()  # step 7 ended without run.finished
    assert s.runner.busy
    other = await started_agent(tmp_path / "ws2", ScriptedLLM())
    for call in (
        s.runner.start_turn("next"),
        s.runner.run_turn("next"),
        s.runner.resume(),
        s.runner.undo(),
    ):
        with pytest.raises(SessionBusyError, match="is still finishing its last turn"):
            await call
    with pytest.raises(SessionBusyError, match="still finishing"):
        s.runner.replace_agent(other)
    turn = await last_turn(s)
    assert (turn.status, turn.cause) == ("cancelled", "user_stop")
    assert turn.result is None  # it ended without run.finished
    gate.set()
    for _ in range(100):
        if not s.runner.busy:
            break
        await asyncio.sleep(0.02)
    assert not s.runner.busy
    s.llm.replies.append(answer("now"))
    assert (await s.runner.run_turn("next")).status == "completed"
    check_invariants(s.path, "gateway")
    await other.close()
    await s.aclose()


async def test_a_turn_cancelled_before_its_task_ran_still_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    gate = asyncio.Event()
    original = SessionStore.begin_turn

    async def slow_begin(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        await gate.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "begin_turn", slow_begin)
    starting = asyncio.create_task(s.runner.start_turn("never runs"))
    await asyncio.sleep(0.05)
    closing = asyncio.create_task(s.runner.close())
    await asyncio.sleep(0.05)
    gate.set()
    turn = await starting
    await closing
    waited = await s.runner.wait()  # what run_turn would return
    assert waited is not None
    assert (waited.status, waited.error) == ("cancelled", "Session closed")
    stored = (await s.store.list_turns(turn.session_id))[-1]
    assert (stored.status, stored.cause) == ("cancelled", "closed")
    assert stored.answer == INTERRUPT_LABEL and stored.result is None
    assert await s.store.get_window(turn.session_id) is None  # no run.finished
    assert s.llm.requests == []
    check_invariants(s.path, "gateway")
    await s.agent.close()
    await s.store.aclose()


# ── Takeover (CLI file) ─────────────────────────────────────────────────────


async def test_a_takeover_ends_the_old_turn_once_and_leaves_the_new_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "_LEASE_REFRESH_S", 0.02)
    clock = Clock()
    s = await setup(tmp_path, writer="cli", clock=clock)
    s.runner._holder = "otherhost:4242:aaaaaaaa"  # its lease_until decides
    turn = await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    before = rows(s.path, "SELECT lease_until FROM sessions")[0][0]
    clock.advance(1)
    await asyncio.sleep(0.1)
    assert rows(s.path, "SELECT lease_until FROM sessions")[0][0] > before
    # The process sleeps in a refresh while its lease lapses.
    original = s.store.refresh_lease
    awake = asyncio.Event()

    async def asleep(session_id: str, *, holder: str) -> bool:
        await awake.wait()
        return await original(session_id, holder=holder)

    monkeypatch.setattr(s.store, "refresh_lease", asleep)
    await asyncio.sleep(0.05)
    clock.advance(LEASE_S + 1)
    other = SessionStore(s.path, writer="cli", clock=clock)
    new = await other.begin_turn(turn.session_id, "mine", holder="newhost:1:bbbbbbbb")
    assert new.turn == 2
    awake.set()  # the next refresh says the lease is gone
    result = await asyncio.wait_for(s.runner.wait(), timeout=10)
    assert result is not None and result.status == "cancelled"
    assert result.error == TAKEOVER_REASON
    events = await other.list_events(turn.session_id)
    old = [e for e in events if e.turn == 1]
    assert [e.event["type"] for e in old].count("turn.finished") == 1
    # Recovery under the takeover: the expired lease was another host's.
    first = (await other.list_turns(turn.session_id))[0]
    assert (first.status, first.cause) == ("cancelled", "takeover")
    assert old[-1].event["data"]["error"] == TAKEOVER_REASON
    started = next(
        e for e in events if e.event["type"] == "turn.started" and e.turn == 2
    )
    assert all((e.id or 0) < (started.id or 0) for e in old)
    assert rows(s.path, "SELECT lease_holder FROM sessions") == [
        ("newhost:1:bbbbbbbb",)
    ]
    check_invariants(s.path, "cli")
    await other.aclose()
    await s.aclose()


async def test_the_lease_task_keeps_a_turn_live_while_its_run_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "_LEASE_REFRESH_S", 0.02)
    clock = Clock()
    s = await setup(tmp_path, writer="cli", clock=clock)
    s.runner._holder = "otherhost:4242:aaaaaaaa"  # its lease_until decides
    turn = await s.runner.start_turn("wait")
    await s.llm.blocked.wait()  # the run waits for the model, for ever
    other = SessionStore(s.path, writer="cli", clock=clock)
    for _ in range(9):  # 3 × LEASE_S on the clock
        clock.advance(LEASE_S / 3)
        await asyncio.sleep(0.06)  # the lease task's ticks
        info = await other.get_session(turn.session_id)
        assert info is not None and info.busy
        assert await other.recover_stale_turns() == 0
    assert s.runner.busy
    s.runner.cancel()
    await s.runner.wait()
    await other.aclose()
    await s.aclose()


async def test_a_failing_refresh_cancels_nothing_and_a_recovery_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "_LEASE_REFRESH_S", 0.02)
    clock = Clock()
    s = await setup(tmp_path, writer="cli", clock=clock)
    s.runner._holder = "otherhost:4242:aaaaaaaa"
    original = s.store.refresh_lease
    failures = 0
    awake = asyncio.Event()

    async def flaky(session_id: str, *, holder: str) -> bool:
        nonlocal failures
        if failures < 3:
            failures += 1
            raise sqlite3.OperationalError("database is locked")
        await awake.wait()
        return await original(session_id, holder=holder)

    monkeypatch.setattr(s.store, "refresh_lease", flaky)
    turn = await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    await asyncio.sleep(0.2)
    assert failures == 3 and s.runner.busy  # a failed refresh cancels nothing
    # Another terminal's recovery ends the turn once its lease lapses.
    other = SessionStore(s.path, writer="cli", clock=clock)
    await other.open()  # the lease is live yet: nothing to recover
    clock.advance(LEASE_S + 1)
    assert await other.recover_stale_turns(turn.session_id) == 1
    awake.set()
    result = await asyncio.wait_for(s.runner.wait(), timeout=10)
    assert result is not None and result.error == TAKEOVER_REASON
    stored = (await other.list_turns(turn.session_id))[-1]
    # Recovery's end: the expired lease was another host's (revision 48).
    assert (stored.status, stored.cause) == ("cancelled", "takeover")
    assert result.status == "cancelled"  # the old runner ends the same way
    check_invariants(s.path, "cli")
    await other.aclose()
    await s.aclose()


# ── Keys ────────────────────────────────────────────────────────────────────


async def test_a_repeated_key_returns_the_stored_turn_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    first = await s.runner.start_turn("hello", client_request_id="k1")
    await s.llm.blocked.wait()
    with pytest.raises(SubmissionRepeated) as repeated:
        await s.runner.start_turn("hello", client_request_id="k1")
    assert repeated.value.turn.id == first.id
    await asyncio.sleep(0.05)  # time for a task the repeat might have started
    assert len(s.llm.requests) == 1
    assert s.runner.current_turn == first
    assert rows(s.path, "SELECT COUNT(*) FROM turns") == [(1,)]
    calls: list[Any] = []
    with monkeypatch.context() as m:
        m.setattr(SessionStore, "begin_turn", lambda *a, **k: calls.append(a))
        with pytest.raises(SubmissionConflict):
            await s.runner.start_turn("other text", client_request_id="k1")
        assert calls == []  # refused at step 0, before begin_turn
    assert (
        "client_request_id" not in inspect.signature(SessionRunner.run_turn).parameters
    )
    s.runner.cancel()
    await s.runner.wait()
    with pytest.raises(SubmissionRepeated):  # also once the turn has ended
        await s.runner.start_turn("hello", client_request_id="k1")
    await asyncio.sleep(0.05)
    assert not s.runner.busy and len(s.llm.requests) == 1
    await s.aclose()


async def test_a_repeat_while_its_own_turn_begins_gets_that_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    gate = asyncio.Event()
    original = SessionStore.begin_turn

    async def slow_begin(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        await gate.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "begin_turn", slow_begin)
    first = asyncio.create_task(s.runner.start_turn("hello", client_request_id="k1"))
    await asyncio.sleep(0.05)
    # The repeat finds no submission yet: the first begin_turn still runs.
    second = asyncio.create_task(s.runner.start_turn("hello", client_request_id="k1"))
    await asyncio.sleep(0.05)
    gate.set()
    turn = await first
    with pytest.raises(SubmissionRepeated) as repeated:
        await second
    assert repeated.value.turn.id == turn.id
    assert rows(s.path, "SELECT COUNT(*) FROM turns") == [(1,)]
    await s.llm.blocked.wait()
    s.runner.cancel()
    await s.runner.wait()
    await s.aclose()


async def test_a_first_turn_refused_as_busy_leaves_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path)
    await s.store.open()
    gate = asyncio.Event()
    original = SessionStore.begin_turn

    async def slow_begin(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        await gate.wait()
        return await original(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "begin_turn", slow_begin)
        first = asyncio.create_task(s.runner.start_turn("one"))
        await asyncio.sleep(0.05)
        with pytest.raises(SessionBusyError, match="is already running a turn"):
            await s.runner.start_turn("two")
        assert s.runner.session_id is None
        assert rows(s.path, "SELECT COUNT(*) FROM sessions") == [(0,)]
        gate.set()
        turn = await first
    assert rows(s.path, "SELECT COUNT(*) FROM sessions") == [(1,)]
    assert s.runner.session_id == turn.session_id
    await s.llm.blocked.wait()
    s.runner.cancel()
    await s.runner.wait()
    await s.aclose()


async def test_a_refused_first_turn_leaves_no_session(tmp_path: Path) -> None:
    s = await setup(tmp_path, [answer("ok"), answer("mine")])
    with pytest.raises(ValueError, match="text must not be blank"):
        await s.runner.start_turn("   ")
    await s.store.withdraw_submission(None, "w1", user_text="draft")
    with pytest.raises(SubmissionWithdrawn):
        await s.runner.start_turn("draft", client_request_id="w1")
    other = SessionRunner(s.agent, s.store, None, new_session=NewSession())
    accepted = await other.start_turn("hello", client_request_id="k1")
    await other.wait()
    with pytest.raises(SubmissionRepeated) as repeated:
        await s.runner.start_turn("hello", client_request_id="k1")
    assert repeated.value.turn.id == accepted.id
    assert s.runner.session_id is None
    assert rows(s.path, "SELECT COUNT(*) FROM sessions") == [(1,)]
    turn = await s.runner.start_turn("now mine")
    assert s.runner.session_id == turn.session_id != accepted.session_id
    await s.runner.wait()
    await other.close()
    await s.aclose()


# ── observe ─────────────────────────────────────────────────────────────────


async def test_observe_stores_the_request_and_the_answer(tmp_path: Path) -> None:
    s = await setup(
        tmp_path,
        [
            tool_call("file_write", {"path": "a.txt", "content": "x"}),
            answer("written"),
            tool_call("file_write", {"path": "b.txt", "content": "y"}, "call_2"),
        ],
    )
    seen_before_answer: list[list[str]] = []
    waiting = asyncio.Event()

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        if seen_before_answer:  # the second request waits for ever
            waiting.set()
            await asyncio.Event().wait()
        assert s.runner.session_id is not None
        await s.runner.flush()  # what a client sees once this task yields
        stored = await s.store.list_events(s.runner.session_id)
        seen_before_answer.append(types_of(stored))
        return ApprovalDecision(True, remember=False)

    policy = RememberingPolicy(s.runner.observe(ask))
    assert (await s.runner.run_turn("write", approval=policy)).status == "completed"
    stored_types = seen_before_answer[0]
    assert stored_types.index("tool.started") < stored_types.index("approval.requested")
    assert "approval.resolved" not in stored_types
    events = await stored_events(s)
    resolved = [
        e.event["data"] for e in events if e.event["type"] == "approval.resolved"
    ]
    assert resolved[0]["outcome"] == "allowed" and resolved[0]["origin"] == "user"
    # After replace_agent the same policy, unchanged, still stores them; a
    # cancel withdraws.
    other_llm = ScriptedLLM(
        [tool_call("file_write", {"path": "c.txt", "content": "z"})]
    )
    other = await started_agent(tmp_path / "ws", other_llm)
    s.runner.replace_agent(other)
    task = asyncio.create_task(s.runner.run_turn("write more", approval=policy))
    await waiting.wait()
    s.runner.cancel()
    assert (await task).status == "cancelled"
    events = await stored_events(s)
    last = [
        e.event
        for e in events
        if e.turn == 2 and e.event["type"].startswith("approval.")
    ]
    assert [e["type"] for e in last] == ["approval.requested", "approval.resolved"]
    assert last[-1]["data"]["outcome"] == last[-1]["data"]["origin"] == "withdrawn"
    check_invariants(s.path, "gateway")
    await other.close()
    await s.aclose()


async def test_observe_outside_a_turn_only_asks(tmp_path: Path) -> None:
    s = await setup(tmp_path)
    asked: list[str] = []

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.tool_name)
        return ApprovalDecision(False, "no")

    result = await s.agent.run_tool(
        "file_write",
        {"path": "a.txt", "content": "x"},
        approval=RememberingPolicy(s.runner.observe(ask)),
    )
    assert asked == ["file_write"] and result.error
    assert s.runner.session_id is None
    await s.aclose()


# ── publish ─────────────────────────────────────────────────────────────────


async def test_publish(tmp_path: Path) -> None:
    s = await setup(tmp_path)
    with pytest.raises(RuntimeError, match="No turn is running"):
        s.runner.publish("sched.held", {"why": "gpu"})
    with pytest.raises(ValueError):
        s.runner.publish("foo.bar", {})
    await s.runner.start_turn("wait")
    await s.llm.blocked.wait()
    s.runner.publish(
        "approval.requested",
        {
            "request_id": "r1",
            "call_id": "c1",
            "tool_name": "shell_exec",
            "arguments": {"x": float("nan"), "long": "a" * 150_000},
            "preview": "p",
            "workspace_root": None,
        },
    )
    s.runner.publish("sched.held", {"why": "gpu"})
    with pytest.raises(ValueError):
        s.runner.publish("foo.bar", {})
    with pytest.raises(ValueError):
        s.runner.publish("approval.resolved", {"request_id": "r1"})
    await s.runner.flush()
    events = {e.event["type"]: e.event for e in await stored_events(s)}
    arguments = events["approval.requested"]["data"]["arguments"]
    assert arguments["x"] == "nan"
    assert len(arguments["long"]) < 150_000 and arguments["long"].endswith("chars]")
    assert events["sched.held"]["data"] == {"why": "gpu"}
    s.runner.cancel()
    await s.runner.wait()
    await s.aclose()
    # A CLI store holds no scheduler events.
    c = await setup(tmp_path / "cli", writer="cli")
    await c.runner.start_turn("wait")
    await c.llm.blocked.wait()
    with pytest.raises(ValueError, match="gateway"):
        c.runner.publish("sched.held", {})
    c.runner.cancel()
    await c.runner.wait()
    await c.aclose()


# ── Checkpoints ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["pairs", "window"])
async def test_every_turn_end_with_run_finished_writes_a_checkpoint(
    tmp_path: Path, mode: str
) -> None:
    s = await setup(tmp_path, [answer("done")])
    s.runner._resume_mode = mode  # type: ignore[assignment]
    await s.runner.run_turn("first")
    assert s.runner.session_id is not None
    window = await s.store.get_window(s.runner.session_id)
    assert window is not None and window.turn == 1
    await s.runner.start_turn("second")
    await s.llm.blocked.wait()
    s.runner.cancel()
    await s.runner.wait()
    window = await s.store.get_window(s.runner.session_id)
    assert window is not None and window.turn == 2
    assert decode_window(window.data)[-1].content == INTERRUPT_LABEL  # type: ignore[index]
    await s.aclose()


async def test_a_turn_without_run_finished_writes_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path, [answer("done")])
    await s.runner.run_turn("first")

    async def broken(self: Agent, *args: Any, **kwargs: Any) -> RunResult:
        raise RuntimeError("run_task failed before its first event")

    monkeypatch.setattr(Agent, "run_task", broken)
    with pytest.raises(RuntimeError):
        await s.runner.run_turn("second")
    turn = await last_turn(s)
    assert (turn.status, turn.cause, turn.result) == ("crashed", "run", None)
    window = await s.store.get_window(turn.session_id)
    assert window is not None and window.turn == 1
    check_invariants(s.path, "gateway")
    await s.aclose()


# ── Write failures ──────────────────────────────────────────────────────────


async def test_a_gateway_finish_is_tried_once_more_then_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner_mod.GATEWAY_FINISH_RETRY_DELAYS_S == (1.0,)
    s = await setup(tmp_path, [answer("done")])
    calls = 0

    async def failing(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "finish_turn", failing)
        loop = asyncio.get_running_loop()
        began = loop.time()
        assert (await s.runner.run_turn("hi")).status == "completed"
        assert calls == 2 and loop.time() - began >= 1.0
        assert (await last_turn(s)).status == "running"
        # No later try in the gateway's file: the next process's open ends it.
        with pytest.raises(SessionBusyError):
            await s.runner.run_turn("next")
        assert calls == 2
    await s.aclose()


@pytest.mark.parametrize("closing", [False, True])
async def test_a_cli_finish_is_retried_unless_closing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, closing: bool
) -> None:
    assert CLI_FINISH_RETRY_DELAYS_S == (0.5, 1.0, 2.0)
    delays = (0.05, 0.1, 0.2)  # shorter, and each one its own
    monkeypatch.setattr(runner_mod, "CLI_FINISH_RETRY_DELAYS_S", delays)
    s = await setup(tmp_path, [] if closing else [answer("done")], writer="cli")
    calls = 0
    times: list[float] = []

    async def failing(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        nonlocal calls
        calls += 1
        times.append(asyncio.get_running_loop().time())
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "finish_turn", failing)
        if closing:
            await s.runner.start_turn("wait")
            await s.llm.blocked.wait()
            await s.runner.close()
            assert calls == 1  # one try, so the close keeps its bound
        else:
            await s.runner.run_turn("hi")
            assert calls == 4
            gaps = [
                later - earlier
                for earlier, later in zip(times, times[1:], strict=False)
            ]
            assert all(gap >= delay for gap, delay in zip(gaps, delays, strict=True))
        # The turn is left to the lease rule.
        assert (await last_turn(s)).status == "running"
    await s.agent.close()
    await s.store.aclose()


async def test_an_unrecorded_cli_end_is_retried_before_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "CLI_FINISH_RETRY_DELAYS_S", (0.01, 0.01, 0.01))
    s = await setup(tmp_path, [answer("one"), answer("two")], writer="cli")
    original = SessionStore.finish_turn

    async def failing(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SessionStore, "finish_turn", failing)
    assert (await s.runner.run_turn("first")).status == "completed"
    # This process lives, so the lease rule never frees its turn.
    assert (await last_turn(s)).status == "running"
    monkeypatch.setattr(SessionStore, "finish_turn", original)
    assert (await s.runner.run_turn("second")).status == "completed"
    turns = await s.store.list_turns(s.runner.session_id)  # type: ignore[arg-type]
    assert [(t.turn, t.status, t.cause) for t in turns] == [
        (1, "completed", "run"),
        (2, "completed", "run"),
    ]
    assert turns[0].answer == "one"
    check_invariants(s.path, "cli")
    await s.aclose()


async def test_a_cli_end_lost_at_a_close_is_stored_by_the_next_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path, [], writer="cli")

    async def failing(self: SessionStore, *args: Any, **kwargs: Any) -> TurnInfo:
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "finish_turn", failing)
        await s.runner.start_turn("wait")
        await s.llm.blocked.wait()
        await s.runner.close()  # one try only, which fails
    assert (await last_turn(s)).status == "running"
    # Another runner of this process for the session stores the lost end
    # before its first turn.
    s.llm.replies.append(answer("now"))
    runner = SessionRunner(s.agent, s.store, s.runner.session_id)
    assert (await runner.run_turn("next")).status == "completed"
    turns = await s.store.list_turns(s.runner.session_id)  # type: ignore[arg-type]
    assert [(t.status, t.cause) for t in turns] == [
        ("cancelled", "closed"),
        ("completed", "run"),
    ]
    check_invariants(s.path, "cli")
    await runner.close()
    await s.aclose()


# ── Undo ────────────────────────────────────────────────────────────────────


async def test_undo_marks_the_turn_resumes_and_drops_its_checkpoint(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path, [answer("one"), answer("two")])
    await s.runner.run_turn("first")
    await s.runner.run_turn("second")
    undone = await s.runner.undo()
    assert undone is not None and undone.turn == 2 and undone.undone
    assert s.runner.session_id is not None
    assert await s.store.get_window(s.runner.session_id) is None
    assert [(m.role, m.content) for m in s.agent.conversation()] == [
        ("user", "first"),
        ("assistant", "one"),
    ]
    assert s.runner.last_resume is not None and s.runner.last_resume.turns == 1
    await s.aclose()


async def test_undo_and_a_new_turn_at_once_never_show_the_undone_turn(
    tmp_path: Path,
) -> None:
    s = await setup(tmp_path, [answer("AKITA answer"), answer("second answer")])
    await s.runner.run_turn("AKITA question")
    # No delay is injected: the undo and the turn race as two requests would.
    undone, started = await asyncio.gather(
        s.runner.undo(), s.runner.start_turn("second question")
    )
    assert undone is not None and undone.turn == 1 and started.turn == 2
    result = await s.runner.wait()
    assert result is not None and result.status == "completed"
    assert not any("AKITA" in m.content for m in s.llm.requests[-1])
    window = await s.store.get_window(started.session_id)
    assert window is not None and window.turn == 2
    decoded = decode_window(window.data)
    assert decoded is not None
    assert not any("AKITA" in m.content for m in decoded)
    await s.aclose()


async def test_an_undo_cancelled_after_its_commit_still_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path, [answer("AKITA answer"), answer("second answer")])
    await s.runner.run_turn("AKITA question")
    reached, gate = asyncio.Event(), asyncio.Event()
    original = SessionStore.list_turns

    async def slow_list(self: SessionStore, *args: Any, **kwargs: Any) -> Any:
        reached.set()  # undo_last_turn has committed; the resume reads turns
        await gate.wait()
        return await original(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "list_turns", slow_list)
        undoing = asyncio.create_task(s.runner.undo())
        await reached.wait()
        undoing.cancel()  # the request that asked for the undo went away
        await asyncio.sleep(0.02)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await undoing
    assert s.agent.conversation() == []  # the resume ran to its end
    assert (await s.runner.run_turn("second question")).status == "completed"
    assert not any("AKITA" in m.content for m in s.llm.requests[-1])
    await s.aclose()


async def test_an_undo_whose_resume_fails_leaves_no_undone_turn_in_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await setup(tmp_path, [answer("AKITA answer"), answer("second answer")])
    await s.runner.run_turn("AKITA question")

    async def failing(self: SessionStore, *args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "list_turns", failing)
        with pytest.raises(sqlite3.OperationalError):
            await s.runner.undo()
    assert (await last_turn(s)).undone  # the undo is stored
    assert s.agent.conversation() == []
    assert (await s.runner.run_turn("second question")).status == "completed"
    assert not any("AKITA" in m.content for m in s.llm.requests[-1])
    await s.aclose()


def test_the_lease_is_refreshed_every_third_of_it() -> None:
    assert runner_mod._LEASE_REFRESH_S == LEASE_S / 3 == 10.0


async def test_a_compress_turn_stores_its_digest_in_turns_note(tmp_path: Path) -> None:
    """R4 writes the turn's final attempt digest through finish_turn."""
    from localmelo.melo.schema import Message, ToolCall

    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer="gateway", clock=Clock())  # type: ignore[arg-type]
    llm = ScriptedLLM(
        [
            Message(
                role="assistant",
                content="<note>plan: read notes.txt</note>",
                tool_calls=[
                    ToolCall(
                        tool_name="file_read",
                        arguments={"path": "notes.txt"},
                        id="call_1",
                    )
                ],
            ),
            answer("The test passed because the notes say buy milk."),
        ]
    )
    agent = await started_agent(tmp_path / "ws", llm, reasoning="compress")
    s = Setup(path, store, llm, agent)
    try:
        result = await s.runner.run_turn("What do my notes say?")
        assert result.status == "completed"
        with sqlite3.connect(path) as conn:
            note = conn.execute(
                "SELECT note FROM turns WHERE session_id = ? AND turn = 1",
                (s.runner.session_id,),
            ).fetchone()[0]
        assert note.startswith("task: What do my notes say?")
        assert "outcome: completed" in note
        events = await stored_events(s)
        notes = [
            event
            for event in events
            if event.event["type"] == "note" and event.event["data"]["kind"] == "digest"
        ]
        assert len(notes) == 1
        assert notes[0].event["data"]["text"] == note
        # resume() still loads the conversation, not the note.
        assert await s.runner.resume() is not None
    finally:
        await s.aclose()


async def test_turn_finished_reaches_on_event_after_the_run(tmp_path: Path) -> None:
    """The committed turn.finished, with its cause, follows run.finished."""
    s = await setup(tmp_path, [answer("done")])
    try:
        await s.runner.run_turn("q")
        types = types_of(s.events)
        assert types[-2:] == ["run.finished", "turn.finished"], types
        forwarded = s.events[-1]
        assert forwarded.id is not None  # the stored event, not a copy
        data = forwarded.event["data"]
        assert (data["status"], data["cause"]) == ("completed", "run")
    finally:
        await s.aclose()
