"""Details of a run's progress and learning that the review's mutants found unpinned.

Each test pins one thing that a plausible bug would change and that no
other test caught: where the lookup sits in the run's timeout, what the
finalizer's ended event says, what the episode records (attempts, a named
routine's text, the workspace root, the model id), that a stuck notice
and a reflection mark the trace, that a run computes no progress without
a sink, and the emitter's own flags.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import agent as agent_module
from localmelo.melo.agent._events import _Emitter
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.procedures import tracker
from localmelo.melo.memory.procedures.store import MemoryProcedures, Procedures
from localmelo.melo.memory.procedures.template import TemplateView
from localmelo.melo.memory.procedures.tracker import ProgressState
from localmelo.tests.agent.procedural_helpers import (
    QUERY,
    Recorder,
    ScriptedLLM,
    Tools,
    answer,
    call,
    fix_run,
    make_agent,
    passing_tools,
    read,
    remember,
)

REFLECT_CONTINUE = (
    '{"recommended_action": "continue", "task_still_feasible": true, '
    '"next_step_is_concrete": true, "next_step_is_novel": true}'
)


async def rows(store: Procedures, table: str) -> list[dict[str, Any]]:
    return [row for row in await store.export_rows() if row["table"] == table]


def progress(recorder: Recorder) -> list[dict[str, Any]]:
    return [dict(event.data) for event in recorder.of("run.progress")]


class HangingLookup(MemoryProcedures):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def lookup(self, *args: Any, **kwargs: Any) -> TemplateView | None:
        self.entered.set()
        await asyncio.Event().wait()
        return None


async def test_the_lookup_counts_against_the_run_s_timeout() -> None:
    agent = make_agent(ScriptedLLM([answer("done")]), timeout_s=0.1)
    agent.hippo.procedures = HangingLookup()
    started = time.perf_counter()
    result = await asyncio.wait_for(
        agent.run_task(QUERY, on_event=Recorder(), progress=True), timeout=5
    )
    assert result.status == "timeout"
    assert time.perf_counter() - started < agent_module.LOOKUP_TIMEOUT_S


async def test_a_cancellation_during_the_lookup_propagates() -> None:
    store = HangingLookup()
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    running = asyncio.create_task(
        agent.run_task(QUERY, observe=Recorder(), progress=True)
    )
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_a_cancelled_run_s_ended_event_leaves_its_current_item_open() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(
        ScriptedLLM([read("/p/test_parser.py"), call("hang", "call_h")]), tools=tools
    )
    await remember(agent, 4)
    running = asyncio.create_task(
        agent.run_task(QUERY, on_event=recorder, progress=True)
    )
    await asyncio.wait_for(tools.hanging.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    ended = progress(recorder)[-1]
    assert (ended["level"], ended["state"]) == (1, "ended")
    assert [i["state"] for i in ended["items"]] == ["current", "pending", "pending"]


async def test_learning_gets_the_lookup_s_view() -> None:
    class ViewSpy(MemoryProcedures):
        def __init__(self) -> None:
            super().__init__()
            self.views: list[TemplateView | None] = []

        async def learn(self, facts: Any, *, view: TemplateView | None = None) -> Any:
            self.views.append(view)
            return await super().learn(facts, view=view)

    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    store = ViewSpy()
    agent.hippo.procedures = store
    await remember(agent, 4, 4)
    store.views.clear()
    await agent.run_task(QUERY, on_event=Recorder(), progress=True)
    (view,) = store.views
    assert view is not None and view.level == 2


async def test_the_episode_records_the_run_s_attempts() -> None:
    script = [read("/a", f"call_{i}") for i in range(3)]
    script += [answer(REFLECT_CONTINUE), answer("done")]
    agent = make_agent(ScriptedLLM(script), steps_per_attempt=3)
    result = await agent.run_task(QUERY)
    assert result.attempts == 2
    (episode,) = await rows(agent.hippo.procedures, "episodes")
    assert episode["attempts"] == 2


async def test_a_run_whose_folder_cannot_be_read_completes_unlearned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def gone(root: str | None) -> str:
        raise FileNotFoundError("the working directory was removed")

    monkeypatch.setattr(agent_module, "run_place", gone)
    caplog.set_level(logging.DEBUG)
    agent = make_agent(ScriptedLLM([answer("done")]))
    recorder = Recorder()
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "completed"
    assert await rows(agent.hippo.procedures, "episodes") == []
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert {s["level"] for s in progress(recorder)} == {0}


async def test_a_named_routine_keeps_its_query_text() -> None:
    agent = make_agent(ScriptedLLM([answer("done")]))
    await agent.run_task("Fix it on 2026-10-26", routine="daily-fix")
    (routine,) = await rows(agent.hippo.procedures, "routines")
    assert routine["query_text"] == "fix it on <date>"


async def test_the_place_is_the_workspace_root(tmp_path: Path) -> None:
    agent = make_agent(ScriptedLLM([answer("done")]), workspace_root=str(tmp_path))
    await agent.run_task(QUERY)
    (routine,) = await rows(agent.hippo.procedures, "routines")
    assert routine["place"] == os.path.realpath(tmp_path)
    assert routine["place"] != os.path.realpath(os.getcwd())


async def test_a_model_that_is_not_a_str_is_unknown() -> None:
    llm = ScriptedLLM([answer("done")])
    llm.model = None  # type: ignore[assignment]
    agent = make_agent(llm)
    await agent.run_task(QUERY)
    (episode,) = await rows(agent.hippo.procedures, "episodes")
    assert episode["model"] == ""


def test_hippo_close_closes_the_store() -> None:
    class SyncClosing(MemoryProcedures):
        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    store = SyncClosing()
    Hippo(procedures=store).close()
    assert store.closed == 1


async def test_a_progress_payload_is_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def odd(*args: Any, **kwargs: Any) -> ProgressState:
        return ProgressState(fraction=float("nan"))

    monkeypatch.setattr(agent_module, "progress_state", odd)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([answer("done")]))
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    for event in recorder.of("run.progress"):
        assert event.data["fraction"] == "nan"
        json.dumps(dict(event.data), allow_nan=False)


def test_the_emitter_s_flags() -> None:
    recorder = Recorder()
    emitter = _Emitter(recorder, "r")
    assert emitter.interactive and not emitter.reports_progress
    emitter.progress(ProgressState())
    assert recorder.events == []


async def test_a_trace_failure_has_no_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from localmelo.melo.memory.procedures.trace import Trace

    def broken(self: Trace, *args: Any) -> None:
        raise RuntimeError("tracker bug")

    monkeypatch.setattr(Trace, "tool", broken)
    caplog.set_level(logging.DEBUG)
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    await agent.run_task(QUERY, on_event=Recorder(), progress=True)
    assert all(r.exc_info is None for r in caplog.records)


async def test_level_2_goes_off_template_when_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", True)
    recorder = Recorder()
    script = [read("/a", f"call_{i}") for i in range(3)]
    script += [answer(REFLECT_CONTINUE), answer("done")]
    agent = make_agent(ScriptedLLM(script), steps_per_attempt=3)
    await remember(agent, 8, 8)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    stuck = next(
        i
        for i, e in enumerate(recorder.events)
        if e.type == "notice" and e.data["kind"] == "stuck"
    )
    after = recorder.events[stuck + 1].data
    assert (after["level"], after["state"], after["off_reason"]) == (
        2,
        "off_template",
        "stuck",
    )


async def test_level_2_goes_off_template_on_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", True)
    recorder = Recorder()
    script = [read("/a", "c1"), read("/b", "c2"), answer(REFLECT_CONTINUE)]
    script += [answer("done")]
    agent = make_agent(ScriptedLLM(script), steps_per_attempt=2)
    await remember(agent, 8, 8)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    i = recorder.types.index("reflection.started")
    after = recorder.events[i + 1].data
    assert (after["level"], after["state"], after["off_reason"]) == (
        2,
        "off_template",
        "retrying",
    )


async def test_no_progress_is_computed_without_a_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def counting(*args: Any, **kwargs: Any) -> ProgressState:
        calls.append(args)
        return ProgressState()

    monkeypatch.setattr(agent_module, "progress_state", counting)
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    assert (await agent.run_task(QUERY, progress=True)).status == "completed"
    assert calls == []


async def test_a_slow_lookup_s_line_names_the_bound(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(agent_module, "LOOKUP_TIMEOUT_S", 0.05)
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = HangingLookup()
    result = await agent.run_task(QUERY, on_event=Recorder(), progress=True)
    (line,) = [r for r in caplog.records if "progress template" in r.getMessage()]
    assert line.levelno == logging.INFO and line.exc_info is None
    assert "no answer within 0.05 s" in line.getMessage()
    assert result.task_id in line.getMessage()


async def test_a_secret_in_a_test_call_s_tail_never_reaches_progress() -> None:
    secret = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    tools = passing_tools()
    tools.shell["pytest -q"] = f"token {secret}\n8 passed in 0.12s"
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run()), tools=tools)
    await remember(agent, 4)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    states = progress(recorder)
    assert states[-1]["level"] == 1
    assert states[-1]["tests"] == {"passed": 8, "failed": 0, "total": 8}
    assert all(secret not in json.dumps(s) for s in states)
    assert secret not in json.dumps(await agent.hippo.procedures.export_rows())


async def test_two_learns_that_overlap_leave_one_active_version(
    tmp_path: Path,
) -> None:
    from localmelo.melo.memory.procedures.store import SqliteProcedures

    barrier = asyncio.Barrier(2)

    class Meeting(SqliteProcedures):
        async def learn(self, facts: Any, *, view: Any = None) -> Any:
            await barrier.wait()  # both runs' ends are under way together
            return await super().learn(facts, view=view)

    agents = []
    for _ in range(2):
        agent = make_agent(ScriptedLLM([answer("done")]), memory_dir=str(tmp_path))
        await agent.hippo.procedures.aclose()
        agent.hippo.procedures = Meeting(tmp_path / "procedures.db")
        agents.append(agent)
    try:
        await asyncio.wait_for(
            asyncio.gather(*(a.run_task(QUERY) for a in agents)), timeout=10
        )
        versions = await rows(agents[0].hippo.procedures, "templates")
        episodes = await rows(agents[0].hippo.procedures, "episodes")
    finally:
        for agent in agents:
            await agent.close()
    assert len(episodes) == 2
    assert sorted((v["version"], v["status"]) for v in versions) == [
        (1, "superseded"),
        (2, "active"),
    ]


async def test_an_observed_run_that_is_cancelled_ends_its_progress() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools)
    running = asyncio.create_task(
        agent.run_task(QUERY, observe=recorder, progress=True)
    )
    await asyncio.wait_for(tools.hanging.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    # The stopped call is neither traced nor followed by progress of its own.
    assert recorder.types[-4:] == [
        "tool.started",
        "tool.finished",
        "run.progress",
        "run.finished",
    ]
    ended = progress(recorder)[-1]
    assert (ended["state"], ended["step"]) == ("ended", 1)
    assert recorder.events[-1].data["result"]["status"] == "crashed"


async def test_a_policy_without_a_sink_reads_nothing_at_the_start() -> None:
    from localmelo.melo.contracts.approvals import AllowAllPolicy

    class Counting(MemoryProcedures):
        lookups = 0

        async def lookup(self, *args: Any, **kwargs: Any) -> TemplateView | None:
            self.lookups += 1
            return None

    store = Counting()
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    await agent.run_task(QUERY, approval=AllowAllPolicy(), progress=True)
    assert store.lookups == 0


async def test_the_learning_bound_cuts_a_locked_file_and_the_next_run_learns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import sqlite3

    caplog.set_level(logging.DEBUG)
    agent = make_agent(
        ScriptedLLM([answer("one"), answer("two"), answer("three")]),
        memory_dir=str(tmp_path),
    )
    try:
        await agent.run_task(QUERY)
        # Below the store's 500 ms busy timeout: the bound cuts the write.
        monkeypatch.setattr(agent_module, "LEARN_TIMEOUT_S", 0.2)
        holder = sqlite3.connect(tmp_path / "procedures.db", isolation_level=None)
        try:
            holder.execute("BEGIN IMMEDIATE")
            started = time.perf_counter()
            cut = await agent.run_task(QUERY)
            assert time.perf_counter() - started < 1.0
        finally:
            holder.close()
        lines = [
            r
            for r in caplog.records
            if cut.task_id in r.getMessage() and r.levelno >= logging.INFO
        ]
        assert [(r.levelno, r.exc_info) for r in lines] == [(logging.INFO, None)]
        assert "no answer within 0.2 s" in lines[0].getMessage()
        await agent.run_task(QUERY)
        episodes = await rows(agent.hippo.procedures, "episodes")
    finally:
        await agent.close()
    assert len(episodes) == 2
