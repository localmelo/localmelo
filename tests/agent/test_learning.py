"""Learning at a run's end: the episode and template version (section 5.6).

``run_task`` records each run that ends through its normal path in the
procedural store, after ``run.finished`` and before it returns, so the
run's events and ``RunResult.duration_s`` are as they were. Skipped during
an exit and on a cancelled interactive run; bounded by 2 s; a failure is
one INFO line without a traceback; a cancellation propagates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import agent as agent_module
from localmelo.melo.memory.procedures import tracker
from localmelo.melo.memory.procedures.store import (
    LearnResult,
    MemoryProcedures,
    Procedures,
    RunFacts,
    harness_version,
)
from localmelo.melo.memory.procedures.template import TemplateView
from localmelo.melo.memory.procedures.trace import Trace
from localmelo.melo.schema import ToolDef
from localmelo.tests.agent.procedural_helpers import (
    MODEL,
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

AGENT_LOGGER = "localmelo.melo.agent.agent"


class SpyStore(MemoryProcedures):
    """A store in RAM that records each learn and what the sink had seen then.

    With *hold*, a learn waits for it to be set; *fail* is raised instead.
    """

    def __init__(
        self,
        recorder: Recorder | None = None,
        *,
        hold: asyncio.Event | None = None,
        delay_s: float = 0.0,
        fail: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.hold = hold
        self.delay_s = delay_s
        self.fail = fail
        self.learned: list[RunFacts] = []
        self.seen: list[list[str]] = []
        self.started = asyncio.Event()

    async def learn(
        self, facts: RunFacts, *, view: TemplateView | None = None
    ) -> LearnResult:
        self.learned.append(facts)
        self.seen.append(list(self.recorder.types) if self.recorder else [])
        self.started.set()
        if self.hold is not None:
            await self.hold.wait()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail is not None:
            raise self.fail
        return await super().learn(facts, view=view)


async def rows(store: Procedures, table: str) -> list[dict[str, Any]]:
    return [row for row in await store.export_rows() if row["table"] == table]


def info_lines(
    caplog: pytest.LogCaptureFixture, task_id: str
) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == AGENT_LOGGER
        and task_id in r.getMessage()
        and "learn" in r.getMessage()
    ]


# ── What a run's end records ────────────────────────────────────────────────


async def test_a_completed_run_records_its_episode_and_version_1() -> None:
    agent = make_agent(ScriptedLLM(fix_run(writes=2)), tools=passing_tools())
    result = await agent.run_task("Fix the parser on 2026-10-26 at 09:30")
    assert result.status == "completed"
    store = agent.hippo.procedures
    (routine,) = await rows(store, "routines")
    (episode,) = await rows(store, "episodes")
    (version,) = await rows(store, "templates")
    assert routine["query_text"] == "fix the parser on <date> at <time>"
    assert routine["place"] == os.path.realpath(os.getcwd())
    assert routine["name"] is None
    assert episode["task_id"] == result.task_id
    assert (episode["status"], episode["steps"], episode["attempts"]) == (
        "completed",
        6,
        1,
    )
    assert (episode["n_tool"], episode["first_pass_step"], episode["cycles"]) == (
        5,
        5,
        2,
    )
    assert (episode["match"], episode["level"], episode["model"]) == ("new", 0, MODEL)
    assert episode["harness_version"] == harness_version()
    assert episode["duration_s"] == pytest.approx(result.duration_s)
    labels = [s["label"] for s in json.loads(episode["segments_json"])]
    assert labels == [
        "Read test_parser.py",
        "Edit parser.py",
        "pytest",
        "Edit parser.py",
        "pytest",
    ]
    assert (version["version"], version["status"], version["source"]) == (
        1,
        "active",
        "online",
    )
    options = json.loads(version["options_json"])
    assert options == {
        "attempt_policy": "reflect",
        "reflection_gate": "flags",
        "max_steps": 30,
        "steps_per_attempt": 10,
        "max_attempts": 5,
    }


async def test_other_statuses_add_only_the_episode() -> None:
    agent = make_agent(
        ScriptedLLM([read("/a", f"call_{i}") for i in range(3)]),
        attempt_policy="none",
        max_steps=3,
    )
    result = await agent.run_task(QUERY)
    assert result.status in ("stuck", "budget_exhausted")
    store = agent.hippo.procedures
    (episode,) = await rows(store, "episodes")
    assert episode["status"] == result.status
    assert await rows(store, "templates") == []


async def test_each_run_appends_a_version_and_a_named_routine_has_no_place() -> None:
    agent = make_agent(ScriptedLLM(fix_run() + fix_run()), tools=passing_tools())
    await agent.run_task("fix it", routine="daily-fix")
    await agent.run_task("please fix the parser", routine="daily-fix")
    store = agent.hippo.procedures
    (routine,) = await rows(store, "routines")
    assert (routine["name"], routine["place"]) == ("daily-fix", "")
    versions = await rows(store, "templates")
    assert [(v["version"], v["status"]) for v in versions] == [
        (1, "superseded"),
        (2, "active"),
    ]
    assert [e["match"] for e in await rows(store, "episodes")] == ["new", "routine"]


async def test_run_learns_too() -> None:
    agent = make_agent(ScriptedLLM([answer("hi")]))
    assert await agent.run("say hi") == "hi"
    assert len(await rows(agent.hippo.procedures, "episodes")) == 1


async def test_the_episode_records_the_gate_s_level_not_the_shown_one() -> None:
    assert tracker.LEVEL2_ENABLED is False
    for progress in (True, False):
        recorder = Recorder()
        agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
        await remember(agent, 4, 4)
        result = await agent.run_task(QUERY, on_event=recorder, progress=progress)
        if progress:
            assert {e.data["level"] for e in recorder.of("run.progress")} == {1}
        episode = next(
            e
            for e in await rows(agent.hippo.procedures, "episodes")
            if e["task_id"] == result.task_id
        )
        assert (episode["level"], episode["template_version"]) == (2, 2)
        # The gate passed, so the run is scored as level 2 would have shown it.
        assert episode["shadow_bar_error"] == pytest.approx(
            (0 + 0 + 0 + abs(0.8 - 1.0)) / 4
        )


async def test_a_bench_run_reads_nothing_at_its_start_and_learns_once() -> None:
    class Counting(SpyStore):
        lookups = 0

        async def lookup(self, *args: Any, **kwargs: Any) -> TemplateView | None:
            self.lookups += 1
            return None

    store = Counting()
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    await agent.run_task(QUERY)
    assert (store.lookups, len(store.learned)) == (0, 1)


# ── When it happens ─────────────────────────────────────────────────────────


async def test_learning_comes_after_run_finished_and_outside_duration() -> None:
    recorder = Recorder()
    store = SpyStore(recorder, delay_s=0.3)
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    agent.hippo.procedures = store
    started = time.perf_counter()
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    elapsed = time.perf_counter() - started
    assert store.seen == [recorder.types]
    assert recorder.types[-1] == "run.finished"
    assert result.duration_s < 0.3 <= elapsed


async def test_learning_is_skipped_during_an_exit() -> None:
    store = SpyStore()
    tools = Tools()
    agent = make_agent(ScriptedLLM([call("exit_now"), answer("done")]), tools=tools)

    async def exit_now() -> str:
        agent.begin_exit("Cancelled by SIGTERM", within_s=5.0)
        return "exiting"

    agent.hippo.register_tool(ToolDef(name="exit_now", description="", parameters={}))
    agent.executor.register("exit_now", exit_now)
    agent.hippo.procedures = store
    result = await agent.run_task(QUERY)
    assert result.status == "completed"
    assert store.learned == []


@pytest.mark.parametrize("sink", ["on_event", "observe"])
async def test_a_cancelled_run_is_not_learned(sink: str) -> None:
    store = SpyStore()
    tools = Tools()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools)
    agent.hippo.procedures = store
    sinks: dict[str, Any] = {sink: Recorder()}
    running = asyncio.create_task(agent.run_task(QUERY, **sinks))
    await asyncio.wait_for(tools.hanging.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert store.learned == []


async def test_a_trace_that_failed_is_not_learned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(self: Trace, *args: Any) -> None:
        raise RuntimeError("tracker bug")

    monkeypatch.setattr(Trace, "tool", broken)
    store = SpyStore()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    agent.hippo.procedures = store
    assert (await agent.run_task(QUERY)).status == "completed"
    assert store.learned == []


# ── Bounded, quiet and never in the way ─────────────────────────────────────


def test_the_bound_is_the_spec_s() -> None:
    assert agent_module.LEARN_TIMEOUT_S == 2.0


async def test_a_hanging_store_costs_at_most_the_bound(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(agent_module, "LEARN_TIMEOUT_S", 0.1)
    store = SpyStore(hold=asyncio.Event())
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    started = time.perf_counter()
    result = await asyncio.wait_for(agent.run_task(QUERY), timeout=5)
    assert time.perf_counter() - started < 1.0
    assert result.status == "completed"
    (line,) = info_lines(caplog, result.task_id)
    assert line.levelno == logging.INFO and line.exc_info is None
    assert "0.1 s" in line.getMessage()


async def test_a_failing_store_logs_one_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    store = SpyStore(fail=RuntimeError("disk on fire"))
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    result = await agent.run_task(QUERY)
    assert result.status == "completed"
    (line,) = info_lines(caplog, result.task_id)
    assert line.levelno == logging.INFO and line.exc_info is None
    assert "disk on fire" in line.getMessage()
    assert all(r.levelno < logging.WARNING for r in caplog.records)


async def test_a_locked_file_logs_one_info_line_and_the_next_run_learns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    agent = make_agent(
        ScriptedLLM([answer("one"), answer("two"), answer("three")]),
        memory_dir=str(tmp_path),
    )
    try:
        await agent.run_task(QUERY)
        locker = sqlite3.connect(tmp_path / "procedures.db", isolation_level=None)
        try:
            locker.execute("BEGIN EXCLUSIVE")
            locked = await agent.run_task(QUERY)
        finally:
            locker.close()
        assert locked.status == "completed"
        (line,) = info_lines(caplog, locked.task_id)
        assert line.levelno == logging.INFO and line.exc_info is None
        assert "locked" in line.getMessage()
        await agent.run_task(QUERY)
        assert len(await rows(agent.hippo.procedures, "episodes")) == 2
    finally:
        await agent.close()
    assert all(r.levelno < logging.WARNING for r in caplog.records)


async def test_a_damaged_file_logs_one_info_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    (tmp_path / "procedures.db").write_bytes(b"not a database, " * 256)
    agent = make_agent(ScriptedLLM([answer("done")]), memory_dir=str(tmp_path))
    try:
        result = await agent.run_task(QUERY)
    finally:
        await agent.close()
    assert result.status == "completed"
    (line,) = info_lines(caplog, result.task_id)
    assert line.levelno == logging.INFO and line.exc_info is None
    assert (tmp_path / "procedures.db").read_bytes() == b"not a database, " * 256


async def test_a_cancellation_during_learning_propagates() -> None:
    recorder = Recorder()
    store = SpyStore(recorder, hold=asyncio.Event())
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    running = asyncio.create_task(agent.run_task(QUERY, on_event=recorder))
    await asyncio.wait_for(store.started.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    # The run had ended: its last event and its task record say completed.
    assert recorder.events[-1].data["result"]["status"] == "completed"
    task = await agent.hippo.history.get_task(recorder.events[-1].run_id)
    assert task is not None and task.status == "completed"
    assert agent._active_runs == 0


# ── The file ────────────────────────────────────────────────────────────────


async def test_persistence_writes_a_private_procedures_db(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    agent = make_agent(
        ScriptedLLM([answer("one"), answer("two")]), memory_dir=str(memory)
    )
    try:
        first = await agent.run_task(QUERY)
        second = await agent.run_task(QUERY)
    finally:
        await agent.close()
    path = memory / "procedures.db"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with sqlite3.connect(path) as db:
        episodes = db.execute("SELECT task_id FROM episodes ORDER BY rowid").fetchall()
        active = db.execute(
            "SELECT version FROM templates WHERE status = 'active'"
        ).fetchall()
    assert episodes == [(first.task_id,), (second.task_id,)]
    assert active == [(2,)]


async def test_two_runs_of_one_template_ending_together_leave_one_active_row(
    tmp_path: Path,
) -> None:
    agents = [
        make_agent(ScriptedLLM([answer("done")]), memory_dir=str(tmp_path))
        for _ in range(2)
    ]
    try:
        results = await asyncio.gather(*(a.run_task(QUERY) for a in agents))
    finally:
        for agent in agents:
            await agent.close()
    assert {r.status for r in results} == {"completed"}
    with sqlite3.connect(tmp_path / "procedures.db") as db:
        episodes = db.execute("SELECT COUNT(*) FROM episodes").fetchone()
        versions = db.execute(
            "SELECT version, status FROM templates ORDER BY version"
        ).fetchall()
    assert episodes == (2,)
    assert versions == [(1, "superseded"), (2, "active")]


async def test_without_persistence_nothing_is_written(tmp_path: Path) -> None:
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        agent = make_agent(ScriptedLLM([answer("done")]))
        await agent.run_task(QUERY)
        await agent.close()
    finally:
        os.chdir(cwd)
    assert list(tmp_path.iterdir()) == []
    assert isinstance(agent.hippo.procedures, MemoryProcedures)
