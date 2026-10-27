"""run.progress: where a running task stands, for a caller that asks (section 5.7).

``run_task(..., progress=True)`` with a sink (``on_event`` or ``observe``)
adds ``run.progress`` events: one right after ``run.started``, one after
each ``step.started``, ``tool.finished``, ``reflection.started`` and stuck
notice, and one with ``state="ended"`` right before ``run.finished``. They
rest on the template that the run-start lookup finds in the procedural
store. Level 2, the bar, is off (decision L-19): a run shows at most
level 1 unless ``tracker.LEVEL2_ENABLED`` is set, which these tests patch
to cover both values.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from localmelo.melo.agent import agent as agent_module
from localmelo.melo.contracts.events import EVENT_FIELDS
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.memory.procedures import tracker
from localmelo.melo.memory.procedures.normalize import (
    RoutineKey,
    key_for_routine,
)
from localmelo.melo.memory.procedures.store import MemoryProcedures
from localmelo.melo.memory.procedures.template import TemplateView
from localmelo.melo.memory.procedures.trace import Trace
from localmelo.melo.memory.procedures.tracker import PROGRESS_KEYS
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
    shell,
)


@pytest.fixture(params=[False, True], ids=["level2-off", "level2-on"])
def level2(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", request.param)
    return bool(request.param)


@pytest.fixture
def level2_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", False)


class CountingStore(MemoryProcedures):
    """A store in RAM that counts lookups; with *fail*, a lookup raises it."""

    def __init__(self, fail: BaseException | None = None) -> None:
        super().__init__()
        self.lookups = 0
        self.fail = fail

    async def lookup(
        self, key: RoutineKey, *, model: str, options: Any
    ) -> TemplateView | None:
        self.lookups += 1
        if self.fail is not None:
            raise self.fail
        return await super().lookup(key, model=model, options=options)


def progress(recorder: Recorder) -> list[dict[str, Any]]:
    return [dict(event.data) for event in recorder.of("run.progress")]


# ── When run.progress comes ─────────────────────────────────────────────────


async def test_the_events_come_where_the_spec_puts_them() -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "completed"
    assert recorder.types == [
        "run.started",
        "run.progress",
        "step.started",
        "run.progress",
        "assistant.message",
        "tool.started",
        "tool.finished",
        "run.progress",
        "step.started",
        "run.progress",
        "assistant.message",
        "run.progress",
        "run.finished",
    ]
    states = [data["state"] for data in progress(recorder)]
    assert states == ["running"] * 4 + ["ended"]
    assert [event.seq for event in recorder.events] == list(
        range(1, len(recorder.events) + 1)
    )


async def test_every_event_has_exactly_the_pinned_keys() -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert EVENT_FIELDS["run.progress"] == frozenset(PROGRESS_KEYS)
    for event in recorder.events:
        assert set(event.data) == EVENT_FIELDS[event.type]
        assert event.to_dict()["v"] == 1


async def test_reflections_and_stuck_notices_report_progress() -> None:
    recorder = Recorder()
    script = [read("/a", f"call_{i}") for i in range(3)]
    script += [
        answer(
            '{"recommended_action": "continue", "task_still_feasible": true, '
            '"next_step_is_concrete": true, "next_step_is_novel": true}'
        ),
        answer("done"),
    ]
    agent = make_agent(ScriptedLLM(script), steps_per_attempt=3)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    types = recorder.types
    stuck = next(
        i
        for i, e in enumerate(recorder.events)
        if e.type == "notice" and e.data["kind"] == "stuck"
    )
    assert types[stuck + 1] == "run.progress"
    reflection = types.index("reflection.started")
    assert types[reflection + 1] == "run.progress"
    assert progress(recorder)[-1]["attempt"] == 2


async def test_no_progress_without_asking_and_no_lookup() -> None:
    store = CountingStore()
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    agent.hippo.procedures = store
    await agent.run_task(QUERY, on_event=recorder)
    assert "run.progress" not in recorder.types
    assert store.lookups == 0


async def test_no_lookup_without_a_sink() -> None:
    store = CountingStore()
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    result = await agent.run_task(QUERY, progress=True)
    assert result.status == "completed"
    assert store.lookups == 0


async def test_an_observer_gets_progress_and_the_lookup_runs_once() -> None:
    store = CountingStore()
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = store
    await agent.run_task(QUERY, observe=recorder, progress=True)
    assert recorder.types == [
        "run.started",
        "run.progress",
        "step.started",
        "run.progress",
        "assistant.message",
        "run.progress",
        "run.finished",
    ]
    assert store.lookups == 1


# ── Level 0: facts ──────────────────────────────────────────────────────────


async def test_a_first_run_shows_facts_only() -> None:
    recorder = Recorder()
    agent = make_agent(
        ScriptedLLM(fix_run(writes=2)),
        tools=passing_tools(),
        steps_per_attempt=10,
        max_attempts=2,
        timeout_s=120.0,
    )
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    states = progress(recorder)
    first, last = states[0], states[-1]
    assert first == {
        "level": 0,
        "state": "running",
        "step": 0,
        "attempt": 0,
        "limits": {
            "max_steps": 30,
            "steps_per_attempt": 10,
            "max_attempts": 2,
            "timeout_s": 120.0,
        },
        "phase": None,
        "cycle": 0,
        "tests": None,
        "items": None,
        "fraction": None,
        "expected_steps": None,
        "past": None,
        "off_reason": None,
        "basis": None,
    }
    phases = [s["phase"] for s in states]
    assert phases[:3] == [None, None, "Explore"]
    assert "Write" in phases and "Verify" in phases
    assert last["state"] == "ended" and last["phase"] == "Finishing"
    assert (last["step"], last["attempt"], last["cycle"]) == (6, 1, 2)
    assert last["tests"] == {"passed": 8, "failed": 0, "total": 8}
    failed_round = next(s for s in states if s["tests"] is not None)
    assert failed_round["tests"] == {"passed": 7, "failed": 1, "total": 8}
    for s in states:
        assert s["level"] == 0
        assert (s["fraction"], s["expected_steps"], s["items"]) == (None, None, None)


async def test_one_attempt_has_no_attempt_limits() -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([answer("done")]), attempt_policy="none")
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert progress(recorder)[0]["limits"] == {
        "max_steps": 30,
        "steps_per_attempt": None,
        "max_attempts": None,
        "timeout_s": None,
    }


# ── Levels 1 and 2 ──────────────────────────────────────────────────────────


async def test_one_past_run_gives_level_1(level2: bool) -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    await remember(agent, 4)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    states = progress(recorder)
    assert {s["level"] for s in states} == {1}
    first, last = states[0], states[-1]
    assert first["past"] == {
        "runs": 1,
        "steps_lo": 4,
        "steps_median": 4,
        "steps_hi": 4,
        "wall_lo_s": 60.0,
        "wall_hi_s": 60.0,
        "last_steps": 4,
        "last_wall_s": 60.0,
    }
    assert first["basis"]["match"] == "exact"
    assert first["basis"]["model"] == MODEL
    assert [item["label"] for item in first["items"]] == [
        "Read test_parser.py",
        "Edit parser.py",
        "pytest",
    ]
    assert [item["state"] for item in first["items"]] == ["pending"] * 3
    assert [item["state"] for item in last["items"]] == ["done"] * 3
    for s in states:
        assert (s["fraction"], s["expected_steps"], s["off_reason"]) == (
            None,
            None,
            None,
        )


async def test_a_passing_gate_shows_level_1_while_level_2_is_off(
    level2_off: None,
) -> None:
    recorder = Recorder()
    # Twice as long as both past runs: at level 2 the bar would stop.
    agent = make_agent(
        ScriptedLLM(fix_run(writes=3)), tools=passing_tools(), steps_per_attempt=12
    )
    await remember(agent, 4, 4)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    states = progress(recorder)
    assert {s["level"] for s in states} == {1}
    assert [s["state"] for s in states] == ["running"] * (len(states) - 1) + ["ended"]
    for s in states:
        assert s["fraction"] is None
        assert s["expected_steps"] is None
        assert s["off_reason"] is None
    assert states[0]["past"]["runs"] == 2


async def test_with_level_2_on_the_bar_shows_and_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", True)
    recorder = Recorder()
    agent = make_agent(
        ScriptedLLM(fix_run(writes=3)), tools=passing_tools(), steps_per_attempt=12
    )
    await remember(agent, 4, 4)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    states = progress(recorder)
    assert {s["level"] for s in states} == {2}
    fractions = [s["fraction"] for s in states]
    assert fractions[0] == 0.0
    assert fractions == sorted(fractions)
    assert max(fractions[:-1]) <= 0.95 and fractions[-1] == 1.0
    longer = [s for s in states if s["state"] == "off_template"]
    assert longer and {s["off_reason"] for s in longer} == {"longer"}
    first_off = states.index(longer[0])
    assert longer[0]["step"] == 5
    # Sticky: every later event before the end is off the template.
    assert all(s["state"] == "off_template" for s in states[first_off:-1])
    assert states[1]["expected_steps"] == 4


async def test_a_completed_short_run_at_level_2_ends_at_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", True)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    await remember(agent, 4, 4)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    states = progress(recorder)
    assert [s["state"] for s in states] == ["running"] * (len(states) - 1) + ["ended"]
    assert [s["fraction"] for s in states][-1] == 1.0
    assert all(s["off_reason"] is None for s in states)


async def test_an_unknown_model_id_shows_level_1_at_most(level2: bool) -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run(), model="default"), tools=passing_tools())
    await remember(agent, 4, 4, model="default")
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert {s["level"] for s in progress(recorder)} == {1}


async def test_another_model_starts_at_level_0() -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    await remember(agent, 4, 4, model="another-model")
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert {s["level"] for s in progress(recorder)} == {0}


async def test_a_named_routine_is_found_from_any_wording(level2_off: None) -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    await remember(agent, 4, key=key_for_routine("daily-fix"))
    await agent.run_task(
        "please fix the parser tests",
        on_event=recorder,
        progress=True,
        routine="daily-fix",
    )
    first = progress(recorder)[0]
    assert first["level"] == 1 and first["basis"]["match"] == "routine"


@pytest.mark.parametrize(
    ("routine", "error"), [("  ", ValueError), (7, TypeError)], ids=["blank", "int"]
)
async def test_a_bad_routine_name_is_refused_before_the_run(
    routine: Any, error: type[Exception]
) -> None:
    llm = ScriptedLLM([answer("never")])
    recorder = Recorder()
    agent = make_agent(llm)
    with pytest.raises(error, match="routine"):
        await agent.run_task(QUERY, on_event=recorder, routine=routine)
    assert (recorder.events, llm.requests) == ([], [])
    assert agent.hippo.history._tasks == {}  # type: ignore[attr-defined]


# ── Failures never fail a run ───────────────────────────────────────────────


async def test_a_failed_lookup_gives_level_0(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    await remember(agent, 4, 4)
    store = CountingStore(fail=RuntimeError("disk on fire"))
    store._state = agent.hippo.procedures._state  # type: ignore[attr-defined]
    agent.hippo.procedures = store
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "completed"
    assert {s["level"] for s in progress(recorder)} == {0}
    lines = [r for r in caplog.records if "disk on fire" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO and lines[0].exc_info is None
    assert result.task_id in lines[0].getMessage()


async def test_a_slow_lookup_gives_level_0_within_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowStore(MemoryProcedures):
        async def lookup(self, *args: Any, **kwargs: Any) -> TemplateView | None:
            await asyncio.Event().wait()
            return None

    monkeypatch.setattr(agent_module, "LOOKUP_TIMEOUT_S", 0.05)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([answer("done")]))
    agent.hippo.procedures = SlowStore()
    result = await asyncio.wait_for(
        agent.run_task(QUERY, on_event=recorder, progress=True), timeout=5
    )
    assert result.status == "completed"
    assert {s["level"] for s in progress(recorder)} == {0}


def test_the_bounds_are_the_spec_s() -> None:
    assert agent_module.LOOKUP_TIMEOUT_S == 1.0


async def test_a_trace_that_fails_ends_progress_for_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: Trace, *args: Any) -> None:
        raise RuntimeError("tracker bug")

    monkeypatch.setattr(Trace, "tool", broken)
    caplog.set_level(logging.DEBUG)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), read("/p/b.py"), answer("done")]))
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "completed"
    # run.started, progress, step 1 and its progress; none after the tool.
    assert recorder.types.count("run.progress") == 2
    assert recorder.types[-1] == "run.finished"
    lines = [r for r in caplog.records if "tracker bug" in r.getMessage()]
    assert lines and all(r.levelno == logging.DEBUG for r in lines)
    assert all(r.exc_info is None for r in caplog.records if r.levelno > logging.DEBUG)


async def test_a_tracker_that_fails_ends_progress_but_not_learning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = tracker._track
    calls = 0

    def flaky(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("tracker bug")
        return real(*args)

    monkeypatch.setattr(tracker, "_track", flaky)
    caplog.set_level(logging.DEBUG)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), read("/p/b.py"), answer("done")]))
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "completed"
    # The first event came; the second failed, and none came after it.
    assert recorder.types.count("run.progress") == 1
    lines = [r for r in caplog.records if "tracker bug" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.DEBUG
    assert all(r.exc_info is None for r in caplog.records if r.levelno > logging.DEBUG)
    # The trace is sound, so the run is still learned.
    episodes = [
        row
        for row in await agent.hippo.procedures.export_rows()
        if row["table"] == "episodes"
    ]
    assert [(e["task_id"], e["n_tool"]) for e in episodes] == [(result.task_id, 2)]


# ── The end, on every path ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        (ProviderError("HTTP 400: refused", status_code=400), "provider_error"),
        (RuntimeError("boom"), "crashed"),
    ],
)
async def test_the_ended_event_comes_on_a_failed_run(
    reply: BaseException, status: str
) -> None:
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), reply]))
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == status
    assert recorder.types[-2:] == ["run.progress", "run.finished"]
    assert progress(recorder)[-1]["state"] == "ended"


async def test_the_ended_event_comes_from_the_finalizer() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools)
    running = asyncio.create_task(
        agent.run_task(QUERY, on_event=recorder, progress=True)
    )
    await asyncio.wait_for(tools.hanging.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert recorder.types[-2:] == ["run.progress", "run.finished"]
    ended = progress(recorder)[-1]
    assert (ended["state"], ended["step"]) == ("ended", 1)
    assert recorder.events[-1].data["result"]["status"] == "cancelled"


async def test_the_ended_event_comes_on_a_timeout() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools, timeout_s=0.2)
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "timeout"
    assert recorder.types[-2:] == ["run.progress", "run.finished"]


async def test_a_shell_command_s_output_never_reaches_progress() -> None:
    recorder = Recorder()
    tools = Tools()
    secret = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    tools.shell["cat notes.txt"] = f"token {secret}"
    agent = make_agent(ScriptedLLM([shell("cat notes.txt"), answer("ok")]), tools=tools)
    await agent.run_task(QUERY, on_event=recorder, progress=True)
    for data in progress(recorder):
        assert secret not in str(data)
