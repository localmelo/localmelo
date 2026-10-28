"""The gate's worker-task lifecycle: registration, calls, levels and the end."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.scheduler import Clock, Machine, Task


class ClockBox:
    def __init__(self) -> None:
        self.mono_us = 1_000_000
        self.wall_us = 1_700_000_000_000_000

    def __call__(self) -> Clock:
        return Clock(self.mono_us, self.wall_us)

    def advance(self, seconds: float) -> None:
        step = round(seconds * 1_000_000)
        self.mono_us += step
        self.wall_us += step


def task(
    task_id: str = "worker",
    *,
    level: int = 2,
    level_source: str = "computed",
    phase: str = "queued",
    origin: str = "user",
) -> Task:
    return Task(
        task_id=task_id,
        kind="task",
        origin=origin,  # type: ignore[arg-type]
        level=level,
        level_source=level_source,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        created_mono_us=1_000_000,
        created_wall_us=1_700_000_000_000_000,
    )


def state_task(gate: ChatGate, task_id: str = "worker") -> Task | None:
    return next((t for t in gate._state.tasks if t.task_id == task_id), None)


async def test_a_registered_task_runs_task_calls_and_is_not_ended_by_one() -> None:
    clock = ClockBox()
    gate = ChatGate(policy="class", clock=clock)
    gate.register_task(task(phase="running"))
    registered = state_task(gate)
    assert registered is not None and registered.kind == "task"

    call = gate.submit("worker", kind="plan")
    assert call.meta.session_kind == "task"
    assert call.meta.lane is False
    await gate.acquire(call)  # admitted at once: the queue is empty
    gate.finish(call)
    still = state_task(gate)
    assert still is not None
    assert still.phase == "running"
    gate.finish_task("worker", "done")
    done = state_task(gate)
    assert done is not None and done.phase == "done"
    await gate.close()


async def test_a_worker_embedding_also_belongs_to_the_task() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task(phase="running"))
    call = gate.submit("worker", kind="embed")
    assert call.meta.session_kind == "task"
    await gate.acquire(call)
    call_events = [
        event for event in gate._state.calls if event.call_id == call.meta.call_id
    ]
    assert call_events and call_events[0].task_id == "worker"
    gate.finish(call)
    await gate.close()


async def test_the_trace_says_task_and_never_reopens_a_chat(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="class", call_log_dir=tmp_path)
    gate.register_task(task(phase="running"))
    call = gate.submit("worker")
    await gate.acquire(call)
    gate.finish(call)
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    line = json.loads(path.read_text().splitlines()[0])
    assert line["session_kind"] == "task"
    assert line["task_key"] == "worker"


async def test_registering_again_updates_the_task_instead_of_adding_it() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task(phase="queued"))
    gate.register_task(task(phase="running", level=1, level_source="user"))
    tasks = [t for t in gate._state.tasks if t.task_id == "worker"]
    assert len(tasks) == 1
    assert (tasks[0].phase, tasks[0].level) == ("running", 1)
    assert gate.level("worker") == 1
    await gate.close()


async def test_a_user_p0_demotes_the_prior_p0_and_first_and_auto_apply() -> None:
    clock = ClockBox()
    gate = ChatGate(policy="class", clock=clock)
    gate.register_task(task("first", phase="running"))
    gate.register_task(task("second", phase="running"))
    assert gate.set_task_level("first", "P0") == 0
    assert gate.set_task_level("second", "p0") == 0
    assert gate.level("first") == 1
    demoted = state_task(gate, "first")
    assert demoted is not None and demoted.user_rank is None
    assert gate.set_task_level("first", "first") == 1
    ranked = state_task(gate, "first")
    assert ranked is not None and ranked.user_rank is not None
    assert ranked.user_rank > 0
    clock.advance(60)
    assert gate.set_task_level("second", "first") == 0
    later = state_task(gate, "second")
    assert later is not None and later.user_rank is not None
    assert ranked.user_rank < later.user_rank
    assert gate.set_task_level("first", "auto") == 2
    auto = state_task(gate, "first")
    assert auto is not None
    assert (auto.level, auto.level_source) == (2, "computed")
    assert gate.set_task_level("second", "P5") == 5
    later = state_task(gate, "second")
    assert later is not None and later.paused is True
    with pytest.raises(ValueError, match="priority must be"):
        gate.set_task_level("first", "urgent")
    await gate.close()


async def test_a_finished_task_is_not_demoted_by_a_new_p0() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task("done", phase="running", level=0, level_source="user"))
    gate.finish_task("done", "done")
    gate.register_task(task("new", phase="running"))
    assert gate.set_task_level("new", "P0") == 0
    done = state_task(gate, "done")
    assert done is not None and done.level == 0
    await gate.close()


async def test_registering_a_second_p0_demotes_the_first() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task("a", level=0, level_source="user"))
    gate.register_task(task("b", level=0, level_source="user"))
    first = state_task(gate, "a")
    second = state_task(gate, "b")
    assert first is not None and second is not None
    assert (first.level, first.user_rank) == (1, None)
    assert second.level == 0
    await gate.close()


async def test_finish_task_cancels_a_call_parked_at_the_gate() -> None:
    gate = ChatGate(policy="class", lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    gate.register_task(task(phase="running"))
    call = gate.submit("worker")
    assert call.lock_waiting is True
    gate.finish_task("worker", "cancelled")
    assert call.done is True
    with pytest.raises(asyncio.CancelledError):
        await gate.acquire(call)
    assert gate.held("worker") == ""
    await gate.close()


async def test_a_non_user_task_cannot_claim_a_user_level() -> None:
    gate = ChatGate(policy="class")
    with pytest.raises(ValueError, match="user-origin"):
        gate.register_task(task("front", origin="front", level_source="user"))
    gate.register_task(task("front", origin="front"))
    with pytest.raises(ValueError, match="user-origin"):
        gate.update_task("front", level=1, level_source="user")
    await gate.close()


async def test_a_pinned_gate_level_needs_force_to_become_computed() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task())
    gate.set_task_level("worker", "P1")
    with pytest.raises(ValueError, match="force_level"):
        gate.update_task("worker", level=3, level_source="computed")
    changed = gate.update_task(
        "worker", level=3, level_source="computed", force_level=True
    )
    assert (changed.level, changed.level_source) == (3, "computed")
    await gate.close()


async def test_a_focused_worker_is_never_the_lane() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task(phase="running"))
    gate.focus("worker")
    call = gate.submit("worker", kind="plan")
    assert call.meta.lane is False
    assert call.meta.session_kind == "task"
    await gate.acquire(call)
    gate.finish(call)
    await gate.close()


async def test_update_task_rejects_identifying_fields_and_a_computed_p0() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task())
    changed = gate.update_task("worker", wait_reason="input", phase="waiting")
    assert (changed.phase, changed.wait_reason) == ("waiting", "input")
    with pytest.raises(ValueError, match="may not change"):
        gate.update_task("worker", kind="chat")
    with pytest.raises(ValueError, match="P0 is the user"):
        gate.update_task("worker", level=0, level_source="computed")
    with pytest.raises(ValueError, match="No task is registered"):
        gate.update_task("missing", phase="running")
    with pytest.raises(ValueError, match="finished task"):
        gate.finish_task("worker", "running")
    await gate.close()


async def test_off_still_tags_the_call_and_forget_drops_the_task() -> None:
    gate = ChatGate(policy="off")
    gate.register_task(task())
    call = gate.submit("worker")
    assert call.meta.session_kind == "task"
    assert call.future is None
    gate.finish(call)
    gate.finish_task("worker", "cancelled")
    assert [t.task_id for t in gate.registered_tasks] == ["worker"]
    gate.forget("worker")
    assert gate.registered_tasks == ()
    await gate.close()


def test_the_gate_s_p0_is_the_user_s_task_p0() -> None:
    gate = ChatGate(policy="class")
    assert gate._params is not None
    assert gate._params.p0_meaning == "task"
