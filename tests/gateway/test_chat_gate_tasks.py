"""The gate's worker-task lifecycle: registration, calls, levels and the end."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
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
    assert (tasks[0].phase, tasks[0].level) == ("running", 2)
    assert gate.level("worker") == 2
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


async def test_relevel_preserves_policy_phase_after_admission() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task(phase="queued"))
    call = gate.submit("worker")
    await gate.acquire(call)
    assert state_task(gate).phase == "running"  # type: ignore[union-attr]
    gate.set_task_level("worker", "P1")
    assert state_task(gate).phase == "running"  # type: ignore[union-attr]
    gate.finish(call)
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


@pytest.mark.parametrize(
    "origin", ["user", "user_rule", "front", "harness", "agent", "channel", "sleep"]
)
async def test_user_may_relevel_any_origin_and_recover_it(origin: str) -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task("worker", origin=origin))
    assert gate.set_task_level("worker", "P1") == 1
    recovered = gate.registered_tasks[0]
    assert (recovered.level, recovered.level_source) == (1, "user")
    await gate.close()
    gate = ChatGate(policy="class")
    gate.register_task(recovered)
    assert gate.set_task_level("worker", "P0") == 0
    await gate.close()


@pytest.mark.parametrize("origin", ["agent", "front", "channel", "sleep"])
async def test_recover_user_rule_level_on_any_origin(origin: str) -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task(origin=origin, level=1, level_source="user_rule"))
    assert gate.level("worker") == 1
    assert gate.level_source("worker") == "user_rule"
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "class"])
@pytest.mark.parametrize("phase", ["done", "failed", "cancelled", "expired"])
async def test_terminal_task_cannot_be_revived(policy: str, phase: str) -> None:
    gate = ChatGate(policy=policy)
    gate.register_task(task("live", level=0, level_source="user"))
    gate.register_task(task("dead", phase=phase))
    with pytest.raises(ValueError, match="terminal|finished"):
        gate.update_task("dead", phase="running")
    with pytest.raises(ValueError, match="terminal|finished"):
        gate.register_task(task("dead", phase="running", level=0, level_source="user"))
    with pytest.raises(ValueError, match="terminal|finished"):
        gate.set_task_level("dead", "P0")
    assert gate.level("live") == 0
    assert next(t for t in gate.registered_tasks if t.task_id == "dead").phase == phase
    await gate.close()


@pytest.mark.parametrize("source", ["user", "user_rule"])
async def test_reregistration_cannot_replace_a_pinned_level_with_computed(
    source: str,
) -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task(level=1, level_source=source))
    gate.register_task(task(level=3, level_source="computed"))
    assert (gate.level("worker"), gate.level_source("worker")) == (1, source)
    assert gate.set_task_level("worker", "auto") == 2
    gate.register_task(task(level=3, level_source="computed"))
    assert (gate.level("worker"), gate.level_source("worker")) == (2, "computed")
    await gate.close()


async def test_update_task_cannot_change_level_or_source() -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task("first", phase="running"))
    gate.register_task(task("second", phase="running"))
    gate.set_task_level("first", "P0")
    for changes in (
        {"level": 0},
        {"level_source": "user"},
        {"level": 0, "level_source": "user"},
    ):
        with pytest.raises(ValueError, match="may not change"):
            gate.update_task("second", **changes)
    assert gate.level("first") == 0
    assert gate.level("second") == 2
    busy = gate.submit("busy")
    first = gate.submit("first")
    second = gate.submit("second")
    assert first.future is not None and not first.future.done()
    assert second.future is not None and not second.future.done()
    gate.finish(busy)
    assert first.future.done() and not second.future.done()
    gate.finish(first)
    await gate.acquire(second)
    gate.finish(second)
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
    with pytest.raises(ValueError, match="may not change"):
        gate.update_task("worker", level=0, level_source="computed")
    with pytest.raises(ValueError, match="No task is registered"):
        gate.update_task("missing", phase="running")
    with pytest.raises(ValueError, match="finished task"):
        gate.finish_task("worker", "running")
    await gate.close()


async def test_off_still_tags_the_call_and_forget_keeps_the_terminal_task() -> None:
    gate = ChatGate(policy="off")
    gate.register_task(task())
    call = gate.submit("worker")
    assert call.meta.session_kind == "task"
    assert call.future is None
    gate.finish(call)
    gate.finish_task("worker", "cancelled")
    assert [t.task_id for t in gate.registered_tasks] == ["worker"]
    gate.forget("worker")
    assert gate.registered_tasks[0].phase == "cancelled"
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
@pytest.mark.parametrize("lock", ["record", "wait"])
@pytest.mark.parametrize("kind", ["plan", "embed"])
async def test_terminal_task_submissions_are_immediately_refused(
    policy: str, lock: str, kind: str
) -> None:
    gate = ChatGate(policy=policy, lock_gate=lock)
    gate.register_task(task())
    gate.finish_task("worker", "cancelled")
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("worker", kind=kind)
    assert call.future is not None and call.future.cancelled()
    assert not call.lock_waiting
    with pytest.raises(asyncio.CancelledError):
        await gate.acquire(call)
    gate.finish(call, "refused")
    await gate.close()


async def test_terminal_or_invalid_target_does_not_demote_live_p0() -> None:
    gate = ChatGate(policy="class")
    gate.register_task(task("live", level=0, level_source="user"))
    gate.register_task(task("dead", phase="done"))
    for target, choice in [("dead", "P0"), ("missing", "P0"), ("dead", "P9")]:
        with pytest.raises(ValueError):
            gate.set_task_level(target, choice)
        assert gate.level("live") == 0
    gate.register_task(task("old", phase="done", level=0, level_source="user"))
    assert gate.level("live") == 0
    await gate.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"level": 2.5},
        {"level": True},
        {"origin": "robot"},
        {"phase": "bogus"},
        {"level_source": "bad"},
        {"size": "huge"},
        {"wait_reason": "bogus"},
    ],
)
async def test_register_task_rejects_invalid_fields(changes: dict[str, Any]) -> None:
    gate = ChatGate(policy="class")
    with pytest.raises(ValueError):
        gate.register_task(replace(task(), **changes))
    assert gate.registered_tasks == ()
    await gate.close()


async def test_finish_task_does_not_release_a_blocked_provider_slot() -> None:
    class BlockingProvider(BaseLLMProvider):
        model = "fake"
        last_reasoning_key = ""
        last_usage = {"prompt_tokens": 3, "completion_tokens": 2}

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            self.started.set()
            await self.release.wait()
            return Message(role="assistant", content="ok")

        async def close(self) -> None:
            pass

    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task("first", phase="running"))
    gate.register_task(task("second", phase="running"))
    provider = BlockingProvider()
    running = asyncio.create_task(GatedLLM(provider, gate.client("first")).chat([]))
    await asyncio.wait_for(provider.started.wait(), 1)
    pending = gate.submit("first")
    assert pending.future is not None and not pending.future.done()
    gate.finish_task("first", "cancelled")
    assert pending.done and pending.future.cancelled()
    assert gate._inflight() == 1
    second = gate.submit("second")
    assert second.future is not None and not second.future.done()
    provider.release.set()
    await asyncio.wait_for(running, 1)
    await asyncio.wait_for(gate.acquire(second), 1)
    assert gate._inflight() == 1
    gate.finish(second)
    await gate.close()


@pytest.mark.parametrize("method", ["update", "register"])
@pytest.mark.parametrize("phase", ["done", "failed", "cancelled", "expired"])
@pytest.mark.parametrize("parked", [False, True])
async def test_terminal_snapshots_cannot_strand_live_calls(
    method: str, phase: str, parked: bool
) -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0, lock_gate="wait")
    gate.register_task(task(phase="running"))
    admitted = gate.submit("worker")
    await gate.acquire(admitted)
    if parked:
        gate.set_machine(Machine(lock="foreign"))
    waiting = gate.submit("worker")
    assert waiting.future is not None and not waiting.future.done()
    assert waiting.lock_waiting is parked
    with pytest.raises(ValueError, match="finish_task"):
        if method == "update":
            gate.update_task("worker", phase=phase)
        else:
            gate.register_task(task(phase=phase))
    assert gate.registered_tasks[0].phase == "running"
    assert not waiting.future.done()
    gate.finish_task("worker", phase)
    assert waiting.done and waiting.future.cancelled()
    gate.finish(admitted)
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
async def test_terminal_recovery_cannot_rewrite_final_phase(policy: str) -> None:
    gate = ChatGate(policy=policy)
    recovered = task(phase="failed", level=1, level_source="user")
    gate.register_task(recovered)
    assert gate.finish_task("worker", "cancelled").phase == "failed"
    with pytest.raises(ValueError, match="finished"):
        gate.register_task(task(phase="done"))
    with pytest.raises(ValueError, match="finished"):
        gate.update_task("worker", phase="done")
    gate.register_task(recovered)
    assert gate.registered_tasks[0].phase == "failed"
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
async def test_forget_with_live_call_does_not_mutate_worker(policy: str) -> None:
    gate = ChatGate(policy=policy, n_srv=1, r_f=0)
    gate.register_task(task(phase="running", level=1, level_source="user"))
    first = gate.submit("worker")
    await gate.acquire(first)
    registry, state = gate.registered_tasks, gate._state
    counts, levels = dict(gate._counts), dict(gate._levels)
    gate.forget("worker")
    assert gate.registered_tasks == registry and gate._state == state
    assert gate._counts == counts and gate._levels == levels
    second = gate.submit("worker", kind="embed")
    assert first.meta.call_id != second.meta.call_id
    assert second.meta.session_kind == "task"
    assert gate._calls[first.meta.call_id] is first
    assert not any(a.startswith("sched_transition_failed") for a in gate.alarms)
    gate.finish(second)
    gate.finish(first)
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
async def test_forget_idle_live_worker_preserves_policy_state(policy: str) -> None:
    gate = ChatGate(policy=policy)
    gate.register_task(task(phase="running"))
    gate.forget("worker")
    call = gate.submit("worker")
    await gate.acquire(call)
    assert call.meta.session_kind == "task"
    assert not any(a.startswith("sched_transition_failed") for a in gate.alarms)
    if policy != "off":
        assert state_task(gate) is not None
    gate.finish(call)
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "class"])
async def test_discard_deleted_task_waits_for_its_live_calls(policy: str) -> None:
    gate = ChatGate(policy=policy)
    gate.register_task(task(phase="running"))
    assert gate.discard_task("worker") is False
    admitted = gate.submit("worker")
    await gate.acquire(admitted)
    gate.finish_task("worker", "done")
    assert gate.discard_task("worker") is False
    assert gate.registered_tasks[0].phase == "done"
    gate.finish(admitted)
    assert gate.discard_task("worker") is True
    assert gate.registered_tasks == ()
    assert gate.discard_task("worker") is False
    assert "worker" not in gate._counts and "worker" not in gate._levels
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
async def test_forget_finished_task_keeps_refusing_and_unique_ids(policy: str) -> None:
    gate = ChatGate(policy=policy)
    gate.register_task(task(phase="running"))
    active = gate.submit("worker")
    await gate.acquire(active)
    gate.finish_task("worker", "done")
    gate.forget("worker")  # admitted calls still own their slot
    assert gate.registered_tasks[0].phase == "done"
    refused_while_active = gate.submit("worker")
    assert refused_while_active.meta.call_id != active.meta.call_id
    assert refused_while_active.future is not None
    assert refused_while_active.future.cancelled()
    assert gate._calls[active.meta.call_id] is active
    gate.finish(active)
    gate.forget("worker")
    refused = gate.submit("worker")
    assert refused.meta.session_kind == "task"
    assert refused.meta.call_id != active.meta.call_id
    assert refused.future is not None and refused.future.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await gate.acquire(refused)
    with pytest.raises(ValueError, match="finished"):
        gate.register_task(task(phase="running"))
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
async def test_stale_reregistration_does_not_steal_priority(policy: str) -> None:
    gate = ChatGate(policy=policy)
    stale = task("a", level=0, level_source="user")
    gate.register_task(stale)
    gate.register_task(task("b", level=0, level_source="user"))
    gate.register_task(replace(stale, phase="running", user_rank=42, paused=True))
    a, b = gate.registered_tasks
    assert (a.phase, a.level, a.level_source, a.user_rank, a.paused) == (
        "running",
        1,
        "user",
        None,
        False,
    )
    assert (b.level, b.level_source) == (0, "user")
    assert gate.level("a") == 1 and gate.level("b") == 0
    if policy != "off":
        assert state_task(gate, "a").level == 1  # type: ignore[union-attr]
        assert state_task(gate, "b").level == 0  # type: ignore[union-attr]
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs", "class"])
@pytest.mark.parametrize("lock", ["record", "wait"])
async def test_terminal_refusals_are_logged_once(
    tmp_path: Path, policy: str, lock: str
) -> None:
    gate = ChatGate(policy=policy, lock_gate=lock, call_log_dir=tmp_path)
    gate.register_task(task(phase="done"))
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("worker")
    with pytest.raises(asyncio.CancelledError):
        await gate.acquire(call)
    gate.finish(call, "refused")
    gate.finish(call, "refused")
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["status"] == "refused"
    assert lines[0]["session_kind"] == "task"
    assert lines[0]["admit_s"] is None
    assert lines[0]["call_index"] == call.meta.call_index


@pytest.mark.parametrize("kind", ["plan", "reflect"])
async def test_park_wins_before_admission(kind: str) -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task(phase="running"))
    busy = gate.submit("busy")
    call = gate.submit("worker", kind=kind)
    assert call.future is not None and not call.future.done()

    def cancel() -> bool:
        # This is the deterministic admission barrier: dispatch runs inside cancel.
        gate.finish(busy)
        gate.wake()
        assert call.future is not None
        assert call.meta.admitted_mono_us is None and not call.future.done()
        return True

    assert gate.try_park("worker", cancel) is call
    gate.wake()
    assert call.meta.admitted_mono_us is None and not call.future.done()
    gate.finish(call, "refused")
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "class"])
async def test_park_fences_foreign_lock_release_and_failure(policy: str) -> None:
    gate = ChatGate(policy=policy, lock_gate="wait")
    gate.register_task(task(phase="running"))
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("worker")

    def cancel() -> bool:
        gate.set_machine(Machine(lock="none"))
        gate._failed(gate._clock(), RuntimeError())
        assert call.meta.admitted_mono_us is None
        assert call.future is not None and not call.future.done()
        return True

    assert gate.try_park("worker", cancel) is call
    gate.finish(call, "refused")
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "class"])
@pytest.mark.parametrize("error", [False, True])
async def test_park_rollback_releases_lock_waiter(policy: str, error: bool) -> None:
    gate = ChatGate(policy=policy, lock_gate="wait")
    gate.register_task(task(phase="running"))
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("worker")

    def cancel() -> bool:
        gate.set_machine(Machine(lock="none"))
        if error:
            raise RuntimeError("cancel failed")
        return False

    if error:
        with pytest.raises(RuntimeError, match="cancel failed"):
            gate.try_park("worker", cancel)
    else:
        assert gate.try_park("worker", cancel) is None
    await gate.acquire(call)
    assert call.meta.admitted_mono_us is not None
    gate.finish(call)
    await gate.close()


async def test_park_rollback_after_stop_does_not_resurrect_call() -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task(phase="running"))
    busy = gate.submit("busy")
    call = gate.submit("worker")

    def cancel() -> bool:
        gate.finish_task("worker", "cancelled")
        return False

    assert gate.try_park("worker", cancel) is None
    assert call.done and call.future is not None and call.future.cancelled()
    assert all(c.call_id != call.meta.call_id for c in gate._state.calls)
    gate.finish(busy)
    await gate.close()


async def test_park_rollback_redispatches_policy_call() -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task(phase="running"))
    busy = gate.submit("busy")
    call = gate.submit("worker")

    def cancel() -> bool:
        gate.finish(busy)
        assert call.meta.admitted_mono_us is None
        return False

    assert gate.try_park("worker", cancel) is None
    await gate.acquire(call)
    assert call.meta.admitted_mono_us is not None
    gate.finish(call)
    await gate.close()


@pytest.mark.parametrize("kind", ["plan", "reflect", "embed", "retrieve", "estimate"])
async def test_park_refuses_unsafe_or_admitted_calls(kind: str) -> None:
    gate = ChatGate(policy="off", lock_gate="wait")
    gate.register_task(task(phase="running"))
    admitted = gate.submit("worker", kind=kind)
    assert gate.try_park("worker", lambda: pytest.fail("cancel called")) is None
    gate.finish(admitted)
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("worker", kind=kind)
    if kind not in ("plan", "reflect"):
        assert gate.try_park("worker", lambda: pytest.fail("cancel called")) is None
    gate.finish(call, "refused")
    chat = gate.submit("chat")
    assert gate.try_park("chat", lambda: pytest.fail("cancel called")) is None
    gate.finish(chat, "refused")
    await gate.close()


async def test_park_cancels_runner_at_real_acquire_barrier() -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task(phase="running"))
    busy = gate.submit("busy")
    reached = asyncio.Event()
    sent = False

    async def run() -> None:
        nonlocal sent
        call = gate.submit("worker")
        reached.set()
        try:
            await gate.acquire(call)
            sent = True
        finally:
            gate.finish(call, "refused")

    runner = asyncio.create_task(run())
    await reached.wait()
    parked = gate.try_park("worker", lambda: runner.cancel("park"))
    assert parked is not None
    gate.finish(busy)
    with pytest.raises(asyncio.CancelledError, match="park"):
        await runner
    assert not sent and parked.done and parked.meta.admitted_mono_us is None
    await gate.close()


async def test_admission_wins_and_multiple_calls_are_not_safe() -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0)
    gate.register_task(task(phase="running"))
    busy = gate.submit("busy")
    call = gate.submit("worker")
    gate.finish(busy)
    assert gate.try_park("worker", lambda: pytest.fail("cancel called")) is None
    second = gate.submit("worker")
    assert gate.try_park("worker", lambda: pytest.fail("cancel called")) is None
    gate.finish(second, "refused")
    gate.finish(call)
    assert gate.try_park("worker", lambda: pytest.fail("cancel called")) is None
    await gate.close()


def test_the_gate_s_p0_is_the_user_s_task_p0() -> None:
    gate = ChatGate(policy="class")
    assert gate._params is not None
    assert gate._params.p0_meaning == "task"
