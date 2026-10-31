"""Who runs first after approvals, and when an open chat takes slots from tasks."""

from __future__ import annotations

from localmelo.support.gateway.chat_gate import ChatGate, GateCall
from localmelo.support.scheduler import Clock, Task


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


def worker(task_id: str, level: int, *, source: str = "computed") -> Task:
    return Task(
        task_id=task_id,
        kind="task",
        origin="user",
        level=level,
        level_source=source,  # type: ignore[arg-type]
        phase="running",
        created_mono_us=1_000_000,
        created_wall_us=1_700_000_000_000_000,
    )


def admitted(call: GateCall) -> bool:
    return call.meta.admitted_mono_us is not None


def tasks(gate: ChatGate, *levels: int) -> list[str]:
    ids = [f"task{index}" for index, _ in enumerate(levels)]
    for task_id, level in zip(ids, levels, strict=True):
        gate.register_task(worker(task_id, level))
    return ids


# ── Admission after an approval ──


async def test_answer_order_does_not_decide_who_runs_first() -> None:
    clock = ClockBox()
    # One non-lane slot, held by a Today task while two approvals are answered.
    gate = ChatGate(policy="class", n_srv=3, r_f=1, clock=clock)
    today, whenever, normal = tasks(gate, 1, 3, 2)
    busy = gate.submit(today)
    assert admitted(busy)
    # Whenever is answered first, Normal a second later.
    first = gate.submit(whenever)
    clock.advance(1)
    second = gate.submit(normal)
    assert not admitted(first) and not admitted(second)
    # Normal is first in line, so Whenever waits behind a higher task.
    assert gate.held(normal) == "Resources" and gate.held(whenever) == "Priority"
    clock.advance(30)
    gate.finish(busy)
    assert admitted(second) and not admitted(first)
    clock.advance(5)
    gate.finish(second)
    assert admitted(first)
    await gate.close()


async def test_a_lower_call_in_flight_ends_before_a_higher_one_takes_the_slot() -> None:
    clock = ClockBox()
    gate = ChatGate(policy="class", n_srv=3, r_f=1, clock=clock)
    whenever, normal = tasks(gate, 3, 2)
    low = gate.submit(whenever)
    assert admitted(low)
    clock.advance(1)
    high = gate.submit(normal)
    # The call in flight is never cut short; the higher one is next.
    assert not admitted(high) and gate.held(normal) == "Resources"
    clock.advance(5)
    gate.finish(low)
    assert admitted(high)
    # The lower task's next step waits behind it, at its step boundary.
    again = gate.submit(whenever)
    assert not admitted(again) and gate.held(whenever) == "Resources"
    gate.finish(high)
    assert admitted(again)
    await gate.close()


# ── Focus ──


def gate_with_tasks(clock: ClockBox, count: int = 3) -> tuple[ChatGate, list[str]]:
    gate = ChatGate(policy="class", n_srv=8, r_f=1, n_fg=1, focus_s=10, clock=clock)
    return gate, tasks(gate, *([2] * count))


async def test_the_focus_window_is_the_hold_wake() -> None:
    gate = ChatGate(policy="class", focus_s=7)
    assert gate.hold_wake_s == 7
    await gate.close()


async def test_an_idle_open_chat_leaves_its_slots_to_tasks() -> None:
    clock = ClockBox()
    gate, ids = gate_with_tasks(clock)
    gate.presence("chat")  # the page opened this chat
    clock.advance(11)
    calls = [gate.submit(task_id) for task_id in ids]
    assert all(admitted(call) for call in calls)
    await gate.close()


async def test_a_chat_typed_in_a_moment_ago_still_takes_its_slot() -> None:
    clock = ClockBox()
    gate, ids = gate_with_tasks(clock)
    gate.presence("chat")
    clock.advance(9)
    calls = [gate.submit(task_id) for task_id in ids]
    assert [admitted(call) for call in calls] == [True, False, False]
    # The window ends on its own: the next decision lets the held ones in.
    clock.advance(2)
    gate.wake()
    assert all(admitted(call) for call in calls)
    await gate.close()


async def test_a_running_turn_takes_its_slot_back_however_long_its_call() -> None:
    clock = ClockBox()
    gate, ids = gate_with_tasks(clock)
    gate.turn_start("chat")
    gate.focus("chat")
    turn = gate.submit("chat")
    assert admitted(turn)
    clock.advance(60)  # a call longer than the focus window
    calls = [gate.submit(task_id) for task_id in ids]
    assert [admitted(call) for call in calls] == [True, False, False]
    gate.finish(turn)
    clock.advance(11)  # the turn ended and nobody typed since
    gate.wake()
    assert all(admitted(call) for call in calls)
    await gate.close()
