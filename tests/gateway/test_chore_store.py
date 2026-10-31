"""The chore store: rows, revisions, plans with jitter, and the migration of an older file."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, date
from pathlib import Path

import pytest

from localmelo.support.gateway.chore_schedule import JITTER_S, Schedule, ScheduleError
from localmelo.support.gateway.chore_store import (
    MAX_CHORES,
    ChoreLimitError,
    ChoreNotFoundError,
    ChoreStaleError,
    ChoreStore,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import (
    COMPONENT,
    SCHEDULER_VERSION,
    TaskStore,
)
from localmelo.support.sessions.store import SessionStore

NOW = 1_785_000_000.0  # 2026-07-25 17:20 UTC, a Saturday
WEEKDAYS = Schedule("weekdays", minute=8 * 60 + 30)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def store(tmp_path: Path, clock: Clock) -> AsyncIterator[SessionStore]:
    opened = SessionStore(tmp_path / "sessions.db", writer="gateway", clock=clock)
    await opened.open()
    await TaskStore(opened, clock=clock).open()
    try:
        yield opened
    finally:
        await opened.aclose()


def make(store: SessionStore, clock: Clock, jitter: float = 0.0) -> ChoreStore:
    return ChoreStore(store, clock=clock, tz=UTC, jitter=lambda: jitter)


async def test_a_chore_is_created_with_its_next_run_and_listed(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock, jitter=42.0)
    chore = await chores.create(
        title="Calendar brief",
        request="Brief me on my calendar",
        schedule=WEEKDAYS,
        front_id="main",
    )
    assert (chore.title, chore.level, chore.enabled, chore.revision) == (
        "Calendar brief",
        2,
        True,
        1,
    )
    # The first run is the schedule's next time plus the jitter, never before it.
    assert chore.next_run == WEEKDAYS.next_after(NOW, chores._tz) + 42.0
    assert chore.last_run is None and chore.last_task_id == ""
    assert [c.chore_id for c in await chores.list_chores()] == [chore.chore_id]
    assert await chores.list_chores("another-chat") == []
    assert [c.chore_id for c in await chores.list_chores("main")] == [chore.chore_id]
    assert await chores.get(chore.chore_id) == chore


async def test_the_jitter_stays_within_its_bounds(
    store: SessionStore, clock: Clock
) -> None:
    chores = ChoreStore(store, clock=clock)  # the real jitter
    nominal = Schedule("hours", hours=1).next_after(NOW)
    runs = [chores.plan(Schedule("hours", hours=1), NOW) for _ in range(200)]
    assert all(nominal <= run <= nominal + JITTER_S for run in runs)
    assert len(set(runs)) > 100  # it does vary


async def test_a_change_needs_the_revision_it_read(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    chore = await chores.create(title="A", request="do a", schedule=WEEKDAYS)
    changed = await chores.update(chore.chore_id, revision=1, title="B")
    assert (changed.title, changed.revision) == ("B", 2)
    with pytest.raises(ChoreStaleError):
        await chores.update(chore.chore_id, revision=1, title="C")
    with pytest.raises(ChoreStaleError):
        await chores.delete(chore.chore_id, revision=1)
    assert (await chores.get(chore.chore_id)).title == "B"
    await chores.delete(chore.chore_id, revision=2)
    with pytest.raises(ChoreNotFoundError):
        await chores.get(chore.chore_id)
    with pytest.raises(ChoreNotFoundError):
        await chores.update(chore.chore_id, revision=3, title="D")


async def test_a_new_time_or_a_switch_back_on_plans_from_now_and_off_clears_it(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    chore = await chores.create(title="A", request="do a", schedule=WEEKDAYS)
    off = await chores.update(chore.chore_id, revision=1, enabled=False)
    assert off.enabled is False and off.next_run is None
    assert await chores.due(NOW + 10 * 86400) == []
    clock.now = NOW + 3 * 86400  # a week-end later
    on = await chores.update(chore.chore_id, revision=2, enabled=True)
    assert on.next_run is not None and on.next_run > clock.now
    moved = await chores.update(
        chore.chore_id, revision=3, schedule=Schedule("daily", minute=60)
    )
    assert moved.schedule == Schedule("daily", minute=60)
    assert moved.next_run == Schedule("daily", minute=60).next_after(
        clock.now, chores._tz
    )
    # Only the title changed: the planned time stays.
    kept = await chores.update(chore.chore_id, revision=4, title="Renamed")
    assert kept.next_run == moved.next_run


async def test_input_is_checked_and_the_file_holds_a_limited_number(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    for bad in (
        {"title": " ", "request": "x"},
        {"title": "t", "request": ""},
        {"title": "t" * 201, "request": "x"},
        {"title": "t", "request": "x", "level": 0},
        {"title": "t", "request": "x", "level": 5},
        {"title": "t", "request": "x", "level": True},
    ):
        with pytest.raises(ValueError):
            await chores.create(schedule=WEEKDAYS, **bad)  # type: ignore[arg-type]
    for index in range(MAX_CHORES):
        await chores.create(title=f"c{index}", request="x", schedule=WEEKDAYS)
    with pytest.raises(ChoreLimitError):
        await chores.create(title="one more", request="x", schedule=WEEKDAYS)


async def test_due_lists_enabled_chores_whose_time_has_come_and_advance_is_a_swap(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    early = await chores.create(
        title="early", request="x", schedule=Schedule("hours", hours=1)
    )
    late = await chores.create(
        title="late", request="x", schedule=Schedule("hours", hours=3)
    )
    assert await chores.due(NOW) == []
    clock.now = NOW + 3600
    assert [c.title for c in await chores.due(clock.now)] == ["early"]
    clock.now = NOW + 4 * 3600
    assert [c.title for c in await chores.due(clock.now)] == ["early", "late"]
    assert await chores.advance(
        early, next_run=clock.now + 3600, task_id="t1", ran_at=clock.now
    )
    # The old view is stale now: a second advance from it changes nothing.
    assert not await chores.advance(early, next_run=0.0, task_id="t2", ran_at=0.0)
    moved = await chores.get(early.chore_id)
    assert (moved.last_task_id, moved.last_run, moved.revision) == ("t1", clock.now, 2)
    assert [c.title for c in await chores.due(clock.now)] == ["late"]
    # A skip moves the time and keeps the last run.
    assert await chores.advance(late, next_run=clock.now + 3600)
    kept = await chores.get(late.chore_id)
    assert (kept.last_task_id, kept.last_run) == ("", None)


async def test_the_checkin_switch_is_on_until_turned_off_and_is_kept(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    assert await chores.checkin_on() is True
    await chores.set_checkin(False)
    assert await chores.checkin_on() is False
    assert await make(store, clock).checkin_on() is False
    await chores.set_checkin(True)
    assert await chores.checkin_on() is True


async def test_an_older_file_gets_the_chore_tables_and_keeps_its_rows(
    tmp_path: Path, clock: Clock
) -> None:
    path = tmp_path / "old.db"
    first = SessionStore(path, writer="gateway", clock=clock)
    await first.open()
    tasks = TaskStore(first, clock=clock)
    await tasks.open()
    record, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "keep me"}),
        origin="user",
        workspace=str(tmp_path),
    )
    # The file as version 5 left it: no chore tables, version row 5.
    async with first.component_transaction("downgrade") as conn:
        await conn.execute("DROP TABLE chores")
        await conn.execute("DROP TABLE chore_meta")
        await conn.execute(
            "UPDATE schema_components SET version = 5 WHERE name = ?", (COMPONENT,)
        )
    await first.aclose()

    second = SessionStore(path, writer="gateway", clock=clock)
    await second.open()
    try:
        migrated = TaskStore(second, clock=clock)
        assert await migrated.open() == SCHEDULER_VERSION == 6
        assert await migrated.get_task(record.task_id) == record
        chores = make(second, clock)
        created = await chores.create(title="A", request="x", schedule=WEEKDAYS)
        assert [c.chore_id for c in await chores.list_chores()] == [created.chore_id]
        # Opening again changes nothing.
        assert await TaskStore(second, clock=clock).open() == SCHEDULER_VERSION
        assert len(await chores.list_chores()) == 1
    finally:
        await second.aclose()


# ── One-time chores (reminders) ──


def once(day: int, hour: int = 9, month: int = 7) -> Schedule:
    return Schedule("once", minute=hour * 60, on=date(2026, month, day))


async def test_a_one_time_chore_is_planned_at_its_own_time_plus_jitter(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock, jitter=42.0)
    schedule = once(27)  # NOW is Saturday 2026-07-25 17:20 UTC
    chore = await chores.create(
        title="Report", request="Reminder: send it", schedule=schedule
    )
    assert chore.next_run == schedule.when(UTC) + 42.0
    assert chore.schedule == schedule and chore.is_reminder
    # Its time has not come: it is not due; at its time plus the jitter it is.
    assert await chores.due(schedule.when(UTC)) == []
    assert [c.chore_id for c in await chores.due(schedule.when(UTC) + 42.0)] == [
        chore.chore_id
    ]


async def test_a_one_time_chore_in_the_past_or_over_a_year_ahead_is_refused(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    far = Schedule("once", minute=0, on=date(2027, 8, 1))
    for bad in (once(25, hour=17), once(1), once(24), far):
        with pytest.raises(ScheduleError):
            await chores.create(title="A", request="Reminder: a", schedule=bad)
    assert await chores.list_chores() == []


async def test_a_one_time_chore_that_ran_stays_in_the_list_switched_off(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    chore = await chores.create(title="A", request="Reminder: a", schedule=once(27))
    clock.now = once(27).when(UTC) + 60
    assert await chores.advance(
        chore, next_run=None, task_id="t1", ran_at=clock.now, enabled=False
    )
    done = await chores.get(chore.chore_id)
    assert (done.enabled, done.next_run, done.last_run, done.last_task_id) == (
        False, None, clock.now, "t1"
    )  # fmt: skip
    assert [c.chore_id for c in await chores.list_chores()] == [chore.chore_id]
    assert await chores.due(clock.now + 86400) == []
    # A stale view does not move it.
    assert not await chores.advance(chore, next_run=None, enabled=False)


async def test_a_done_reminder_is_armed_again_by_a_new_time_but_not_by_a_switch(
    store: SessionStore, clock: Clock
) -> None:
    chores = make(store, clock)
    chore = await chores.create(title="A", request="Reminder: a", schedule=once(27))
    clock.now = once(27).when(UTC) + 60
    await chores.advance(
        chore, next_run=None, task_id="t1", ran_at=clock.now, enabled=False
    )
    done = await chores.get(chore.chore_id)
    # Switching it on would run a time that has passed: refused, nothing changes.
    with pytest.raises(ScheduleError, match="passed"):
        await chores.update(chore.chore_id, revision=done.revision, enabled=True)
    assert (await chores.get(chore.chore_id)).enabled is False
    # A new time arms it again.
    new = once(28, hour=8)
    again = await chores.update(chore.chore_id, revision=done.revision, schedule=new)
    assert (again.enabled, again.next_run, again.schedule) == (True, new.when(UTC), new)
    # A title change keeps a pending reminder's planned time, even one that is late.
    clock.now = new.when(UTC) + 3 * 3600
    kept = await chores.update(chore.chore_id, revision=again.revision, title="Renamed")
    assert kept.next_run == again.next_run
