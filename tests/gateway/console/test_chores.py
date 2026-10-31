"""Chores in the console: the API, the clock that starts them, and the check-in.

Every test runs a real console with the main chat's scripted model and a fake
clock the test moves by hand; the chore loop's beat is set so long that only
``tick()`` ever runs it. Workers never start (the scheduler's start is a
no-op), so a started chore is a queued task and no request reaches a model.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support.gateway.chore_schedule import Schedule
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.chores import CHECKIN_S, ChoreOptions
from localmelo.support.gateway.console.front import ECHO_EVENT
from localmelo.tests.sessions.helpers import ScriptedLLM, answer

from .conftest import HOLDER, run_result, seed_turn
from .test_front import _say, front_console
from .test_turn_writes import _wait_turn

HOUR = 3600.0
DAY = 86400.0


def at(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp()


class Clock:
    """2026-10-28 is a Wednesday; the test moves the time."""

    def __init__(self, start: str = "2026-10-28 07:00") -> None:
        self.now = at(start)

    def __call__(self) -> float:
        return self.now

    def to(self, text: str) -> None:
        self.now = at(text)


class Summary:
    """The check-in's summary, counting its calls."""

    def __init__(self, note: str = "All quiet.", fail: bool = False) -> None:
        self.note, self.fail = note, fail
        self.facts: list[list[str]] = []

    async def __call__(self, facts: list[str]) -> str:
        self.facts.append(facts)
        if self.fail:
            raise RuntimeError("no model")
        return self.note


@contextlib.asynccontextmanager
async def chore_console(
    tmp_path: Path,
    *replies: Any,
    jitter: float = 0.0,
    summary: Summary | None = None,
    tick_s: float = 1e9,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, Clock, Summary, ScriptedLLM]]:
    clock, summarize = Clock(), summary or Summary()
    options = ChoreOptions(
        clock=clock, tz=UTC, jitter=lambda: jitter, tick_s=tick_s, summarize=summarize
    )
    async with front_console(tmp_path, *replies, chores=options) as (
        console,
        client,
        llm,
    ):
        # The first check-in is half an hour after the gateway starts ...
        assert console.chore_runner.next_checkin == clock.now + CHECKIN_S
        # ... and a test gets none until it asks: moving the clock never starts one by accident.
        console.chore_runner.next_checkin = 1e12
        yield console, client, clock, summarize, llm


async def add(
    client: httpx.AsyncClient, request: str, schedule: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    response = await client.post(
        "/api/chores", json={"request": request, "schedule": schedule, **extra}
    )
    assert response.status_code == 201, response.text
    chore: dict[str, Any] = response.json()["chore"]
    return chore


BRIEF = {"kind": "weekdays", "at": "08:30"}


async def tasks_of(console: Console) -> list[Any]:
    return await console.workers.tasks.list_tasks()


async def set_state(console: Console, task_id: str, state: str, at: float) -> None:
    """Put a task in *state* as of *at*, as the scheduler's own transitions would leave it."""
    ended = at if state in ("done", "failed", "cancelled", "expired") else None
    async with console.store.component_transaction("test_state") as conn:
        await conn.execute(
            "UPDATE tasks SET state = ?, started_at = COALESCE(started_at, ?), "
            "finished_at = ?, updated_at = ? WHERE task_id = ?",
            (state, at, ended, at, task_id),
        )


async def finish(
    console: Console, task_id: str, at: float, text: str = "Three meetings today."
) -> None:
    """End a task as done with a one-turn answer, as a worker would."""
    store = console.store
    turn = await store.begin_turn(task_id, "x", holder=HOLDER)
    await store.finish_turn(
        task_id, turn.turn, holder=HOLDER, status="completed", cause="run",
        result=run_result(), answer=text,
    )  # fmt: skip
    await set_state(console, task_id, "done", at)


async def echoes(console: Console, kind: str) -> list[dict[str, Any]]:
    assert console.main_id is not None
    found = await console.store.events_of_type(console.main_id, ECHO_EVENT)
    return [e.event["data"] for e in found if e.event["data"]["echo"] == kind]


# ── The API ──


async def test_a_chore_is_created_listed_changed_and_deleted(tmp_path: Path) -> None:
    async with chore_console(tmp_path, jitter=30.0) as (console, client, clock, _, _):
        chore = await add(
            client, "Brief me on my calendar", BRIEF, title="Calendar brief"
        )
        assert (
            chore["title"],
            chore["when"],
            chore["level_word"],
            chore["enabled"],
        ) == (
            "Calendar brief",
            "every weekday at 8:30",
            "Normal",
            True,
        )
        assert chore["next_run"] == at("2026-10-28 08:30") + 30
        assert chore["front_id"] == console.main_id
        assert chore["last"] is None and chore["last_run"] is None
        # The title is the request's first clause when none is given.
        derived = await add(
            client, "Water the plants. Then rest.", {"kind": "hours", "every": 6}
        )
        assert derived["title"] == "Water the plants"

        listed = (await client.get("/api/chores")).json()
        assert [c["id"] for c in listed["chores"]] == [chore["id"], derived["id"]]
        assert listed["scope"] == "all"
        assert listed["checkin"] == {
            "on": True,
            "every_min": 30,
            "next_at": console.chore_runner.next_checkin,
            "last_at": None,
        }

        moved = await client.patch(
            f"/api/chores/{chore['id']}",
            json={
                "revision": chore["revision"],
                "schedule": {"kind": "daily", "at": "21:00"},
                "level": "Today",
            },
        )
        body = moved.json()["chore"]
        assert (moved.status_code, body["when"], body["level"], body["revision"]) == (
            200,
            "every day at 21:00",
            1,
            2,
        )
        assert body["next_run"] == at("2026-10-28 21:00") + 30

        stale = await client.patch(
            f"/api/chores/{chore['id']}", json={"revision": 1, "title": "Other"}
        )
        assert (
            stale.status_code == 409
            and stale.json()["error"]["code"] == "stale_revision"
        )
        stale_delete = await client.delete(f"/api/chores/{chore['id']}?revision=1")
        assert stale_delete.status_code == 409
        assert (
            await client.delete(f"/api/chores/{chore['id']}?revision=2")
        ).status_code == 200
        gone = await client.delete(f"/api/chores/{chore['id']}")
        assert gone.status_code == 404 and gone.json()["error"]["code"] == "not_found"
        # Without a revision a delete is not checked.
        assert (await client.delete(f"/api/chores/{derived['id']}")).status_code == 200
        assert (await client.get("/api/chores")).json()["chores"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"request": "Brief me", "schedule": {"kind": "weekly", "at": "08:30"}},
        {"request": "Brief me", "schedule": {"kind": "daily", "at": "8:30pm"}},
        {"request": "   ", "schedule": BRIEF},
        {"request": "Brief me"},
        {"request": "Brief me", "schedule": BRIEF, "level": "Soon"},
        {"request": "Brief me", "schedule": BRIEF, "level": "Later"},
        {"request": "Brief me", "schedule": BRIEF, "level": "Now"},
        {"request": "Brief me", "schedule": BRIEF, "level": 0},
        {"request": "Brief me", "schedule": BRIEF, "level": "P0"},
        {"request": "Brief me", "schedule": BRIEF, "level": "P5"},
        {"request": "Brief me", "schedule": BRIEF, "front_id": "no-such-chat"},
        {"request": "Brief me", "schedule": BRIEF, "origin": "user"},
        {"request": "x" * 100_001, "schedule": BRIEF},
    ],
)
async def test_the_api_refuses_what_is_not_a_chore(
    tmp_path: Path, body: dict[str, Any]
) -> None:
    async with chore_console(tmp_path) as (_, client, _, _, _):
        response = await client.post("/api/chores", json=body)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"
        if "level" in body:
            assert response.json()["error"]["message"] == "level: P1 to P4"
        assert (await client.get("/api/chores")).json()["chores"] == []


async def test_a_chore_takes_its_level_as_a_code_or_a_word(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (_, client, _, _, _):
        chore = await add(client, "Brief me", BRIEF, level="P3")
        # The API still answers the word beside the number.
        assert (chore["level"], chore["level_word"]) == (3, "Whenever")
        for value, level in (("Today", 1), ("p4", 4), ("normal", 2), ("1", 1)):
            sent = await client.patch(
                f"/api/chores/{chore['id']}",
                json={"revision": chore["revision"], "level": value},
            )
            chore = sent.json()["chore"]
            assert chore["level"] == level


async def test_a_change_names_a_revision_and_something_to_change(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (_, client, _, _, _):
        chore = await add(client, "Brief me", BRIEF)
        path = f"/api/chores/{chore['id']}"
        assert (await client.patch(path, json={"revision": 1})).status_code == 400
        assert (await client.patch(path, json={"title": "x"})).status_code == 400
        assert (
            await client.patch(path, json={"revision": 1, "enabled": "yes"})
        ).status_code == 400
        missing = await client.patch(
            "/api/chores/nope", json={"revision": 1, "title": "x"}
        )
        assert missing.status_code == 404


async def test_a_side_chat_lists_its_own_chores_and_the_main_chat_all(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, _):
        side = (await seed_turn(console.store, str(tmp_path))).session_id
        mine = await add(client, "Main chore", BRIEF)
        theirs = await add(client, "Side chore", BRIEF, front_id=side)
        assert theirs["front_id"] == side
        everything = (
            await client.get(f"/api/chores?session_id={console.main_id}")
        ).json()
        assert [c["id"] for c in everything["chores"]] == [mine["id"], theirs["id"]]
        own = (await client.get(f"/api/chores?session_id={side}")).json()
        assert (own["scope"], [c["id"] for c in own["chores"]]) == (
            "chat",
            [theirs["id"]],
        )
        assert (await client.get("/api/chores?session_id=gone")).status_code == 404


async def test_the_checkin_switch_is_kept(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (_, client, _, _, _):
        off = await client.put("/api/checkin", json={"on": False})
        assert (
            off.json()["checkin"]["on"] is False
            and off.json()["checkin"]["next_at"] is None
        )
        assert (await client.get("/api/chores")).json()["checkin"]["on"] is False
        assert (await client.put("/api/checkin", json={"on": "no"})).status_code == 400
        on = await client.put("/api/checkin", json={"on": True})
        assert (
            on.json()["checkin"]["on"] is True
            and on.json()["checkin"]["next_at"] is not None
        )


# ── The clock ──


async def test_a_due_chore_starts_one_user_rule_task_and_plans_its_next_run(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, llm):
        chore = await add(
            client,
            "Brief me on my calendar",
            BRIEF,
            title="Calendar brief",
            level="Today",
        )
        runner = console.chore_runner
        assert await runner.tick() == []
        clock.to("2026-10-28 08:29")
        assert await runner.tick() == [] and await tasks_of(console) == []

        clock.to("2026-10-28 08:31")
        [task_id] = await runner.tick()
        [record] = await tasks_of(console)
        assert record.task_id == task_id
        assert (record.origin, record.front_id, record.title, record.level) == (
            "user_rule", console.main_id, "Calendar brief", 1
        )  # fmt: skip
        assert record.spec["objective"] == "Brief me on my calendar"
        assert record.reason == "Chore, every weekday at 8:30"
        assert (
            record.request_key == f"chore:{chore['id']}:{int(at('2026-10-28 08:30'))}"
        )

        after = (await client.get("/api/chores")).json()["chores"][0]
        assert after["next_run"] == at("2026-10-29 08:30")
        assert (after["last_run"], after["last"]["task_id"], after["last"]["state"]) == (
            clock.now, task_id, "queued"
        )  # fmt: skip
        # A second tick in the same minute starts nothing, and the model was never called.
        assert await runner.tick() == [] and len(await tasks_of(console)) == 1
        assert llm.requests == []


async def test_a_run_waits_for_its_jitter_and_never_starts_early(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, jitter=90.0) as (console, client, clock, _, _):
        await add(client, "Brief me", BRIEF)
        clock.to("2026-10-28 08:31")  # 60 s after 8:30: inside the jitter
        assert await console.chore_runner.tick() == []
        clock.to("2026-10-28 08:32")  # past 8:31:30
        assert len(await console.chore_runner.tick()) == 1


async def test_a_weekday_chore_does_not_run_on_the_weekend(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        await add(client, "Brief me", BRIEF)
        runner = console.chore_runner
        clock.to("2026-10-30 08:31")  # Friday
        assert len(await runner.tick()) == 1
        chore = (await client.get("/api/chores")).json()["chores"][0]
        assert chore["next_run"] == at("2026-11-02 08:30")  # Monday
        [friday] = await runner.store.list_chores()
        assert friday.last_task_id
        await finish(console, friday.last_task_id, clock.now)
        for stamp in ("2026-10-31 08:31", "2026-11-01 08:31", "2026-11-02 08:29"):
            clock.to(stamp)
            assert await runner.tick() == []
        clock.to("2026-11-02 08:31")
        assert len(await runner.tick()) == 1


async def test_a_run_missed_while_the_computer_slept_runs_once_at_wake(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        await add(client, "Brief me", {"kind": "daily", "at": "08:30"})
        await add(client, "Check disk", {"kind": "hours", "every": 2})
        runner = console.chore_runner
        # Asleep from Wednesday 07:00 until Saturday 15:10: three morning slots and many 2-hour slots.
        clock.to("2026-10-31 15:10")
        assert len(await runner.tick()) == 2
        assert len(await tasks_of(console)) == 2
        assert await runner.tick() == []
        chores = {
            c["title"]: c for c in (await client.get("/api/chores")).json()["chores"]
        }
        assert chores["Brief me"]["next_run"] == at("2026-11-01 08:30")
        assert chores["Check disk"]["next_run"] == clock.now + 2 * HOUR


async def test_a_chore_switched_off_never_runs_and_switching_on_plans_from_now(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        chore = await add(client, "Brief me", BRIEF)
        off = await client.patch(
            f"/api/chores/{chore['id']}", json={"revision": 1, "enabled": False}
        )
        assert off.json()["chore"]["next_run"] is None
        clock.to("2026-11-05 12:00")
        assert await console.chore_runner.tick() == []
        on = await client.patch(
            f"/api/chores/{chore['id']}", json={"revision": 2, "enabled": True}
        )
        assert on.json()["chore"]["next_run"] == at("2026-11-06 08:30")
        assert await console.chore_runner.tick() == []  # the missed days are not run


async def test_a_chore_whose_last_run_has_not_ended_skips_its_next_slot(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        await add(client, "Brief me", {"kind": "daily", "at": "08:30"})
        runner = console.chore_runner
        clock.to("2026-10-28 08:31")
        [first] = await runner.tick()
        clock.to("2026-10-29 08:31")  # the task is still queued: no pile-up
        assert await runner.tick() == []
        assert len(await tasks_of(console)) == 1
        assert (await client.get("/api/chores")).json()["chores"][0]["next_run"] == at(
            "2026-10-30 08:30"
        )
        await finish(console, first, clock.now)
        clock.to("2026-10-30 08:31")
        assert len(await runner.tick()) == 1
        assert len(await tasks_of(console)) == 2


async def test_a_start_that_died_before_the_row_moved_does_not_run_twice(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        chore = await add(client, "Brief me", BRIEF)
        runner = console.chore_runner
        clock.to("2026-10-28 08:31")
        row = await runner.store.get(chore["id"])
        # The task was made, then the gateway stopped before the chore moved on.
        first = await runner._start(
            row, f"chore:{row.chore_id}:{int(row.next_run or 0)}"
        )
        [again] = await runner.tick()
        assert again == first and len(await tasks_of(console)) == 1


async def test_run_now_starts_a_task_and_leaves_the_schedule(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        chore = await add(client, "Brief me", BRIEF, title="Calendar brief")
        clock.to("2026-10-28 07:30")
        response = await client.post(f"/api/chores/{chore['id']}/run")
        assert response.status_code == 202, response.text
        body = response.json()
        assert (
            body["task"]["origin"] == "user_rule"
            and body["task"]["title"] == "Calendar brief"
        )
        assert body["chore"]["next_run"] == chore["next_run"]
        assert (body["chore"]["last_run"], body["chore"]["last"]["task_id"]) == (
            clock.now,
            body["task"]["task_id"],
        )
        assert (await client.post("/api/chores/nope/run")).status_code == 404


async def test_the_loop_runs_the_tick_on_its_beat_and_stops_with_the_console(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, tick_s=0.02) as (console, client, clock, _, _):
        await add(client, "Brief me", BRIEF)
        clock.to("2026-10-28 08:31")
        for _ in range(250):
            if await tasks_of(console):
                break
            await asyncio.sleep(0.02)
        assert len(await tasks_of(console)) == 1
    assert (
        console.chore_runner._loop_task is not None
        and console.chore_runner._loop_task.done()
    )


# ── The check-in ──


async def beat(console: Console, clock: Clock) -> None:
    """Move a whole beat on, run the tick that starts the check-in, and let it finish."""
    runner = console.chore_runner
    clock.now += CHECKIN_S + 1
    runner.next_checkin = clock.now - 1
    await runner.tick()
    if runner.checkin_task is not None:
        await runner.checkin_task


async def test_the_checkin_comes_once_every_half_hour(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        runner = console.chore_runner
        runner.next_checkin = clock.now + CHECKIN_S
        await runner.tick()
        assert runner.checkin_task is None and runner.last_checkin is None
        clock.now += CHECKIN_S - 1
        await runner.tick()
        assert runner.checkin_task is None
        clock.now += 1
        await runner.tick()
        assert runner.checkin_task is not None
        await runner.checkin_task
        assert runner.last_checkin == clock.now
        assert runner.next_checkin == clock.now + CHECKIN_S
        body = (await client.get("/api/chores")).json()["checkin"]
        assert (body["next_at"], body["last_at"]) == (runner.next_checkin, clock.now)


async def test_a_checkin_with_nothing_due_makes_no_model_call_and_no_card(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, summary, llm):
        await add(
            client, "Brief me", {"kind": "daily", "at": "08:30"}
        )  # not yet run: nothing to report
        for _ in range(3):
            await beat(console, clock)
        assert summary.facts == [] and llm.requests == []
        assert await echoes(console, "checkin") == []
        assert (await client.get("/api/chores")).json()["checkin"][
            "last_at"
        ] is not None


async def test_a_finished_chore_run_is_summarized_once_in_one_card(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, summary=Summary("Your calendar brief is ready.")) as (
        console, client, clock, summary, llm
    ):  # fmt: skip
        chore = await add(
            client,
            "Brief me on my calendar",
            {"kind": "daily", "at": "08:30"},
            title="Calendar brief",
        )
        clock.to("2026-10-28 08:31")
        [task_id] = await console.chore_runner.tick()
        await finish(
            console, task_id, clock.now, "Three meetings today. The first is at ten."
        )
        await beat(console, clock)

        assert summary.facts == [["Calendar brief: Three meetings today."]]
        [card] = await echoes(console, "checkin")
        assert (card["line"], card["items"]) == ("Your calendar brief is ready.", 1)
        row = await console.chore_runner.store.get(chore["id"])
        assert row.reported_task_id == task_id
        # Nothing new: the next beats say nothing and call nothing.
        await beat(console, clock)
        assert len(summary.facts) == 1 and len(await echoes(console, "checkin")) == 1
        assert llm.requests == []


async def test_a_task_that_waits_for_the_person_is_mentioned_after_a_whole_beat(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, summary=Summary("Slides waits for you.")) as (
        console, client, clock, summary, _
    ):  # fmt: skip
        task, _ = await console.workers.create(
            {"objective": "Make slides", "title": "Slides"}
        )
        await set_state(console, task["task_id"], "waiting", clock.now)
        # Just began waiting: the approval card already told the person, so not yet.
        await console.chore_runner.checkin()
        assert summary.facts == []
        clock.now += CHECKIN_S + 1
        card = await console.chore_runner.checkin()
        assert card is not None and card["line"] == "Slides waits for you."
        assert summary.facts == [["Slides: waiting for the user"]]
        # Mentioned once; the next check-in says nothing.
        clock.now += CHECKIN_S
        assert await console.chore_runner.checkin() is None


async def test_the_checkin_switch_off_posts_nothing_and_keeps_results_for_later(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, summary, _):
        await add(
            client, "Brief me", {"kind": "daily", "at": "08:30"}, title="Calendar brief"
        )
        clock.to("2026-10-28 08:31")
        [task_id] = await console.chore_runner.tick()
        await finish(console, task_id, clock.now)
        await client.put("/api/checkin", json={"on": False})
        await beat(console, clock)
        assert summary.facts == [] and await echoes(console, "checkin") == []
        await client.put("/api/checkin", json={"on": True})
        await beat(console, clock)
        assert len(await echoes(console, "checkin")) == 1


async def test_when_the_model_cannot_write_the_note_code_does(tmp_path: Path) -> None:
    async with chore_console(tmp_path, summary=Summary(fail=True)) as (
        console,
        client,
        clock,
        _,
        _,
    ):
        await add(
            client, "Brief me", {"kind": "daily", "at": "08:30"}, title="Calendar brief"
        )
        clock.to("2026-10-28 08:31")
        [task_id] = await console.chore_runner.tick()
        await finish(console, task_id, clock.now)
        await beat(console, clock)
        [card] = await echoes(console, "checkin")
        assert card["line"] == "Calendar brief: Three meetings today."


async def test_the_note_is_one_short_line(tmp_path: Path) -> None:
    async with chore_console(tmp_path, summary=Summary("word " * 300)) as (
        console,
        client,
        clock,
        _,
        _,
    ):
        await add(client, "Brief me", {"kind": "daily", "at": "08:30"})
        clock.to("2026-10-28 08:31")
        [task_id] = await console.chore_runner.tick()
        await finish(console, task_id, clock.now)
        await beat(console, clock)
        [card] = await echoes(console, "checkin")
        assert len(card["line"]) <= 400 and card["line"].endswith("…")


async def test_the_default_note_is_one_call_to_the_chat_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = ScriptedLLM([answer("  Your brief\nis ready.  ")])
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    llm.close = close  # type: ignore[method-assign]

    class Backend:
        def build_chat_provider(self, cfg: Any) -> ScriptedLLM:
            built.append(cfg)
            return llm

    built: list[Any] = []
    monkeypatch.setattr(
        "localmelo.support.backends.registry.get_backend", lambda name: Backend()
    )
    clock = Clock()
    options = ChoreOptions(clock=clock, tz=UTC, jitter=lambda: 0.0, tick_s=1e9)
    async with front_console(tmp_path, chores=options) as (console, client, _):
        console.chore_runner.next_checkin = (
            1e12  # this test asks for the check-in itself
        )
        forgotten: list[str] = []
        real_forget = console.gate.forget

        def spy(session_id: str) -> None:
            forgotten.append(session_id)
            real_forget(session_id)

        console.gate.forget = spy  # type: ignore[method-assign]
        await add(
            client, "Brief me", {"kind": "daily", "at": "08:30"}, title="Calendar brief"
        )
        clock.to("2026-10-28 08:31")
        [task_id] = await console.chore_runner.tick()
        await finish(console, task_id, clock.now)
        card = await console.chore_runner.checkin()
        assert card is not None and card["line"] == "Your brief is ready."
        [request] = llm.requests
        assert [m.role for m in request] == ["system", "user"]
        assert "Calendar brief: Three meetings today." in request[1].content
        assert closed == [True] and len(built) == 1
        # The note's own gate session is dropped when the call is done.
        assert len(forgotten) == 1
        assert forgotten[0].startswith(f"checkin:{console.main_id}:")


# ── The main chat: a repeating phrase proposes a chore ──


async def tap(client: httpx.AsyncClient, echo_id: int, do: str) -> httpx.Response:
    return await client.post(f"/api/front/echoes/{echo_id}", json={"do": do})


async def test_a_repeating_phrase_proposes_a_chore_and_saves_nothing(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, llm):
        response = await _say(
            client, console, "every weekday at 8:30, brief me on my calendar"
        )
        assert response.status_code == 200, response.text
        echo = response.json()["echo"]
        assert (
            echo["echo"] == "chore"
            and echo["line"] == "Run this every weekday at 8:30?"
        )
        assert echo["typed"] == "every weekday at 8:30, brief me on my calendar"
        assert (echo["title"], echo["text"], echo["when"], echo["level"]) == (
            "brief me on my calendar", "brief me on my calendar", "every weekday at 8:30", 2
        )  # fmt: skip
        assert echo["schedule"] == {"kind": "weekdays", "at": "08:30"}
        # No model call for the schedule, and nothing is saved or started without the tap.
        assert llm.requests == []
        assert (await client.get("/api/chores")).json()["chores"] == []
        assert await tasks_of(console) == []


async def test_a_chinese_phrase_is_read_the_same_way(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, llm):
        echo = (await _say(client, console, "每天早上九点 总结我的邮件")).json()["echo"]
        assert (echo["line"], echo["text"], echo["schedule"]) == (
            "Run this every day at 9:00?", "总结我的邮件", {"kind": "daily", "at": "09:00"}
        )  # fmt: skip
        assert llm.requests == []


async def test_save_chore_keeps_it_and_undo_removes_it(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, _):
        proposal = (
            await _say(client, console, "every day at 21:00 urgent: back up notes.md")
        ).json()["echo"]
        assert proposal["level"] == 1  # an urgency word in the work, as for a task
        saved = await tap(client, proposal["id"], "save")
        assert saved.status_code == 200, saved.text
        card = saved.json()["echo"]
        assert card["echo"] == "chore_saved" and card["of"] == proposal["id"]
        assert card["line"] == "Saved: urgent: back up notes.md · every day at 21:00"
        [chore] = (await client.get("/api/chores")).json()["chores"]
        assert (chore["when"], chore["level"], chore["front_id"], chore["id"]) == (
            "every day at 21:00", 1, console.main_id, card["chore_id"]
        )  # fmt: skip
        # A second tap on the same card is refused.
        again = await tap(client, proposal["id"], "save")
        assert (
            again.status_code == 409
            and again.json()["error"]["code"] == "already_answered"
        )
        undone = await tap(client, card["id"], "undo")
        assert undone.json()["echo"]["echo"] == "undone"
        assert (
            undone.json()["echo"]["line"]
            == "Undone: urgent: back up notes.md is not saved"
        )
        assert (await client.get("/api/chores")).json()["chores"] == []


async def test_just_once_runs_it_as_a_task_and_saves_no_chore(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, _):
        proposal = (
            await _say(
                client, console, "every weekday at 8:30, brief me on my calendar"
            )
        ).json()["echo"]
        started = await tap(client, proposal["id"], "once")
        assert started.status_code == 200, started.text
        echo = started.json()["echo"]
        assert (echo["echo"], echo["of"]) == ("started", proposal["id"])
        [record] = await tasks_of(console)
        assert (record.origin, record.spec["objective"]) == (
            "user",
            "brief me on my calendar",
        )
        assert (await client.get("/api/chores")).json()["chores"] == []
        assert (await tap(client, proposal["id"], "cancel")).status_code == 409


async def test_cancel_saves_and_starts_nothing(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, _):
        proposal = (
            await _say(
                client, console, "every weekday at 8:30, brief me on my calendar"
            )
        ).json()["echo"]
        done = await tap(client, proposal["id"], "cancel")
        assert done.json()["echo"]["line"] == "Not saved: brief me on my calendar"
        assert (await client.get("/api/chores")).json()[
            "chores"
        ] == [] and await tasks_of(console) == []
        # A save and an undo only fit their own lines.
        other = (await _say(client, console, "daily at 9 check the news")).json()[
            "echo"
        ]
        assert (await tap(client, other["id"], "undo")).status_code == 400
        assert (await tap(client, other["id"], "apply")).status_code in (400, 404)


async def test_other_messages_reach_the_model_as_before(tmp_path: Path) -> None:
    messages = [
        "hello",
        "make slides from notes.md",
        # Repeating, but code does not read the schedule: the model takes it, as before.
        "every Monday at 9 write a report",
        "every 30 minutes ping the server",
        # A question about a schedule is a question.
        "what happens every day at noon?",
        "every day at 9 is it worth it?",
    ]
    async with chore_console(tmp_path, *[answer("ok")] * len(messages)) as (
        console,
        client,
        _,
        _,
        llm,
    ):
        for text in messages:
            response = await _say(client, console, text)
            assert response.status_code == 202, (text, response.text)
            await _wait_turn(client, str(console.main_id))
        assert len(llm.requests) == len(messages)
        assert (await client.get("/api/chores")).json()["chores"] == []
        assert await echoes(console, "chore") == []


async def test_a_polite_request_with_a_schedule_is_a_proposal_but_an_attachment_is_not(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, llm):
        echo = (
            await _say(client, console, "every day at 9, can you check the news?")
        ).json()["echo"]
        assert (echo["echo"], echo["text"]) == ("chore", "can you check the news?")
        # An attached file is one-time: the message is not turned into a chore.
        front = console.front
        assert (
            await front.route(
                "every day at 9 check the news",
                "every day at 9 check the news\n[file]",
                as_task=False,
                key=None,
            )
            is None
        )
        assert llm.requests == []


# ── Reminders: a one-time chore ──

# The test clock is Wednesday 2026-10-28 07:00 in UTC (the chore options' zone).
REMIND = "remind me to send the report on Friday at 9"
FRIDAY = {"kind": "once", "at": "2026-10-30T09:00"}


async def remind(client: httpx.AsyncClient, console: Console, text: str) -> Any:
    response = await _say(client, console, text)
    assert response.status_code == 200, response.text
    return response.json()["echo"]


async def test_a_reminder_phrase_proposes_a_one_time_chore_with_the_absolute_time(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, llm):
        echo = await remind(client, console, REMIND)
        assert (echo["echo"], echo["reminder"], echo["typed"]) == (
            "chore",
            True,
            REMIND,
        )
        assert echo["line"] == "Remind you on Fri Oct 30 at 9:00?"
        assert echo["said"] == {
            "key": "front.echo.remind",
            "params": {"at": Schedule.from_json(FRIDAY).said_at()},
        }
        assert (echo["title"], echo["work"], echo["text"]) == (
            "send the report", "send the report", "Reminder: send the report"
        )  # fmt: skip
        assert echo["schedule"] == FRIDAY and echo["when"] == "once, Fri Oct 30 at 9:00"
        assert "level" not in echo  # a reminder shows no level chip
        # No model call, and nothing is saved or started before the tap.
        assert llm.requests == []
        assert (await client.get("/api/chores")).json()["chores"] == []
        assert await tasks_of(console) == []


@pytest.mark.parametrize(
    ("text", "at"),
    [
        ("Remind me tomorrow at 9am to call mom", "2026-10-29T09:00"),
        ("remind me in 2 hours to drink water", "2026-10-28T09:00"),
        ("remind me at 6:30 to stretch", "2026-10-28T18:30"),
        ("提醒我明天九点给导师发邮件", "2026-10-29T09:00"),
        ("记得明天九点给导师发邮件", "2026-10-29T09:00"),
        ("两小时后提醒我喝水", "2026-10-28T09:00"),
        ("提醒我下周一交作业", "2026-11-02T09:00"),
    ],
)
async def test_relative_times_become_absolute_in_the_persons_time_zone(
    tmp_path: Path, text: str, at: str
) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, llm):
        echo = await remind(client, console, text)
        assert (echo["echo"], echo["schedule"]) == ("chore", {"kind": "once", "at": at})
        assert llm.requests == []


async def test_a_reminder_without_a_time_asks_when_and_the_next_message_answers(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, answer("ok")) as (console, client, _, _, llm):
        ask = await remind(client, console, "remind me to send the report")
        assert (ask["echo"], ask["ask"], ask["work"]) == (
            "ask",
            "when",
            "send the report",
        )
        assert ask["said"]["key"] == "front.echo.askWhen" and ask["line"].startswith(
            "When should I remind you?"
        )
        assert (await client.get("/api/chores")).json()["chores"] == []
        # The answer is only a time: the reminder is proposed, tied to the question.
        echo = await remind(client, console, "Friday at 9")
        assert (echo["echo"], echo["of"], echo["work"]) == (
            "chore",
            ask["id"],
            "send the report",
        )
        assert echo["schedule"] == FRIDAY and echo["typed"] == "Friday at 9"
        assert llm.requests == []
        # The question is answered once: a time sent later is not tied to it.
        later = await _say(client, console, "Friday at 9")
        assert later.status_code == 202
        await _wait_turn(client, str(console.main_id))
        assert len(llm.requests) == 1


async def test_a_time_that_has_passed_asks_again_and_a_reply_that_is_not_a_time_goes_on(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, answer("ok")) as (console, client, _, _, llm):
        ask = await remind(client, console, "remind me today at 6am to call mom")
        assert ask["ask"] == "when" and ask["said"]["key"] == "front.echo.askWhenPast"
        again = await remind(client, console, "remind me on 2028-01-05 to renew it")
        assert again["said"]["key"] == "front.echo.askWhenFar"
        # Not a time: the message is an ordinary one for the model.
        reply = await _say(client, console, "never mind, what is a reminder?")
        assert reply.status_code == 202
        await _wait_turn(client, str(console.main_id))
        assert len(llm.requests) == 1


async def test_saving_a_reminder_keeps_a_one_time_chore_and_undo_removes_it(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        proposal = await remind(client, console, REMIND)
        saved = await tap(client, proposal["id"], "save")
        assert saved.status_code == 200, saved.text
        card = saved.json()["echo"]
        assert card["echo"] == "chore_saved" and card["of"] == proposal["id"]
        assert card["line"] == "Saved: send the report · once, Fri Oct 30 at 9:00"
        [chore] = (await client.get("/api/chores")).json()["chores"]
        assert (chore["schedule"], chore["when"], chore["level"], chore["enabled"]) == (
            FRIDAY, "once, Fri Oct 30 at 9:00", 1, True
        )  # fmt: skip
        assert (chore["request"], chore["title"]) == (
            "Reminder: send the report",
            "send the report",
        )
        assert chore["next_run"] == at("2026-10-30 09:00") and chore["late_min"] is None
        assert (await tap(client, card["id"], "undo")).json()["echo"][
            "echo"
        ] == "undone"
        assert (await client.get("/api/chores")).json()["chores"] == []
        # "Just once" does not fit a reminder, and a tap after its time is refused in words.
        other = await remind(client, console, REMIND)
        assert (await tap(client, other["id"], "once")).status_code == 400
        clock.to("2026-10-30 09:30")
        late = await tap(client, other["id"], "save")
        assert late.status_code == 400 and "passed" in late.json()["error"]["message"]
        assert (await client.get("/api/chores")).json()[
            "chores"
        ] == [] and await tasks_of(console) == []


async def test_cancel_saves_no_reminder(tmp_path: Path) -> None:
    async with chore_console(tmp_path) as (console, client, _, _, _):
        proposal = await remind(client, console, REMIND)
        done = await tap(client, proposal["id"], "cancel")
        assert done.json()["echo"]["line"] == "Not saved: send the report"
        assert (await client.get("/api/chores")).json()["chores"] == []


async def test_a_reminder_runs_once_at_its_time_as_a_task_then_is_done(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, llm):
        proposal = await remind(client, console, REMIND)
        await tap(client, proposal["id"], "save")
        runner = console.chore_runner
        clock.to("2026-10-30 08:59")
        assert await runner.tick() == []
        clock.to("2026-10-30 09:00")
        [task_id] = await runner.tick()
        [record] = await tasks_of(console)
        assert record.task_id == task_id
        assert (record.origin, record.title, record.level, record.reason) == (
            "user_rule", "send the report", 1, "Reminder"
        )  # fmt: skip
        assert record.spec["objective"].startswith("Reminder: send the report\n\n")
        assert "Do not do the task yourself" in record.spec["objective"]
        [done] = (await client.get("/api/chores")).json()["chores"]
        # Done: switched off, no next time, kept in the list with its last run.
        assert (done["enabled"], done["next_run"], done["last_run"]) == (
            False,
            None,
            clock.now,
        )
        assert done["last"]["task_id"] == task_id and done["late_min"] is None
        # It never runs again, whatever the days after.
        for stamp in ("2026-10-30 09:30", "2026-11-06 09:00", "2027-01-01 09:00"):
            clock.to(stamp)
            assert await runner.tick() == []
        assert len(await tasks_of(console)) == 1 and llm.requests == []


async def test_a_reminder_missed_while_the_computer_slept_runs_at_wake_and_says_it_ran_late(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        proposal = await remind(client, console, REMIND)
        await tap(client, proposal["id"], "save")
        clock.to("2026-10-30 11:00")  # two hours after its time
        assert len(await console.chore_runner.tick()) == 1
        [done] = (await client.get("/api/chores")).json()["chores"]
        assert (done["enabled"], done["late_min"]) == (False, 120)
        # A reminder that waits for its jitter is not late.
        clock.to("2026-10-30 11:00")
        await add(
            client, "Reminder: stand up", {"kind": "once", "at": "2026-10-30T11:30"}
        )
        clock.to("2026-10-30 11:31")
        assert len(await console.chore_runner.tick()) == 1
        bodies = {
            c["request"]: c for c in (await client.get("/api/chores")).json()["chores"]
        }
        assert bodies["Reminder: stand up"]["late_min"] is None


async def test_a_reminder_runs_even_when_an_earlier_run_is_still_open(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        chore = await add(client, "Reminder: send the report", FRIDAY)
        runner = console.chore_runner
        clock.to("2026-10-29 12:00")
        await client.post(
            f"/api/chores/{chore['id']}/run"
        )  # Run now: a queued task stays open
        clock.to("2026-10-30 09:01")
        assert len(await runner.tick()) == 1
        assert len(await tasks_of(console)) == 2


async def test_the_api_checks_a_one_time_chore_and_a_done_one_wakes_with_a_new_time(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        for at_text in (
            "2026-10-28T06:59",
            "2028-01-05T09:00",
            "2026-13-01T09:00",
            "09:00",
        ):
            response = await client.post(
                "/api/chores",
                json={
                    "request": "Reminder: call mom",
                    "schedule": {"kind": "once", "at": at_text},
                },
            )
            assert response.status_code == 400, at_text
            assert response.json()["error"]["code"] == "invalid_request"
        chore = await add(client, "Reminder: call mom", FRIDAY)
        assert chore["when"] == "once, Fri Oct 30 at 9:00" and chore["next_run"] == at(
            "2026-10-30 09:00"
        )
        clock.to("2026-10-30 09:05")
        await console.chore_runner.tick()
        [done] = (await client.get("/api/chores")).json()["chores"]
        assert done["enabled"] is False
        # Switching a done reminder on would run a time that has passed.
        on = await client.patch(
            f"/api/chores/{chore['id']}",
            json={"revision": done["revision"], "enabled": True},
        )
        assert on.status_code == 400 and "passed" in on.json()["error"]["message"]
        # A new time arms it again.
        again = await client.patch(
            f"/api/chores/{chore['id']}",
            json={
                "revision": done["revision"],
                "schedule": {"kind": "once", "at": "2026-11-02T08:00"},
            },
        )
        body = again.json()["chore"]
        assert (body["enabled"], body["next_run"]) == (True, at("2026-11-02 08:00"))


async def test_a_reminders_card_shows_the_persons_words_even_when_its_run_failed(
    tmp_path: Path,
) -> None:
    long = "send the report to the whole committee before noon"
    async with chore_console(tmp_path) as (console, client, clock, _, _):
        await add(client, "Reminder: stretch", FRIDAY)
        await add(client, f"Reminder: {long}", FRIDAY, title="send the report")
        await add(client, "Brief me on my calendar", FRIDAY)
        clock.to("2026-10-30 09:01")
        done, failed, plain = await console.chore_runner.tick()

        async def last_card() -> Any:
            return (await client.get("/api/sessions")).json()["main"]["last_card"]

        # Done: the card is flagged and keeps the answer.
        await finish(console, done, clock.now, "Time to stretch.")
        await console.store.append_session_event(
            done, "sched.finished", {"state": "done"}
        )
        card = await last_card()
        assert (card["state"], card["reminder"], card["line"]) == (
            "done",
            True,
            "Time to stretch.",
        )
        # Failed: the line is the person's full words, not a reason.
        await console.workers.tasks.change(
            failed, state="failed", actor="scheduler", failure_reason="No model"
        )
        await console.store.append_session_event(
            failed, "sched.finished", {"state": "failed"}
        )
        card = await last_card()
        assert (card["state"], card["reminder"], card["title"]) == (
            "failed",
            True,
            "send the report",
        )
        assert card["line"] == f"The reminder did not run. It was: {long}"
        # A once chore that is not a reminder (no prefix) is a task like any other.
        await finish(console, plain, clock.now)
        await console.store.append_session_event(
            plain, "sched.finished", {"state": "done"}
        )
        assert "reminder" not in await last_card()
