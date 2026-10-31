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
        # The title comes from the request when none is given.
        derived = await add(
            client, "Water the plants. Then rest.", {"kind": "hours", "every": 6}
        )
        assert derived["title"] == "Water the plants. Then rest."

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
        assert (await client.get("/api/chores")).json()["chores"] == []


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
