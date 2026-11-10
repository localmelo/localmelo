"""A scheduled task keeps the person's answer to a typing step on its site (#619, rules K1, K4, K5, K6).

A real chore run in Auto with nobody to ask, a scripted model and the
``browser_type`` of ``test_allow_typing`` (the browser's own questions, no
browser). Only the second choice on a blocked step of a chore's run keeps an
answer; the next run uses it; a new request, Remove and Delete end it.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import EarlierBlocks
from localmelo.melo.schema import Message
from localmelo.support.gateway.chore_schedule import Schedule
from localmelo.support.gateway.chore_store import Chore, ChoreStore
from localmelo.support.gateway.console import Console
from localmelo.tests.agent.test_allow_question import NOTE, SENDS, SITE, typing
from localmelo.tests.agent.test_approvals import _answer, _calls
from localmelo.tests.agent.test_auto_mode import verdict

from .conftest import all_events
from .test_allow_typing import console_with, finished_calls, scheduled_task, settled

BLOCKED = "It needs the note.\nEnding: blocked: may it type the note?"
LATER = "Checked in again, all is well."
DAILY = {"kind": "daily", "at": "08:30"}


def sending(text: str, call_id: str) -> Message:
    """The check-in's second step: the form sent to the site's own address."""
    return _calls(
        ("browser_type", {"text": text, "site": SITE, "kind": "submit"}, call_id)
    )


async def a_chore(console: Console) -> Chore:
    assert console.main_id is not None
    await console.store.update_session(
        console.main_id, meta_update={"approvals": "auto"}
    )
    return await console.chore_runner.store.create(
        title="Check in",
        request="do the daily check-in",
        schedule=Schedule.from_json(DAILY),
    )


async def kept_on(console: Console, chore: Chore) -> Chore:
    """*chore*, keeping the person's answer for :data:`SITE`."""
    await console.chore_runner.store.keep(chore.chore_id, SITE, words=chore.words)
    return await console.chore_runner.store.get(chore.chore_id)


@pytest.mark.asyncio
async def test_a_kept_answer_is_made_from_the_block_used_by_the_next_run_and_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        typing(), _answer(BLOCKED),            # run 1: blocked
        # Kept: the typing goes ahead once, and the send after it on the kept answer.
        typing(NOTE, "call_2"), sending(NOTE, "call_s"), _answer("Sent the note."),
        typing(LATER, "call_3"), _answer("Checked in."),     # run 2: on the kept answer
        typing(LATER, "call_b"), _answer(BLOCKED),           # run 2 moved on by the person
        typing(LATER, "call_4"), _answer(BLOCKED),           # run 3: the request changed
        _answer("All right."),                 # the person's typed yes to run 3
    ]  # fmt: skip
    async with console_with(tmp_path, monkeypatch, plans, [SENDS] * 5) as (
        console,
        client,
        types,
        _,
    ):
        assert console.main_id is not None
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "auto"}
        )
        runner = console.chore_runner
        chore = await runner.store.create(
            title="Check in",
            request="do the daily check-in",
            schedule=Schedule.from_json({"kind": "daily", "at": "08:30"}),
        )
        run = (await runner.run_now(chore))["task_id"]
        await settled(console, run)
        [block] = (await client.get(f"/api/tasks/{run}")).json()["task"]["blocked"]
        assert (block["keep_site"], block["kept"]) == (SITE, False)
        # The chat's cards of the run offer it too: the blocked step and the end.
        await console.cards.flush()
        cards = [
            e.event["data"]
            for e in await all_events(console.store, console.main_id)
            if e.event["type"] == "attention.card"
        ]
        [shown] = [c for c in cards if c.get("state") == "blocked"]
        [end] = [c["block"] for c in cards if "block" in c]
        assert shown["keep_site"] == end["keep_site"] == SITE

        # K1: a request that names no stored block keeps nothing.
        missing = await client.post(f"/api/tasks/{run}/blocks/not-a-block/keep")
        assert missing.status_code == 404
        assert await runner.store.keeps(chore.chore_id) == []

        # The second choice: the site comes from the stored block; the step runs once.
        kept = await client.post(f"/api/tasks/{run}/blocks/{block['request_id']}/keep")
        assert kept.status_code == 202, kept.text
        # The run ended, so the step goes ahead in a fresh task, allowed once there.
        assert kept.json()["message"] == "successor"
        going = str(kept.json()["task"]["task_id"])
        await settled(console, going)
        [keep] = await runner.store.keeps(chore.chore_id)
        assert (keep.site, keep.kinds, keep.dropped_at) == (
            SITE,
            ("type", "submit"),
            None,
        )
        # C: that task holds the kept answer: the send after the typing is not asked.
        assert types.typed == [NOTE, NOTE]
        typed, sent = [c["tool_asks"][0] for c in await finished_calls(console, going)]
        assert (typed["allowance"], sent["allowance"]) == ("once", "kept")
        assert (await console.workers.get(going))["blocked"] == []

        # K5: the next run types another note on the kept answer, and its page says so.
        run2 = (await runner.run_now(await runner.store.get(chore.chore_id)))["task_id"]
        await settled(console, run2)
        assert types.typed == [NOTE, NOTE, LATER]
        task = (await client.get(f"/api/tasks/{run2}")).json()["task"]
        assert task["blocked"] == []
        [use] = task["kept_uses"]
        assert (use["question"], use["arguments"]["text"]) == (
            f"browser-type:{SITE}",
            LATER,
        )
        listed = (await client.get("/api/chores")).json()["chores"][0]["keeps"]
        assert [(k["site"], k["dropped"]) for k in listed] == [(SITE, False)]

        # B1: a task the person's message moves the run on in holds nothing kept.
        moved, how = await console.workers.successor(
            await console.workers.tasks.get_task(run2), "Type a hello too.", by="person"
        )
        assert how == "successor"
        [held] = (await settled(console, str(moved["task_id"])))["blocked"]
        assert types.typed == [NOTE, NOTE, LATER] and held["keep_site"] is None

        # K4: a new request drops it, and the page says so; the next run is blocked.
        fresh = await runner.store.get(chore.chore_id)
        changed = await client.patch(
            f"/api/chores/{chore.chore_id}",
            json={"revision": fresh.revision, "request": "check in and say hello"},
        )
        assert changed.json()["chore"]["keeps"][0]["dropped"] is True
        run3 = (await runner.run_now(await runner.store.get(chore.chore_id)))["task_id"]
        await settled(console, run3)
        assert types.typed == [NOTE, NOTE, LATER]
        # Remove takes it off the list.
        removed = await client.delete(
            f"/api/chores/{chore.chore_id}/keeps", params={"site": SITE}
        )
        assert removed.json()["chore"]["keeps"] == []
        # K1: the person's typed yes goes on with the work and keeps nothing.
        record = await console.workers.tasks.get_task(run3)
        went_on, how = await console.workers.successor(
            record, "Yes, always allow typing there.", by="person"
        )
        assert how == "successor"
        await settled(console, str(went_on["task_id"]))
        assert await runner.store.keeps(chore.chore_id) == []
        # Deleting the scheduled task deletes what it kept.
        chore = await runner.store.get(chore.chore_id)
        await runner.store.keep(chore.chore_id, SITE, words=chore.words)
        await runner.delete(chore.chore_id)
        assert await runner.store.keeps(chore.chore_id) == []


@pytest.mark.asyncio
async def test_nothing_is_kept_for_a_task_that_is_not_a_scheduled_tasks_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K6: a one-time task offers no second choice and the route refuses it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [typing(), _answer(BLOCKED)]
    async with console_with(tmp_path, monkeypatch, plans, [SENDS]) as (
        console,
        client,
        _,
        _,
    ):
        task_id = await scheduled_task(console, tmp_path)
        [block] = (await console.workers.get(task_id))["blocked"]
        assert block["can_allow"] is True and block["keep_site"] is None

        refused = await client.post(
            f"/api/tasks/{task_id}/blocks/{block['request_id']}/keep"
        )
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            409,
            "not_keepable",
        )


@pytest.mark.asyncio
async def test_a_task_from_the_tasks_route_with_a_runs_key_holds_nothing_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K3a: a request key shaped like a scheduled run's makes no task a run of it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [typing(), _answer(BLOCKED)]
    async with console_with(tmp_path, monkeypatch, plans, [SENDS]) as (
        console,
        client,
        types,
        _,
    ):
        chore = await kept_on(console, await a_chore(console))
        made = await client.post(
            "/api/tasks",
            json={
                "objective": "do the daily check-in",
                "request_key": f"chore:{chore.chore_id}:1",
            },
        )
        assert made.status_code in (200, 201, 202), made.text
        task_id = str(made.json()["task"]["task_id"])
        [block] = (await settled(console, task_id))["blocked"]

        # The kept answer did not run the step, and its block offers no keep.
        assert types.typed == [] and block["keep_site"] is None
        refused = await client.post(
            f"/api/tasks/{task_id}/blocks/{block['request_id']}/keep"
        )
        assert refused.json()["error"]["code"] == "not_keepable"
        assert [
            k.site for k in await console.chore_runner.store.keeps(chore.chore_id)
        ] == [SITE]


def _refused_earlier(policy: Any) -> EarlierBlocks:
    """*policy*, with the site refused earlier in the job."""
    return EarlierBlocks(
        policy, {"k0": ("Auto mode", "")}, {"k0": ["ask:site:notes.example"]}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "review", "text"),
    [
        ("credentials", verdict(verdict="ask", category="credentials"), NOTE),
        ("a refused site", SENDS, NOTE),
        ("could not be reviewed", None, NOTE),
        ("does not fit the task", verdict(verdict="ask", category="send_out", fits_request="no"), NOTE),
        ("personal details", SENDS, "Call me at 415-555-0134 today."),
    ],
)  # fmt: skip
async def test_the_keep_route_refuses_a_block_a_kept_answer_would_not_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, review: Any, text: str
) -> None:
    """A5: in a scheduled task's run, only a block a kept answer would have run offers the second choice."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [typing(text), _answer(BLOCKED)]
    reviews = (
        [review]
        if review is not None
        else [Message(role="assistant", content="not json")] * 2
    )
    async with console_with(tmp_path, monkeypatch, plans, reviews) as (
        console,
        client,
        types,
        _,
    ):
        if case == "a refused site":
            guard = console.workers._guard

            async def refusing(*args: Any, **kwargs: Any) -> Any:
                return _refused_earlier(await guard(*args, **kwargs))

            monkeypatch.setattr(console.workers, "_guard", refusing)
        chore = await a_chore(console)
        run = (await console.chore_runner.run_now(chore))["task_id"]
        [block] = (await settled(console, run))["blocked"]
        assert types.typed == [] and block["keep_site"] is None, case

        refused = await client.post(
            f"/api/tasks/{run}/blocks/{block['request_id']}/keep"
        )
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            409,
            "not_keepable",
        ), case
        assert await console.chore_runner.store.keeps(chore.chore_id) == []


class Held(asyncio.Event):
    """Holds the run's first step until the test lets it go."""

    def __init__(self) -> None:
        super().__init__()
        self.reached = asyncio.Event()


@pytest.mark.asyncio
async def test_a_run_going_on_stops_using_an_answer_removed_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [typing(), _answer(BLOCKED)]
    async with console_with(tmp_path, monkeypatch, plans, [SENDS]) as (
        console,
        client,
        types,
        _,
    ):
        held = Held()
        typing_step = types._type

        async def later(*args: Any, **kwargs: Any) -> str:
            held.reached.set()
            await held.wait()
            return await typing_step(*args, **kwargs)

        monkeypatch.setattr(types, "_type", later)
        chore = await kept_on(console, await a_chore(console))
        run = (await console.chore_runner.run_now(chore))["task_id"]
        await asyncio.wait_for(held.reached.wait(), 10)
        # The run started holding the answer; the person removes it now.
        removed = await client.delete(
            f"/api/chores/{chore.chore_id}/keeps", params={"site": SITE}
        )
        assert removed.json()["chore"]["keeps"] == []
        held.set()

        [block] = (await settled(console, run))["blocked"]
        assert types.typed == [] and block["can_allow"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"schedule": {"kind": "daily", "at": "09:15"}},
        {"request": "  do the daily check-in \n"},
    ],
)
async def test_an_edit_that_leaves_the_request_as_it_was_keeps_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    async with console_with(tmp_path, monkeypatch, [], []) as (console, client, *_):
        chore = await kept_on(console, await a_chore(console))
        changed = await client.patch(
            f"/api/chores/{chore.chore_id}", json={"revision": chore.revision, **change}
        )
        assert changed.status_code == 200, changed.text
        assert [
            (k["site"], k["dropped"]) for k in changed.json()["chore"]["keeps"]
        ] == [(SITE, False)]
        fresh = await console.chore_runner.store.get(chore.chore_id)
        assert await console.chore_runner.store.live_keep(chore.chore_id, fresh.words)


@pytest.mark.asyncio
async def test_a_store_from_before_opens_with_nothing_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    async with console_with(tmp_path, monkeypatch, [], []) as (console, *_):
        chore = await console.chore_runner.store.create(
            title="Check in",
            request="do the daily check-in",
            schedule=Schedule.from_json({"kind": "daily", "at": "08:30"}),
        )
    with sqlite3.connect(tmp_path / "sessions.db") as conn:
        conn.execute("DROP TABLE chore_keeps")
    async with console_with(tmp_path, monkeypatch, [], []) as (console, *_):
        store = ChoreStore(console.store)
        assert await store.keeps(chore.chore_id) == []
        assert await store.live_keep(chore.chore_id, chore.words) is None
