"""A scheduled task keeps the person's answer to a typing step on its site (#619, rules K1, K4, K5, K6).

A real chore run in Auto with nobody to ask, a scripted model and the
``browser_type`` of ``test_allow_typing`` (the browser's own questions, no
browser). Only the second choice on a blocked step of a chore's run keeps an
answer; the next run uses it; a new request, Remove and Delete end it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from localmelo.support.gateway.chore_schedule import Schedule
from localmelo.support.gateway.chore_store import ChoreStore
from localmelo.tests.agent.test_allow_question import NOTE, SENDS, SITE, typing
from localmelo.tests.agent.test_approvals import _answer

from .test_allow_typing import console_with, scheduled_task, settled

BLOCKED = "It needs the note.\nEnding: blocked: may it type the note?"
LATER = "Checked in again, all is well."


@pytest.mark.asyncio
async def test_a_kept_answer_is_made_from_the_block_used_by_the_next_run_and_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        typing(), _answer(BLOCKED),            # run 1: blocked
        typing(NOTE, "call_2"), _answer("Typed the note."),  # kept, allowed once
        typing(LATER, "call_3"), _answer("Checked in."),     # run 2: on the kept answer
        typing(LATER, "call_4"), _answer(BLOCKED),           # run 3: the request changed
        _answer("All right."),                 # the person's typed yes to run 3
    ]  # fmt: skip
    async with console_with(tmp_path, monkeypatch, plans, [SENDS, SENDS, SENDS]) as (
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

        # K1: a request that names no stored block keeps nothing.
        missing = await client.post(f"/api/tasks/{run}/blocks/not-a-block/keep")
        assert missing.status_code == 404
        assert await runner.store.keeps(chore.chore_id) == []

        # The second choice: the site comes from the stored block; the step runs once.
        kept = await client.post(f"/api/tasks/{run}/blocks/{block['request_id']}/keep")
        assert kept.status_code == 202, kept.text
        # The run ended, so the step goes ahead in a fresh task, allowed once there.
        assert kept.json()["message"] == "successor"
        await settled(console, kept.json()["task"]["task_id"])
        [keep] = await runner.store.keeps(chore.chore_id)
        assert (keep.site, keep.kinds, keep.dropped_at) == (
            SITE,
            ("type", "submit"),
            None,
        )
        assert types.typed == [NOTE]

        # K5: the next run types another note on the kept answer, and its page says so.
        run2 = (await runner.run_now(await runner.store.get(chore.chore_id)))["task_id"]
        await settled(console, run2)
        assert types.typed == [NOTE, LATER]
        task = (await client.get(f"/api/tasks/{run2}")).json()["task"]
        assert task["blocked"] == []
        [use] = task["kept_uses"]
        assert (use["question"], use["arguments"]["text"]) == (
            f"browser-type:{SITE}",
            LATER,
        )
        listed = (await client.get("/api/chores")).json()["chores"][0]["keeps"]
        assert [(k["site"], k["dropped"]) for k in listed] == [(SITE, False)]

        # K4: a new request drops it, and the page says so; the next run is blocked.
        fresh = await runner.store.get(chore.chore_id)
        changed = await client.patch(
            f"/api/chores/{chore.chore_id}",
            json={"revision": fresh.revision, "request": "check in and say hello"},
        )
        assert changed.json()["chore"]["keeps"][0]["dropped"] is True
        run3 = (await runner.run_now(await runner.store.get(chore.chore_id)))["task_id"]
        await settled(console, run3)
        assert types.typed == [NOTE, LATER]
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
@pytest.mark.parametrize("chat_started", [False, True])
async def test_nothing_is_kept_for_a_task_that_is_not_a_scheduled_tasks_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chat_started: bool
) -> None:
    """K6: a one-time task, and a chat's task, offer no second choice and the route refuses it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [typing(), _answer(BLOCKED)]
    async with console_with(tmp_path, monkeypatch, plans, [SENDS]) as (
        console,
        client,
        _,
        _,
    ):
        task_id = await scheduled_task(console, tmp_path, chat_started=chat_started)
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
