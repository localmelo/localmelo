"""Side chats are equals: they start tasks, and their tasks report back to them.

A side chat has the task tools and code's rules of the main chat; a task
records the chat that started it, and its echo, card, news and result go to
that chat. The main chat's Activity lists every task with the owner named, and
its next turn reads one line when a side chat finished work. Each test drives
the console's routes with real Agents on scripted models; no worker starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import socket
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.melo.schema import Message
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console import front as front_module
from localmelo.support.gateway.console.asks import ASK_ORIGIN, FOLLOWUP_ORIGIN
from localmelo.support.gateway.console.browser_cards import BROWSER_EVENT
from localmelo.support.gateway.console.cards import CARD_EVENT
from localmelo.support.gateway.console.chores import ChoreOptions
from localmelo.support.gateway.console.front import (
    BROWSING_OFF_NOTE,
    ECHO_EVENT,
    FRONT_TEXT,
    SIDE_SUMMARY_EVENT,
    FrontAgent,
)
from localmelo.support.gateway.console.registry import chat_options
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.identity import ACTING_NOTE
from localmelo.support.sessions.store import StoredEvent
from localmelo.tests.browser.fakes import FakeDns, FakeDriver
from localmelo.tests.browser.test_agent import SIBLING, SITE, _collect
from localmelo.tests.browser.test_live import HOME, site
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, tool_call

from .conftest import PORT, FakeProbe, make_cfg, quiet_gate
from .test_ask_owner import Script, held, until
from .test_auto_mode import KEYS
from .test_front import _say, _task
from .test_front_followup import _finished, _stopped
from .test_turn_writes import _post_turn, _wait_turn

# The first message of a side chat: it answers, and starts nothing.
OPENING = answer("Hello from the side chat.")


@contextlib.asynccontextmanager
async def chats_console(
    tmp_path: Path,
    *,
    main: list[Message] | None = None,
    sides: list[list[Message]] | None = None,
    chores: ChoreOptions | None = None,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, dict[str, ScriptedLLM]]]:
    """A console whose main chat and side chats each run a FrontAgent.

    *sides* are the scripts of the side chats in the order they are built; the
    returned dict holds each Agent's model by session id, the main chat's under
    ``"main"``. Workers never start.
    """
    models: dict[str, ScriptedLLM] = {"main": ScriptedLLM(main or [])}
    scripts = sides if sides is not None else []
    holder: list[Console] = []

    async def front(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        options = session_options(chat_options(settings, workspace), workspace)
        agent = FrontAgent(
            llm=models["main"], options=options, tool_providers=[holder[0].front.tools]
        )
        agent.front = holder[0].front
        await agent.start()
        return agent

    async def side(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        mine = holder[0].front_for(session_id)
        llm = models[session_id] = ScriptedLLM(scripts.pop(0) if scripts else [OPENING])
        options = session_options(AgentOptions(mode="gateway"), workspace)
        agent = FrontAgent(llm=llm, options=options, tool_providers=[mine.tools])
        agent.front = mine
        await agent.start()
        return agent

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=side,
            front_builder=front,
            probe=FakeProbe(),
            gate=quiet_gate(),
            chores=chores,
        ),
    )
    holder.append(console)

    async def no_schedule() -> None:
        pass

    console.workers._schedule = no_schedule  # type: ignore[method-assign]
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, models
    finally:
        await client.aclose()
        await console.aclose()


def _fails(event: StoredEvent) -> None:
    raise RuntimeError("a listener that fails")


async def _open(client: httpx.AsyncClient, title: str) -> dict[str, Any]:
    """A side chat whose first message is *title*, working in a folder of the home."""
    folder = Path.home() / "side-folder"
    folder.mkdir(exist_ok=True)
    response = await client.post(
        "/api/sessions",
        json={"turn": {"text": title}, "workspace": str(folder)},
    )
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


async def _side(client: httpx.AsyncClient, tmp_path: Path, title: str) -> str:
    """A side chat titled *title*, after its first turn."""
    side = str((await _open(client, title))["session"]["id"])
    await _wait_turn(client, side)
    return side


async def _say_in(
    client: httpx.AsyncClient, chat: str, text: str, **extra: Any
) -> dict[str, Any]:
    response = await _post_turn(client, chat, text, **extra)
    assert response.status_code == 200, response.text
    echo: dict[str, Any] = response.json()["echo"]
    return echo


async def _events(console: Console, chat: str, kind: str) -> list[dict[str, Any]]:
    found = await console.store.events_of_type(chat, kind, limit=100)
    return [{"id": e.id, **e.event["data"]} for e in found]


async def _task_cards(console: Console, chat: str) -> list[dict[str, Any]]:
    """The cards of tasks in *chat* (a side chat's own run is a card in the main chat too)."""
    return [
        c
        for c in await _events(console, chat, CARD_EVENT)
        if c["source_kind"] == "task"
    ]


def _hear(console: Console, session: str, kind: str, data: dict[str, Any]) -> None:
    """The cards hear a run's event, as the store's listener passes it."""
    event = {"type": kind, "ts": time.time(), "data": data}
    console.cards._heard(StoredEvent(1, session, 1, event))


def _asked(llm: ScriptedLLM, request: int = -1) -> str:
    """The user message the model was last sent, without the chat's reminder."""
    said = str(llm.requests[request][-1].content).removesuffix("\n\n" + ACTING_NOTE)
    return said.removesuffix("\n\n" + BROWSING_OFF_NOTE)


# ── 1. The task tools and code's rules ──


async def test_a_side_chat_starts_a_task_with_the_task_tools_and_owns_it(
    tmp_path: Path,
) -> None:
    replies = [
        tool_call("task_create", {"title": "Slides", "objective": "Make 6 slides"}),
        answer("Started the slides."),
    ]
    async with chats_console(tmp_path, sides=[replies]) as (console, client, models):
        created = await _open(client, "make slides from my notes today, urgent")
        side = created["session"]["id"]
        turn = await _wait_turn(client, side)
        assert turn["status"] == "completed"
        # A request the chat cannot do itself starts at once (#410): the person's
        # message is its request, the model's title only names it.
        [started] = await _events(console, side, ECHO_EVENT)
        assert (started["echo"], started["line"]) == ("started", "Started: Slides · P1")
        record = await _task(console, started["task_id"])
        assert record.spec["user_words"] == "make slides from my notes today, urgent"
        # The side chat owns it, in the folder it works in; the main chat holds no echo of it.
        assert (record.front_id, record.level) == (side, 1)
        folder = created["session"]["workspace"]
        session = await console.store.get_session(record.task_id)
        assert session is not None and session.workspace == folder
        assert await _events(console, str(console.main_id), ECHO_EVENT) == []
        entry = console.sessions.get(side)
        assert entry is not None and entry.agent is not None
        offered = {tool.name for tool in entry.agent.list_tools()}
        assert {"task_create", "task_list", "task_control", "task_message"} <= offered
        assert "search_files" in offered
        assert FRONT_TEXT in models[side].requests[0][0].content
        result = str(models[side].requests[1][-1].content)
        assert result.startswith(f"Started task {record.task_id}: Slides (P1)")


async def test_code_runs_the_same_rules_in_a_side_chat_without_a_model_call(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        side = await _side(client, tmp_path, "Slides")
        asked = len(models[side].requests)
        echo = await _say_in(client, side, "/task make slides from notes.md")
        assert (echo["echo"], echo["line"]) == (
            "started",
            "Started: make slides from notes.md · P2",
        )
        record = await _task(console, echo["task_id"])
        assert (record.front_id, record.reason) == (side, "you asked for it")
        # "Run as a task" and a control phrase are code's too.
        urgent = await _say_in(client, side, "fix the build, it's urgent", as_task=True)
        assert urgent["line"].endswith("· P1")
        paused = await _say_in(client, side, "pause the slides")
        assert (paused["echo"], paused["task_id"]) == ("control", echo["task_id"])
        assert len(models[side].requests) == asked
        # The echoes are in the side chat, and its page carries them.
        assert len(await _events(console, side, ECHO_EVENT)) == 3
        assert await _events(console, str(console.main_id), ECHO_EVENT) == []
        page = (await client.get(f"/api/sessions/{side}/turns")).json()
        assert [e["id"] for e in page["echoes"]] == [
            e["id"] for e in await _events(console, side, ECHO_EVENT)
        ]


async def test_a_tap_on_a_side_chats_echo_is_found_in_that_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Slides")
        task = await _say_in(client, side, "/task make slides from notes.md")
        paused = await _say_in(client, side, "pause the slides")
        url = f"/api/front/echoes/{paused['id']}"
        # Without the chat the main chat is asked, and it has no such echo.
        assert (await client.post(url, json={"do": "undo"})).status_code == 404
        for wrong in ("no-such-chat", task["task_id"]):
            refused = await client.post(url, json={"do": "undo", "session_id": wrong})
            assert refused.status_code == 404
        undone = await client.post(url, json={"do": "undo", "session_id": side})
        assert undone.status_code == 200, undone.text
        assert undone.json()["echo"]["of"] == paused["id"]
        assert (await _task(console, task["task_id"])).state == "queued"


async def test_a_run_as_a_task_in_a_chat_that_is_a_task_is_refused(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Slides")
        echo = await _say_in(client, side, "/task tidy photos")
        refused = await _post_turn(client, echo["task_id"], "more", as_task=True)
        assert refused.status_code == 400


# ── 3. A chat acts on its own tasks first, then the rest, and says which ──


async def test_a_chat_acts_on_its_own_tasks_first_and_then_on_the_rest(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        mine = await _say_in(client, side, "/task Slides draft for Tuesday")
        theirs = await _say(client, console, "/task Slides for Monday")
        main_task = theirs.json()["echo"]["task_id"]
        # Both titles hold "slides": the side chat's own task wins.
        own = await _say_in(client, side, "pause the slides")
        assert (own["task_id"], "from_main" in own, "from_chat" in own) == (
            mine["task_id"],
            False,
            False,
        )
        # Only the main chat's task holds "Monday": the side chat reaches it and says whose it is.
        reached = await _say_in(client, side, "stop Monday")
        assert (reached["task_id"], reached["from_main"]) == (main_task, True)
        assert (await _task(console, main_task)).state == "cancelled"
        # The main chat: Tuesday's slides belong to the side chat.
        tuesday = (await _say(client, console, "stop Tuesday")).json()["echo"]
        assert (tuesday["task_id"], tuesday["from_chat"]) == (
            mine["task_id"],
            "Trip plan",
        )


async def test_continue_the_task_picks_this_chats_stopped_task_first(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        side = await _side(client, tmp_path, "Trip plan")
        mine = (await _say_in(client, side, "/task Fix the recovery card test"))[
            "task_id"
        ]
        theirs = (
            await _say(client, console, "/task Hide the recovery card section")
        ).json()["echo"]["task_id"]
        await _stopped(console, mine)
        await _stopped(console, theirs)
        # Both titles hold "recovery card": this chat's stopped task goes on at once,
        # in a fresh task of this chat (#571).
        echo = await _say_in(client, side, "continue the recovery card task")
        assert (echo["echo"], echo["follows"]) == ("started", mine)
        assert "from_main" not in echo and "from_chat" not in echo
        assert (await _task(console, echo["task_id"])).state == "queued"
        assert (await _task(console, mine)).state == "failed"
        assert (await _task(console, theirs)).state == "failed"
        # With none of its own stopped task left to name, it reaches the main chat's and says so.
        other = await _say_in(client, side, "continue the section task")
        assert (other["echo"], other["follows"], other["from_main"]) == (
            "started",
            theirs,
            True,
        )
        assert len(models[side].requests) == 1, "no model call, only the first message"


async def test_the_task_tools_list_own_tasks_first_and_name_the_other_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        await _say(client, console, "/task Main chat work")
        await _say_in(client, side, "/task Side chat work")
        listed = await console.front_for(side).tools.list_tasks()
        rows = listed.splitlines()[1:]
        assert [r.split(" · ")[1] for r in rows] == ["Side chat work", "Main chat work"]
        assert rows[1].endswith("· P2 · from the main chat")
        assert "from" not in rows[0]
        # The main chat lists its own first too, and names the side chat for the rest.
        main_rows = (await console.front.tools.list_tasks()).splitlines()[1:]
        assert [r.split(" · ")[1] for r in main_rows] == [
            "Main chat work",
            "Side chat work",
        ]
        assert main_rows[1].endswith("· from Trip plan")


async def test_a_tool_call_on_another_chats_task_says_whose_it_is(
    tmp_path: Path,
) -> None:
    scripts: list[list[Message]] = []
    async with chats_console(tmp_path, sides=scripts) as (console, client, models):
        main_task = (await _say(client, console, "/task Slides for Monday")).json()[
            "echo"
        ]["task_id"]
        scripts.append(
            [
                tool_call("task_control", {"task_id": main_task, "action": "stop"}),
                answer("Stopped it."),
            ]
        )
        side = await _side(client, tmp_path, "stop the Monday slides please")
        [echo] = [e for e in await _events(console, side, ECHO_EVENT)]
        # The person's own words name it, so the tool acts, and the echo says whose it is.
        assert (echo["echo"], echo["task_id"], echo["from_main"]) == (
            "control",
            main_task,
            True,
        )
        assert (await _task(console, main_task)).state == "cancelled"
        result = str(models[side].requests[1][-1].content)
        assert "(a task of the main chat, not of this chat)" in result


# ── 2. Echo, card and result go to the chat that started the task ──


async def test_a_side_tasks_card_goes_to_the_side_chat_not_the_main_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        echo = await _say_in(client, side, "/task Find hotels")
        mine = await _say(client, console, "/task Book a table")
        await _finished(console, echo["task_id"], "Three hotels.\nSaved hotels.md.")
        await _finished(console, mine.json()["echo"]["task_id"], "Booked for two.")
        main = str(console.main_id)
        listed = (
            await client.get("/api/sessions")
        ).json()  # reading the list writes the cards
        in_side = await _events(console, side, CARD_EVENT)
        in_main = await _events(console, main, CARD_EVENT)
        assert [(c["source_id"], c["state"], c["line"]) for c in in_side] == [
            (echo["task_id"], "done", "Three hotels.\nSaved hotels.md.")
        ]
        # The main chat holds its own task's card and the side chat's run (the chat
        # answered its first message), never the side chat's task.
        assert sorted(c["source_id"] for c in in_main) == sorted(
            [side, mine.json()["echo"]["task_id"]]
        )
        assert listed["main"]["last_card"]["source_id"] in {
            c["source_id"] for c in in_main
        }
        page = (await client.get(f"/api/sessions/{side}/turns")).json()
        assert [c["source_id"] for c in page["cards"]] == [echo["task_id"]]


async def test_a_waiting_card_and_its_answer_belong_to_the_chat_that_owns_the_task(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        task_id = (await _say_in(client, side, "/task Edit notes"))["task_id"]
        request = ApprovalRequest(
            request_id="r1",
            run_id=task_id,
            call_id="c1",
            tool_name="file_write",
            arguments={"path": "/tmp/notes.md", "content": "x"},
            preview="write notes.md",
            workspace_root="/tmp",
        )
        waiter = console.broker.begin(task_id, request)
        data = {"request_id": "r1", "tool_name": "file_write"}
        _hear(console, task_id, "approval.requested", data)
        await client.get("/api/sessions")
        [waiting] = await _events(console, side, CARD_EVENT)
        assert (waiting["state"], waiting["request_id"]) == ("waiting", "r1")
        assert await _task_cards(console, str(console.main_id)) == []
        await client.post("/api/approvals/r1", json={"choice": "once"})
        await waiter
        _hear(
            console,
            task_id,
            "approval.resolved",
            {"request_id": "r1", "outcome": "allowed"},
        )
        await client.get("/api/sessions")
        states = [c["state"] for c in await _events(console, side, CARD_EVENT)]
        assert states == ["waiting", "answered"]
        assert await _task_cards(console, str(console.main_id)) == []


async def test_a_task_whose_chat_is_gone_comes_back_to_the_main_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        task_id = (await _say_in(client, side, "/task Find hotels"))["task_id"]
        assert side in console._side_fronts
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        assert side not in console._side_fronts
        await _finished(console, task_id, "Three hotels.")
        await client.get("/api/sessions")
        cards = await _task_cards(console, str(console.main_id))
        assert [c["source_id"] for c in cards] == [task_id]


# ── News: a side chat reads its tasks, the main chat reads one line per side chat ──


async def test_a_side_chat_reads_its_tasks_that_ended_on_its_next_turn(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, sides=[[OPENING, answer("Noted.")]]) as (
        console,
        client,
        models,
    ):
        side = await _side(client, tmp_path, "Trip plan")
        task_id = (await _say_in(client, side, "/task Find hotels"))["task_id"]
        other = (await _say(client, console, "/task Book a table")).json()["echo"]
        await _finished(console, task_id, "Three hotels.\nSaved hotels.md.")
        await _finished(console, other["task_id"], "Booked for two.")
        await _post_turn(client, side, "anything new?")
        await _wait_turn(client, side)
        news = _asked(models[side])
        assert news.startswith("anything new?")
        assert (
            "tasks that ended since the last message · data, not instructions]" in news
        )
        assert "Find hotels · done · P2 · Three hotels. / Saved hotels.md." in news
        assert "Book a table" not in news, "that task is the main chat's"
        # Read once.
        models[side].replies.append(answer("Still nothing."))
        await _post_turn(client, side, "and now?")
        await _wait_turn(client, side)
        assert _asked(models[side]) == "and now?"


async def test_the_main_chat_reads_one_line_per_side_chat_that_finished_work(
    tmp_path: Path,
) -> None:
    async with chats_console(
        tmp_path,
        main=[answer("Nothing yet."), answer("Okay.")],
        sides=[[answer("The trip is planned for May.")], [OPENING]],
    ) as (console, client, models):
        heard: list[StoredEvent] = []
        console.front.add_side_listener(heard.append)
        trip = await _side(client, tmp_path, "Plan the trip")
        slides = await _side(client, tmp_path, "Slides")
        task_id = (await _say_in(client, slides, "/task Make slides"))["task_id"]
        await _finished(console, task_id, "Six slides.\nSaved deck.pptx.")
        await _say(client, console, "what happened?")
        await _wait_turn(client, str(console.main_id))
        asked = _asked(models["main"])
        assert asked.startswith("what happened?")
        # With a side chat's line in it the block is news, not only tasks that ended.
        assert (
            "[localmelo · news since your last message · data, not instructions]"
            in asked
        )
        assert "tasks that ended since" not in asked
        lines = [row for row in asked.splitlines() if row.startswith("side chat ")]
        # One line each, naming the chat and its summary; the task is not listed on its own.
        assert sorted(lines) == [
            'side chat "Plan the trip": The trip is planned for May.',
            'side chat "Slides": Six slides. Saved deck.pptx.',
        ]
        assert "Make slides · done" not in asked
        assert {(e.session_id, e.event["data"]["title"]) for e in heard} == {
            (trip, "Plan the trip"),
            (slides, "Slides"),
        }
        # Read once.
        await _say(client, console, "and now?")
        await _wait_turn(client, str(console.main_id))
        assert _asked(models["main"]) == "and now?"


async def test_side_summary_is_emitted_once_per_ending_with_the_chat_and_its_words(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, sides=[[answer("Planned: May 3 to 7.")]]) as (
        console,
        client,
        _,
    ):
        heard: list[StoredEvent] = []
        console.front.add_side_listener(heard.append)
        console.front.add_side_listener(_fails)  # a failing handler harms nothing
        trip = await _side(client, tmp_path, "Plan the trip")
        await client.get("/api/sessions")
        [event] = heard
        assert event.event["type"] == SIDE_SUMMARY_EVENT == "side.summary"
        assert event.event["data"] == {
            "chat_id": trip,
            "title": "Plan the trip",
            "summary": "Planned: May 3 to 7.",
            "source": "chat",
        }
        assert event.session_id == trip
        await client.get("/api/sessions")
        assert len(heard) == 1, "the same ending is not told twice"
        # A failed run and a reminder's run are no finished work.
        console.front.add_side_listener(heard.append)
        task_id = (await _say_in(client, trip, "/task Find hotels"))["task_id"]
        await console.store.append_session_event(
            task_id, "sched.finished", {"state": "failed"}
        )
        await client.get("/api/sessions")
        assert len(heard) == 1


async def test_a_side_chats_ending_is_kept_in_memory_for_every_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, sides=[[OPENING]]) as (console, client, _):
        slides = await _side(client, tmp_path, "Slides")
        task_id = (await _say_in(client, slides, "/task Make slides"))["task_id"]
        await _finished(console, task_id, "Six slides.\nSaved deck.pptx.")
        await client.get("/api/sessions")
        await console.memory.drain()
        found = await console.memory.search("how many slides, deck.pptx?")
        summaries = [i for i in found if i.source.startswith("chat “Slides” · summary")]
        assert [i.text for i in summaries] == [
            "Summary of chat “Slides”: Six slides. Saved deck.pptx."
        ]


async def test_the_main_chats_own_tasks_and_runs_are_not_side_news(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, main=[answer("ok")]) as (
        console,
        client,
        models,
    ):
        heard: list[StoredEvent] = []
        console.front.add_side_listener(heard.append)
        task_id = (await _say(client, console, "/task Book a table")).json()["echo"][
            "task_id"
        ]
        await _finished(console, task_id, "Booked.")
        await _say(client, console, "news?")
        await _wait_turn(client, str(console.main_id))
        asked = _asked(models["main"])
        assert "Book a table · done" in asked and "side chat" not in asked
        assert heard == []


async def test_a_side_chat_opened_late_still_reads_what_its_tasks_did_before(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, sides=[[answer("Noted.")]]) as (
        console,
        client,
        models,
    ):
        folder = Path.home() / "late"
        folder.mkdir()
        info = await console.store.create_session(
            model="stub-model", workspace=str(folder)
        )
        # A task the chat owns ended before anything opened the chat.
        body, _ = await console.workers.create(
            {"objective": "Find hotels", "front_id": info.id}
        )
        await _finished(console, body["task_id"], "Three hotels.")
        sent = await _post_turn(client, info.id, "anything new?")
        assert sent.status_code == 202, sent.text
        await _wait_turn(client, info.id)
        assert "Find hotels · done · P2 · Three hotels." in _asked(models[info.id])


async def test_a_flush_that_finds_nothing_waits_for_the_one_that_is_writing(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, _, _):
        written: list[int] = []

        async def write(main: str, heard: list[StoredEvent]) -> None:
            if heard:
                await asyncio.sleep(0.1)  # only the flush that holds events is slow
            written.append(len(heard))

        console.cards._write = write  # type: ignore[method-assign]
        _hear(console, "gone", "turn.finished", {})
        first = asyncio.create_task(console.cards.flush())
        await asyncio.sleep(0.02)
        await console.cards.flush()
        # The main chat's turn reads the news right after a flush: what the first
        # flush took must be written by the time the second returns.
        assert written == [1, 0]
        await first


# ── A side chat's task that waits for you shows on the chat list ──


def _hold(console: Console, *task_ids: str, reason: str = "approval") -> None:
    """Read these tasks as held on *reason*, as the scheduler holds one that asks for the person."""
    real = console.workers.tasks.list_tasks

    async def listed(*args: Any, **kwargs: Any) -> Any:
        return [
            (
                dataclasses.replace(r, state="held", wait_reason=reason)
                if r.task_id in task_ids
                else r
            )
            for r in await real(*args, **kwargs)
        ]

    console.workers.tasks.list_tasks = listed  # type: ignore[method-assign]


async def test_a_side_chats_waiting_task_marks_its_row_and_counts_in_the_main_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        trip = await _side(client, tmp_path, "Trip plan")
        quiet = await _side(client, tmp_path, "Slides")
        asks = (await _say_in(client, trip, "/task Edit notes"))["task_id"]
        await _say_in(client, quiet, "/task Tidy photos")
        mine = (await _say(client, console, "/task Book a table")).json()["echo"]
        listed = (await client.get("/api/sessions")).json()
        assert all("waiting_tasks" not in s for s in listed["sessions"])
        assert "waiting_in_side" not in listed["main"]
        _hold(console, asks, mine["task_id"])
        listed = (await client.get("/api/sessions")).json()
        by_id = {s["id"]: s for s in listed["sessions"]}
        assert by_id[trip]["waiting_tasks"] == 1
        assert "waiting_tasks" not in by_id[quiet]
        # The main chat counts the side chats' tasks; its own wait on its card.
        assert listed["main"]["waiting_in_side"] == 1
        # A task held for a slot, not for the person, marks nothing.
        _hold(console, asks, reason="slot")
        listed = (await client.get("/api/sessions")).json()
        assert "waiting_in_side" not in listed["main"]


async def test_a_waiting_task_whose_chat_is_gone_is_the_main_chats_to_answer(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        trip = await _side(client, tmp_path, "Trip plan")
        asks = (await _say_in(client, trip, "/task Edit notes"))["task_id"]
        assert (await client.delete(f"/api/sessions/{trip}")).status_code == 204
        _hold(console, asks)
        listed = (await client.get("/api/sessions")).json()
        assert listed["sessions"] == []
        assert "waiting_in_side" not in listed["main"]


# ── 5. Chores belong to the chat that made them ──


async def test_a_chore_made_in_a_side_chat_belongs_to_it_and_reports_there(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        proposed = await console.front_for(side).propose_chore(
            "Brief me on the news every weekday at 8:30"
        )
        assert proposed is not None and proposed["echo"] == "chore"
        assert await _events(console, str(console.main_id), ECHO_EVENT) == []
        saved = await client.post(
            f"/api/front/echoes/{proposed['id']}",
            json={"do": "save", "session_id": side},
        )
        assert saved.status_code == 200, saved.text
        [chore] = await console.chore_runner.store.list_chores()
        assert chore.front_id == side
        task = await console.chore_runner.run_now(chore)
        assert task["front_id"] == side
        await _finished(console, task["task_id"], "Three headlines.")
        # The chore's result is a card in its chat, and the main chat's check-in leaves it out.
        await client.get("/api/sessions")
        assert [c["source_id"] for c in await _events(console, side, CARD_EVENT)] == [
            task["task_id"]
        ]
        assert await console.chore_runner.checkin() is None
        listed = (await client.get(f"/api/chores?session_id={side}")).json()
        assert [c["front_id"] for c in listed["chores"]] == [side]


async def test_the_main_chats_chores_still_report_in_the_check_in(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        proposed = await console.front.propose_chore(
            "Brief me on the news every weekday at 8:30"
        )
        assert proposed is not None
        await client.post(f"/api/front/echoes/{proposed['id']}", json={"do": "save"})
        [chore] = await console.chore_runner.store.list_chores()
        assert chore.front_id == console.main_id
        task = await console.chore_runner.run_now(chore)
        await _finished(console, task["task_id"], "Three headlines.")
        assert await console.chore_runner.checkin() is not None


# ── Activity names the owner ──


async def test_the_whole_days_activity_names_the_side_chat_that_started_a_task(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        mine = await _say_in(client, side, "/task Find hotels")
        main_task = (await _say(client, console, "/task Book a table")).json()["echo"]
        rows = {r["id"]: r for r in (await client.get("/api/activity")).json()["rows"]}
        assert rows[mine["task_id"]]["from"] == "Trip plan"
        assert "from" not in rows[main_task["task_id"]]
        assert "_chat" not in rows[mine["task_id"]]
        # A side chat's own list is its tasks, with no owner to name.
        own = (await client.get(f"/api/activity?session_id={side}")).json()["rows"]
        assert {r["id"] for r in own} >= {side, mine["task_id"]}
        assert all("from" not in r for r in own)


# ── The builder: every chat gets the same class, tools and options ──


@pytest.mark.parametrize("kind", ["main", "side"])
async def test_every_chat_is_built_the_same_on_its_own_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    seen: dict[str, Any] = {}

    async def fake(
        settings: Any, workspace: str, mcp: Any, session_id: str, **kw: Any
    ) -> Any:
        seen.update(kw, session_id=session_id)
        return kw["agent_class"](
            llm=ScriptedLLM([]),
            options=session_options(chat_options(settings, workspace), workspace),
        )

    monkeypatch.setattr("localmelo.support.gateway.console.default_builder", fake)
    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    try:
        chat = console.main_id if kind == "main" else "side-1"
        assert chat is not None
        agent = await console._chat_agent(console.cfg, str(tmp_path), None, chat)
        front = console.front_for(chat)
        assert seen["tools"] == [front.tools]
        assert isinstance(agent, FrontAgent) and agent.front is front
        assert (front is console.front) is (kind == "main")
    finally:
        await console.aclose()


@pytest.mark.parametrize("kind", ["main", "side"])
async def test_a_chat_that_reads_past_four_steps_goes_on_and_starts_nothing(
    tmp_path: Path, kind: str
) -> None:
    steps = [tool_call("task_list", {"n": i}, call_id=f"c{i}") for i in range(1, 6)]
    async with chats_console(
        tmp_path, main=steps if kind == "main" else None, sides=[steps]
    ) as (console, client, models):
        if kind == "main":
            assert console.main_id is not None
            sent = await _post_turn(client, console.main_id, "think about the trip")
            turn, chat = sent.json(), "main"
        else:
            turn = await _open(client, "think about the trip")
            chat = turn["session"]["id"]
        while chat not in models:
            await asyncio.sleep(0.01)
        model = models[chat]
        await asyncio.wait_for(model.blocked.wait(), timeout=10)
        # Five steps and it still runs: no cap of four, and nothing sent to a task.
        assert len(model.requests) == 6
        assert await console.workers.tasks.list_tasks() == []
        await client.post(f"/api/turns/{turn['turn_id']}/cancel")


def test_the_front_module_names_what_the_memory_lane_hears() -> None:
    assert front_module.SIDE_SUMMARY_EVENT == "side.summary"


# ── A chat does the person's work itself (#571) ──

WRITE = tool_call("file_write", {"path": "notes.txt", "content": "done\n"}, "w-1")
ASKED = "write the word done into notes.txt"


@contextlib.asynccontextmanager
async def built_chats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *scripts: list[Message],
    driver: str = "off",
    search_order: tuple[str, ...] = (),
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, Path, list[Script]]]:
    """A console built as the gateway builds it: only the model is a script.

    Each chat's Agent comes from the real composition, on the next of *scripts*
    (the order the chats open in), in a home folder of the test's.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # A page's name is looked up before it opens: never on the network here.
    monkeypatch.setattr(socket, "getaddrinfo", FakeDns().getaddrinfo)
    folder = tmp_path / "work"
    folder.mkdir()
    waiting = [Script(tuple(script)) for script in scripts]
    models: list[Script] = []

    def providers(*_: Any) -> tuple[Script, None]:
        models.append(waiting.pop(0))
        return models[-1], None

    monkeypatch.setattr("localmelo.support.sessions.agents.providers_for", providers)
    cfg = make_cfg()
    cfg.gateway.workspace = str(folder)
    cfg.browser.driver = driver
    cfg.web.search_order = list(search_order)
    cfg.web.search = bool(search_order)
    console = Console(
        cfg,
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, folder, models
    finally:
        await client.aclose()
        await console.aclose()


async def _chat(
    client: httpx.AsyncClient, console: Console, where: str, words: str, folder: Path
) -> str:
    """Send *words* to the main chat, a side chat's second turn, or a new chat's first."""
    if where == "main":
        chat = str(console.main_id)
    else:
        first = "hello" if where == "side" else words
        made = await client.post(
            "/api/sessions",
            json={"turn": {"text": first}, "workspace": str(folder)},
        )
        assert made.status_code == 201, made.text
        chat = str(made.json()["session"]["id"])
        if where == "first":
            return chat
        await _wait_turn(client, chat)
    sent = await _post_turn(client, chat, words)
    assert sent.status_code == 202, sent.text
    return chat


@pytest.mark.parametrize("where", ["main", "side", "first"])
async def test_a_chat_does_the_work_in_its_own_run_and_starts_no_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    script = [WRITE, answer("Wrote it.")]
    if where == "side":
        script = [answer("Hello."), *script]
    async with built_chats(tmp_path, monkeypatch, script) as (
        console,
        client,
        folder,
        models,
    ):
        chat = await _chat(client, console, where, ASKED, folder)
        # The approval is the chat's own: the step waits for the person's card.
        await until(lambda: console.broker.pending(chat))
        [request] = console.broker.pending(chat)
        answered = await client.post(
            f"/api/approvals/{request.request_id}", json={"choice": "once"}
        )
        assert answered.status_code == 200, answered.text
        finished = await _wait_turn(client, chat)
        assert finished["status"] == "completed"
        assert (folder / "notes.txt").read_text() == "done\n"
        assert await console.workers.tasks.list_tasks() == []
        offered = set(models[0].offered[-1])
        assert {
            "file_read",
            "file_write",
            "file_edit",
            "list_files",
            "git_status",
            "python_exec",
            "search_files",
            "task_create",
            "task_schedule",
        } <= offered
        assert "shell_exec" not in offered


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_chats_own_browser_shows_its_card_with_no_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    release = asyncio.Event()
    step = tool_call("browser_open", {"url": HOME}, "b-1")
    script: list[Any] = [held(release, step), answer("Read it.")]
    if where == "side":
        script = [answer("Hello."), *script]
    monkeypatch.setattr(
        "localmelo.support.browser.provider_for",
        lambda *_, **__: BrowserToolProvider(site()),  # type: ignore[arg-type]
    )
    async with built_chats(tmp_path, monkeypatch, script, driver="chrome") as (
        console,
        client,
        folder,
        _,
    ):
        chat = await _chat(client, console, where, f"read {HOME}", folder)

        async def shown() -> list[Any]:
            return await console.store.events_of_type(chat, BROWSER_EVENT, limit=50)

        await until(lambda: console.browser_cards.source_of(chat))
        release.set()
        await until(lambda: console.broker.pending(chat))
        await until(shown)
        card = (await shown())[-1].event["data"]
        assert (card["owner_chat_id"], card["source_session_id"]) == (chat, chat)
        assert card["state"] == "working" and "source_kind" not in card
        [request] = console.broker.pending(chat)
        await client.post(
            f"/api/approvals/{request.request_id}", json={"choice": "once"}
        )
        assert (await _wait_turn(client, chat))["status"] == "completed"
        await until(
            lambda: console.browser_cards.source_of(chat) is None
        )  # its run is over: nothing of it is live
        assert await console.workers.tasks.list_tasks() == []


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_chat_with_a_browser_page_keeps_its_work_and_hands_off_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    hand = tool_call(
        "task_create",
        {"title": "Read on", "objective": "Read the rest.", "continue_current": True},
        "h-1",
    )
    script: list[Any] = [
        tool_call("browser_open", {"url": HOME}, "b-1"),
        hand,
        answer("The page stays here, so I finish it in this chat."),
    ]
    if where == "side":
        script = [answer("Hello."), *script]
    monkeypatch.setattr(
        "localmelo.support.browser.provider_for",
        lambda *_, **__: BrowserToolProvider(site()),  # type: ignore[arg-type]
    )
    async with built_chats(tmp_path, monkeypatch, script, driver="chrome") as (
        console,
        client,
        folder,
        models,
    ):
        chat = await _chat(client, console, where, f"read {HOME}", folder)
        [request] = [await _waiting(console, chat)]
        await _answer(client, request.request_id, "once")
        finished = await _wait_turn(client, chat)
        assert finished["status"] == "completed"
        assert finished["answer"].startswith("The page stays here")
        told = [str(m.content) for m in models[0].requests[-1] if m.role == "tool"]
        assert told[-1].startswith("Not done: this chat has a browser page open.")
        assert await console.workers.tasks.list_tasks() == []


KEYS_STEP = tool_call("python_exec", {"code": KEYS}, "k-1")
MORE = tool_call("file_write", {"path": "more.txt", "content": "more\n"}, "w-2")


async def _mode(client: httpx.AsyncClient, chat: str, mode: str) -> None:
    sent = await client.patch(f"/api/sessions/{chat}", json={"approvals": mode})
    assert sent.status_code == 200, sent.text


async def _waiting(console: Console, chat: str) -> ApprovalRequest:
    await until(lambda: console.broker.pending(chat))
    [request] = console.broker.pending(chat)
    return request


async def _answer(client: httpx.AsyncClient, request_id: str, choice: str) -> None:
    sent = await client.post(f"/api/approvals/{request_id}", json={"choice": choice})
    assert sent.status_code == 200, sent.text


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_chats_mode_decides_each_step_of_its_own_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    release = asyncio.Event()
    script: list[Any] = [WRITE, KEYS_STEP, held(release, MORE), answer("Done.")]
    if where == "side":
        script = [answer("Hello."), *script]
    async with built_chats(tmp_path, monkeypatch, script) as (
        console,
        client,
        folder,
        models,
    ):
        chat = await _chat(client, console, where, ASKED, folder)
        first = await _waiting(console, chat)
        await _mode(client, chat, "auto")
        # The card that waits stays for the person.
        assert console.broker.pending(chat) == [first]
        planned = len(models[0].requests)
        await _answer(client, first.request_id, "once")
        # Ask to Auto: the next step is Auto's, and nobody is asked.
        await until(lambda: len(models[0].requests) == planned + 2)
        assert console.broker.pending(chat) == []
        # Auto to Ask: the step the run is about to take asks.
        await _mode(client, chat, "ask")
        release.set()
        last = await _waiting(console, chat)
        assert last.call_id == "w-2"
        await _answer(client, last.request_id, "once")
        assert (await _wait_turn(client, chat))["status"] == "completed"
        assert (folder / "notes.txt").read_text() == "done\n"
        assert (folder / "more.txt").read_text() == "more\n"


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_task_a_chats_own_run_started_follows_that_chats_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    start = tool_call(
        "task_create", {"objective": "tidy the notes", "title": "Tidy"}, "t-1"
    )
    script: list[Any] = [start, answer("Started.")]
    if where == "side":
        script = [answer("Hello."), *script]
    work = [WRITE, KEYS_STEP, answer("It was blocked.")]
    async with built_chats(tmp_path, monkeypatch, script, work) as (
        console,
        client,
        folder,
        _,
    ):
        chat = await _chat(client, console, where, "tidy the notes in a task", folder)
        assert (await _wait_turn(client, chat))["status"] == "completed"
        [record] = await console.workers.tasks.list_tasks()
        task_id = record.task_id
        first = await _waiting(console, task_id)
        await _mode(client, chat, "auto")
        assert console.broker.pending(task_id) == [first]
        await _answer(client, first.request_id, "once")

        async def ended() -> bool:
            return (await console.workers.get(task_id))["state"] in ("done", "failed")

        await until(ended)
        # The task's next decision was Auto's: its chat's switch reached it.
        body = await console.workers.get(task_id)
        assert body["approvals"] == "auto" and len(body["blocked"]) == 1
        assert console.broker.pending(task_id) == []


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_site_refused_in_a_chats_turn_takes_its_kept_answer_and_its_tabs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    kept = tool_call("browser_open", {"url": SITE}, "b-1")
    refused = tool_call("browser_open", {"url": SIBLING}, "b-2")
    again = tool_call("browser_read", {}, "b-3")
    script: list[Any] = [kept, refused, answer("It was refused."), again, answer("Ok.")]
    if where == "side":
        script = [answer("Hello."), *script]
    monkeypatch.setattr(
        "localmelo.support.browser.provider_for",
        lambda *_, **__: BrowserToolProvider(_collect()),  # type: ignore[arg-type]
    )
    async with built_chats(tmp_path, monkeypatch, script, driver="chrome") as (
        console,
        client,
        folder,
        models,
    ):
        chat = await _chat(client, console, where, "read the site", folder)
        first = await _waiting(console, chat)
        await _answer(client, first.request_id, "session")
        entry = console.sessions.get(chat)
        assert entry is not None
        await until(lambda: first.tool_name in entry.policy.allowed_tools)
        await until(lambda: console.broker.pending(chat))
        [second] = console.broker.pending(chat)
        assert second.tool_name != first.tool_name
        await _answer(client, second.request_id, "deny")
        assert (await _wait_turn(client, chat))["status"] == "completed"
        # The person's newest word on the site stands: the chat keeps nothing for it.
        assert first.tool_name not in entry.policy.allowed_tools
        # A later turn asks again, about the page of the tab that was approved.
        assert (await _post_turn(client, chat, "read it again")).status_code == 202
        third = await _waiting(console, chat)
        assert third.tool_name == first.tool_name
        await _answer(client, third.request_id, "deny")
        assert (await _wait_turn(client, chat))["status"] == "completed"
        tools = [m.content for m in models[0].requests[-1] if m.role == "tool"]
        assert "Welcome" not in str(tools[-1])


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_step_the_person_refuses_stays_undone_in_the_chats_own_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    script = [WRITE, answer("I did not write it.")]
    if where == "side":
        script = [answer("Hello."), *script]
    async with built_chats(tmp_path, monkeypatch, script) as (
        console,
        client,
        folder,
        _,
    ):
        chat = await _chat(client, console, where, ASKED, folder)
        await until(lambda: console.broker.pending(chat))
        [request] = console.broker.pending(chat)
        answered = await client.post(
            f"/api/approvals/{request.request_id}", json={"choice": "deny"}
        )
        assert answered.status_code == 200, answered.text
        assert (await _wait_turn(client, chat))["status"] == "completed"
        assert not (folder / "notes.txt").exists()


async def test_a_step_refused_in_the_chat_stays_refused_in_the_task_it_hands_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusals carry from the chat's turn to the task it hands its work to: the
    same step is refused there before any review, not asked again (#571)."""
    hand = tool_call(
        "task_create",
        {"title": "Notes", "objective": "Write the word into notes.txt.",
         "continue_current": True},
        "h-1",
    )  # fmt: skip
    chat_script = [WRITE, hand, answer("Never asked.")]
    worker_script = [WRITE, answer("Done.\nEnding: ready")]
    async with built_chats(tmp_path, monkeypatch, chat_script, worker_script) as (
        console,
        client,
        folder,
        _,
    ):
        chat = str(console.main_id)
        sent = await _post_turn(client, chat, ASKED)
        assert sent.status_code == 202, sent.text
        await until(lambda: console.broker.pending(chat))
        [request] = console.broker.pending(chat)
        denied = await client.post(
            f"/api/approvals/{request.request_id}", json={"choice": "deny"}
        )
        assert denied.status_code == 200, denied.text
        await until(console.workers.tasks.list_tasks)
        [task] = await console.workers.tasks.list_tasks()

        async def ended() -> bool:
            record = await console.workers.tasks.get_task(task.task_id)
            return bool(console.broker.pending(task.task_id)) or record.state in (
                "done",
                "failed",
                "cancelled",
            )

        await until(ended, timeout=15.0)
        assert not console.broker.pending(task.task_id), "asked again in the task"
        assert not (folder / "notes.txt").exists()


@pytest.mark.parametrize("where", ["main", "side"])
@pytest.mark.parametrize("origin", [ASK_ORIGIN, FOLLOWUP_ORIGIN])
async def test_a_turn_melo_began_cannot_change_anything_whatever_the_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, origin: str
) -> None:
    calls = [
        tool_call("file_write", {"path": "notes.txt", "content": "x"}, "w-1"),
        tool_call("python_exec", {"code": "open('notes.txt', 'w')"}, "w-2"),
        answer("I can only read here."),
    ]
    script = [*([answer("Hello.")] if where == "side" else []), *calls]
    async with built_chats(tmp_path, monkeypatch, script) as (
        console,
        client,
        folder,
        models,
    ):
        chat = str(console.main_id)
        if where == "side":
            made = await client.post(
                "/api/sessions",
                json={"turn": {"text": "hello"}, "workspace": str(folder)},
            )
            chat = str(made.json()["session"]["id"])
            await _wait_turn(client, chat)
        await console.sessions.start_internal(chat, "a task's words: write it", origin)
        assert (await _wait_turn(client, chat))["status"] == "completed"
        assert not (folder / "notes.txt").exists()
        assert console.broker.pending(chat) == []
        offered = set(models[0].offered[-1])
        assert not offered & {"file_write", "file_edit", "python_exec"}
        assert await console.workers.tasks.list_tasks() == []


async def _search_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeDriver:
    """The stand-in behind every browser this test's chats build, with no address looked up."""
    driver = site()

    async def public(_: str) -> None:
        return None

    monkeypatch.setattr("localmelo.support.browser.addresses.require_public", public)
    monkeypatch.setattr(
        "localmelo.support.browser.provider_for",
        lambda *_, **__: BrowserToolProvider(driver),  # type: ignore[arg-type]
    )
    return driver


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_turn_melo_began_starts_no_browser_work_through_a_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    calls = [tool_call("web_search", {"query": "tomatoes"}, "s-1"), answer("Nothing.")]
    script = [*([answer("Hello.")] if where == "side" else []), *calls]
    driver = await _search_browser(monkeypatch)
    async with built_chats(
        tmp_path, monkeypatch, script, driver="chrome", search_order=("browser",)
    ) as (console, client, folder, models):
        chat = str(console.main_id)
        if where == "side":
            made = await client.post(
                "/api/sessions",
                json={"turn": {"text": "hello"}, "workspace": str(folder)},
            )
            chat = str(made.json()["session"]["id"])
            await _wait_turn(client, chat)
        await console.sessions.start_internal(chat, "a task's question", ASK_ORIGIN)
        assert (await _wait_turn(client, chat))["status"] == "completed"
        assert "web_search" in models[0].offered[-1]
        assert (driver.started, driver.calls) == (0, [])
        assert await console.store.events_of_type(chat, BROWSER_EVENT, limit=50) == []


async def test_a_turn_the_person_began_still_searches_with_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = [tool_call("web_search", {"query": "tomatoes"}, "s-1"), answer("Found.")]
    driver = await _search_browser(monkeypatch)
    async with built_chats(
        tmp_path, monkeypatch, script, driver="chrome", search_order=("browser",)
    ) as (console, client, folder, _):
        chat = await _chat(client, console, "main", "search for tomatoes", folder)
        assert (await _wait_turn(client, chat))["status"] == "completed"
        assert driver.started >= 1


async def test_a_reply_that_only_says_what_it_will_do_is_the_reply_and_starts_nothing(
    tmp_path: Path,
) -> None:
    said = "I'll look into it."
    async with chats_console(tmp_path, main=[answer(said)]) as (console, client, _):
        chat = str(console.main_id)
        sent = await _post_turn(client, chat, "find the openings and apply for me")
        assert sent.status_code == 202, sent.text
        finished = await _wait_turn(client, chat)
        assert (finished["status"], finished["answer"]) == ("completed", said)
        assert await console.workers.tasks.list_tasks() == []


async def test_a_change_to_a_named_file_is_the_chats_to_do_not_a_task_by_its_words(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, main=[answer("Fixed.")]) as (
        console,
        client,
        models,
    ):
        chat = str(console.main_id)
        sent = await _post_turn(client, chat, "fix the typo in docs/notes.md")
        assert sent.status_code == 202, sent.text
        await _wait_turn(client, chat)
        assert len(models["main"].requests) == 1
        assert await console.workers.tasks.list_tasks() == []


async def test_no_list_of_words_stands_between_the_models_call_and_the_task(
    tmp_path: Path,
) -> None:
    create = tool_call(
        "task_create", {"title": "Plan", "objective": "plan", "uses_website": False}
    )
    async with chats_console(tmp_path, main=[create, answer("Started.")]) as (
        console,
        client,
        _,
    ):
        chat = str(console.main_id)
        sent = await _post_turn(client, chat, "do not start a task, plan the trip")
        assert sent.status_code == 202, sent.text
        await _wait_turn(client, chat)
        assert len(await console.workers.tasks.list_tasks()) == 1


@pytest.mark.parametrize("where", ["main", "side"])
async def test_a_chat_in_a_folder_no_task_may_use_reads_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """The rule a person's task is held to, read when a chat is built."""
    outside = tmp_path / "outside"
    outside.mkdir()
    home = tmp_path / "person"
    home.mkdir()
    script = [WRITE, answer("I cannot change anything here.")]
    async with built_chats(home, monkeypatch, script) as (console, client, _, models):
        if where == "main":
            console.cfg.gateway.workspace = str(outside)
            chat = await _chat(client, console, where, ASKED, outside)
        else:
            chat = await _chat(client, console, "first", ASKED, outside)
        assert (await _wait_turn(client, chat))["status"] == "completed"
        assert not (outside / "notes.txt").exists()
        offered = set(models[0].offered[0])
        assert {"file_read", "list_files"} <= offered
        assert not offered & {"file_write", "file_edit", "python_exec"}
        assert "inside your home folder" in str(models[0].requests[0][-1].content)
