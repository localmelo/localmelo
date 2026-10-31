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
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console import front as front_module
from localmelo.support.gateway.console.cards import CARD_EVENT
from localmelo.support.gateway.console.chores import ChoreOptions
from localmelo.support.gateway.console.front import (
    ECHO_EVENT,
    SIDE_SUMMARY_EVENT,
    SIDE_TEXT,
    FrontAgent,
    SideAgent,
    front_options,
)
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.store import StoredEvent
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, tool_call

from .conftest import PORT, FakeProbe, make_cfg, quiet_gate
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
    """A console whose main chat runs a FrontAgent and whose side chats run SideAgents.

    *sides* are the scripts of the side chats in the order they are built; the
    returned dict holds each Agent's model by session id, the main chat's under
    ``"main"``. Workers never start.
    """
    models: dict[str, ScriptedLLM] = {"main": ScriptedLLM(main or [])}
    scripts = sides if sides is not None else []
    holder: list[Console] = []

    async def front(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        options = session_options(front_options(settings), workspace)
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
        agent = SideAgent(llm=llm, options=options, tool_providers=[mine.tools])
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
    """The user message the model was last sent."""
    return str(llm.requests[request][-1].content)


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
        [started] = await _events(console, side, ECHO_EVENT)
        assert (started["echo"], started["line"]) == ("started", "Started: Slides · P1")
        record = await _task(console, started["task_id"])
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
        assert "search_files" not in offered, "that one is the main chat's"
        assert SIDE_TEXT in models[side].requests[0][0].content
        result = str(models[side].requests[1][-1].content)
        assert result.startswith(f"Started task {record.task_id}: Slides (P1).")
        assert "comes back to this chat as a card" in result


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
        # Both titles hold "recovery card": this chat's stopped task continues at once.
        echo = await _say_in(client, side, "continue the recovery card task")
        assert (echo["echo"], echo["task_id"]) == ("sent", mine)
        assert "from_main" not in echo and "from_chat" not in echo
        assert (await _task(console, mine)).state == "queued"
        assert (await _task(console, theirs)).state == "failed"
        # With none of its own stopped task left to name, it reaches the main chat's and says so.
        other = await _say_in(client, side, "continue the section task")
        assert (other["echo"], other["task_id"], other["from_main"]) == (
            "sent",
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


# ── 5. Chores belong to the chat that made them ──


async def test_a_chore_made_in_a_side_chat_belongs_to_it_and_reports_there(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        proposed = await _say_in(
            client, side, "Brief me on the news every weekday at 8:30"
        )
        assert proposed["echo"] == "chore"
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
    async with chats_console(tmp_path, chores=ChoreOptions(summarize=_join)) as (
        console,
        client,
        _,
    ):
        proposed = (
            await _say(client, console, "Brief me on the news every weekday at 8:30")
        ).json()["echo"]
        await client.post(f"/api/front/echoes/{proposed['id']}", json={"do": "save"})
        [chore] = await console.chore_runner.store.list_chores()
        assert chore.front_id == console.main_id
        task = await console.chore_runner.run_now(chore)
        await _finished(console, task["task_id"], "Three headlines.")
        assert await console.chore_runner.checkin() is not None


async def _join(facts: list[str]) -> str:
    return " ".join(facts)


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


# ── The builder: a side chat gets the tools and the class ──


async def test_the_default_builder_gives_a_side_chat_its_front_s_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def fake(
        settings: Any, workspace: str, mcp: Any, session_id: str, **kw: Any
    ) -> Any:
        seen.update(kw, session_id=session_id)
        return SideAgent(
            llm=ScriptedLLM([]),
            options=session_options(front_options(settings), workspace),
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
        agent = await console._side_agent(console.cfg, str(tmp_path), None, "side-1")
        front = console.front_for("side-1")
        assert (seen["tools"], seen["agent_class"]) == ([front.tools], SideAgent)
        assert isinstance(agent, SideAgent) and agent.front is front
        assert console.front_for("side-1") is front
        assert console.front_for(console.main_id) is console.front
        assert type(front.tools).__name__ == "TaskTools"
        assert type(console.front.tools).__name__ == "FrontTools"
    finally:
        await console.aclose()


async def test_a_side_agent_is_not_handed_off_when_it_runs_out_of_steps(
    tmp_path: Path,
) -> None:
    steps = [tool_call("task_list", {}, call_id=f"c{i}") for i in range(1, 6)]
    async with chats_console(tmp_path, sides=[steps]) as (console, client, models):
        created = await _open(client, "think about the trip")
        side = created["session"]["id"]
        await asyncio.wait_for(_until_blocked(models, side), timeout=10)
        # Five steps, past the main chat's cap of four, and no task was started for it.
        assert len(models[side].requests) == 6
        assert await console.workers.tasks.list_tasks() == []
        await client.post(f"/api/turns/{created['turn_id']}/cancel")


async def _until_blocked(models: dict[str, ScriptedLLM], side: str) -> None:
    while side not in models or not models[side].blocked.is_set():
        await asyncio.sleep(0.01)


def test_the_front_module_names_what_the_memory_lane_hears() -> None:
    assert front_module.SIDE_SUMMARY_EVENT == "side.summary"
