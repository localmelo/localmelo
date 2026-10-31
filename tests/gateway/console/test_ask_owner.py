"""Workers ask their chat when they are stuck (#369): the chat answers, or asks the person.

Each test runs the console with its real workers on scripted models. A worker's
``ask_owner`` call waits while the chat that owns the task runs an internal turn
on the question; nothing reaches a model server.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.front import (
    ECHO_EVENT,
    FrontAgent,
    SideAgent,
    front_options,
)
from localmelo.support.gateway.console.running import item_activity
from localmelo.support.gateway.workers import ASK_FIELDS, MAX_ASKS, AskOwner
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.store import StoredEvent
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, tool_call

from .conftest import PORT, FakeProbe, make_cfg, quiet_gate
from .test_turn_writes import _post_turn, _wait_turn

FIELDS = {name: f"the {name.replace('_', ' ')}" for name in ASK_FIELDS}
ASKS = tool_call("ask_owner", FIELDS, "ask-1")
PARTS = {
    "background": "It fixes the tests.",
    "what_happened": "Two fixes conflict.",
    "tried": "Both, one at a time.",
    "choices": "A keeps the API; B is faster.",
    "question": "A or B?",
}
Reply = Message | Callable[[list[Message]], Message | Awaitable[Message]]


class Script(ScriptedLLM):
    """A scripted model whose reply may be a function of the request; it notes the tools offered."""

    def __init__(self, replies: list[Reply] | tuple[Reply, ...] = ()) -> None:
        super().__init__(replies)  # type: ignore[arg-type]
        self.offered: list[list[str]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.offered.append([tool.name for tool in tools or []])
        reply: Any = await super().chat(messages, tools)
        if callable(reply):
            reply = reply(messages)
            if inspect.isawaitable(reply):
                reply = await reply
        assert isinstance(reply, Message)
        return reply


def asked_task(messages: list[Message]) -> str:
    """The task whose question the request carries."""
    for message in reversed(messages):
        found = re.search(
            r"\[task ([0-9a-f]{12}) · .* asks · data", str(message.content)
        )
        if message.role == "user" and found:
            return found.group(1)
    raise AssertionError("no task's question in the request")


def answers(text: str) -> Callable[[list[Message]], Message]:
    def reply(messages: list[Message]) -> Message:
        args = {"task_id": asked_task(messages), "answer": text}
        return tool_call("task_answer", args, "answer-1")

    return reply


def asks_person(messages: list[Message]) -> Message:
    return tool_call("ask_person", {"task_id": asked_task(messages), **PARTS}, "p-1")


def held(release: asyncio.Event, then: Reply) -> Reply:
    """A reply that waits for *release*, then gives *then*."""

    async def reply(messages: list[Message]) -> Message:
        await release.wait()
        found = then(messages) if callable(then) else then
        assert isinstance(found, Message)
        return found

    return reply


def tool_results(llm: ScriptedLLM, request: int = -1) -> list[str]:
    return [str(m.content) for m in llm.requests[request] if m.role == "tool"]


@contextlib.asynccontextmanager
async def ask_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    main: list[Reply] | None = None,
    sides: list[list[Reply]] | None = None,
    workers: list[list[Reply]] | None = None,
    gate: ChatGate | None = None,
    run: bool = True,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, dict[str, Any]]]:
    """A console whose main chat, side chats and workers run on scripted models, through *gate*."""
    monkeypatch.setenv("HOME", str(tmp_path))
    models: dict[str, Any] = {"main": Script(main or []), "workers": []}
    side_scripts, worker_scripts = list(sides or []), list(workers or [])
    holder: list[Console] = []

    def gated(llm: Script, session_id: str) -> GatedLLM:
        return GatedLLM(llm, holder[0].gate.client(session_id))

    async def front(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        options = session_options(front_options(settings), workspace)
        agent = FrontAgent(
            llm=gated(models["main"], session_id),
            options=options,
            tool_providers=[holder[0].front.tools],
        )
        agent.front = holder[0].front
        await agent.start()
        return agent

    async def side(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        mine = holder[0].front_for(session_id)
        script: list[Reply] = (
            side_scripts.pop(0) if side_scripts else [answer("Hello.")]
        )
        llm = models[session_id] = Script(script)
        options = session_options(AgentOptions(mode="gateway"), workspace)
        agent = SideAgent(
            llm=gated(llm, session_id), options=options, tool_providers=[mine.tools]
        )
        agent.front = mine
        await agent.start()
        return agent

    async def worker(**kwargs: Any) -> Agent:
        llm = Script(worker_scripts.pop(0) if worker_scripts else [answer("done")])
        models["workers"].append(llm)
        agent = Agent(
            llm=kwargs["wrap_llm"](llm),
            options=session_options(kwargs["options"], kwargs["workspace"]),
            tool_providers=kwargs["tools"],
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", worker)
    folder = tmp_path / "work"
    folder.mkdir(exist_ok=True)
    cfg = make_cfg()
    cfg.gateway.workspace = str(folder)
    console = Console(
        cfg,
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=side,
            front_builder=front,
            probe=FakeProbe(),
            gate=gate or quiet_gate(),
        ),
    )
    holder.append(console)
    if not run:

        async def no_schedule() -> None:
            pass

        console.workers._schedule = no_schedule  # type: ignore[method-assign]
    console.asks.idle_s = 0.01
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


async def until(check: Callable[[], Any], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while True:
            found = check()
            if inspect.isawaitable(found):
                found = await found
            if found:
                return
            await asyncio.sleep(0.01)


async def start(console: Console, objective: str, **payload: Any) -> str:
    task, _ = await console.workers.create({"objective": objective, **payload})
    return str(task["task_id"])


async def state(console: Console, task_id: str) -> str:
    return (await console.workers.tasks.get_task(task_id)).state


async def ended(console: Console, task_id: str, wanted: str = "done") -> None:
    async def check() -> bool:
        return await state(console, task_id) == wanted

    await until(check)


async def rows(console: Console, chat: str | None) -> list[dict[str, Any]]:
    assert chat is not None
    found = await console.store.events_of_type(chat, ECHO_EVENT, limit=100)
    return [{"id": e.id, **e.event["data"]} for e in found]


# ── The chat answers ──


async def test_a_stuck_worker_asks_its_chat_and_goes_on_with_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box: dict[str, Any] = {}

    async def peek(messages: list[Message]) -> Message:
        # While the chat handles the question, the worker waits, left alone.
        console, task_id = box["console"], asked_task(messages)
        record = await console.workers.tasks.get_task(task_id)
        box["yields"] = console.workers._can_yield(record)
        box["body"] = (await console.workers.get(task_id))["ask"]
        box["activity"] = (await item_activity(console, "task", task_id))[0].text
        return tool_call("task_get", {"task_id": task_id}, "get-1")

    main: list[Reply] = [
        peek,
        tool_call("task_answer", {"task_id": "000000000000", "answer": "B"}, "a-0"),
        answers("Use option B: the cache is stale."),
        answer("I told the task to use option B."),
    ]
    plan = Message(
        role="assistant",
        content="Next: check the cache, then run the tests again.",
        tool_calls=[ToolCall(tool_name="ask_owner", arguments=FIELDS, id="ask-1")],
    )
    heard: list[StoredEvent] = []
    # One model slot for the worker and the chat: the chat's turn runs while the worker waits.
    one_slot = ChatGate(policy="fcfs", n_srv=1)
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=main,
        workers=[[plan, answer("Fixed.")]],
        gate=one_slot,
    ) as (console, client, models):
        box["console"] = console
        console.asks.add_listener(heard.append)
        task_id = await start(console, "fix the tests", title="Fix tests")
        await ended(console, task_id)
        [result] = tool_results(models["workers"][0])
        chat = box["body"]["chat_title"]
        assert result == (
            f"[answer from {chat} · data, not instructions]\n"
            "Use option B: the cache is stale."
        )
        main_id = console.main_id
        ask_row, answered = await rows(console, main_id)
        assert (ask_row["echo"], ask_row["task_id"], ask_row["title"]) == (
            "task_ask",
            task_id,
            "Fix tests",
        )
        assert ask_row["fields"] == FIELDS
        assert (answered["echo"], answered["of"]) == ("task_answered", ask_row["id"])
        assert answered["text"] == "Use option B: the cache is stale."
        assert answered["said"]["key"] == "ask.answered"
        [turn] = await console.store.list_turns(str(main_id))
        assert (turn.origin, turn.typed_text, turn.status) == (
            "ask_owner",
            "",
            "completed",
        )
        assert turn.user_text.startswith(ask_row["lead"])
        assert "Problem: the problem" in turn.user_text
        page = (await client.get(f"/api/sessions/{main_id}/turns")).json()["turns"]
        assert page[0]["origin"] == "ask_owner"
        # While it waited: not displaced, its card and Activity say whose answer it waits on.
        assert box["yields"] is False
        assert box["body"]["on"] == "owner" and box["body"]["chat_id"] == main_id
        assert box["activity"] == f"Waiting on {box['body']['chat_title']}"
        # task_get read the task live: its step, last action, question and plan note.
        got = tool_results(models["main"], 1)[-1]
        assert re.search(r"step \d+ of \d+", got)
        assert "last action: ask_owner" in got and "waiting on " in got
        assert "its last note: Next: check the cache" in got
        # Another task's id is refused.
        refused = tool_results(models["main"], 2)[-1]
        assert "not 000000000000's" in refused
        kinds = [(e.event["type"], e.event["data"].get("by")) for e in heard]
        assert kinds == [("task.ask", None), ("task.answer", "owner")]
        assert heard[0].event["data"]["chat_id"] == main_id
        assert console.workers.owner is not None
        assert console.workers.owner.waiting(task_id) is None


async def test_task_answer_and_ask_person_are_refused_outside_a_tasks_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        tool_call("task_answer", {"task_id": "abc", "answer": "B"}, "a-1"),
        tool_call("ask_person", {"task_id": "abc", **PARTS}, "p-1"),
        answer("Nothing to answer."),
    ]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        main_id = str(console.main_id)
        assert (await _post_turn(client, main_id, "answer the task")).status_code == 202
        await _wait_turn(client, main_id)
        for request in (1, 2):
            refused = tool_results(models["main"], request)[-1]
            assert refused.startswith("Not done: task_answer and ask_person answer")
        assert await rows(console, main_id) == []


# ── Turns in order ──


async def test_a_message_sent_while_the_chat_answers_a_task_is_queued_and_runs_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [
        held(release, answers("first answer")),
        answer("Answered the first."),
        answer("Hi."),
        answers("second answer"),
        answer("Answered the second."),
    ]
    workers: list[list[Reply]] = [[ASKS, answer("done 1")], [ASKS, answer("done 2")]]
    async with ask_console(tmp_path, monkeypatch, main=main, workers=workers) as (
        console,
        client,
        models,
    ):
        main_id = str(console.main_id)
        first = await start(console, "the first job")
        await until(lambda: len(models["main"].requests) == 1)
        second = await start(console, "the second job")
        await until(lambda: console.asks.waiting(second))
        response = await _post_turn(client, main_id, "hello there")
        # Not refused as busy: kept, and shown queued.
        assert response.status_code == 200, response.text
        queued = response.json()["echo"]
        assert (queued["echo"], queued["typed"]) == ("queued", "hello there")
        assert queued["said"]["key"] == "ask.queued"
        release.set()
        await ended(console, first)
        await ended(console, second)
        await _wait_turn(client, main_id)
        turns = await console.store.list_turns(main_id)
        # The person's message runs right after the first answer, before the second question.
        assert [t.origin for t in turns] == ["ask_owner", "", "ask_owner"]
        assert first in turns[0].user_text and second in turns[2].user_text
        assert (turns[1].user_text, turns[1].typed_text) == (
            "hello there",
            "hello there",
        )
        found = await rows(console, main_id)
        assert [r["of"] for r in found if r["echo"] == "dequeued"] == [queued["id"]]


async def test_two_tasks_asking_one_chat_get_one_internal_turn_at_a_time_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [
        held(release, answers("one")),
        answer("Answered."),
        answers("two"),
        answer("Answered."),
    ]
    workers: list[list[Reply]] = [[ASKS, answer("done")], [ASKS, answer("done")]]
    async with ask_console(tmp_path, monkeypatch, main=main, workers=workers) as (
        console,
        client,
        models,
    ):
        main_id = str(console.main_id)
        tasks = [await start(console, "job a"), await start(console, "job b")]
        await until(lambda: all(console.asks.waiting(t) for t in tasks))
        await until(lambda: len(models["main"].requests) == 1)
        # Both asked; one internal turn runs, the other waits for it.
        assert len(console.asks._busy) == 1
        assert sum(len(q) for q in console.asks._waiting.values()) == 1
        release.set()
        for task_id in tasks:
            await ended(console, task_id)
        asked = [
            r["task_id"]
            for r in await rows(console, main_id)
            if r["echo"] == "task_ask"
        ]
        turns = await console.store.list_turns(main_id)
        assert [asked_in(t.user_text) for t in turns] == asked


def asked_in(text: str) -> str:
    found = re.match(r"\[task ([0-9a-f]{12}) ", text)
    assert found
    return found.group(1)


async def test_the_main_chat_answers_when_the_tasks_side_chat_was_deleted_while_it_waited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [answers("from the main chat"), answer("Answered.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, workers=[[ASKS, answer("done")]]
    ) as (console, client, models):
        created = await client.post(
            "/api/sessions",
            json={"turn": {"text": "side work"}, "workspace": str(tmp_path / "work")},
        )
        side = created.json()["session"]["id"]
        await _wait_turn(client, side)
        go = asyncio.Event()
        idle = console.asks._idle

        async def waits(chat_id: str) -> None:
            await go.wait()
            await idle(chat_id)

        monkeypatch.setattr(console.asks, "_idle", waits)
        task_id = await start(console, "side job", title="Side job", front_id=side)
        await until(lambda: console.asks.waiting(task_id))
        assert [r["echo"] for r in await rows(console, side)] == ["task_ask"]
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        go.set()
        await ended(console, task_id)
        [ask_row, answered] = await rows(console, console.main_id)
        assert (ask_row["echo"], ask_row["from_chat"]) == ("task_ask", "Side job")
        assert answered["echo"] == "task_answered"
        [result] = tool_results(models["workers"][0])
        assert result.endswith("from the main chat")
        [turn] = await console.store.list_turns(str(console.main_id))
        assert turn.origin == "ask_owner"


async def test_a_side_chat_answers_its_own_tasks_question_and_the_main_chat_gets_no_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    side_script: list[Reply] = [
        answer("Hello."),
        answers("Pin the version."),
        answer("Told the task."),
    ]
    async with ask_console(
        tmp_path, monkeypatch, sides=[side_script], workers=[[ASKS, answer("done")]]
    ) as (console, client, models):
        created = await client.post(
            "/api/sessions",
            json={"turn": {"text": "side work"}, "workspace": str(tmp_path / "work")},
        )
        side = created.json()["session"]["id"]
        await _wait_turn(client, side)
        task_id = await start(console, "side job", front_id=side)
        await ended(console, task_id)
        await _wait_turn(client, side)
        assert [r["echo"] for r in await rows(console, side)] == [
            "task_ask",
            "task_answered",
        ]
        assert [t.origin for t in await console.store.list_turns(side)] == [
            "",
            "ask_owner",
        ]
        assert models[side].offered[1].count("task_answer") == 1
        await console.cards.flush()
        cards = await console.store.events_of_type(
            str(console.main_id), "attention.card", limit=10
        )
        # The side chat's own first answer is a card; its answer to the task is not.
        assert len(cards) == 1
        assert await rows(console, console.main_id) == []


# ── No answer, stops, restarts ──


async def test_the_chat_has_its_time_from_its_internal_turns_start_then_the_worker_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, never = asyncio.Event(), asyncio.Event()
    main: list[Reply] = [
        held(release, answer("Here you go.")),
        held(never, answer("x")),
    ]
    async with ask_console(
        tmp_path, monkeypatch, main=main, workers=[[ASKS, answer("went on")]]
    ) as (console, client, models):
        console.asks.timeout_s = 0.3
        main_id = str(console.main_id)
        # The person's own long turn: the question waits for it.
        assert (await _post_turn(client, main_id, "a long question")).status_code == 202
        task_id = await start(console, "a job")
        await until(lambda: console.asks.waiting(task_id))
        await asyncio.sleep(0.6)
        assert console.asks.waiting(task_id) is not None, "its time has not started"
        release.set()
        started = time.monotonic()
        await ended(console, task_id)
        assert time.monotonic() - started >= 0.3
        [result] = tool_results(models["workers"][0])
        assert result.startswith("No answer from ")
        assert result.endswith("go on with your best option and say so in your report")
        ask_row, end = [
            r for r in await rows(console, main_id) if r["task_id"] == task_id
        ]
        assert (end["echo"], end["how"], end["of"]) == (
            "ask_end",
            "timeout",
            ask_row["id"],
        )
        turns = await console.store.list_turns(main_id)
        assert [t.origin for t in turns] == ["", "ask_owner"]


async def test_a_question_to_the_person_waits_with_no_time_limit_and_a_stop_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [asks_person, answer("The task needs your answer.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, workers=[[ASKS, answer("never")]]
    ) as (console, client, models):
        console.asks.timeout_s = 0.05
        task_id = await start(console, "a job")
        await until(lambda: (console.asks.waiting(task_id) or {}).get("on") == "person")
        await _wait_turn(client, str(console.main_id))
        await asyncio.sleep(0.2)
        assert (await console.workers.get(task_id))["ask"]["on"] == "person"
        activity = (await item_activity(console, "task", task_id))[0]
        assert (activity.text, activity.icon) == ("Waiting on you", "wait")
        await console.workers.update(task_id, state="stop")
        await ended(console, task_id, "cancelled")
        await console.asks.settle()
        ask_row, escalation, end = await rows(console, console.main_id)
        assert escalation["echo"] == "task_escalate" and escalation["parts"] == PARTS
        assert (end["echo"], end["how"], end["of"]) == (
            "ask_end",
            "stopped",
            escalation["id"],
        )
        info = await console.store.get_session(task_id)
        assert info is not None and info.meta.get("ask") is None


@pytest.mark.parametrize("via", ["box", "message"])
async def test_the_persons_reply_answers_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    main: list[Reply] = [asks_person, answer("The task needs your answer.")]
    heard: list[StoredEvent] = []
    async with ask_console(
        tmp_path, monkeypatch, main=main, workers=[[ASKS, answer("done with B")]]
    ) as (console, client, models):
        console.asks.add_listener(heard.append)
        task_id = await start(console, "a job")
        await until(lambda: (console.asks.waiting(task_id) or {}).get("on") == "person")
        escalation = (await rows(console, console.main_id))[-1]
        if via == "box":
            url = f"/api/front/echoes/{escalation['id']}"
            response = await client.post(url, json={"do": "reply", "text": "B, please"})
            assert response.status_code == 200, response.text
            assert response.json()["echo"]["how"] == "person"
            again = await client.post(url, json={"do": "reply", "text": "A"})
            assert again.status_code == 409
        else:
            _, how = await console.workers.message(task_id, "B, please")
            assert how == "answered"
        await ended(console, task_id)
        [result] = tool_results(models["workers"][0])
        assert result == "[answer from the user, in their own words]\nB, please"
        end = (await rows(console, console.main_id))[-1]
        assert (end["how"], end["text"], end["of"]) == (
            "person",
            "B, please",
            escalation["id"],
        )
        assert heard[-1].event["data"]["by"] == "person"


async def test_a_stop_ends_the_wait_at_once_and_a_late_answer_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, answers("too late")), answer("ok")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, workers=[[ASKS, answer("never")]]
    ) as (console, client, models):
        task_id = await start(console, "a job")
        await until(lambda: len(models["main"].requests) == 1)
        await console.workers.update(task_id, state="stop")
        await ended(console, task_id, "cancelled")
        await console.asks.settle()
        ask_row, end = await rows(console, console.main_id)
        assert (end["echo"], end["how"], end["of"]) == (
            "ask_end",
            "stopped",
            ask_row["id"],
        )
        release.set()
        await _wait_turn(client, str(console.main_id))
        late = tool_results(models["main"], 1)[-1]
        assert (
            late == "Not done: the question is closed (stopped); nothing waits for it."
        )


async def test_a_restart_marks_a_pending_ask_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, client, _):
        task_id = await start(console, "a job", title="A job")
        main_id = str(console.main_id)
        # What a crash leaves: the ask's mirror on the task, with its row in the chat.
        mirror = {"ask_id": "a1", "chat_id": main_id, "row": 7}
        await console.store.update_session(task_id, meta_update={"ask": mirror})
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, client, _):
        [expired] = await rows(console, main_id)
        assert (expired["echo"], expired["how"], expired["of"]) == (
            "ask_end",
            "expired",
            7,
        )
        assert (expired["task_id"], expired["said"]["key"]) == (task_id, "ask.expired")
        info = await console.store.get_session(task_id)
        assert info is not None and info.meta.get("ask") is None


# ── Approvals stay the person's ──


async def test_the_internal_turn_cannot_approve_a_workers_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "work" / "out.txt"
    writes = tool_call("file_write", {"path": str(target), "content": "x"}, "w-1")
    approve = {**FIELDS, "help_needed": "Approve the other task's file write."}
    main: list[Reply] = [answers("I cannot approve that."), answer("Answered.")]
    workers: list[list[Reply]] = [
        [writes, answer("written")],
        [tool_call("ask_owner", approve, "ask-1"), answer("done")],
    ]
    async with ask_console(tmp_path, monkeypatch, main=main, workers=workers) as (
        console,
        client,
        models,
    ):
        writer = await start(console, "write the file")
        await until(lambda: console.broker.pending(writer))
        asker = await start(console, "ask for it")
        await ended(console, asker)
        offered = set(models["main"].offered[0])
        assert {"task_answer", "ask_person"} <= offered
        assert not [name for name in offered if "approv" in name or "allow" in name]
        # The person's approval still waits; nothing the chat did answered it.
        assert console.broker.pending(writer)
        assert not target.exists()


# ── The worker's tool ──


class _Owner:
    def __init__(self) -> None:
        self.asked: list[dict[str, str]] = []

    async def ask(self, task_id: str, fields: dict[str, str]) -> str:
        self.asked.append(fields)
        return "the answer"

    async def reply(
        self, task_id: str, text: str, ask_id: str | None = None
    ) -> dict[str, Any] | None:
        return None

    def waiting(self, task_id: str) -> dict[str, Any] | None:
        return None


async def test_ask_owner_needs_every_field_and_asks_at_most_three_times_a_run() -> None:
    owner = _Owner()
    tool = AskOwner("t1", lambda: owner)
    assert await tool.ask(**{**FIELDS, "problem": "  "}) == "Not asked: fill problem"
    missing = {k: v for k, v in FIELDS.items() if k != "options"}
    assert await tool.ask(**missing) == "Not asked: fill options"
    too_long = {**FIELDS, "situation": "x" * 601}
    assert (
        await tool.ask(**too_long) == "Not asked: situation has at most 600 characters"
    )
    for _ in range(MAX_ASKS):
        assert await tool.ask(**FIELDS) == "the answer"
    assert await tool.ask(**FIELDS) == (
        "Not asked: you asked 3 times; go on with your best option and say so in "
        "your report"
    )
    assert len(owner.asked) == MAX_ASKS
    lone = AskOwner("t2", lambda: None)
    assert (await lone.ask(**FIELDS)).startswith("Not asked: no chat can answer;")
