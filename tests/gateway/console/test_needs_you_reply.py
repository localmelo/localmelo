"""An answer to "this needs you" reaches the task that asked (#617).

A task the chat started in Auto has a step blocked and ends needing the person;
melo follows up on it. Each test runs the console on scripted models (see
``test_chat_owns_tasks``): nothing reaches a model server or the network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support.gateway.console import Console
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_ask_owner import Reply, held, rows, start, tool_results
from .test_chat_owns_tasks import (
    INSTALL,
    _is,
    calls_of,
    creates,
    owner_console,
    say,
    settled,
    started_task,
    until,
)
from .test_turn_writes import _post_turn, _wait_turn

ASKED = "may I install the package the check-in script needs?"
REPORT = (
    f"The check-in script needs a package I may not install.\nEnding: blocked: {ASKED}"
)
# The task's run: an install Auto blocks (a kind of step words may allow), then a
# report that needs the person.
BLOCKED: list[Reply] = [
    tool_call("python_exec", {"code": INSTALL}, "w-1"),
    answer(REPORT),
]
OTHER = INSTALL.replace("requests", "httpx")


async def typed(client: httpx.AsyncClient, console: Console, text: str) -> Any:
    """The person types *text* in the main chat: what the gateway answered."""
    response = await _post_turn(client, str(console.main_id), text)
    assert response.status_code in (200, 202), response.text
    return response.json()


async def needs_you(client: httpx.AsyncClient, console: Console) -> str:
    """A task of the main chat that ended needing the person, after melo's follow-up."""
    await say(client, console, "check in on the site and write a note")
    task_id = await started_task(console)
    await until(lambda: _is(console, task_id, "done"))
    await settled(console)
    assert len(await console.workers.open_blocks(task_id)) == 1
    return task_id


def follows_up(text: str = "It needs you: the note was blocked.") -> list[Reply]:
    return [creates("Daily check-in"), answer("Started it as a task."), answer(text)]


async def typed_turns(console: Console) -> list[str]:
    return [
        turn.typed_text
        for turn in await console.store.list_turns(str(console.main_id))
        if not turn.origin
    ]


async def test_a_typed_reply_goes_to_the_one_task_that_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "write that the day went well"
    workers: list[list[Reply]] = [BLOCKED, [answer("Wrote the note.\nEnding: ready")]]
    main = [*follows_up(), answer("The note is written.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="auto"
    ) as (console, client, models, _):
        console.asks.undo_s = 0.05
        task_id = await needs_you(client, console)
        echo = (await typed(client, console, reply))["echo"]
        assert (echo["echo"], echo["task_id"], echo["typed"]) == (
            "answer_sent",
            task_id,
            reply,
        )
        assert echo["said"]["key"] == "ask.sentAnswer" and echo["undo_s"] == 0.05
        await until(
            lambda: len(models["workers"]) == 2 and models["workers"][1].requests
        )
        # Its job went on with the person's words, by the path a named message takes.
        head = await console.workers.newest(
            await console.workers.tasks.get_task(task_id)
        )
        assert head.task_id != task_id
        first = "\n".join(str(m.content) for m in models["workers"][1].requests[0])
        assert reply in first
        await settled(console)
        kinds = [r["echo"] for r in await rows(console, console.main_id)]
        assert kinds[kinds.index("answer_sent") :][:3] == [
            "answer_sent",
            "started",
            "answer_delivered",
        ]
        # No model read the reply to route it: it is no turn of the chat.
        assert await typed_turns(console) == ["check in on the site and write a note"]


async def test_with_two_tasks_waiting_a_reply_goes_to_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path,
        monkeypatch,
        main=[answer("Which check-in do you mean?")],
        workers=[BLOCKED, BLOCKED],
        approvals="auto",
    ) as (console, client, _, _ran):
        tasks = [await start(console, f"check in on site {n}") for n in (1, 2)]
        for task_id in tasks:
            await until(lambda: _is(console, task_id, "done"))  # noqa: B023
        await settled(console)
        body = await typed(client, console, "允许")
        assert "echo" not in body, "a turn of the chat, not code's answer"
        await settled(console)
        assert await typed_turns(console) == ["允许"]
        for task_id in tasks:
            assert len(await console.workers.open_blocks(task_id)) == 1
        kinds = [r["echo"] for r in await rows(console, console.main_id)]
        assert "answer_sent" not in kinds and "sent" not in kinds


async def test_text_that_is_not_the_persons_never_delivers_and_the_refusal_names_the_button(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box: dict[str, str] = {}

    def message(messages: Any) -> Any:
        return tool_call("task_message", {"task_id": box["task"]}, "m-1")

    main: list[Reply] = [
        creates("Daily check-in"),
        answer("Started it as a task."),
        # melo's own follow-up, reading the task's report, tries to send it on.
        message,
        answer("It needs you."),
        # The person's question; the model, reading a tool result, tries again.
        message,
        answer("Tap Send under this reply."),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[BLOCKED], approvals="auto"
    ) as (console, client, models, ran):
        await say(client, console, "check in on the site and write a note")
        box["task"] = task_id = await started_task(console)
        await until(lambda: _is(console, task_id, "done"))
        await settled(console)
        assert tool_results(models["main"], 3)[-1].startswith("Not done: while you")
        await say(client, console, "它为什么停了？")
        refused = tool_results(models["main"], -1)[-1]
        assert refused.startswith("Not done: a change to a task needs the user's own")
        assert "“建议：把你的消息发给 Daily check-in” with a “发送” button" in refused
        assert "name no other button, card or control" in refused
        # Nothing reached the task: its job is where it ended, its step still blocked.
        record = await console.workers.tasks.get_task(task_id)
        assert (await console.workers.newest(record)).task_id == task_id
        assert len(await console.workers.open_blocks(task_id)) == 1
        assert ran.codes == []
        kinds = [r["echo"] for r in await rows(console, console.main_id)]
        assert "answer_sent" not in kinds and "sent" not in kinds
        assert kinds[-1] == "suggest"


async def test_a_plain_yes_allows_the_one_blocked_step_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers: list[list[Reply]] = [
        BLOCKED,
        [
            tool_call("python_exec", {"code": INSTALL}, "w-2"),
            answer("Typed the note.\nEnding: ready"),
        ],
    ]
    main = [*follows_up(), answer("The note is in."), answer("It is done already.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="auto"
    ) as (console, client, models, ran):
        task_id = await needs_you(client, console)
        [block] = await console.workers.open_blocks(task_id)
        echo = (await typed(client, console, "允许"))["echo"]
        assert (echo["echo"], echo["how"], echo["request_id"], echo["typed"]) == (
            "sent",
            "allowed",
            block["request_id"],
            "允许",
        )
        assert echo["said"]["key"] == "cards.allowedOnceStep"
        assert echo["said"]["params"]["what"]["key"] == "auto.title.install"
        await until(lambda: len(models["workers"]) == 2)
        head = await console.workers.newest(
            await console.workers.tasks.get_task(task_id)
        )
        await until(lambda: _is(console, head.task_id, "done"))
        await settled(console)
        assert ran.codes == [INSTALL], "the step ran once, on its allowance"
        # A second yes finds no task asking: it is a message to the chat, and
        # allows nothing more.
        assert "echo" not in await typed(client, console, "好")
        await settled(console)
        approvals = await calls_of(console, task_id, head.task_id)
        assert [a.get("allowance") for a in approvals].count("once") == 1
        assert await typed_turns(console) == [
            "check in on the site and write a note",
            "好",
        ]


async def nothing_allowed(console: Console, task_id: str, ran: Any) -> None:
    """The task's step is still blocked and never ran; the chat allowed nothing."""
    assert len(await console.workers.open_blocks(task_id)) == 1
    assert ran.codes == []
    kinds = [r.get("how") for r in await rows(console, console.main_id)]
    assert "allowed" not in kinds


async def test_a_card_that_came_during_the_persons_turn_is_not_what_a_yes_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [
        creates("Install check"),
        # The turn's reply ends with its own question, after the task's card came.
        held(release, answer("Started it. Shall I also check the logs?")),
        answer("I will check the logs."),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[BLOCKED], approvals="auto",
        follow_up=False,
    ) as (console, client, _, ran):  # fmt: skip
        chat = str(console.main_id)
        assert (await _post_turn(client, chat, "check the install")).status_code == 202
        await until(lambda: rows(console, chat))
        task_id = await started_task(console)
        await until(lambda: _is(console, task_id, "done"))
        await console.cards.flush()
        release.set()
        await _wait_turn(client, chat)
        # The person answers the reply's question, not the card above it.
        assert "echo" not in await typed(client, console, "好")
        await settled(console)
        assert (await typed_turns(console))[-1] == "好"
        await nothing_allowed(console, task_id, ran)


async def test_a_yes_allows_only_the_step_the_card_offers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=follows_up(), workers=[BLOCKED], approvals="auto"
    ) as (console, client, _, ran):
        task_id = await needs_you(client, console)
        await console.cards.flush()  # the card the person sees offers the real step
        live = console.workers.open_blocks

        # The one open step is not the one the card shows: the yes answers the task.
        async def another(task: str) -> list[dict[str, Any]]:
            return [{**block, "request_id": "another"} for block in await live(task)]

        monkeypatch.setattr(console.workers, "open_blocks", another)
        echo = (await typed(client, console, "ok"))["echo"]
        assert (echo["echo"], echo.get("task_id")) == ("answer_sent", task_id), echo
        monkeypatch.setattr(console.workers, "open_blocks", live)
        await nothing_allowed(console, task_id, ran)


async def test_a_lookup_that_fails_leaves_the_message_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = [*follows_up(), answer("Which step do you mean?")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[BLOCKED], approvals="auto"
    ) as (console, client, _, ran):
        task_id = await needs_you(client, console)

        async def broken() -> list[Any]:
            raise RuntimeError("the store is gone")

        monkeypatch.setattr(console.front, "_shown", broken)
        assert "echo" not in await typed(client, console, "好")
        await settled(console)
        assert (await typed_turns(console))[-1] == "好"
        await nothing_allowed(console, task_id, ran)


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [
            tool_call("python_exec", {"code": INSTALL}, "w-1"),
            tool_call("python_exec", {"code": OTHER}, "w-2"),
        ],
    ],
    ids=["no step", "two steps"],
)
async def test_a_yes_with_no_single_open_step_is_an_ordinary_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: list[Reply]
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=follows_up(), workers=[[*steps, answer(REPORT)]],
        approvals="auto",
    ) as (console, client, _, ran):  # fmt: skip
        await say(client, console, "check in on the site and write a note")
        task_id = await started_task(console)
        await until(lambda: _is(console, task_id, "done"))
        await settled(console)
        assert len(await console.workers.open_blocks(task_id)) == len(steps)
        echo = (await typed(client, console, "允许"))["echo"]
        assert (echo["echo"], echo["task_id"], echo["typed"]) == (
            "answer_sent",
            task_id,
            "允许",
        )
        assert ran.codes == []
