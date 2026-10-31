"""Side chats as equals, the landing edges: what authorizes a task, a chat that goes while work runs, news across a restart.

A task starts from a model's call only for a message code reads as work, with the
person's own message as its objective; deleting a side chat is one transition held
against the cards; news is told by task ending and kept in a note of the chat; a
chat made for a task is a submission of the create scope. Each test drives the
console's routes with real Agents on scripted models; no worker starts.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.console import front as front_module
from localmelo.support.gateway.console.front import (
    ECHO_EVENT,
    NEWS_ROWS,
    READING_HEAD,
    Front,
)
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_front import _say, _task
from .test_front_followup import _finished
from .test_side_equals import (
    _asked,
    _events,
    _hear,
    _open,
    _say_in,
    _side,
    _task_cards,
    chats_console,
)
from .test_side_equals_review import _as_task_body, _chats, _request
from .test_turn_writes import _wait_turn

# ── R1. A question, a matching title, a negation: no task from the model's call ──


@pytest.mark.parametrize(
    ("typed", "call"),
    [
        (
            "What does report.md say?",
            {"title": "Delete report", "objective": "Delete report.md"},
        ),
        (
            "Tell me about report.md",
            {"title": "Report", "objective": "Delete every other file"},
        ),
        (
            "Do not delete report.md; just explain what it contains.",
            {"title": "Delete report", "objective": "Delete report.md"},
        ),
    ],
)
async def test_a_title_that_matches_the_words_is_no_authority_to_start_a_task(
    tmp_path: Path, typed: str, call: dict[str, str]
) -> None:
    scripts = [[tool_call("task_create", call), answer("I offered it.")]]
    async with chats_console(tmp_path, sides=scripts) as (console, client, models):
        side = (await _open(client, typed))["session"]["id"]
        await _wait_turn(client, side)
        assert await console.workers.tasks.list_tasks() == []
        [offer] = await _events(console, side, ECHO_EVENT)
        assert (offer["echo"], offer["text"]) == ("ask", typed)
        assert str(models[side].requests[1][-1].content).startswith("Not done")


async def test_an_admitted_task_is_the_persons_message_and_the_models_reading_is_data(
    tmp_path: Path,
) -> None:
    # A work request that also holds a follow-up word and a task's name is left to the model.
    typed = (
        "Fix the Activity row text in support/gateway/console/activity.py and finish it"
    )
    call = {"title": "Activity row fix", "objective": "Also delete the build folder"}
    main = [tool_call("task_create", call), answer("Started.")]
    async with chats_console(tmp_path, main=main) as (console, client, _):
        await _say(client, console, "/task Activity row text")
        await _say(client, console, typed)
        await _wait_turn(client, str(console.main_id))
        [started] = [
            e
            for e in await _events(console, str(console.main_id), ECHO_EVENT)
            if e["echo"] == "started" and e["title"] == "Activity row fix"
        ]
        objective = (await _task(console, started["task_id"])).spec["objective"]
        # The person's words are the task; the title only names it; the rest is data.
        assert objective.startswith(typed)
        assert objective.endswith(f"{READING_HEAD}\nAlso delete the build folder")


# ── R2. A chat deleted while work ends and cards arrive hands all of it over ──


async def test_a_chat_deleted_while_tasks_end_and_cards_arrive_hands_them_to_the_main_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, main=[answer("Okay.")]) as (
        console,
        client,
        models,
    ):
        side = await _side(client, tmp_path, "Trip plan")
        finishing = (await _say_in(client, side, "/task Find hotels"))["task_id"]
        asking = (await _say_in(client, side, "/task Edit notes"))["task_id"]
        waiter = console.broker.begin(asking, _request(asking))
        real = console.sessions.delete
        racing: list[asyncio.Task[Any]] = []

        async def delete_with_traffic(*args: Any, **kwargs: Any) -> Any:
            # While the chat goes: a task ends, an approval is asked, a card is written.
            await _finished(console, finishing, "Three hotels.")
            data = {"request_id": "r1", "tool_name": "file_write"}
            _hear(console, asking, "approval.requested", data)
            racing.append(asyncio.create_task(client.get("/api/sessions")))
            await asyncio.sleep(0.05)
            return await real(*args, **kwargs)

        console.sessions.delete = delete_with_traffic  # type: ignore[method-assign]
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        await asyncio.gather(*racing)
        main = str(console.main_id)
        assert (await _task(console, finishing)).front_id == main
        assert (await _task(console, asking)).front_id == main
        cards = {(c["source_id"], c["state"]) for c in await _task_cards(console, main)}
        assert cards == {(finishing, "done"), (asking, "waiting")}
        # The cards and the news agree: the main chat's model reads the ended task.
        await _say(client, console, "what happened?")
        await _wait_turn(client, main)
        assert "Find hotels · done · P2 · Three hotels." in _asked(models["main"])
        waiter.cancel()


# ── R3, N2. News is told by task ending, and what was not told survives a restart ──


async def test_news_that_did_not_fit_is_told_after_a_restart_and_nothing_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(front_module, "REBUILD_PAGE", 2)  # the cards are read in pages
    count = NEWS_ROWS + 1
    sides = [[answer(f"Done {n}.")] for n in range(count)]
    async with chats_console(tmp_path, main=[answer("One.")], sides=sides) as (
        console,
        client,
        models,
    ):
        for n in range(count):
            await _side(client, tmp_path, f"Chat {n}")
        for n in range(count):
            echo = (await _say(client, console, f"/task Job {n}")).json()["echo"]
            await _finished(console, echo["task_id"], "Finished.")
        await _say(client, console, "first")
        await _wait_turn(client, str(console.main_id))
        first = _asked(models["main"])
        assert first.count("\nside chat ") == NEWS_ROWS
        assert first.count(" · done · ") == NEWS_ROWS
    async with chats_console(tmp_path, main=[answer("Two."), answer("Three.")]) as (
        console,
        client,
        models,
    ):
        await _say(client, console, "second")
        await _wait_turn(client, str(console.main_id))
        second = _asked(models["main"])
        # The one of each that did not fit before the restart, and no more.
        assert second.count("\nside chat ") == 1
        assert second.count(" · done · ") == 1
        await _say(client, console, "third")
        await _wait_turn(client, str(console.main_id))
        assert _asked(models["main"]) == "third"


async def test_a_task_that_continues_and_ends_again_is_news_again(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, main=[answer("One."), answer("Two.")]) as (
        console,
        client,
        models,
    ):
        # More ended than a message has room for: the ones told stay on the chat's note.
        tasks = []
        for n in range(NEWS_ROWS + 1):
            echo = (await _say(client, console, f"/task Job {n}")).json()["echo"]
            await _finished(console, echo["task_id"], f"First result {n}.")
            tasks.append(echo["task_id"])
        await _say(client, console, "one")
        await _wait_turn(client, str(console.main_id))
        assert f"First result {NEWS_ROWS}." in _asked(models["main"])
        # A follow-up continues the newest in its own session, and it ends again.
        await console.workers.message(tasks[-1], "and the summary too")
        await _finished(console, tasks[-1], "Second result.")
        await _say(client, console, "two")
        await _wait_turn(client, str(console.main_id))
        asked = _asked(models["main"])
        assert "Second result." in asked, "its new ending is news"
        assert "First result 0." in asked, "the one that did not fit comes next"
        assert f"First result {NEWS_ROWS}." not in asked


# ── R4. Every console chat's page is as new as what came back to it ──


async def test_a_task_made_through_the_api_shows_its_card_on_the_chats_page(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        body, _ = await console.workers.create(
            {"objective": "Find hotels", "front_id": side}
        )
        await _finished(console, body["task_id"], "Three hotels.")
        # No echo, no card yet, and the task is not live: only this page is read.
        page = (await client.get(f"/api/sessions/{side}/turns")).json()
        assert [(c["source_id"], c["state"]) for c in page["cards"]] == [
            (body["task_id"], "done")
        ]


# ── N1. A create for a task is a submission of the create scope ──


async def test_a_cancelled_key_does_not_make_a_chat_for_a_task(tmp_path: Path) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        cancelled = await client.post(
            "/api/submissions/create-w/cancel", json={"text": "Find hotels"}
        )
        assert cancelled.json()["submission"]["state"] == "withdrawn"
        made = await client.post(
            "/api/sessions", json=_as_task_body("Find hotels", "create-w")
        )
        assert made.status_code == 409
        assert made.json()["error"]["code"] == "submission_withdrawn"
        assert await _chats(console) == []
        assert await console.workers.tasks.list_tasks() == []


async def test_a_create_that_made_a_task_is_found_and_cancelled_as_a_submission(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        made = (
            await client.post(
                "/api/sessions", json=_as_task_body("Find hotels", "create-f")
            )
        ).json()
        found = (await client.get("/api/submissions/create-f")).json()["submission"]
        assert (found["state"], found["session_id"], found["task_id"]) == (
            "accepted",
            made["session"]["id"],
            made["echo"]["task_id"],
        )
        cancelled = await client.post(
            "/api/submissions/create-f/cancel", json={"text": "Find hotels"}
        )
        assert cancelled.status_code == 202, cancelled.text
        assert cancelled.json()["cancelling"] is True
        assert (await _task(console, made["echo"]["task_id"])).state == "cancelled"
        # The task is the answer to the key: no withdrawal was stored over it.
        again = await client.post(
            "/api/sessions", json=_as_task_body("Find hotels", "create-f")
        )
        assert again.status_code == 200 and again.json()["repeated"] is True
        assert await _chats(console) == [made["session"]["id"]]


async def test_a_key_accepted_for_a_message_does_not_make_a_chat_for_a_task(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        folder = Path.home() / "side-folder"
        folder.mkdir(exist_ok=True)
        first = await client.post(
            "/api/sessions",
            json={
                "workspace": str(folder),
                "turn": {"text": "Find hotels", "client_request_id": "create-m"},
            },
        )
        assert first.status_code == 201, first.text
        again = await client.post(
            "/api/sessions", json=_as_task_body("Find hotels", "create-m")
        )
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "submission_conflict"
        assert len(await _chats(console)) == 1


# ── N4. A task admitted without its echo is recovered on a retry ──


async def test_a_task_admitted_without_its_echo_keeps_its_chat_and_a_retry_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        real = Front._echo
        calls: list[str] = []

        async def broken(self: Front, *args: Any, **kwargs: Any) -> Any:
            calls.append("echo")
            if len(calls) == 1:
                raise RuntimeError("the echo could not be stored")
            return await real(self, *args, **kwargs)

        monkeypatch.setattr(Front, "_echo", broken)
        body = _as_task_body("Find hotels", "create-n")
        # The app's error reaches the test: the first attempt failed after the task.
        with contextlib.suppress(RuntimeError):
            await client.post("/api/sessions", json=body)
        [record] = await console.workers.tasks.list_tasks()
        owner = await console.store.get_session(str(record.front_id))
        assert (
            owner is not None
        ), "the task's chat was not deleted after the task was admitted"
        again = await client.post("/api/sessions", json=body)
        assert again.status_code == 200, again.text
        echo = again.json()["echo"]
        assert (echo["echo"], echo["task_id"]) == ("started", record.task_id)
        assert again.json()["session"]["id"] == owner.id
        assert len(await console.workers.tasks.list_tasks()) == 1
        assert owner.id in await _chats(console)
