"""Side chats as equals, the edges: whose task a model's call may change, what news a chat is told, a chat that goes.

A chat's words reach its own tasks first and another chat's only when they fit none
of its own; a model's task_create starts only what the person's words ask for; news
is told by id and survives a restart; deleting a side chat hands its unfinished
tasks and its waiting cards to the main chat. Each test drives the console's routes
with real Agents on scripted models; no worker starts.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.front import ECHO_EVENT, NEWS_ROWS
from localmelo.support.gateway.front_rules import (
    TaskRef,
    named_tasks,
    names_job,
    route,
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
from .test_turn_writes import _post_turn, _wait_turn

# A message longer than code's own reading of a control phrase (80 characters).
LONG_PAUSE = (
    "Pause the slides task please; keep the current files and notes untouched "
    "and do not change the task priority."
)


def _suggested(echoes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(e["action"], e["task_id"]) for e in echoes if e["echo"] == "suggest"]


# ── 1. A model's call reaches another chat's task only when the words do ──


async def test_a_model_call_on_another_chats_task_asks_which_when_this_chat_has_one(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        theirs = (await _say(client, console, "/task Slides")).json()["echo"]["task_id"]
        side = await _side(client, tmp_path, "Trip plan")
        mine = (await _say_in(client, side, "/task Slides draft"))["task_id"]
        assert (
            len(LONG_PAUSE) > 80
        ), "too long for code's own reading of a control phrase"
        # The model is asked in the chat that owns "Slides draft" and picks the other chat's "Slides".
        models[side].replies.extend(
            [
                tool_call("task_control", {"task_id": theirs, "action": "pause"}),
                answer("Asked."),
            ]
        )
        assert (await _post_turn(client, side, LONG_PAUSE)).status_code == 202
        await _wait_turn(client, side)
        assert (
            await _task(console, theirs)
        ).state == "queued", "the other chat's task is left alone"
        result = str(models[side].requests[-1][-1].content)
        assert result.startswith("Not done: the user's words fit more than one task")
        picks = _suggested(await _events(console, side, ECHO_EVENT))
        assert ("pause", mine) in picks and ("pause", theirs) in picks


async def test_a_model_call_on_another_chats_task_acts_when_the_words_name_it(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        theirs = (await _say(client, console, "/task Slides")).json()["echo"]["task_id"]
        setup = await _side(client, tmp_path, "Trip plan")
        await _say_in(client, setup, "/task Slides draft")
        for words, name in (
            (f"{LONG_PAUSE} It is task {theirs}.", "its id"),
            (f"{LONG_PAUSE} I mean the one in the main chat.", "its chat"),
        ):
            models[setup].replies.extend(
                [
                    tool_call("task_control", {"task_id": theirs, "action": "pause"}),
                    answer(name),
                ]
            )
            assert (await _post_turn(client, setup, words)).status_code == 202
            await _wait_turn(client, setup)
            assert (await _task(console, theirs)).state == "parked", name
            await console.workers.update(theirs, state="resume")


async def test_a_model_priority_or_message_for_another_chats_task_asks_which_too(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        theirs = (await _say(client, console, "/task Slides")).json()["echo"]["task_id"]
        setup = await _side(client, tmp_path, "Trip plan")
        mine = (await _say_in(client, setup, "/task Slides draft"))["task_id"]
        words = (
            "Move the slides to today, it is urgent and the review is this afternoon, "
            "so do it before anything else in the queue."
        )
        models[setup].replies.extend(
            [
                tool_call("task_set_priority", {"task_id": theirs, "level": "P1"}),
                answer("Asked."),
            ]
        )
        assert (await _post_turn(client, setup, words)).status_code == 202
        await _wait_turn(client, setup)
        assert (
            await _task(console, theirs)
        ).level == 2, "the other chat's level is kept"
        picks = _suggested(await _events(console, setup, ECHO_EVENT))
        assert ("level", mine) in picks and ("level", theirs) in picks
        models[setup].replies.extend(
            [tool_call("task_message", {"task_id": theirs}), answer("Asked.")]
        )
        said = (
            "Tell the slides task to use a dark theme and a wide layout for the room."
        )
        assert (await _post_turn(client, setup, said)).status_code == 202
        await _wait_turn(client, setup)
        assert theirs not in console.workers._inboxes, "nothing was sent to it"
        assert ("send", mine) in _suggested(await _events(console, setup, ECHO_EVENT))


# ── 2. Words that fit several tasks of this chat ask which, and stop there ──


async def test_words_that_fit_several_own_tasks_ask_which_and_never_reach_another_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        other = (await _say(client, console, "/task Slides")).json()["echo"]["task_id"]
        side = await _side(client, tmp_path, "Trip plan")
        tuesday = (await _say_in(client, side, "/task Slides Tuesday"))["task_id"]
        wednesday = (await _say_in(client, side, "/task Slides Wednesday"))["task_id"]
        asked = len(models[side].requests)
        echo = await _say_in(client, side, "pause the slides")
        assert (echo["echo"], echo["action"]) == ("suggest", "pause")
        picks = _suggested(await _events(console, side, ECHO_EVENT))[-2:]
        assert sorted(task for _, task in picks) == sorted([tuesday, wednesday])
        assert (await _task(console, other)).state == "queued"
        assert len(models[side].requests) == asked, "no model call"
        # A tap on one is the person's choice.
        tapped = await client.post(
            f"/api/front/echoes/{echo['id']}",
            json={"do": "apply", "session_id": side},
        )
        assert tapped.status_code == 200, tapped.text
        assert (await _task(console, echo["task_id"])).state == "parked"


def test_the_rules_ask_which_when_own_words_are_ambiguous_and_use_the_rest_only_for_none() -> (
    None
):
    own = [TaskRef("a1", "Slides Tuesday"), TaskRef("a2", "Slides Wednesday")]
    rest = [TaskRef("b1", "Slides"), TaskRef("b2", "Photos")]
    asked = route("pause the slides", own, fallback=rest)
    assert (asked.route, asked.pick) == ("choose", "pause")
    assert [m.task_id for m in asked.matches] == ["a1", "a2"]
    # Words that fit none of the chat's own reach the rest.
    reached = route("pause the photos", own, fallback=rest)
    assert (reached.route, reached.task) == ("control", rest[1])
    follow = route("finish the slides", own, fallback=rest)
    assert (follow.route, follow.pick) == ("choose", "send")
    assert [t.task_id for t in named_tasks("slides", "", own)] == ["a1", "a2"]


# ── 3. A model starts a task only for what the person's words ask ──


async def test_a_task_the_model_made_up_is_offered_not_started_in_a_side_chat(
    tmp_path: Path,
) -> None:
    scripts = [
        [
            tool_call(
                "task_create",
                {"title": "Clean up", "objective": "Delete the other folder"},
            ),
            answer("I offered it."),
        ]
    ]
    async with chats_console(tmp_path, sides=scripts) as (console, client, models):
        side = (await _open(client, "summarize report.md"))["session"]["id"]
        await _wait_turn(client, side)
        assert await console.workers.tasks.list_tasks() == []
        [offer] = await _events(console, side, ECHO_EVENT)
        assert (offer["echo"], offer["line"]) == ("ask", "Start this as a task?")
        assert (
            offer["text"] == "summarize report.md"
        ), "the person's words, not the model's"
        assert str(models[side].requests[1][-1].content).startswith("Not done")
        # One tap starts the person's own request.
        tapped = await client.post(
            f"/api/front/echoes/{offer['id']}", json={"do": "start", "session_id": side}
        )
        assert tapped.status_code == 200, tapped.text
        [record] = await console.workers.tasks.list_tasks()
        assert (record.front_id, record.spec["objective"]) == (
            side,
            "summarize report.md",
        )


async def test_a_task_the_model_made_up_is_offered_not_started_in_the_main_chat(
    tmp_path: Path,
) -> None:
    main = [
        tool_call(
            "task_create", {"title": "Clean up", "objective": "Delete the other folder"}
        ),
        answer("I offered it."),
    ]
    async with chats_console(tmp_path, main=main) as (console, client, models):
        await _say(client, console, "summarize report.md")
        await _wait_turn(client, str(console.main_id))
        assert await console.workers.tasks.list_tasks() == []
        [offer] = await _events(console, str(console.main_id), ECHO_EVENT)
        assert (offer["echo"], offer["text"]) == ("ask", "summarize report.md")


def test_a_job_is_the_persons_when_its_words_are() -> None:
    typed = "can you make slides from my notes today, urgent"
    assert names_job(typed, "Slides", "Make 6 slides")
    assert names_job(
        "research three laptops for me", "Laptop research", "Compare three laptops"
    )
    assert not names_job("summarize report.md", "Clean up", "Delete the other folder")
    assert not names_job(
        "summarize report.md",
        "Remove old folders",
        "Delete old folders, then summarize the report",
    )
    assert not names_job("", "Slides", "Make slides")
    assert names_job("把笔记做成幻灯片", "笔记幻灯片", "做幻灯片")


# ── 6, 7. A deleted chat's unfinished tasks and waiting cards are the main chat's ──


def _request(task_id: str) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r1",
        run_id=task_id,
        call_id="c1",
        tool_name="file_write",
        arguments={"path": "/tmp/notes.md", "content": "x"},
        preview="write notes.md",
        workspace_root="/tmp",
    )


async def test_deleting_a_chat_moves_its_waiting_cards_to_the_main_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        task_id = (await _say_in(client, side, "/task Edit notes"))["task_id"]
        waiter = console.broker.begin(task_id, _request(task_id))
        _hear(
            console,
            task_id,
            "approval.requested",
            {"request_id": "r1", "tool_name": "file_write"},
        )
        await client.get("/api/sessions")
        assert [c["state"] for c in await _task_cards(console, side)] == ["waiting"]
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        main = str(console.main_id)
        [waiting] = await _task_cards(console, main)
        assert (waiting["state"], waiting["request_id"]) == ("waiting", "r1")
        assert (await _task(console, task_id)).front_id == main
        # The approval is answered where the card is: the answer follows it in the main chat.
        await client.post("/api/approvals/r1", json={"choice": "once"})
        await waiter
        _hear(
            console,
            task_id,
            "approval.resolved",
            {"request_id": "r1", "outcome": "allowed"},
        )
        await client.get("/api/sessions")
        assert [c["state"] for c in await _task_cards(console, main)] == [
            "waiting",
            "answered",
        ]


async def test_a_task_of_a_deleted_chat_reports_to_the_main_chat_and_its_model(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, main=[answer("Okay.")]) as (
        console,
        client,
        models,
    ):
        side = await _side(client, tmp_path, "Trip plan")
        running = (await _say_in(client, side, "/task Find hotels"))["task_id"]
        kept = (await _say_in(client, side, "/task Book flights"))["task_id"]
        await _finished(console, kept, "Booked.")
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        main = str(console.main_id)
        # The unfinished task is the main chat's; the finished one keeps its record.
        assert (await _task(console, running)).front_id == main
        assert (await _task(console, kept)).front_id == side
        await _finished(console, running, "Three hotels.")
        await _say(client, console, "what happened?")
        await _wait_turn(client, main)
        assert "Find hotels · done · P2 · Three hotels." in _asked(models["main"])
        assert [c["source_id"] for c in await _task_cards(console, main)] == [running]


async def test_a_deleted_chats_chore_is_the_main_chats(tmp_path: Path) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        proposed = await _say_in(
            client, side, "Brief me on the news every weekday at 8:30"
        )
        saved = await client.post(
            f"/api/front/echoes/{proposed['id']}",
            json={"do": "save", "session_id": side},
        )
        assert saved.status_code == 200, saved.text
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        [chore] = await console.chore_runner.store.list_chores()
        assert chore.front_id == console.main_id


# ── 8, 9. News is told by id, and survives a restart ──


async def test_news_leaves_what_does_not_fit_for_the_next_turn_and_tells_nothing_twice(
    tmp_path: Path,
) -> None:
    sides = [[answer(f"Done {n}.")] for n in range(NEWS_ROWS + 1)]
    async with chats_console(
        tmp_path, main=[answer("One."), answer("Two."), answer("Three.")], sides=sides
    ) as (console, client, models):
        for n in range(NEWS_ROWS + 1):
            await _side(client, tmp_path, f"Chat {n}")
        tasks = [
            (await _say(client, console, f"/task Job {n}")).json()["echo"]["task_id"]
            for n in range(NEWS_ROWS + 1)
        ]
        for task_id in tasks:
            await _finished(console, task_id, "Finished.")
        told: list[str] = []
        for text in ("first", "second", "third"):
            await _say(client, console, text)
            await _wait_turn(client, str(console.main_id))
            told.append(_asked(models["main"]))
        sides_told = [
            r
            for block in told
            for r in block.splitlines()
            if r.startswith("side chat ")
        ]
        jobs_told = [
            r for block in told for r in block.splitlines() if " · done · " in r
        ]
        assert len(sides_told) == len(set(sides_told)) == NEWS_ROWS + 1
        assert len(jobs_told) == NEWS_ROWS + 1
        assert (
            sum("side chat " in block for block in told) == 2
        ), "the sixth waits for the next turn"
        assert "side chat" not in told[2] and " · done · " not in told[2]


async def test_a_summary_that_lands_while_the_news_is_read_is_told_once(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path, main=[answer("One."), answer("Two.")]) as (
        console,
        client,
        models,
    ):
        real = console.cards.flush
        landed: list[bool] = []

        async def flush_and_land() -> None:
            await real()
            if not landed:  # once, in the middle of the first reading
                landed.append(True)
                text = "Landed during the read."
                await console.front.side_done("late-chat", "Late", text, time.time())

        console.cards.flush = flush_and_land  # type: ignore[method-assign]
        await _say(client, console, "first")
        await _wait_turn(client, str(console.main_id))
        console.cards.flush = real  # type: ignore[method-assign]
        await _say(client, console, "second")
        await _wait_turn(client, str(console.main_id))
        assert 'side chat "Late": Landed during the read.' in _asked(models["main"], 0)
        assert _asked(models["main"]) == "second"


async def test_a_restart_does_not_lose_what_the_main_chat_was_never_told(
    tmp_path: Path,
) -> None:
    async with chats_console(
        tmp_path, sides=[[answer("The trip is planned for May.")]]
    ) as (
        console,
        client,
        _,
    ):
        trip = await _side(client, tmp_path, "Plan the trip")
        await client.get("/api/sessions")  # the card is written
        assert trip
    # A new gateway on the same file: nothing in memory, the main chat not yet told.
    async with chats_console(tmp_path, main=[answer("Okay."), answer("Again.")]) as (
        console,
        client,
        models,
    ):
        heard: list[Any] = []
        console.front.add_side_listener(heard.append)
        await _say(client, console, "what happened?")
        await _wait_turn(client, str(console.main_id))
        assert 'side chat "Plan the trip": The trip is planned for May.' in _asked(
            models["main"]
        )
        assert heard == [], "rebuilding tells the memory lane nothing again"
        # Told, and a restart after that does not tell it again.
        await _say(client, console, "and now?")
        await _wait_turn(client, str(console.main_id))
        assert _asked(models["main"]) == "and now?"
    async with chats_console(tmp_path, main=[answer("Fine.")]) as (
        console,
        client,
        models,
    ):
        await _say(client, console, "once more")
        await _wait_turn(client, str(console.main_id))
        assert _asked(models["main"]) == "once more"


# ── 10, 13. A chat that took a task by code has no turn, and is in the list ──


async def test_a_side_chat_that_took_a_task_by_code_is_in_the_list_without_a_turn(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        made = await client.post("/api/sessions", json={})
        assert made.status_code == 201, made.text
        empty = made.json()["session"]["id"]
        before = (await client.get("/api/sessions")).json()["sessions"]
        assert empty not in [
            s["id"] for s in before
        ], "an empty chat is still not listed"
        echo = await _say_in(client, empty, "/task Find hotels")
        assert echo["echo"] == "started"
        listed = (await client.get("/api/sessions")).json()["sessions"]
        assert empty in [s["id"] for s in listed]


async def test_as_task_on_a_new_chat_makes_the_chat_first_and_the_task_belongs_to_it(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, models):
        folder = Path.home() / "side-folder"
        folder.mkdir(exist_ok=True)
        made = await client.post(
            "/api/sessions",
            json={
                "workspace": str(folder),
                "turn": {"text": "Find hotels", "as_task": True},
            },
        )
        assert made.status_code == 201, made.text
        body = made.json()
        chat = body["session"]["id"]
        assert body["echo"]["echo"] == "started" and "turn" not in body
        assert body["session"]["title"] == "Find hotels"
        assert (await _task(console, body["echo"]["task_id"])).front_id == chat
        assert models[chat].requests == [], "no model was asked"
        assert chat in [
            s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]
        ]


async def test_as_task_on_a_new_chat_that_cannot_start_leaves_no_chat(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from localmelo.support.gateway.console.errors import ApiError
    from localmelo.support.gateway.console.front import Front

    async def refuse(self: Front, *args: Any, **kwargs: Any) -> Any:
        raise ApiError(400, "invalid_request", "No.")

    monkeypatch.setattr(Front, "route", refuse)
    async with chats_console(tmp_path) as (console, client, _):
        made = await client.post(
            "/api/sessions", json={"turn": {"text": "Find hotels", "as_task": True}}
        )
        assert made.status_code == 400
        listed = await console.store.list_sessions(kinds=("chat",))
        assert listed == []


def _as_task_body(text: str, key: str | None = None) -> dict[str, Any]:
    folder = Path.home() / "side-folder"
    folder.mkdir(exist_ok=True)
    turn: dict[str, Any] = {"text": text, "as_task": True}
    if key is not None:
        turn["client_request_id"] = key
    return {"workspace": str(folder), "turn": turn}


async def _chats(console: Console) -> list[str]:
    return [s.id for s in await console.store.list_sessions(kinds=("chat",))]


async def test_a_repeated_as_task_create_answers_the_same_chat_and_echo(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        body = _as_task_body("Find hotels", "create-1")
        first = await client.post("/api/sessions", json=body)
        again = await client.post("/api/sessions", json=body)
        assert (first.status_code, again.status_code) == (201, 200), again.text
        one, two = first.json(), again.json()
        assert two["repeated"] is True and "repeated" not in one
        assert two["session"]["id"] == one["session"]["id"]
        assert two["echo"]["id"] == one["echo"]["id"]
        assert two["echo"]["task_id"] == one["echo"]["task_id"]
        assert await _chats(console) == [one["session"]["id"]], "no second chat"
        assert len(await console.workers.tasks.list_tasks()) == 1


async def test_two_as_task_creates_at_once_make_one_chat(tmp_path: Path) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        body = _as_task_body("Find hotels", "create-2")
        answers = await asyncio.gather(
            client.post("/api/sessions", json=body),
            client.post("/api/sessions", json=body),
        )
        assert sorted(a.status_code for a in answers) == [200, 201]
        assert len({a.json()["session"]["id"] for a in answers}) == 1
        assert len(await _chats(console)) == 1
        assert len(await console.workers.tasks.list_tasks()) == 1


async def test_a_key_used_for_another_message_is_a_conflict_and_makes_no_chat(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        made = await client.post(
            "/api/sessions", json=_as_task_body("Find hotels", "create-3")
        )
        other = await client.post(
            "/api/sessions", json=_as_task_body("Book flights", "create-3")
        )
        assert other.status_code == 409
        assert other.json()["error"]["code"] == "submission_conflict"
        assert await _chats(console) == [made.json()["session"]["id"]]


async def test_a_key_used_by_a_task_in_a_chat_returns_that_chat(tmp_path: Path) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        sent = await _post_turn(
            client, side, "/task Find hotels", client_request_id="create-4"
        )
        assert sent.status_code == 200, sent.text
        again = await client.post(
            "/api/sessions", json=_as_task_body("/task Find hotels", "create-4")
        )
        # The same job under the same key: the chat that has it, not a second one.
        assert again.status_code == 200, again.text
        assert again.json()["session"]["id"] == side
        assert again.json()["repeated"] is True
        assert await _chats(console) == [side]


async def test_an_as_task_create_whose_task_does_not_start_leaves_no_chat(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from localmelo.support.gateway.task_policy import TaskPolicyError

    async with chats_console(tmp_path) as (console, client, _):

        async def refuse(*args: Any, **kwargs: Any) -> Any:
            raise TaskPolicyError("The objective is not allowed.")

        monkeypatch.setattr(console.workers, "create_spec", refuse)
        made = await client.post("/api/sessions", json=_as_task_body("Find hotels"))
        assert made.status_code == 400, made.text
        assert "not allowed" in made.json()["error"]["message"]
        assert await _chats(console) == []


# ── 11. A side chat's page is as new as its tasks ──


async def test_reading_a_side_chats_page_writes_the_cards_of_its_task(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        task_id = (await _say_in(client, side, "/task Find hotels"))["task_id"]
        await _finished(console, task_id, "Three hotels.")
        # No list, no main chat read: only this chat's page.
        page = (await client.get(f"/api/sessions/{side}/turns")).json()
        assert [(c["source_id"], c["state"]) for c in page["cards"]] == [
            (task_id, "done")
        ]


# ── 14. The task list keeps this chat's tasks on the page ──


async def test_the_task_list_keeps_this_chats_finished_task_among_many_of_the_others(
    tmp_path: Path,
) -> None:
    async with chats_console(tmp_path) as (console, client, _):
        side = await _side(client, tmp_path, "Trip plan")
        mine = (await _say_in(client, side, "/task Find hotels"))["task_id"]
        await _finished(console, mine, "Three hotels.")
        for n in range(21):
            await _say(client, console, f"/task Other job {n}")
        listed = await console.front_for(side).tools.list_tasks()
        rows = listed.splitlines()[1:]
        assert rows[0].startswith(mine) and "Three hotels." in rows[0]
        assert len(rows) <= 20
