"""Words the person adds to a turn that runs: read at its next step, or sent as the next turn.

Each test runs the console with its main chat on a scripted model. A model reply
that waits for an event keeps a turn running while the test posts notes; nothing
reaches a model server.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import Message
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.asks import FORGET_TRIES
from localmelo.support.gateway.console.notes import NOTE_MARK
from localmelo.support.sessions.store import SessionStoreError

from .test_ask_owner import (
    ASKS,
    Reply,
    answer,
    answers,
    ask_console,
    ended,
    held,
    start,
    tool_call,
    until,
)
from .test_turn_writes import _post_turn, _wait_turn

# A harmless step: the run goes on to another planning request after it.
STEP = tool_call("task_get", {"task_id": "000000000000"}, "step-1")


async def _start(client: httpx.AsyncClient, console: Console, text: str) -> str:
    response = await _post_turn(client, str(console.main_id), text)
    assert response.status_code == 202, response.text
    return str(response.json()["turn_id"])


async def _note(
    client: httpx.AsyncClient, turn_id: str, text: str, **extra: Any
) -> httpx.Response:
    return await client.post(
        f"/api/turns/{turn_id}/notes", json={"text": text, **extra}
    )


async def _events(console: Console, turn: int) -> list[dict[str, Any]]:
    found = await console.store.list_events(str(console.main_id), turns=(turn, turn))
    return [{"type": e.event["type"], **e.event["data"]} for e in found]


def _seen(request: list[Message]) -> str:
    """The last user message of a planning request."""
    return str([m for m in request if m.role == "user"][-1].content)


async def test_a_note_is_read_at_the_next_step_and_marked_as_added_while_it_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("Done, with your note.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        turn_id = await _start(client, console, "start the job")
        await until(lambda: len(models["main"].requests) == 1)
        response = await _note(
            client, turn_id, "use the blue one", client_request_id="n1"
        )
        assert response.status_code == 202, response.text
        note = response.json()["note"]
        assert (note["state"], note["text"], note["turn_id"]) == (
            "sent",
            "use the blue one",
            turn_id,
        )
        # Kept in the queue store while it waits for the next step.
        info = await console.store.get_session(str(console.main_id))
        assert info is not None
        assert [q["typed"] for q in info.meta["queued"]] == ["use the blue one"]
        release.set()
        finished = await _wait_turn(client, str(console.main_id))
        assert finished["status"] == "completed"
        # The second request carries it as the person's message, marked.
        request = models["main"].requests[1]
        assert _seen(request).endswith(f"{NOTE_MARK}\nuse the blue one")
        events = await _events(console, 1)
        types = [event["type"] for event in events]
        assert types.index("turn.note") < types.index("inbox.accepted")
        assert (
            next(e for e in events if e["type"] == "turn.note")["note"] == note["note"]
        )
        assert next(e for e in events if e["type"] == "inbox.accepted")["offset"] == (
            note["note"]
        )
        # Read: it is not a next turn, and the queue store is empty again.
        await console.asks.settle()
        assert len(await console.store.list_turns(str(console.main_id))) == 1
        info = await console.store.get_session(str(console.main_id))
        assert info is not None and info.meta.get("queued") is None


async def test_a_note_after_the_last_step_becomes_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, answer("All done.")), answer("Got your note.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "start the job")
        await until(lambda: len(models["main"].requests) == 1)
        # The only step is under way: nothing will read the note before the turn ends.
        assert (await _note(client, turn_id, "one more thing")).status_code == 202
        release.set()

        async def ran() -> bool:
            turns = await console.store.list_turns(main_id)
            return len(turns) == 2 and turns[1].status != "running"

        await until(ran)
        first, second = await console.store.list_turns(main_id)
        assert first.user_text == "start the job" and first.status == "completed"
        # The note is the person's message of the next turn, as typed.
        assert (second.user_text, second.typed_text, second.origin) == (
            "one more thing",
            "one more thing",
            "",
        )
        assert [e["type"] for e in await _events(console, 1)].count(
            "inbox.accepted"
        ) == 0
        assert "turn.note" in [e["type"] for e in await _events(console, 1)]
        await console.asks.settle()
        info = await console.store.get_session(main_id)
        assert info is not None and info.meta.get("queued") is None


async def test_notes_keep_their_order_whether_they_are_read_or_become_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_release, second_release = asyncio.Event(), asyncio.Event()
    main: list[Reply] = [
        held(first_release, STEP),
        # Both notes arrive before the second request: it carries them in order.
        held(second_release, answer("Read both.")),
        answer("Third."),
        answer("Fourth."),
    ]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        for text in ("note a", "note b"):
            assert (await _note(client, turn_id, text)).status_code == 202
        first_release.set()
        await until(lambda: len(models["main"].requests) == 2)
        seen = _seen(models["main"].requests[1])
        assert seen.index("note a") < seen.index("note b")
        assert seen.count(NOTE_MARK) == 2
        # Two more arrive after that last step: they run as two turns, in order.
        for text in ("note c", "note d"):
            assert (await _note(client, turn_id, text)).status_code == 202
        second_release.set()

        async def ran() -> bool:
            turns = await console.store.list_turns(main_id)
            return len(turns) == 3 and all(t.status != "running" for t in turns)

        await until(ran)
        turns = await console.store.list_turns(main_id)
        assert [t.user_text for t in turns] == ["go", "note c", "note d"]


async def test_a_repeated_send_of_a_note_is_one_note_also_after_it_was_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("Done.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        one = await _note(client, turn_id, "only once", client_request_id="same-key")
        again = await _note(client, turn_id, "only once", client_request_id="same-key")
        assert one.json() == again.json()
        release.set()
        await _wait_turn(client, str(console.main_id))
        await console.asks.settle()
        # Read by now, and the same key still answers with the note it made.
        late = await client.post(
            f"/api/turns/{turn_id}/notes",
            json={"text": "only once", "client_request_id": "same-key"},
        )
        assert late.status_code == 202 and late.json() == one.json()
        events = await _events(console, 1)
        assert [e["type"] for e in events].count("turn.note") == 1
        assert len(await console.store.list_turns(str(console.main_id))) == 1


async def test_a_note_unread_before_a_restart_runs_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    async with ask_console(
        tmp_path, monkeypatch, main=[held(never, answer("stalled"))], run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        assert (await _note(client, turn_id, "after the restart")).status_code == 202
        info = await console.store.get_session(main_id)
        assert info is not None
        assert [q["typed"] for q in info.meta["queued"]] == ["after the restart"]
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Hi again.")], run=False
    ) as (console, client, _):

        async def ran() -> bool:
            turns = await console.store.list_turns(main_id)
            return turns[-1].user_text == "after the restart"

        await until(ran)
        await _wait_turn(client, main_id)
        info = await console.store.get_session(main_id)
        assert info is not None and info.meta.get("queued") is None


async def test_stop_hands_the_unread_notes_back_and_starts_no_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    async with ask_console(
        tmp_path, monkeypatch, main=[held(never, answer("stalled"))], run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        await _note(client, turn_id, "wait, not that")
        stopped = await client.post(f"/api/turns/{turn_id}/cancel")
        assert stopped.status_code == 202
        assert [n["text"] for n in stopped.json()["unread_notes"]] == ["wait, not that"]
        # A turn being stopped takes no more words.
        late = await _note(client, turn_id, "too late")
        assert (
            late.status_code == 409
            and late.json()["error"]["code"] == "turn_not_running"
        )
        await _wait_turn(client, main_id)
        await console.asks.settle()
        await asyncio.sleep(0.1)
        assert len(await console.store.list_turns(main_id)) == 1
        info = await console.store.get_session(main_id)
        assert info is not None and info.meta.get("queued") is None


async def test_a_note_is_refused_for_a_turn_that_is_not_running_or_a_bad_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    main: list[Reply] = [held(never, answer("Done."))]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        blank = await _note(client, turn_id, "   ")
        assert blank.status_code == 400 and blank.json()["error"]["code"] == (
            "invalid_request"
        )
        long = await _note(client, turn_id, "x" * 8_001)
        assert long.status_code == 400
        for number in range(20):
            assert (await _note(client, turn_id, f"n{number}")).status_code == 202
        many = await _note(client, turn_id, "one too many")
        assert (
            many.status_code == 400 and "at most 20" in many.json()["error"]["message"]
        )
        # Stop takes the unread ones back, so none starts another turn.
        assert (await client.post(f"/api/turns/{turn_id}/cancel")).status_code == 202
        await _wait_turn(client, str(console.main_id))
        unknown = await _note(client, "0" * 32, "hello")
        assert unknown.status_code == 409
        assert unknown.json()["error"]["code"] == "turn_not_running"
        ended = await _note(client, turn_id, "too late")
        assert ended.status_code == 409


async def _queued(console: Console, session_id: str) -> list[str]:
    """The typed words of the notes and messages waiting in *session_id*'s queue."""
    info = await console.store.get_session(session_id)
    assert info is not None
    return [q["typed"] for q in info.meta.get("queued") or []]


async def test_a_note_is_refused_for_the_turn_that_answers_a_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn that answers a worker is the system's, not the person's: words cannot join it."""
    release = asyncio.Event()
    main: list[Reply] = [held(release, answers("an answer")), answer("Answered.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, workers=[[ASKS, answer("done")]]
    ) as (console, client, models):
        main_id = str(console.main_id)
        task_id = await start(console, "the job")
        await until(lambda: len(models["main"].requests) == 1)
        turn = console.sessions._sessions[main_id].runner.current_turn
        assert turn is not None and turn.origin == "ask_owner"
        refused = await _note(client, turn.id, "also do this", client_request_id="n1")
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "not_your_turn"
        assert await _queued(console, main_id) == []
        release.set()
        await until(lambda: console.asks.waiting(task_id) is None)
        await console.asks.settle()
        # The refused words reached no turn.
        assert all(
            "also do this" not in str(m.content)
            for request in models["main"].requests
            for m in request
        )


async def test_a_note_is_refused_for_the_turn_of_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker's turn is the worker's: the person talks to its chat, never into the run."""
    release = asyncio.Event()
    async with ask_console(
        tmp_path,
        monkeypatch,
        workers=[[held(release, answer("done"))]],
    ) as (console, client, models):
        task_id = await start(console, "the job")
        await until(lambda: models["workers"] and models["workers"][0].requests)
        turns = await console.store.list_turns(task_id)
        assert [turn.status for turn in turns] == ["running"]
        refused = await _note(client, turns[0].id, "extra words", client_request_id="w")
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] in ("turn_not_running", "not_your_turn")
        assert await _queued(console, task_id) == []
        assert await _queued(console, str(console.main_id)) == []
        release.set()
        await ended(console, task_id)


async def test_a_note_goes_to_the_chat_that_owns_the_turn_and_to_no_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_release, side_release = asyncio.Event(), asyncio.Event()
    main: list[Reply] = [held(main_release, STEP), answer("Main done.")]
    side: list[Reply] = [held(side_release, STEP), answer("Side done.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, sides=[side], run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        main_turn = await _start(client, console, "main work")
        created = await client.post(
            "/api/sessions",
            json={"turn": {"text": "side work"}, "workspace": str(tmp_path / "work")},
        )
        side_id = created.json()["session"]["id"]
        side_turn = str(created.json()["turn_id"])
        await until(
            lambda: len(models["main"].requests) == 1
            and side_id in models
            and models[side_id].requests
        )
        response = await _note(client, side_turn, "for the side chat")
        assert response.status_code == 202, response.text
        assert response.json()["note"]["turn_id"] == side_turn
        assert await _queued(console, side_id) == ["for the side chat"]
        assert await _queued(console, main_id) == []
        side_release.set()
        await _wait_turn(client, side_id)
        main_release.set()
        await _wait_turn(client, main_id)
        assert main_turn
        assert all(
            "for the side chat" not in str(m.content)
            for request in models["main"].requests
            for m in request
        )
        assert any(
            "for the side chat" in str(m.content)
            for request in models[side_id].requests
            for m in request
        )


async def test_a_note_is_refused_once_stop_was_pressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    main: list[Reply] = [held(never, answer("Done."))]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        assert (await client.post(f"/api/turns/{turn_id}/cancel")).status_code == 202
        # Right after Stop, before the turn is seen to end, the words are not taken.
        late = await _note(client, turn_id, "wait, also this", client_request_id="late")
        assert late.status_code == 409
        assert late.json()["error"]["code"] == "turn_not_running"
        await _wait_turn(client, str(console.main_id))
        assert await _queued(console, str(console.main_id)) == []


@pytest.mark.parametrize("fails", [1, 99])
async def test_a_read_note_is_forgotten_by_the_stored_queue_or_never_runs_twice_in_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: int
) -> None:
    """The stored queue is what a restart runs, so removing a read note is tried again.

    One failed write is made good by the next try. If every try fails the note is
    still read once and starts no turn of its own; it stays stored, which is the
    known limit of a queue that is written after the fact (a crash then would run
    it again).
    """
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("Done, with your note.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        assert (await _note(client, turn_id, "read once")).status_code == 202
        assert await _queued(console, main_id) == ["read once"]
        update = console.store.update_session
        failed: list[int] = []

        async def flaky(session_id: str, **kwargs: Any) -> Any:
            if len(failed) < fails and "queued" in (kwargs.get("meta_update") or {}):
                failed.append(1)
                raise SessionStoreError("disk full")
            return await update(session_id, **kwargs)

        monkeypatch.setattr(console.store, "update_session", flaky)
        release.set()
        finished = await _wait_turn(client, main_id)
        assert finished["status"] == "completed"
        await console.asks.settle()
        # Read by the run exactly once, and not a turn of its own.
        assert _seen(models["main"].requests[1]).endswith(f"{NOTE_MARK}\nread once")
        assert len(await console.store.list_turns(main_id)) == 1
        assert not console.asks._people.get(main_id)
        assert await _queued(console, main_id) == (["read once"] if fails > 1 else [])
        assert len(failed) == min(fails, FORGET_TRIES)
