"""The main chat: one per file, never deleted, side chats under it, cards in it."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.cards import CARD_EVENT
from localmelo.support.sessions.store import (
    MAIN_CHAT_TITLE,
    MainChatError,
    SessionStore,
    StoredEvent,
)
from localmelo.tests.sessions.helpers import answer, tool_call

from .conftest import (
    HOLDER,
    FakeBuilder,
    FakeProbe,
    all_events,
    make_cfg,
    quiet_gate,
    run_result,
    seed_turn,
)
from .test_turn_writes import (
    PORT,
    _create,
    _post_turn,
    _wait_pending,
    _wait_turn,
    console_with,
)


def _console(path: Path) -> Console:
    return Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(path),
            builder=FakeBuilder(),
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )


def _fronts(path: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as conn:
        return list(conn.execute("SELECT id, title FROM sessions WHERE kind = 'front'"))


# ── One per file ──


async def test_the_main_chat_is_made_once_and_kept(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    first = _console(path)
    await first.start()
    main_id = first.main_id
    await first.aclose()
    again = _console(path)
    await again.start()
    try:
        assert again.main_id == main_id
        assert _fronts(path) == [(main_id, MAIN_CHAT_TITLE)]
    finally:
        await again.aclose()


async def test_two_gateways_racing_make_one_main_chat(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    one, two = _console(path), _console(path)
    await asyncio.gather(one.start(), two.start())
    try:
        states = sorted([one.store_state(), two.store_state()])
        assert states == ["ok", "unavailable"]  # the owner lock: one gateway writes
        assert len(_fronts(path)) == 1
    finally:
        await asyncio.gather(one.aclose(), two.aclose())


async def test_one_store_asked_at_once_makes_one_and_the_index_refuses_two(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(str(path), writer="gateway")
    try:
        found = await asyncio.gather(*(store.main_chat() for _ in range(5)))
        assert len({info.id for info in found}) == 1
    finally:
        await store.aclose()
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO sessions (id, kind, created_at, updated_at) "
            "VALUES ('second', 'front', 0, 0)"
        )


async def test_a_file_with_two_fronts_keeps_its_oldest_and_still_opens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(str(path), writer="gateway")
    await store.list_sessions()  # creates the file, without the main chat's index
    await store.aclose()
    with sqlite3.connect(path) as conn:
        for name, at in (("newer", 2), ("older", 1)):
            conn.execute(
                "INSERT INTO sessions (id, kind, created_at, updated_at) "
                "VALUES (?, 'front', ?, ?)",
                (name, at, at),
            )
    again = SessionStore(str(path), writer="gateway")
    try:
        assert (await again.main_chat()).id == "older"
    finally:
        await again.aclose()


async def test_a_terminal_file_has_no_main_chat(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "cli.db"), writer="cli")
    try:
        with pytest.raises(Exception, match="gateway's store"):
            await store.main_chat()
    finally:
        await store.aclose()


async def test_existing_chats_become_side_chats(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    old = SessionStore(str(path), writer="gateway")
    chats = [await seed_turn(old, str(workspace), title=title) for title in ("A", "B")]
    await old.aclose()
    assert _fronts(path) == []
    console = _console(path)
    await console.start()
    app = _app(console)
    try:
        async with _client(app) as client:
            body = (await client.get("/api/sessions")).json()
            assert {s["id"] for s in body["sessions"]} == {c.session_id for c in chats}
            assert all(s["kind"] == "chat" and not s["main"] for s in body["sessions"])
            assert body["main"]["id"] == console.main_id
            assert body["main"]["main"] is True and body["main"]["kind"] == "front"
            turns = await client.get(f"/api/sessions/{chats[0].session_id}/turns")
            assert turns.json()["turns"][0]["answer"] == "an answer"
    finally:
        await console.aclose()


# ── Never deleted; side chats under it ──


async def test_the_main_chat_cannot_be_deleted(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        main_id = console.main_id
        assert main_id is not None
        response = await client.delete(f"/api/sessions/{main_id}")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "main_chat"
        with pytest.raises(MainChatError):
            await console.store.delete_session(main_id)
        assert await console.store.get_session(main_id) is not None


async def test_the_list_pins_the_main_chat_and_a_create_is_a_side_chat(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        body = (await client.get("/api/sessions")).json()
        assert body["sessions"] == []
        assert body["main"]["title"] == MAIN_CHAT_TITLE
        assert body["main"]["last_card"] is None
        # The main chat is listed before its first message, by its own kind.
        fronts = (await client.get("/api/sessions?kind=front")).json()["sessions"]
        assert [s["id"] for s in fronts] == [console.main_id]
        created = await _create(client, tmp_path, "Plan the trip")
        assert created["session"]["kind"] == "chat"
        assert created["session"]["main"] is False
        listed = (await client.get("/api/sessions")).json()
        assert [s["id"] for s in listed["sessions"]] == [created["session"]["id"]]
        assert console.main_id not in [s["id"] for s in listed["sessions"]]


# ── Cards ──


async def test_a_side_chat_run_comes_back_as_a_card_not_a_turn(
    tmp_path: Path,
) -> None:
    scripts = [[answer("The trip is planned. Details follow.")], [answer("Hi.")]]
    async with console_with(tmp_path, scripts=scripts) as (console, client, llms):
        side = (await _create(client, tmp_path, "Plan the trip"))["session"]["id"]
        await _wait_turn(client, side)
        main = (await client.get("/api/sessions")).json()["main"]
        card = main["last_card"]
        assert card["state"] == "done" and card["source_id"] == side
        assert card["source_kind"] == "chat"
        assert card["title"] == "Plan the trip"
        # A card carries the first two lines of the answer.
        assert card["line"] == "The trip is planned.\nDetails follow."
        page = (await client.get(f"/api/sessions/{console.main_id}/turns")).json()
        assert page["turns"] == []
        assert [c["id"] for c in page["cards"]] == [card["id"]]
        assert page["replay_from"] == card["id"] - 1
        assert console.main_id is not None
        [stored] = await all_events(console.store, console.main_id)
        assert stored.event["type"] == CARD_EVENT and stored.turn is None
        # The main chat's next turn: the card is not in what the model reads.
        response = await _post_turn(client, console.main_id, "Anything new?")
        assert response.status_code == 202
        await _wait_turn(client, console.main_id)
        seen = " ".join(str(m.content) for m in llms[1].requests[0])
        assert "Anything new?" in seen and "trip is planned" not in seen
        main_turns = await console.store.list_turns(console.main_id)
        assert [t.user_text for t in main_turns] == ["Anything new?"]


async def test_a_waiting_approval_gets_a_card_unless_answered_first(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "todo.md"
    write = tool_call("file_write", {"path": str(target), "content": "# TODO"})
    async with console_with(tmp_path, write, answer("done")) as (console, client, _):
        side = (await _create(client, tmp_path, "Write the todo"))["session"]["id"]
        [pending] = await _wait_pending(client, side)
        card = (await client.get("/api/sessions")).json()["main"]["last_card"]
        assert card["state"] == "waiting" and card["source_id"] == side
        assert card["line"] == "Wants to write todo.md"
        assert card["request_id"] == pending["request_id"]
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, side)
    other = tmp_path / "b"
    write = tool_call(
        "file_write", {"path": str(other / "ws" / "a.md"), "content": "a"}
    )
    async with console_with(other, write, answer("done")) as (console, client, _):
        side = (await _create(client, other, "Write"))["session"]["id"]
        [pending] = await _wait_pending(client, side)
        # Answered before anything read the main chat: no waiting card.
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, side)
        assert console.main_id is not None
        cards = await console.store.events_of_type(console.main_id, CARD_EVENT)
        assert cards == []
        await client.get("/api/sessions")
        cards = await console.store.events_of_type(console.main_id, CARD_EVENT)
        assert [c.event["data"]["state"] for c in cards] == ["done"]


async def test_an_answered_approval_follows_its_card_and_is_no_news(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "todo.md"
    write = tool_call("file_write", {"path": str(target), "content": "# TODO"})
    async with console_with(tmp_path, write, answer("done")) as (console, client, _):
        side = (await _create(client, tmp_path, "Write the todo"))["session"]["id"]
        [pending] = await _wait_pending(client, side)
        waiting = (await client.get("/api/sessions")).json()["main"]["last_card"]
        assert waiting["state"] == "waiting"
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "once"}
        )
        await _wait_turn(client, side)
        assert console.main_id is not None
        page = (await client.get(f"/api/sessions/{console.main_id}/turns")).json()
        states = [card["state"] for card in page["cards"]]
        assert states == ["waiting", "answered", "done"]
        _, answered, done = page["cards"]
        assert answered["request_id"] == waiting["request_id"] == pending["request_id"]
        assert answered["line"] == "Allowed once" and answered["source_id"] == side
        assert answered["title"] == "Write the todo"
        # The answer is the person's own act: the list's newest card is the done one.
        newest = (await client.get("/api/sessions")).json()["main"]["last_card"]
        assert newest["id"] == done["id"]
        assert console.cards.last is not None and console.cards.last["id"] == done["id"]


async def test_each_answer_reads_in_words_and_a_task_grant_is_for_the_task(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("ok")) as (console, client, _):
        main = console.main_id
        assert main is not None
        sources = {
            "chat": (await _create(client, tmp_path, "A chat"))["session"]["id"],
        }
        store = console.store
        record, _ = await console.workers.create(
            {"objective": "A task", "title": "A task", "request_key": "k"}
        )
        sources["task"] = record["task_id"]
        await _wait_turn(client, sources["chat"])
        ask = 0

        async def resolved(kind: str, **data: Any) -> dict[str, Any]:
            nonlocal ask
            ask += 1
            rid = f"r{ask}"
            source = sources[kind]
            await store.append_session_event(
                main,
                CARD_EVENT,
                {
                    "state": "waiting",
                    "request_id": rid,
                    "source_id": source,
                    "source_kind": kind,
                    "title": "x",
                    "line": "",
                },
            )
            event = {
                "type": "approval.resolved",
                "ts": 1.0,
                "data": {
                    "request_id": rid,
                    "outcome": "allowed",
                    "origin": "user",
                    **data,
                },
            }
            console.cards._heard(StoredEvent(0, source, None, event))
            await console.cards.flush()
            cards = await store.events_of_type(main, CARD_EVENT, limit=2)
            return cards[-1].event["data"]  # type: ignore[no-any-return]

        said = await resolved("task", remember=True)
        assert said["state"] == "answered" and said["line"] == "Allowed for this task"
        said = await resolved("chat", remember=True)
        assert said["line"] == "Allowed for this chat"
        assert (await resolved("task", remember=False))["line"] == "Allowed once"
        assert (await resolved("task", origin="policy"))[
            "line"
        ] == "Allowed automatically"
        assert (await resolved("task", outcome="denied"))["line"] == "Denied"
        assert (await resolved("task", origin="timeout", outcome="denied"))[
            "line"
        ] == "No answer in time"
        assert (await resolved("task", outcome="withdrawn", origin="withdrawn"))[
            "line"
        ] == "Not run: it stopped first"
        # An answer whose waiting card was never written adds no card.
        before = await store.events_of_type(main, CARD_EVENT, limit=100)
        event = {
            "type": "approval.resolved",
            "ts": 1.0,
            "data": {"request_id": "never", "outcome": "denied", "origin": "user"},
        }
        console.cards._heard(StoredEvent(0, sources["task"], None, event))
        await console.cards.flush()
        after = await store.events_of_type(main, CARD_EVENT, limit=100)
        assert len(after) == len(before)


async def test_a_card_title_is_short_and_a_failed_card_gives_the_reason(
    tmp_path: Path,
) -> None:
    long = (
        "Fix this today: in tests/cli/test_e2e_pty.py the list_files test only "
        "checks that the word list_files appears on screen."
    )
    async with console_with(tmp_path) as (console, client, _):
        store = console.store
        record, _ = await console.workers.create(
            {"objective": long, "request_key": "k"}
        )
        assert record["title"] == "Fix this today: in tests/cli/test_e2e_pty.py…"
        # A title stored long before (cut mid-word at 60) is cut at a word on the card.
        await store.rename_session(record["task_id"], long[:60])
        started = await store.begin_turn(record["task_id"], long, holder=HOLDER)
        await store.finish_turn(
            record["task_id"],
            started.turn,
            holder=HOLDER,
            status="provider_error",
            cause="run",
            result={**run_result(), "error": "connection refused\nsecond line"},
            answer="",
        )
        await store.append_session_event(
            record["task_id"], "sched.finished", {"state": "failed"}
        )
        card = (await client.get("/api/sessions")).json()["main"]["last_card"]
        assert card["state"] == "failed"
        assert card["title"] == "Fix this today: in tests/cli/test_e2e_pty.py…"
        assert card["line"] == (
            "Could not get an answer from the model server: connection refused"
        )


async def test_tasks_belong_to_the_main_chat_and_come_back_as_cards(
    tmp_path: Path,
) -> None:
    # A user's task works strictly inside the home folder (a temporary one here).
    workspace = Path.home() / "jobs"
    workspace.mkdir()
    async with console_with(tmp_path) as (console, client, _):
        payload = {
            "objective": "Find jobs",
            "title": "Find jobs",
            "level": "p5",
            "workspace": str(workspace),
            "request_key": "k1",
        }
        task, repeated = await console.workers.create(dict(payload))
        again, repeated = await console.workers.create(dict(payload))
        assert repeated and again["task_id"] == task["task_id"]
        record = await console.workers.tasks.get_task(task["task_id"])
        assert record.front_id == console.main_id
        store = console.store
        started = await store.begin_turn(task["task_id"], "Find jobs", holder=HOLDER)
        await store.finish_turn(
            task["task_id"],
            started.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(),
            answer="Two new roles. Saved jobs.md.",
        )
        await store.append_session_event(
            task["task_id"], "sched.finished", {"state": "done"}
        )
        card = (await client.get("/api/sessions")).json()["main"]["last_card"]
        assert card["source_kind"] == "task" and card["state"] == "done"
        assert card["title"] == "Find jobs"
        assert card["line"] == "Two new roles.\nSaved jobs.md."
        assert console.main_id is not None
        # A stop by the person is no card; a turn of a task is none either.
        await store.append_session_event(
            task["task_id"], "sched.stopped", {"state": "cancelled"}
        )
        await client.get("/api/sessions")
        cards = await store.events_of_type(console.main_id, CARD_EVENT)
        assert len(cards) == 1


async def test_cards_page_with_the_main_chats_turns(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        main_id = console.main_id
        assert main_id is not None
        store = console.store
        ids: list[int] = []
        for text in ("one", "two"):
            card = await store.append_session_event(
                main_id, CARD_EVENT, {"state": "done", "title": text}
            )
            assert card.id is not None
            ids.append(card.id)
            turn = await store.begin_turn(main_id, text, holder=HOLDER)
            await store.finish_turn(
                main_id,
                turn.turn,
                holder=HOLDER,
                status="completed",
                cause="run",
                result=run_result(),
                answer=text,
            )
        newest = (await client.get(f"/api/sessions/{main_id}/turns?limit=1")).json()
        assert [t["turn"] for t in newest["turns"]] == [2]
        assert newest["has_earlier"] is True
        assert [c["title"] for c in newest["cards"]] == []
        earlier = (
            await client.get(f"/api/sessions/{main_id}/turns?before=2&limit=1")
        ).json()
        assert [c["title"] for c in earlier["cards"]] == ["one", "two"]
        assert [c["id"] for c in earlier["cards"]] == ids


# ── Helpers ──


def _app(console: Console) -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    console.mount(app)
    return app


def _client(app: Any) -> Any:
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )


async def test_a_chat_that_starts_a_task_owns_it(tmp_path: Path) -> None:
    workspace = Path.home() / "slides"
    workspace.mkdir()
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        side = (await _create(client, tmp_path, "Slides"))["session"]["id"]
        await _wait_turn(client, side)
        body = {"objective": "Make slides", "level": "p5", "workspace": str(workspace)}
        response = await client.post("/api/tasks", json={**body, "front_id": side})
        assert response.status_code == 202, response.text
        task = response.json()["task"]
        assert task["front_id"] == side
        # Without one, the main chat owns it.
        plain = (await client.post("/api/tasks", json=body)).json()["task"]
        assert plain["front_id"] == console.main_id
        for wrong in ("gone", task["task_id"]):
            refused = await client.post("/api/tasks", json={**body, "front_id": wrong})
            assert refused.status_code == 400
            assert refused.json()["error"]["code"] == "invalid_request"
