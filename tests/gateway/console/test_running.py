"""``GET /api/health`` ``status``: what runs now, by kind, with a live or a stale lease."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.melo.checker.redaction import REDACTED
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.running import MAX_ITEMS, task_counts
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    SessionStoreError,
)

from .conftest import HOLDER, FakeBuilder, FakeProbe, make_cfg, open_session, seed_turn


@pytest.fixture()
def cli_path(tmp_path: Path, console: Console) -> Path:
    path = tmp_path / "cli" / "sessions.db"
    console.worker_history = WorkerReader(str(path))
    return path


async def begin(path: Path, kind: str, title: str, *, finish: bool = False) -> str:
    """A run (``worker``) or terminal chat (``chat``) with a turn that runs."""
    store = SessionStore(path, writer="cli")
    try:
        turn = await store.begin_turn(
            None,
            "do it",
            holder=HOLDER,
            title=title,
            new_session=NewSession(kind=kind, model="test"),
        )
        if finish:
            await store.finish_turn(
                turn.session_id,
                turn.turn,
                holder=HOLDER,
                status="completed",
                cause="run",
                result={"status": "completed", "result": "ok"},
                answer="ok",
            )
        return turn.session_id
    finally:
        await store.aclose()


def lapse(path: Path, *session_ids: str) -> None:
    """Let the leases of *session_ids* run out, as a killed process leaves them.

    Done after every session is made: opening the file ends a stale turn.
    """
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "UPDATE sessions SET lease_until = 1.0 WHERE id = ?",
            [(sid,) for sid in session_ids],
        )


class Record:
    def __init__(
        self,
        task_id: str,
        state: str,
        level_eff: int,
        title: str,
        wait_reason: str = "",
    ) -> None:
        self.task_id, self.state, self.level_eff, self.title = (
            task_id,
            state,
            level_eff,
            title,
        )
        self.wait_reason = wait_reason


def status(client: TestClient) -> dict[str, Any]:
    body = client.get("/api/health").json()
    assert isinstance(body["status"], dict)
    return body["status"]  # type: ignore[no-any-return]


async def test_nothing_runs_on_a_fresh_console(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == {
        "running": {"chats": 0, "tasks": 0, "runs": 0, "terminal": 0},
        "tasks": {"running": 0, "queued": 0, "held": 0, "top_level": None},
        "items": [],
        "waiting": [],
        "recent": None,
        "focus": None,
    }
    # Every field health had before stays.
    assert body["sessions"]["running"] == 0 and body["model"]["name"] == "stub-model"


async def test_a_running_chat_is_counted_and_listed_by_title(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = await seed_turn(console.store, workspace, title="Fix the build")
    entry = await open_session(console.sessions, turn.session_id)
    assert status(client)["running"]["chats"] == 0
    monkeypatch.setattr(entry.runner, "_active", object())
    found = status(client)
    assert found["running"]["chats"] == 1
    item = {"kind": "chat", "id": turn.session_id, "title": "Fix the build"}
    assert found["items"] == [{**item, "activity": "Thinking", "icon": "think"}]
    assert client.get("/api/health").json()["sessions"]["running"] == 1
    # The app's pill reads these words: in the language it asks for, English for a client that asks for none.
    asked = client.get("/api/health", headers={"X-LocalMelo-Language": "zh-Hans"})
    assert asked.json()["status"]["items"][0]["activity"] == "正在思考"
    assert asked.json()["status"]["focus"]["activity"] == "正在思考"
    monkeypatch.undo()


def ask(console: Console, owner: str, request_id: str) -> Any:
    """A tool approval that waits for an answer; *owner* is a chat's session id or a task id."""
    request = ApprovalRequest(
        request_id=request_id,
        run_id="",
        call_id="call_1",
        tool_name="file_write",
        arguments={},
        preview="a.txt",
        workspace_root=None,
    )
    return console.broker.begin(owner, request)


async def test_chats_and_tasks_that_wait_for_an_approval_are_listed(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app's menu bar reads this list: an approval in any chat or task, and where to find it."""
    turn = await seed_turn(console.store, workspace)
    await open_session(console.sessions, turn.session_id)

    async def listed(states: Any = None) -> list[Record]:
        return [
            Record("t1", "held", 2, "Index the repo"),
            Record("t2", "queued", 2, ""),
        ]

    monkeypatch.setattr(console.workers.tasks, "list_tasks", listed)
    assert status(client)["waiting"] == []
    chat = ask(console, turn.session_id, "req-chat")
    task = ask(console, "t1", "req-task")
    assert status(client)["waiting"] == [
        {"kind": "chat", "id": turn.session_id},
        {"kind": "task", "id": "t1"},
    ]
    # An answer takes the chat off the list within the next health read.
    assert console.broker.answer("req-chat", ApprovalDecision(True, "ok")) == "resolved"
    await chat
    assert status(client)["waiting"] == [{"kind": "task", "id": "t1"}]
    console.broker.withdraw_all(reason="done")
    await task
    assert status(client)["waiting"] == []


async def test_running_tasks_are_counted_with_the_queue(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def listed(states: Any = None) -> list[Record]:
        return [
            Record("t1", "running", 2, "Index the repo"),
            Record("t2", "running", 3, "Summarize"),
            Record("t3", "queued", 1, "Later"),
            Record("t4", "held", 4, "Held", "Waiting for a free model slot"),
            Record("t5", "done", 0, "Over"),
        ]

    monkeypatch.setattr(console.workers.tasks, "list_tasks", listed)
    found = status(client)
    assert found["running"]["tasks"] == 2
    assert found["tasks"] == {"running": 2, "queued": 1, "held": 1, "top_level": 1}
    thinking = {"activity": "Thinking", "icon": "think"}
    assert found["items"] == [
        {"kind": "task", "id": "t1", "title": "Index the repo", **thinking},
        {"kind": "task", "id": "t2", "title": "Summarize", **thinking},
    ]


def test_only_a_task_held_on_the_person_is_waiting_for_you() -> None:
    live = [
        Record("t1", "held", 2, "Asks to write", "approval"),
        Record("t2", "held", 2, "Asks a question", "input"),
        Record(
            "t3", "held", 2, "Behind a higher one", "Waiting for a higher-priority task"
        ),
        Record("t4", "held", 2, "No slot", "scheduler"),
        Record("t5", "held", 2, "No reason"),
        # Only a held task can wait for the person: a stale reason on another state does not count.
        Record("t6", "running", 2, "Working", "approval"),
        Record("t7", "queued", 2, "Queued", "approval"),
    ]
    counts = task_counts(live)
    assert counts is not None
    assert (counts.running, counts.queued, counts.held, counts.waiting) == (1, 1, 5, 2)
    slot = task_counts([Record("t4", "held", 2, "No slot", "scheduler")])
    assert slot is not None
    assert (slot.held, slot.waiting) == (1, 0)
    empty = task_counts([])
    assert empty is not None and empty.waiting == 0
    assert task_counts(None) is None


async def test_runs_and_terminal_chats_count_only_under_a_live_lease(
    client: TestClient, cli_path: Path
) -> None:
    live_run = await begin(cli_path, "worker", "live run")
    live_chat = await begin(cli_path, "chat", "live chat")
    stale_run = await begin(cli_path, "worker", "stale run")
    stale_chat = await begin(cli_path, "chat", "stale chat")
    await begin(cli_path, "worker", "done run", finish=True)
    await begin(cli_path, "chat", "done chat", finish=True)
    lapse(cli_path, stale_run, stale_chat)
    found = status(client)
    assert found["running"] == {"chats": 0, "tasks": 0, "runs": 1, "terminal": 1}
    assert {(item["kind"], item["id"], item["title"]) for item in found["items"]} == {
        ("run", live_run, "live run"),
        ("terminal", live_chat, "live chat"),
    }


async def test_a_terminal_chat_title_is_redacted_and_the_file_is_not_written(
    client: TestClient, cli_path: Path
) -> None:
    key = "sk-" + "a1b2c3d4e5" * 3
    await begin(cli_path, "chat", f"use {key} please")
    before = hashlib.sha256(cli_path.read_bytes()).hexdigest()
    item = status(client)["items"][0]
    assert key not in item["title"] and REDACTED in item["title"]
    assert hashlib.sha256(cli_path.read_bytes()).hexdigest() == before


async def test_the_list_is_capped_and_the_counts_are_not(
    client: TestClient, cli_path: Path
) -> None:
    for n in range(MAX_ITEMS + 5):
        await begin(cli_path, "worker", f"run {n}")
    found = status(client)
    assert found["running"]["runs"] == MAX_ITEMS + 5
    assert len(found["items"]) == MAX_ITEMS


async def test_a_missing_cli_file_counts_as_nothing_running(
    client: TestClient, cli_path: Path
) -> None:
    assert not cli_path.exists()
    assert status(client)["running"]["runs"] == 0


async def test_an_unavailable_store_still_answers(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    options = ConsoleOptions(
        store_path=str(tmp_path / "sessions.db"), builder=builder, probe=probe
    )
    first, second = Console(make_cfg(), port=8401, options=options), Console(
        make_cfg(), port=8401, options=options
    )
    await first.start()
    await second.start()
    try:
        app = FastAPI()
        second.mount(app)
        body = TestClient(app, base_url="http://127.0.0.1:8401").get("/api/health")
        assert body.status_code == 200
        assert body.json()["status"]["running"] == {
            "chats": 0,
            "tasks": 0,
            "runs": 0,
            "terminal": 0,
        }
    finally:
        await second.aclose()
        await first.aclose()


def set_turn(console: Console, session_id: str, **columns: Any) -> None:
    """Change columns of a session's turns in the gateway's file, as an older or odder turn would have them."""
    sets = ", ".join(f"{name} = ?" for name in columns)
    with sqlite3.connect(console.store.path) as conn:
        conn.execute(
            f"UPDATE turns SET {sets} WHERE session_id = ?",
            (*columns.values(), session_id),
        )


async def test_the_newest_finished_chat_turn_is_the_recent_outcome(
    client: TestClient, console: Console, workspace: str
) -> None:
    older = await seed_turn(console.store, workspace, title="Older")
    set_turn(console, older.session_id, finished_at=time.time() - 500)
    newest = await seed_turn(console.store, workspace, title="Newest")
    found = status(client)["recent"]
    assert found.keys() == {"kind", "id", "title", "outcome", "ended_s_ago"}
    assert (found["kind"], found["id"], found["outcome"]) == (
        "chat",
        newest.session_id,
        "done",
    )
    assert found["title"] == "Newest"
    assert 0 <= found["ended_s_ago"] < 5
    # The age counts from the end of the turn, to a tenth of a second.
    set_turn(console, newest.session_id, finished_at=time.time() - 42.26)
    assert status(client)["recent"]["ended_s_ago"] == pytest.approx(42.3, abs=0.5)


@pytest.mark.parametrize(
    "status_word, cause, outcome",
    [
        ("completed", "run", "done"),
        ("cancelled", "user_stop", "stopped"),
        ("failed", "run", "failed"),
        ("provider_error", "run", "failed"),
        ("budget_exhausted", "run", "failed"),
        ("stuck", "run", "failed"),
        ("timeout", "run", "failed"),
        ("crashed", "run", "failed"),
    ],
)
async def test_a_turns_status_says_done_failed_or_stopped(
    client: TestClient,
    console: Console,
    workspace: str,
    status_word: str,
    cause: str,
    outcome: str,
) -> None:
    turn = await seed_turn(console.store, workspace, status=status_word)
    set_turn(console, turn.session_id, cause=cause)
    assert status(client)["recent"]["outcome"] == outcome


@pytest.mark.parametrize("cause", ["shutdown", "closed", "crash", "takeover", "park"])
async def test_a_turn_that_something_else_ended_is_no_outcome(
    client: TestClient, console: Console, workspace: str, cause: str
) -> None:
    """A restart, a crash recovery or a park is not a result the person should see."""
    turn = await seed_turn(console.store, workspace)
    set_turn(console, turn.session_id, cause=cause)
    assert status(client)["recent"] is None


async def test_the_last_real_outcome_of_a_session_shows_past_a_later_recovery(
    client: TestClient, console: Console, workspace: str
) -> None:
    done = await seed_turn(console.store, workspace, status="failed")
    stale = await console.store.begin_turn(
        done.session_id, "again", holder=HOLDER, title=None
    )
    await console.store.finish_turn(
        stale.session_id,
        stale.turn,
        holder=HOLDER,
        status="crashed",
        cause="crash",
        result=None,
        answer="",
    )
    assert status(client)["recent"]["outcome"] == "failed"
    # An undone turn counts for nothing either.
    set_turn(console, done.session_id, undone=1)
    assert status(client)["recent"] is None


async def test_runs_and_terminal_chats_have_their_own_kinds_and_the_newest_wins(
    client: TestClient, console: Console, workspace: str, cli_path: Path
) -> None:
    chat = await seed_turn(console.store, workspace)
    set_turn(console, chat.session_id, finished_at=time.time() - 300)
    run = await begin(cli_path, "worker", "a run", finish=True)
    assert status(client)["recent"]["kind"] == "run"
    assert status(client)["recent"]["id"] == run
    with sqlite3.connect(cli_path) as conn:
        conn.execute("UPDATE turns SET finished_at = ?", (time.time() - 900,))
    terminal = await begin(cli_path, "chat", "a terminal chat", finish=True)
    with sqlite3.connect(cli_path) as conn:
        conn.execute(
            "UPDATE turns SET finished_at = ? WHERE session_id = ?",
            (time.time() - 100, terminal),
        )
    found = status(client)["recent"]
    assert (found["kind"], found["id"]) == ("terminal", terminal)
    # A running turn is not an outcome: only the finished one shows.
    await begin(cli_path, "worker", "still going")
    assert status(client)["recent"]["id"] == terminal


@pytest.mark.parametrize(
    "state, outcome",
    [
        ("done", "done"),
        ("failed", "failed"),
        ("cancelled", "stopped"),
        ("expired", "stopped"),
    ],
)
async def test_a_finished_task_is_a_recent_outcome(
    client: TestClient,
    console: Console,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    outcome: str,
) -> None:
    async def latest() -> tuple[str, str, float]:
        return ("t9", state, time.time() - 3)

    monkeypatch.setattr(console.workers.tasks, "latest_finished", latest)
    found = status(client)["recent"]
    assert (found["kind"], found["id"], found["outcome"]) == ("task", "t9", outcome)
    assert 2.5 <= found["ended_s_ago"] <= 8
    assert (
        found["title"] == ""
    )  # no task t9 in the store: empty, and health still answers


async def test_a_finished_task_and_run_say_what_they_were_called(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The activity pill reads "Done: <title>" from ``recent.title``: tasks, runs and long titles too."""
    await begin(cli_path, "worker", "A worker run", finish=True)
    assert status(client)["recent"]["title"] == "A worker run"
    with sqlite3.connect(cli_path) as conn:
        conn.execute("UPDATE turns SET finished_at = ?", (time.time() - 900,))
    chat = await seed_turn(console.store, workspace, title="Fix the build")
    assert status(client)["recent"]["title"] == "Fix the build"
    set_turn(console, chat.session_id, finished_at=1.0)

    async def latest() -> tuple[str, str, float]:
        return ("t1", "done", time.time())

    class Record:
        title = "Slides for the 3 pm meeting"

    async def get_task(task_id: str) -> Record:
        assert task_id == "t1"
        return Record()

    monkeypatch.setattr(console.workers.tasks, "latest_finished", latest)
    monkeypatch.setattr(console.workers.tasks, "get_task", get_task)
    found = status(client)["recent"]
    assert (found["kind"], found["title"]) == ("task", "Slides for the 3 pm meeting")

    async def broken(task_id: str) -> Record:
        raise SessionStoreError("unreadable")

    monkeypatch.setattr(console.workers.tasks, "get_task", broken)
    assert (
        status(client)["recent"]["title"] == ""
    )  # an unreadable title never fails health


async def test_an_unreadable_part_leaves_the_others_and_never_fails_health(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_turn(console.store, workspace)
    run = await begin(cli_path, "worker", "a run", finish=True)
    set_turn(console, (await console.store.list_sessions())[0].id, finished_at=1.0)

    async def broken(*_: Any, **__: Any) -> Any:
        raise SessionStoreError("cannot read")

    monkeypatch.setattr(console.worker_history, "latest_finished_turn", broken)
    monkeypatch.setattr(console.workers.tasks, "latest_finished", broken)
    assert status(client)["recent"]["kind"] == "chat"
    monkeypatch.setattr(console.store, "latest_finished_turn", broken)
    assert status(client)["recent"] is None
    monkeypatch.undo()
    assert status(client)["recent"]["id"] == run
    # A store that is not ok is not asked at all.
    monkeypatch.setattr(console, "store_state", lambda: "unavailable")
    assert client.get("/api/health").json()["status"]["recent"]["id"] == run
