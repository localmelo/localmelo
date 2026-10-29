"""``GET /api/health`` ``status``: what runs now, by kind, with a live or a stale lease."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.melo.checker.redaction import REDACTED
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.running import MAX_ITEMS
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.sessions.store import NewSession, SessionStore

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
    def __init__(self, task_id: str, state: str, level_eff: int, title: str) -> None:
        self.task_id, self.state, self.level_eff, self.title = (
            task_id,
            state,
            level_eff,
            title,
        )


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
    assert found["items"] == [
        {"kind": "chat", "id": turn.session_id, "title": "Fix the build"}
    ]
    assert client.get("/api/health").json()["sessions"]["running"] == 1
    monkeypatch.undo()


async def test_running_tasks_are_counted_with_the_queue(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def listed(states: Any = None) -> list[Record]:
        return [
            Record("t1", "running", 2, "Index the repo"),
            Record("t2", "running", 3, "Summarize"),
            Record("t3", "queued", 1, "Later"),
            Record("t4", "held", 4, "Held"),
            Record("t5", "done", 0, "Over"),
        ]

    monkeypatch.setattr(console.workers.tasks, "list_tasks", listed)
    found = status(client)
    assert found["running"]["tasks"] == 2
    assert found["tasks"] == {"running": 2, "queued": 1, "held": 1, "top_level": 1}
    assert found["items"] == [
        {"kind": "task", "id": "t1", "title": "Index the repo"},
        {"kind": "task", "id": "t2", "title": "Summarize"},
    ]


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
