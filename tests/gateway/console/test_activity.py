"""What runs now and what ran today: ``status.items[*].activity``, ``status.focus`` and ``GET /api/activity``."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.melo.checker.redaction import REDACTED
from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.activity import (
    MAX_ROWS,
    MAX_SPAN_S,
    local_midnight,
    result_line,
)
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    SessionStoreError,
)

from .conftest import HOLDER, core_event, open_session, run_result, seed_turn


@pytest.fixture()
def cli_path(tmp_path: Path, console: Console) -> Path:
    path = tmp_path / "cli" / "sessions.db"
    console.worker_history = WorkerReader(str(path))
    return path


async def cli_turn(
    path: Path, kind: str, title: str, *, answer: str = "", events: Any = ()
) -> str:
    """A run (``worker``) or terminal chat (``chat``); finished when it has an *answer*."""
    store = SessionStore(path, writer="cli")
    try:
        turn = await store.begin_turn(
            None,
            "do it",
            holder=HOLDER,
            title=title,
            new_session=NewSession(kind=kind, model="test"),
        )
        if events:
            await store.append_events(
                turn.session_id, turn.turn, list(events), holder=HOLDER
            )
        if answer:
            await store.finish_turn(
                turn.session_id,
                turn.turn,
                holder=HOLDER,
                status="completed",
                cause="run",
                result=run_result(result=answer),
                answer=answer,
            )
        return turn.session_id
    finally:
        await store.aclose()


def set_turn(path: str | Path, session_id: str, **columns: Any) -> None:
    sets = ", ".join(f"{name} = ?" for name in columns)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"UPDATE turns SET {sets} WHERE session_id = ?",
            (*columns.values(), session_id),
        )


class Task:
    """A task row as ``tasks_since`` gives it, with the fields the list reads."""

    def __init__(self, task_id: str, state: str, title: str, **more: Any) -> None:
        now = time.time()
        self.task_id, self.state, self.title = task_id, state, title
        self.level_eff = more.get("level", 2)
        self.wait_reason = more.get("wait_reason", "")
        self.finished_at = more.get("finished_at")
        self.updated_at = more.get("updated_at", now)


def tasks(monkeypatch: pytest.MonkeyPatch, console: Console, *rows: Task) -> None:
    async def since(_: float) -> list[Task]:
        return list(rows)

    monkeypatch.setattr(console.workers.tasks, "tasks_since", since)


def activity(client: TestClient, **params: Any) -> dict[str, Any]:
    response = client.get("/api/activity", params=params)
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


# ── status.items and status.focus ──


def ask(console: Console, owner: str) -> Any:
    request = ApprovalRequest(
        request_id=f"req-{owner}",
        run_id="",
        call_id="call_1",
        tool_name="file_write",
        arguments={},
        preview="a.txt",
        workspace_root=None,
    )
    return console.broker.begin(owner, request)


async def test_each_running_item_says_what_it_does_and_focus_prefers_a_wait(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = await seed_turn(console.store, workspace, title="Slides", status="running")
    await console.store.append_events(
        chat.session_id,
        chat.turn,
        [
            core_event(
                "tool.started",
                ts=time.time() - 60,
                name="file_write",
                arguments={"path": "/w/slides.md", "content": "x"},
            )
        ],
        holder=HOLDER,
    )
    entry = await open_session(console.sessions, chat.session_id)
    monkeypatch.setattr(entry.runner, "_active", object())
    command = core_event(
        "tool.started",
        ts=time.time(),
        name="shell_exec",
        arguments={"command": "uv run pytest -q"},
    )
    run = await cli_turn(cli_path, "worker", "Check the build", events=[command])
    found = client.get("/api/health").json()["status"]
    words = {item["id"]: (item["activity"], item["icon"]) for item in found["items"]}
    assert words == {
        chat.session_id: ("Writing slides.md", "write"),
        run: ("Running the tests", "run"),
    }
    # The most recently changed item is the one a glance shows ...
    assert found["focus"]["id"] == run
    assert found["focus"]["activity"] == "Running the tests"
    # ... until one waits for the person.
    waiting = ask(console, chat.session_id)
    found = client.get("/api/health").json()["status"]
    assert found["focus"]["id"] == chat.session_id
    assert found["focus"]["activity"] == "Waiting for your approval"
    console.broker.withdraw_all(reason="done")
    await waiting


async def test_an_unreadable_event_store_reads_thinking(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = await seed_turn(console.store, workspace, status="running")
    entry = await open_session(console.sessions, chat.session_id)
    monkeypatch.setattr(entry.runner, "_active", object())

    async def broken(*_: Any, **__: Any) -> Any:
        raise SessionStoreError("cannot read")

    monkeypatch.setattr(console.store, "latest_events", broken)
    item = client.get("/api/health").json()["status"]["items"][0]
    assert (item["activity"], item["icon"]) == ("Thinking", "think")


# ── GET /api/activity ──


async def test_the_day_merges_chats_tasks_runs_and_terminal_chats(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    since = now - 3600
    old = await seed_turn(console.store, workspace, title="Yesterday's chat")
    set_turn(console.store.path, old.session_id, finished_at=since - 60)
    chat = await seed_turn(console.store, workspace, title="Plan the trip")
    set_turn(console.store.path, chat.session_id, finished_at=now - 300)
    task_session = await console.store.begin_turn(
        None,
        "find jobs",
        holder=HOLDER,
        title="",
        new_session=NewSession(kind="task", model="m", workspace=workspace),
    )
    await console.store.finish_turn(
        task_session.session_id,
        task_session.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="Two new roles. Both match.",
    )
    set_turn(console.store.path, task_session.session_id, finished_at=now - 200)
    await cli_turn(
        cli_path, "worker", "Check the build", answer="All 412 tests passed."
    )
    terminal = await cli_turn(
        cli_path, "chat", "Read the notes", answer="Version 0.7.1 adds batching. More."
    )
    set_turn(cli_path, terminal, finished_at=now - 100)
    tasks(
        monkeypatch,
        console,
        Task(task_session.session_id, "done", "Find jobs", finished_at=now - 200),
        Task("t-queued", "queued", "Later", level=3),
        Task("t-parked", "parked", "Paused one"),
        Task(
            "t-failed", "failed", "Broke", finished_at=now - 50, wait_reason="no disk"
        ),
    )
    body = activity(client, since=since)
    assert body["since"] == since and body["unavailable"] == [] and not body["more"]
    rows = {row["title"]: row for row in body["rows"]}
    assert "Yesterday's chat" not in rows
    assert [row["kind"] for row in body["rows"]].count("chat") == 1
    assert {row["kind"] for row in body["rows"]} == {"chat", "task", "run", "terminal"}
    # What waits comes first, then the rest, newest first.
    states = [row["state"] for row in body["rows"]]
    assert states[:2] == ["queued", "paused"] or states[:2] == ["paused", "queued"]
    finished = [row["at"] for row in body["rows"][2:]]
    assert finished == sorted(finished, reverse=True)
    assert (
        rows["Later"]["activity"] == "Waiting to start" and rows["Later"]["level"] == 3
    )
    assert rows["Paused one"]["activity"] == "Paused"
    assert (rows["Broke"]["state"], rows["Broke"]["result"]) == ("failed", "no disk")
    assert rows["Find jobs"]["result"] == "Two new roles."
    assert rows["Check the build"]["kind"] == "run"
    assert rows["Check the build"]["result"] == "All 412 tests passed."
    assert rows["Read the notes"]["result"] == "Version 0.7.1 adds batching."
    assert rows["Plan the trip"] | {"at": 0} == {
        "kind": "chat",
        "id": chat.session_id,
        "title": "Plan the trip",
        "state": "done",
        "activity": "",
        "icon": "done",
        "result": "an answer",
        "at": 0,
        "turns": 1,
    }


async def test_a_running_chat_says_what_it_does_and_a_finished_one_its_last_kind(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = await seed_turn(console.store, workspace, title="Live", status="running")
    await console.store.append_events(
        running.session_id,
        running.turn,
        [
            core_event(
                "tool.started",
                name="browser_open",
                arguments={"url": "https://example.com"},
            )
        ],
        holder=HOLDER,
    )
    tasks(monkeypatch, console)
    row = activity(client)["rows"][0]
    assert (row["state"], row["activity"], row["icon"]) == (
        "running",
        "Reading example.com",
        "web",
    )
    await console.store.finish_turn(
        running.session_id,
        running.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="Done.",
    )
    row = activity(client)["rows"][0]
    assert (row["state"], row["icon"], row["activity"]) == ("done", "web", "")


async def test_the_list_is_capped_and_says_so(
    client: TestClient,
    console: Console,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks(monkeypatch, console)
    for n in range(MAX_ROWS + 3):
        await cli_turn(cli_path, "worker", f"run {n}", answer="ok")
    body = activity(client)
    assert len(body["rows"]) == MAX_ROWS and body["more"] is True


@pytest.mark.parametrize("broken", ["day_turns", "tasks_since", "worker_day_turns"])
async def test_an_unreadable_store_counts_as_empty(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken: str,
) -> None:
    await seed_turn(console.store, workspace, title="A chat")
    await cli_turn(cli_path, "worker", "A run", answer="ok")
    tasks(monkeypatch, console, Task("t1", "queued", "A task"))

    async def fail(*_: Any, **__: Any) -> Any:
        raise SessionStoreError("cannot read")

    target, name, gone = {
        "day_turns": (console.store, "day_turns", ("chats", "A chat")),
        "tasks_since": (console.workers.tasks, "tasks_since", ("tasks", "A task")),
        "worker_day_turns": (console.worker_history, "day_turns", ("runs", "A run")),
    }[broken]
    monkeypatch.setattr(target, name, fail)
    body = activity(client)
    assert body["unavailable"] == [gone[0]]
    titles = {row["title"] for row in body["rows"]}
    assert titles == {"A chat", "A task", "A run"} - {gone[1]}


async def test_a_missing_cli_file_is_an_empty_source(
    client: TestClient,
    console: Console,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks(monkeypatch, console)
    assert activity(client) == {
        "since": activity(client)["since"],
        "rows": [],
        "more": False,
        "unavailable": [],
    }
    assert not cli_path.exists()


async def test_a_terminal_chat_is_redacted(
    client: TestClient,
    console: Console,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks(monkeypatch, console)
    key = "sk-" + "a1b2c3d4e5" * 3
    await cli_turn(cli_path, "chat", f"use {key}", answer=f"Set {key} as the key.")
    row = activity(client)["rows"][0]
    assert key not in row["title"] and REDACTED in row["title"]
    assert key not in row["result"]


def test_since_is_today_by_default_a_past_time_and_at_most_a_week(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks(monkeypatch, console)
    now = time.time()
    assert activity(client)["since"] == pytest.approx(local_midnight(now), abs=2)
    assert activity(client, since=0)["since"] == pytest.approx(now - MAX_SPAN_S, abs=5)
    future = client.get("/api/activity", params={"since": now + 3600})
    assert future.status_code == 400
    assert future.json()["error"]["code"] == "invalid_request"
    assert client.get("/api/activity", params={"since": "nan"}).status_code == 400


@pytest.mark.parametrize(
    "note, answer, line",
    [
        ("", "Two new roles. Both match.", "Two new roles."),
        ("Saved jobs.md", "Two new roles. Both match.", "Saved jobs.md"),
        (
            "",
            "<think>why</think>## Results\n- Latency fell 38%.",
            "Results - Latency fell 38%.",
        ),
        ("", "找到了两个职位。都在湾区。", "找到了两个职位。"),
        ("", "", ""),
    ],
)
def test_the_result_is_one_line(note: str, answer: str, line: str) -> None:
    assert result_line(note, answer) == line


def test_a_long_result_is_cut_at_a_word() -> None:
    found = result_line("", "word " * 60)
    assert len(found) <= 140 and found.endswith("word…")
