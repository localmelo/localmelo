"""The console reads worker history without opening the CLI store for writing."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.melo.schema import ImagePart
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.sessions.store import NewSession, SessionStore, SessionStoreError

from .conftest import HOLDER, core_event, run_result


@pytest.fixture()
def worker_path(tmp_path: Path, console: Console) -> Path:
    path = tmp_path / "cli" / "sessions.db"
    console.worker_history = WorkerReader(str(path))
    return path


def test_default_console_keeps_task_workers_and_history_separate(
    client: TestClient, console: Console
) -> None:
    assert isinstance(console.worker_history, WorkerReader)
    assert client.get("/api/tasks").status_code == 200
    assert client.get("/api/sessions?kind=worker").status_code == 200


async def test_missing_file_then_live_worker(
    client: TestClient, console: Console, worker_path: Path
) -> None:
    assert client.get("/api/sessions?kind=worker").json()["sessions"] == []
    assert not worker_path.parent.exists()
    store = SessionStore(worker_path, writer="cli")
    try:
        turn = await store.begin_turn(
            None,
            "worker task",
            holder=HOLDER,
            new_session=NewSession(kind="worker", model="test", meta={"source": "dsh"}),
        )
        chat = await store.begin_turn(
            None, "private chat", holder=HOLDER, new_session=NewSession(kind="chat")
        )
        await store.append_events(
            turn.session_id,
            turn.turn,
            [core_event("assistant.message", content="hello")],
            holder=HOLDER,
        )
        items = client.get("/api/sessions?kind=worker").json()["sessions"]
        assert [item["id"] for item in items] == [turn.session_id]
        assert items[0]["meta"] == {"source": "dsh"}
        assert items[0]["running_here"] is False
        assert isinstance(items[0]["started_at"], float)
        assert items[0]["duration_s"] is None
        assert items[0]["busy"] is True
        assert client.get("/api/sessions").json()["sessions"] == []
        # A chat of the CLI file is a terminal chat: read-only, under its own kind.
        shown = client.get(f"/api/sessions/{chat.session_id}").json()["session"]
        assert shown["kind"] == "terminal"
        detail = client.get(f"/api/sessions/{turn.session_id}").json()
        assert detail["agent"] == "not_loaded"
        assert detail["grants"] == []
        body = client.get(f"/api/sessions/{turn.session_id}/turns?events=true").json()
        assert body["turns"][0]["running_here"] is False
        assert any(e["type"] == "assistant.message" for e in body["turns"][0]["events"])
        # A subsequent commit by the CLI writer is visible to the same reader.
        await store.finish_turn(
            turn.session_id,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(),
            answer="done",
        )
        assert (
            client.get("/api/sessions?kind=worker").json()["sessions"][0]["duration_s"]
            == 0.5
        )
        assert (
            client.get(f"/api/sessions/{turn.session_id}/turns").json()["turns"][0][
                "status"
            ]
            == "completed"
        )
        with pytest.raises(SessionStoreError):
            await console.worker_history.rename_session(turn.session_id, "no")
        stored_session = await store.get_session(turn.session_id)
        assert stored_session is not None and stored_session.title != "no"
        # The events of a worker run are read, never written.
        events = client.get(f"/api/sessions/{turn.session_id}/events?follow=false")
        assert events.status_code == 200 and "assistant.message" in events.text
        assert (
            client.patch(
                f"/api/sessions/{turn.session_id}", json={"title": "no"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                f"/api/sessions/{turn.session_id}/turns", json={"text": "no"}
            ).status_code
            == 404
        )
        assert client.delete(f"/api/sessions/{turn.session_id}").status_code == 404
    finally:
        await store.aclose()


async def test_worker_blob_is_read_from_the_cli_store(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli")
    turn = await store.begin_turn(
        None, "task", holder=HOLDER, new_session=NewSession(kind="worker")
    )
    image = ImagePart(
        mime_type="image/png", data=base64.b64encode(b"fixture-image").decode()
    )
    await store.append_events(
        turn.session_id, turn.turn, [], holder=HOLDER, images=[image]
    )
    response = client.get(f"/api/sessions/{turn.session_id}/blobs/{image.sha256}")
    assert response.status_code == 200 and response.content == b"fixture-image"
    await store.aclose()


async def test_worker_rejects_foreign_file(
    client: TestClient, worker_path: Path
) -> None:
    worker_path.parent.mkdir()
    with sqlite3.connect(worker_path) as conn:
        conn.execute("CREATE TABLE other (x INTEGER)")
    assert client.get("/api/sessions?kind=worker").status_code == 503
    assert client.get("/api/sessions").status_code == 200


async def test_read_does_not_recover_stale_turn(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli", clock=lambda: 1.0)
    turn = await store.begin_turn(
        None, "stale", holder=HOLDER, new_session=NewSession(kind="worker")
    )
    await store.aclose()
    before = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    detail = client.get(f"/api/sessions/{turn.session_id}").json()
    assert hashlib.sha256(worker_path.read_bytes()).hexdigest() == before
    assert detail["session"]["busy"] is False
    with sqlite3.connect(f"{worker_path.as_uri()}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT status FROM turns").fetchone() == ("running",)


async def test_worker_filters_and_cursor_over_200(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli")
    await store.open()
    await store.aclose()
    with sqlite3.connect(worker_path) as conn:
        for n in range(235):
            sid = f"worker-{n:04d}"
            meta: dict[str, Any] = {
                "source": "dsh" if n % 2 else "localmelo",
                "parent_run": "p1" if n % 3 else "p2",
            }
            if n == 234:
                meta.update(
                    {
                        "status": "gate_stopped",
                        "steps": 0,
                        "usage": {"total_tokens": 0, "completion_tokens": 0},
                        "exit_code": 2,
                        "git_head": "abc",
                        "git_branch": "main",
                    }
                )
            conn.execute(
                "INSERT INTO sessions (id, kind, title, created_at, updated_at, meta) VALUES (?, 'worker', ?, ?, ?, ?)",
                (sid, "worker", 1, n, json.dumps(meta)),
            )
            conn.execute(
                "INSERT INTO turns (session_id, turn, id, status, user_text, holder, started_at) VALUES (?, 1, ?, ?, 'text', 'test', ?)",
                (sid, f"turn-{n}", "completed" if n % 5 else "failed", 10 + n // 100),
            )
    seen: list[str] = []
    cursor = None
    while True:
        params = {"kind": "worker", "limit": "63"}
        if cursor is not None:
            params["cursor"] = cursor
        page = client.get("/api/sessions", params=params).json()
        seen.extend(item["id"] for item in page["sessions"])
        cursor = page["next_cursor"]
        if cursor is None:
            assert page["has_more"] is False
            break
        assert page["has_more"] is True
    expected = [
        f"worker-{n:04d}"
        for n in sorted(
            range(235), key=lambda n: (10 + n // 100, f"worker-{n:04d}"), reverse=True
        )
    ]
    assert seen == expected
    filtered = client.get(
        "/api/sessions",
        params={
            "kind": "worker",
            "source": "dsh",
            "parent_run": "p1",
            "status": "completed",
            "limit": 200,
        },
    ).json()
    expected_filtered = [
        f"worker-{n:04d}"
        for n in sorted(
            range(235), key=lambda n: (10 + n // 100, f"worker-{n:04d}"), reverse=True
        )
        if n % 2 and n % 3 and n % 5
    ]
    assert [row["id"] for row in filtered["sessions"]] == expected_filtered
    assert filtered["next_cursor"] is None
    paged: list[str] = []
    next_cursor = None
    while True:
        params = {
            "kind": "worker",
            "source": "dsh",
            "parent_run": "p1",
            "status": "completed",
            "limit": "7",
        }
        if next_cursor is not None:
            params["cursor"] = next_cursor
        page = client.get("/api/sessions", params=params).json()
        paged.extend(row["id"] for row in page["sessions"])
        next_cursor = page["next_cursor"]
        if next_cursor is None:
            break
    assert paged == expected_filtered
    row = filtered["sessions"][0]
    assert row["status"] == "completed" and row["source"] == "dsh"
    assert all(
        row[field] is None
        for field in ("git_branch", "git_head", "exit_code", "steps", "tokens")
    )
    assert (
        client.get(f"/api/sessions/{row['id']}").json()["session"]["status"]
        == "completed"
    )
    exact = client.get("/api/sessions?kind=worker&status=gate_stopped").json()[
        "sessions"
    ]
    assert len(exact) == 1 and exact[0]["id"] == "worker-0234"
    assert exact[0]["steps"] == 0 and exact[0]["tokens"] == 0
    assert exact[0]["meta"]["usage"] == {"total_tokens": 0, "completion_tokens": 0}
    detail = client.get("/api/sessions/worker-0234").json()
    assert detail["session"]["tokens"] == 0
    assert detail["session"]["meta"]["usage"] == exact[0]["meta"]["usage"]
    assert exact[0]["git_branch"] == "main" and exact[0]["exit_code"] == 2
    assert (
        client.get("/api/sessions?kind=worker&cursor=not-a-cursor").status_code == 400
    )
    assert client.get("/api/sessions?source=dsh").status_code == 400


async def test_worker_pagination_and_event_paging(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli")
    try:
        first = await store.begin_turn(
            None, "first", holder=HOLDER, new_session=NewSession(kind="worker")
        )
        await store.finish_turn(
            first.session_id,
            first.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(),
            answer="ok",
        )
        second = await store.begin_turn(first.session_id, "second", holder=HOLDER)
        events = [core_event("assistant.message", content=str(n)) for n in range(1001)]
        for start in range(0, len(events), 200):
            await store.append_events(
                first.session_id,
                second.turn,
                events[start : start + 200],
                holder=HOLDER,
            )
        page = client.get(
            f"/api/sessions/{first.session_id}/turns?limit=1&events=true"
        ).json()
        assert page["has_earlier"] is True
        assert len(page["turns"][0]["events"]) >= 1002
        older = client.get(f"/api/sessions/{first.session_id}/turns?before=2").json()
        assert [turn["turn"] for turn in older["turns"]] == [1]
        assert "replay_from" not in older
        assert (
            client.get("/api/sessions?kind=worker&query=second").json()["sessions"][0][
                "id"
            ]
            == first.session_id
        )
    finally:
        await store.aclose()


def test_a_plain_direct_run_appears_in_the_worker_list(
    tmp_path: Path, client: TestClient, console: Console
) -> None:
    import os
    import subprocess
    import sys

    from localmelo.tests.fixtures.stub_llm import StubServer

    server = StubServer().start()
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
        env["HOME"] = str(tmp_path)
        done = subprocess.run(
            [sys.executable, "-m", "localmelo", "--base-url", server.base_url]
            + ["--chat-model", "stub-model", "--", "say hello"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=120,
        )
    finally:
        server.close()
    assert done.returncode == 0, done.stderr
    console.worker_history = WorkerReader(
        str(tmp_path / ".cache/localmelo/cli/sessions.db")
    )
    [item] = client.get("/api/sessions?kind=worker").json()["sessions"]
    assert (item["title"], item["status"], item["source"]) == (
        "say hello",
        "completed",
        "localmelo",
    )
