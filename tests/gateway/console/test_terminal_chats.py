"""Terminal chats in the console: read-only, listed apart, redacted on every read."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.melo.checker.redaction import REDACTED
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.support.usage_snapshot import TaskCounts, UsageSnapshot

from .conftest import HOLDER, all_events, core_event, run_result, seed_turn
from .test_status import run_events
from .test_worker_follow import WRITES, begin_run, finish_run

API_KEY = "sk-" + "a1b2c3d4e5" * 3
BEARER_TOKEN = "q1w2e3r4t5y6u7i8o9p0"
PEM_BODY = "MIIEowIBAAKCAQEAx3secretkeybody"
PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n" f"{PEM_BODY}\n" "-----END RSA PRIVATE KEY-----"
)
DOT_ENV_LINE = "DATABASE_PASSWORD=hunter2-plain"
SECRETS = (API_KEY, BEARER_TOKEN, PEM_BODY, "hunter2-plain")
SHELL_OUTPUT = f"$ cat notes\nAuthorization: Bearer {BEARER_TOKEN}\n"


@pytest.fixture()
def cli_path(tmp_path: Path, console: Console) -> Path:
    path = tmp_path / "cli" / "sessions.db"
    console.worker_history = WorkerReader(str(path))
    return path


def tool_pair(call_id: str, name: str, arguments: dict[str, Any], output: str) -> Any:
    return (
        core_event(
            "tool.started", call_id=call_id, name=name, arguments=arguments, preview=""
        ),
        core_event(
            "tool.finished",
            call_id=call_id,
            name=name,
            output=output,
            output_chars=len(output),
        ),
    )


async def begin_chat(
    store: SessionStore,
    workspace: str,
    *events: dict[str, Any],
    title: str = "terminal chat",
    text: str = "hello",
) -> tuple[str, int]:
    """A terminal chat with *events* stored unredacted, a turn still running."""
    turn = await store.begin_turn(
        None,
        text,
        holder=HOLDER,
        title=title,
        new_session=NewSession(kind="chat", model="test", workspace=workspace),
    )
    await store.append_events(turn.session_id, turn.turn, list(events), holder=HOLDER)
    return turn.session_id, turn.turn


async def leaking_chat(store: SessionStore, workspace: str) -> str:
    """A finished chat that holds every secret kind in every place it can."""
    started, finished = tool_pair(
        "call_env", "file_read", {"path": ".env"}, DOT_ENV_LINE
    )
    shell = tool_pair(
        "call_sh", "shell_exec", {"command": "echo hi"}, f"key {API_KEY}\n{PRIVATE_KEY}"
    )
    sid, turn = await begin_chat(
        store,
        workspace,
        core_event("run.started"),
        started,
        finished,
        *shell,
        core_event("assistant.message", content=f"the token is Bearer {BEARER_TOKEN}"),
        title=f"debug {API_KEY}",
        text=f"use {API_KEY}\n{SHELL_OUTPUT}",
    )
    await store.finish_turn(
        sid,
        turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(result=f"done {API_KEY}", config={"mode": "x"}),
        answer=f"done {API_KEY}",
    )
    return sid


def routes(sid: str) -> list[str]:
    return [
        "/api/sessions?kind=terminal",
        f"/api/sessions/{sid}",
        f"/api/sessions/{sid}/turns?events=true",
        f"/api/sessions/{sid}/turns",
        f"/api/sessions/{sid}/events?follow=false",
        f"/api/sessions/{sid}/status",
    ]


async def test_a_terminal_chat_is_redacted_on_every_read_route(
    client: TestClient, cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        sid = await leaking_chat(store, workspace)
        # The file really holds the secrets: the redaction is on read.
        stored = json.dumps([e.event for e in await all_events(store, sid)])
        assert all(secret in stored for secret in SECRETS)
        for path in routes(sid):
            resp = client.get(path)
            assert resp.status_code == 200, path
            for secret in SECRETS:
                assert secret not in resp.text, (path, secret)
        detail = client.get(f"/api/sessions/{sid}").json()["session"]
        assert detail["title"] == f"debug {REDACTED}"
        assert detail["kind"] == "terminal"
        [turn] = client.get(f"/api/sessions/{sid}/turns").json()["turns"]
        assert REDACTED in turn["user_text"] and turn["answer"] == f"done {REDACTED}"
        assert "config" not in turn["result"]
        events = client.get(f"/api/sessions/{sid}/events?follow=false").text
        assert f"Bearer {REDACTED}" in events
        assert "[Sensitive file content omitted: .env]" in events
        assert f"key {REDACTED}" in events
        assert f"BEGIN RSA PRIVATE KEY-----\\n{REDACTED}\\n-----END" in events
    finally:
        await store.aclose()


async def test_a_secret_file_read_in_an_earlier_page_is_still_omitted(
    cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    reader = WorkerReader(str(cli_path))
    try:
        started, finished = tool_pair(
            "call_env", "file_read", {"path": ".env"}, DOT_ENV_LINE
        )
        sid, turn = await begin_chat(store, workspace, started)
        last = (await reader.list_events(sid))[-1]
        await store.append_events(sid, turn, [finished], holder=HOLDER)
        [second] = await reader.list_events(sid, after_id=last.id or 0)
        assert second.event["data"]["output"].startswith("[Sensitive file content")
        assert DOT_ENV_LINE not in json.dumps(second.event)
    finally:
        await reader.aclose()
        await store.aclose()


async def test_a_repeated_call_id_hides_only_the_secret_read(
    cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    reader = WorkerReader(str(cli_path))
    try:
        env = tool_pair("call_1", "file_read", {"path": ".env"}, DOT_ENV_LINE)
        plain = tool_pair("call_1", "file_read", {"path": "a.txt"}, "plain text")
        sid, _ = await begin_chat(store, workspace, *env, *plain)
        outputs = [
            e.event["data"]["output"]
            for e in await reader.list_events(sid)
            if e.event["type"] == "tool.finished"
        ]
        assert outputs == ["[Sensitive file content omitted: .env]", "plain text"]
    finally:
        await reader.aclose()
        await store.aclose()


async def test_a_worker_run_is_served_as_stored_and_never_redacted_twice(
    client: TestClient, cli_path: Path
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        text = f"saw {REDACTED} and Bearer {REDACTED}"
        sid, turn = await begin_run(
            store, core_event("assistant.message", content=text)
        )
        await finish_run(store, sid, turn)
        before = [e.event for e in await all_events(store, sid)]
        body = client.get(f"/api/sessions/{sid}/events?follow=false").text
        served = [
            json.loads(line[6:]) for line in body.splitlines() if line[:6] == "data: "
        ]
        assert [
            {k: v for k, v in e.items() if k not in ("id", "session_id", "turn")}
            for e in served
        ] == before
        assert REDACTED + REDACTED not in body and f"{REDACTED}{REDACTED}" not in body
        # A chat that already holds the marker keeps one of it.
        chat, _ = await begin_chat(
            store, "/tmp", core_event("assistant.message", content=text)
        )
        again = client.get(f"/api/sessions/{chat}/events?follow=false").text
        assert text in again
    finally:
        await store.aclose()


async def test_the_lists_keep_chats_apart(
    client: TestClient, console: Console, cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        await seed_turn(console.store, workspace, title="console chat")
        chat, _ = await begin_chat(store, workspace, title="from the terminal")
        run, _ = await begin_run(store)
        rows = {
            kind: client.get(f"/api/sessions?kind={kind}").json()["sessions"]
            for kind in ("chat", "worker", "terminal")
        }
        assert [r["title"] for r in rows["chat"]] == ["console chat"]
        assert [r["id"] for r in rows["worker"]] == [run]
        [row] = rows["terminal"]
        assert (row["id"], row["kind"], row["status"]) == (chat, "terminal", "running")
        assert row["title"] == "from the terminal" and row["busy"] is True
        assert (
            client.get("/api/sessions?kind=terminal&query=terminal").json()["sessions"][
                0
            ]["id"]
            == chat
        )
        assert (
            client.get("/api/sessions?kind=terminal&status=completed").json()[
                "sessions"
            ]
            == []
        )
        assert client.get("/api/sessions?kind=terminal&source=x").status_code == 200
        assert client.get("/api/sessions?kind=chat&source=x").status_code == 400
    finally:
        await store.aclose()


async def test_a_chat_row_stands_for_its_latest_turn(
    client: TestClient, cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        sid, turn = await begin_chat(store, workspace)
        await store.finish_turn(
            sid,
            turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(),
            answer="a",
        )
        second = await store.begin_turn(sid, "again", holder=HOLDER)
        assert second.turn == turn + 1
        [row] = client.get("/api/sessions?kind=terminal").json()["sessions"]
        assert row["status"] == "running" and row["started_at"] == second.started_at
        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["session"]["status"] == "running"
        assert detail["agent"] == "not_loaded" and detail["pending_approvals"] == []
    finally:
        await store.aclose()


async def test_the_status_of_a_terminal_chat_equals_the_terminal_fold(
    client: TestClient, cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        now = time.time()
        sid, _ = await begin_chat(store, workspace, *run_events(now, 400))
        body = client.get(f"/api/sessions/{sid}/status").json()
        snap = UsageSnapshot()
        for stored in await all_events(store, sid):
            snap.observe(stored.event)
        expected = snap.to_dict(local=True, now=now, tasks=TaskCounts(0, 0, 0, None))
        assert body.pop("session_id") == sid
        assert abs(body["cache"].pop("idle_s") - expected["cache"].pop("idle_s")) < 5
        assert body == expected
    finally:
        await store.aclose()


async def test_every_write_route_refuses_a_terminal_chat(
    client: TestClient, cli_path: Path, workspace: str
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        sid, _ = await begin_chat(store, workspace, core_event("run.started"))
        [info] = await store.list_turns(sid)
        before = len(await all_events(store, sid))
        ids = {
            "session_id": sid,
            "task_id": sid,
            "turn_id": info.id,
            "tool": "file_write",
            "client_request_id": "k",
        }
        for (method, path), body in WRITES.items():
            payload = (
                None
                if body is None
                else {
                    k: v.format(**ids) if isinstance(v, str) else v
                    for k, v in body.items()
                }
            )
            resp = client.request(method, path.format(**ids), json=payload)
            want = 409 if "cancel" in path and "turns" in path else 404
            assert resp.status_code == want, (method, path, resp.status_code)
        stored = await store.get_session(sid)
        assert stored is not None and stored.title == "terminal chat"
        assert len(await all_events(store, sid)) == before
        assert [t.undone for t in await store.list_turns(sid)] == [False]
    finally:
        await store.aclose()


async def test_the_reader_does_not_serve_other_kinds(
    client: TestClient, cli_path: Path
) -> None:
    store = SessionStore(cli_path, writer="cli")
    try:
        turn = await store.begin_turn(
            None, "q", holder=HOLDER, new_session=NewSession(kind="task", model="m")
        )
        assert client.get(f"/api/sessions/{turn.session_id}").status_code == 404
        assert client.get("/api/sessions?kind=terminal").json()["sessions"] == []
    finally:
        await store.aclose()
