"""Worker runs on the events and status routes, and the polling follow."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import localmelo.support.gateway.console.sse as sse_mod
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.sse import EventStream
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.support.usage_snapshot import TaskCounts, UsageSnapshot

from .conftest import HOLDER, all_events, core_event, run_result
from .test_events import _line_starting, _serve, _stop, parse_frames
from .test_status import run_events


@pytest.fixture()
def worker_path(tmp_path: Path, console: Console) -> Path:
    path = tmp_path / "cli" / "sessions.db"
    console.worker_history = WorkerReader(str(path))
    return path


async def begin_run(store: SessionStore, *events: dict[str, Any]) -> tuple[str, int]:
    """A worker run with *events* stored, as the CLI runner leaves it mid-flight."""
    turn = await store.begin_turn(
        None,
        "worker task",
        holder=HOLDER,
        new_session=NewSession(kind="worker", model="test", meta={"source": "dsh"}),
    )
    await store.append_events(turn.session_id, turn.turn, list(events), holder=HOLDER)
    return turn.session_id, turn.turn


async def finish_run(store: SessionStore, session_id: str, turn: int) -> None:
    await store.finish_turn(
        session_id,
        turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="done",
    )


def message(text: str) -> dict[str, Any]:
    return core_event("assistant.message", content=text)


async def test_a_worker_run_serves_its_events_after_an_id(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli")
    try:
        sid, turn = await begin_run(store, message("a"), message("b"))
        await finish_run(store, sid, turn)
        stored = await all_events(store, sid)
        ids = [event.id for event in stored]
        resp = client.get(f"/api/sessions/{sid}/events?follow=false")
        events, comments = parse_frames(resp.text)
        assert resp.status_code == 200 and comments == ["retry: 2000"]
        assert [event["id"] for event in events] == ids
        assert [event["type"] for event in events] == [e.event["type"] for e in stored]
        after = client.get(f"/api/sessions/{sid}/events?after={ids[1]}&follow=false")
        assert [e["id"] for e in parse_frames(after.text)[0]] == ids[2:]
        resumed = client.get(
            f"/api/sessions/{sid}/events?follow=false",
            headers={"Last-Event-ID": str(ids[-2])},
        )
        assert [e["id"] for e in parse_frames(resumed.text)[0]] == ids[-1:]
    finally:
        await store.aclose()


async def test_a_following_stream_polls_a_growing_file_and_ends_at_the_finish(
    console: Console, worker_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sse_mod, "SSE_POLL_S", 0.02)
    store = SessionStore(worker_path, writer="cli")
    server, task, port = await _serve(console)
    try:
        sid, turn = await begin_run(store, message("first"))
        url = f"http://127.0.0.1:{port}/api/sessions/{sid}/events"
        async with (
            httpx.AsyncClient(timeout=10) as client,
            client.stream("GET", url) as response,
        ):
            assert response.status_code == 200
            lines = response.aiter_lines()
            seen = [await _line_starting(lines, "id: ")]
            assert console.open_streams == 1
            await store.append_events(sid, turn, [message("second")], holder=HOLDER)
            seen.append(await _line_starting(lines, "id: "))
            await store.append_events(sid, turn, [message("third")], holder=HOLDER)
            await finish_run(store, sid, turn)
            seen.extend(line for line in [x async for x in lines] if line[:3] == "id:")
        stored = [f"id: {event.id}" for event in await all_events(store, sid)]
        assert seen == stored, "every frame arrives once, in order, then the end"
        for _ in range(500):
            if console.open_streams == 0:
                break
            await asyncio.sleep(0.01)
        assert console.open_streams == 0
    finally:
        await _stop(server, task)
        await store.aclose()


async def test_a_stream_resumes_a_worker_run_from_the_last_event_id(
    console: Console, worker_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sse_mod, "SSE_POLL_S", 0.02)
    store = SessionStore(worker_path, writer="cli")
    server, task, port = await _serve(console)
    try:
        sid, turn = await begin_run(store, message("one"), message("two"))
        ids = [event.id for event in await all_events(store, sid)]
        url = f"http://127.0.0.1:{port}/api/sessions/{sid}/events"
        async with (
            httpx.AsyncClient(timeout=10) as client,
            client.stream("GET", url, headers={"Last-Event-ID": str(ids[-2])}) as r,
        ):
            lines = r.aiter_lines()
            assert await _line_starting(lines, "id: ") == f"id: {ids[-1]}"
            await store.append_events(sid, turn, [message("three")], holder=HOLDER)
            assert await _line_starting(lines, "id: ") == f"id: {ids[-1] + 1}"
    finally:
        await _stop(server, task)
        await store.aclose()


async def test_a_run_with_no_live_lease_replays_and_ends(
    console: Console, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli", clock=lambda: 1.0)
    server, task, port = await _serve(console)
    try:
        sid, _ = await begin_run(store, message("left behind"))
        url = f"http://127.0.0.1:{port}/api/sessions/{sid}/events"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await asyncio.wait_for(client.get(url), timeout=10)
        events, _ = parse_frames(resp.text)
        assert [e["type"] for e in events][-1] == "assistant.message"
    finally:
        await _stop(server, task)
        await store.aclose()


async def test_a_silent_live_run_sends_keepalives(
    console: Console, worker_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sse_mod, "SSE_POLL_S", 0.01)
    monkeypatch.setattr(sse_mod, "SSE_KEEPALIVE_S", 0.03)
    store = SessionStore(worker_path, writer="cli")
    try:
        sid, _ = await begin_run(store, message("quiet"))
        last = (await all_events(store, sid))[-1].id
        assert last is not None

        async def live() -> bool:
            return True

        stream = EventStream(
            console.worker_history,
            console.hub,
            console.shutdown,
            sid,
            after_id=last,
            live=live,
        )
        assert await anext(stream) == "retry: 2000\n\n"
        assert await anext(stream) == ": keepalive\n\n"
        console.begin_shutdown()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        await stream.aclose()
    finally:
        await store.aclose()


async def test_an_unknown_id_is_not_found_on_both_routes(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli")
    try:
        await begin_run(store, message("x"))
        for path in ("events", "status"):
            resp = client.get(f"/api/sessions/nope/{path}")
            assert resp.status_code == 404
            assert resp.json()["error"]["code"] == "not_found"
    finally:
        await store.aclose()


async def test_the_status_of_a_worker_run_equals_the_terminal_fold(
    client: TestClient, worker_path: Path
) -> None:
    store = SessionStore(worker_path, writer="cli")
    try:
        now = time.time()
        sid, _ = await begin_run(store, *run_events(now, 400))
        body = client.get(f"/api/sessions/{sid}/status").json()
        snap = UsageSnapshot()
        for stored in await all_events(store, sid):
            snap.observe(stored.event)
        expected = snap.to_dict(local=True, now=now, tasks=TaskCounts(0, 0, 0, None))
        assert body.pop("session_id") == sid
        assert abs(body["cache"].pop("idle_s") - expected["cache"].pop("idle_s")) < 5
        assert body == expected
        assert body["segments"][:3] == ["cache 67% idle 3m", "ctx 1.0k/8.0k", "step 3"]
    finally:
        await store.aclose()


# Every route that writes and names a session, a task or a turn by id. A new
# one must be added here, which makes it prove that it refuses a worker run.
# A turn that is not running in this process is a 409, every other id a 404.
WRITES: dict[tuple[str, str], dict[str, Any] | None] = {
    ("POST", "/api/sessions/{session_id}/turns"): {"text": "no"},
    ("POST", "/api/sessions/{session_id}/retry"): {"turn_id": "t"},
    ("POST", "/api/sessions/{session_id}/submissions/{client_request_id}/cancel"): {
        "text": "no"
    },
    ("PATCH", "/api/sessions/{session_id}"): {"title": "no"},
    ("DELETE", "/api/sessions/{session_id}"): None,
    ("DELETE", "/api/sessions/{session_id}/grants/{tool}"): None,
    ("DELETE", "/api/sessions/{session_id}/grants"): None,
    ("POST", "/api/sessions/{session_id}/level"): {"level": 2},
    ("POST", "/api/presence"): {"session_id": "{session_id}"},
    ("POST", "/api/turns/{turn_id}/cancel"): None,
    ("PATCH", "/api/tasks/{task_id}"): {"level": 2},
    ("POST", "/api/tasks/{task_id}/resume"): {"operation_id": "op"},
    ("POST", "/api/v2/tasks/{task_id}/resume"): {"operation_id": "op"},
}


async def test_every_write_route_refuses_a_worker_run(
    client: TestClient, worker_path: Path
) -> None:
    named = {
        (method, route.path)
        for route in client.app.routes  # type: ignore[attr-defined]
        for method in getattr(route, "methods", None) or ()
        if method not in ("GET", "HEAD")
        and any(key in route.path for key in ("{session_id}", "{task_id}", "{turn_id}"))
    }
    assert named <= set(WRITES), sorted(named - set(WRITES))
    store = SessionStore(worker_path, writer="cli")
    try:
        sid, _ = await begin_run(store, message("x"))
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
        assert stored is not None and stored.title != "no"
        assert len(await all_events(store, sid)) == before
        assert client.get(f"/api/sessions/{sid}/events?follow=false").status_code == 200
    finally:
        await store.aclose()


def supported_event_types() -> set[str]:
    source = Path(sse_mod.__file__).parents[1] / "static" / "worker-history.mjs"
    body = re.search(
        r"SUPPORTED_EVENTS = new Set\(\[(.*?)\]\)", source.read_text(), re.S
    )
    assert body is not None
    return set(re.findall(r"'([^']+)'", body.group(1)))


def test_the_console_fold_knows_every_event_type_of_the_contract() -> None:
    from localmelo.melo.contracts.events import EVENT_TYPES

    assert supported_event_types() >= EVENT_TYPES


def test_a_real_direct_run_is_served_and_summarised(
    tmp_path: Path, client: TestClient, console: Console
) -> None:
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
    path = tmp_path / ".cache/localmelo/cli/sessions.db"
    console.worker_history = WorkerReader(str(path))
    [item] = client.get("/api/sessions?kind=worker").json()["sessions"]
    resp = client.get(f"/api/sessions/{item['id']}/events?follow=false")
    events, _ = parse_frames(resp.text)
    assert {event["type"] for event in events} <= supported_event_types()
    assert "turn.finished" in {event["type"] for event in events}
    body = client.get(f"/api/sessions/{item['id']}/status").json()
    snap = UsageSnapshot()
    for event in events:
        snap.observe({k: v for k, v in event.items() if k not in ("id", "turn")})
    now = time.time()
    expected = snap.to_dict(local=True, now=now, tasks=TaskCounts(0, 0, 0, None))
    assert body.pop("session_id") == item["id"]
    del body["cache"]["idle_s"], expected["cache"]["idle_s"]
    assert body == expected
    assert body["cache"]["requests"] >= 1
