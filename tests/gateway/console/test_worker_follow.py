"""Worker runs on the events and status routes, and the polling follow."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
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
from .test_events import _collect, _line_starting, _serve, _stop, parse_frames
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
        new_session=NewSession(
            kind="worker", model="test", meta={"source": "localmelo"}
        ),
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
    ("POST", "/api/tasks/{task_id}/continue"): None,
    ("POST", "/api/v2/tasks/{task_id}/resume"): {"operation_id": "op"},
    ("POST", "/api/sessions/{session_id}/blocks/{request_id}/allow"): None,
    ("POST", "/api/tasks/{task_id}/blocks/{request_id}/allow"): None,
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
            "request_id": "r",
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


@pytest.mark.parametrize(
    "scenario,answer",
    [
        (
            "unfinished_server",
            "Hello! The folder is empty. What would you like to make?",
        ),
        ("unfinished_twice", ""),
    ],
    ids=["answers on the second request", "never answers"],
)
def test_a_direct_run_never_returns_unfinished_thinking_and_the_console_folds_it(
    tmp_path: Path, client: TestClient, console: Console, scenario: str, answer: str
) -> None:
    """#278: the reply that is only thinking is served as thinking with no
    content, and the run's result is the answer or a plain error, never it."""
    from localmelo.melo.schema import UNFINISHED_ANSWER_ERROR
    from localmelo.tests.fixtures.stub_llm import (
        SCENARIOS_DIR,
        StubServer,
        load_scenarios,
    )

    server = StubServer(load_scenarios([SCENARIOS_DIR]), default=scenario).start()
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
        env["HOME"] = str(tmp_path)
        done = subprocess.run(
            [sys.executable, "-m", "localmelo", "--base-url", server.base_url]
            + ["--chat-model", "stub-model", "--", "Say hello"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=120,
        )
    finally:
        server.close()
    assert "keep it simple and friendly" not in done.stdout
    if answer:
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == answer
    else:
        assert done.returncode != 0
        assert UNFINISHED_ANSWER_ERROR in done.stderr
    path = tmp_path / ".cache/localmelo/cli/sessions.db"
    console.worker_history = WorkerReader(str(path))
    [item] = client.get("/api/sessions?kind=worker").json()["sessions"]
    resp = client.get(f"/api/sessions/{item['id']}/events?follow=false")
    events, _ = parse_frames(resp.text)
    replies = [e["data"] for e in events if e["type"] == "assistant.message"]
    folded = [r for r in replies if not r["tool_calls"] and not r["content"]]
    assert folded and all("keep it simple" in r["thinking"] for r in folded)
    notices = [e["data"]["kind"] for e in events if e["type"] == "notice"]
    assert notices == ["unfinished_answer"]
    [finished] = [e["data"]["result"] for e in events if e["type"] == "run.finished"]
    assert finished["result"] == answer
    assert finished["error"] == ("" if answer else UNFINISHED_ANSWER_ERROR)


# What a loaded machine slows down in the test below: the run's process start
# (the CLI creates its file, then the schema), the writer task that stores each
# event, and the model call. Every wait is for the event it needs, under one
# generous deadline, never for a fixed time.
DEADLINE_S = 120.0

# A child that stands still between creating its sessions file and creating
# the schema in it: a process start that is slow at that point.
SLOW_SCHEMA = """\
import asyncio

from localmelo.support.sessions import store

_create_schema = store.SessionStore._create_schema


async def _slow(self, conn):
    await asyncio.sleep({delay})
    return await _create_schema(self, conn)


store.SessionStore._create_schema = _slow
"""


async def first_worker_session(
    client: httpx.AsyncClient, base: str, run: subprocess.Popen[bytes]
) -> str:
    """The id of the worker run, once the console lists it.

    The CLI creates its sessions file before the schema in it, so for a while
    the list answers 503 ``store_unavailable``: that is not yet, not a failure.
    """
    deadline = time.monotonic() + DEADLINE_S
    last = "no answer"
    while time.monotonic() < deadline:
        resp = await client.get(f"{base}/api/sessions?kind=worker")
        last = f"{resp.status_code} {resp.text}"
        if resp.status_code == 200 and (rows := resp.json()["sessions"]):
            return str(rows[0]["id"])
        assert run.poll() is None, f"the run ended ({run.returncode}) unlisted: {last}"
        await asyncio.sleep(0.1)
    raise AssertionError(f"no worker run was listed in {DEADLINE_S:g} s: {last}")


async def read_until(lines: AsyncIterator[str], needle: str) -> None:
    """Read stream lines up to the first one holding *needle*."""
    try:
        async with asyncio.timeout(DEADLINE_S):
            async for line in lines:
                if needle in line:
                    return
    except TimeoutError:
        raise AssertionError(f"{needle!r} did not arrive in {DEADLINE_S:g} s") from None
    raise AssertionError(f"the stream ended before {needle!r}")


async def wait_for(check: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + DEADLINE_S
    while not check():
        assert time.monotonic() < deadline, f"{what} did not happen in {DEADLINE_S:g} s"
        await asyncio.sleep(0.05)


@pytest.mark.parametrize("schema_delay_s", [0.0, 1.5], ids=["prompt", "schema-late"])
async def test_a_direct_run_held_mid_flight_is_followed_to_its_end(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    console: Console,
    monkeypatch: pytest.MonkeyPatch,
    schema_delay_s: float,
) -> None:
    """The stream carries a real run's steps while it is blocked, then ends."""
    from localmelo.tests.fixtures.stub_llm import StubServer

    monkeypatch.setattr(sse_mod, "SSE_POLL_S", 0.05)
    (tmp_path / "notes.txt").write_text("three decisions\n")
    scenario = {
        "live": {
            "models": ["stub-model"],
            "replies": [
                {
                    "content": "Reading.",
                    "tool_calls": [
                        {"name": "file_read", "arguments": {"path": "notes.txt"}}
                    ],
                },
                {"hold": True, "then": {"content": "Finished the job."}},
            ],
        }
    }
    stub = StubServer(scenario, default="live").start()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
    env["HOME"] = str(tmp_path)
    if schema_delay_s:
        hook = tmp_path_factory.mktemp("slow-schema")
        (hook / "sitecustomize.py").write_text(SLOW_SCHEMA.format(delay=schema_delay_s))
        env["PYTHONPATH"] = os.pathsep.join(
            [str(hook), *filter(None, [env.get("PYTHONPATH")])]
        )
    run = subprocess.Popen(
        [sys.executable, "-m", "localmelo", "--base-url", stub.base_url]
        + ["--chat-model", "stub-model", "--", "read the notes"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=tmp_path,
    )
    console.worker_history = WorkerReader(
        str(tmp_path / ".cache/localmelo/cli/sessions.db")
    )
    server, task, port = await _serve(console)
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=None)) as client:
            sid = await first_worker_session(client, base, run)
            async with client.stream("GET", f"{base}/api/sessions/{sid}/events") as r:
                lines = r.aiter_lines()
                await read_until(lines, 'data: {"id":')
                await read_until(lines, '"type":"tool.finished"')
                # The next model call is the one the stub holds: wait until it
                # is there, so the run is blocked on the model, not about to be.
                await wait_for(lambda: len(stub.requests) >= 2, "the held model call")
                assert run.poll() is None, "the run is still blocked on the model"
                stub.released.set()
                rest = await asyncio.wait_for(_collect(lines), timeout=DEADLINE_S)
            assert any('"type":"turn.finished"' in line for line in rest)
        assert await asyncio.to_thread(run.wait, DEADLINE_S) == 0
    finally:
        stub.close()
        if run.poll() is None:
            run.kill()
        await _stop(server, task)
