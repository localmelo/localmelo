"""``GET /api/sessions/{id}/events``: the SSE frames, the start point and the end."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.support.gateway import _ConsoleServer
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.sse import EventStream

from .conftest import seed_turn


def parse_frames(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The event frames and the comment frames of a one-shot stream."""
    events: list[dict[str, Any]] = []
    comments: list[str] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if block.startswith("retry:"):
            comments.append(block)
            continue
        if block.startswith(":"):
            comments.append(block)
            continue
        lines = block.splitlines()
        ids = [line[len("id: ") :] for line in lines if line.startswith("id: ")]
        data = [line[len("data: ") :] for line in lines if line.startswith("data: ")]
        assert len(ids) == 1 and len(data) == 1
        event = json.loads(data[0])
        assert event["id"] == int(ids[0])
        events.append(event)
    return events, comments


async def test_a_one_shot_stream_carries_every_event(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    resp = client.get(f"/api/sessions/{turn.session_id}/events?follow=false")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-store"
    events, comments = parse_frames(resp.text)
    stored = await console.store.list_events(turn.session_id)
    assert [event["id"] for event in events] == [event.id for event in stored]
    assert comments[0] == "retry: 2000"
    assert console.open_streams == 0


async def test_the_last_event_id_wins_over_after(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    stored = await console.store.list_events(turn.session_id)
    stored_ids = [event.id for event in stored if event.id is not None]
    assert len(stored_ids) == len(stored) >= 3
    cutoff = stored_ids[1]
    for after in ("0", "-1", "not-a-number"):
        resp = client.get(
            f"/api/sessions/{turn.session_id}/events?after={after}&follow=false",
            headers={"Last-Event-ID": str(cutoff)},
        )
        events, _ = parse_frames(resp.text)
        assert [event["id"] for event in events] == [
            event_id for event_id in stored_ids if event_id > cutoff
        ]


async def test_after_starts_the_replay(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    stored = await console.store.list_events(turn.session_id)
    stored_ids = [event.id for event in stored if event.id is not None]
    assert len(stored_ids) == len(stored) >= 2
    after = stored_ids[-2]
    resp = client.get(
        f"/api/sessions/{turn.session_id}/events?after={after}&follow=false"
    )
    events, _ = parse_frames(resp.text)
    assert [event["id"] for event in events] == [stored_ids[-1]]


async def test_one_shot_stops_at_the_id_the_request_read(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = await seed_turn(console.store, workspace)
    high_water = await console.store.latest_event_id(turn.session_id)
    await console.store.append_session_event(turn.session_id, "sched.note", {"x": 1})
    assert await console.store.latest_event_id(turn.session_id) > high_water

    async def frozen(session_id: str) -> int:
        assert session_id == turn.session_id
        return high_water

    monkeypatch.setattr(console.store, "latest_event_id", frozen)
    resp = client.get(f"/api/sessions/{turn.session_id}/events?follow=false")
    events, _ = parse_frames(resp.text)
    assert events
    assert max(event["id"] for event in events) == high_water


async def test_after_the_newest_event_yields_only_retry(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    stored = await console.store.list_events(turn.session_id)
    newest = stored[-1].id
    assert newest is not None
    resp = client.get(
        f"/api/sessions/{turn.session_id}/events?after={newest}&follow=false"
    )
    events, comments = parse_frames(resp.text)
    assert events == []
    assert comments == ["retry: 2000"]
    assert console.open_streams == 0


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", "1_0", "+1", " 1"])
async def test_a_bad_last_event_id_is_a_400(
    client: TestClient, console: Console, workspace: str, value: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    resp = client.get(
        f"/api/sessions/{turn.session_id}/events",
        headers={"Last-Event-ID": value},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("value", ["-1", "1_0", "+1", "abc", "1.5"])
async def test_a_bad_after_is_a_400(
    client: TestClient, console: Console, workspace: str, value: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    resp = client.get(f"/api/sessions/{turn.session_id}/events?after={value}")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


async def test_an_unknown_session_is_404_json(client: TestClient) -> None:
    resp = client.get("/api/sessions/0123456789ab/events")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def _serve(console: Console) -> tuple[_ConsoleServer, asyncio.Task[None], int]:
    app = FastAPI()
    console.mount(app)
    server = _ConsoleServer(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            timeout_graceful_shutdown=3,
        ),
        console=console,
    )
    task = asyncio.create_task(server.serve())
    for _ in range(500):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


async def _stop(server: _ConsoleServer, task: asyncio.Task[None]) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _line_starting(lines: AsyncIterator[str], prefix: str) -> str:
    while True:
        line = await asyncio.wait_for(anext(lines), timeout=10)
        if line.startswith(prefix):
            return line


async def test_a_following_stream_ends_at_shutdown(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    server, task, port = await _serve(console)
    url = f"http://127.0.0.1:{port}/api/sessions/{turn.session_id}/events"
    try:
        async with (
            httpx.AsyncClient(timeout=5) as client,
            client.stream("GET", url) as response,
        ):
            assert response.status_code == 200
            lines = response.aiter_lines()
            assert await anext(lines) == "retry: 2000"
            assert await anext(lines) == ""
            assert (await anext(lines)).startswith("id: ")
            assert console.open_streams == 1
            # An event stored after the subscription arrives live.
            live = await console.store.append_session_event(
                turn.session_id, "sched.note", {"x": 1}
            )
            assert live.id is not None
            assert await _line_starting(lines, f"id: {live.id}") == f"id: {live.id}"
            server.should_exit = True
            rest = await asyncio.wait_for(_collect(lines), timeout=10)
            assert isinstance(rest, list)
        for _ in range(500):
            if console.open_streams == 0:
                break
            await asyncio.sleep(0.01)
        assert console.open_streams == 0
    finally:
        await _stop(server, task)


async def test_a_client_disconnect_unregisters_the_stream(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    server, task, port = await _serve(console)
    url = f"http://127.0.0.1:{port}/api/sessions/{turn.session_id}/events"
    try:
        async with (
            httpx.AsyncClient(timeout=5) as client,
            client.stream("GET", url) as response,
        ):
            assert response.status_code == 200
            lines = response.aiter_lines()
            assert await anext(lines) == "retry: 2000"
            assert await anext(lines) == ""
            assert (await anext(lines)).startswith("id: ")
            assert console.open_streams == 1
        for _ in range(500):
            if console.open_streams == 0:
                break
            await asyncio.sleep(0.01)
        assert console.open_streams == 0
    finally:
        await _stop(server, task)


async def test_a_silent_stream_sends_a_keepalive(
    console: Console, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import localmelo.support.gateway.console.sse as sse_mod

    monkeypatch.setattr(sse_mod, "SSE_KEEPALIVE_S", 0.01)
    turn = await seed_turn(console.store, workspace)
    stored = await console.store.list_events(turn.session_id)
    after = stored[-1].id
    assert after is not None
    stream = EventStream(
        console.store, console.hub, console.shutdown, turn.session_id, after_id=after
    )
    assert await anext(stream) == "retry: 2000\n\n"
    assert await anext(stream) == ": keepalive\n\n"
    console.begin_shutdown()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    await stream.aclose()
    await stream.aclose()


async def test_a_deleted_session_ends_the_stream(
    console: Console, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import localmelo.support.gateway.console.sse as sse_mod

    monkeypatch.setattr(sse_mod, "SSE_KEEPALIVE_S", 0.01)
    turn = await seed_turn(console.store, workspace)
    stored = await console.store.list_events(turn.session_id)
    after = stored[-1].id
    assert after is not None
    stream = EventStream(
        console.store, console.hub, console.shutdown, turn.session_id, after_id=after
    )
    assert await anext(stream) == "retry: 2000\n\n"
    await console.store.delete_session(turn.session_id)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    await stream.aclose()


async def _collect(lines: AsyncIterator[str]) -> list[str]:
    return [line async for line in lines]
