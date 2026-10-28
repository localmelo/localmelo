"""API v2 writes: create a session, start, retry, stop and approve turns.

Each test drives the routes over ``httpx.AsyncClient`` against the console's
own app, in the test's loop, so a turn's background task, the store and the
teardown share one loop. A real Agent with a scripted model answers; the
builder gives every open a fresh script.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.sessions.store import SessionBusyError
from localmelo.tests.sessions.helpers import (
    ScriptedLLM,
    answer,
    started_agent,
    tool_call,
)

from .conftest import PORT, FakeProbe, make_cfg, quiet_gate, seed_turn

TIMEOUT_S = 10.0


@contextlib.asynccontextmanager
async def console_with(
    tmp_path: Path, *replies: Message, scripts: list[list[Message]] | None = None
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]]:
    """A started console whose builder scripts every Agent with *replies*.

    With *scripts*, the Agent of the Nth build gets the Nth list instead,
    for a test whose sessions need their own replies.
    """
    llms: list[ScriptedLLM] = []

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        # A rebuilt Agent takes the replies its predecessors did not use:
        # every Agent here answers with one reply per turn.
        script = (
            list(scripts[len(llms)])
            if scripts is not None
            else list(replies[len(llms) :])
        )
        llm = ScriptedLLM(script)
        llms.append(llm)
        return await started_agent(Path(workspace), llm, mode="gateway")

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=f"http://127.0.0.1:{PORT}",
    )
    try:
        yield console, client, llms
    finally:
        await client.aclose()
        await console.aclose()


def _workspace(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "ws"
    path.mkdir()
    return {"workspace": str(path)}


async def _post_turn(
    client: httpx.AsyncClient,
    session_id: str,
    text: str,
    **extra: Any,
) -> httpx.Response:
    return await client.post(
        f"/api/sessions/{session_id}/turns", json={"text": text, **extra}
    )


async def _create(
    client: httpx.AsyncClient, tmp_path: Path, text: str, **extra: Any
) -> dict[str, Any]:
    response = await client.post(
        "/api/sessions",
        json={"turn": {"text": text, **extra}, **_workspace(tmp_path)},
    )
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


async def _wait_turn(
    client: httpx.AsyncClient, session_id: str, *, timeout: float = TIMEOUT_S
) -> dict[str, Any]:
    """The newest turn, once it is no longer running."""
    deadline = time.monotonic() + timeout
    while True:
        body = (await client.get(f"/api/sessions/{session_id}/turns")).json()
        if body["turns"] and body["turns"][-1]["status"] != "running":
            return cast(dict[str, Any], body["turns"][-1])
        if time.monotonic() > deadline:
            raise AssertionError("the turn did not finish")
        await asyncio.sleep(0.01)


async def _wait_pending(
    client: httpx.AsyncClient, session_id: str, *, timeout: float = TIMEOUT_S
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        body = (await client.get(f"/api/sessions/{session_id}")).json()
        if body["pending_approvals"]:
            return cast(list[dict[str, Any]], body["pending_approvals"])
        if time.monotonic() > deadline:
            raise AssertionError("no approval was asked")
        await asyncio.sleep(0.01)


# ── Create ──


async def test_create_with_a_first_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")

        session = body["session"]
        assert session["title"] == "Hi there"
        assert session["model"] == "stub-model"
        assert body["turn"]["status"] == "running"
        assert body["resumed_turns"] is None
        turn = await _wait_turn(client, session["id"])
        assert (turn["status"], turn["answer"]) == ("completed", "hello")
        assert turn["id"] == body["turn_id"]


async def test_create_without_a_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("unused")) as (_console, client, _):
        response = await client.post("/api/sessions", json=_workspace(tmp_path))

        assert response.status_code == 201
        body = response.json()
        assert "turn_id" not in body
        turns = (
            await client.get(f"/api/sessions/{body['session']['id']}/turns")
        ).json()
        assert turns["turns"] == []


async def test_a_repeated_create_key_answers_the_same_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        request = {
            "turn": {"text": "Hi there", "client_request_id": "8d2f6c1e"},
            **_workspace(tmp_path),
        }
        first = await client.post("/api/sessions", json=request)
        again = await client.post("/api/sessions", json=request)

        assert (first.status_code, again.status_code) == (201, 200)
        assert again.json()["repeated"] is True
        assert again.json()["session"]["id"] == first.json()["session"]["id"]
        assert again.json()["turn_id"] == first.json()["turn_id"]


async def test_a_bad_workspace_is_invalid_workspace(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("unused")) as (_console, client, _):
        response = await client.post(
            "/api/sessions",
            json={"workspace": "relative/path", "turn": {"text": "Hi"}},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_workspace"


async def test_a_builder_that_fails_is_agent_start_failed(tmp_path: Path) -> None:
    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        raise RuntimeError("no model server")

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=f"http://127.0.0.1:{PORT}",
        ) as client:
            response = await client.post(
                "/api/sessions",
                json={"turn": {"text": "Hi"}, **_workspace(tmp_path)},
            )
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "agent_start_failed"
        assert response.json()["error"]["message"] == "no model server"
        # The row a create makes before its Agent is gone again.
        assert await console.store.list_sessions() == []
    finally:
        await console.aclose()


# ── Turns ──


async def test_a_turn_starts_on_an_existing_session(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("first"), answer("second")) as (
        _console,
        client,
        _,
    ):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        response = await _post_turn(client, session_id, "Hi there")

        assert response.status_code == 202
        assert response.json()["turn"]["status"] == "running"
        assert (await _wait_turn(client, session_id))["answer"] == "first"
        await _post_turn(client, session_id, "and again")
        assert (await _wait_turn(client, session_id))["answer"] == "second"


async def test_a_running_turn_is_session_busy(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]
        # The scripted model blocks once its replies run out.
        assert (await _post_turn(client, session_id, "Hi there")).status_code == 202

        second = await _post_turn(client, session_id, "again")

        assert second.status_code == 409
        assert second.json()["error"]["code"] == "session_busy"
        assert second.json()["error"]["message"] == (
            "A turn is already running in this session."
        )


async def test_retry_runs_the_newest_finished_turn_again(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("first"), answer("second")) as (
        _console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        first = await _wait_turn(client, session_id)

        response = await client.post(
            f"/api/sessions/{session_id}/retry", json={"turn_id": first["id"]}
        )

        assert response.status_code == 202
        retried = response.json()["turn"]
        assert retried["turn"] == 2
        assert retried["user_text"] == "Hi there"
        assert (await _wait_turn(client, session_id))["answer"] == "second"
        old = (await client.get(f"/api/sessions/{session_id}/turns")).json()
        assert [turn["undone"] for turn in old["turns"]] == [True, False]


async def test_retry_of_an_older_turn_is_not_latest_turn(tmp_path: Path) -> None:
    async with console_with(
        tmp_path, answer("first"), answer("second"), answer("third")
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        first = await _wait_turn(client, session_id)
        await _post_turn(client, session_id, "and again")
        await _wait_turn(client, session_id)

        response = await client.post(
            f"/api/sessions/{session_id}/retry", json={"turn_id": first["id"]}
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "not_latest_turn"


async def test_stop_ends_a_running_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id, turn_id = body["session"]["id"], body["turn_id"]

        response = await client.post(f"/api/turns/{turn_id}/cancel")

        assert response.status_code == 202
        assert response.json() == {"turn_id": turn_id, "cancelling": True}
        turn = await _wait_turn(client, session_id)
        assert (turn["status"], turn["cause"]) == ("cancelled", "user_stop")
        late = await client.post(f"/api/turns/{turn_id}/cancel")
        assert late.status_code == 409
        assert late.json()["error"]["code"] == "turn_not_running"


async def test_a_cancel_by_key_stops_the_turn_it_names(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there", client_request_id="8d2f6c1e")
        session_id = body["session"]["id"]

        response = await client.post(
            "/api/submissions/8d2f6c1e/cancel", json={"text": "Hi there"}
        )

        assert response.status_code == 202
        assert response.json()["cancelling"] is True
        turn = await _wait_turn(client, session_id)
        assert (turn["status"], turn["cause"]) == ("cancelled", "user_stop")
        # An accepted key stays accepted: its repeat answers the same turn.
        replay = await _post_turn(
            client, session_id, "Hi there", client_request_id="8d2f6c1e"
        )
        assert replay.status_code == 200
        assert replay.json()["repeated"] is True
        assert replay.json()["turn_id"] == turn["id"]


async def test_a_cancel_before_acceptance_withdraws_the_key(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.post(
            "/api/submissions/8d2f6c1e/cancel", json={"text": "Hi there"}
        )

        assert response.status_code == 200
        assert response.json()["cancelling"] is False
        assert response.json()["submission"]["state"] == "withdrawn"
        later = await client.post(
            "/api/sessions",
            json={
                "turn": {"text": "Hi there", "client_request_id": "8d2f6c1e"},
                **_workspace(tmp_path),
            },
        )
        assert later.status_code == 409
        assert later.json()["error"]["code"] == "submission_withdrawn"


async def test_a_used_key_for_another_text_is_submission_conflict(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("first"), answer("second")) as (
        _console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "Hi there", client_request_id="8d2f6c1e")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        response = await _post_turn(
            client, session_id, "Something else", client_request_id="8d2f6c1e"
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "submission_conflict"


async def test_a_rejected_text_does_not_start_a_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]

        response = await _post_turn(client, session_id, "x" * 100_001)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "text_rejected"
        listed = await client.get(f"/api/sessions/{session_id}/turns")
        # Only the create's own turn: the refused text started nothing.
        assert len(listed.json()["turns"]) == 1


async def test_a_bad_key_is_judged_after_the_text(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]

        response = await _post_turn(
            client, session_id, "x" * 100_001, client_request_id="bad key!"
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "text_rejected"


async def test_attachments_are_refused_for_now(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.post(
            "/api/sessions",
            json={
                "turn": {
                    "text": "Read this",
                    "attachments": [{"name": "notes.txt", "text": "x", "chars": 1}],
                },
                **_workspace(tmp_path),
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


# ── Approvals ──


async def test_an_approval_answers_the_waiting_turn(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]

        [pending] = await _wait_pending(client, session_id)
        assert pending["tool_name"] == "file_write"
        response = await client.post(
            f"/api/approvals/{pending['request_id']}",
            json={"choice": "deny", "reason": "Not that file"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "request_id": pending["request_id"],
            "outcome": "denied",
            "remember": False,
        }
        turn = await _wait_turn(client, session_id)
        assert turn["status"] == "completed"
        assert not target.exists()
        again = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "once"}
        )
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "already_answered"


async def test_stopping_a_turn_while_an_approval_waits(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id, turn_id = body["session"]["id"], body["turn_id"]
        [pending] = await _wait_pending(client, session_id)

        stopped = await client.post(f"/api/turns/{turn_id}/cancel")

        assert stopped.status_code == 202
        turn = await _wait_turn(client, session_id)
        assert (turn["status"], turn["cause"]) == ("cancelled", "user_stop")
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["pending_approvals"] == []
        # A waiting ask that the stop cancels cannot be answered any more:
        # the broker remembers it as resolved.
        late = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "once"}
        )
        assert late.status_code == 409
        # The stored frame says the card was withdrawn, not answered.
        listed = await client.get(
            f"/api/sessions/{session_id}/turns?limit=1&events=true"
        )
        [resolved] = [
            event
            for event in listed.json()["turns"][0]["events"]
            if event["type"] == "approval.resolved"
        ]
        assert resolved["data"]["outcome"] == "withdrawn"
        assert late.json()["error"]["code"] == "already_answered"


async def test_an_unknown_approval_is_not_found(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.post("/api/approvals/0d6c2a9e", json={"choice": "once"})

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "payload",
    [
        {"choice": "sometimes"},
        {"choice": "once", "reason": "not a denial"},
        {"choice": "deny", "reason": "y" * 501},
    ],
    ids=["choice", "reason_without_deny", "reason_too_long"],
)
async def test_a_malformed_approval_answer_is_invalid(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.post("/api/approvals/0d6c2a9e", json=payload)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


# ── Session edits ──


async def test_rename_and_delete_a_session(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("unused")) as (_console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        renamed = await client.patch(
            f"/api/sessions/{session_id}", json={"title": "Release notes"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["session"]["title"] == "Release notes"

        other = tmp_path / "other"
        other.mkdir()
        moved = await client.patch(
            f"/api/sessions/{session_id}", json={"workspace": str(other)}
        )
        assert moved.status_code == 200
        assert moved.json()["session"]["workspace"] == str(other)

        deleted = await client.delete(f"/api/sessions/{session_id}")
        assert deleted.status_code == 204
        assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 404


# ── Shutdown ──


async def test_shutdown_ends_a_running_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            [turn] = (await client.get(f"/api/sessions/{session_id}/turns")).json()[
                "turns"
            ]
            if turn["status"] == "running":
                break
            await asyncio.sleep(0.01)

        await console.sessions.close_all()

        turns = await console.store.list_turns(session_id)
        assert [(t.status, t.cause) for t in turns] == [("cancelled", "shutdown")]


async def test_concurrent_creates_with_one_key_make_one_session(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        request = {
            "turn": {"text": "Hi there", "client_request_id": "8d2f6c1e"},
            **_workspace(tmp_path),
        }

        first, second = await asyncio.gather(
            client.post("/api/sessions", json=request),
            client.post("/api/sessions", json=request),
        )

        assert sorted([first.status_code, second.status_code]) == [200, 201]
        listed = (await client.get("/api/sessions")).json()["sessions"]
        assert [session["id"] for session in listed] == [first.json()["session"]["id"]]


async def test_a_workspace_change_while_a_turn_runs_is_session_busy(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        other = tmp_path / "other"
        other.mkdir()

        response = await client.patch(
            f"/api/sessions/{session_id}", json={"workspace": str(other)}
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "session_busy"


async def test_a_workspace_change_keeps_the_conversation(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("first"), answer("second")) as (
        _console,
        client,
        llms,
    ):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        other = tmp_path / "other"
        other.mkdir()

        moved = await client.patch(
            f"/api/sessions/{session_id}", json={"workspace": str(other)}
        )
        assert moved.status_code == 200
        await _post_turn(client, session_id, "and again")

        assert (await _wait_turn(client, session_id))["answer"] == "second"
        # The second Agent was built for the new folder and got the
        # conversation of the first.
        assert llms[1].requests[0][0].role == "system"
        assert any(m.content == "Hi there" for m in llms[1].requests[0])


async def test_a_turn_past_its_delete_is_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        original = console.store.delete_session
        deleting = asyncio.Event()

        async def slow_delete(session_id: str) -> bool:
            # Hold the session's lock long enough for a turn to queue on
            # it, then delete the row under the queued turn.
            deleting.set()
            await asyncio.sleep(0.05)
            return await original(session_id)

        monkeypatch.setattr(console.store, "delete_session", slow_delete)
        removed = asyncio.create_task(client.delete(f"/api/sessions/{session_id}"))
        await deleting.wait()
        queued = asyncio.create_task(
            client.post(f"/api/sessions/{session_id}/turns", json={"text": "again"})
        )
        assert (await removed).status_code == 204
        response = await queued
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


async def _stream_events(client: httpx.AsyncClient, url: str) -> list[dict[str, Any]]:
    """Every ``data:`` frame of a one-shot event stream, in order."""
    events: list[dict[str, Any]] = []
    async with client.stream("GET", url) as response:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


async def test_a_refused_workspace_change_renames_nothing(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        response = await client.patch(
            f"/api/sessions/{session_id}",
            json={"title": "Renamed", "workspace": "relative/path"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_workspace"
        # Both changes are judged before either is written.
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["session"]["title"] == "Hi there"


async def test_a_deleted_workspace_stops_the_next_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        shutil.rmtree(tmp_path / "ws")

        response = await _post_turn(client, session_id, "again")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_workspace"


async def test_a_reopened_session_keeps_its_conversation_without_a_resume(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("first"), answer("second")) as (
        console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        # What the open cap or a settings change does: close, keep, reopen.
        await console.sessions.keep(session_id)

        response = await _post_turn(client, session_id, "again")

        assert response.status_code == 202
        # The kept conversation is loaded as it was; the store is not read.
        assert response.json()["resumed_turns"] is None
        turn = await _wait_turn(client, session_id)
        assert turn["answer"] == "second"


async def test_a_first_open_after_a_restart_reports_the_resume(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        # A session this gateway has never opened, with a turn in its file.
        workspace = tmp_path / "ws"
        workspace.mkdir(exist_ok=True)
        seeded = await seed_turn(console.store, str(workspace))

        response = await client.post(
            f"/api/sessions/{seeded.session_id}/turns", json={"text": "again"}
        )

        assert response.status_code == 202
        assert response.json()["resumed_turns"] == 1
        assert response.json()["resume_mode"] == "pairs"
        assert response.json()["resumed_checkpoint_turns"] == 0


async def test_a_route_started_turn_reaches_a_reconnecting_stream(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        events = await _stream_events(
            client, f"/api/sessions/{session_id}/events?after=0&follow=false"
        )

        assert events[0]["type"] == "turn.started"
        assert events[-1]["type"] == "turn.finished"
        # A reconnect after an id gets exactly the frames it missed.
        rest = await _stream_events(
            client,
            f"/api/sessions/{session_id}/events"
            f"?after={events[0]['id']}&follow=false",
        )
        assert [frame["id"] for frame in rest] == [frame["id"] for frame in events[1:]]


async def test_retry_while_a_turn_runs_is_session_busy(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id, turn_id = body["session"]["id"], body["turn_id"]
        [pending] = await _wait_pending(client, session_id)

        response = await client.post(
            f"/api/sessions/{session_id}/retry", json={"turn_id": turn_id}
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "session_busy"
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, session_id)


async def test_a_creates_key_is_found_without_a_session_id(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(
            client, tmp_path, "Hi there", client_request_id="create-key-1"
        )
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        # The create makes its row before the Agent; the key still lands in
        # the create scope, where a client without the id looks it up.
        found = (await client.get("/api/submissions/create-key-1")).json()
        assert found["submission"]["state"] == "accepted"
        assert found["submission"]["session_id"] == session_id
        by_session = (
            await client.get(f"/api/sessions/{session_id}/submissions/create-key-1")
        ).json()
        assert by_session["submission"]["turn_id"] == body["turn_id"]


async def test_a_cancel_during_a_slow_create_withdraws_its_key(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        entered.set()
        await release.wait()
        return await started_agent(Path(workspace), ScriptedLLM([answer("hello")]))

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=f"http://127.0.0.1:{PORT}",
        ) as client:
            creating = asyncio.create_task(
                client.post(
                    "/api/sessions",
                    json={
                        "turn": {"text": "hello", "client_request_id": "slow-key"},
                        **_workspace(tmp_path),
                    },
                )
            )
            await entered.wait()
            withdrew = await client.post(
                "/api/submissions/slow-key/cancel", json={"text": "hello"}
            )
            assert withdrew.status_code == 200
            assert withdrew.json()["submission"]["state"] == "withdrawn"

            release.set()
            response = await creating

        # The key was withdrawn while the Agent started: the turn is
        # refused, not written over the withdrawal.
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "submission_withdrawn"
        assert await console.store.list_sessions() == []
    finally:
        await console.aclose()


async def test_a_stop_for_an_unregistered_turn_lands_when_it_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    slow_model = asyncio.Event()

    class SlowLLM(BaseLLMProvider):
        """A model that holds its answer until the test lets go."""

        model = "slow"

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            await slow_model.wait()
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            return None

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        return await started_agent(Path(workspace), SlowLLM())

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=f"http://127.0.0.1:{PORT}",
        ) as client:
            begin = console.store.begin_turn

            async def slow_begin(*args: Any, **kwargs: Any) -> Any:
                turn = await begin(*args, **kwargs)
                if kwargs.get("client_request_id") == "late-key":
                    entered.set()
                    await release.wait()
                return turn

            monkeypatch.setattr(console.store, "begin_turn", slow_begin)
            create = asyncio.create_task(
                client.post(
                    "/api/sessions",
                    json={
                        "turn": {"text": "Hi there", "client_request_id": "late-key"},
                        **_workspace(tmp_path),
                    },
                )
            )
            await asyncio.wait_for(entered.wait(), TIMEOUT_S)

            # The window a create has: its row and the turn are stored, but
            # the registry has not registered the turn yet, so only the
            # deferred stop can reach it.
            response = await client.post(
                "/api/submissions/late-key/cancel", json={"text": "Hi there"}
            )
            release.set()
            created = await create

            assert created.status_code == 201
            assert response.status_code == 202
            assert response.json()["cancelling"] is True
            session_id = created.json()["session"]["id"]
            turn = await _wait_turn(client, session_id)
            assert (turn["status"], turn["cause"]) == ("cancelled", "user_stop")
    finally:
        slow_model.set()
        await console.aclose()


async def test_a_turn_on_a_row_still_being_created_waits_for_it(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    builds: list[str] = []

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        builds.append(session_id)
        if len(builds) == 1:
            entered.set()
            await release.wait()
        return await started_agent(Path(workspace), ScriptedLLM([answer("hello")]))

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=f"http://127.0.0.1:{PORT}",
        ) as client:
            create = asyncio.create_task(
                client.post(
                    "/api/sessions",
                    json={"turn": {"text": "Hi there"}, **_workspace(tmp_path)},
                )
            )
            await asyncio.wait_for(entered.wait(), TIMEOUT_S)
            # The row exists while its Agent is still being built.
            [row] = await console.store.list_sessions()
            second = asyncio.create_task(
                client.post(f"/api/sessions/{row.id}/turns", json={"text": "Later"})
            )
            await asyncio.sleep(0.05)
            # It waits on the session's lock instead of building a second
            # Agent for a row no entry owns yet.
            assert not second.done()
            assert len(builds) == 1
            release.set()
            created = await create
            queued = await second

        assert created.status_code == 201
        # The queued turn then finds the create's turn running (409) or done.
        assert queued.status_code in (202, 409)
        assert len(builds) == 1
    finally:
        await console.aclose()


# ── Workspaces ──


async def test_a_symlinked_workspace_is_stored_as_its_real_path(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    async with console_with(
        tmp_path, answer("hello"), answer("again"), answer("third")
    ) as (console, client, _):
        created = await client.post("/api/sessions", json={"workspace": str(link)})
        session_id = created.json()["session"]["id"]

        # The row keeps the folder the Agent works in, not the link.
        assert created.json()["session"]["workspace"] == os.path.realpath(real)
        moved = await client.patch(
            f"/api/sessions/{session_id}", json={"workspace": str(link)}
        )
        assert moved.json()["session"]["workspace"] == os.path.realpath(real)

        # A closed session reopens at that folder even once the link is gone.
        await console.sessions.keep(session_id)
        link.unlink()
        started = await _post_turn(client, session_id, "Hi there")
        assert started.status_code == 202
        await _wait_turn(client, session_id)


async def test_a_create_cancelled_after_its_row_is_committed_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, _client, _):
        entered = asyncio.Event()
        release = asyncio.Event()
        ids: list[str] = []
        begin = console.store.begin_turn

        async def slow_begin(*args: Any, **kwargs: Any) -> Any:
            turn = await begin(*args, **kwargs)
            ids.append(turn.session_id)
            entered.set()
            await release.wait()
            return turn

        monkeypatch.setattr(console.store, "begin_turn", slow_begin)
        workspace = _workspace(tmp_path)["workspace"]
        task = asyncio.create_task(console.sessions.create(workspace, ("Hi", "Hi")))
        await asyncio.wait_for(entered.wait(), TIMEOUT_S)
        # The row is committed; the client that asked never gets its id.
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        [session_id] = ids
        deadline = time.monotonic() + TIMEOUT_S
        while await console.store.get_session(session_id) is not None:
            assert time.monotonic() < deadline, "the row the create made stayed"
            await asyncio.sleep(0.01)
        assert console.sessions.get(session_id) is None


# ── Approvals: the allow answers ──


async def test_an_allow_once_lets_the_tool_run_and_asks_again(
    tmp_path: Path,
) -> None:
    first = tmp_path / "ws" / "one.md"
    second = tmp_path / "ws" / "two.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(first), "content": "1"}),
        answer("done"),
        tool_call("file_write", {"path": str(second), "content": "2"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write one")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)

        allowed = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "once"}
        )

        assert allowed.json() == {
            "request_id": pending["request_id"],
            "outcome": "allowed",
            "remember": False,
        }
        await _wait_turn(client, session_id)
        assert first.read_text() == "1"
        # "once" grants nothing: the next write of the tool asks again.
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["grants"] == [] and detail["granted_at"] == {}

        await _post_turn(client, session_id, "Write two")
        [again] = await _wait_pending(client, session_id)
        assert again["tool_name"] == "file_write"
        await client.post(
            f"/api/approvals/{again['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, session_id)
        assert not second.exists()


async def test_a_session_grant_covers_its_session_only(tmp_path: Path) -> None:
    one = tmp_path / "ws" / "one.md"
    two = tmp_path / "ws" / "two.md"
    other = tmp_path / "ws" / "other.md"
    scripts = [
        # The session that is granted: a card, then a write with none.
        [
            tool_call("file_write", {"path": str(one), "content": "1"}),
            answer("one"),
            tool_call("file_write", {"path": str(two), "content": "2"}),
            answer("two"),
        ],
        # The other session: its own write must ask.
        [
            tool_call("file_write", {"path": str(other), "content": "3"}),
            answer("three"),
        ],
    ]
    async with console_with(tmp_path, scripts=scripts) as (_console, client, _):
        granted = await _create(client, tmp_path, "Write one")
        granted_id = granted["session"]["id"]
        [pending] = await _wait_pending(client, granted_id)

        answered = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
        )

        assert answered.json()["outcome"] == "allowed"
        assert answered.json()["remember"] is True
        await _wait_turn(client, granted_id)
        assert one.read_text() == "1"
        detail = (await client.get(f"/api/sessions/{granted_id}")).json()
        assert detail["grants"] == ["file_write"]
        assert set(detail["granted_at"]) == {"file_write"}

        # The grant covers this session's next turn, with no card.
        started = await _post_turn(client, granted_id, "Write two")
        turn = await _wait_turn(client, granted_id)
        assert turn["id"] == started.json()["turn_id"]
        assert two.read_text() == "2"
        assert (await client.get(f"/api/sessions/{granted_id}")).json()[
            "pending_approvals"
        ] == []

        # Another session of the same gateway still asks.
        other_body = await client.post(
            "/api/sessions", json={"workspace": str(tmp_path / "ws")}
        )
        other_id = other_body.json()["session"]["id"]
        await _post_turn(client, other_id, "Write three")
        [asked] = await _wait_pending(client, other_id)
        assert asked["tool_name"] == "file_write"
        assert not other.exists()


async def test_a_workspace_change_forgets_the_grants(tmp_path: Path) -> None:
    one = tmp_path / "ws" / "one.md"
    other = tmp_path / "other"
    other.mkdir()
    two = other / "two.md"
    scripts = [
        [tool_call("file_write", {"path": str(one), "content": "1"}), answer("one")],
        [tool_call("file_write", {"path": str(two), "content": "2"}), answer("two")],
    ]
    async with console_with(tmp_path, scripts=scripts) as (_console, client, _):
        granted = await _create(client, tmp_path, "Write one")
        session_id = granted["session"]["id"]
        [pending] = await _wait_pending(client, session_id)
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
        )
        await _wait_turn(client, session_id)
        assert (await client.get(f"/api/sessions/{session_id}")).json()["grants"] == (
            ["file_write"]
        )

        moved = await client.patch(
            f"/api/sessions/{session_id}", json={"workspace": str(other)}
        )

        assert moved.status_code == 200
        # The move forgets the grant: the session's tools all ask again.
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["grants"] == [] and detail["granted_at"] == {}
        await _post_turn(client, session_id, "Write two")
        [again] = await _wait_pending(client, session_id)
        assert again["tool_name"] == "file_write"
        await client.post(
            f"/api/approvals/{again['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, session_id)


# ── Shutdown and delete while a card waits ──


def _write_card(tmp_path: Path) -> tuple[Path, Message, Message]:
    target = tmp_path / "ws" / "todo.md"
    return (
        target,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    )


async def test_a_shutdown_during_an_approval_wait_withdraws_the_card(
    tmp_path: Path,
) -> None:
    _target, card, said = _write_card(tmp_path)
    async with console_with(tmp_path, card, said) as (console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)

        await console.sessions.close_all()

        turns = await console.store.list_turns(session_id)
        assert [(turn.status, turn.cause) for turn in turns] == [
            ("cancelled", "shutdown")
        ]
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["pending_approvals"] == []
        stored = await client.get(
            f"/api/sessions/{session_id}/turns?limit=1&events=true"
        )
        [resolved] = [
            event
            for event in stored.json()["turns"][0]["events"]
            if event["type"] == "approval.resolved"
        ]
        assert resolved["data"]["outcome"] == "withdrawn"
        late = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "once"}
        )
        assert late.status_code == 409
        assert late.json()["error"]["code"] == "already_answered"


async def test_a_delete_during_an_approval_wait_ends_the_turn(
    tmp_path: Path,
) -> None:
    _target, card, said = _write_card(tmp_path)
    async with console_with(tmp_path, card, said) as (console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)

        deleted = await client.delete(f"/api/sessions/{session_id}")

        assert deleted.status_code == 204
        assert await console.store.get_session(session_id) is None
        assert console.sessions.get(session_id) is None
        late = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "once"}
        )
        # The card was withdrawn by the close; it cannot be answered.
        assert late.status_code == 409
        assert late.json()["error"]["code"] == "already_answered"


# ── Locks, retries and deletes ──


async def test_a_rename_takes_the_sessions_lock(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        async with console.sessions.lock(session_id):
            task = asyncio.create_task(
                client.patch(f"/api/sessions/{session_id}", json={"title": "Later"})
            )
            await asyncio.sleep(0.05)
            assert not task.done()

        renamed = await task
        assert renamed.status_code == 200
        assert renamed.json()["session"]["title"] == "Later"


async def test_a_retry_reports_the_resume_it_undid_into(tmp_path: Path) -> None:
    async with console_with(
        tmp_path, answer("first"), answer("second"), answer("again")
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        second = (await _post_turn(client, session_id, "and again")).json()["turn_id"]
        await _wait_turn(client, session_id)

        retried = await client.post(
            f"/api/sessions/{session_id}/retry", json={"turn_id": second}
        )

        assert retried.status_code == 202
        # The undo's resume loaded the one turn that is left.
        assert retried.json()["resumed_turns"] == 1
        assert retried.json()["resume_mode"] == "pairs"
        assert retried.json()["resumed_checkpoint_turns"] == 0


async def test_a_retry_records_a_changed_model(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("first"), answer("again")) as (
        console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        first = await _wait_turn(client, session_id)
        console.cfg.mlc.chat_model = "other-model"

        retried = await client.post(
            f"/api/sessions/{session_id}/retry", json={"turn_id": first["id"]}
        )

        assert retried.status_code == 202
        info = await console.store.get_session(session_id)
        assert info is not None and info.model == "other-model"
        await _wait_turn(client, session_id)


async def test_a_delete_the_store_refuses_is_the_standard_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        async def busy(id_: str) -> bool:
            raise SessionBusyError(f"Session {id_} is running a turn")

        monkeypatch.setattr(console.store, "delete_session", busy)
        refused = await client.delete(f"/api/sessions/{session_id}")

        assert refused.status_code == 409
        assert refused.json()["error"] == {
            "code": "session_busy",
            "message": "A turn is already running in this session.",
            "details": {},
        }


# ── Text at the limit ──


async def test_a_text_at_the_limit_still_starts_a_turn(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("ok")) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        # The limit applies to the text the Agent gets, after the trim.
        started = await _post_turn(client, session_id, " " + "x" * 100_000)

        assert started.status_code == 202
        turns = await console.store.list_turns(session_id)
        assert len(turns[-1].user_text) == 100_000


async def test_a_text_over_the_limit_after_the_trim_is_rejected(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        refused = await _post_turn(client, session_id, " " + "x" * 100_001)

        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "text_rejected"
        assert await console.store.list_turns(session_id) == []


async def test_a_cancel_by_key_matches_a_padded_text(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        created = await _create(
            client, tmp_path, "  Hi there  ", client_request_id="8d2f6c1e"
        )
        session_id = created["session"]["id"]

        # Both texts are compared trimmed, as the checker trims them.
        response = await client.post(
            "/api/submissions/8d2f6c1e/cancel", json={"text": "Hi there"}
        )

        assert response.status_code == 202
        assert response.json()["cancelling"] is True
        turn = await _wait_turn(client, session_id)
        assert (turn["status"], turn["cause"]) == ("cancelled", "user_stop")


async def test_a_malformed_session_id_is_not_found(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        # The same 404 a read of that path gives, not the checker's 400.
        response = await _post_turn(client, "a" * 65, "Hi there")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
