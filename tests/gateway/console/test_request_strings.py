"""Every request string encodes as UTF-8: a lone surrogate is a 400.

SQLite cannot store a lone surrogate, so a request that carries one must
be refused before any write: a refusal after the write wedges the turn, as
a deny reason did before this check (2.1, 2.2).

``json.dumps`` escapes the surrogate, so the body is ASCII and reaches the
gateway exactly as a browser would send it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from localmelo.tests.sessions.helpers import answer, tool_call

from .test_turn_writes import (
    _create,
    _wait_pending,
    _wait_turn,
    _workspace,
    console_with,
)

# A lone surrogate: JSON carries it as an escape that Python accepts.
BAD = "\ud800"


async def _post(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any]
) -> httpx.Response:
    return await client.post(
        url,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


async def _patch(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any]
) -> httpx.Response:
    return await client.patch(
        url,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def _surrogate_error(response: httpx.Response) -> str:
    assert response.status_code == 400, response.text
    body: dict[str, Any] = response.json()
    assert body["error"]["code"] == "invalid_request"
    message = body["error"]["message"]
    assert isinstance(message, str)
    return message


async def test_a_create_with_a_non_utf8_text_is_invalid(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        response = await _post(
            client,
            "/api/sessions",
            {"turn": {"text": f"Hi {BAD}"}, **_workspace(tmp_path)},
        )

        assert _surrogate_error(response) == "turn.text: the text must be valid UTF-8"
        # Nothing was written: no session, so no turn went running.
        assert await console.store.list_sessions() == []


async def test_a_create_with_a_non_utf8_workspace_is_invalid(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        response = await _post(client, "/api/sessions", {"workspace": f"/tmp/{BAD}"})

        assert "workspace: the text must be valid UTF-8" in _surrogate_error(response)
        assert await console.store.list_sessions() == []


async def test_a_turn_with_a_non_utf8_text_is_invalid(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        refused = await _post(
            client, f"/api/sessions/{session_id}/turns", {"text": f"Hi {BAD}"}
        )

        assert _surrogate_error(refused) == "text: the text must be valid UTF-8"
        assert await console.store.list_turns(session_id) == []


async def test_a_rename_with_a_non_utf8_title_is_invalid(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        refused = await _patch(
            client, f"/api/sessions/{session_id}", {"title": f"x{BAD}"}
        )

        assert _surrogate_error(refused) == "title: the text must be valid UTF-8"
        info = await console.store.get_session(session_id)
        assert info is not None and info.title != f"x{BAD}"


async def test_a_patch_with_a_non_utf8_workspace_moves_nothing(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]
        before = await console.store.get_session(session_id)
        assert before is not None

        refused = await _patch(
            client,
            f"/api/sessions/{session_id}",
            {"title": "Release notes", "workspace": f"/tmp/{BAD}"},
        )

        assert "workspace: the text must be valid UTF-8" in _surrogate_error(refused)
        after = await console.store.get_session(session_id)
        assert after is not None
        # Both changes are judged before either is written.
        assert (after.title, after.workspace) == (before.title, before.workspace)


async def test_a_deny_reason_that_is_not_utf8_leaves_the_turn_running(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)

        refused = await _post(
            client,
            f"/api/approvals/{pending['request_id']}",
            {"choice": "deny", "reason": f"no {BAD}"},
        )

        assert _surrogate_error(refused) == "reason: the text must be valid UTF-8"
        # The card is still open, so the turn still finishes when answered.
        answered = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "deny"}
        )
        assert answered.status_code == 200
        turn = await _wait_turn(client, session_id)
        assert turn["status"] == "completed"
        assert not target.exists()


async def test_a_retry_turn_id_that_is_not_utf8_is_invalid(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        refused = await _post(
            client, f"/api/sessions/{session_id}/retry", {"turn_id": f"x{BAD}"}
        )

        assert _surrogate_error(refused) == "turn_id: the text must be valid UTF-8"


async def test_a_cancel_text_that_is_not_utf8_is_invalid(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        refused = await _post(
            client, "/api/submissions/8d2f6c1e/cancel", {"text": f"Hi {BAD}"}
        )

        assert _surrogate_error(refused) == "text: the text must be valid UTF-8"
