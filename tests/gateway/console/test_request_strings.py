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
import pytest

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


async def _send(
    client: httpx.AsyncClient, method: str, url: str, payload: dict[str, Any]
) -> httpx.Response:
    return await client.request(
        method,
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


# The check runs before the request is routed, so the route does not change the answer: one case
# per route that takes a string a person typed (the deny reason has its own test below).
ROUTES = [
    pytest.param(
        "POST",
        "/api/sessions",
        {"turn": {"text": f"Hi {BAD}"}},
        "turn.text: the text must be valid UTF-8",
        id="create-text",
    ),
    pytest.param(
        "POST",
        "/api/sessions",
        {"workspace": f"/tmp/{BAD}"},
        "workspace: the text must be valid UTF-8",
        id="create-workspace",
    ),
    pytest.param(
        "POST",
        "/api/sessions/{id}/turns",
        {"text": f"Hi {BAD}"},
        "text: the text must be valid UTF-8",
        id="turn-text",
    ),
    pytest.param(
        "PATCH",
        "/api/sessions/{id}",
        {"title": f"x{BAD}"},
        "title: the text must be valid UTF-8",
        id="rename-title",
    ),
    pytest.param(
        "PATCH",
        "/api/sessions/{id}",
        {"title": "Release notes", "workspace": f"/tmp/{BAD}"},
        "workspace: the text must be valid UTF-8",
        id="patch-workspace",
    ),
    pytest.param(
        "POST",
        "/api/sessions/{id}/retry",
        {"turn_id": f"x{BAD}"},
        "turn_id: the text must be valid UTF-8",
        id="retry-turn-id",
    ),
    pytest.param(
        "POST",
        "/api/submissions/8d2f6c1e/cancel",
        {"text": f"Hi {BAD}"},
        "text: the text must be valid UTF-8",
        id="cancel-text",
    ),
]


@pytest.mark.parametrize(("method", "url", "payload", "message"), ROUTES)
async def test_a_request_string_that_is_not_utf8_is_invalid_and_writes_nothing(
    tmp_path: Path, method: str, url: str, payload: dict[str, Any], message: str
) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]
        sessions = await console.store.list_sessions()

        refused = await _send(client, method, url.format(id=session_id), payload)

        assert _surrogate_error(refused) == message
        # Every change of a request is judged before any is written.
        assert await console.store.list_sessions() == sessions
        assert await console.store.list_turns(session_id) == []


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

        refused = await _send(
            client,
            "POST",
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
