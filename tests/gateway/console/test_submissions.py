"""The submission lookup, by session and by the create scope."""

from __future__ import annotations

from fastapi.testclient import TestClient

from localmelo.support.gateway.console import Console
from localmelo.support.sessions.submissions import WITHDRAWN_ERROR

from .conftest import seed_turn


async def test_a_create_scope_key_is_found_from_either_route(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace, key="req-1")
    by_session = client.get(
        f"/api/sessions/{turn.session_id}/submissions/req-1"
    ).json()["submission"]
    by_create = client.get("/api/submissions/req-1").json()["submission"]
    assert by_session == by_create
    assert by_session["client_request_id"] == "req-1"
    assert by_session["state"] == "accepted"
    assert by_session["session_id"] == turn.session_id
    assert by_session["turn_id"] == turn.id
    assert by_session["status"] == "completed"
    assert by_session["cause"] == "run"
    assert by_session["turn"]["id"] == turn.id
    assert by_session["error"] == ""
    assert by_session["created_at"] > 0


async def test_an_unknown_key_reads_unknown(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    body = client.get(f"/api/sessions/{turn.session_id}/submissions/never-used").json()[
        "submission"
    ]
    assert body == {
        "client_request_id": "never-used",
        "state": "unknown",
        "session_id": None,
        "turn_id": None,
        "status": None,
        "cause": None,
        "turn": None,
        "error": "",
        "created_at": None,
    }


async def test_a_withdrawn_create_scope_key(
    client: TestClient, console: Console
) -> None:
    await console.store.withdraw_submission(None, "gone-1", user_text="hello")
    body = client.get("/api/submissions/gone-1").json()["submission"]
    assert body["state"] == "withdrawn"
    assert body["turn"] is None
    assert body["turn_id"] is None
    assert body["error"] == WITHDRAWN_ERROR
    assert body["session_id"] is None


async def test_a_withdrawn_session_scope_key(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    await console.store.withdraw_submission(
        turn.session_id, "gone-2", user_text="a different message"
    )
    body = client.get(f"/api/sessions/{turn.session_id}/submissions/gone-2").json()[
        "submission"
    ]
    assert body["state"] == "withdrawn"
    assert body["session_id"] == turn.session_id
    assert body["error"] == WITHDRAWN_ERROR


async def test_a_malformed_key_is_400(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    for key in ("has space", "-starts-with-a-dash", "a" * 129):
        resp = client.get(f"/api/sessions/{turn.session_id}/submissions/{key}")
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_request"
    # An empty key has no path segment: it is an unknown route, not a key.
    assert (
        client.get(f"/api/sessions/{turn.session_id}/submissions/").status_code == 404
    )


async def test_an_unknown_session_is_404(client: TestClient) -> None:
    resp = client.get("/api/sessions/0123456789ab/submissions/req-1")
    assert resp.status_code == 404
    # The session is judged before the key: still 404, not a key error.
    resp = client.get("/api/sessions/0123456789ab/submissions/has space")
    assert resp.status_code == 404


async def test_the_create_scope_does_not_need_a_session(client: TestClient) -> None:
    resp = client.get("/api/submissions/req-1")
    assert resp.status_code == 200
    assert resp.json()["submission"]["state"] == "unknown"
