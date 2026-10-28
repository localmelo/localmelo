"""``GET /api/sessions`` and ``GET /api/sessions/{id}``."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from localmelo.support.gateway.console import Console
from localmelo.support.sessions.store import SessionStore

from .conftest import FakeBuilder, open_session, seed_turn

SESSION_KEYS = {
    "id",
    "kind",
    "title",
    "created_at",
    "updated_at",
    "model",
    "workspace",
    "workspace_label",
    "busy",
    "running_here",
    "pending_approvals",
}


async def test_the_list_shape_and_a_session(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace, title="Notes")
    body = client.get("/api/sessions").json()
    assert body["has_more"] is False
    assert len(body["sessions"]) == 1
    session = body["sessions"][0]
    assert set(session) == SESSION_KEYS
    assert session["id"] == turn.session_id
    assert session["kind"] == "chat"
    assert session["title"] == "Notes"
    assert session["model"] == "stub-model"
    assert session["workspace"] == workspace
    assert session["workspace_label"] == workspace
    assert session["busy"] is False
    assert session["running_here"] is False
    assert session["pending_approvals"] == 0


async def test_the_list_is_newest_first_and_pages(
    client: TestClient, console: Console, workspace: str
) -> None:
    first = await seed_turn(console.store, workspace, title="First")
    await asyncio.sleep(0.01)
    second = await seed_turn(console.store, workspace, title="Second")
    body = client.get("/api/sessions?limit=1").json()
    assert [s["id"] for s in body["sessions"]] == [second.session_id]
    assert body["has_more"] is True
    page = client.get("/api/sessions?limit=1&query=First").json()
    assert [s["id"] for s in page["sessions"]] == [first.session_id]


async def test_search_matches_a_title_or_a_turn_text(
    client: TestClient, console: Console, workspace: str
) -> None:
    titled = await seed_turn(console.store, workspace, title="Release notes")
    await seed_turn(console.store, workspace, title="Other", text="a note about milk")
    await seed_turn(console.store, workspace, title="Third", text="nothing")

    by_title = client.get("/api/sessions?query=release").json()
    assert [s["id"] for s in by_title["sessions"]] == [titled.session_id]
    by_text = client.get("/api/sessions?query=milk").json()
    assert len(by_text["sessions"]) == 1


async def test_a_query_over_200_characters_is_400(client: TestClient) -> None:
    resp = client.get("/api/sessions?query=" + "x" * 201)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


async def test_the_kind_filters_the_list(
    client: TestClient, console: Console, workspace: str
) -> None:
    await seed_turn(console.store, workspace, title="Chat")
    task = await console.store.create_session(
        model="stub-model", workspace=workspace, kind="task"
    )
    chats = client.get("/api/sessions").json()["sessions"]
    assert all(s["kind"] == "chat" for s in chats)
    tasks = client.get("/api/sessions?kind=task").json()["sessions"]
    assert [s["id"] for s in tasks] == [task.id]
    assert client.get("/api/sessions?kind=everything").status_code == 400


async def test_a_session_with_its_tokens_and_agent_state(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace, usage={"completion_tokens": 7})
    body = client.get(f"/api/sessions/{turn.session_id}").json()
    assert set(body) == {
        "session",
        "pending_approvals",
        "grants",
        "granted_at",
        "mcp_error",
        "agent",
        "tokens",
    }
    assert body["agent"] == "not_loaded"
    assert body["grants"] == []
    assert body["granted_at"] == {}
    assert body["mcp_error"] == ""
    assert body["tokens"] == 7

    entry = await open_session(console.sessions, turn.session_id)
    entry.policy._granted.add("shell_exec")
    entry.granted_at["shell_exec"] = 123.0
    body = client.get(f"/api/sessions/{turn.session_id}").json()
    assert body["agent"] == "open"
    assert body["grants"] == ["shell_exec"]
    assert body["granted_at"] == {"shell_exec": 123.0}

    await console.sessions.keep(turn.session_id)
    body = client.get(f"/api/sessions/{turn.session_id}").json()
    assert body["agent"] == "paused"


async def test_an_unknown_session_is_404(client: TestClient) -> None:
    resp = client.get("/api/sessions/0123456789ab")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_the_list_is_503_when_the_store_failed(
    client: TestClient, console: Console
) -> None:
    console.store._failed = RuntimeError("The disk is gone")  # type: ignore[attr-defined]
    resp = client.get("/api/sessions")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "store_unavailable"


async def test_the_workspace_label_uses_home(
    client: TestClient, console: Console, home: Path
) -> None:
    project = home / "project"
    project.mkdir()
    turn = await seed_turn(console.store, str(project))
    body = client.get(f"/api/sessions/{turn.session_id}").json()
    assert body["session"]["workspace"] == str(project)
    assert body["session"]["workspace_label"] == "~/project"


async def test_another_stores_session_is_not_listed(
    client: TestClient, console: Console, tmp_path: Path, workspace: str
) -> None:
    # The console lists only the gateway file it opened.
    other = SessionStore(tmp_path / "other.db", writer="gateway")
    await other.open()
    try:
        await other.create_session(model="m", workspace=workspace)
        assert client.get("/api/sessions").json()["sessions"] == []
    finally:
        await other.aclose()


def test_the_builder_is_only_called_by_open(
    console: Console, builder: FakeBuilder
) -> None:
    # Constructing the console and reading health never builds an Agent.
    assert builder.calls == []
