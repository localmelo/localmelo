"""``GET /api/sessions/{id}/turns``: pages, events and the replay point."""

from __future__ import annotations

from fastapi.testclient import TestClient

from localmelo.support.gateway.console import Console
from localmelo.support.sessions.store import SessionStore, TurnInfo

from .conftest import (
    HOLDER,
    all_events,
    core_event,
    run_result,
    seed_many_events,
    seed_turn,
)

TURN_KEYS = {
    "id",
    "turn",
    "run_id",
    "user_text",
    "status",
    "cause",
    "answer",
    "result",
    "undone",
    "started_at",
    "finished_at",
    "client_request_id",
    "running_here",
}


async def add_turn(
    store: SessionStore, session_id: str, text: str, *, status: str = "completed"
) -> TurnInfo:
    turn = await store.begin_turn(session_id, text, holder=HOLDER)
    await store.append_events(
        session_id,
        turn.turn,
        [
            core_event("run.started", run_id=f"run-{turn.turn}"),
            core_event("assistant.message", content=text),
        ],
        holder=HOLDER,
    )
    if status == "running":
        return turn
    return await store.finish_turn(
        session_id,
        turn.turn,
        holder=HOLDER,
        status=status,
        cause="run",
        result=run_result(),
        answer=text,
    )


async def test_the_newest_page_and_its_replay_point(
    client: TestClient, console: Console, workspace: str
) -> None:
    first = await seed_turn(console.store, workspace, title="One")
    session_id = first.session_id
    await add_turn(console.store, session_id, "second")
    await add_turn(console.store, session_id, "third")
    body = client.get(f"/api/sessions/{session_id}/turns?limit=2").json()
    assert [t["turn"] for t in body["turns"]] == [2, 3]
    assert body["has_earlier"] is True
    assert set(body["turns"][0]) == TURN_KEYS
    started = [
        event
        for event in (await console.store.list_events(session_id, turns=(2, 2)))
        if event.event["type"] == "turn.started"
    ]
    started_id = started[0].id
    assert started_id is not None
    assert body["replay_from"] == started_id - 1


async def test_before_pages_older_turns_and_has_no_replay_point(
    client: TestClient, console: Console, workspace: str
) -> None:
    first = await seed_turn(console.store, workspace, title="One")
    session_id = first.session_id
    await add_turn(console.store, session_id, "second")
    await add_turn(console.store, session_id, "third")
    body = client.get(f"/api/sessions/{session_id}/turns?before=3&limit=2").json()
    assert [t["turn"] for t in body["turns"]] == [1, 2]
    assert "replay_from" not in body
    assert body["has_earlier"] is False


async def test_events_are_included_and_shaped_like_the_stream(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    body = client.get(f"/api/sessions/{turn.session_id}/turns?events=true").json()
    events = body["turns"][0]["events"]
    assert events[0]["type"] == "turn.started"
    assert set(events[0]) == {
        "id",
        "session_id",
        "turn",
        "v",
        "type",
        "run_id",
        "seq",
        "ts",
        "data",
    }
    assert [event["id"] for event in events] == sorted(event["id"] for event in events)


async def test_events_read_past_one_page(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_many_events(console.store, workspace)
    body = client.get(f"/api/sessions/{turn.session_id}/turns?events=true").json()
    events = body["turns"][0]["events"]
    stored = await all_events(console.store, turn.session_id)
    assert len(stored) > 1000
    assert [event["id"] for event in events] == [event.id for event in stored]


async def test_undone_turns_are_listed(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    await add_turn(console.store, turn.session_id, "second")
    await console.store.undo_last_turn(turn.session_id)
    body = client.get(f"/api/sessions/{turn.session_id}/turns").json()
    assert body["turns"][-1]["undone"] is True
    assert body["turns"][-2]["undone"] is False


async def test_an_empty_session_has_no_turns(
    client: TestClient, console: Console, workspace: str
) -> None:
    session = await console.store.create_session(
        model="stub-model", workspace=workspace
    )
    body = client.get(f"/api/sessions/{session.id}/turns").json()
    assert body == {"turns": [], "has_earlier": False, "replay_from": 0}


async def test_a_running_turn_is_listed(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace, status="running")
    body = client.get(f"/api/sessions/{turn.session_id}/turns").json()
    assert body["turns"][0]["status"] == "running"
    assert body["turns"][0]["running_here"] is True
    assert body["turns"][0]["result"] is None


async def test_the_turn_parameters_are_checked(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    assert (
        client.get(f"/api/sessions/{turn.session_id}/turns?limit=0").status_code == 400
    )
    assert (
        client.get(f"/api/sessions/{turn.session_id}/turns?limit=51").status_code == 400
    )
    assert (
        client.get(f"/api/sessions/{turn.session_id}/turns?before=0").status_code == 400
    )


async def test_an_unknown_session_for_turns_is_404(client: TestClient) -> None:
    resp = client.get("/api/sessions/0123456789ab/turns")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
