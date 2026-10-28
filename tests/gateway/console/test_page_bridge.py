"""API v2 header bridge: level, presence, machine and the session's S2 facts.

The page talks only to ``/api``, so the S2/S2b header it keeps (the machine
line, the P1-P4 chip, ``Waiting: <held>``) reads and writes the one gate
through these routes instead of ``/v1/playground``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.tests.sessions.helpers import answer

from .test_turn_writes import (
    _create,
    _post_turn,
    _wait_turn,
    _workspace,
    console_with,
)


async def test_the_session_list_carries_the_machine_and_s2_facts(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        listed = (await client.get("/api/sessions")).json()

        machine = listed["machine"]
        assert set(machine) >= {
            "lock",
            "holder",
            "purpose",
            "stale",
            "readable",
            "gate",
        }
        [session] = listed["sessions"]
        assert session["level"] == 2
        assert session["held"] == ""


async def test_a_level_is_set_and_read_back(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        response = await client.post(
            f"/api/sessions/{session_id}/level", json={"level": 3}
        )

        assert response.status_code == 200
        assert response.json() == {"session_id": session_id, "level": 3}
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["session"]["level"] == 3
        listed = (await client.get("/api/sessions")).json()
        assert listed["machine"]["gate"] in ("record", "wait")


@pytest.mark.parametrize("level", [0, 5, "two", True])
async def test_a_level_outside_p1_p4_is_invalid(tmp_path: Path, level: Any) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        response = await client.post(
            f"/api/sessions/{session_id}/level", json={"level": level}
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


async def test_a_level_for_an_unknown_session_is_not_found(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.post(
            "/api/sessions/0123456789ab/level", json={"level": 2}
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


async def test_presence_is_a_heartbeat(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, client, _):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        response = await client.post("/api/presence", json={"session_id": session_id})

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        # The gate's focus moved to this chat.
        assert console.gate._focus == session_id


async def test_presence_for_an_unknown_session_is_not_found(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.post(
            "/api/presence", json={"session_id": "0123456789ab"}
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


async def test_a_create_with_a_first_turn_sets_the_gates_turn_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        started: list[tuple[str, str]] = []
        monkeypatch.setattr(
            console.gate, "turn_start", lambda sid: started.append(("start", sid))
        )
        monkeypatch.setattr(
            console.gate, "focus", lambda sid: started.append(("focus", sid))
        )

        body = await _create(client, tmp_path, "Hi there")

        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        # A create's first turn is a turn like any other (S2).
        assert started == [("start", session_id), ("focus", session_id)]


async def test_a_turn_resets_the_gates_turn_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        started: list[tuple[str, str]] = []
        monkeypatch.setattr(
            console.gate, "turn_start", lambda sid: started.append(("start", sid))
        )
        monkeypatch.setattr(
            console.gate, "focus", lambda sid: started.append(("focus", sid))
        )
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]

        response = await _post_turn(client, session_id, "Hi there")

        assert response.status_code == 202
        await _wait_turn(client, session_id)
        # A turn starts under the same gate the page's chats use (S2).
        assert started == [("start", session_id), ("focus", session_id)]


async def test_a_retry_resets_the_gates_turn_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with console_with(tmp_path, answer("hello"), answer("hello again")) as (
        console,
        client,
        _,
    ):
        created = await client.post("/api/sessions", json=_workspace(tmp_path))
        session_id = created.json()["session"]["id"]
        posted = await _post_turn(client, session_id, "Hi there")
        turn_id = posted.json()["turn_id"]
        await _wait_turn(client, session_id)

        started: list[tuple[str, str]] = []
        monkeypatch.setattr(
            console.gate, "turn_start", lambda sid: started.append(("start", sid))
        )
        monkeypatch.setattr(
            console.gate, "focus", lambda sid: started.append(("focus", sid))
        )

        retried = await client.post(
            f"/api/sessions/{session_id}/retry", json={"turn_id": turn_id}
        )

        assert retried.status_code == 202
        await _wait_turn(client, session_id)
        assert started == [("start", session_id), ("focus", session_id)]
