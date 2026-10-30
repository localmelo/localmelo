"""The Memory tab's routes: reword, delete, forget everything, the switch, and chats that go.

A note outlives its chat: deleting a chat keeps its notes and says "from a deleted
chat" unless the person also forgets them (``?notes=forget``); the index entries
always go. The routes run against the console's app as ``test_memory_routes`` does.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

from localmelo.support import config
from localmelo.support.gateway.console import Console

from .test_memory_routes import ScriptedLLM, say, setup  # noqa: F401  (fixture)

Setup = tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    """The config file of the switch is the test's own."""
    path = tmp_path / "cfg" / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_DIR", str(path.parent)),
        mock.patch.object(config, "CONFIG_PATH", str(path)),
    ):
        yield path


async def notes(client: httpx.AsyncClient, query: str = "") -> dict[str, Any]:
    return dict((await client.get("/api/memory/records" + query)).json())


async def test_a_note_says_where_it_came_from_and_whether_it_was_reworded(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "I prefer short answers")
    await console.memory.drain()
    body = await notes(client)
    (row,) = body["records"]
    assert row["from"].startswith("from chat “I prefer short answers”, ")
    assert (row["chat_gone"], row["edited"], row["session_id"]) == (
        False,
        False,
        session_id,
    )
    assert (body["on"], body["documents"], body["embedding_model"]) == (True, 1, "")
    assert (await notes(client, f"?session_id={session_id}"))["records"] == [row]
    assert (await notes(client, "?session_id=other"))["records"] == []


async def test_rewording_a_note_returns_it_and_refuses_what_cannot_be_kept(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    _, client, _ = setup
    await say(client, tmp_path, "My dentist is Lee Park")
    (row,) = (await notes(client))["records"]
    url = f"/api/memory/records/{row['id']}"
    done = await client.patch(url, json={"text": "Your dentist is Kim Soo."})
    assert done.status_code == 200
    assert (done.json()["text"], done.json()["edited"]) == (
        "Your dentist is Kim Soo.",
        True,
    )
    assert [r["text"] for r in (await notes(client))["records"]] == [
        "Your dentist is Kim Soo."
    ]
    for text, why in (("  ", "empty"), ("My password is hunter2", "password")):
        refused = await client.patch(url, json={"text": text})
        assert refused.status_code == 400 and why in refused.json()["error"]["message"]
    assert (await client.patch(url, json={"text": 3})).status_code == 400
    assert (
        await client.patch("/api/memory/records/999", json={"text": "x y"})
    ).status_code == 404


async def test_a_deleted_chat_keeps_its_notes_and_says_so(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "My dentist is Lee Park")
    await console.memory.drain()
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204
    body = await notes(client)
    (row,) = body["records"]
    assert row["chat_gone"] is True and row["from"].startswith("from a deleted chat, ")
    assert body["documents"] == 0, "the index entries go with the chat"


async def test_a_deleted_chat_can_forget_its_notes_too(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    keep = await say(client, tmp_path, "I live in Pittsburgh")
    forget = await say(client, tmp_path, "My dentist is Lee Park")
    await console.memory.drain()
    assert (
        await client.delete(f"/api/sessions/{forget}?notes=forget")
    ).status_code == 204
    assert [r["session_id"] for r in (await notes(client))["records"]] == [keep]
    assert (
        await client.delete(f"/api/sessions/{keep}?notes=nonsense")
    ).status_code == 400


async def test_forget_everything_needs_its_flag_and_reports_what_went(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    await say(client, tmp_path, "I prefer short answers; my dentist is Lee Park")
    await console.memory.drain()
    assert (await client.delete("/api/memory/records")).status_code == 400
    assert len((await notes(client))["records"]) == 2
    done = await client.delete("/api/memory/records?all=true")
    assert (done.status_code, done.json()) == (200, {"notes": 2, "turns": 1})
    assert await notes(client) == {
        "on": True,
        "records": [],
        "documents": 0,
        "embedding_model": "",
        "last_error": "",
    }


async def test_the_switch_is_saved_and_a_chat_with_memory_off_remembers_nothing(
    setup: Setup, tmp_path: Path, isolated_config: Path  # noqa: F811
) -> None:
    _, client, _ = setup
    off = await client.put("/api/memory", json={"on": False})
    assert (off.status_code, off.json()) == (200, {"on": False})
    assert config.load().memory.across_chats is False
    await say(client, tmp_path, "I prefer short answers")
    body = await notes(client)
    assert (body["on"], body["records"]) == (False, [])
    assert (await client.put("/api/memory", json={"on": True})).json() == {"on": True}
    assert config.load().memory.across_chats is True
    assert (await client.put("/api/memory", json={"on": "yes"})).status_code == 400


async def test_reindex_rebuilds_the_index_from_the_chats(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    await say(client, tmp_path, "Plan the tomato garden for spring")
    await console.memory.drain()
    await console.memory.forget_all()
    assert (await notes(client))["documents"] == 0
    # What the user forgot stays forgotten; only turns after it are indexed again.
    assert (await client.post("/api/memory/reindex")).json() == {"notes": 0, "turns": 0}
    await say(client, tmp_path, "Plan the pepper garden for summer")
    await console.memory.drain()
    assert (await client.post("/api/memory/reindex")).json() == {"notes": 0, "turns": 1}
    assert (await notes(client))["documents"] == 1
