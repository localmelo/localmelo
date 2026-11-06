"""The Memory tab's routes for memory items: list, delete at once, put back.

Items are what a chat model wrote from the chats (``recall_source = "items"``); the
routes run against the console's app as ``test_memory_manage_routes`` does, with items
stored the way the encoder stores them. A delete leaves nothing on disk and can be put
back for a while: from memory, not from the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import httpx

from localmelo.support.gateway.console import Console
from localmelo.support.memory import service
from localmelo.support.memory.index import MemoryItem

from .test_memory_routes import ScriptedLLM, say, setup  # noqa: F401  (fixture)

Setup = tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]


async def store(
    console: Console, session_id: str, title: str, text: str, at: float = 1000.0
) -> int:
    """Store an item of the chat's first turn, as the encoder does."""
    return await console.memory.stores()[1].add_item(
        MemoryItem(
            session_id=session_id,
            title=title,
            text=text,
            kind="semantic",
            created_at=at,
            sources=((session_id, 1),),
        )
    )


async def listed(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    return list((await client.get("/api/memory/items")).json()["items"])


async def test_an_item_is_listed_with_its_chat_and_a_link_to_it(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    assert (await client.get("/api/memory/items")).json() == {"items": []}
    session_id = await say(client, tmp_path, "Plan the tomato garden for spring")
    await console.memory.drain()
    number = await store(console, session_id, "Tomato garden", "Bamboo stakes.", 1.0)
    (row,) = await listed(client)
    assert row == {
        "id": number,
        "title": "Tomato garden",
        "text": "Bamboo stakes.",
        "kind": "semantic",
        "status": "active",
        "replaced_by": None,
        "conflicts": [],
        "from": row["from"],
        "chat_gone": False,
        "link": f"#/s/{session_id}",
        "session_id": session_id,
        "created_at": 1.0,
    }
    assert row["from"].startswith("from chat “Plan the tomato garden for spring”, ")
    # Deleting the chat takes the items that came from it.
    await client.delete(f"/api/sessions/{session_id}")
    assert await listed(client) == []


async def test_an_item_of_a_chat_the_console_does_not_know_says_so(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    await store(console, "terminal-chat", "Tea", "Likes green tea.")
    (row,) = await listed(client)
    assert row["chat_gone"] is True and row["link"] == ""
    assert row["from"].startswith("from a deleted chat, ")


async def test_a_replaced_item_names_what_replaced_it_until_that_is_deleted(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "Where is the weekly sync?")
    old = await store(console, session_id, "Sync", "On Wednesday.", 1.0)
    new = await store(console, session_id, "Sync", "Now on Friday.", 2.0)
    await console.memory.stores()[1].mark_replaced(old, new)
    rows = {row["id"]: row for row in await listed(client)}
    assert [r["id"] for r in await listed(client)] == [new, old], "newest first"
    assert (rows[old]["status"], rows[old]["replaced_by"]) == ("replaced", new)
    assert (rows[new]["status"], rows[new]["replaced_by"]) == ("active", None)
    # Deleting the newer item leaves the older one replaced, with nothing to name.
    assert (await client.delete(f"/api/memory/items/{new}")).status_code == 200
    (left,) = await listed(client)
    assert (left["id"], left["status"], left["replaced_by"]) == (old, "replaced", None)


async def test_a_delete_takes_effect_at_once_and_an_undo_puts_the_item_back(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "Plan the tomato garden for spring")
    await console.memory.drain()
    keep = await store(console, session_id, "Keep", "Stays.", 1.0)
    gone = await store(console, session_id, "Gone", "Is deleted.", 2.0)
    before = await listed(client)
    done = await client.delete(f"/api/memory/items/{gone}")
    assert (done.status_code, done.json()) == (
        200,
        {"undo_seconds": service.ITEM_UNDO_S},
    )
    assert [r["id"] for r in await listed(client)] == [keep]
    # Nothing of it is on disk: only this process can put it back.
    index = console.memory.stores()[1]
    assert await index.count_items() == 1
    back = await client.post(f"/api/memory/items/{gone}/restore")
    assert (back.status_code, back.content) == (204, b"")
    assert await listed(client) == before
    # An undo is used once, and a delete of a missing item says so.
    again = await client.post(f"/api/memory/items/{gone}/restore")
    assert again.status_code == 404 and again.json()["error"]["code"] == "not_found"
    assert "no longer be put back" in again.json()["error"]["message"]
    for method, url in (
        ("DELETE", "/api/memory/items/999"),
        ("POST", "/api/memory/items/999/restore"),
    ):
        missing = await client.request(method, url)
        assert (
            missing.status_code == 404
            and missing.json()["error"]["code"] == "not_found"
        )
    assert (await client.delete("/api/memory/items/nope")).status_code == 400


async def test_an_undo_after_the_time_is_up_finds_nothing(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "Plan the tomato garden for spring")
    number = await store(console, session_id, "Gone", "Is deleted.")
    await client.delete(f"/api/memory/items/{number}")
    with mock.patch.object(service, "ITEM_UNDO_S", 0):
        late = await client.post(f"/api/memory/items/{number}/restore")
    assert late.status_code == 404 and await listed(client) == []


async def test_forgetting_everything_takes_the_items_and_their_undo(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "Plan the tomato garden for spring")
    await console.memory.drain()
    gone = await store(console, session_id, "Gone", "Is deleted.", 1.0)
    await store(console, session_id, "Kept", "Is forgotten with the rest.", 2.0)
    await client.delete(f"/api/memory/items/{gone}")
    done = await client.delete("/api/memory/records?all=true")
    assert done.status_code == 200
    assert await listed(client) == []
    assert (await client.post(f"/api/memory/items/{gone}/restore")).status_code == 404
