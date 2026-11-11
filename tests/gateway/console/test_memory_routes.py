"""Memory across chats in the console: records, Undo, recall, and deletes that cascade.

The routes run over ``httpx.AsyncClient`` against the console's app in the
test's loop, with a scripted model, as the turn tests do. The memory folder is
the test's own.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.memory import records
from localmelo.support.memory.recall import BLOCK_HEADER
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, started_agent

from .conftest import PORT, FakeProbe, make_cfg, quiet_gate


@pytest.fixture()
async def setup(
    tmp_path: Path,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]]:
    llms: list[ScriptedLLM] = []

    async def build(settings: Any, workspace: str, mcp: str | None, sid: str) -> Any:
        llms.append(ScriptedLLM([answer("OK.")]))
        return await started_agent(Path(workspace), llms[-1], mode="gateway")

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
            memory_dir=str(tmp_path / "memory"),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, llms
    finally:
        await client.aclose()
        await console.aclose()


async def say(client: httpx.AsyncClient, tmp_path: Path, text: str) -> str:
    """A new chat whose first turn is *text*; returns its id once the turn ended."""
    folder = tmp_path / f"ws{time.monotonic_ns()}"
    folder.mkdir()
    made = await client.post(
        "/api/sessions", json={"workspace": str(folder), "turn": {"text": text}}
    )
    assert made.status_code == 201, made.text
    session_id = str(made.json()["session"]["id"])
    for _ in range(500):
        turns = (await client.get(f"/api/sessions/{session_id}/turns")).json()["turns"]
        if turns and turns[-1]["status"] != "running":
            return session_id
        await asyncio.sleep(0.01)
    raise AssertionError("the turn did not finish")


async def memory_event(console: Console, session_id: str) -> dict[str, Any]:
    events = await console.store.list_events(session_id)
    (found,) = [e.event["data"] for e in events if e.event["type"] == "turn.memory"]
    return dict(found)


async def test_records_list_undo_and_a_second_chat_recalls_them(
    setup: tuple[Console, httpx.AsyncClient, list[ScriptedLLM]], tmp_path: Path
) -> None:
    console, client, llms = setup
    first = await say(
        client,
        tmp_path,
        "Remember that I prefer short answers; remember that my group meeting is "
        "on Thursdays",
    )
    said = await memory_event(console, first)
    assert [r["text"] for r in said["remembered"]] == [
        "You prefer short answers.",
        "Your group meeting is on Thursdays.",
    ]
    listed = (await client.get("/api/memory/records")).json()
    assert listed["on"] is True
    assert [r["topic"] for r in listed["records"]] == ["about", "preferences"]
    assert listed["records"][0]["source"].startswith(
        "you said it · chat “Remember that I prefer short"
    )

    second = await say(client, tmp_path, "when is my group meeting?")
    used = (await memory_event(console, second))["used"]
    assert [u["text"] for u in used] == ["Your group meeting is on Thursdays."]
    asked = llms[-1].requests[0][-1].content
    assert asked.startswith("when is my group meeting?\n\n" + BLOCK_HEADER)

    note = said["remembered"][0]["id"]
    gone = await client.delete(f"/api/memory/records/{note}?undo=true")
    assert gone.status_code == 204
    again = await client.delete(f"/api/memory/records/{note}")
    assert again.status_code == 404
    assert again.json()["error"]["code"] == "not_found"
    assert len((await client.get("/api/memory/records")).json()["records"]) == 1


async def test_deleting_a_chat_deletes_its_documents(
    setup: tuple[Console, httpx.AsyncClient, list[ScriptedLLM]], tmp_path: Path
) -> None:
    console, client, _ = setup
    session_id = await say(client, tmp_path, "Plan the tomato garden for spring")
    await console.memory.drain()
    _, index = console.memory.stores()
    assert await index.count() == 1
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204
    assert await index.count() == 0
    assert await console.memory.search("tomato garden", docs_k=8) == []


async def test_a_note_from_material_a_result_or_inferred_says_so_and_a_typed_one_reads_as_before(
    setup: tuple[Console, httpx.AsyncClient, list[ScriptedLLM]], tmp_path: Path
) -> None:
    """The notes page names the page, the file, the finished task's result or the
    model as a note's source (``[memory] note_gate``, #442) in place of "you said
    it"; a typed note and a judged one (typed words, worded differently) read as
    they always did."""
    console, client, _ = setup
    session_id = await say(client, tmp_path, "Plan my job search")
    at = {"session_id": session_id, "turn": 1, "chat": "Plan my job search"}

    async def keep(text: str, **provenance: Any) -> int:
        (written,) = await records.write(
            console.memory.stores()[0],
            [records.Candidate(text, "", "about", "user_statement")],
            [None],
            source={**at, **provenance},
            embed_model="",
        )
        return written.id

    typed = await keep("You prefer short answers.")
    page = await keep(
        "You have an MSc in Robotics.",
        source="pointed",
        **{"from": "https://www.jane.example.com/about"},
    )
    file = await keep(
        "You know Python and Rust.", source="pointed", **{"from": "resume.pdf"}
    )
    result = await keep("Your flat lease ends June 30.", source="result")
    judged = await keep("你喜欢慢跑。", judged=True)
    inferred = await keep(
        "You are looking for roles at large tech companies.", source="inferred"
    )
    rows = {
        row["id"]: row
        for row in (await client.get("/api/memory/records")).json()["records"]
    }
    day = rows[typed]["source"].rsplit(", ", 1)[1]
    chat = "chat “Plan my job search”"
    # A typed note reads as it always did.
    assert rows[typed]["source"] == f"you said it · {chat}, {day}"
    assert rows[typed]["from"] == f"from {chat}, {day}"
    # A note from material the user pointed to names the page or the file.
    assert rows[page]["source"] == f"from jane.example.com, {day}"
    assert rows[page]["from"] == f"from jane.example.com, {day}"
    assert rows[file]["source"] == f"from resume.pdf, {day}"
    assert rows[file]["from"] == f"from the file resume.pdf, {day}"
    # A note a finished task's result supports says so, and names no chat.
    assert rows[result]["source"] == f"from a finished task's result, {day}"
    assert rows[result]["from"] == f"from a finished task's result, {day}"
    # A judged note is the user's own words, worded differently.
    assert rows[judged]["source"] == f"you said it · {chat}, {day}"
    assert rows[judged]["from"] == f"from {chat}, {day}"
    # A note the model inferred says so, and in which chat.
    assert rows[inferred]["source"] == f"melo inferred it · {chat}, {day}"
    assert rows[inferred]["from"] == f"melo inferred it from {chat}, {day}"
    assert not any(row["imported"] for row in rows.values())
