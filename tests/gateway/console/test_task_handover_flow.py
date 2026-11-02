"""A task the chat starts begins with the chat's turns and what the chat recalled.

The person talks about a friend, then sends a short request ("find backend
roles from his résumé"). The task the chat starts keeps,
with its spec, the chat's newest earlier turns and the memory the chat's turn
recalled, so its first request knows whom the work is about. A task the
person makes on the tasks page, and one started again, get neither. Scripted
models; memory words only; nothing reaches a model server or the network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localmelo.melo.agent._earlier import EARLIER_HEAD
from localmelo.support.gateway.handover import PICKED_HEADER
from localmelo.support.gateway.workers import TASK_PACK_HEADER
from localmelo.tests.memory.test_recall_list import add_turn
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_ask_owner import Reply, tool_results
from .test_chat_owns_tasks import owner_console, say, started_task

FRIEND = "My friend Alex is looking for a new job."
WANTS = "He wants a staff role at a large company."
# A short request whose words alone do not say whom it is about.
REQUEST = "find backend roles from his résumé"


def creates(objective: str) -> Reply:
    return tool_call(
        "task_create", {"title": "Roles for Alex", "objective": objective}, "create-1"
    )


async def test_a_short_request_hands_the_task_the_chats_turns_and_its_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        answer("Noted: Alex is looking."),
        answer("A staff role at a large company, got it."),
        creates("Look up roles for Alex."),
        answer("Started; the result comes back here."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        # Another chat talked about Alex's résumé earlier.
        await add_turn(
            console.memory,
            console.store,
            "Alex's résumé lists eight years of backend work at a large company.",
            "Noted.",
            title="Résumé",
        )
        await say(client, console, FRIEND)
        await say(client, console, WANTS)
        await say(client, console, REQUEST)
        assert tool_results(models["main"])[-1].startswith("Started task ")
        record = await console.workers.tasks.get_task(await started_task(console))
        spec = record.spec
        # The chat's two earlier turns, oldest first; never the request itself.
        turns = spec["chat_turns"].splitlines()
        assert turns[0] == f"- the user: {FRIEND}"
        assert f"- the user: {WANTS}" in turns
        assert REQUEST not in spec["chat_turns"]
        # What the chat's turn recalled: the other chat's résumé turn.
        assert "eight years of backend work" in spec["chat_picked"]
        assert "[t" not in spec["chat_picked"]
        # The objective is the person's message: their words are not repeated.
        assert "own_words" not in spec


async def test_a_task_made_on_the_tasks_page_gets_no_chat_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[answer("Noted.")], run=False
    ) as (console, client, _, _):
        await say(client, console, FRIEND)
        response = await client.post(
            "/api/tasks", json={"prompt": "Look up roles for Alex."}
        )
        assert response.status_code == 202, response.text
        record = await console.workers.tasks.get_task(
            response.json()["task"]["task_id"]
        )
        for name in ("chat_turns", "chat_picked", "own_words"):
            assert name not in record.spec


async def test_the_task_reads_them_as_data_after_its_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        answer("Noted: Alex is looking."),
        creates("Look up openings for Alex."),
        answer("Started."),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[[answer("Done.\nEnding: ready")]]
    ) as (console, client, models, _):
        await add_turn(
            console.memory,
            console.store,
            "Alex's résumé lists eight years of backend work.",
            "Noted.",
            title="Résumé",
        )
        await say(client, console, FRIEND)
        await say(client, console, REQUEST)
        task_id = await started_task(console)
        record = await console.workers.tasks.get_task(task_id)
        while not models["workers"] or not models["workers"][0].requests:
            await _tick()
        first = "\n".join(
            str(message.content) for message in models["workers"][0].requests[0]
        )
        request = record.spec["objective"]
        assert first.index(request) < first.index(EARLIER_HEAD)
        assert f"- the user: {FRIEND}" in first
        assert first.index(EARLIER_HEAD) < first.index(PICKED_HEADER)
        assert "eight years of backend work" in first[first.index(PICKED_HEADER) :]
        if TASK_PACK_HEADER in first:
            assert first.index(PICKED_HEADER) < first.index(TASK_PACK_HEADER)
        # Not in the task's stored turn: that keeps its request only.
        turns = await console.store.list_turns(task_id, limit=5)
        assert turns and turns[0].user_text == request


async def _tick() -> None:
    await asyncio.sleep(0.02)
