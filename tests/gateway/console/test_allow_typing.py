"""A typing step Auto blocked in a task nobody can be asked for, and its Allow once.

A real Agent with a scripted model, and a ``browser_type`` that asks its
question through the browser's own ``require_type`` (the self-asking path,
no browser), in a scheduled task's Auto: nobody is asked, the step is blocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent
from localmelo.melo.schema import Message
from localmelo.support.gateway.auto_blocks import ALLOWED_HEAD
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.tests.agent.test_allow_question import NOTE, SENDS, SITE, Types, typing
from localmelo.tests.agent.test_approvals import _answer
from localmelo.tests.agent.test_auto_mode import FakeModel
from localmelo.tests.sessions.helpers import started_agent

from .conftest import PORT, FakeProbe, all_events, make_cfg, quiet_gate

TIMEOUT_S = 10.0


@contextlib.asynccontextmanager
async def console_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plans: list[Message],
    reviews: list[Any],
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, Types, FakeModel]]:
    llm = FakeModel(plans, reviews)
    types = Types()

    async def build_task_agent(**kwargs: Any) -> Agent:
        options = dataclasses.replace(
            kwargs["options"],
            workspace_root=kwargs["workspace"],
            record_answers=True,
        )
        agent = Agent(llm=llm, options=options, tool_providers=[types])
        await agent.start()
        return agent

    monkeypatch.setattr(
        "localmelo.support.gateway.workers.build_session_agent", build_task_agent
    )

    async def build(settings: Any, workspace: str, *_: Any) -> Any:
        return await started_agent(Path(workspace), llm, mode="gateway")

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, types, llm
    finally:
        await client.aclose()
        await console.aclose()


async def scheduled_task(
    console: Console, tmp_path: Path, chat_started: bool = False
) -> str:
    """A task in Auto, run to its end: one nobody can be asked for (a chore's
    origin), or with *chat_started* a worker a chat started, nobody watching."""
    folder = tmp_path / "localmelo-workspace" / "ws"
    folder.mkdir(parents=True, exist_ok=True)
    assert console.main_id is not None
    await console.store.update_session(
        console.main_id, meta_update={"approvals": "auto"}
    )
    spec = TaskSpec.from_payload({"objective": "do the daily check-in"})
    task, _ = await console.workers.create_spec(
        dataclasses.replace(spec, workspace=str(folder)),
        **({"chat_started": True} if chat_started else {"origin": "user_rule"}),
    )
    task_id = str(task["task_id"])
    await settled(console, task_id)
    return task_id


async def settled(console: Console, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        task = await console.workers.get(task_id)
        if task["state"] not in ("queued", "running", "starting"):
            return task
        assert time.monotonic() < deadline, f"the task did not settle: {task['state']}"
        await asyncio.sleep(0.02)


def told(request: list[Message]) -> str:
    """All a planning request held."""
    return "\n".join(m.content for m in request)


async def finished_calls(console: Console, task_id: str) -> list[dict[str, Any]]:
    return [
        e.event["data"]
        for e in await all_events(console.store, task_id)
        if e.event["type"] == "tool.finished"
    ]


@pytest.mark.asyncio
async def test_a_blocked_typing_step_is_listed_carded_and_allowed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        typing(),
        _answer("It needs the note.\nEnding: blocked: may it type the note?"),
        # After Allow once: the allowed text, then the same text again.
        typing(NOTE, "call_2"),
        typing(NOTE, "call_3"),
        _answer("Typed the note."),
    ]
    async with console_with(tmp_path, monkeypatch, plans, [SENDS, SENDS]) as (
        console,
        client,
        types,
        llm,
    ):
        task_id = await scheduled_task(console, tmp_path)
        # No allowance yet: no note of one.
        assert not any(ALLOWED_HEAD in told(r) for r in llm.requests)
        body = (await client.get(f"/api/tasks/{task_id}")).json()["task"]
        [blocked] = body["blocked"]
        assert blocked["can_allow"] is True and blocked["used_at"] is None
        assert (blocked["name"], blocked["question"]) == (
            "browser_type",
            f"browser-type:{SITE}",
        )
        assert blocked["arguments"]["text"] == NOTE, "the whole text, for the page"
        # Its chat shows it as a blocked step, with the same facts.
        assert console.main_id is not None
        await console.cards.flush()
        [card] = [
            e.event["data"]
            for e in await all_events(console.store, console.main_id)
            if e.event["type"] == "attention.card"
            and e.event["data"].get("state") == "blocked"
        ]
        assert (card["request_id"], card["question"], card["can_allow"]) == (
            blocked["request_id"],
            f"browser-type:{SITE}",
            True,
        )
        assert card["arguments"]["text"] == NOTE
        # The task's end card offers its one open step, with what it types.
        [end] = [
            e.event["data"]
            for e in await all_events(console.store, console.main_id)
            if e.event["type"] == "attention.card" and "block" in e.event["data"]
        ]
        assert end["block"]["request_id"] == blocked["request_id"]
        assert end["block"]["arguments"]["text"] == NOTE

        allowed = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert allowed.status_code == 202, allowed.text
        assert set(allowed.json()["task"]["allowances"]) == {blocked["step_key"]}
        await settled(console, task_id)
        # The continued run read the allowed step, its whole text, in melo's note.
        went_on = told(llm.requests[2])
        assert ALLOWED_HEAD in went_on
        assert f"text:\n{NOTE}" in went_on.split(ALLOWED_HEAD, 1)[1]

        # The allowed text ran once; the same text again was a new block (the
        # job finished since, so nothing is offered any more).
        assert types.typed == [NOTE]
        first, again = (await console.workers.get(task_id))["blocked"]
        assert first["request_id"] == blocked["request_id"]
        assert first["used_at"] is not None
        assert again["used_at"] is None and again["finished"] is True
        refused = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            409,
            "already_allowed",
        )


@pytest.mark.asyncio
async def test_a_fresh_task_its_job_goes_on_in_reads_the_allowed_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker a chat started runs once (#571): Allow once goes on in a fresh task,
    which reads the step from the ended task's block and runs it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        typing(),
        _answer("It needs the note.\nEnding: blocked: may it type the note?"),
        typing(NOTE, "call_2"),
        _answer("Typed the note."),
    ]
    async with console_with(tmp_path, monkeypatch, plans, [SENDS]) as (
        console,
        client,
        types,
        llm,
    ):
        task_id = await scheduled_task(console, tmp_path, chat_started=True)
        [blocked] = (await console.workers.get(task_id))["blocked"]
        allowed = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert allowed.status_code == 202, allowed.text
        assert allowed.json()["message"] == "successor"
        await settled(console, allowed.json()["task"]["task_id"])

        went_on = told(llm.requests[2])
        assert f"text:\n{NOTE}" in went_on.split(ALLOWED_HEAD, 1)[1]
        assert types.typed == [NOTE]
