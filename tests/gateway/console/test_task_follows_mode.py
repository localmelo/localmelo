"""A chat's mode decides for the tasks it started that are still running (#576).

Ask to Auto takes effect at the task's next decision, Auto to Ask at once, and a
card already waiting stays for the person. A chore keeps the mode it was saved with.
A task that has ended is left as it was made (``test_auto_mode``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path

import httpx
import pytest

from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.tests.agent.test_approvals import _answer

from .test_auto_mode import KEYS, approvals, auto_console, step

TIMEOUT_S = 5.0
FIRST = "print(sorted(__import__('os').listdir('.')))"
SECOND = "print('second')"


async def eventually(check) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + TIMEOUT_S
    while not check():
        assert time.monotonic() < deadline, "it did not come"
        await asyncio.sleep(0.01)


async def waiting(console: Console, task_id: str, call_id: str) -> ApprovalRequest:
    """The approval of *call_id* once it waits for the person."""
    await eventually(
        lambda: any(r.call_id == call_id for r in console.broker.pending(task_id))
    )
    return next(r for r in console.broker.pending(task_id) if r.call_id == call_id)


async def finished(console: Console, task_id: str) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while (await console.workers.get(task_id))["state"] not in ("done", "failed"):
        assert time.monotonic() < deadline, "the task did not finish"
        await asyncio.sleep(0.02)


async def switch(client: httpx.AsyncClient, console: Console, mode: str) -> None:
    sent = await client.patch(
        f"/api/sessions/{console.main_id}", json={"approvals": mode}
    )
    assert sent.status_code == 200, sent.text


async def test_a_switch_to_auto_reaches_a_task_that_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [step(FIRST, "call_1"), step(KEYS, "call_2"), _answer("It was blocked.")]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        _,
    ):
        folder = tmp_path / "ws"
        folder.mkdir()
        task, _ = await console.workers.create(
            {"prompt": "tidy the notes", "workspace": str(folder)}
        )
        task_id = task["task_id"]
        assert task["approvals"] == "ask"
        first = await waiting(console, task_id, "call_1")
        await switch(client, console, "auto")
        # The card that waited stays for the person; the task now shows the chat's mode.
        assert [r.request_id for r in console.broker.pending(task_id)] == [
            first.request_id
        ]
        assert (await console.workers.get(task_id))["approvals"] == "auto"
        console.broker.answer(first.request_id, ApprovalDecision(True, origin="user"))
        await finished(console, task_id)
        # The next decision was Auto's: the key step was stopped, nobody was asked.
        assert ran.codes == [FIRST]
        found = await approvals(console, task_id)
        assert [(a["origin"], a["allowed"]) for a in found] == [
            ("user", True),
            ("auto", False),
        ]
        assert console.broker.pending(task_id) == []


async def test_a_switch_to_ask_reaches_a_task_that_is_running_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [step(FIRST, "call_1"), step(SECOND, "call_2"), _answer("done")]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        _,
    ):
        folder = tmp_path / "ws"
        folder.mkdir()
        assert console.main_id is not None
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "auto"}
        )
        running, release = asyncio.Event(), asyncio.Event()

        async def python_exec(code: str) -> str:
            ran.codes.append(code)
            if code == FIRST:
                running.set()
                await release.wait()
            return "ok"

        ran.python_exec = python_exec  # type: ignore[method-assign]
        task, _ = await console.workers.create(
            {"prompt": "tidy the notes", "workspace": str(folder)}
        )
        task_id = task["task_id"]
        assert task["approvals"] == "auto"
        await asyncio.wait_for(running.wait(), TIMEOUT_S)
        await switch(client, console, "ask")
        release.set()
        # The next step asks the person instead of running on Auto's word.
        second = await waiting(console, task_id, "call_2")
        assert ran.codes == [FIRST]
        console.broker.answer(second.request_id, ApprovalDecision(True, origin="user"))
        await finished(console, task_id)
        assert ran.codes == [FIRST, SECOND]


async def test_a_chore_keeps_the_mode_it_was_saved_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [step(FIRST, "call_1"), step(KEYS, "call_2"), _answer("done")]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        _,
    ):
        spec = TaskSpec.from_payload({"objective": "tidy the notes", "title": "Tidy"})
        spec = dataclasses.replace(spec, repeats=True)
        task, _ = await console.workers.create_spec(
            spec, origin="user_rule", front_id=console.workers.front_id
        )
        task_id = task["task_id"]
        first = await waiting(console, task_id, "call_1")
        await switch(client, console, "auto")
        assert (await console.workers.get(task_id))["approvals"] == "ask"
        console.broker.answer(first.request_id, ApprovalDecision(True, origin="user"))
        # Still Ask: the key step asks the person, it is not stopped by Auto.
        second = await waiting(console, task_id, "call_2")
        console.broker.answer(second.request_id, ApprovalDecision(False, "no"))
        await finished(console, task_id)
        assert ran.codes == [FIRST]
