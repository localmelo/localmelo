"""A chat's mode decides for the tasks it started that are still running (#576).

Ask to Auto takes effect at the task's next decision, Auto to Ask at once, and a
card already waiting stays for the person. A chore keeps the mode it was saved with.
A task that has ended is left as it was made (``test_auto_mode``).

There is one source of the mode: the chat's. A task reads it at each decision, so
no switch has anything to hand on to a task, and what a task is left holding by an
overlapping switch, an interrupted one or a restart cannot differ from the chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from pathlib import Path

import httpx
import pytest

from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.tests.agent.test_approvals import _answer

from .test_auto_mode import KEYS, approvals, auto_console, create, step, wait_turn

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


def hold_mode_writes(
    console: Console, monkeypatch: pytest.MonkeyPatch, task_id: str
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold a write of *task_id*'s mode at the store, if anything makes one.

    Returns ``(reached, release)``: ``reached`` is set when a write waits, and
    the write goes on once ``release`` is set. The store and its locks are real.
    """
    reached, release = asyncio.Event(), asyncio.Event()
    real = console.store.update_session

    async def update_session(session_id: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if session_id == task_id and "approvals" in (kwargs.get("meta_update") or {}):
            reached.set()
            await release.wait()
        return await real(session_id, *args, **kwargs)

    monkeypatch.setattr(console.store, "update_session", update_session)
    return reached, release


async def parked(console: Console, folder: Path) -> str:
    """A task that has not started, made while the chat has the mode it has."""
    task, _ = await console.workers.create(
        {"prompt": "later", "workspace": str(folder), "level": 5}
    )
    assert task["state"] == "parked"
    return str(task["task_id"])


async def test_the_last_of_two_overlapping_switches_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older switch that is slow to finish never undoes a newer one."""
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
        running, release_run = asyncio.Event(), asyncio.Event()

        async def python_exec(code: str) -> str:
            ran.codes.append(code)
            if code == FIRST:
                running.set()
                await release_run.wait()
            return "ok"

        ran.python_exec = python_exec  # type: ignore[method-assign]
        live, _ = await console.workers.create(
            {"prompt": "tidy the notes", "workspace": str(folder)}
        )
        live_id = live["task_id"]
        await asyncio.wait_for(running.wait(), TIMEOUT_S)
        later = await parked(console, folder)
        reached, release = hold_mode_writes(console, monkeypatch, later)
        older = asyncio.create_task(switch(client, console, "auto"))
        await eventually(lambda: reached.is_set() or older.done())
        # The task the older switch was to reach ends; the newer switch has fewer to reach.
        await console.workers.update(later, state="stop")
        await switch(client, console, "ask")
        release.set()
        await older
        release_run.set()
        second = await waiting(console, live_id, "call_2")
        assert ran.codes == [FIRST]
        assert (await console.workers.get(live_id))["approvals"] == "ask"
        assert (await client.get(f"/api/sessions/{console.main_id}")).json()["session"][
            "approvals"
        ] == "ask"
        console.broker.answer(second.request_id, ApprovalDecision(True, origin="user"))
        await finished(console, live_id)


async def test_a_task_that_ended_keeps_the_mode_it_ended_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch that was on its way to a task does not change it once it ended."""
    monkeypatch.setenv("HOME", str(tmp_path))
    async with auto_console(tmp_path, [], monkeypatch=monkeypatch) as (
        console,
        client,
        _,
        _,
    ):
        folder = tmp_path / "ws"
        folder.mkdir()
        task_id = await parked(console, folder)
        reached, release = hold_mode_writes(console, monkeypatch, task_id)
        sent = asyncio.create_task(switch(client, console, "auto"))
        await eventually(lambda: reached.is_set() or sent.done())
        await console.workers.update(task_id, state="stop")
        release.set()
        await sent
        ended = await console.store.get_session(task_id)
        assert ended is not None and ended.meta["approvals"] == "ask"


async def test_a_task_never_started_starts_with_the_chats_mode_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch that was cut short leaves nothing a later start could read."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [step(FIRST, "call_1"), _answer("done")]
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
        task_id = await parked(console, folder)
        real = console.store.update_session

        async def cut_short(session_id: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if session_id == task_id and "approvals" in (
                kwargs.get("meta_update") or {}
            ):
                raise RuntimeError("the gateway stopped here")
            return await real(session_id, *args, **kwargs)

        monkeypatch.setattr(console.store, "update_session", cut_short)
        with contextlib.suppress(RuntimeError):
            # Whether the request is answered or fails is not what is asked here.
            await client.patch(
                f"/api/sessions/{console.main_id}", json={"approvals": "ask"}
            )
        monkeypatch.setattr(console.store, "update_session", real)
        chat = await console.store.get_session(console.main_id)
        assert chat is not None and chat.meta["approvals"] == "ask"
        await console.workers.update(task_id, state="resume")
        first = await waiting(console, task_id, "call_1")
        assert ran.codes == []
        console.broker.answer(first.request_id, ApprovalDecision(True, origin="user"))
        await finished(console, task_id)


async def test_a_switch_is_seen_by_the_next_decision_of_a_task_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a read shows Ask, no task decides on Auto's word any more."""
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
        running, release_run = asyncio.Event(), asyncio.Event()

        async def python_exec(code: str) -> str:
            ran.codes.append(code)
            if code == FIRST:
                running.set()
                await release_run.wait()
            return "ok"

        ran.python_exec = python_exec  # type: ignore[method-assign]
        task, _ = await console.workers.create(
            {"prompt": "tidy the notes", "workspace": str(folder)}
        )
        task_id = task["task_id"]
        await asyncio.wait_for(running.wait(), TIMEOUT_S)
        # Whatever else a switch waits for in the store stays held.
        reached, release = asyncio.Event(), asyncio.Event()
        real = console.workers.tasks.unfinished

        async def unfinished(*args, **kwargs):  # type: ignore[no-untyped-def]
            reached.set()
            await release.wait()
            return await real(*args, **kwargs)

        monkeypatch.setattr(console.workers.tasks, "unfinished", unfinished)
        sent = asyncio.create_task(switch(client, console, "ask"))
        shown = ""
        await eventually(lambda: shown != "" or sent.done() or reached.is_set())
        read = await client.get(f"/api/sessions/{console.main_id}")
        shown = read.json()["session"]["approvals"]
        assert shown == "ask"
        release_run.set()
        await eventually(
            lambda: SECOND in ran.codes
            or any(r.call_id == "call_2" for r in console.broker.pending(task_id))
        )
        assert ran.codes == [FIRST], "the step ran on Auto's word after Ask was shown"
        release.set()
        await sent
        second = await waiting(console, task_id, "call_2")
        console.broker.answer(second.request_id, ApprovalDecision(True, origin="user"))
        await finished(console, task_id)


async def test_a_task_of_a_deleted_chat_follows_the_main_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task its chat went from is the main chat's, and takes its mode."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        _answer("hi"),
        step(FIRST, "call_1"),
        step(SECOND, "call_2"),
        _answer("ok"),
    ]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        _,
    ):
        folder = tmp_path / "ws"
        side = await create(client, tmp_path, "hello", approvals="auto")
        await wait_turn(client, side)
        running, release_run = asyncio.Event(), asyncio.Event()

        async def python_exec(code: str) -> str:
            ran.codes.append(code)
            if code == FIRST:
                running.set()
                await release_run.wait()
            return "ok"

        ran.python_exec = python_exec  # type: ignore[method-assign]
        task, _ = await console.workers.create(
            {"prompt": "tidy", "workspace": str(folder), "front_id": side}
        )
        task_id = task["task_id"]
        assert task["approvals"] == "auto"
        await asyncio.wait_for(running.wait(), TIMEOUT_S)
        gone = await client.delete(f"/api/sessions/{side}")
        assert gone.status_code in (200, 204), gone.text
        release_run.set()
        # The main chat is in Ask: the next step asks the person.
        second = await waiting(console, task_id, "call_2")
        assert ran.codes == [FIRST]
        console.broker.answer(second.request_id, ApprovalDecision(True, origin="user"))
        await finished(console, task_id)
