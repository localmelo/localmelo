"""Sleep on its own: the idle gates, the line in the main chat, and its Undo.

The gateway tidies the notes by itself only when the setting is not ``off`` and the
computer is idle (no recent turn, no task, no GPU lock, on AC power, 6 hours after the last
run). Each test drives a console with a fake clock and fake readers of the lock and the
power; the notes are made through the console's own routes with a scripted model that never
keeps a note, so a rules sleep has something to recover.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.front import ECHO_EVENT
from localmelo.support.gateway.console.idle_sleep import (
    META_KEY,
    SHOWN,
    SPACING_S,
    TEXT_CHARS,
    IdleSleepOptions,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.memory.sleep import Change, SleepReport
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, started_agent

from .conftest import PORT, FakeProbe, make_cfg, quiet_gate, seed_turn
from .test_memory_routes import say

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n"
BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t64%; discharging; 3:10 remaining present: true\n"
HOUR = 3600.0


@dataclass
class Machine:
    """What the idle sleep reads of the computer: the time, the GPU lock and the power."""

    now: float = field(default_factory=lambda: time.time() + 3 * HOUR)
    lock: bool = False
    power: str = AC


@dataclass
class World:
    console: Console
    client: httpx.AsyncClient
    machine: Machine
    tmp_path: Path
    runs: list[str | None] = field(default_factory=list)

    async def note(self, text: str = "I live in Utrecht.") -> str:
        """A chat whose turn the model keeps no note of; a rules sleep recovers it."""
        session_id = await say(self.client, self.tmp_path, text)
        await self.console.memory.drain()
        return session_id

    async def echoes(self) -> list[dict[str, Any]]:
        assert self.console.main_id is not None
        found = await self.console.store.events_of_type(
            self.console.main_id, ECHO_EVENT
        )
        return [{"id": e.id, **e.event["data"]} for e in found]

    async def tap(self, echo_id: int, do: str = "undo", **headers: str) -> Any:
        return await self.client.post(
            f"/api/front/echoes/{echo_id}", json={"do": do}, headers=headers
        )

    async def notes(self) -> list[str]:
        listed = (await self.client.get("/api/memory/records")).json()["records"]
        return [str(row["text"]) for row in listed]


@contextlib.asynccontextmanager
async def world(
    tmp_path: Path, *, sleep: str = "rules", poll_s: float = 1e9, lock: bool = False
) -> AsyncIterator[World]:
    machine = Machine(lock=lock)
    llms: list[ScriptedLLM] = []

    async def build(settings: Any, workspace: str, mcp: str | None, sid: str) -> Any:
        llms.append(ScriptedLLM([answer("Noted.")]))
        return await started_agent(Path(workspace), llms[-1], mode="gateway")

    cfg = make_cfg()
    cfg.memory.sleep = sleep
    console = Console(
        cfg,
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
            memory_dir=str(tmp_path / "memory"),
            idle_sleep=IdleSleepOptions(
                clock=lambda: machine.now,
                poll_s=poll_s,
                lock_present=lambda: machine.lock,
                pmset=lambda: machine.power,
            ),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    found = World(console, client, machine, tmp_path)
    # Count the sleeps that really begin.
    original = console.memory.sleep

    async def counted(variant: str | None = None, **kwargs: Any) -> SleepReport:
        found.runs.append(variant)
        return await original(variant, **kwargs)

    console.memory.sleep = counted  # type: ignore[method-assign]
    try:
        yield found
    finally:
        await client.aclose()
        await console.aclose()


async def stamped(w: World) -> float:
    return await w.console.memory.stores()[1].meta(META_KEY)


# ── The gates ──


async def test_nothing_happens_while_the_setting_is_off(tmp_path: Path) -> None:
    async with world(tmp_path, sleep="off") as w:
        await w.note()
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "off"
        assert w.runs == [] and await w.echoes() == []
        assert await stamped(w) == 0


async def test_memory_across_chats_off_blocks_it(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        w.console.cfg.memory.across_chats = False
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "off" and w.runs == []


async def test_a_turn_in_the_last_twenty_minutes_blocks_it(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        real = time.time()
        w.machine.now = real + 19 * 60
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "active" and w.runs == []
        w.machine.now = real + 21 * 60
        report = await w.console.idle_sleep.tick()
        assert report is not None and len(w.runs) == 1


async def test_a_turn_that_runs_now_blocks_it_however_old_it_is(
    tmp_path: Path,
) -> None:
    async with world(tmp_path) as w:
        folder = tmp_path / "ws"
        folder.mkdir()
        await seed_turn(w.console.store, str(folder), status="running")
        w.machine.now = time.time() + 30 * HOUR
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "active" and w.runs == []


@pytest.mark.parametrize("state", ["queued", "running", "held", "parked"])
async def test_a_task_that_runs_or_waits_for_its_turn_blocks_it(
    tmp_path: Path, state: str
) -> None:
    async with world(tmp_path) as w:
        await w.note()
        folder = tmp_path / "tasks"
        folder.mkdir()
        spec = TaskSpec.from_payload(
            {"objective": "tidy the photos", "title": "Photos"}
        )
        record, _ = await w.console.workers.tasks.create_task(
            spec, origin="user", workspace=str(folder), front_id=w.console.main_id
        )
        async with w.console.store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE tasks SET state = ?, updated_at = ? WHERE task_id = ?",
                (state, time.time(), record.task_id),
            )
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "tasks" and w.runs == []
        # A finished task does not.
        async with w.console.store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE tasks SET state = 'done', finished_at = ?, updated_at = ? "
                "WHERE task_id = ?",
                (time.time(), time.time(), record.task_id),
            )
        assert await w.console.idle_sleep.tick() is not None


async def test_the_gpu_lock_blocks_it(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        w.machine.lock = True
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "lock" and w.runs == []
        w.machine.lock = False
        assert await w.console.idle_sleep.tick() is not None


async def test_the_lock_file_of_the_scheduler_setting_is_read_not_written(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "gpu-lock"
    async with world(tmp_path) as w:
        await w.note()
        w.console.idle_sleep.options = IdleSleepOptions(
            clock=lambda: w.machine.now, pmset=lambda: AC
        )
        w.console.cfg.scheduler.gpu_lock_file = str(lock)
        lock.write_text("holder: a bench\npurpose: a run\n")
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "lock"
        assert lock.read_text() == "holder: a bench\npurpose: a run\n"
        lock.unlink()
        assert await w.console.idle_sleep.tick() is not None
        assert not lock.exists()


@pytest.mark.parametrize(
    "power",
    [BATTERY, "", "pmset said something else\n"],
    ids=["battery", "none", "odd"],
)
async def test_only_ac_power_lets_it_run(tmp_path: Path, power: str) -> None:
    async with world(tmp_path) as w:
        await w.note()
        w.machine.power = power
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "battery" and w.runs == []
        w.machine.power = AC
        assert await w.console.idle_sleep.tick() is not None


async def test_a_power_reading_that_fails_blocks_it(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()

        def broken() -> str:
            raise OSError("pmset exited with 1")

        w.console.idle_sleep.options = IdleSleepOptions(
            clock=lambda: w.machine.now, lock_present=lambda: False, pmset=broken
        )
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "battery"


async def test_a_run_is_six_hours_after_the_last(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        start = w.machine.now
        first = await w.console.idle_sleep.tick()
        assert first is not None and await stamped(w) == start
        w.machine.now = start + SPACING_S - 1
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "spacing" and len(w.runs) == 1
        w.machine.now = start + SPACING_S
        assert await w.console.idle_sleep.tick() is not None
        assert len(w.runs) == 2


async def test_the_spacing_survives_a_restart(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        start = w.machine.now
        assert await w.console.idle_sleep.tick() is not None
    async with world(tmp_path) as again:
        again.machine.now = start + HOUR
        assert await again.console.idle_sleep.tick() is None
        assert again.console.idle_sleep.blocked == "spacing" and again.runs == []
        again.machine.now = start + SPACING_S + 1
        assert await again.console.idle_sleep.tick() is not None


async def test_a_run_that_fails_is_not_tried_again_for_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with world(tmp_path) as w:
        await w.note()

        async def broken(*args: Any, **kwargs: Any) -> SleepReport:
            raise RuntimeError("the index is locked")

        monkeypatch.setattr(w.console.memory, "sleep", broken)
        assert await w.console.idle_sleep.tick() is None
        assert await w.echoes() == []
        assert await stamped(w) == w.machine.now
        assert await w.console.idle_sleep.tick() is None
        assert w.console.idle_sleep.blocked == "spacing"


# ── The line ──


async def test_a_run_that_changed_nothing_says_nothing_in_the_chat(
    tmp_path: Path,
) -> None:
    async with world(tmp_path) as w:
        report = await w.console.idle_sleep.tick()
        assert report is not None and report.changes == []
        assert len(w.runs) == 1 and await w.echoes() == []
        assert await stamped(w) == w.machine.now


async def test_a_run_that_changed_notes_adds_one_line_to_the_main_chat(
    tmp_path: Path,
) -> None:
    async with world(tmp_path) as w:
        await w.note()
        report = await w.console.idle_sleep.tick()
        assert report is not None and len(report.changes) == 1
        (row,) = await w.echoes()
        assert row["echo"] == "sleep" and row["run_id"] == report.run_id
        assert (row["count"], row["variant"]) == (1, "rules")
        assert row["line"] == "While you were away, melo tidied 1 note"
        assert row["said"] == {"key": "front.echo.slept", "params": {"n": 1}}
        assert row["items"] == [{"change": "added", "text": "You live in Utrecht."}]
        assert "more" not in row
        assert await w.notes() == ["You live in Utrecht."]


async def test_the_line_lists_five_changes_cut_short_and_counts_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ops = ["O1", "O2", "O3", "O4", "RETIRE", "ADD", "UPDATE", "NEW"]
    long = "You keep a very long note.  " * 12

    async def fake(variant: str | None = None, **kwargs: Any) -> SleepReport:
        changes = [
            Change(op, [], 1, long if op == "O3" else f"Note {op}") for op in ops
        ]
        return SleepReport("run-long", "model", 0.0, 0.0, changes=changes)

    async with world(tmp_path) as w:
        monkeypatch.setattr(w.console.memory, "sleep", fake)
        await w.console.idle_sleep.tick()
        (row,) = await w.echoes()
        assert (row["count"], row["more"]) == (8, 8 - SHOWN)
        assert row["line"] == "While you were away, melo tidied 8 notes"
        assert [item["change"] for item in row["items"]] == [
            "merged", "updated", "added", "reworded", "retired",
        ]  # fmt: skip
        third = row["items"][2]["text"]
        assert TEXT_CHARS - 10 < len(third) <= TEXT_CHARS and third.endswith("…")
        assert "  " not in third


async def test_the_line_is_in_the_readers_language(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        await w.console.idle_sleep.tick()
        (row,) = await w.echoes()
        page = (await w.client.get(f"/api/sessions/{w.console.main_id}/turns")).json()
        assert page["echoes"][0]["line"] == "While you were away, melo tidied 1 note"
        zh = await w.client.get(
            f"/api/sessions/{w.console.main_id}/turns",
            headers={"X-LocalMelo-Language": "zh-Hans"},
        )
        assert zh.json()["echoes"][0]["line"] == "你不在的时候，melo 整理了 1 条笔记"
        assert row["said"]["key"] == "front.echo.slept"


# ── Undo ──


async def test_undo_puts_the_notes_back_in_the_default_mode(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        assert w.console.cfg.ui.mode != "developer"
        assert not w.console.options.dev_routes
        await w.note()
        await w.console.idle_sleep.tick()
        (row,) = await w.echoes()
        assert await w.notes() == ["You live in Utrecht."]

        tapped = await w.tap(row["id"])
        assert tapped.status_code == 200, tapped.text
        undone = tapped.json()["echo"]
        assert undone["echo"] == "undone" and undone["of"] == row["id"]
        assert undone["line"] == "Put back the 1 note melo tidied"
        assert undone["said"] == {
            "key": "front.echo.sleptBack",
            "params": {"n": 1, "total": 1},
        }
        assert await w.notes() == []
        # The line was answered: a second tap on it is refused.
        again = await w.tap(row["id"])
        assert (again.status_code, again.json()["error"]["code"]) == (
            409,
            "already_answered",
        )


async def test_undo_of_a_run_that_is_gone_says_so(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        report = await w.console.idle_sleep.tick()
        assert report is not None
        (row,) = await w.echoes()
        # Another way back (the developer route) took the run first.
        assert await w.console.memory.undo_sleep(report.run_id) is not None
        gone = await w.tap(row["id"])
        assert gone.status_code == 404
        error = gone.json()["error"]
        assert error["code"] == "not_found"
        assert error["message"] == "No such tidy-up run, or it was undone already."
        chinese = await w.tap(row["id"], **{"X-LocalMelo-Language": "zh-Hans"})
        assert chinese.json()["error"]["message"] == "没有这次整理，或者它已经撤销了。"
        assert [e["echo"] for e in await w.echoes()] == ["sleep"]


async def test_a_note_the_person_reworded_since_stays_and_the_line_says_so(
    tmp_path: Path,
) -> None:
    async with world(tmp_path) as w:
        await w.note()
        await w.console.idle_sleep.tick()
        (row,) = await w.echoes()
        (note,) = (await w.client.get("/api/memory/records")).json()["records"]
        edited = await w.client.patch(
            f"/api/memory/records/{note['id']}", json={"text": "You live in Delft."}
        )
        assert edited.status_code == 200
        undone = (await w.tap(row["id"])).json()["echo"]
        assert undone["line"] == (
            "Nothing was put back: the notes melo tidied have changed since."
        )
        assert await w.notes() == ["You live in Delft."]


async def test_only_undo_applies_to_the_line(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        await w.console.idle_sleep.tick()
        (row,) = await w.echoes()
        for do in ("apply", "start", "save", "revoke"):
            refused = await w.tap(row["id"], do)
            assert refused.status_code == 400, do
            assert refused.json()["error"]["code"] == "invalid_request"
        assert await w.notes() == ["You live in Utrecht."]


async def test_the_developer_routes_stay_developer_only(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        report = await w.console.idle_sleep.tick()
        assert report is not None
        for call in (
            w.client.post("/api/memory/sleep"),
            w.client.delete(f"/api/memory/sleep/{report.run_id}"),
        ):
            assert (await call).status_code == 404


# ── The loop ──


async def test_the_gateway_runs_it_by_itself_and_stops_it_on_close(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, poll_s=0.01, lock=True) as w:
        await w.note()
        w.machine.lock = False
        task = w.console.idle_sleep._task
        assert task is not None and not task.done()
        for _ in range(500):
            if await w.echoes():
                break
            await asyncio.sleep(0.01)
        (row,) = await w.echoes()
        assert row["echo"] == "sleep" and row["count"] == 1
        # It waits six hours from the run it made: the loop did not run it again.
        await asyncio.sleep(0.1)
        assert len(w.runs) == 1
    assert task.done()


async def test_with_the_setting_off_the_loop_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with world(tmp_path, sleep="off", poll_s=0.01) as w:
        touched: list[str] = []

        def spy() -> Any:
            touched.append("stores")
            raise AssertionError("the memory files are not opened while sleep is off")

        monkeypatch.setattr(w.console.memory, "stores", spy)
        await asyncio.sleep(0.15)
        assert touched == [] and w.console.idle_sleep.blocked == "off"
