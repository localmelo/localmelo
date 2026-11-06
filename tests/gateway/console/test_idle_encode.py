"""Memory items from the chats, written on their own while the computer is idle (#428).

With ``recall_source = "items"`` and a local chat model, the gateway encodes the chats
into items a window at a time, oldest chat first, and stops when a gate says the person
is back. These tests reuse the idle sleep's fake machine (clock, lock and power) and its
console; the chats are written straight into the console's store and a stub chat model
answers the encoder. No model server.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.backends.openai_family import CLOUD_BACKENDS
from localmelo.support.gateway.console.idle_encode import MAX_FAILURES
from localmelo.support.gateway.console.idle_gates import IdleOptions
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.memory.service import MemoryTools
from localmelo.support.sessions.store import NewSession
from localmelo.tests.memory.hippo_helpers import StubEncoder, item, says
from localmelo.tests.memory.test_hippo_recall import refs_of
from localmelo.tests.memory.test_waiting_turns import PAD, WINDOW, reply

from .conftest import HOLDER, seed_turn
from .test_idle_sleep import AC, BATTERY, HOUR, World, world

# The first turn of a chat written by :func:`chat` ended this long before the real clock.
LONG_AGO = 30 * HOUR
LOGGER = "localmelo.support.gateway.console.idle_encode"


async def chat(
    w: World,
    exchanges: list[tuple[str, str]],
    *,
    at: float | None = None,
    title: str = "Chat",
    last: str = "completed",
) -> str:
    """A chat in the console's store: one turn for each of *exchanges* (the last one
    ends as *last* says; ``running`` leaves it open), a minute apart from *at*."""
    store = w.console.store
    folder = w.tmp_path / f"ws{time.monotonic_ns()}"
    folder.mkdir()
    start = at if at is not None else time.time() - LONG_AGO
    session_id: str | None = None
    for n, (typed, answer) in enumerate(exchanges):
        status = last if n == len(exchanges) - 1 else "completed"
        info = await store.begin_turn(
            session_id,
            typed,
            holder=HOLDER,
            title=title,
            new_session=None if session_id else NewSession(workspace=str(folder)),
        )
        session_id = info.session_id
        if status != "running":
            await store.finish_turn(
                session_id,
                info.turn,
                holder=HOLDER,
                status=status,
                cause="run",
                result=None,
                answer=answer,
            )
        async with store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE turns SET started_at = ?, finished_at = ? "
                "WHERE session_id = ? AND turn = ?",
                (start + 60 * n, None if status == "running" else start + 60 * n + 5,
                 session_id, info.turn),
            )  # fmt: skip
    assert session_id is not None
    return session_id


def padded(count: int) -> list[tuple[str, str]]:
    """*count* exchanges of a size that makes two of them fill a window."""
    return [(f"turn {n}", PAD) for n in range(1, count + 1)]


def arm(w: World, answers: Any = reply, *, poll_s: float = 1e9) -> StubEncoder:
    """Put the encoder to work in *w*: items as the recall source, a small window, a
    stub chat model, and the idle sleep's fake machine for the gates."""
    memory = w.console.cfg.memory
    memory.recall_source = "items"
    memory.compress_at_chars = WINDOW
    stub = StubEncoder(answers)
    w.console.memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    w.console.idle_encode.options = IdleOptions(
        clock=lambda: w.machine.now,
        poll_s=poll_s,
        lock_present=lambda: w.machine.lock,
        pmset=lambda: w.machine.power,
    )
    return stub


@dataclass
class Windows:
    """The calls ``compress_chat`` got: the chat and the turn numbers of each; *after*
    runs once each call has ended, with the number of calls so far."""

    calls: list[tuple[str, list[int]]] = field(default_factory=list)
    after: Callable[[int], Awaitable[None]] | None = None


def watch(w: World) -> Windows:
    seen = Windows()
    original = w.console.memory.compress_chat

    async def spy(session_id: str, title: str, turns: Any, **kwargs: Any) -> Any:
        seen.calls.append((session_id, [t.turn for t in turns]))
        out = await original(session_id, title, turns, **kwargs)
        if seen.after is not None:
            await seen.after(len(seen.calls))
        return out

    w.console.memory.compress_chat = spy  # type: ignore[method-assign]
    return seen


async def mark(w: World, session_id: str) -> int | None:
    return await w.console.memory.stores()[1].encoded_turn(session_id)


def lock_after_each_window(
    w: World, seen: Windows, *, back_in: float | None = None
) -> None:
    """Take the GPU lock once a window ends, so the run stops there; with *back_in*,
    give it back that many seconds later."""

    async def after(_: int) -> None:
        w.machine.lock = True
        if back_in is not None:
            asyncio.get_running_loop().call_later(
                back_in, lambda: setattr(w.machine, "lock", False)
            )

    seen.after = after


# ── It does nothing unless asked ──


async def test_with_the_default_source_nothing_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with world(tmp_path, sleep="off") as w:
        await chat(w, padded(1))
        stub = arm(w, poll_s=0.01)
        w.console.cfg.memory.recall_source = "turns"
        touched: list[str] = []

        def spy() -> Any:
            touched.append("stores")
            raise AssertionError("the memory files are not opened with turns recall")

        monkeypatch.setattr(w.console.memory, "stores", spy)
        assert await w.console.idle_encode.tick() is None
        assert w.console.idle_encode.blocked == "off"
        # The loop reads nothing either.
        await w.console.idle_encode.aclose()
        w.console.idle_encode.start()
        await asyncio.sleep(0.15)
        assert touched == [] and stub.prompts == []


async def test_memory_across_chats_off_blocks_it(tmp_path: Path) -> None:
    async with world(tmp_path, sleep="off") as w:
        await chat(w, padded(1))
        stub = arm(w)
        w.console.cfg.memory.across_chats = False
        assert await w.console.idle_encode.tick() is None
        assert w.console.idle_encode.blocked == "off" and stub.prompts == []


async def test_a_hosted_chat_model_is_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with world(tmp_path, sleep="off") as w:
        await chat(w, padded(1))
        stub = arm(w)
        w.console.cfg.chat_backend = CLOUD_BACKENDS[0]
        monkeypatch.setattr(
            w.console.memory,
            "stores",
            lambda: (_ for _ in ()).throw(AssertionError("the memory is not read")),
        )
        assert await w.console.idle_encode.tick() is None
        assert w.console.idle_encode.blocked == "hosted" and stub.prompts == []


async def test_without_a_chat_model_it_waits_and_counts_no_failure(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(1))
        arm(w)
        w.console.memory.chat_provider = lambda: None  # type: ignore[method-assign]
        assert await w.console.idle_encode.tick() is None
        assert w.console.idle_encode.blocked == "model"
        assert w.console.idle_encode.failures == {} and await mark(w, sid) is None


async def close_gate(w: World, gate: str) -> Callable[[], Awaitable[None]]:
    """Make the idle gate *gate* say no; the call that opens it again."""
    if gate == "lock":
        w.machine.lock = True

        async def reopen() -> None:
            w.machine.lock = False

    elif gate == "battery":
        w.machine.power = BATTERY

        async def reopen() -> None:
            w.machine.power = AC

    elif gate == "active":
        # A chat had a turn 19 minutes ago, by the machine's clock.
        w.machine.now = time.time() + 19 * 60

        async def reopen() -> None:
            w.machine.now = time.time() + 21 * 60

    else:
        folder = w.tmp_path / "tasks"
        folder.mkdir()
        spec = TaskSpec.from_payload(
            {"objective": "tidy the photos", "title": "Photos"}
        )
        record, _ = await w.console.workers.tasks.create_task(
            spec, origin="user", workspace=str(folder), front_id=w.console.main_id
        )
        queue = "UPDATE tasks SET state = ?, updated_at = ? WHERE task_id = ?"

        async def put(state: str) -> None:
            async with w.console.store.component_transaction("test_state") as conn:
                await conn.execute(queue, (state, time.time(), record.task_id))

        await put("queued")

        async def reopen() -> None:
            await put("done")

    return reopen


@pytest.mark.parametrize("gate", ["lock", "active", "tasks", "battery"])
async def test_each_idle_gate_stops_it(tmp_path: Path, gate: str) -> None:
    async with world(tmp_path, sleep="off") as w:
        await chat(w, padded(1))
        stub = arm(w)
        seen = watch(w)
        reopen = await close_gate(w, gate)
        assert await w.console.idle_encode.tick() is None
        assert w.console.idle_encode.blocked == gate
        assert seen.calls == [] and stub.prompts == []
        await reopen()
        run = await w.console.idle_encode.tick()
        assert run is not None and run.windows == 1 and len(seen.calls) == 1


# ── One window at a time ──


async def test_one_window_for_each_call_and_the_mark_follows_it(tmp_path: Path) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(5))
        arm(w)
        seen = watch(w)
        lock_after_each_window(w, seen)
        run = await w.console.idle_encode.tick()
        assert run is not None and (run.windows, run.stopped) == (1, "lock")
        assert seen.calls == [(sid, [1, 2])] and await mark(w, sid) == 2
        w.machine.lock, seen.after = False, None
        run = await w.console.idle_encode.tick()
        assert run is not None and (run.windows, run.stopped) == (2, "empty")
        assert seen.calls[1:] == [(sid, [3, 4]), (sid, [5])]
        assert await mark(w, sid) == 5
        # Nothing waits: the next look makes no call.
        run = await w.console.idle_encode.tick()
        assert run is not None and run.windows == 0 and len(seen.calls) == 3


async def test_no_turn_at_or_before_the_mark_or_below_one_is_passed(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(5))
        arm(w)
        seen = watch(w)
        await w.console.memory.stores()[1].mark_encoded(sid, 3)
        run = await w.console.idle_encode.tick()
        assert run is not None and run.windows == 1
        assert seen.calls == [(sid, [4, 5])]
        assert all(n >= 1 for _, turns in seen.calls for n in turns)
        assert await mark(w, sid) == 5


async def test_a_turn_that_starts_between_two_windows_stops_the_run(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(5))
        arm(w)
        seen = watch(w)

        async def someone_comes_back(n: int) -> None:
            folder = w.tmp_path / "back"
            folder.mkdir()
            await seed_turn(w.console.store, str(folder), status="running")

        seen.after = someone_comes_back
        run = await w.console.idle_encode.tick()
        assert run is not None and (run.windows, run.stopped) == (1, "active")
        assert seen.calls == [(sid, [1, 2])] and await mark(w, sid) == 2


# ── Which chat, which turns ──


async def test_the_oldest_chat_goes_first_and_a_cancelled_turn_is_skipped(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        now = time.time() - LONG_AGO
        # Written first, but its turns are the newest; the older chat is made last.
        newer = await chat(w, padded(1), at=now + 5 * HOUR, title="Newer")
        broken = await chat(
            w, [("turn 1", "Fine."), ("turn 2", ""), ("turn 3", "Kept.")], at=now + HOUR
        )
        older = await chat(w, padded(1), at=now, title="Older")
        # The middle turn of the second chat was cancelled: it is not read, and nor
        # does it hold up the turn after it.
        async with w.console.store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE turns SET status = 'cancelled' WHERE session_id = ? AND turn = 2",
                (broken,),
            )
        arm(w)
        seen = watch(w)
        run = await w.console.idle_encode.tick()
        assert run is not None and run.stopped == "empty"
        assert [sid for sid, _ in seen.calls] == [older, broken, newer]
        assert seen.calls[1] == (broken, [1, 3])
        assert await mark(w, broken) == 3
        assert await w.console.memory.waiting_turns(w.console.store, broken) == []


async def test_a_turn_that_runs_is_not_read_and_nor_is_anything_after_it(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, [("turn 1", "Fine."), ("turn 2", "")], last="running")
        waiting = await w.console.memory.waiting_turns(w.console.store, sid)
        assert [t.turn for t in waiting] == [1]


# ── A forget while a window is with the model ──


async def forget_during_a_window(
    w: World, forget: Callable[[], Awaitable[Any]]
) -> tuple[Any, StubEncoder]:
    """Run the encoder, and do *forget* once its window is with the model; the run."""
    stub = arm(w)
    original = w.console.memory.encode_chat
    reading, release = asyncio.Event(), asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> Any:
        found = await original(*args, **kwargs)
        reading.set()
        await release.wait()
        return found

    w.console.memory.encode_chat = slow  # type: ignore[method-assign]
    encoding = asyncio.create_task(w.console.idle_encode.tick())
    await asyncio.wait_for(reading.wait(), 5)
    await forget()
    release.set()
    return await asyncio.wait_for(encoding, 5), stub


async def test_forgetting_everything_during_a_window_keeps_it_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(1))
        with caplog.at_level(logging.INFO, logger=LOGGER):
            run, stub = await forget_during_a_window(w, w.console.memory.forget_all)
        assert run is not None and run.stopped == "forgotten"
        assert (run.windows, run.failures, run.items) == (0, 0, 0)
        assert w.console.idle_encode.failures == {}
        assert await w.console.memory.stores()[1].count_items() == 0
        assert await mark(w, sid) is None
        assert any("stopped: forgotten" in r.getMessage() for r in caplog.records)
        # The turns the user forgot are not read again, and later ones are.
        asked = len(stub.prompts)
        again = await w.console.idle_encode.tick()
        assert again is not None and (again.windows, again.stopped) == (0, "empty")
        assert len(stub.prompts) == asked


async def test_deleting_the_chat_during_a_window_keeps_it_out(tmp_path: Path) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(2))
        other = await chat(
            w, padded(1), at=time.time() - LONG_AGO + HOUR, title="Other"
        )

        async def delete() -> None:
            gone = await w.client.delete(f"/api/sessions/{sid}")
            assert gone.status_code == 204, gone.text

        run, _ = await forget_during_a_window(w, delete)
        assert run is not None and run.stopped == "forgotten"
        assert (run.windows, run.failures) == (0, 0)
        assert w.console.idle_encode.failures == {}
        index = w.console.memory.stores()[1]
        assert await index.count_items() == 0 and await mark(w, sid) is None
        # The chat after it is encoded at the next look.
        again = await w.console.idle_encode.tick()
        assert again is not None and again.windows == 1
        assert await mark(w, other) == 1 and await index.count_items() == 1


# ── Failures ──


@pytest.mark.parametrize("how", ["raises", "fails"])
async def test_three_failures_skip_a_chat_and_the_others_go_on(
    tmp_path: Path, how: str
) -> None:
    async with world(tmp_path, sleep="off") as w:
        now = time.time() - LONG_AGO
        bad = await chat(w, padded(1), at=now, title="Bad")
        good = await chat(w, padded(1), at=now + HOUR, title="Good")
        # Either the call raises, or the model's reply for the chat is no use.
        stub = arm(w, lambda prompt: "RAISE" if "Bad" in prompt else reply(prompt))
        if how == "raises":
            original = w.console.memory.compress_chat

            async def broken(session_id: str, *args: Any, **kwargs: Any) -> Any:
                if session_id == bad:
                    raise RuntimeError("the model server is down")
                return await original(session_id, *args, **kwargs)

            w.console.memory.compress_chat = broken  # type: ignore[method-assign]
        encoder = w.console.idle_encode
        for count in range(1, MAX_FAILURES + 1):
            run = await encoder.tick()
            assert run is not None and (run.failures, run.stopped) == (1, "failed")
            assert encoder.failures[bad] == count and run.windows == 0
        before = len(stub.prompts)
        # The next look skips that chat and encodes the one after it.
        run = await encoder.tick()
        assert run is not None and (run.windows, run.failures) == (1, 0)
        assert await mark(w, bad) is None and await mark(w, good) == 1
        assert len(stub.prompts) == before + 1, "the skipped chat is not asked again"
        waiting = await w.console.memory.waiting_turns(w.console.store, bad)
        assert [t.turn for t in waiting] == [1], "its turns stay recalled"


async def test_a_success_resets_the_count(tmp_path: Path) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(1))
        down = [True]
        arm(w, lambda prompt: "RAISE" if down[0] else reply(prompt))
        encoder = w.console.idle_encode
        for _ in range(MAX_FAILURES - 1):
            await encoder.tick()
        assert encoder.failures == {sid: MAX_FAILURES - 1}
        down[0] = False
        run = await encoder.tick()
        assert run is not None and run.windows == 1 and encoder.failures == {}
        # A chat that fails again counts from one, not from where it was.
        down[0] = True
        info = await w.console.store.begin_turn(sid, "turn 2", holder=HOLDER)
        await w.console.store.finish_turn(
            sid, info.turn, holder=HOLDER, status="completed", cause="run",
            result=None, answer=PAD,
        )  # fmt: skip
        await encoder.tick()
        assert encoder.failures == {sid: 1}


# ── What it writes and logs ──


async def test_nothing_is_written_to_the_cache_file_and_the_log_has_no_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, padded(3))
        arm(w)
        mine = tmp_path / "mine.jsonl"
        w.console.cfg.memory.compress_cache = str(mine)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            run = await w.console.idle_encode.tick()
        assert run is not None and (run.chats, run.windows, run.items) == ({sid}, 2, 3)
        assert not mine.exists()
        assert not (tmp_path / "memory" / "compress_cache.jsonl").exists()
        assert not list(tmp_path.rglob("*.jsonl"))
        (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
        assert line == (
            "Idle encode: 1 chats, 2 windows, 3 items stored, 0 refused, "
            "0 failures, stopped: empty"
        )


async def test_an_item_that_holds_a_secret_is_refused_and_counted(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        sid = await chat(w, [("What time is it?", PAD)])
        leak = says([item("Card", "The user's password is hunter2.", "profile", [0])])
        arm(w, lambda _: leak)
        run = await w.console.idle_encode.tick()
        assert run is not None and (run.windows, run.refused, run.items) == (1, 1, 0)
        assert await w.console.memory.stores()[1].count_items() == 0
        assert await mark(w, sid) == 1


# ── Taking turns with the sleep ──


async def test_the_encoder_and_the_sleep_never_run_at_once(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        arm(w)
        started, release = asyncio.Event(), asyncio.Event()
        original = w.console.memory.compress_chat

        async def slow(*args: Any, **kwargs: Any) -> Any:
            started.set()
            await release.wait()
            return await original(*args, **kwargs)

        w.console.memory.compress_chat = slow  # type: ignore[method-assign]
        encoding = asyncio.create_task(w.console.idle_encode.tick())
        await asyncio.wait_for(started.wait(), 5)
        sleeping = asyncio.create_task(w.console.idle_sleep.tick())
        await asyncio.sleep(0.1)
        assert not sleeping.done() and w.runs == [], "the sleep waits for the encoder"
        release.set()
        assert await asyncio.wait_for(encoding, 5) is not None
        assert await asyncio.wait_for(sleeping, 5) is not None and w.runs == [None]


async def test_the_sleep_in_progress_holds_the_encoder_back(tmp_path: Path) -> None:
    async with world(tmp_path) as w:
        await w.note()
        stub = arm(w)
        started, release = asyncio.Event(), asyncio.Event()
        original = w.console.memory.sleep

        async def slow(variant: str | None = None, **kwargs: Any) -> Any:
            started.set()
            await release.wait()
            return await original(variant, **kwargs)

        w.console.memory.sleep = slow  # type: ignore[method-assign]
        sleeping = asyncio.create_task(w.console.idle_sleep.tick())
        await asyncio.wait_for(started.wait(), 5)
        encoding = asyncio.create_task(w.console.idle_encode.tick())
        await asyncio.sleep(0.1)
        assert not encoding.done() and stub.prompts == [], "the encoder waits"
        release.set()
        assert await asyncio.wait_for(sleeping, 5) is not None
        run = await asyncio.wait_for(encoding, 5)
        assert run is not None and run.windows == 1


# ── End to end ──

BEES = "The user keeps bees on a rooftop."
MARATHON = "The marathon route crosses the old bridge."


def facts(prompt: str) -> str:
    """The encoder's answer: one item for each exchange that states a fact."""
    found = []
    for n, words in re.findall(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE):
        if "bees" in words:
            found.append(item("Bees", BEES, "profile", [int(n)]))
        if "marathon" in words:
            found.append(item("Marathon route", MARATHON, "episodic", [int(n)]))
    return says(found)


async def found(w: World, query: str) -> list[str]:
    """The ids of the memory lines a search for *query* gives: ``m`` items, ``t`` turns."""
    return refs_of(await MemoryTools(w.console.memory).search(query=query))


async def test_chats_are_encoded_over_several_looks_and_a_fact_is_found_throughout(
    tmp_path: Path,
) -> None:
    async with world(tmp_path, sleep="off") as w:
        now = time.time() - LONG_AGO
        bees = await chat(
            w,
            [
                ("I keep bees on a rooftop.", "Bees like rooftops."),
                ("Any tips?", "Smoke."),
            ],
            at=now,
            title="Bees",
        )
        run_chat = await chat(
            w,
            [("The marathon route crosses the old bridge.", "A fine route.")],
            at=now + HOUR,
            title="Run",
        )
        # The turns are in the index as the live index holds them; no item yet.
        await w.console.memory.reindex(w.console.store)
        arm(w, facts)
        w.console.cfg.memory.recall_form = "list"
        seen = watch(w)
        lock_after_each_window(w, seen)
        before = await found(w, "bees on a rooftop")
        assert before and all(ref.startswith("t") for ref in before), "found as a turn"
        assert await found(w, "marathon route old bridge")

        # First look: the older chat is encoded, the newer one is still turns.
        run = await w.console.idle_encode.tick()
        assert run is not None and (run.windows, run.stopped) == (1, "lock")
        assert await mark(w, bees) == 2 and await mark(w, run_chat) is None
        first = await found(w, "bees on a rooftop")
        assert first and all(ref.startswith("m") for ref in first), "found as an item"
        assert all(
            ref.startswith("t") for ref in await found(w, "marathon route old bridge")
        )

        # Second look: the other chat follows.
        w.machine.lock = False
        run = await w.console.idle_encode.tick()
        assert run is not None and run.windows == 1
        assert await mark(w, run_chat) == 1
        for query in ("bees on a rooftop", "marathon route old bridge"):
            refs = await found(w, query)
            assert refs and all(ref.startswith("m") for ref in refs), query
        assert [sid for sid, _ in seen.calls] == [bees, run_chat]


async def test_the_loop_encodes_by_itself_and_stops_on_close(tmp_path: Path) -> None:
    async with world(tmp_path, sleep="off") as w:
        now = time.time() - LONG_AGO
        first = await chat(w, padded(1), at=now)
        second = await chat(w, padded(1), at=now + HOUR, title="Second")
        await w.console.idle_encode.aclose()
        arm(w, poll_s=0.01)
        seen = watch(w)
        # The lock comes back after each window, so the second chat needs another look.
        lock_after_each_window(w, seen, back_in=0.05)
        w.console.idle_encode.start()
        task = w.console.idle_encode._task
        assert task is not None and not task.done()
        for _ in range(500):
            if await mark(w, second) is not None:
                break
            await asyncio.sleep(0.01)
        assert await mark(w, first) == 1 and await mark(w, second) == 1
        assert [sid for sid, _ in seen.calls] == [first, second]
        await w.console.idle_encode.aclose()
        assert task.done() and len(seen.calls) == 2
