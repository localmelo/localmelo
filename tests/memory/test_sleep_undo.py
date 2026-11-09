"""Sleep: an undo sticks, a stopped run can still be undone, the GPU lock is honoured.

Synthetic notes and turns on a temporary memory folder, a stub embedding model and
temporary lock files; no server, and never the real GPU lock. An undo keeps the
pairs it separated apart (``kept_apart``), and a note it removed is not written
again from the same turn. A run that is cancelled saves what it changed. Every
request to the embedding model first checks the lock (``[memory] sleep_lock``,
else ``[scheduler] gpu_lock_file``) and whether the user is active. An undo log
an earlier version saved, with the changes of its model pass, still undoes.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, records, sleep
from localmelo.support.memory.index import ConversationIndex
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

from . import test_sleep as rules
from .test_sleep_checks import AT, C3, C4, TRAIN, pair, states, tidy

memory = rules.memory
chats = rules.chats


def saved_runs(memory: ChatMemory) -> list[str]:
    """The ids of the runs whose undo log is kept."""
    with sqlite3.connect(Path(memory.directory) / "index.db") as conn:
        return [str(row[0]) for row in conn.execute("SELECT run_id FROM sleep_runs")]


def reset_watermark(memory: ChatMemory) -> None:
    """As if a cut-short run had not moved the watermark: its turns come back."""
    with sqlite3.connect(Path(memory.directory) / "index.db") as conn:
        conn.execute("DELETE FROM meta WHERE key = 'sleep_until'")


async def apart(memory: ChatMemory) -> dict[int, list[int]]:
    """Each active note that lists notes an undo kept apart from it."""
    return {
        n.id: n.provenance["kept_apart"]
        for n in await memory.records()
        if n.provenance.get("kept_apart")
    }


@pytest.fixture()
async def watched(tmp_path: Path) -> AsyncIterator[tuple[ChatMemory, Path]]:
    """A memory with the shared GPU lock at a temporary path (``[scheduler]
    gpu_lock_file``) and no ``[memory] sleep_lock``; the lock does not exist yet."""
    lock = tmp_path / "gpu-lock"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory.sleep = "rules"
    cfg.scheduler.gpu_lock_file = str(lock)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        yield memory, lock
    finally:
        await memory.aclose()


FOREIGN = "holder: bench:7\npurpose: a benchmark\nrun: 7\n"
BUS = "You take the 8:15 bus to work."
SAID_TRAIN = "My usual train to the office is the 7:40."
SAID_BUS = "My usual bus to the office is the 8:15."


class Embedder(FakeEmbedding):
    """An embedding model whose vector is far from the notes' ``[1, 0]`` for every
    text (so no fact counts as held), and which runs ``after`` once it answered."""

    after: Callable[[], Awaitable[None]] | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        await super().embed(texts)  # keeps the call
        if self.after is not None:
            await self.after()
        return [[0.0, 1.0] for _ in texts]


async def embedder(memory: ChatMemory, monkeypatch: pytest.MonkeyPatch) -> Embedder:
    """Give *memory* the embedding model and two notes with vectors that a turn
    can restate: a run that reads such a turn embeds what it says."""
    provider = Embedder("m")
    monkeypatch.setattr(memory, "embedding", lambda: (provider, "m"))
    await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    await rules.put(memory, BUS, vector=[1.0, 0.0])
    return provider


# ── 1. An undo sticks ──


async def test_an_undone_merge_is_not_merged_again_by_the_next_run(
    memory: ChatMemory,
) -> None:
    older = await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO)
    newer = await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO + 5)
    first = await memory.sleep()
    (change,) = first.changes
    assert (change.op, change.before_ids, change.after_id) == ("O1", [older], newer)
    assert await memory.undo_sleep(first.run_id) is not None
    assert await apart(memory) == {older: [newer], newer: [older]}
    second = await memory.sleep()
    # Nothing to do: no change, no stop, and no undo log to keep.
    assert (second.changes, second.aborted, second.capped) == ([], "", False)
    assert saved_runs(memory) == []
    assert await states(memory) == {older: "active", newer: "active"}


async def test_rules_do_not_retire_an_undone_replacement_again(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    first = await tidy(memory)
    assert [c.op for c in first.changes] == ["O2b"]
    assert await memory.undo_sleep(first.run_id) is not None
    assert await apart(memory) == {older: [newer], newer: [older]}
    assert (await tidy(memory)).changes == []
    assert await states(memory) == {older: "active", newer: "active"}


async def test_a_third_newer_note_still_replaces_a_restored_note(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    first = await tidy(memory)
    assert await memory.undo_sleep(first.run_id) is not None
    third = await rules.put(
        memory,
        "Your new manager is Mia Costa from May.",
        AT + 40,
        session="t",
        quote="My new manager is Mia Costa from May.",
        topic="people",
    )
    report = await tidy(memory)
    # Only the pair stays apart: the third note takes the restored one (O2b) and
    # the note it was separated from (O1).
    assert sorted((c.op, c.before_ids[0], c.after_id) for c in report.changes) == [
        ("O1", newer, third),
        ("O2b", older, third),
    ]
    assert await states(memory) == {
        older: "superseded",
        newer: "superseded",
        third: "active",
    }


async def test_a_user_edit_ends_what_an_undo_kept_apart(memory: ChatMemory) -> None:
    older = await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO)
    newer = await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO + 5)
    first = await memory.sleep()
    await memory.undo_sleep(first.run_id)
    assert (await memory.sleep()).changes == []
    assert await memory.edit_record(newer, "You live in Utrecht.") is not None
    # Neither list is left: the user changed it, so the old reason is over.
    assert await apart(memory) == {}
    again = await memory.sleep()
    assert [(c.op, c.before_ids, c.after_id) for c in again.changes] == [
        ("O1", [older], newer)
    ]


async def test_a_user_edit_of_a_retired_note_clears_its_own_list(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    await memory.undo_sleep((await tidy(memory)).run_id)
    await memory.stores()[0].set_provenance(older, {"kept_apart": [older, newer]})
    await memory.edit_record(older, "Your manager is Priya Shah.")
    assert await apart(memory) == {}


async def test_sleep_never_removes_what_an_undo_kept_apart(memory: ChatMemory) -> None:
    older = await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO)
    newer = await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO + 5)
    await memory.undo_sleep((await memory.sleep()).run_id)
    for _ in range(2):
        await memory.sleep()
        assert await apart(memory) == {older: [newer], newer: [older]}


# ── An undone written note is not written again ──


async def test_a_note_an_undo_removed_is_not_written_again_even_when_its_turn_returns(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await rules.chat(chats, "I live in Utrecht.")
    first = await memory.sleep(chats=chats)
    assert [c.op for c in first.changes] == ["O3"]
    assert await memory.undo_sleep(first.run_id) is not None
    index = memory.stores()[1]
    assert await index.meta("sleep_until") > 0
    # The watermark stops the turn from being read again ...
    assert (await memory.sleep(chats=chats)).changes == []
    # ... and when it is read again (it was never marked), the note is not rewritten.
    reset_watermark(memory)
    assert await index.meta("sleep_until") == 0
    assert (await memory.sleep(chats=chats)).changes == []
    assert await memory.records() == []


async def test_a_capped_run_that_was_undone_does_not_write_the_same_notes_again(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sleep, "MAX_CHANGES", 2)
    for who in ("advisor", "dentist", "coach"):
        await rules.chat(chats, f"My {who} is Sam.")
    first = await memory.sleep(chats=chats)
    assert first.capped and len(first.changes) == 2
    assert await memory.stores()[1].meta("sleep_until") == 0  # the cursor stays
    assert await memory.undo_sleep(first.run_id) is not None
    second = await memory.sleep(chats=chats)
    # The two undone notes (newest first: coach, dentist) are not written again.
    assert [c.text for c in second.changes] == ["Your advisor is Sam."]
    assert await rules.texts(memory) == {"Your advisor is Sam."}


async def test_the_same_words_in_a_new_turn_are_new_evidence(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await rules.chat(chats, "I live in Utrecht.")
    await memory.undo_sleep((await memory.sleep(chats=chats)).run_id)
    await rules.chat(chats, "I live in Utrecht.")
    again = await memory.sleep(chats=chats)
    assert [c.text for c in again.changes] == ["You live in Utrecht."]


async def test_forgetting_the_chat_forgets_what_was_undone(
    memory: ChatMemory, chats: SessionStore
) -> None:
    said = await rules.chat(chats, "I live in Utrecht.")
    await memory.undo_sleep((await memory.sleep(chats=chats)).run_id)
    await memory.forget_session(said.session_id)
    reset_watermark(memory)
    assert [c.text for c in (await memory.sleep(chats=chats)).changes] == [
        "You live in Utrecht."
    ]


# ── An undo log an earlier version saved ──


async def test_an_undo_log_with_the_changes_of_the_removed_model_pass_still_undoes(
    memory: ChatMemory,
) -> None:
    """Notes as an earlier model pass left them, and the log it saved: ``UPDATE``
    (a note superseded by a new one), ``RETIRE``, ``ADD`` and ``O4`` (a pronoun
    reworded). The log loads and undoes, and is gone afterwards."""
    store, index = memory.stores()
    gym = await rules.put(memory, "Your gym day is Monday.", AT)
    cashews = await rules.put(memory, "You are allergic to cashews.", AT)
    pronoun = await rules.put(memory, "Her office is on Pine Street.", AT)
    baseline = {str(note): 0 for note in (gym, cashews, pronoun)}
    moved = await rules.put(memory, "Your gym day moved to Friday.", AT + 20)
    kite = await rules.put(memory, "You enjoy kitesurfing.", AT + 20)
    ada = await rules.put(memory, "Dr. Ada's office is on Pine Street.", AT + 20)
    for old, keeper in ((gym, moved), (pronoun, ada), (cashews, None)):
        assert await store.set_status(
            [old], "superseded", keeper=keeper, admitted_by="sleep", reason="run"
        )
    await index.save_run(
        "earlier",
        AT + 30,
        [
            {"op": "UPDATE", "before": [gym], "after": moved},
            {"op": "RETIRE", "before": [cashews], "after": None, "baseline": baseline},
            {"op": "ADD", "before": [], "after": kite},
            {"op": "O4", "before": [pronoun], "after": ada},
        ],
    )
    assert saved_runs(memory) == ["earlier"]
    undone = await memory.undo_sleep("earlier")
    assert undone == {"run_id": "earlier", "restored": 3, "removed": 3, "kept": 0}
    assert await rules.texts(memory) == {
        "Your gym day is Monday.",
        "You are allergic to cashews.",
        "Her office is on Pine Street.",
    }
    assert saved_runs(memory) == [] and await memory.undo_sleep("earlier") is None


async def test_an_index_with_the_removed_turn_table_opens_with_it_empty(
    tmp_path: Path,
) -> None:
    """The removed model pass kept turn ids in ``sleep_processed``: its rows go when
    the file opens, the table stays; a file without it opens as before."""
    old, new = str(tmp_path / "old.db"), str(tmp_path / "new.db")
    raw = sqlite3.connect(old)
    raw.executescript("""
        CREATE TABLE sleep_processed (
            session_id TEXT NOT NULL, turn INTEGER NOT NULL,
            note_id INTEGER NOT NULL DEFAULT 0, finished REAL NOT NULL,
            failures INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (session_id, turn, note_id));
        INSERT INTO sleep_processed (session_id, turn, finished) VALUES ('s1', 4, 1.0);
        """)
    raw.commit()
    raw.close()
    for path in (old, new):
        index = ConversationIndex(path)
        try:
            assert await index.count() == 0
        finally:
            await index.aclose()

    def tables(path: str, name: str) -> list[Any]:
        with sqlite3.connect(path) as conn:
            return conn.execute(
                "SELECT name FROM sqlite_master WHERE name = ?", (name,)
            ).fetchall()

    with sqlite3.connect(old) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sleep_processed").fetchone() == (0,)
    assert tables(old, "sleep_processed") and not tables(new, "sleep_processed")


# ── 2. A cancelled run keeps its undo log ──


def block_second_write(monkeypatch: pytest.MonkeyPatch, *, first: bool = False) -> Any:
    """Let ``records.write`` through once (never with *first*), then wait for
    ever, as a slow model call would; the event is set when it waits."""
    real, state = records.write, {"calls": 0}
    waiting = asyncio.Event()

    async def blocking(*args: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        if state["calls"] >= (1 if first else 2):
            waiting.set()
            await asyncio.Event().wait()
        return await real(*args, **kwargs)

    monkeypatch.setattr(records, "write", blocking)
    return waiting


async def test_a_cancelled_run_keeps_what_it_changed_and_it_can_be_undone(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rules.chat(chats, "I live in Utrecht.")
    await rules.chat(chats, "My dentist is Lee Park.")
    waiting = block_second_write(monkeypatch)
    task = asyncio.create_task(memory.sleep(chats=chats))
    await asyncio.wait_for(waiting.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (note,) = await memory.records()
    assert note.text == "Your dentist is Lee Park."
    run_id = note.provenance["run_id"]
    assert saved_runs(memory) == [run_id]
    # The turns it did not finish stay unread.
    assert await memory.stores()[1].meta("sleep_until") == 0
    undone = await memory.undo_sleep(run_id)
    assert undone == {"run_id": run_id, "restored": 0, "removed": 1, "kept": 0}
    assert await memory.records() == []


async def test_a_run_cancelled_before_any_change_saves_no_run(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rules.chat(chats, "I live in Utrecht.")
    waiting = block_second_write(monkeypatch, first=True)
    task = asyncio.create_task(memory.sleep(chats=chats))
    await asyncio.wait_for(waiting.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert saved_runs(memory) == [] and await memory.records() == []
    assert await memory.stores()[1].meta("sleep_until") == 0


async def test_recall_sees_the_turns_a_cancelled_run_replaced(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    older, newer = await pair(memory, *C3)
    await rules.chat(chats, "My dentist is Lee Park.")
    await memory._sync()  # recall has read the notes as they are
    assert memory._replaced(memory._corpus) == {}
    waiting = block_second_write(monkeypatch, first=True)
    task = asyncio.create_task(memory.sleep(chats=chats))
    await asyncio.wait_for(waiting.wait(), 10)  # the rule O2b is done by now
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (run_id,) = saved_runs(memory)
    await memory._sync()
    assert list(memory._replaced(memory._corpus)) == [("o", 1)]
    assert await memory.undo_sleep(run_id) is not None
    await memory._sync()
    assert memory._replaced(memory._corpus) == {}
    assert await states(memory) == {older: "active", newer: "active"}


async def test_a_second_cancel_while_the_log_is_saved_does_not_drop_it(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rules.chat(chats, "I live in Utrecht.")
    await rules.chat(chats, "My dentist is Lee Park.")
    waiting = block_second_write(monkeypatch)
    index = memory.stores()[1]
    real_save = index.save_run

    async def slow_save(*args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0.05)
        await real_save(*args, **kwargs)

    monkeypatch.setattr(index, "save_run", slow_save)
    task = asyncio.create_task(memory.sleep(chats=chats))
    await asyncio.wait_for(waiting.wait(), 10)
    task.cancel()
    await asyncio.sleep(0.01)  # the run is saving its log now
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    (note,) = await memory.records()
    assert saved_runs(memory) == [note.provenance["run_id"]]


# ── 3. The GPU lock is honoured ──


async def test_the_gpu_lock_is_the_default_sleep_lock_and_stops_a_run_before_its_first_embedding(
    watched: tuple[ChatMemory, Path],
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, lock = watched
    lock.write_text(FOREIGN)
    provider = await embedder(memory, monkeypatch)
    await rules.chat(chats, SAID_TRAIN)
    report = await memory.sleep("rules", chats=chats)
    assert report.aborted == "stopped: gpu_lock"
    assert not provider.calls and await memory.stores()[1].meta("sleep_until") == 0
    assert lock.read_text() == FOREIGN  # read only


async def test_a_lock_taken_mid_run_stops_it_before_the_next_embedding_and_it_is_saved(
    watched: tuple[ChatMemory, Path],
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, lock = watched
    await rules.put(memory, "You live in Lisbon.", rules.LONG_AGO)
    await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO + 5)
    provider = await embedder(memory, monkeypatch)
    before = {n.id for n in await memory.records()}
    await rules.chat(chats, SAID_BUS)
    await rules.chat(chats, SAID_TRAIN)  # the newest turn is read first

    async def a_benchmark_starts() -> None:
        lock.write_text(FOREIGN)

    provider.after = a_benchmark_starts
    report = await memory.sleep("rules", chats=chats)
    assert report.aborted == "stopped: gpu_lock" and len(provider.calls) == 1
    # What the run did before the lock was taken is kept, saved and undoable.
    assert [c.op for c in report.changes] == ["O1", "O3"]
    assert saved_runs(memory) == [report.run_id]
    assert await memory.stores()[1].meta("sleep_until") == 0
    assert await memory.undo_sleep(report.run_id) is not None
    assert {n.id for n in await memory.records()} == before


async def test_a_lock_that_is_gone_lets_the_next_run_go_on(
    watched: tuple[ChatMemory, Path],
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, lock = watched
    lock.write_text(FOREIGN)
    provider = await embedder(memory, monkeypatch)
    await rules.chat(chats, SAID_TRAIN)
    assert (await memory.sleep("rules", chats=chats)).aborted == "stopped: gpu_lock"
    assert not provider.calls
    lock.unlink()
    assert (await memory.sleep("rules", chats=chats)).aborted == ""
    assert len(provider.calls) == 1


async def test_a_run_that_embeds_nothing_needs_no_lock(
    watched: tuple[ChatMemory, Path], chats: SessionStore
) -> None:
    memory, lock = watched
    lock.write_text(FOREIGN)
    await rules.put(memory, "You live in Lisbon.", rules.LONG_AGO)
    await rules.put(memory, "You live in Utrecht.", rules.LONG_AGO + 5)
    await rules.chat(chats, "My dentist is Lee Park.")
    report = await memory.sleep("rules", chats=chats)  # no embedding model is set up
    assert [c.op for c in report.changes] == ["O1", "O3"]
    assert report.aborted == ""
    assert await memory.stores()[1].meta("sleep_until") > 0


async def test_a_lock_of_our_own_holder_does_not_stop_a_run(
    watched: tuple[ChatMemory, Path],
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, lock = watched
    lock.write_text("holder: melo-test\nrun: 3\n")
    memory._settings().scheduler.lock_self = "melo-test"  # type: ignore[union-attr]
    provider = await embedder(memory, monkeypatch)
    await rules.chat(chats, SAID_TRAIN)
    report = await memory.sleep("rules", chats=chats)
    assert report.aborted == "" and len(provider.calls) == 1
    lock.write_text(FOREIGN)
    await rules.chat(chats, SAID_BUS)
    assert (await memory.sleep("rules", chats=chats)).aborted == "stopped: gpu_lock"


async def test_an_explicit_sleep_lock_replaces_the_gpu_lock(
    watched: tuple[ChatMemory, Path],
    chats: SessionStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, gpu = watched
    gpu.write_text(FOREIGN)
    own = tmp_path / "sleep-lock"
    memory._settings().memory.sleep_lock = str(own)  # type: ignore[union-attr]
    provider = await embedder(memory, monkeypatch)
    await rules.chat(chats, SAID_TRAIN)
    assert (await memory.sleep("rules", chats=chats)).aborted == ""
    own.write_text("held")  # any file counts, whatever its words
    await rules.chat(chats, SAID_BUS)
    assert (await memory.sleep("rules", chats=chats)).aborted == "stopped: gpu_lock"
    assert len(provider.calls) == 1


LOCKS = [
    None,
    "",
    "   \n",
    "busy",
    "holder:\nrun: 1\n",
    "holder: bench\n",
    "holder: bench\nrun: r1\n",
    "holder: bench\r\nrun: r1\r\n",
    "holder: bench\nholder: other\n",
    "purpose: x\nholder:  bench  \n",
    "holder: bench:r1\n",
]


@pytest.mark.parametrize("text", LOCKS)
@pytest.mark.parametrize(
    "mine", ["", "bench", "bench:r1", "bench:r2", "other", "bench:"]
)
def test_the_lock_reader_agrees_with_the_schedulers_gate(
    tmp_path: Path, text: str | None, mine: str
) -> None:
    from localmelo.support.gateway.machine_monitor import classify_lock, read_lock_file

    path = tmp_path / "lock"
    if text is not None:
        path.write_text(text)
    gate = classify_lock(
        read_lock_file(path, now_wall=time.time(), self_holder=mine), mine
    )
    assert sleep.lock_held(str(path), mine) == (gate == "foreign")


def test_the_lock_reader_holds_for_what_is_no_plain_lock(tmp_path: Path) -> None:
    from localmelo.support.gateway.machine_monitor import classify_lock, read_lock_file

    big = tmp_path / "big"
    big.write_text("holder: bench\n" + "x" * 70_000)
    bad = tmp_path / "bad"
    bad.write_bytes(b"holder: \xff\xfe\n")
    for path in (big, bad, tmp_path):  # a directory too
        gate = classify_lock(
            read_lock_file(path, now_wall=time.time(), self_holder="bench"), "bench"
        )
        assert gate == "foreign" and sleep.lock_held(str(path), "bench")
    assert not sleep.lock_held(str(tmp_path / "missing"), "bench")
    assert not sleep.lock_held(str(tmp_path / "missing"))


# ── Low: the start time, and tasks ──


async def test_a_turn_that_began_before_the_runs_start_time_counts_as_activity(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    began = time.time()  # the caller took it before its idle checks
    await asyncio.sleep(0.01)
    await rules.chat(chats, SAID_TRAIN)  # started in the gap
    provider = await embedder(memory, monkeypatch)
    report = await memory.sleep("rules", chats=chats, began=began)
    assert report.aborted == "stopped: user_active" and not provider.calls


async def test_a_turn_before_the_run_is_not_activity_and_none_means_now(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rules.chat(chats, SAID_TRAIN)
    provider = await embedder(memory, monkeypatch)
    report = await memory.sleep("rules", chats=chats)  # began=None: at entry
    assert report.aborted == "" and len(provider.calls) == 1


async def test_a_task_that_starts_during_the_run_stops_it(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rules.chat(chats, SAID_BUS)
    await rules.chat(chats, SAID_TRAIN)
    provider = await embedder(memory, monkeypatch)

    async def a_task_starts() -> None:
        await chats.begin_turn(
            None,
            "Tidy the folder.",
            holder=rules.HOLDER,
            new_session=NewSession(kind="task"),
        )

    provider.after = a_task_starts
    report = await memory.sleep("rules", chats=chats)
    assert report.aborted == "stopped: user_active" and len(provider.calls) == 1


async def test_a_task_that_began_earlier_is_not_activity(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await chats.begin_turn(
        None,
        "Tidy the folder.",
        holder=rules.HOLDER,
        new_session=NewSession(kind="task"),
    )
    await rules.chat(chats, SAID_TRAIN)
    provider = await embedder(memory, monkeypatch)
    report = await memory.sleep("rules", chats=chats)
    assert report.aborted == "" and len(provider.calls) == 1
