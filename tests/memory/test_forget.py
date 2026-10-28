"""``forget(task_id)`` and ``forget_all()`` on every memory backend.

The in-memory stores answer the same two calls, so a caller sees one
behaviour whichever backend a :class:`Hippo` holds. For the SQLite stores
the tests also check what the brief asks for: a reopened file no longer
sees the rows, and the deleted text is not left in the database or its WAL
after the checkpoint that a forget runs.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from localmelo.melo.memory import History, LongTerm
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import LLMCallRecord, Message, StepRecord, TaskRecord


def _call(task_id: str, seq: int = 0, *, text: str = "") -> LLMCallRecord:
    return LLMCallRecord(
        task_id=task_id,
        seq=seq,
        kind="plan",
        step=0,
        attempt=0,
        outcome="answer",
        content=text,
        thinking=text,
        reasoning_key="",
        redactions=0,
        tool_calls=(),
        usage=None,
        duration_s=0.1,
        timestamp=1.0,
    )


def _count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _disk_bytes(path: Path) -> bytes:
    """The database file and its ``-wal`` and ``-shm`` files, as they are."""
    parts: list[bytes] = []
    for suffix in ("", "-wal", "-shm"):
        file = path.parent / (path.name + suffix)
        if file.exists():
            parts.append(file.read_bytes())
    return b"".join(parts)


# ── In-memory History ──


async def test_in_memory_history_forgets_a_task_with_its_steps() -> None:
    history = History()
    await history.save_task(TaskRecord(query="one", task_id="t1"))
    await history.add_step("t1", StepRecord(thought="first"))
    await history.save_task(TaskRecord(query="two", task_id="t2"))

    await history.forget("t1")

    assert await history.get_task("t1") is None
    assert await history.get_steps("t1") == []
    kept = await history.get_task("t2")
    assert kept is not None
    assert kept.query == "two"

    # An unknown id removes nothing and is no error.
    await history.forget("missing")
    assert await history.get_task("t2") is not None


async def test_in_memory_history_forget_all_empties_it() -> None:
    history = History()
    await history.save_task(TaskRecord(query="one", task_id="t1"))
    await history.save_task(TaskRecord(query="two", task_id="t2"))

    await history.forget_all()

    assert await history.get_task("t1") is None
    assert await history.get_task("t2") is None


# ── In-memory LongTerm ──


async def test_in_memory_long_term_forgets_entries_that_name_the_task() -> None:
    long = LongTerm()
    await long.add(
        "reflection one", [1.0, 0.0], {"task_id": "t1", "type": "reflection"}
    )
    await long.add("reflection two", [0.0, 1.0], {"task_id": "t2"})
    await long.add("step summary", [1.0, 1.0], {"step_id": "s1"})
    await long.add("no provenance", [0.5, 0.5])

    await long.forget("t1")

    texts = {text for text, _score, _meta in await long.search([1.0, 0.0], top_k=10)}
    assert texts == {"reflection two", "step summary", "no provenance"}

    await long.forget_all()
    assert await long.search([1.0, 0.0], top_k=10) == []


# ── SqliteHistory ──


async def test_sqlite_history_forget_removes_only_that_task(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    try:
        for task_id in ("t1", "t2"):
            await history.save_task(TaskRecord(query=task_id, task_id=task_id))
            await history.add_step(task_id, StepRecord(thought=f"{task_id}-step"))
            await history.add_llm_call(_call(task_id))
        await history.forget("t1")

        assert await history.count_tasks() == 1
        assert await history.count_steps() == 1
        assert await history.get_task("t1") is None
        assert await history.get_steps("t1") == []
        kept = await history.get_task("t2")
        assert kept is not None
        assert [step.thought for step in kept.steps] == ["t2-step"]
    finally:
        await history.aclose()

    assert _count(path, "tasks") == 1
    assert _count(path, "steps") == 1
    assert _count(path, "llm_calls") == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT task_id FROM llm_calls").fetchall() == [("t2",)]


async def test_a_reopened_history_does_not_see_a_forgotten_task(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    await history.save_task(TaskRecord(query="one", task_id="t1"))
    await history.add_step("t1", StepRecord(thought="step"))
    await history.add_llm_call(_call("t1"))
    await history.forget("t1")
    await history.aclose()

    reopened = SqliteHistory(path)
    try:
        assert await reopened.get_task("t1") is None
        assert await reopened.count_tasks() == 0
        assert await reopened.count_steps() == 0
    finally:
        await reopened.aclose()
    assert _count(path, "llm_calls") == 0


async def test_sqlite_history_forget_keeps_a_working_connection(tmp_path: Path) -> None:
    history = SqliteHistory(tmp_path / "history.db")
    try:
        await history.save_task(TaskRecord(query="one", task_id="t1"))
        await history.forget("t1")
        # The checkpoint leaves no transaction open and the next write works.
        assert history._conn is not None
        assert not history._conn.in_transaction
        await history.save_task(TaskRecord(query="after", task_id="t2"))
        assert await history.count_tasks() == 1
    finally:
        await history.aclose()


async def test_sqlite_history_forget_deletes_llm_calls_without_a_task_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    try:
        # llm_calls has no foreign key, so a row can outlive its task.
        await history.add_llm_call(_call("orphan"))
        await history.forget("orphan")
        assert await history.count_tasks() == 0
    finally:
        await history.aclose()
    assert _count(path, "llm_calls") == 0


async def test_sqlite_history_forget_all_empties_every_table(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    try:
        for task_id in ("t1", "t2"):
            await history.save_task(TaskRecord(query=task_id, task_id=task_id))
            await history.add_step(task_id, StepRecord(thought="step"))
            await history.add_llm_call(_call(task_id))
        await history.forget_all()

        assert await history.count_tasks() == 0
        assert await history.count_steps() == 0
    finally:
        await history.aclose()
    assert _count(path, "tasks") == 0
    assert _count(path, "steps") == 0
    assert _count(path, "llm_calls") == 0


async def test_history_forget_leaves_no_forgotten_text_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    sentinel = "SENTINEL_HISTORY_FORGOTTEN_7f3a"
    history = SqliteHistory(path)
    try:
        await history.save_task(TaskRecord(query=sentinel, task_id="t1"))
        await history.add_step("t1", StepRecord(thought=sentinel))
        await history.add_llm_call(_call("t1", text=sentinel))
        assert sentinel.encode() in _disk_bytes(path)

        await history.forget("t1")

        assert sentinel.encode() not in _disk_bytes(path)
        # The checkpoint ran: no WAL frame is left to hold an old copy.
        wal = path.parent / (path.name + "-wal")
        assert wal.exists()
        assert wal.stat().st_size == 0
    finally:
        await history.aclose()


async def test_a_busy_checkpoint_is_logged_but_the_forget_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    other = sqlite3.connect(path, isolation_level=None)
    try:
        await history.save_task(TaskRecord(query="one", task_id="t1"))
        assert history._conn is not None
        await history._conn.execute("PRAGMA busy_timeout=0")
        # A reader on another connection keeps TRUNCATE from resetting the
        # WAL; the rows are still deleted, only the old copies wait.
        other.execute("BEGIN")
        other.execute("SELECT COUNT(*) FROM tasks").fetchone()
        with caplog.at_level(
            logging.WARNING, logger="localmelo.melo.memory.history.sqlite"
        ):
            await history.forget("t1")
        assert await history.count_tasks() == 0
        assert "could not truncate the WAL" in caplog.text
    finally:
        other.execute("ROLLBACK")
        other.close()
        await history.aclose()


# ── SqliteLongTerm ──


async def test_sqlite_long_term_forget_matches_metadata_task_id_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    long = SqliteLongTerm(path)
    try:
        await long.add("reflection one", [1.0, 0.0], {"task_id": "t1"})
        await long.add("reflection two", [0.0, 1.0], {"task_id": "t2"})
        # A step summary names its step, not its task, even when the text
        # happens to contain the id: the metadata is what counts.
        await long.add("t1 step summary", [1.0, 1.0], {"step_id": "s1"})
        await long.forget("t1")

        results = await long.search([1.0, 0.0], top_k=10)
        assert {text for text, _score, _meta in results} == {
            "reflection two",
            "t1 step summary",
        }
    finally:
        await long.aclose()


async def test_a_reopened_long_term_does_not_see_forgotten_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    long = SqliteLongTerm(path)
    await long.add("reflection one", [1.0, 0.0], {"task_id": "t1"})
    await long.add("reflection two", [0.0, 1.0], {"task_id": "t2"})
    await long.forget("t1")
    await long.aclose()

    reopened = SqliteLongTerm(path)
    try:
        assert await reopened.count_entries() == 1
        results = await reopened.search([1.0, 0.0], top_k=10)
        assert {text for text, _score, _meta in results} == {"reflection two"}
    finally:
        await reopened.aclose()


async def test_sqlite_long_term_forget_all_empties_it(tmp_path: Path) -> None:
    path = tmp_path / "long_term.db"
    long = SqliteLongTerm(path)
    try:
        await long.add("reflection one", [1.0, 0.0], {"task_id": "t1"})
        await long.add("no provenance", [0.0, 1.0])
        await long.forget_all()
        assert await long.count_entries() == 0
        await long.forget("t1")  # still a no-op
        assert await long.count_entries() == 0
    finally:
        await long.aclose()


async def test_long_term_forget_all_leaves_no_forgotten_text_on_disk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    sentinel = "SENTINEL_LONG_TERM_FORGOTTEN_4c1d"
    long = SqliteLongTerm(path)
    try:
        await long.add(sentinel, [1.0, 0.0], {"task_id": "t1"})
        assert sentinel.encode() in _disk_bytes(path)

        await long.forget_all()

        assert sentinel.encode() not in _disk_bytes(path)
        wal = path.parent / (path.name + "-wal")
        assert wal.exists()
        assert wal.stat().st_size == 0
    finally:
        await long.aclose()


# ── Hippo facade ──


async def test_hippo_forget_forgets_history_and_long_term(tmp_path: Path) -> None:
    history = SqliteHistory(tmp_path / "history.db")
    long = SqliteLongTerm(tmp_path / "long_term.db")
    hippo = Hippo(embedding=None, history=history, long=long)
    try:
        await history.save_task(TaskRecord(query="one", task_id="t1"))
        await history.add_step("t1", StepRecord(thought="step"))
        await history.save_task(TaskRecord(query="two", task_id="t2"))
        await long.add("reflection one", [1.0, 0.0], {"task_id": "t1"})
        await long.add("reflection two", [0.0, 1.0], {"task_id": "t2"})
        hippo.working.append(Message(role="user", content="keep me"))

        await hippo.forget("t1")

        assert await history.get_task("t1") is None
        assert await history.count_steps() == 0
        assert await history.get_task("t2") is not None
        results = await long.search([1.0, 0.0], top_k=10)
        assert {text for text, _score, _meta in results} == {"reflection two"}
    finally:
        await hippo.aclose()


async def test_hippo_forget_all_forgets_both_stores_not_working_memory(
    tmp_path: Path,
) -> None:
    history = SqliteHistory(tmp_path / "history.db")
    long = SqliteLongTerm(tmp_path / "long_term.db")
    hippo = Hippo(embedding=None, history=history, long=long)
    try:
        await history.save_task(TaskRecord(query="one", task_id="t1"))
        await long.add("reflection one", [1.0, 0.0], {"task_id": "t1"})
        hippo.working.append(Message(role="user", content="keep me"))

        await hippo.forget_all()

        assert await history.count_tasks() == 0
        assert await long.count_entries() == 0
        assert [msg.content for msg in hippo.working.get_window()] == ["keep me"]
    finally:
        await hippo.aclose()
