"""History under several processes and a killed writer (S2's history slice).

Each child is ``sys.executable`` running a short script that writes through
``SqliteHistory``; the database lives under ``tmp_path``. The default size
runs in a few seconds.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from localmelo.melo.memory.history.sqlite import _SCHEMA

WRITERS = 4
ROWS_PER_WRITER = 20

# Waits for the "go" file, then writes ROWS llm_calls rows and as many steps
# rows for task WRITER, printing the seq of each call row once committed.
CHILD = textwrap.dedent("""
    import asyncio, os, sys, time
    from localmelo.melo.memory.history.sqlite import SqliteHistory
    from localmelo.melo.schema import LLMCallRecord, StepRecord, TaskRecord

    path, go, writer, rows = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

    async def main() -> None:
        while not os.path.exists(go):
            time.sleep(0.005)
        history = SqliteHistory(path)
        try:
            await history.save_task(TaskRecord(query="q", task_id=writer))
            seq = 0
            while rows < 0 or seq < rows:
                await history.add_step(writer, StepRecord(thought=f"{writer}-{seq}"))
                await history.add_llm_call(
                    LLMCallRecord(
                        task_id=writer, seq=seq, kind="plan", step=seq + 1,
                        attempt=1, outcome="tools", content=f"{writer}-{seq}",
                        thinking="t", reasoning_key="", redactions=0,
                        tool_calls=(), usage=None, duration_s=0.0,
                        timestamp=time.time(),
                    )
                )
                print(seq, flush=True)
                seq += 1
        finally:
            await history.aclose()

    asyncio.run(main())
    """)


def _child(path: Path, go: Path, writer: str, rows: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, str(path), str(go), writer, str(rows)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # A temporary HOME: nothing reaches the user's own files.
        env={**os.environ, "HOME": str(path.parent)},
    )


def _write_v0(path: Path) -> None:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT INTO tasks (task_id, query, status, result) "
            "VALUES ('old', 'q', 'completed', 'a')"
        )
        connection.execute(
            "INSERT INTO steps (step_id, task_id, thought, timestamp, seq) "
            "VALUES ('old-step', 'old', 'kept', 1.0, 0)"
        )
        connection.commit()


def _check(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    return connection


def test_four_processes_migrate_and_write_one_file(tmp_path: Path) -> None:
    path, go = tmp_path / "history.db", tmp_path / "go"
    _write_v0(path)
    children = [_child(path, go, f"w{i}", ROWS_PER_WRITER) for i in range(WRITERS)]
    go.touch()
    for child in children:
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err
        assert out.split() == [str(seq) for seq in range(ROWS_PER_WRITER)]

    with contextlib.closing(_check(path)) as connection:
        assert connection.execute(
            "SELECT thought FROM steps WHERE task_id = 'old'"
        ).fetchall() == [("kept",)]
        for i in range(WRITERS):
            calls = connection.execute(
                "SELECT seq, content FROM llm_calls WHERE task_id = ? ORDER BY seq",
                (f"w{i}",),
            ).fetchall()
            steps = connection.execute(
                "SELECT thought FROM steps WHERE task_id = ? ORDER BY seq", (f"w{i}",)
            ).fetchall()
            expected = [f"w{i}-{seq}" for seq in range(ROWS_PER_WRITER)]
            assert calls == list(enumerate(expected))
            assert [thought for (thought,) in steps] == expected


def test_a_killed_writer_leaves_every_committed_row(tmp_path: Path) -> None:
    path, go = tmp_path / "history.db", tmp_path / "go"
    _write_v0(path)
    go.touch()
    child = _child(path, go, "w", -1)
    assert child.stdout is not None
    committed: list[int] = []
    deadline = time.monotonic() + 30
    while len(committed) < 30 and time.monotonic() < deadline:
        line = child.stdout.readline()
        if not line:
            break
        committed.append(int(line))
    os.kill(child.pid, signal.SIGKILL)
    child.wait(timeout=30)
    assert len(committed) == 30, child.stderr.read() if child.stderr else ""

    with contextlib.closing(_check(path)) as connection:
        seqs = [
            seq
            for (seq,) in connection.execute(
                "SELECT seq FROM llm_calls WHERE task_id = 'w' ORDER BY seq"
            )
        ]
    # Every row the child saw committed reads back; a later one may too.
    assert seqs[: len(committed)] == committed
    assert seqs == list(range(len(seqs)))


# ── The typed long store's slice (R6-lite) ──


def test_twenty_sessions_migrate_once_and_read_the_same_store(tmp_path: Path) -> None:
    import asyncio

    from localmelo.melo.memory.long.sqlite import SqliteLongTerm

    path = tmp_path / "long_term.db"

    async def run() -> None:
        first = SqliteLongTerm(path)
        await first.apply(
            text="I live in Berlin near the river and work at Acme",
            kind="profile",
            source="user",
            admitted_by="user_statement",
            provenance={"source_ids": ["r1"]},
            task_id="t1",
            session_id="",
            created_at=1.0,
            embedding=None,
            action="NEW",
            target_id=None,
            reason="new",
            source_ids=["r1"],
            keep_text=False,
            candidate_id="c1",
        )
        await first.aclose()
        for _session in range(20):
            store = SqliteLongTerm(path)
            try:
                entries = await store.active_entries()
                decisions = await store.decisions()
                assert len(entries) == 1
                assert len(decisions) == 1
            finally:
                await store.aclose()

    asyncio.run(run())
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_repeated_identical_facts_do_not_grow_the_store(tmp_path: Path) -> None:
    import asyncio

    from localmelo.melo.memory.hippo import decide
    from localmelo.melo.memory.long.sqlite import SqliteLongTerm

    path = tmp_path / "long_term.db"
    text = "the deployment checklist is owned by Alice"

    async def run() -> None:
        store = SqliteLongTerm(path)
        try:
            for session in range(20):
                active = list(await store.active_entries("episodic"))
                decision = decide(
                    text,
                    kind="episodic",
                    source="reasoning_note",
                    provenance={"task_text": "the release"},
                    active=active,
                )
                await store.apply(
                    text=text,
                    kind="episodic",
                    source="reasoning_note",
                    admitted_by="rehearsal",
                    provenance={"task_text": "the release", "source_ids": []},
                    task_id=f"t{session}",
                    session_id="",
                    created_at=float(session + 1),
                    embedding=None,
                    action=decision.action,
                    target_id=decision.target_id,
                    reason=decision.reason,
                    source_ids=[],
                    keep_text=False,
                    candidate_id=f"c{session}",
                )
            assert len(await store.active_entries()) == 1
            assert len(await store.decisions()) == 20
        finally:
            await store.aclose()

    asyncio.run(run())


def test_a_forgotten_task_leaves_no_row_behind_on_restart(tmp_path: Path) -> None:
    import asyncio

    from localmelo.melo.memory.long.sqlite import SqliteLongTerm

    path = tmp_path / "long_term.db"

    async def run() -> None:
        store = SqliteLongTerm(path)
        try:
            await store.apply(
                text="I live in Berlin near the river and work at Acme",
                kind="profile",
                source="user",
                admitted_by="user_statement",
                provenance={"source_ids": []},
                task_id="t1",
                session_id="",
                created_at=1.0,
                embedding=None,
                action="NEW",
                target_id=None,
                reason="new",
                source_ids=[],
                keep_text=False,
                candidate_id="c1",
            )
            await store.forget("t1")
        finally:
            await store.aclose()
        reopened = SqliteLongTerm(path)
        try:
            assert await reopened.active_entries() == []
            assert await reopened.decisions() == []
        finally:
            await reopened.aclose()

    asyncio.run(run())
