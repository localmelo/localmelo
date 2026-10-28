"""In-place migration of ``long_term.db`` from v0 to v1."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from localmelo.melo.memory.long import LongTermSchemaError
from localmelo.melo.memory.long.sqlite import SqliteLongTerm, read_export_entries


def _write_v0(path: Path) -> tuple[bytes, bytes]:
    """A v0 file: a step summary and a reflection; returns their bytes."""
    summary = "read list_ops.py and fixed the import"
    reflection = "Attempt 1: the test was wrong"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE long_term (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, vector BLOB NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}')"
        )
        summary_meta = json.dumps({"step_id": "s1"})
        reflection_meta = json.dumps({"task_id": "t1", "type": "reflection"})
        conn.execute(
            "INSERT INTO long_term (text, vector, metadata) VALUES (?, ?, ?)",
            (
                summary,
                np.asarray([0.0, 1.0], dtype=np.float32).tobytes(),
                summary_meta,
            ),
        )
        conn.execute(
            "INSERT INTO long_term (text, vector, metadata) VALUES (?, ?, ?)",
            (
                reflection,
                np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
                reflection_meta,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT metadata FROM long_term WHERE text = ?", (summary,)
        ).fetchone()
        assert row is not None
        return row[0].encode(), reflection_meta.encode()
    finally:
        conn.close()


def _user_version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


async def test_a_v0_file_migrates_in_place_and_keeps_its_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    summary_meta, reflection_meta = _write_v0(path)
    store = SqliteLongTerm(path)
    try:
        entries = {entry.text: entry for entry in await store.active_entries()}
        assert entries["read list_ops.py and fixed the import"].kind == "episodic"
        assert entries["read list_ops.py and fixed the import"].source == "step_summary"
        reflection = entries["Attempt 1: the test was wrong"]
        assert reflection.kind == "reflection"
        assert reflection.source == "reflection"
        assert reflection.task_id == "t1"
        assert reflection.status == "active"
        assert all(entry.status == "active" for entry in entries.values())
    finally:
        await store.aclose()
    assert _user_version(path) == 1
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT text, vector, metadata FROM long_term WHERE text = ?",
            ("read list_ops.py and fixed the import",),
        ).fetchone()
    assert row is not None
    assert row[2].encode() == summary_meta
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT metadata FROM long_term WHERE text = ?",
            ("Attempt 1: the test was wrong",),
        ).fetchone()
    assert row is not None
    assert row[0].encode() == reflection_meta


async def test_the_legacy_cosine_search_returns_the_same_rows_in_the_same_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    store = SqliteLongTerm(path)
    try:
        rows = await store.search([1.0, 0.0], top_k=5)
        assert [text for text, _score, _meta in rows] == [
            "Attempt 1: the test was wrong",
            "read list_ops.py and fixed the import",
        ]
    finally:
        await store.aclose()


async def test_the_migration_is_idempotent_and_a_second_opener_does_not_redo_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    first = SqliteLongTerm(path)
    await first.active_entries()
    await first.aclose()
    second = SqliteLongTerm(path)
    try:
        entries = await second.active_entries()
        assert len(entries) == 2
        assert _user_version(path) == 1
    finally:
        await second.aclose()


async def test_a_half_migrated_file_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "ALTER TABLE long_term ADD COLUMN kind TEXT NOT NULL DEFAULT 'episodic'"
        )
        conn.execute(
            "ALTER TABLE long_term ADD COLUMN source TEXT NOT NULL "
            "DEFAULT 'step_summary'"
        )
        conn.commit()
    store = SqliteLongTerm(path)
    try:
        entries = {entry.text: entry for entry in await store.active_entries()}
        # The derivation is idempotent: the reflection is classified again
        # and the missing columns are added, so the file reaches v1.
        assert entries["Attempt 1: the test was wrong"].kind == "reflection"
        assert _user_version(path) == 1
    finally:
        await store.aclose()


async def test_a_newer_file_raises_and_names_the_version_and_the_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    store = SqliteLongTerm(path)
    with pytest.raises(LongTermSchemaError) as caught:
        await store.count_entries()
    assert "2" in str(caught.value)
    assert str(path) in str(caught.value)
    await store.aclose()


async def test_two_openers_migrate_once(tmp_path: Path) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    first = SqliteLongTerm(path)
    second = SqliteLongTerm(path)
    try:
        await first.count_entries()
        await second.count_entries()
        assert len(await second.active_entries()) == 2
        assert _user_version(path) == 1
    finally:
        await first.aclose()
        await second.aclose()


async def test_the_read_only_export_of_a_v0_file_does_not_migrate_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    entries = read_export_entries(str(path))
    assert [entry["text"] for entry in entries] == [
        "read list_ops.py and fixed the import",
        "Attempt 1: the test was wrong",
    ]
    assert entries[0]["kind"] == "episodic"
    assert entries[1]["kind"] == "reflection"
    assert _user_version(path) == 0


def test_the_export_of_a_newer_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "long_term.db"
    _write_v0(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    with pytest.raises(LongTermSchemaError):
        read_export_entries(str(path))
