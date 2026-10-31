from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from localmelo.melo.memory._sqlite import (
    checkpoint_wal,
    create_private_file,
    prepare_connection,
    secure_deletes,
)
from localmelo.melo.memory.assertions import decide_assertion
from localmelo.melo.memory.bm25 import BM25Index
from localmelo.melo.memory.long import (
    ACTIONS,
    KINDS,
    SOURCES,
    STORE_SIGNALS,
    LongTerm,
    LongTermSchemaError,
    StoredEntry,
    _history_order,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
# A records file (``SqliteLongTerm(path, records=True)``) is version 2: v1
# plus the columns of a record the user sees and edits.
_RECORDS_VERSION = 2
_RECORD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("topic", "TEXT NOT NULL DEFAULT ''"),
    ("quote", "TEXT NOT NULL DEFAULT ''"),
    ("edited_at", "REAL NOT NULL DEFAULT 0"),
)

# The base table; kept under its old name for callers that read it.
_SCHEMA = """CREATE TABLE IF NOT EXISTS long_term (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    text     TEXT NOT NULL,
    vector   BLOB NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS long_term (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    text     TEXT NOT NULL,
    vector   BLOB NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""

# The typed columns, in the order the v1 schema adds them.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("kind", "TEXT NOT NULL DEFAULT 'episodic'"),
    ("source", "TEXT NOT NULL DEFAULT 'step_summary'"),
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("admitted_by", "TEXT NOT NULL DEFAULT ''"),
    ("supersedes", "INTEGER"),
    ("links_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("provenance_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("task_id", "TEXT NOT NULL DEFAULT ''"),
    ("session_id", "TEXT NOT NULL DEFAULT ''"),
    ("uses", "INTEGER NOT NULL DEFAULT 0"),
    ("created_at", "REAL NOT NULL DEFAULT 0"),
    ("last_used_at", "REAL NOT NULL DEFAULT 0"),
    ("embed_model", "TEXT NOT NULL DEFAULT ''"),
    ("embed_dim", "INTEGER NOT NULL DEFAULT 0"),
)

_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT NOT NULL, action TEXT NOT NULL, kind TEXT,
  admitted_by TEXT, entry_id INTEGER, target_id INTEGER, reason TEXT,
  source_ids_json TEXT, text_sha256 TEXT, text TEXT, timestamp REAL NOT NULL,
  task_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS decisions_task ON decisions(task_id);
"""

_SELECT_COLUMNS = (
    "id, text, vector, metadata, kind, source, status, admitted_by, supersedes, "
    "links_json, provenance_json, task_id, session_id, uses, created_at, "
    "last_used_at, embed_model, embed_dim"
)


def _row_meta(row: tuple[Any, ...]) -> dict[str, Any]:
    """The search shape of one typed row: metadata plus the typed columns."""
    (
        row_id,
        text,
        _vector,
        metadata_json,
        kind,
        source,
        status,
        admitted_by,
        supersedes,
        links_json,
        provenance_json,
        task_id,
        session_id,
        uses,
        created_at,
        last_used_at,
        embed_model,
        embed_dim,
    ) = row
    try:
        metadata = json.loads(metadata_json)
    except (TypeError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        links = json.loads(links_json)
    except (TypeError, ValueError):
        links = []
    if not isinstance(links, list):
        links = []
    try:
        provenance = json.loads(provenance_json)
    except (TypeError, ValueError):
        provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    origin = provenance.get("origin_task_ids")
    if isinstance(origin, list) and all(isinstance(i, str) for i in origin):
        origin_task_ids = origin
    elif task_id:
        origin_task_ids = [task_id]
    else:
        origin_task_ids = []
    return {
        **metadata,
        "id": int(row_id),
        "kind": str(kind),
        "source": str(source),
        "status": str(status),
        "admitted_by": str(admitted_by or ""),
        "provenance": provenance,
        "task_id": str(task_id or ""),
        "session_id": str(session_id or ""),
        "uses": int(uses),
        "created_at": float(created_at or 0.0),
        "last_used_at": float(last_used_at or 0.0),
        "supersedes": None if supersedes is None else int(supersedes),
        "links": [int(link) for link in links],
        "embed_model": str(embed_model or ""),
        "embed_dim": int(embed_dim or 0),
        "origin_task_ids": origin_task_ids,
    }


def _stored(row: tuple[Any, ...]) -> StoredEntry:
    meta = _row_meta(row)
    vector_bytes = bytes(row[2] or b"")
    vector = (
        tuple(float(v) for v in np.frombuffer(vector_bytes, dtype=np.float32))
        if vector_bytes
        else None
    )
    return StoredEntry(
        id=meta["id"],
        text=str(row[1]),
        kind=meta["kind"],
        source=meta["source"],
        status=meta["status"],
        admitted_by=meta["admitted_by"],
        provenance=meta["provenance"],
        task_id=meta["task_id"],
        session_id=meta["session_id"],
        uses=meta["uses"],
        created_at=meta["created_at"],
        last_used_at=meta["last_used_at"],
        supersedes=meta["supersedes"],
        links=tuple(meta["links"]),
        embed_model=meta["embed_model"],
        embed_dim=meta["embed_dim"],
        vector=vector,
    )


class SqliteLongTerm(LongTerm):
    """Persistent, typed long-term memory backed by SQLite via aiosqlite.

    Vectors are stored as raw float32 ``BLOB``s. The connection is opened
    lazily on first awaited call; a missing file is created with mode 0600,
    an existing one keeps its mode. Writes serialize through
    ``_write_lock``; the cosine scan and BM25 ranking run in a worker thread
    so a long fetch does not stall the loop.

    With *records*, the file is a records file (schema version 2), whose
    rows also keep the record's ``topic``, the user's ``quote`` and when it
    was ``edited_at``; a plain store refuses such a file.
    """

    def __init__(self, db_path: str | Path, *, records: bool = False) -> None:
        self._db_path = str(db_path)
        self._version = _RECORDS_VERSION if records else _SCHEMA_VERSION
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._index: BM25Index | None = None
        self._index_rows: dict[str, dict[str, Any]] = {}

    async def _ensure_ready(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            if self._conn is not None:
                return self._conn
            if self._db_path != ":memory:":
                create_private_file(self._db_path)
            conn = await aiosqlite.connect(self._db_path)
            try:
                await prepare_connection(conn, _BASE_SCHEMA)
                await self._migrate(conn)
            except BaseException:
                with contextlib.suppress(Exception):
                    await conn.close()
                raise
            self._conn = conn
            return conn

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        """Migrate the file to its version in one transaction; a current one is left alone."""
        await conn.execute("BEGIN IMMEDIATE")
        try:
            # Read the version inside the transaction, as history does, so a
            # second opener sees the version the first one committed.
            async with conn.execute("PRAGMA user_version") as cur:
                row = await cur.fetchone()
            version = int(row[0]) if row else 0
            if version > self._version:
                raise LongTermSchemaError(
                    f"{self._db_path} is at schema version {version}; this "
                    f"localmelo understands version {self._version} and older"
                )
            if version == self._version:
                # A v1 file may still lack the decisions table after an
                # unusual interruption; CREATE IF NOT EXISTS repairs it
                # without touching the entries.
                for statement in _DECISIONS.split(";"):
                    if statement.strip():
                        await conn.execute(statement)
                await conn.commit()
                return
            async with conn.execute("PRAGMA table_info(long_term)") as cur:
                existing = {str(row[1]) for row in await cur.fetchall()}
            columns = _COLUMNS
            if self._version == _RECORDS_VERSION:
                columns += _RECORD_COLUMNS
            for name, ddl in columns:
                if name in existing:
                    continue
                await conn.execute(f"ALTER TABLE long_term ADD COLUMN {name} {ddl}")
            # The derivation is idempotent, so a half-migrated file is
            # repaired too: every v0 row still gets its legacy kind/source
            # and task id, and a row already derived keeps that value.
            await conn.execute(
                "UPDATE long_term SET kind = 'reflection', source = 'reflection' "
                "WHERE json_extract(metadata, '$.type') = 'reflection'"
            )
            await conn.execute(
                "UPDATE long_term SET task_id = "
                "json_extract(metadata, '$.task_id') "
                "WHERE task_id = '' AND json_extract(metadata, '$.task_id') "
                "IS NOT NULL"
            )
            for statement in _DECISIONS.split(";"):
                if statement.strip():
                    await conn.execute(statement)
            await conn.execute(f"PRAGMA user_version = {self._version}")
            await conn.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                await conn.rollback()
            raise

    async def add(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """The legacy write: an episodic step summary with the given metadata."""
        conn = await self._ensure_ready()
        meta = dict(metadata or {})
        vec = np.asarray(embedding, dtype=np.float32)
        task_id = str(meta.get("task_id") or "")
        session_id = str(meta.get("session_id") or "")
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO long_term (text, vector, metadata, kind, source, "
                "status, admitted_by, task_id, session_id, created_at, embed_dim) "
                "VALUES (?, ?, ?, 'episodic', 'step_summary', 'active', '', ?, ?, "
                "?, ?)",
                (
                    text,
                    vec.tobytes(),
                    json.dumps(meta, ensure_ascii=False),
                    task_id,
                    session_id,
                    time.time(),
                    int(vec.size),
                ),
            )
            await conn.commit()
        self._index = None

    async def apply(
        self,
        *,
        text: str,
        kind: str,
        source: str,
        admitted_by: str,
        provenance: dict[str, Any],
        task_id: str,
        session_id: str,
        created_at: float,
        embedding: list[float] | None,
        action: str,
        target_id: int | None,
        reason: str,
        source_ids: Any,
        keep_text: bool,
        candidate_id: str,
        typed_assertion: bool = False,
        decision_details: dict[str, Any] | None = None,
        record: dict[str, str] | None = None,
    ) -> tuple[int | None, str]:
        """Write one decision, its entry and its decision row, in one transaction.

        *record* (a records file only) sets the new row's ``topic``,
        ``quote`` and ``embed_model`` columns.
        """
        for name, value, allowed in (
            ("kind", kind, KINDS),
            ("source", source, SOURCES),
            ("admitted_by", admitted_by, STORE_SIGNALS),
            ("action", action, ACTIONS),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown long-term {name}: {value!r}")
        conn = await self._ensure_ready()
        vector = (
            b""
            if embedding is None
            else np.asarray(embedding, dtype=np.float32).tobytes()
        )
        now = time.time()
        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                actual = action
                assertion = provenance.get("assertion")
                if typed_assertion and isinstance(assertion, dict) and task_id:
                    async with conn.execute(
                        "SELECT id, provenance_json, created_at FROM long_term "
                        "WHERE status = 'active' AND kind = ? AND task_id = ?",
                        (kind, task_id),
                    ) as cur:
                        existing = await cur.fetchall()
                    typed_rows = []
                    for existing_id, raw, observed_at in existing:
                        try:
                            other = json.loads(raw).get("assertion")
                        except (ValueError, AttributeError, TypeError):
                            continue
                        if isinstance(other, dict):
                            typed_rows.append(
                                (int(existing_id), other, float(observed_at))
                            )
                    actual, target_id, reason = decide_assertion(
                        assertion, typed_rows, created_at
                    )
                target_status = ""
                if target_id is not None:
                    async with conn.execute(
                        "SELECT status FROM long_term WHERE id = ?", (target_id,)
                    ) as cur:
                        row = await cur.fetchone()
                    target_status = str(row[0]) if row else ""
                if actual == "UPDATE" and target_status != "active":
                    actual = "NEW"
                    target_id = None
                entry_id: int | None = None
                if actual == "IGNORE":
                    if target_id is not None:
                        await conn.execute(
                            "UPDATE long_term SET uses = uses + 1, last_used_at = ? "
                            "WHERE id = ?",
                            (now, target_id),
                        )
                else:
                    links = (
                        [target_id]
                        if actual in ("UPDATE", "RELATED", "CONFLICT")
                        and target_id is not None
                        else []
                    )
                    supersedes = (
                        target_id
                        if actual == "UPDATE" and target_id is not None
                        else None
                    )
                    origin = provenance.get("origin_task_ids") or (
                        [task_id] if task_id else []
                    )
                    metadata: dict[str, Any] = {
                        "task_id": task_id,
                        "session_id": session_id,
                        "created_at": created_at or now,
                        "origin_task_ids": list(origin),
                    }
                    if kind == "reflection":
                        metadata["type"] = "reflection"
                    embed_dim = 0 if embedding is None else len(embedding)
                    cursor = await conn.execute(
                        "INSERT INTO long_term (text, vector, metadata, kind, source, "
                        "status, admitted_by, supersedes, links_json, "
                        "provenance_json, task_id, session_id, uses, created_at, "
                        "last_used_at, embed_dim) VALUES (?, ?, ?, ?, ?, 'active', "
                        "?, ?, ?, ?, ?, ?, 0, ?, 0, ?)",
                        (
                            text,
                            vector,
                            json.dumps(metadata, ensure_ascii=False),
                            kind,
                            source,
                            admitted_by,
                            supersedes,
                            json.dumps(links),
                            json.dumps(provenance, ensure_ascii=False),
                            task_id,
                            session_id,
                            created_at or now,
                            embed_dim,
                        ),
                    )
                    entry_id = int(cursor.lastrowid or 0) or None
                    if record and self._version == _RECORDS_VERSION:
                        await conn.execute(
                            "UPDATE long_term SET topic = ?, quote = ?, "
                            "embed_model = ? WHERE id = ?",
                            (
                                record.get("topic", ""),
                                record.get("quote", ""),
                                record.get("embed_model", ""),
                                entry_id,
                            ),
                        )
                    if supersedes is not None:
                        await conn.execute(
                            "UPDATE long_term SET status = 'superseded' WHERE id = ?",
                            (supersedes,),
                        )
                await conn.execute(
                    "INSERT INTO decisions (candidate_id, action, kind, admitted_by, "
                    "entry_id, target_id, reason, source_ids_json, text_sha256, text, "
                    "timestamp, task_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate_id,
                        actual,
                        kind,
                        admitted_by,
                        entry_id,
                        target_id,
                        reason[:500],
                        json.dumps(list(source_ids)),
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        text if keep_text else None,
                        now,
                        task_id,
                    ),
                )
                await conn.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await conn.rollback()
                raise
        self._index = None
        if decision_details is not None:
            decision_details.update(target_id=target_id, reason=reason)
        return entry_id, actual

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        kinds: Any = None,
        exclude_task_ids: Any = (),
        task_key: str = "",
    ) -> list[tuple[str, float, dict[str, Any]]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_unit = q / q_norm
        conn = await self._ensure_ready()
        sql = (
            f"SELECT {_SELECT_COLUMNS} FROM long_term "
            "WHERE status = 'active' AND length(vector) > 0"
        )
        params: list[Any] = []
        if kinds:
            names = tuple(kinds)
            sql += f" AND kind IN ({', '.join('?' for _ in names)})"
            params.extend(names)
        excluded = tuple(exclude_task_ids)
        if excluded:
            sql += f" AND task_id NOT IN ({', '.join('?' for _ in excluded)})"
            params.extend(excluded)
        sql += " ORDER BY id"
        async with conn.execute(sql, params) as cur:
            rows = [tuple(row) for row in await cur.fetchall()]
        if not rows:
            return []
        return await asyncio.to_thread(_rank_rows, rows, q_unit, top_k, task_key)

    async def search_embedding(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Alias of :meth:`search`, the name the typed API uses."""
        return await self.search(*args, **kwargs)

    async def search_text(
        self,
        query: str,
        top_k: int = 5,
        *,
        kinds: Any = None,
        exclude_task_ids: Any = (),
        task_key: str = "",
        include_history: bool = False,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        conn = await self._ensure_ready()
        filtered = bool(kinds) or bool(tuple(exclude_task_ids))
        if include_history or self._db_path == ":memory:":
            statuses = "('active', 'superseded')" if include_history else "('active')"
            async with conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM long_term "
                f"WHERE status IN {statuses} ORDER BY created_at, id"
            ) as cur:
                rows = [tuple(row) for row in await cur.fetchall()]
            metas, texts = _index_rows(
                rows, tuple(kinds or ()), tuple(exclude_task_ids)
            )
            index = BM25Index()
            for doc_id, text in texts.items():
                index.add(doc_id, text)
            filtered = True
        elif filtered:
            metas, texts = await asyncio.to_thread(
                _fetch_and_index,
                self._db_path,
                tuple(kinds or ()),
                tuple(exclude_task_ids),
            )
            index = BM25Index()
            for doc_id, text in texts.items():
                index.add(doc_id, text)
        else:
            if self._index is None:
                metas, texts = await asyncio.to_thread(
                    _fetch_and_index, self._db_path, (), ()
                )
                index = BM25Index()
                for doc_id, text in texts.items():
                    index.add(doc_id, text)
                self._index = index
                self._index_rows = metas
            index = self._index
            metas = self._index_rows
        if not metas or not query.strip():
            return []
        scored = [
            (metas[doc_id], score)
            for doc_id, score in index.search(query, len(metas))
            if score > 0
        ]
        if task_key:
            scored = [
                item
                for item in scored
                if item[0]["provenance"].get("task_key") == task_key
            ]
        scored.sort(key=lambda item: _tie_key(item[0], item[1]), reverse=True)
        return [
            (str(meta["text"]), score, _strip_index_meta(meta))
            for meta, score in scored[:top_k]
        ]

    async def active_entries(self, kind: str | None = None) -> list[StoredEntry]:
        conn = await self._ensure_ready()
        sql = f"SELECT {_SELECT_COLUMNS} FROM long_term WHERE status = 'active'"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at, id"
        async with conn.execute(sql, params) as cur:
            rows = [tuple(row) for row in await cur.fetchall()]
        return [_stored(row) for row in rows]

    async def entry(self, entry_id: int) -> StoredEntry | None:
        """Read one record, including a superseded one (for a guarded undo)."""
        conn = await self._ensure_ready()
        async with conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM long_term WHERE id = ?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
        return _stored(tuple(row)) if row is not None else None

    async def history_entries(self, task_id: str) -> list[StoredEntry]:
        """Return a task's entries, including superseded rows, newest first."""
        if not task_id:
            return []
        conn = await self._ensure_ready()
        async with conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM long_term WHERE task_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (task_id,),
        ) as cur:
            rows = [tuple(row) for row in await cur.fetchall()]
        entries = [_stored(row) for row in rows]
        entries.sort(key=_history_order, reverse=True)
        return entries

    async def decisions(self) -> list[dict[str, Any]]:
        conn = await self._ensure_ready()
        async with conn.execute(
            "SELECT seq, candidate_id, action, kind, admitted_by, entry_id, "
            "target_id, reason, source_ids_json, text_sha256, text, timestamp, "
            "task_id FROM decisions ORDER BY seq"
        ) as cur:
            return [_decision_dict(tuple(row)) for row in await cur.fetchall()]

    async def count_entries(self) -> int:
        """Diagnostic helper: total long-term entries persisted."""
        conn = await self._ensure_ready()
        async with conn.execute("SELECT COUNT(*) FROM long_term") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def delete_entry(self, entry_id: int, *, restore: bool = False) -> bool:
        """Delete one entry and its vector; ``False`` when there is none.

        The text of every decision row about it is blanked (``decisions.text``
        would keep it otherwise), and a ``DELETE`` row records the delete.
        With *restore*, the entry it superseded is active again, as an undo
        of the write wants. The delete overwrites the freed pages.
        """
        conn = await self._ensure_ready()
        async with self._write_lock, secure_deletes(conn):
            await conn.execute("BEGIN IMMEDIATE")
            try:
                async with conn.execute(
                    "SELECT kind, supersedes FROM long_term WHERE id = ?", (entry_id,)
                ) as cur:
                    row = await cur.fetchone()
                if row is not None:
                    gone = [entry_id]
                    # Without a restore, the wordings it replaced go with it.
                    older = None if restore else row[1]
                    while older is not None and older not in gone:
                        gone.append(older)
                        async with conn.execute(
                            "SELECT supersedes FROM long_term WHERE id = ?", (older,)
                        ) as cur:
                            found = await cur.fetchone()
                        older = found[0] if found else None
                    marks = ", ".join("?" * len(gone))
                    await conn.execute(
                        f"DELETE FROM long_term WHERE id IN ({marks})", gone
                    )
                    await conn.execute(
                        "UPDATE decisions SET text = NULL, text_sha256 = '' "
                        f"WHERE entry_id IN ({marks}) OR target_id IN ({marks})",
                        [*gone, *gone],
                    )
                    if restore and row[1] is not None:
                        await conn.execute(
                            "UPDATE long_term SET status = 'active' "
                            "WHERE id = ? AND status = 'superseded'",
                            (row[1],),
                        )
                    await conn.execute(
                        "INSERT INTO decisions (candidate_id, action, kind, "
                        "admitted_by, target_id, reason, source_ids_json, "
                        "text_sha256, timestamp) VALUES ('', 'DELETE', ?, "
                        "'user_edit', ?, 'deleted by the user', '[]', '', ?)",
                        (row[0], entry_id, time.time()),
                    )
                await conn.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await conn.rollback()
                raise
            # The write-ahead log would keep the old wording until it is folded in.
            await checkpoint_wal(conn)
        self._index = None
        return row is not None

    async def edit_entry(
        self,
        entry_id: int,
        *,
        text: str,
        embedding: list[float] | None,
        embed_model: str,
    ) -> bool:
        """Rewrite an active record in place; ``False`` when there is none.

        The user's new wording replaces the old one and is the record's quote
        now; the old wording is overwritten (not kept as a superseded row),
        the text of every decision row about the record is blanked, and an
        ``EDIT`` row records the edit. Only a records file keeps edits.
        """
        if self._version != _RECORDS_VERSION:
            raise ValueError("only a records file keeps edited notes")
        conn = await self._ensure_ready()
        vector = (
            b"" if embedding is None else np.asarray(embedding, np.float32).tobytes()
        )
        async with self._write_lock, secure_deletes(conn):
            await conn.execute("BEGIN IMMEDIATE")
            try:
                async with conn.execute(
                    "SELECT kind, provenance_json FROM long_term "
                    "WHERE id = ? AND status = 'active'",
                    (entry_id,),
                ) as cur:
                    row = await cur.fetchone()
                if row is not None:
                    now = time.time()
                    provenance = json.loads(row[1])
                    provenance.update(
                        quote=text,
                        edited=True,
                        edited_at=now,
                        embed_model=embed_model if vector else "",
                    )
                    await conn.execute(
                        "UPDATE long_term SET text = ?, vector = ?, provenance_json = ?, "
                        "embed_dim = ?, embed_model = ?, quote = ?, edited_at = ? "
                        "WHERE id = ?",
                        (
                            text,
                            vector,
                            json.dumps(provenance, ensure_ascii=False),
                            0 if embedding is None else len(embedding),
                            embed_model if vector else "",
                            text,
                            now,
                            entry_id,
                        ),
                    )
                    await conn.execute(
                        "UPDATE decisions SET text = NULL, text_sha256 = '' "
                        "WHERE entry_id = ? OR target_id = ?",
                        (entry_id, entry_id),
                    )
                    await conn.execute(
                        "INSERT INTO decisions (candidate_id, action, kind, "
                        "admitted_by, entry_id, reason, source_ids_json, "
                        "text_sha256, timestamp) VALUES ('', 'EDIT', ?, "
                        "'user_edit', ?, 'edited by the user', '[]', '', ?)",
                        (row[0], entry_id, now),
                    )
                await conn.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await conn.rollback()
                raise
            await checkpoint_wal(conn)
        self._index = None
        return row is not None

    async def set_status(
        self,
        entry_ids: Sequence[int],
        status: str,
        *,
        keeper: int | None,
        admitted_by: str,
        reason: str,
    ) -> list[int]:
        """Mark active records superseded (by *keeper*, which stays active), or
        superseded ones active again; the ids that changed.

        Nothing is deleted. Each change writes an ``UPDATE`` decision row
        (``entry_id`` the keeper, ``target_id`` the record) with *reason*.
        """
        if status not in ("active", "superseded"):
            raise ValueError(f"Unknown long-term status: {status!r}")
        before = "superseded" if status == "active" else "active"
        conn = await self._ensure_ready()
        changed: list[int] = []
        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                for entry_id in entry_ids:
                    cur = await conn.execute(
                        "UPDATE long_term SET status = ? WHERE id = ? AND status = ?",
                        (status, entry_id, before),
                    )
                    if not cur.rowcount:
                        continue
                    changed.append(entry_id)
                    await conn.execute(
                        "INSERT INTO decisions (candidate_id, action, kind, "
                        "admitted_by, entry_id, target_id, reason, source_ids_json, "
                        "text_sha256, timestamp) SELECT '', 'UPDATE', kind, ?, ?, "
                        "id, ?, '[]', '', ? FROM long_term WHERE id = ?",
                        (admitted_by, keeper, reason[:500], time.time(), entry_id),
                    )
                await conn.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await conn.rollback()
                raise
        self._index = None
        return changed

    async def set_vector(
        self, entry_id: int, embedding: list[float] | None, embed_model: str
    ) -> None:
        """Replace an entry's vector and the model it came from (a reindex)."""
        conn = await self._ensure_ready()
        vector = (
            b"" if embedding is None else np.asarray(embedding, np.float32).tobytes()
        )
        async with self._write_lock:
            await conn.execute(
                "UPDATE long_term SET vector = ?, embed_dim = ?, embed_model = ? "
                "WHERE id = ?",
                (
                    vector,
                    0 if embedding is None else len(embedding),
                    embed_model if vector else "",
                    entry_id,
                ),
            )
            await conn.commit()
        self._index = None

    async def forget(self, task_id: str) -> None:
        """Delete the entries and decisions whose ``task_id`` is *task_id*."""
        await self._forget(task_id)

    async def forget_all(self) -> None:
        """Delete every entry and every decision."""
        await self._forget(None)

    async def _forget(self, task_id: str | None) -> None:
        conn = await self._ensure_ready()
        async with self._write_lock:
            async with secure_deletes(conn):
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    if task_id is None:
                        await conn.execute("DELETE FROM long_term")
                        await conn.execute("DELETE FROM decisions")
                    else:
                        await conn.execute(
                            "DELETE FROM long_term WHERE task_id = ?", (task_id,)
                        )
                        await conn.execute(
                            "DELETE FROM decisions WHERE task_id = ?", (task_id,)
                        )
                    await conn.commit()
                except BaseException:
                    with contextlib.suppress(Exception):
                        await conn.rollback()
                    raise
            if not await checkpoint_wal(conn):
                logger.warning(
                    "Forgot %s but could not truncate the WAL of %s: "
                    "another connection holds it",
                    "every entry" if task_id is None else f"the entries of {task_id}",
                    self._db_path,
                )
        self._index = None

    async def aclose(self) -> None:
        """Close the underlying aiosqlite connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def close(self) -> None:
        """Sync compatibility shim. Prefer :meth:`aclose` in new code."""
        self._conn = None


def _tie_key(meta: dict[str, Any], score: float) -> tuple[float, bool, float]:
    # The sort is stable, so the final tie keeps the rows' own order (id
    # ascending): a legacy file's equal scores stay in the order it wrote.
    return (
        score,
        meta.get("admitted_by") == "rehearsal",
        float(meta.get("created_at") or 0.0) if meta.get("admitted_by") else 0.0,
    )


def _strip_index_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """A cached index row, without the copy of its text kept for ranking."""
    return {key: value for key, value in meta.items() if key != "text"}


def _rank_rows(
    rows: list[tuple[Any, ...]], q_unit: np.ndarray, top_k: int, task_key: str = ""
) -> list[tuple[str, float, dict[str, Any]]]:
    scored: list[tuple[dict[str, Any], float, str]] = []
    for row in rows:
        vector = np.frombuffer(bytes(row[2] or b""), dtype=np.float32)
        # A vector from another embedding model has another dimension; it
        # cannot be compared, so it is left out rather than crashing.
        if vector.shape != q_unit.shape:
            continue
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        meta = _row_meta(row)
        if task_key and meta["provenance"].get("task_key") != task_key:
            continue
        scored.append((meta, float(np.dot(q_unit, vector / norm)), str(row[1])))
    scored.sort(key=lambda item: _tie_key(item[0], item[1]), reverse=True)
    return [(text, score, meta) for meta, score, text in scored[:top_k]]


def _index_rows(
    rows: list[tuple[Any, ...]],
    kinds: tuple[str, ...],
    exclude: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, str]]:
    metas: dict[str, Any] = {}
    texts: dict[str, str] = {}
    excluded = set(exclude)
    for row in rows:
        meta = _row_meta(row)
        if kinds and meta["kind"] not in kinds:
            continue
        if meta["task_id"] in excluded:
            continue
        doc_id = str(meta["id"])
        metas[doc_id] = {**meta, "text": str(row[1])}
        texts[doc_id] = str(row[1])
    return metas, texts


def _fetch_and_index(
    path: str, kinds: tuple[str, ...], exclude: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read the active rows and index their texts, for BM25 in a thread."""
    sql = (
        f"SELECT {_SELECT_COLUMNS} FROM long_term "
        "WHERE status = 'active' ORDER BY created_at, id"
    )
    params: list[Any] = []
    if kinds:
        sql = (
            f"SELECT {_SELECT_COLUMNS} FROM long_term "
            f"WHERE status = 'active' AND kind IN ({', '.join('?' for _ in kinds)}) "
            "ORDER BY created_at, id"
        )
        params.extend(kinds)
    with contextlib.closing(
        sqlite3.connect(f"{Path(path).absolute().as_uri()}?mode=ro", uri=True)
    ) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    metas: dict[str, Any] = {}
    texts: dict[str, str] = {}
    excluded = set(exclude)
    for row in rows:
        values = tuple(row)
        meta = _row_meta(values)
        if meta["task_id"] in excluded:
            continue
        doc_id = str(meta["id"])
        metas[doc_id] = {**meta, "text": str(values[1])}
        texts[doc_id] = str(values[1])
    return metas, texts


def _decision_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    try:
        source_ids = json.loads(row[8]) if row[8] else []
    except (TypeError, ValueError):
        source_ids = []
    if not isinstance(source_ids, list):
        source_ids = []
    return {
        "seq": int(row[0]),
        "candidate_id": str(row[1]),
        "action": str(row[2]),
        "kind": None if row[3] is None else str(row[3]),
        "admitted_by": None if row[4] is None else str(row[4]),
        "entry_id": None if row[5] is None else int(row[5]),
        "target_id": None if row[6] is None else int(row[6]),
        "reason": None if row[7] is None else str(row[7]),
        "source_ids": source_ids,
        "text_sha256": None if row[9] is None else str(row[9]),
        "text": None if row[10] is None else str(row[10]),
        "timestamp": float(row[11] or 0.0),
        "task_id": str(row[12] or ""),
    }


# ── Read-only exports (never migrate) ──


def _connect_read_only(path: str) -> sqlite3.Connection:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"{Path(path).absolute().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_version(conn: sqlite3.Connection, path: str) -> int:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > _SCHEMA_VERSION:
        raise LongTermSchemaError(
            f"{path} is at schema version {version}; this localmelo understands "
            f"version {_SCHEMA_VERSION} and older"
        )
    return version


def read_export_entries(path: str) -> list[dict[str, Any]]:
    """The active entries of a store file, v0 derived or v1 as stored."""
    with contextlib.closing(_connect_read_only(path)) as conn:
        version = _read_version(conn, path)
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(long_term)").fetchall()
        }
        if "text" not in columns:
            raise LongTermSchemaError(f"{path} has no long_term table")
        typed = version >= 1 and "kind" in columns
        select = (
            "id, text, metadata, kind, source, status, admitted_by, "
            "links_json, provenance_json, uses, created_at, last_used_at, "
            "supersedes, embed_model, embed_dim"
            if typed
            else "id, text, metadata"
        )
        where = " WHERE status = 'active'" if typed else ""
        order = " ORDER BY created_at, id" if typed else " ORDER BY id"
        rows = conn.execute(f"SELECT {select} FROM long_term{where}{order}").fetchall()
    entries: list[dict[str, Any]] = []
    for row in rows:
        if typed:
            try:
                metadata = json.loads(row[2])
            except (TypeError, ValueError):
                metadata = {}
            try:
                links = json.loads(row[7])
            except (TypeError, ValueError):
                links = []
            try:
                provenance = json.loads(row[8])
            except (TypeError, ValueError):
                provenance = {}
            entries.append(
                {
                    "id": int(row[0]),
                    "text": str(row[1]),
                    "kind": str(row[3]),
                    "source": str(row[4]),
                    "status": str(row[5]),
                    "admitted_by": str(row[6] or ""),
                    "provenance": provenance if isinstance(provenance, dict) else {},
                    "created_at": float(row[10] or 0.0),
                    "last_used_at": float(row[11] or 0.0),
                    "uses": int(row[9]),
                    "supersedes": None if row[12] is None else int(row[12]),
                    "links": links if isinstance(links, list) else [],
                    "embed_model": str(row[13] or ""),
                    "embed_dim": int(row[14] or 0),
                }
            )
            continue
        try:
            metadata = json.loads(row[2])
        except (TypeError, ValueError):
            metadata = {}
        is_reflection = isinstance(metadata, dict) and metadata.get("type") == (
            "reflection"
        )
        entries.append(
            {
                "id": int(row[0]),
                "text": str(row[1]),
                "kind": "reflection" if is_reflection else "episodic",
                "source": "reflection" if is_reflection else "step_summary",
                "status": "active",
                "admitted_by": "",
                "provenance": {},
                "created_at": 0.0,
                "last_used_at": 0.0,
                "uses": 0,
                "supersedes": None,
                "links": [],
                "embed_model": "",
                "embed_dim": 0,
            }
        )
    return entries


def read_export_decisions(path: str) -> list[dict[str, Any]]:
    """The decision rows of a store file, in ``seq`` order; v0 has none."""
    with contextlib.closing(_connect_read_only(path)) as conn:
        _read_version(conn, path)
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "decisions" not in tables:
            return []
        rows = conn.execute(
            "SELECT seq, candidate_id, action, kind, admitted_by, entry_id, "
            "target_id, reason, source_ids_json, text_sha256, text, timestamp, "
            "task_id FROM decisions ORDER BY seq"
        ).fetchall()
    return [_decision_dict(tuple(row)) for row in rows]


__all__ = [
    "LongTermSchemaError",
    "SqliteLongTerm",
    "read_export_decisions",
    "read_export_entries",
]
