"""The conversation index: one document per finished turn of every session.

A document is the user's typed text and the answer without its thinking, cut
to :data:`DOC_CHARS` and passed through ``redact_secrets``; a task's carries
its title too. Tool output, page text and attachments are never indexed. The
file is ``index.db`` in the memory folder, mode 0600, shared by every surface;
deleting a chat deletes its documents. It can be rebuilt from the session files.
Its ``meta`` table also holds the sleep watermark (``sleep_until``), and
``sleep_runs`` what each sleep run changed, by id, for its undo.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

import aiosqlite
import numpy as np

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory._sqlite import (
    checkpoint_wal,
    create_private_file,
    prepare_connection,
    secure_deletes,
)
from localmelo.melo.memory.notes import split_sentences
from localmelo.support.memory.records import sensitive

DOC_CHARS: Final = 2_000

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    surface     TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'chat',
    title       TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    vector      BLOB,
    embed_model TEXT NOT NULL DEFAULT '',
    UNIQUE (session_id, turn)
);
CREATE INDEX IF NOT EXISTS documents_session ON documents(session_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sleep_runs (
    run_id  TEXT PRIMARY KEY,
    started REAL NOT NULL,
    changes TEXT NOT NULL
);
"""
_COLUMNS: Final = (
    "id, session_id, turn, surface, kind, title, text, created_at, vector, embed_model"
)


@dataclass(frozen=True)
class Document:
    """One indexed turn; ``id`` is 0 until it is stored."""

    session_id: str
    turn: int
    surface: str
    kind: str
    title: str
    text: str
    created_at: float
    vector: tuple[float, ...] | None = None
    embed_model: str = ""
    id: int = 0


def _kept(text: str) -> str:
    """*text* without the sentences a note would refuse (a password, a card number)."""
    for _, sentence in reversed(split_sentences(text)):
        if sensitive(sentence):
            text = text.replace(sentence, "[left out]")
    return text


def document_text(typed: str, answer: str, *, title: str = "") -> str:
    """What a turn's document holds: a task's title, who said what, bounded."""
    parts = [f"Task: {title.strip()}"] if title.strip() else []
    parts += [f"User: {_kept(typed.strip())}", f"melo: {_kept(answer.strip())}"]
    return redact_secrets("\n".join(parts)[:DOC_CHARS])[0]


def typed_of(text: str) -> str:
    """What the user typed, in a chat turn's document text (:func:`document_text`)."""
    head = text.split("\nmelo: ", 1)[0]
    return head.removeprefix("User: ") if head.startswith("User: ") else ""


class ConversationIndex:
    """``index.db``: add, list and delete the documents of every session."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _ready(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._conn is None:
                create_private_file(self._path)
                conn = await aiosqlite.connect(self._path)
                try:
                    await prepare_connection(conn, _SCHEMA)
                except BaseException:
                    with contextlib.suppress(Exception):
                        await conn.close()
                    raise
                self._conn = conn
            return self._conn

    async def add(self, doc: Document) -> None:
        """Store *doc*, in place of an earlier document of the same turn."""
        conn = await self._ready()
        vector = None if doc.vector is None else np.asarray(doc.vector, np.float32)
        await conn.execute(
            "INSERT OR REPLACE INTO documents (session_id, turn, surface, kind, "
            "title, text, created_at, vector, embed_model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc.session_id,
                doc.turn,
                doc.surface,
                doc.kind,
                doc.title,
                doc.text,
                doc.created_at or time.time(),
                None if vector is None else vector.tobytes(),
                doc.embed_model if vector is not None else "",
            ),
        )
        await conn.commit()

    async def ids(self) -> set[int]:
        conn = await self._ready()
        async with conn.execute("SELECT id FROM documents") as cur:
            return {int(row[0]) for row in await cur.fetchall()}

    async def fetch(self, ids: Iterable[int]) -> list[Document]:
        """The documents with these ids (in chunks; SQLite caps the parameters)."""
        conn = await self._ready()
        wanted = sorted(ids)
        out: list[Document] = []
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            marks = ", ".join("?" * len(chunk))
            async with conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE id IN ({marks})", chunk
            ) as cur:
                for row in await cur.fetchall():
                    blob = row[8]
                    vector = (
                        tuple(float(v) for v in np.frombuffer(blob, np.float32))
                        if blob
                        else None
                    )
                    out.append(
                        Document(
                            session_id=str(row[1]),
                            turn=int(row[2]),
                            surface=str(row[3]),
                            kind=str(row[4]),
                            title=str(row[5]),
                            text=str(row[6]),
                            created_at=float(row[7]),
                            vector=vector,
                            embed_model=str(row[9]),
                            id=int(row[0]),
                        )
                    )
        return out

    async def count(self) -> int:
        conn = await self._ready()
        async with conn.execute("SELECT COUNT(*) FROM documents") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def delete(self, session_id: str, turn: int | None = None) -> int:
        """Delete a session's documents (one turn's with *turn*); returns how many."""
        conn = await self._ready()
        sql = "DELETE FROM documents WHERE session_id = ?"
        args: list[object] = [session_id]
        if turn is not None:
            sql += " AND turn = ?"
            args.append(turn)
        async with secure_deletes(conn):
            cur = await conn.execute(sql, args)
            await conn.commit()
        return int(cur.rowcount or 0)

    async def forget_all(self, *, mark: bool = True) -> int:
        """Delete every document; returns how many. With *mark*, a turn that ended
        before now is not indexed again (:meth:`forgotten_at`)."""
        conn = await self._ready()
        async with secure_deletes(conn):
            cur = await conn.execute("DELETE FROM documents")
            if mark:
                await conn.execute("DELETE FROM sleep_runs")
                await conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('forgotten_at', ?)",
                    (time.time(),),
                )
            await conn.commit()
        await checkpoint_wal(conn)
        return int(cur.rowcount or 0)

    async def forgotten_at(self) -> float:
        """When the user last forgot everything; 0 for never."""
        conn = await self._ready()
        async with conn.execute(
            "SELECT value FROM meta WHERE key = 'forgotten_at'"
        ) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def meta(self, key: str) -> float:
        """A ``meta`` value; 0 for one never set."""
        conn = await self._ready()
        async with conn.execute("SELECT value FROM meta WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def set_meta(self, key: str, value: float) -> None:
        conn = await self._ready()
        await conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        await conn.commit()

    async def save_run(
        self, run_id: str, started: float, changes: list[dict[str, Any]]
    ) -> None:
        """Keep what a sleep run changed (ids only, no note text) for its undo."""
        conn = await self._ready()
        await conn.execute(
            "INSERT OR REPLACE INTO sleep_runs (run_id, started, changes) "
            "VALUES (?, ?, ?)",
            (run_id, started, json.dumps(changes)),
        )
        await conn.commit()

    async def load_run(self, run_id: str) -> list[dict[str, Any]] | None:
        """What a sleep run changed; ``None`` for a run unknown or already undone."""
        conn = await self._ready()
        async with conn.execute(
            "SELECT changes FROM sleep_runs WHERE run_id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        return list(json.loads(row[0])) if row else None

    async def drop_run(self, run_id: str) -> None:
        conn = await self._ready()
        await conn.execute("DELETE FROM sleep_runs WHERE run_id = ?", (run_id,))
        await conn.commit()

    async def aclose(self) -> None:
        if self._conn is not None:
            conn, self._conn = self._conn, None
            await conn.close()
