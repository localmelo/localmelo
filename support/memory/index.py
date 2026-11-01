"""The conversation index: the documents of every finished turn of every session.

A turn's text is the user's typed words and the answer without its thinking,
passed through ``redact_secrets``; a task's carries its title too. A turn that
fits in :data:`DOC_CHARS` is one document. A longer one becomes up to
:data:`MAX_CHUNKS` documents of the same session and turn (:func:`document_chunks`):
chunk 0 is its head, each later chunk repeats a short head of what the user
typed and then the next window of the text, overlapping the one before; what is
left after the last chunk is not indexed. Tool output, page text and attachments
are never indexed. The file is ``index.db`` in the memory folder, mode 0600,
shared by every surface; deleting a chat or a turn deletes all its documents. It
can be rebuilt from the session files, which re-chunks every turn. Its ``meta``
table also holds the sleep watermark (``sleep_until``), and ``sleep_runs`` what
each sleep run changed, by id, for its undo.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Iterable, Sequence
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
# A turn's text over DOC_CHARS becomes at most this many chunks, each at most
# DOC_CHARS long; a later chunk starts CHUNK_OVERLAP characters before the one
# before it ended, and repeats the first USER_HEAD_CHARS of what the user typed.
MAX_CHUNKS: Final = 8
CHUNK_OVERLAP: Final = 200
USER_HEAD_CHARS: Final = 200
# How far before a window's edge a chunk looks for a paragraph, line or sentence end.
_BREAK_REACH: Final = 400
# A side chat's summary is the one document at this turn of its session; a
# task's exchanges take the turns below it.
SUMMARY_TURN: Final = -1

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
    chunk       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (session_id, turn, chunk)
);
CREATE INDEX IF NOT EXISTS documents_session ON documents(session_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sleep_processed (
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    note_id INTEGER NOT NULL DEFAULT 0,
    finished REAL NOT NULL,
    failures INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (session_id, turn, note_id)
);
CREATE TABLE IF NOT EXISTS sleep_runs (
    run_id  TEXT PRIMARY KEY,
    started REAL NOT NULL,
    changes TEXT NOT NULL
);
"""
_COLUMNS: Final = (
    "id, session_id, turn, surface, kind, title, text, created_at, vector, "
    "embed_model, chunk"
)


@dataclass(frozen=True)
class Document:
    """One indexed piece of a turn; ``id`` is 0 until it is stored.

    ``chunk`` is 0 for a turn's head (its only document when it fits), 1 and up
    for the windows after it (:func:`document_chunks`).
    """

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
    chunk: int = 0


def _kept(text: str) -> str:
    """*text* without the sentences a note would refuse (a password, a card number)."""
    for _, sentence in reversed(split_sentences(text)):
        if sensitive(sentence):
            text = text.replace(sentence, "[left out]")
    return text


def _clean(text: str) -> str:
    """*text* without its secrets: a sentence a note would refuse is left out, the
    rest redacted. A private key block is redacted whole first: left out line by
    line, its header would go and its body stay."""
    text = text.strip()
    if "PRIVATE KEY" in text:
        text = redact_secrets(text)[0]
    return redact_secrets(_kept(text))[0]


def _parts(typed: str, answer: str, title: str) -> tuple[str, str, str]:
    """A turn's task line (``""`` for a chat), typed text and answer, each cleaned
    (:func:`_clean`). Each is cleaned whole, before any cut, so no half of a key
    or key block is left at a chunk's edge."""
    lead = f"Task: {title.strip()}\n" if title.strip() else ""
    return redact_secrets(lead)[0], _clean(typed), _clean(answer)


def _fences(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` of each fenced code block of *text*: from its opening
    fence line to the end of its closing one (to the end of *text* when there
    is none)."""
    spans: list[tuple[int, int]] = []
    opened = -1
    at = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            if opened < 0:
                opened = at
            else:
                spans.append((opened, at + len(line)))
                opened = -1
        at += len(line) + 1
    if opened >= 0:
        spans.append((opened, len(text)))
    return spans


_SENTENCE_END: Final = re.compile(r"""[.!?]["')\]]*(?=\s)|[。！？]""")


def _break(text: str, start: int, edge: int, reach: int) -> int:
    """Where a window of *text* that starts at *start* and may run to *edge* ends:
    at the last paragraph, line or sentence end, or else word end, within *reach*
    of *edge*; *edge* itself when there is none."""
    low = max(start + 1, edge - reach)
    for mark in ("\n\n", "\n"):
        found = text.rfind(mark, low, edge)
        if found >= 0:
            return found
    sentence = [m.end() for m in _SENTENCE_END.finditer(text, low, edge)]
    if sentence:
        return sentence[-1]
    found = text.rfind(" ", low, edge)
    return found if found >= 0 else edge


def _end(
    text: str, start: int, room: int, overlap: int, spans: Sequence[tuple[int, int]]
) -> int:
    """Where the window of *text* from *start*, at most *room* long, ends.

    A break (:func:`_break`), moved out of a fenced code block: past its end when
    that is inside the window, else before it when the block fits in a later
    window with the overlap. A block too long for that is split at the break.
    """
    edge = start + room
    if edge >= len(text):
        return len(text)
    end = _break(text, start, edge, min(_BREAK_REACH, room // 4))
    for opened, closed in spans:
        if opened < end < closed:
            if closed <= edge:
                return closed
            if closed - opened <= room - overlap and opened - start >= room // 4:
                return opened
            break
    return end


def _user_head(typed: str, limit: int) -> str:
    """What a later chunk repeats of the user's words: all of them when they fit
    in *limit* characters, else the first words and ``…``."""
    flat = " ".join(typed.split())
    if len(flat) <= limit:
        return flat
    cut = flat.rfind(" ", limit * 3 // 4, limit)
    return flat[: cut if cut > 0 else limit] + "…"


def _split(typed: str, answer: str, title: str) -> tuple[list[str], bool]:
    """A turn's chunks (:func:`document_chunks`) and whether text was left out."""
    lead, typed, answer = _parts(typed, answer, title)
    text = f"{lead}User: {typed}\nmelo: {answer}"
    if len(text) <= DOC_CHARS:
        return [text], False
    typed_at = len(lead) + len("User: ")
    typed_end = typed_at + len(typed)
    answer_at = typed_end + len("\nmelo: ")
    head = f"User: {_user_head(typed, min(USER_HEAD_CHARS, DOC_CHARS // 8))}\nmelo: "
    spans = _fences(text)
    chunks: list[str] = []
    start = 0
    while True:
        if answer_at > start >= typed_end:
            start = answer_at  # not in the middle of the "melo: " label
        prefix = ""
        if start >= answer_at:
            prefix = head
        elif start >= typed_at:
            prefix = "User: "
        room = DOC_CHARS - len(prefix)
        overlap = min(CHUNK_OVERLAP, room // 8)
        end = _end(text, start, room, overlap, spans)
        chunks.append(redact_secrets(prefix + text[start:end].rstrip())[0])
        if end >= len(text) or len(chunks) == MAX_CHUNKS:
            return chunks, end < len(text)
        # The next window starts inside the overlap, at the start of a word.
        back = max(start + 1, end - overlap)
        gap = re.search(r"\s", text[back:end])
        start = back + gap.end() if gap else back


def document_chunks(typed: str, answer: str, *, title: str = "") -> list[str]:
    """What a turn's documents hold, chunk 0 first: a task's title, who said
    what, redacted, each at most :data:`DOC_CHARS` long.

    A turn that fits is one chunk. A longer one is cut at a paragraph, line or
    sentence end near each window's edge, and never inside a fenced code block
    that fits in a window. Chunk 0 is the turn's head; each later one starts
    ``User: <the first words the user typed>…`` and ``melo: `` and goes on with
    the text where the one before ended, overlapping it by about
    :data:`CHUNK_OVERLAP` characters. At most :data:`MAX_CHUNKS` come out.
    """
    return _split(typed, answer, title)[0]


def document_text(typed: str, answer: str, *, title: str = "") -> str:
    """The head of a turn's text (chunk 0 of :func:`document_chunks`)."""
    return document_chunks(typed, answer, title=title)[0]


def is_cut(typed: str, answer: str, *, title: str = "") -> bool:
    """Whether :func:`document_chunks` leaves part of the turn out, for its length."""
    return _split(typed, answer, title)[1]


def plain_text(text: str) -> str:
    """A summary's or an exchange's text: secrets left out and redacted, bounded."""
    return redact_secrets(_kept(" ".join(text.split()))[:DOC_CHARS])[0]


def typed_of(text: str) -> str:
    """What the user typed, in a chat turn's document text (:func:`document_text`);
    for a later chunk, the head of it that the chunk repeats."""
    head = text.split("\nmelo: ", 1)[0]
    return head.removeprefix("User: ") if head.startswith("User: ") else ""


# An index.db from before chunks has UNIQUE (session_id, turn), which SQLite cannot
# change in place: its rows move to a table with a chunk column, ids kept (a
# vector, a recalled source and a sleep run all name a document by its id).
_REBUILD: Final = (
    "ALTER TABLE documents RENAME TO documents_old",
    _SCHEMA.split(";", 1)[0] + ";",
    "INSERT INTO documents (id, session_id, turn, surface, kind, title, text, "
    "created_at, vector, embed_model) SELECT id, session_id, turn, surface, kind, "
    "title, text, created_at, vector, embed_model FROM documents_old",
    "DELETE FROM sqlite_sequence WHERE name = 'documents'",
    "UPDATE sqlite_sequence SET name = 'documents' WHERE name = 'documents_old'",
    "DROP TABLE documents_old",
    "CREATE INDEX IF NOT EXISTS documents_session ON documents(session_id)",
)


async def _add_chunks(conn: aiosqlite.Connection) -> None:
    """Move an index.db made before chunks to the table with a ``chunk`` column."""
    async with conn.execute("PRAGMA table_info(documents)") as cur:
        if any(row[1] == "chunk" for row in await cur.fetchall()):
            return
    await conn.execute("BEGIN IMMEDIATE")
    try:
        async with conn.execute("PRAGMA table_info(documents)") as cur:
            if any(row[1] == "chunk" for row in await cur.fetchall()):
                await conn.rollback()  # another process moved it first
                return
        for statement in _REBUILD:
            await conn.execute(statement)
        await conn.commit()
    except BaseException:
        await conn.rollback()
        raise


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
                    await _add_chunks(conn)
                except BaseException:
                    with contextlib.suppress(Exception):
                        await conn.close()
                    raise
                self._conn = conn
            return self._conn

    async def add(self, doc: Document) -> None:
        """Store *doc*, in place of the documents of the same turn."""
        await self.add_many([doc])

    async def add_many(self, docs: Sequence[Document]) -> None:
        """Store *docs*, the chunks of one or more turns, each turn's in place of
        every document it had (a turn that is now shorter keeps no old chunk)."""
        conn = await self._ready()
        try:
            for session_id, turn in {(d.session_id, d.turn) for d in docs}:
                await conn.execute(
                    "DELETE FROM documents WHERE session_id = ? AND turn = ?",
                    (session_id, turn),
                )
            for doc in docs:
                vector = (
                    None if doc.vector is None else np.asarray(doc.vector, np.float32)
                )
                await conn.execute(
                    "INSERT INTO documents (session_id, turn, surface, kind, "
                    "title, text, created_at, vector, embed_model, chunk) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        doc.chunk,
                    ),
                )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise

    async def next_special_turn(self, session_id: str) -> int:
        """The next free turn number below 0 of a session, for an exchange
        (a summary uses :data:`SUMMARY_TURN`; a chat's turns start at 0)."""
        conn = await self._ready()
        async with conn.execute(
            "SELECT MIN(turn) FROM documents WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        low = int(row[0]) if row and row[0] is not None else 0
        return min(low, SUMMARY_TURN) - 1

    async def ids(self) -> set[int]:
        conn = await self._ready()
        async with conn.execute("SELECT id FROM documents") as cur:
            return {int(row[0]) for row in await cur.fetchall()}

    async def fetch(self, ids: Iterable[int]) -> list[Document]:
        """The documents with these ids (in batches; SQLite caps the parameters)."""
        conn = await self._ready()
        wanted = sorted(ids)
        out: list[Document] = []
        for start in range(0, len(wanted), 500):
            batch = wanted[start : start + 500]
            marks = ", ".join("?" * len(batch))
            async with conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE id IN ({marks})", batch
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
                            chunk=int(row[10]),
                        )
                    )
        return out

    async def count(self) -> int:
        """How many turns are indexed (a turn's chunks count once)."""
        conn = await self._ready()
        async with conn.execute(
            "SELECT COUNT(*) FROM documents WHERE chunk = 0"
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def delete(self, session_id: str, turn: int | None = None) -> int:
        """Delete a session's documents, every chunk of them (one turn's with
        *turn*); returns how many turns."""
        conn = await self._ready()
        where = " WHERE session_id = ?"
        args: list[object] = [session_id]
        if turn is not None:
            where += " AND turn = ?"
            args.append(turn)
        async with conn.execute(
            f"SELECT COUNT(*) FROM documents{where} AND chunk = 0", args
        ) as cur:
            row = await cur.fetchone()
        async with secure_deletes(conn):
            await conn.execute(f"DELETE FROM documents{where}", args)
            await conn.execute(
                "DELETE FROM sleep_processed WHERE session_id = ?"
                + (" AND turn = ?" if turn is not None else ""),
                args,
            )
            await conn.commit()
        return int(row[0]) if row else 0

    async def forget_all(self, *, mark: bool = True) -> int:
        """Delete every document; returns how many turns. With *mark*, a turn that
        ended before now is not indexed again (:meth:`forgotten_at`)."""
        conn = await self._ready()
        turns = await self.count()
        async with secure_deletes(conn):
            await conn.execute("DELETE FROM documents")
            if mark:
                await conn.execute("DELETE FROM sleep_runs")
                await conn.execute("DELETE FROM sleep_processed")
                await conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('forgotten_at', ?)",
                    (time.time(),),
                )
            await conn.commit()
        await checkpoint_wal(conn)
        return turns

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

    async def processed_since(self, since: float) -> set[tuple[str, int]]:
        """Processed sleep turns; prune expired ids without retaining any text."""
        conn = await self._ready()
        await conn.execute("DELETE FROM sleep_processed WHERE finished < ?", (since,))
        await conn.commit()
        async with conn.execute(
            "SELECT session_id, turn FROM sleep_processed "
            "WHERE note_id = 0 AND completed = 1"
        ) as cur:
            return {(str(row[0]), int(row[1])) for row in await cur.fetchall()}

    async def work_done(self, session_id: str, turn: int, note_id: int) -> bool:
        """A completed per-note rewrite, independent of ordinary turn replay."""
        conn = await self._ready()
        async with conn.execute(
            "SELECT completed FROM sleep_processed "
            "WHERE session_id = ? AND turn = ? AND note_id = ?",
            (session_id, turn, note_id),
        ) as cur:
            row = await cur.fetchone()
        return bool(row and row[0])

    async def failed_attempt(
        self, session_id: str, turn: int, finished: float, note_id: int = 0
    ) -> int:
        """Count an unreadable reply/timeout without completing its work."""
        conn = await self._ready()
        await conn.execute(
            "INSERT INTO sleep_processed "
            "(session_id, turn, note_id, finished, failures, completed) "
            "VALUES (?, ?, ?, ?, 1, 0) ON CONFLICT(session_id, turn, note_id) "
            "DO UPDATE SET failures = failures + 1",
            (session_id, turn, note_id, finished),
        )
        await conn.commit()
        async with conn.execute(
            "SELECT failures FROM sleep_processed "
            "WHERE session_id = ? AND turn = ? AND note_id = ?",
            (session_id, turn, note_id),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        return int(row[0])

    async def mark_processed(
        self, session_id: str, turn: int, finished: float, note_id: int = 0
    ) -> None:
        conn = await self._ready()
        await conn.execute(
            "INSERT OR REPLACE INTO sleep_processed "
            "(session_id, turn, note_id, finished) VALUES (?, ?, ?, ?)",
            (session_id, turn, note_id, finished),
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
