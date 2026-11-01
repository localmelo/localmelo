"""Load past chats from a JSON file as if they had happened on their dates.

A developer tool for memory tests (``melo memory ingest FILE``): a file of
timestamped chats goes through the real session store as ordinary chat sessions,
each turn stamped with its own time, and the turns are indexed and their notes
kept as the chat path does. No chat model is ever called. The file::

    {"sessions": [
      {"id": "s1", "title": "optional", "date": "2023-05-20T13:40:00Z",
       "turns": [{"role": "user", "text": "...", "date": "optional"},
                 {"role": "assistant", "text": "..."}]}]}

A date is ISO 8601 (no zone means local time) or epoch seconds
(:func:`localmelo.melo.clock.parse`). A message without its own date takes its
session's date plus one second for each message before it, so the order holds.
A session without a date takes its first message's.

Sessions are written oldest first (the file's order breaks a tie), so a newer
statement replaces an older note as it would have. Messages pair into chat
turns: a user message and the assistant message after it make one. A user
message with no reply, because another user message or the end follows, is a
turn with an empty answer; it is stored and indexed all the same, though
``melo memory reindex`` leaves a turn with no answer out. A further assistant
message right after a reply continues that reply. An assistant message with no
user message before it, and a message with no text, are skipped and counted.
A user message over :data:`~localmelo.melo.checker.validators.MAX_QUERY_LEN`
characters is cut there and counted as ``trimmed``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from localmelo.melo import clock
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import is_cut
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    process_holder,
)

MAX_FILE_BYTES: Final = 500 * 1024 * 1024
NOTES = ("rules", "none")
ROLES = ("user", "assistant")


class IngestError(ValueError):
    """The file cannot be loaded; the message says why, in one line."""


class Clock:
    """The session store's clock: the instant the ingest sets, real time before."""

    def __init__(self) -> None:
        self.at = time.time()

    def __call__(self) -> float:
        return self.at


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    at: float


@dataclass(frozen=True)
class Chat:
    label: str
    title: str
    at: float
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Exchange:
    """One chat turn: what was typed and answered, and when each happened."""

    typed: str
    answer: str
    asked_at: float
    answered_at: float


@dataclass
class Report:
    sessions: int = 0
    turns: int = 0
    docs: int = 0
    notes: int = 0
    embedded_chars: int = 0
    cut: int = 0
    skipped: int = 0
    trimmed: int = 0
    seconds: float = 0.0
    model: str = ""
    error: str = ""
    ids: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "turns": self.turns,
            "docs": self.docs,
            "notes": self.notes,
            "embedded_chars": self.embedded_chars,
            "cut": self.cut,
            "skipped": self.skipped,
            "trimmed": self.trimmed,
            "seconds": round(self.seconds, 3),
            "model": self.model,
            "error": self.error,
            "ids": self.ids,
        }

    def summary(self) -> str:
        how = f"with {self.model}" if self.model else "by words only"
        line = (
            f"Ingested {self.sessions} session{'s' if self.sessions != 1 else ''}, "
            f"{self.turns} turn{'s' if self.turns != 1 else ''}, "
            f"{self.docs} document{'s' if self.docs != 1 else ''} ({how}) and "
            f"{self.notes} note{'s' if self.notes != 1 else ''} "
            f"in {self.seconds:.1f} s."
        )
        return (
            f"{line} The embedding model failed: {self.error}" if self.error else line
        )


def _date(value: object, where: str) -> float:
    seconds = clock.parse(value)
    if seconds is None:
        raise IngestError(
            f"{where}: {value!r} is not an ISO 8601 date or epoch seconds"
        )
    return seconds


def _text(value: object) -> str:
    """*value* as text the session store can hold (a lone surrogate cannot be)."""
    return (
        value.encode("utf-8", "replace").decode("utf-8")
        if isinstance(value, str)
        else ""
    )


def parse(data: object) -> list[Chat]:
    """The chats of a decoded file, or :class:`IngestError`."""
    sessions = data.get("sessions") if isinstance(data, Mapping) else None
    if not isinstance(sessions, list):
        raise IngestError('the file must be an object with a "sessions" list')
    chats: list[Chat] = []
    seen: set[str] = set()
    for number, item in enumerate(sessions, 1):
        if not isinstance(item, Mapping):
            raise IngestError(f"session {number}: must be an object")
        label = item.get("id")
        label = str(label) if label not in (None, "") else f"session-{number}"
        where = f"session {label}"
        if label in seen:
            raise IngestError(f"{where}: the id is used twice")
        seen.add(label)
        turns = item.get("turns")
        if not isinstance(turns, list):
            raise IngestError(f'{where}: needs a "turns" list')
        start = item.get("date")
        base = None if start is None else _date(start, f"{where} date")
        messages: list[Message] = []
        for index, turn in enumerate(turns):
            at_text = f"{where}, turn {index + 1}"
            if not isinstance(turn, Mapping):
                raise IngestError(f"{at_text}: must be an object")
            if turn.get("role") not in ROLES:
                raise IngestError(f'{at_text}: role must be "user" or "assistant"')
            if not isinstance(turn.get("text"), str):
                raise IngestError(f"{at_text}: text must be a string")
            own = turn.get("date")
            if own is not None:
                stamp = _date(own, f"{at_text} date")
                base = stamp if base is None else base
            elif base is None:
                raise IngestError(f"{where}: has no date")
            else:
                stamp = base + index
            messages.append(Message(str(turn["role"]), _text(turn["text"]), stamp))
        if base is None:
            raise IngestError(f"{where}: has no date")
        title = item.get("title")
        chats.append(
            Chat(
                label,
                _text(title).strip() if isinstance(title, str) else "",
                base,
                tuple(messages),
            )
        )
    return chats


def load(path: str) -> list[Chat]:
    """Read and check the file at *path*; :class:`IngestError` on any problem."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise IngestError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if size > MAX_FILE_BYTES:
        raise IngestError(
            f"{path} is {size // (1024 * 1024)} MB; the limit is "
            f"{MAX_FILE_BYTES // (1024 * 1024)} MB. Split it into several files."
        )
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise IngestError(f"cannot read {path}: {exc}") from exc
    return parse(data)


def exchanges(messages: Sequence[Message]) -> tuple[list[Exchange], int]:
    """The chat turns of *messages* in order, and how many messages were skipped."""
    out: list[Exchange] = []
    skipped = 0
    asked: Message | None = None  # a user message waiting for its reply
    for message in messages:
        if not message.text.strip():
            skipped += 1
        elif message.role == "user":
            if asked is not None:
                out.append(Exchange(asked.text, "", asked.at, asked.at))
            asked = message
        elif asked is not None:
            out.append(
                Exchange(asked.text, message.text, asked.at, max(message.at, asked.at))
            )
            asked = None
        elif out:  # a second assistant message continues the reply before it
            last = out[-1]
            out[-1] = Exchange(
                last.typed,
                f"{last.answer}\n\n{message.text}",
                last.asked_at,
                max(message.at, last.answered_at),
            )
        else:
            skipped += 1
    if asked is not None:
        out.append(Exchange(asked.text, "", asked.at, asked.at))
    return out, skipped


async def run(
    memory: ChatMemory,
    chats: SessionStore,
    pin: Clock,
    found: Sequence[Chat],
    *,
    notes: str = "rules",
) -> Report:
    """Write *found* to *chats* (opened with *pin* as its clock), index it and keep its notes.

    *notes* is ``rules`` (the notes the rule path keeps from each user message,
    no model call) or ``none``. Never calls the chat model.
    """
    if notes not in NOTES:
        raise IngestError(f"notes must be one of {', '.join(NOTES)}, got {notes!r}")
    started = time.perf_counter()
    report = Report()
    holder = process_holder("gateway")
    turns: list[tuple[dict[str, Any], int, str, str, float]] = []
    for chat in sorted(found, key=lambda c: c.at):  # stable: the file's order ties
        pairs, skipped = exchanges(chat.messages)
        report.skipped += skipped
        session_id = ""
        source: dict[str, Any] = {}
        for exchange in pairs:
            typed = exchange.typed
            if len(typed) > MAX_QUERY_LEN:
                typed = typed[:MAX_QUERY_LEN]
                report.trimmed += 1
            pin.at = exchange.asked_at
            info = await chats.begin_turn(
                session_id or None,
                typed,
                holder=holder,
                new_session=None if session_id else NewSession(),
            )
            if not session_id:
                session_id = info.session_id
                report.sessions += 1
                report.ids[chat.label] = session_id
                if chat.title:
                    await chats.rename_session(session_id, chat.title)
                session = await chats.get_session(session_id)
                source = {
                    "session_id": session_id,
                    "kind": "chat",
                    "title": session.title if session is not None else chat.title,
                }
            pin.at = exchange.answered_at
            await chats.finish_turn(
                session_id,
                info.turn,
                holder=holder,
                status="completed",
                cause="run",
                result=None,
                answer=exchange.answer,
            )
            report.turns += 1
            report.cut += int(is_cut(typed, exchange.answer))
            turns.append(
                (source, info.turn, typed, exchange.answer, exchange.answered_at)
            )
            if notes == "rules":
                report.notes += await memory.keep_said(
                    chats, info, typed, exchange.asked_at
                )
    report.docs, report.embedded_chars = await memory.index_turns(turns)
    if notes == "rules":
        await memory.reindex(None)  # a vector for each note, when a model is set
    report.model = memory.embedding()[1]
    report.error = memory.last_error
    report.seconds = time.perf_counter() - started
    return report
