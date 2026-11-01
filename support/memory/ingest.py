"""Load past chats from a JSON file as if they had happened on their dates.

A developer tool for memory tests (``melo memory ingest FILE``): a file of
timestamped chats goes through the real session store as ordinary chat sessions,
each turn stamped with its own time, and the turns are indexed and their notes
kept as the chat path does. The chat model is not called, unless the developer
option ``-X ingest=compress`` asks for it (below). The file::

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

With ``ingest=compress`` (:data:`MODES`) everything above happens first, so one ingest
serves every way of recalling, and then each chat is encoded into memory items by the
chat model (:meth:`~localmelo.support.memory.service.ChatMemory.compress_chat`):
windows of the chat, one model request each, then the items routed against the ones
kept, the chats oldest first as they were written. The report says, for each chat by
its id in the file, what became of every exchange. It does not check the GPU lock:
it is an explicit developer command, for whoever holds the GPU.
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
from localmelo.support.memory import ChatMemory, hippo_encode
from localmelo.support.memory.hippo_route import Router
from localmelo.support.memory.index import is_cut
from localmelo.support.memory.service import Compressed
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    process_holder,
)

MAX_FILE_BYTES: Final = 500 * 1024 * 1024
NOTES = ("rules", "none")
# What ingest does with the chats besides storing and indexing the turns: nothing
# (``turns``), or encode each chat into memory items with the chat model
# (``compress``).
MODES = ("turns", "compress")
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
    # Only with ``ingest=compress``: the model's totals and, for each chat by its id
    # in the file, windows, gate, items and routes (:func:`_chat_report`).
    compress: dict[str, dict[str, Any]] | None = None
    stats: hippo_encode.Stats = field(default_factory=hippo_encode.Stats)
    items: int = 0

    def to_json(self) -> dict[str, Any]:
        data = self._turns_json()
        if self.compress is not None:
            data["ingest"] = "compress"
            data["model_calls"] = self.stats.calls
            data["cache_hits"] = self.stats.cache_hits
            if self.stats.reported:
                data["input_tokens"] = self.stats.prompt_tokens
                data["output_tokens"] = self.stats.completion_tokens
            data["compress"] = self.compress
        return data

    def _turns_json(self) -> dict[str, Any]:
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
        if self.compress is not None:
            line += (
                f" Encoded {self.items} item{'s' if self.items != 1 else ''} "
                f"from {len(self.compress)} chat{'s' if len(self.compress) != 1 else ''} "
                f"({self.stats.calls} model call{'s' if self.stats.calls != 1 else ''}, "
                f"{self.stats.cache_hits} from the cache)."
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
    seen: dict[str, Chat] = {}
    for number, item in enumerate(sessions, 1):
        if not isinstance(item, Mapping):
            raise IngestError(f"session {number}: must be an object")
        label = item.get("id")
        label = str(label) if label not in (None, "") else f"session-{number}"
        where = f"session {label}"
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
        chat = Chat(
            label,
            _text(title).strip() if isinstance(title, str) else "",
            base,
            tuple(messages),
        )
        # A session the file repeats word for word is read once; the same id
        # with other turns, date or title is a mistake the file must fix.
        if label in seen:
            if seen[label] == chat:
                continue
            raise IngestError(f"{where}: the id is used twice, with different turns")
        seen[label] = chat
        chats.append(chat)
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


def _chat_report(label: str, encoded: Compressed) -> dict[str, Any]:
    """One chat's part of the compress report, by exchange index (0-based in the
    chat as the file gives it) and file id *label*.

    ``windows`` (``exchanges``, ``chars``, ``cached``, ``failed``, ``cut``,
    ``dropped``), ``gate`` (``kept``, ``ignored``, ``unmentioned``, ``forced``:
    exchange indexes), ``items`` (each stored: ``id``, ``title``, ``text``,
    ``kind``, ``sources`` as ``[file id, exchange index]``) and ``routes`` (each routed item:
    ``item`` (its id, ``null`` when ignored), ``action``, ``target``, ``reason``,
    ``title``).
    """
    encoding = encoded.encoding
    return {
        "windows": [window.to_json() for window in encoding.windows],
        "gate": {
            "kept": encoding.kept,
            "ignored": encoding.ignored,
            "unmentioned": encoding.unmentioned,
            "forced": encoding.forced,
        },
        "items": [
            {
                "id": route.item,
                "title": piece.title,
                "text": piece.text,
                "kind": piece.kind,
                "sources": [[label, n] for n in piece.sources],
            }
            for piece, route in encoded.routed
            if route.item is not None
        ],
        "routes": [
            {**route.to_json(), "title": piece.title} for piece, route in encoded.routed
        ],
    }


async def run(
    memory: ChatMemory,
    chats: SessionStore,
    pin: Clock,
    found: Sequence[Chat],
    *,
    notes: str = "rules",
    mode: str = "turns",
) -> Report:
    """Write *found* to *chats* (opened with *pin* as its clock), index it and keep its notes.

    *notes* is ``rules`` (the notes the rule path keeps from each user message,
    no model call) or ``none``. *mode* is ``turns`` (the chat model is never
    called) or ``compress``: after all that, each chat is encoded into memory items
    and the items are routed (see the module's text); it needs a chat model, and
    :class:`IngestError` says so before anything is written when there is none.
    """
    if notes not in NOTES:
        raise IngestError(f"notes must be one of {', '.join(NOTES)}, got {notes!r}")
    if mode not in MODES:
        raise IngestError(f"ingest must be one of {', '.join(MODES)}, got {mode!r}")
    if mode == "compress" and memory.chat_provider() is None:
        raise IngestError(
            "ingest=compress needs a chat model: "
            f"{memory.last_error or 'none is set up'}"
        )
    started = time.perf_counter()
    report = Report()
    holder = process_holder("gateway")
    turns: list[tuple[dict[str, Any], int, str, str, float]] = []
    # Each chat of the file, for compress: its id, store id, title and exchanges.
    encode: list[tuple[str, str, str, list[hippo_encode.Turn]]] = []
    for chat in sorted(found, key=lambda c: c.at):  # stable: the file's order ties
        pairs, skipped = exchanges(chat.messages)
        report.skipped += skipped
        session_id = ""
        source: dict[str, Any] = {}
        mine: list[hippo_encode.Turn] = []
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
            mine.append(
                hippo_encode.Turn(
                    info.turn,
                    exchange.asked_at,
                    exchange.answered_at,
                    typed,
                    exchange.answer,
                )
            )
            if notes == "rules":
                report.notes += await memory.keep_said(
                    chats, info, typed, exchange.asked_at
                )
        if session_id:
            encode.append((chat.label, session_id, str(source["title"]), mine))
    report.docs, report.embedded_chars = await memory.index_turns(turns)
    if notes == "rules":
        await memory.reindex(None)  # a vector for each note, when a model is set
    if mode == "compress":
        report.compress = {}
        router = Router(memory.stores()[1])
        cache = memory.compress_cache()
        for label, session_id, title, mine in encode:
            encoded = await memory.compress_chat(
                session_id,
                title,
                mine,
                stats=report.stats,
                router=router,
                cache=cache,
            )
            report.compress[label] = _chat_report(label, encoded)
            report.items += sum(
                1 for _, route in encoded.routed if route.item is not None
            )
    report.model = memory.embedding()[1]
    report.error = memory.last_error
    report.seconds = time.perf_counter() - started
    return report
