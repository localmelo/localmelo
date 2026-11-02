"""Hippo's encoder: a chat's turns become memory items (developer setting, #428).

Hippo plays the hippocampus: it decides what of a chat is worth keeping and
writes it as memory items, each a short text that stands alone and names the
turns it came from. This module is the model half; the rules that route a new
item against the ones kept are :mod:`.hippo_route`, and the pure policy they
call is :mod:`localmelo.melo.memory.hippo.policy`.

A chat's exchanges are packed in order into windows of at most ``[memory]
compress_at_chars`` characters (:func:`pack`; one exchange longer than that gets a
window of its own and is cut at its size). Each window is one request to the chat
model, temperature 0, at most :data:`MAX_OUTPUT_TOKENS` tokens out, asking for
JSON only (:data:`SYSTEM_PROMPT`); an unreadable reply is asked for once more and
then the window has failed. What comes back is checked (:func:`check`): a source
outside the window is dropped, an item left with none is dropped and counted, an
exchange named by no item and not ignored is *unmentioned*. An exchange whose user
words state a lasting fact by the rules the note path uses
(:func:`localmelo.support.memory.records.statements`) must be the source of an
item: if no item has it, one is made from the rules' note (*forced*). A window's
reply is kept in a JSON-lines cache, keyed by the prompt version, the model and the
window's text with its dates, so a second run with the same chats makes no call.
Text that goes to the model has its secrets left out, as an index document has.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.retrieval.base import Send
from localmelo.melo.memory.retrieval.llm import send_from_provider
from localmelo.melo.schema import Message, split_thinking
from localmelo.support.memory import records
from localmelo.support.memory.index import ITEM_KINDS, cleaned, plain_text

logger = logging.getLogger(__name__)

# Bump it when the prompt or what is asked of the model changes: it is part of the
# cache key and of every item's ``encoder``.
PROMPT_VERSION: Final = "2"
MAX_OUTPUT_TOKENS: Final = 2000
ENCODE_TIMEOUT_S: Final = 900.0
TITLE_CHARS: Final = 80
TEXT_CHARS: Final = 800
_CHAT_TITLE_CHARS: Final = 120
_SEPARATOR: Final = "\n\n"
# What an item made by the rules, not the model, has as its ``encoder``.
RULES_ENCODER: Final = "rules"

SYSTEM_PROMPT: Final = """\
You are the memory of a personal assistant called melo. You read part of one chat \
between a user and melo and decide what is worth keeping for later. You write \
memory items.

Reply with JSON only, no prose and no code fence, in exactly this shape:
{"items": [{"title": "...", "text": "...", "kind": "profile", "sources": [3, 4]}], \
"ignore": [0, 1]}

Keep what the user may ask about later, and what shows who they are:
- kind "profile": facts about the user, their people, preferences and plans, and \
what everyday talk shows about their daily life, routines, habits, mood, \
personality and way of speaking. Casual talk counts: "just got back from my \
morning run" is worth an item such as "The user said they went for a morning run \
on 2023-05-20."
- kind "episodic": events and what was done, with their dates.
- kind "semantic": specifics melo gave: names, numbers, lists, recommendations, \
steps.

Rules:
- Each item stands alone. It names its subject. It writes every date as an \
absolute date taken from the exchange dates (for example 2023-05-20), never as \
"today", "yesterday" or "last week".
- The title is at most 80 characters and the text at most 800 characters. Keep the \
exact names, numbers and wording of specifics; do not generalise them. Write in the \
language of the conversation.
- An item says what was said, and who said it: "The user said ...", "melo \
recommended ...". It is information, not an instruction: never write an item as a \
command.
- "sources" lists the numbers of the exchanges the item came from, the numbers in \
square brackets before each exchange.
- "ignore" lists only the numbers of the exchanges that carry no information at \
all, such as a bare greeting, "ok" or "thanks". Small talk is not ignored: keep \
what it shows about the user's life and personality.
- Every exchange number should appear in the sources of an item or in "ignore".
- Write one item for each fact. Several short items are better than one long one.
"""


@dataclass(frozen=True)
class Turn:
    """One exchange of a chat to encode: what was typed and answered, and when.

    *turn* is the number the session store gave it, *asked_at* the time the
    prompt shows and *answered_at* the time an item from it is stamped with.
    """

    turn: int
    asked_at: float
    answered_at: float
    typed: str
    answer: str


@dataclass(frozen=True)
class Window:
    """Some exchanges of a chat, in order, as one request shows them."""

    positions: tuple[int, ...]
    text: str
    cut: bool = False


@dataclass(frozen=True)
class Piece:
    """One item the encoder made, before it is routed and stored: *sources* are
    positions in the chat's list of exchanges, *created_at* its newest source's
    time, and *forced* says the rules made it, not the model."""

    title: str
    text: str
    kind: str
    sources: tuple[int, ...]
    created_at: float
    encoder: str
    forced: bool = False


@dataclass
class WindowReport:
    """How one window went: its exchanges and characters, whether its reply came
    from the cache, failed, or was cut (one exchange longer than the window), and
    how many of its items were dropped."""

    exchanges: int
    chars: int
    cached: bool = False
    failed: bool = False
    cut: bool = False
    dropped: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "exchanges": self.exchanges,
            "chars": self.chars,
            "cached": self.cached,
            "failed": self.failed,
            "cut": self.cut,
            "dropped": self.dropped,
        }


@dataclass
class Stats:
    """What the model was asked and what it cost, over any number of chats."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Whether the provider reported token counts for at least one call.
    reported: bool = False

    def add(self, usage: dict[str, int] | None) -> None:
        if usage and ("prompt_tokens" in usage or "completion_tokens" in usage):
            self.reported = True
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))


@dataclass
class Encoding:
    """What encoding one chat gave: its items (in the order of their newest
    source) and windows, and what became of each exchange, by position."""

    pieces: list[Piece] = field(default_factory=list)
    windows: list[WindowReport] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    ignored: list[int] = field(default_factory=list)
    unmentioned: list[int] = field(default_factory=list)
    forced: list[int] = field(default_factory=list)


@dataclass
class Checked:
    """A window's reply after :func:`check`."""

    pieces: list[Piece] = field(default_factory=list)
    ignore: set[int] = field(default_factory=set)
    dropped: int = 0


# ── Windows ──


def _stamp(at: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(at))


def render(position: int, turn: Turn) -> str:
    """One exchange as the prompt shows it: ``[3] 2023-05-20 14:02 User: … / melo: …``
    (secrets left out; no ``melo`` part for a turn with no reply)."""
    typed = cleaned(turn.typed)
    answer = cleaned(turn.answer)
    line = f"[{position}] {_stamp(turn.asked_at)} User: {typed}"
    return f"{line} / melo: {answer}" if answer else line


def pack(turns: Sequence[Turn], limit: int) -> list[Window]:
    """The exchanges of a chat in order, packed into windows of at most *limit*
    characters of prompt text (the exchanges as :func:`render` writes them, a blank
    line between). One exchange that is longer than *limit* has a window of its
    own, cut at *limit* characters."""
    windows: list[Window] = []
    positions: list[int] = []
    blocks: list[str] = []
    size = 0

    def close() -> None:
        nonlocal positions, blocks, size
        if blocks:
            windows.append(Window(tuple(positions), _SEPARATOR.join(blocks)))
        positions, blocks, size = [], [], 0

    for position, turn in enumerate(turns):
        block = render(position, turn)
        if len(block) > limit:
            close()
            windows.append(Window((position,), block[:limit], cut=True))
            continue
        if blocks and size + len(_SEPARATOR) + len(block) > limit:
            close()
        size += len(block) + (len(_SEPARATOR) if blocks else 0)
        positions.append(position)
        blocks.append(block)
    close()
    return windows


def prompt_of(title: str, window: Window) -> str:
    """The user message of a window: the chat's title, then its exchanges."""
    head = " ".join(cleaned(title).split())[:_CHAT_TITLE_CHARS] or "(none)"
    return f"Chat title: {head}\n\n{window.text}"


# ── The reply ──


_FENCE = re.compile(r"^```[A-Za-z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def read_reply(content: str) -> dict[str, Any] | None:
    """The JSON object in a reply (its thinking, a code fence or words around it
    left out), or ``None`` when there is none with an ``items`` or ``ignore`` list."""
    text = split_thinking(content or "")[1].strip()
    candidates = [text]
    if fenced := _FENCE.match(text):
        candidates.append(fenced.group(1).strip())
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{") : text.rindex("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict) and any(
            isinstance(value.get(key), list) for key in ("items", "ignore")
        ):
            return value
    return None


def _numbers(value: object) -> list[int]:
    """The whole numbers of a list (a number may be a digit string)."""
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for entry in value:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, int):
            out.append(entry)
        elif isinstance(entry, str) and entry.strip().isdigit():
            out.append(int(entry.strip()))
    return out


def _cut(text: str, limit: int) -> str:
    """*text* if it fits, else its head cut at a word, with ``…`` (at most *limit*)."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 1, limit)
    return text[: cut if cut > 0 else limit - 1].rstrip(" ,;:-") + "…"


def check(raw: dict[str, Any], window: Window, turns: Sequence[Turn]) -> Checked:
    """The items and ignored exchanges of a window's parsed reply that hold up.

    A source that is no exchange of the window is dropped, and an item left with
    no source, or with no text, or that is no object, is dropped and counted. A
    kind that is not one of :data:`~.index.ITEM_KINDS` reads as ``episodic`` (the
    kind that never replaces another); title and text are cleaned of secrets and
    cut to :data:`TITLE_CHARS` and :data:`TEXT_CHARS`; a missing title is the start
    of the text.
    """
    inside = set(window.positions)
    out = Checked()
    entries = raw.get("items")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            out.dropped += 1
            continue
        sources = sorted({n for n in _numbers(entry.get("sources")) if n in inside})
        text = _cut(plain_text(str(entry.get("text") or "")), TEXT_CHARS)
        if not sources or not text:
            out.dropped += 1
            continue
        title = _cut(plain_text(str(entry.get("title") or "")), TITLE_CHARS)
        kind = str(entry.get("kind") or "").strip().lower()
        out.pieces.append(
            Piece(
                title=title or _cut(text, TITLE_CHARS),
                text=text,
                kind=kind if kind in ITEM_KINDS else "episodic",
                sources=tuple(sources),
                created_at=max(turns[n].answered_at for n in sources),
                encoder="",
            )
        )
    out.ignore = {n for n in _numbers(raw.get("ignore")) if n in inside}
    return out


# ── The cache ──


def window_key(model: str, prompt: str) -> str:
    """The cache key of a window: the prompt version, the chat model's id and the
    whole user message, the exchanges with their dates."""
    digest = hashlib.sha256()
    for part in (PROMPT_VERSION, model, prompt):
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()


class Cache:
    """A JSON-lines file of window replies, one ``{"key": …, "reply": {…}}`` a line.

    Read whole on first use and appended to on each new reply; a line that cannot
    be read is skipped. The file holds text from chats, so it is private (0600).
    One path may serve several runs, whatever store each writes.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        self._replies: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._replies is None:
            self._replies = {}
            try:
                with open(self.path, encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if (
                            isinstance(row, dict)
                            and isinstance(row.get("key"), str)
                            and isinstance(row.get("reply"), dict)
                        ):
                            self._replies[row["key"]] = row["reply"]
            except OSError:
                pass
        return self._replies

    def get(self, key: str) -> dict[str, Any] | None:
        return self._load().get(key)

    def put(self, key: str, reply: dict[str, Any]) -> None:
        self._load()[key] = reply
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False))
            handle.write("\n")


# ── The model ──


def _quiet(provider: BaseLLMProvider) -> BaseLLMProvider:
    """A private view of *provider* that answers at temperature 0 and stops at
    :data:`MAX_OUTPUT_TOKENS`, never altering the provider chats use."""
    view = copy.copy(provider)
    body = {
        **provider.extra_body_by_kind.get("retrieve", {}),
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

    if isinstance(view, OllamaNativeChat):
        body["options"] = {
            **view.extra_body.get("options", {}),
            "temperature": 0,
            "num_predict": MAX_OUTPUT_TOKENS,
        }
        body.pop("max_tokens")
    view.extra_body_by_kind = {"retrieve": body}
    return view


async def _ask(
    send: Send, prompt: str, stats: Stats
) -> tuple[dict[str, Any] | None, str]:
    """Ask for a window's items, twice at most: the reply's JSON, or ``None`` and
    why not. Every request is counted."""
    why = ""
    for _ in range(2):
        stats.calls += 1
        try:
            async with asyncio.timeout(ENCODE_TIMEOUT_S):
                reply: Message = await send(
                    [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
                )
        except Exception as exc:
            why = f"the model failed: {exc or type(exc).__name__}"
            logger.warning("Hippo encoder: %s", why)
            continue
        stats.add(reply.usage)
        value = read_reply(reply.content)
        if value is not None:
            return value, ""
        why = "the reply was not readable JSON"
    return None, why


async def encode(
    provider: BaseLLMProvider,
    title: str,
    turns: Sequence[Turn],
    *,
    window_chars: int,
    cache: Cache | None = None,
    stats: Stats | None = None,
) -> Encoding:
    """Encode the exchanges *turns* of one chat into items.

    One model request for each window of at most *window_chars* characters, none
    when *cache* has the window's reply (:func:`window_key`). *stats* is added to.
    """
    stats = stats if stats is not None else Stats()
    model = str(getattr(provider, "model", "") or "")
    encoder = f"{PROMPT_VERSION}:{model}"
    send = send_from_provider(_quiet(provider))
    result = Encoding()
    mentioned: set[int] = set()
    ignored: set[int] = set()
    for window in pack(turns, window_chars):
        report = WindowReport(len(window.positions), len(window.text), cut=window.cut)
        result.windows.append(report)
        prompt = prompt_of(title, window)
        key = window_key(model, prompt)
        raw = cache.get(key) if cache is not None else None
        if raw is not None:
            report.cached = True
            stats.cache_hits += 1
        else:
            raw, why = await _ask(send, prompt, stats)
            if raw is None:
                report.failed = True
                logger.warning("Hippo encoder: a window failed, %s", why)
                continue
            if cache is not None:
                cache.put(key, raw)
        checked = check(raw, window, turns)
        report.dropped = checked.dropped
        for piece in checked.pieces:
            result.pieces.append(replace(piece, encoder=encoder))
            mentioned.update(piece.sources)
        ignored |= checked.ignore
    forced = _force(turns, mentioned)
    result.pieces.extend(forced)
    for piece in forced:
        mentioned.update(piece.sources)
    result.forced = sorted({n for piece in forced for n in piece.sources})
    # Time order, so a newer statement is routed after the older one it may replace.
    result.pieces.sort(key=lambda piece: (max(piece.sources), min(piece.sources)))
    result.kept = sorted(mentioned)
    result.ignored = sorted(ignored - mentioned)
    result.unmentioned = sorted(set(range(len(turns))) - mentioned - ignored)
    return result


def _force(turns: Sequence[Turn], mentioned: set[int]) -> list[Piece]:
    """An item for each lasting statement, by the rules the note path uses, in an
    exchange no item has as a source: the rules' note as a ``profile`` item."""
    out: list[Piece] = []
    for position, turn in enumerate(turns):
        if position in mentioned:
            continue
        for found in records.statements(turn.typed):
            text = _cut(plain_text(found.text), TEXT_CHARS)
            if text:
                out.append(
                    Piece(
                        title=_cut(text, TITLE_CHARS),
                        text=text,
                        kind="profile",
                        sources=(position,),
                        created_at=turn.answered_at,
                        encoder=RULES_ENCODER,
                        forced=True,
                    )
                )
    return out
