"""Hippo's encoder: a chat's turns become memory items (developer setting, #428).

Hippo plays the hippocampus: it decides what of a chat is worth keeping and
writes it as memory items, each a short text that stands alone and names the
turns it came from. This module is the model half; the rules that route a new
item against the ones kept are :mod:`.hippo_route`, and the pure policy they
call is :mod:`localmelo.melo.memory.hippo.policy`.

A chat's exchanges are packed in order into windows of at most
:data:`~localmelo.support.memory.service.COMPRESS_AT_CHARS` characters (:func:`pack`; one exchange longer than that is read
in up to :data:`MAX_PARTS` consecutive parts of that size, each part after the first
starting ``[n] (continued) ``). Each window is one request to the chat model,
temperature 0, at most :data:`MAX_OUTPUT_TOKENS` tokens out, asking for JSON only
(:data:`SYSTEM_PROMPT`). An unreadable reply is asked for once more and then the
window has failed, unless the reply *ran out of room* (:func:`_ran_out`: its JSON
was cut off at the output cap): asking the same prompt again at temperature 0 would
cut it at the same place, so the window is split instead, as the hippocampus
encodes an episode too big for one go at its next event boundary down: between
exchanges, then, inside one exchange, between paragraphs or sentences
(:func:`halves`). Each half is asked and cached like any window; one that also runs
out splits once more, so a window makes at most four sub-windows, and a sub-window
that fails fails alone. The encoder asks a local model with thinking off, as
step reviews do (:mod:`localmelo.support.providers.llm.thinking`). What comes back
is checked (:func:`check`): a source outside the window is dropped, an item left
with none is dropped and counted, an exchange named by no item and not ignored is
*unmentioned*. An exchange whose user words state a lasting fact by the rules the
note path uses (:func:`localmelo.support.memory.records.statements`) must be the
source of an item: if no item has it, one is made from the rules' note (*forced*).
A window's reply is kept in a JSON-lines cache, keyed by the prompt version, the
model and the window's text with its dates, so a second run with the same chats
makes no call; a window that was split is kept under a second key (:func:`split_key`),
so the run does not ask it again either. Text that goes to the model has its
secrets left out, as an index document has (:func:`render`). An item whose title or
text, as the model wrote it, holds a secret (:func:`~.index.holds_secret`) is refused
whole, not kept with a ``[left out]`` remnant, and counted in :attr:`Encoding.refused`.
A window that failed is recorded in :attr:`Encoding.failed_from`, the first exchange
a progress mark must not pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, TypeVar

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.retrieval.base import Send
from localmelo.melo.memory.retrieval.llm import send_from_provider
from localmelo.melo.schema import Message, split_thinking
from localmelo.support.memory import records
from localmelo.support.memory.index import (
    ITEM_KINDS,
    cleaned,
    holds_secret,
    plain_text,
)
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.thinking import thinking_off_fields

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# Bump it when the prompt or what is asked of the model changes: it is part of the
# cache key and of every item's ``encoder``.
PROMPT_VERSION: Final = "3"
MAX_OUTPUT_TOKENS: Final = 2000
# A reply that reports at least this many completion tokens ran out of room (a
# server may count a token or two differently from the cap it was given).
_RUN_OUT_MARGIN: Final = 16
# A window shorter than this is not split; the one that has run out fails.
MIN_SPLIT_CHARS: Final = 1500
# Levels of splitting: a window makes two halves, each of which may make two more.
MAX_SPLIT_DEPTH: Final = 2
# Most parts one very long exchange is read in; the text after them is cut.
MAX_PARTS: Final = 4
ENCODE_TIMEOUT_S: Final = 900.0
TITLE_CHARS: Final = 80
TEXT_CHARS: Final = 800
_CHAT_TITLE_CHARS: Final = 120
_SEPARATOR: Final = "\n\n"
# What follows the ``[n]`` label of every part of an exchange after the first.
_CONTINUED: Final = " (continued) "
# What an item made by the rules, not the model, has as its ``encoder``.
RULES_ENCODER: Final = "rules"
# The chat models whose items recall may read with ``[memory] recall_source = "auto"``:
# the ones a check found writing no false fact about the user. The id is compared
# exactly with the model's; a smaller model is not on the list.
CHECKED_ENCODERS: Final = frozenset({"Qwen3.8-Flash-Next-IQ3XXS-splash"})

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
    """Some exchanges of a chat, in order, as one request shows them.

    *cut* says text was left out (more than :data:`MAX_PARTS` parts of one
    exchange), *part* is ``(k, m)`` for part *k* of *m* of one long exchange,
    *starts* the offset in *text* of each exchange's block (what :func:`halves`
    splits a window of several exchanges by; not part of what a window is).
    """

    positions: tuple[int, ...]
    text: str
    cut: bool = False
    part: tuple[int, int] | None = None
    starts: tuple[int, ...] = field(default=(), compare=False, repr=False)


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
    from the cache (no request at all, not for a part of it either), failed, or was
    cut (text of one very long exchange left out), and how many of its items were
    dropped.

    *split* is the number of sub-windows asked for when the window ran out of room
    (0 when it did not; at most four, a half that split again counted by its own
    halves), *failed_parts* how many of them failed, and *part* is ``(k, m)`` for
    part *k* of *m* of one long exchange. *failed* is true only when the window
    gave nothing: it could not split, or every sub-window failed.
    """

    exchanges: int
    chars: int
    cached: bool = False
    failed: bool = False
    cut: bool = False
    dropped: int = 0
    split: int = 0
    failed_parts: int = 0
    part: tuple[int, int] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "exchanges": self.exchanges,
            "chars": self.chars,
            "cached": self.cached,
            "failed": self.failed,
            "cut": self.cut,
            "dropped": self.dropped,
            "split": self.split,
            "failed_parts": self.failed_parts,
            "part": list(self.part) if self.part is not None else None,
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
    source) and windows, and what became of each exchange, by position.

    *refused* counts the items that held a secret and are not among *pieces*.
    *failed_from* is the position of the first exchange of the first window (or
    half of one) that failed, ``None`` when none did: what the chat's progress mark
    must not pass.
    """

    pieces: list[Piece] = field(default_factory=list)
    windows: list[WindowReport] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    ignored: list[int] = field(default_factory=list)
    unmentioned: list[int] = field(default_factory=list)
    forced: list[int] = field(default_factory=list)
    refused: int = 0
    failed_from: int | None = None

    def without_secrets(self) -> Encoding:
        """This encoding without the pieces that hold a secret
        (:func:`~.index.holds_secret`), added to *refused*, with *kept*, *forced* and
        *unmentioned* read from the pieces that remain (an exchange that lost every
        item is unmentioned). ``self`` when no piece holds one."""
        pieces = [p for p in self.pieces if not holds_secret(p.title, p.text)]
        if len(pieces) == len(self.pieces):
            return self
        kept = {n for p in pieces for n in p.sources}
        return replace(
            self,
            pieces=pieces,
            kept=sorted(kept),
            unmentioned=sorted(set(self.unmentioned) | (set(self.kept) - kept)),
            forced=sorted({n for p in pieces if p.forced for n in p.sources}),
            refused=self.refused + len(self.pieces) - len(pieces),
        )


@dataclass
class Checked:
    """A window's reply after :func:`check`."""

    pieces: list[Piece] = field(default_factory=list)
    ignore: set[int] = field(default_factory=set)
    dropped: int = 0
    refused: int = 0


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


_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")
# A sentence end: a full stop, ``!``, ``?`` or ``…`` (and any closing quote or
# bracket) before whitespace, or a CJK full stop, ``！`` or ``？``.
_SENTENCE = re.compile(
    "[.!?\u2026][\"')\\]\u201d\u2019]*\\s+|[\u3002\uff01\uff1f]+\\s*"
)
_BLANK = re.compile(r"\s+")


def _cut_index(text: str, low: int, high: int, target: int) -> int:
    """Where to cut *text*, between *low* and *high* (the first part is
    ``text[:index]``): after the paragraph break nearest *target*, else after the
    sentence end nearest it, else after the whitespace nearest it, else at *target*
    (kept between *low* and *high*). A cut after a break starts the next part at a
    word."""
    low, high = max(low, 1), min(high, len(text) - 1)
    if low > high:
        return max(min(target, len(text) - 1), 1)
    for pattern in (_PARAGRAPH, _SENTENCE, _BLANK):
        ends = [match.end() for match in pattern.finditer(text, low, high)]
        if ends:
            return min(ends, key=lambda end: (abs(end - target), end))
    return min(max(target, low), high)


def _continued(position: int) -> str:
    """What a part of exchange *position* after the first starts with, so the
    model can name its source: ``[3] (continued) ``."""
    return f"[{position}]{_CONTINUED}"


def _parts(position: int, block: str, limit: int) -> list[Window]:
    """One exchange longer than *limit* as consecutive windows of at most *limit*
    characters (each part after the first starts :func:`_continued`), cut at a
    paragraph break, else a sentence end, else whitespace, else exactly at the size.
    At most :data:`MAX_PARTS` of them: what is left after that is cut, and the last
    part says so."""
    pieces: list[tuple[str, bool]] = []
    start = 0
    for k in range(MAX_PARTS):
        lead = _continued(position) if k else ""
        room = max(limit - len(lead), 1)
        if len(block) - start <= room:
            pieces.append((lead + block[start:], False))
            break
        if k == MAX_PARTS - 1:
            pieces.append((lead + block[start : start + room], True))
            break
        end = _cut_index(block, start + room // 2, start + room, start + room)
        pieces.append((lead + block[start:end], False))
        start = end
    return [
        Window((position,), text, cut=cut, part=(k, len(pieces)))
        for k, (text, cut) in enumerate(pieces, 1)
    ]


def pack(turns: Sequence[Turn], limit: int) -> list[Window]:
    """The exchanges of a chat in order, packed into windows of at most *limit*
    characters of prompt text (the exchanges as :func:`render` writes them, a blank
    line between). One exchange that is longer than *limit* is not cut at *limit*:
    it is read in consecutive part windows of at most *limit* characters each (see
    :func:`_parts`), at most :data:`MAX_PARTS` of them, between its neighbours'
    windows."""
    windows: list[Window] = []
    positions: list[int] = []
    blocks: list[str] = []
    starts: list[int] = []
    size = 0

    def close() -> None:
        nonlocal positions, blocks, starts, size
        if blocks:
            windows.append(
                Window(
                    tuple(positions),
                    _SEPARATOR.join(blocks),
                    starts=tuple(starts),
                )
            )
        positions, blocks, starts, size = [], [], [], 0

    for position, turn in enumerate(turns):
        block = render(position, turn)
        if len(block) > limit:
            close()
            windows.extend(_parts(position, block, limit))
            continue
        if blocks and size + len(_SEPARATOR) + len(block) > limit:
            close()
        starts.append(size + (len(_SEPARATOR) if blocks else 0))
        size += len(block) + (len(_SEPARATOR) if blocks else 0)
        positions.append(position)
        blocks.append(block)
    close()
    return windows


def halves(window: Window) -> tuple[Window, Window] | None:
    """The two windows *window* splits into when its reply runs out of room, or
    ``None`` when it cannot (under :data:`MIN_SPLIT_CHARS`).

    A window of several exchanges splits at the boundary between exchanges nearest
    its middle by characters. A window of one exchange splits its text in two: at the
    paragraph break nearest the middle, else a sentence end, else whitespace, else
    exactly in the middle (only within the middle half, so neither part is small);
    the second starts :func:`_continued`. The halves' texts, without that prefix,
    are the window's, in order.
    """
    text = window.text
    count = len(window.positions)
    if len(text) < MIN_SPLIT_CHARS:
        return None
    if count >= 2:
        if len(window.starts) != count:
            return None
        gap = len(_SEPARATOR)
        k = min(
            range(1, count),
            key=lambda i: abs(
                (window.starts[i] - gap) - (len(text) - window.starts[i])
            ),
        )
        at = window.starts[k]
        return (
            Window(window.positions[:k], text[: at - gap], starts=window.starts[:k]),
            Window(
                window.positions[k:],
                text[at:],
                starts=tuple(start - at for start in window.starts[k:]),
            ),
        )
    size = len(text)
    end = _cut_index(text, size // 4, 3 * size // 4, size // 2)
    (position,) = window.positions
    return (
        Window(window.positions, text[:end]),
        Window(window.positions, _continued(position) + text[end:], cut=window.cut),
    )


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
    no source, or with no text, or that is no object, is dropped and counted. An
    item whose title or text, as the model wrote it and before any cut, holds a
    secret (:func:`~.index.holds_secret`, read as the stored text is, with its
    whitespace joined) is refused whole and counted in ``refused``, never kept
    with the secret left out. A kind that is not one of :data:`~.index.ITEM_KINDS`
    reads as ``episodic`` (the kind that never replaces another); title and text
    are cut to :data:`TITLE_CHARS` and :data:`TEXT_CHARS`; a missing title is the
    start of the text.
    """
    inside = set(window.positions)
    out = Checked()
    entries = raw.get("items")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            out.dropped += 1
            continue
        said = [
            " ".join(str(entry.get(key) or "").split()) for key in ("title", "text")
        ]
        if holds_secret(*said):
            out.refused += 1
            continue
        sources = sorted({n for n in _numbers(entry.get("sources")) if n in inside})
        text = _cut(plain_text(said[1]), TEXT_CHARS)
        if not sources or not text:
            out.dropped += 1
            continue
        title = _cut(plain_text(said[0]), TITLE_CHARS)
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


def window_key(model: str, prompt: str, version: str | None = None) -> str:
    """The cache key of a window: the prompt version, the chat model's id and the
    whole user message, the exchanges with their dates. Another job that keeps its
    replies in the same cache (:mod:`.hippo_judge`) keys them with its own *version*."""
    digest = hashlib.sha256()
    for part in (PROMPT_VERSION if version is None else version, model, prompt):
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def split_key(key: str) -> str:
    """The cache key under which a window that ran out of room is kept, so a later
    run splits it without asking it again. Its own key, not the window's, so a
    reader that knows nothing of splits never finds it and reads no empty reply."""
    return hashlib.sha256(f"{key}\0split".encode()).hexdigest()


# What :func:`_window_reply` returns, in place of a reply's JSON, for a window that
# ran out of room and has to be split.
_SPLIT: Final[dict[str, Any]] = {"split": True}


def _is_split(raw: dict[str, Any] | None) -> bool:
    return raw is _SPLIT


class Cache:
    """A JSON-lines file of window replies, one ``{"key": …, "reply": {…}}`` a line.

    Read whole on first use and appended to on each new reply; a line that cannot
    be read is skipped. The file holds text from chats, so it is private (0600).
    One path may serve several runs, whatever store each writes. With no path
    (``""``) the replies are kept in memory only, and nothing is read or written.

    :meth:`put` is synchronous (no ``await`` between its read and its write), so
    coroutines of one event loop may call it as their replies arrive.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        self._replies: dict[str, dict[str, Any]] | None = None
        # Windows being asked for now, by key, so a second ask for the same window
        # (the same text in two chats of one run) waits for the first one's reply.
        self.inflight: dict[str, asyncio.Future[None]] = {}

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._replies is None:
            self._replies = {}
            if not self.path:
                return self._replies
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
        if not self.path:
            return
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False))
            handle.write("\n")


# ── The model ──


def _quiet(provider: BaseLLMProvider, backend: str | None = None) -> BaseLLMProvider:
    """A private view of *provider* that answers at temperature 0, stops at
    :data:`MAX_OUTPUT_TOKENS` and, on a local *backend*, does not think, never
    altering the provider chats use.

    The thinking-off fields come from
    :func:`~localmelo.support.providers.llm.thinking.thinking_off_fields`, over the
    ``chat_template_kwargs`` the provider's requests of this kind carry now.
    *backend* is the chat backend's key; ``None``, a server named by its URL,
    counts as a local one.
    """
    view = copy.copy(provider)
    kept = provider.extra_body_by_kind.get("retrieve", {})
    own = {**(getattr(provider, "extra_body", None) or {}), **kept}
    body = {
        **kept,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        **thinking_off_fields(provider, backend, own),
    }
    if isinstance(view, OllamaNativeChat):
        body["options"] = {
            **view.extra_body.get("options", {}),
            "temperature": 0,
            "num_predict": MAX_OUTPUT_TOKENS,
        }
        body.pop("max_tokens")
    view.extra_body_by_kind = {"retrieve": body}
    return view


def _unfinished(content: str) -> bool:
    """Whether *content*, after its thinking and any code fence, starts a JSON
    object that is never closed (braces outside strings do not balance)."""
    text = split_thinking(content or "")[1].strip()
    text = re.sub(r"^```[A-Za-z]*\s*", "", text)
    if not text.startswith("{"):
        return False
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return False
    return True


def _ran_out(reply: Message) -> bool:
    """Whether *reply* stopped because it ran out of output room, its JSON cut off
    at :data:`MAX_OUTPUT_TOKENS`.

    The chat provider's ``chat()`` hands back the message only, not the finish
    reason, so this reads what the message carries: the completion tokens the server
    reported (at least ``MAX_OUTPUT_TOKENS - 16``), and, when it reported none, whether
    the text is a JSON object that is never closed (:func:`_unfinished`).
    """
    done = (reply.usage or {}).get("completion_tokens")
    if isinstance(done, int) and not isinstance(done, bool) and done > 0:
        return done >= MAX_OUTPUT_TOKENS - _RUN_OUT_MARGIN
    return _unfinished(reply.content)


async def _ask(
    send: Send,
    prompt: str,
    stats: Stats,
    *,
    splittable: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
    parse_reply: Callable[[str], dict[str, Any] | None] = read_reply,
    attempts: int = 2,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Ask for a window's items, twice at most: the reply's JSON, or ``None`` and
    why not. Every request is counted. Other short JSON jobs can supply their
    own system prompt, reply reader, attempt count and request timeout
    (default :data:`ENCODE_TIMEOUT_S`) through this same path.

    An unreadable reply that ran out of room (:func:`_ran_out`) is not asked for
    again: the same prompt at temperature 0 is cut at the same place. It gives the
    split mark (``_SPLIT``) when the window can be split (*splittable*), else
    ``None``.
    """
    why = ""
    for _ in range(attempts):
        stats.calls += 1
        try:
            async with asyncio.timeout(
                ENCODE_TIMEOUT_S if timeout_s is None else timeout_s
            ):
                reply: Message = await send(
                    [Message("system", system_prompt), Message("user", prompt)]
                )
        except Exception as exc:
            # The class only: a provider error can carry part of the server's reply,
            # which may echo the request, and the request holds the user's texts.
            why = f"the model failed: {type(exc).__name__}"
            logger.warning("Hippo encoder: %s", why)
            continue
        stats.add(reply.usage)
        value = parse_reply(reply.content)
        if value is not None:
            return value, ""
        if _ran_out(reply):
            if splittable:
                return _SPLIT, ""
            return None, "the reply ran out of room and the window cannot be split"
        why = "the reply was not readable JSON"
    return None, why


async def _window_reply(
    send: Send,
    prompt: str,
    key: str,
    cache: Cache | None,
    stats: Stats,
    limit: asyncio.Semaphore | None,
    *,
    splittable: bool = False,
) -> tuple[dict[str, Any] | None, bool, str]:
    """A window's reply JSON (``None`` when the window failed; the split mark
    ``_SPLIT`` when it ran out of room and *splittable* says it can be split),
    whether it came from the cache, and why not when it failed.

    A cached reply makes no request, and neither does a window the cache knows
    ran out of room (:func:`split_key`). Otherwise the request (:func:`_ask`) runs
    under *limit* when it is given, and the reply is put in the cache as soon as it
    comes, so a run that is stopped keeps every window already answered. When the
    same window is already being asked for (the same text in two chats of one run),
    this waits for that reply and counts a cache hit, as a later serial ask would.
    """

    def known() -> dict[str, Any] | None:
        if cache is None:
            return None
        raw = cache.get(key)
        if raw is None and splittable and cache.get(split_key(key)) is not None:
            return _SPLIT
        return raw

    while cache is not None:
        raw = known()
        if raw is not None:
            stats.cache_hits += 1
            return raw, True, ""
        waiting = cache.inflight.get(key)
        if waiting is None:
            break
        await asyncio.shield(waiting)
    mine: asyncio.Future[None] | None = None
    if cache is not None:
        mine = cache.inflight[key] = asyncio.get_running_loop().create_future()
    try:
        async with limit if limit is not None else contextlib.nullcontext():
            raw, why = await _ask(send, prompt, stats, splittable=splittable)
        if raw is not None and cache is not None:
            cache.put(split_key(key) if _is_split(raw) else key, raw)
        return raw, False, why
    finally:
        if cache is not None and mine is not None:
            del cache.inflight[key]
            mine.set_result(None)


@dataclass
class _Read:
    """What asking for one window gave: its reply's JSON (``raw``, ``None`` when it
    failed), whether that came from the cache, why it failed, and, when it ran out of
    room, what its halves gave (in order, each read the same way)."""

    window: Window
    raw: dict[str, Any] | None = None
    cached: bool = False
    why: str = ""
    parts: list[_Read] = field(default_factory=list)

    def leaves(self) -> list[_Read]:
        """The sub-windows that were read (the window itself when it did not
        split), in order; a half that split again gives its own halves."""
        if not self.parts:
            return [self]
        return [leaf for part in self.parts for leaf in part.leaves()]

    def untouched(self) -> bool:
        """Whether no request was made for the window or any part of it."""
        return self.cached and all(part.untouched() for part in self.parts)


async def _read_window(
    send: Send,
    title: str,
    model: str,
    window: Window,
    *,
    cache: Cache | None,
    stats: Stats,
    limit: asyncio.Semaphore | None,
    depth: int = 0,
) -> _Read:
    """Ask for one window; when its reply runs out of room, split it (*depth* 0 is
    the window of the chat, at most :data:`MAX_SPLIT_DEPTH` levels) and ask for each
    half the same way, one by one without *limit* and together with it. A half that
    fails fails alone."""
    parts = halves(window) if depth < MAX_SPLIT_DEPTH else None
    prompt = prompt_of(title, window)
    raw, cached, why = await _window_reply(
        send,
        prompt,
        window_key(model, prompt),
        cache,
        stats,
        limit,
        splittable=parts is not None,
    )
    if not _is_split(raw):
        return _Read(window, raw, cached, why)
    assert parts is not None

    def read(half: Window) -> Coroutine[Any, Any, _Read]:
        return _read_window(
            send,
            title,
            model,
            half,
            cache=cache,
            stats=stats,
            limit=limit,
            depth=depth + 1,
        )

    if limit is None:
        reads = [await read(half) for half in parts]
    else:
        reads = await gather([read(half) for half in parts])
    return _Read(window, None, cached, parts=reads)


async def gather(
    coros: Sequence[Coroutine[Any, Any, _T]],
) -> list[_T]:
    """The results of *coros*, run together, in their order. A failure or a
    cancellation cancels the ones still running and is raised."""
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def encode(
    provider: BaseLLMProvider,
    title: str,
    turns: Sequence[Turn],
    *,
    window_chars: int,
    cache: Cache | None = None,
    stats: Stats | None = None,
    backend: str | None = None,
    limit: asyncio.Semaphore | None = None,
) -> Encoding:
    """Encode the exchanges *turns* of one chat into items.

    One model request for each window of at most *window_chars* characters, none
    when *cache* has the window's reply (:func:`window_key`). A window whose reply
    runs out of room is split in two and each half is asked in its place (see
    :func:`halves` and :func:`_read_window`). *stats* is added to.
    *backend* is the key of the chat backend *provider* runs on, which decides how
    the request turns thinking off (:func:`_quiet`); ``None``, as when a caller
    names none, is a local server named by its URL.

    Without *limit* the windows are asked for one by one, in order. With one, the
    windows that are not cached are asked for together, each request waiting for
    the semaphore, which a caller may share across the chats of a run to bound the
    requests in flight, the halves of a split window too; the result is built in
    window order all the same, so it is the one the serial path gives. A window that
    fails fails alone, and so does a half of one.
    """
    stats = stats if stats is not None else Stats()
    model = str(getattr(provider, "model", "") or "")
    encoder = f"{PROMPT_VERSION}:{model}"
    send = send_from_provider(_quiet(provider, backend))
    result = Encoding()
    mentioned: set[int] = set()
    ignored: set[int] = set()
    windows = pack(turns, window_chars)

    def ask(window: Window) -> Coroutine[Any, Any, _Read]:
        return _read_window(
            send, title, model, window, cache=cache, stats=stats, limit=limit
        )

    if limit is None:
        reads = [await ask(window) for window in windows]
    else:
        reads = await gather([ask(window) for window in windows])
    for window, read in zip(windows, reads, strict=True):
        report = WindowReport(
            len(window.positions),
            len(window.text),
            cached=read.untouched(),
            cut=window.cut,
            part=window.part,
        )
        result.windows.append(report)
        leaves = read.leaves()
        if read.parts:
            report.split = len(leaves)
        for leaf in leaves:
            if leaf.raw is None:
                if read.parts:
                    report.failed_parts += 1
                first = min(leaf.window.positions)
                if result.failed_from is None or first < result.failed_from:
                    result.failed_from = first
                logger.warning("Hippo encoder: a window failed, %s", leaf.why)
                continue
            checked = check(leaf.raw, leaf.window, turns)
            report.dropped += checked.dropped
            result.refused += checked.refused
            for piece in checked.pieces:
                result.pieces.append(replace(piece, encoder=encoder))
                mentioned.update(piece.sources)
            ignored |= checked.ignore
        report.failed = all(leaf.raw is None for leaf in leaves)
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


def is_checked(encoder: str) -> bool:
    """Whether an item's *encoder* is one recall may read with ``auto``: the rules
    (:data:`RULES_ENCODER`), or ``<version>:<model>`` (as :func:`encode` writes it)
    with a version and the model in :data:`CHECKED_ENCODERS`. An empty *encoder*, or
    one with no version, is not checked."""
    version, _, model = encoder.partition(":")
    return encoder == RULES_ENCODER or bool(version) and model in CHECKED_ENCODERS


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
