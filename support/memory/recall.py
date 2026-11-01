"""Hybrid retrieval over records and the conversation index, and how an item reads.

BM25 always runs, over words and CJK characters and bigrams; with an embedding
model, cosine similarity over the vectors joins it, and reciprocal rank fusion
(k = :data:`RRF_K`) merges the two lists (:func:`ranked`; :func:`pick` keeps the
best few). The ``llm`` retriever asks :func:`ranked` for loose BM25 candidates
and lets the chat model choose among them. An item takes part in a list only
above that list's bar: most of the question's content words for BM25,
:data:`MIN_COSINE` for vectors, so most turns recall nothing. Documents (not
records) decay with a :data:`HALF_LIFE_DAYS` half-life, records rank first at
an equal score, and at most :data:`PER_SESSION` documents come from one chat.
A long turn is indexed as several chunks (:mod:`.index`); a list holds at most
one chunk of a turn, the best, whatever the chats' cap.
A turn whose fact a newer note or a newer turn replaced comes after the rest,
below the newer one, and reads as older (:func:`newer_first`); the recall block
leaves such a turn out unless the message asks about the past
(:func:`current_only`), so the model is given the current value only.
A chat's own turns are recalled like another chat's once the model's window no
longer holds them (:func:`in_view`), and read as ``this chat`` (:func:`this_chat`).
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.melo.memory.bm25 import BM25Index, tokenize
from localmelo.melo.memory.hippo.policy import STOPWORDS, normalise

RRF_K: Final = 60
MIN_COVERAGE: Final = 0.6
MIN_COSINE: Final = 0.55
HALF_LIFE_DAYS: Final = 30.0
PER_SESSION: Final = 2
# What one turn recalls by itself, and the room its block may take.
AUTO_RECORDS: Final = 3
AUTO_DOCS: Final = 2
BLOCK_CHARS: Final = 900
# What the block says about the message's own words: kept as notes, or refused.
# It forbids saving that fact again, not keeping another fact of the message.
KEPT_LINE: Final = (
    "melo kept from this message: {notes} (saved; do not save it again or in a file)"
)
NOT_KEPT_LINE: Final = (
    "melo did not keep part of this message: it looked like a password or key, "
    "and it must not be written anywhere (no file, no code)"
)
KEPT_NOTES: Final = 3
# How a turn whose fact a newer note, or a newer turn's words, replaced reads,
# to the model and the page.
OLDER: Final = "older; replaced by a newer note"
OLDER_TURN: Final = "older; replaced by a newer message"
# Index document kinds besides a chat's turns (:mod:`.index`).
SIDE_SUMMARY: Final = "side_summary"
AGENT_EXCHANGE: Final = "agent_exchange"
_KEPT_CHARS: Final = 120
_LINE_CHARS: Final = 240
_TITLE_CHARS: Final = 40
# Words that ask rather than say what is asked about; never a content word. The
# last five ask for an older value (:func:`asks_about_past`).
_ASKING_WORDS: Final = (
    "what when where who whom whose which why how do does did can could would should "
    "will tell know remember recall please about any anything there am 时候 "
    "before earlier previously formerly originally"
)
_ASKING: Final = frozenset(_ASKING_WORDS.split())
_CJK_FILLER: Final = frozenset("的了是在我你他她它们和与就也都吗呢吧啊么什怎哪谁几时候")
# A note that starts in the first person ("I prefer tea", "My name is Juno", "我…").
_FIRST_PERSON: Final = re.compile(r"(?:i|my|me|we|our|我|咱)(?![a-z])", re.IGNORECASE)
# A message that asks about an older value, so the block keeps the older item.
_PAST: Final = re.compile(
    r"\b(?:before|used to|did(?:n't| not)? \w+ use to|earlier|previous(?:ly)?"
    r"|formerly|originally"
    r"|(?<!how )old(?:er)?)\b|以前|之前|原来|过去|曾经",
    re.IGNORECASE,
)


@dataclass
class Item:
    """One record (key ``r:<id>``) or document (``d:<id>``) that can be recalled.

    ``text`` is what the person reads: a record in the second person, or a
    turn. ``quote`` is a record's words as the user typed them, which the
    model reads, so that "you" never stands for the user in its prompt.
    ``where`` names the chat and the day. A record's ``replaces`` are the
    turns (session id, turn) whose facts it replaced; a turn's ``said`` is
    what its typed words state about the user, each folded statement with
    its subject key (``""`` for none). ``older`` is how such a turn reads
    when a newer one replaced it: :data:`OLDER` or :data:`OLDER_TURN`.
    """

    key: str
    text: str
    where: str
    session_id: str
    turn: int
    created_at: float
    quote: str = ""
    vector: np.ndarray | None = None  # unit length
    embed_model: str = ""
    replaces: tuple[tuple[str, int], ...] = ()
    said: tuple[tuple[str, str], ...] = ()
    older: str = ""

    @property
    def is_record(self) -> bool:
        return self.key.startswith("r:")

    @property
    def source(self) -> str:
        said = ("you said it · " if self.is_record else "") + self.where
        return f"{said} · {self.older}" if self.older else said

    @property
    def prompt(self) -> str:
        """The item as the model reads it, with where it came from.

        A record reads as the user's words when it has them; without, a note in
        the second person says it is written to the user ("you" is the user),
        and one in the first person says it is the user's own.
        """
        if not self.is_record:
            older = f" ({self.older})" if self.older else ""
            return f"{snippet(self.text)} ({self.where}){older}"
        if self.quote:
            said = f"the user said “{self.quote}”"
        elif _FIRST_PERSON.match(self.text):
            said = f"the user's note “{self.text}”"
        else:
            said = f"a note to the user “{self.text}”"
        return f"{said} ({self.where})"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "record" if self.is_record else "chat",
            "id": int(self.key[2:]),
            "text": self.text if self.is_record else snippet(self.text),
            "source": self.source,
            "session_id": self.session_id,
            "turn": self.turn,
            "at": self.created_at,
        }


def unit(vector: Iterable[float] | None) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(list(vector), dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if array.size and norm > 0 else None


def chat_title(title: str) -> str:
    """A source chat's title as its label quotes it: one line, cut at 40 characters."""
    title = " ".join(title.split())
    return title[: _TITLE_CHARS - 1] + "…" if len(title) > _TITLE_CHARS else title


def chat_label(kind: str, title: str, surface: str) -> str:
    """How a source chat is named: ``Main chat``, ``chat “Fix CI”``, ``task “…”``."""
    title = chat_title(title)
    if kind == "front":
        return "Main chat"
    if kind == SIDE_SUMMARY:
        return f"chat “{title}” · summary" if title else "a chat · summary"
    if kind == AGENT_EXCHANGE:
        return f"task “{title}” · question" if title else "a task · question"
    name = {"task": "task", "front_bg": "main chat task"}.get(kind, "chat")
    if surface == "cli" and kind == "chat":
        name = "terminal chat"
    return f"{name} “{title}”" if title else f"a {name}"


def day(at: float) -> str:
    local = time.localtime(at)
    return f"{time.strftime('%b', local)} {local.tm_mday}"


def in_view(item: Item, session_id: str, keep_from_turn: int | None) -> bool:
    """Whether *item* is a turn of chat *session_id* that the chat's model still
    sees: its turn is *keep_from_turn* or later, or any turn when that is ``None``.

    Such a turn is not recalled: the chat has it already. A summary or an
    exchange (a turn below 0, :mod:`.index`) is never recalled to the chat it belongs to.
    """
    return (
        not item.is_record
        and item.session_id == session_id
        and (keep_from_turn is None or item.turn < 0 or item.turn >= keep_from_turn)
    )


def this_chat(item: Item, session_id: str) -> Item:
    """*item* worded as the earlier talk of chat *session_id*, when it is a turn of it."""
    if item.is_record or item.session_id != session_id:
        return item
    return dataclasses.replace(item, where=f"this chat, {day(item.created_at)}")


def origin(chat: str, gone: bool, at: float) -> str:
    """Where a note came from: ``from chat “Plans”, Oct 30``, or ``from a deleted chat, Oct 30``."""
    return f"from {'a deleted chat' if gone else chat}, {day(at)}"


def snippet(text: str, limit: int = _LINE_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _undoubled(base: str) -> str:
    """*base* (a word without ``-ing`` or ``-ed``) with a doubled last consonant
    reduced to one: "runn" gives "run"; "fall", "miss", "buzz", "stuff" and "add"
    are words as they are."""
    if (
        len(base) >= 4
        and base[-1] == base[-2]
        and base[-1].isalpha()
        and base[-1] not in "aeiouylszfd"
    ):
        return base[:-1]
    return base


def stem(word: str) -> str:
    """A lowercase ASCII word without a plural, ``-ing`` or ``-ed`` ending.

    So "meets" and "meeting" find "meet" and "Tuesdays" finds "tuesday"; both
    sides of a match are stemmed, so a stem need not be a word. ``-ies`` and a
    long ``-ie`` end in ``y`` ("replies" and "reply", "movies" and "movie"). A
    doubled consonant the ending brought drops one ("running" and "run",
    "stopped" and "stop"), unless the word keeps it bare: ``l``, ``s``, ``z``,
    ``f`` and ``d`` stay ("falling", "missing", "buzzing", "stuffed", "adding").
    """
    if len(word) >= 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) >= 5 and word.endswith("ie"):
        return word[:-2] + "y"
    if len(word) >= 6 and word.endswith("ing"):
        return _undoubled(word[:-3])
    if len(word) >= 5 and word.endswith("ed"):
        return _undoubled(word[:-2])
    if len(word) >= 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


_WORD = re.compile(r"[A-Za-z0-9_]+")


def stemmed(text: str) -> str:
    """*text* with each ASCII word stemmed; CJK runs stay whole for the tokenizer."""
    return _WORD.sub(lambda match: stem(match.group(0).lower()), text)


def gate_terms(text: str) -> set[str]:
    """The stemmed content terms of a question: no stopwords, asking words or CJK fillers."""
    return {
        stem(term)
        for term in tokenize(normalise(text))
        if term not in STOPWORDS
        and term not in _ASKING
        and (term.isascii() or not any(char in _CJK_FILLER for char in term))
    }


class Corpus:
    """The items a search ranks, with a BM25 index kept up to date by key."""

    def __init__(self) -> None:
        self.items: dict[str, Item] = {}
        self.terms: dict[str, frozenset[str]] = {}
        self.index = BM25Index()
        # Changes with every add and remove, so what is derived from all the
        # items can be kept until they change.
        self.version = 0

    def add(self, item: Item) -> None:
        self.version += 1
        self.items[item.key] = item
        words = stemmed(normalise(item.text))
        self.terms[item.key] = frozenset(tokenize(words))
        self.index.add(item.key, words)

    def remove(self, key: str) -> None:
        self.version += 1
        self.items.pop(key, None)
        self.terms.pop(key, None)
        self.index.remove(key)

    def keys(self, prefix: str) -> set[str]:
        return {key for key in self.items if key.startswith(prefix)}


def ranked(
    corpus: Corpus,
    query: str,
    query_vector: np.ndarray | None,
    embed_model: str,
    *,
    now: float,
    exclude_session: str = "",
    keep_from_turn: int | None = None,
    coverage: float = MIN_COVERAGE,
) -> list[Item]:
    """The items that pass the bars for *query*, best first.

    *coverage* is the share of the question's content words an item must hold
    to join the BM25 list; a small one lets loose candidates through. The
    turns of *exclude_session* from *keep_from_turn* on (all of them without
    one) are left out (:func:`in_view`).
    """
    wanted = gate_terms(query)
    lists: list[list[str]] = []
    if wanted:
        need = max(1, math.ceil(coverage * len(wanted)))
        hits = corpus.index.search(stemmed(query), len(corpus.items))
        lists.append([k for k, _ in hits if len(wanted & corpus.terms[k]) >= need])
    if query_vector is not None:
        scored = [
            (float(np.dot(query_vector, item.vector)), key)
            for key, item in corpus.items.items()
            if item.vector is not None
            and item.embed_model == embed_model
            and item.vector.shape == query_vector.shape
        ]
        lists.append([k for c, k in sorted(scored, reverse=True) if c >= MIN_COSINE])
    fused: dict[str, float] = {}
    for keys in lists:
        for position, key in enumerate(keys):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + position + 1)
    rows: list[tuple[float, bool, Item]] = []
    for key, score in fused.items():
        item = corpus.items[key]
        if not item.is_record:
            if in_view(item, exclude_session, keep_from_turn):
                continue
            age_days = max(0.0, now - item.created_at) / 86_400
            score *= 0.5 ** (age_days / HALF_LIFE_DAYS)
        rows.append((score, item.is_record, item))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return one_per_turn([item for _, _, item in rows])


def one_per_turn(items: Sequence[Item]) -> list[Item]:
    """*items* with only the first (the best) document of each turn kept: a long
    turn's chunks (:mod:`.index`) share their session and turn. Records stay."""
    seen: set[tuple[str, int]] = set()
    kept: list[Item] = []
    for item in items:
        if not item.is_record:
            turn = (item.session_id, item.turn)
            if turn in seen:
                continue
            seen.add(turn)
        kept.append(item)
    return kept


def replaced_turns(corpus: Corpus) -> dict[tuple[str, int], Item]:
    """The turns whose facts a note in *corpus* replaced, each to that note."""
    return {
        turn: item
        for item in corpus.items.values()
        if item.is_record
        for turn in item.replaces
    }


def newer_first(
    items: Sequence[Item],
    replaced: Mapping[tuple[str, int], Item],
    *,
    exclude_session: str = "",
    keep_from_turn: int | None = None,
) -> list[Item]:
    """*items* with each turn whose fact a newer note or turn *replaced* moved
    after the rest and marked older; the newer one comes in ahead of it if it
    was not found (a turn the chat under way still shows, never: the chat has
    it already, :func:`in_view`).

    The old fact stays as history, as people keep it, but the newer one wins.
    """
    keys = {item.key for item in items}
    fresh: list[Item] = []
    stale: list[Item] = []
    for item in items:
        newer = None if item.is_record else replaced.get((item.session_id, item.turn))
        if newer is None:
            fresh.append(item)
            continue
        # A newer turn a still newer note or turn replaced: that one wins.
        for _ in range(len(replaced)):
            if newer.is_record or (newer.session_id, newer.turn) not in replaced:
                break
            newer = replaced[(newer.session_id, newer.turn)]
        here = in_view(newer, exclude_session, keep_from_turn)
        if newer.key not in keys and not here:
            keys.add(newer.key)
            fresh.append(newer)
        older = OLDER if newer.is_record else OLDER_TURN
        stale.append(dataclasses.replace(item, older=older))
    return fresh + stale


def asks_about_past(message: str) -> bool:
    """Whether *message* asks about an older value ("before", "used to", "以前" …)."""
    return _PAST.search(message) is not None


def current_only(items: Sequence[Item], message: str) -> list[Item]:
    """*items* for the recall block: without a turn a newer note or message
    replaced (:attr:`Item.older`), unless *message* asks about the past.

    The newer value is in the block (:func:`newer_first` brings it in), so the
    old one would only be named as history in an answer that wants the new
    value. The Memory tab and ``search_memory`` still show it.
    """
    if asks_about_past(message):
        return list(items)
    return [item for item in items if not item.older]


def pick(items: Sequence[Item], *, records: int, docs: int) -> list[Item]:
    """From *items*, best first: at most *records* records and *docs* documents,
    one chunk of a turn at most (the best), then at most :data:`PER_SESSION`
    documents of a chat."""
    items = one_per_turn(items)
    kept = [item for item in items if item.is_record][:records]
    # A record already says what the turn it came from said.
    said = {(item.session_id, item.turn) for item in kept}
    per_session: Counter[str] = Counter()
    for item in items:
        if item.is_record or (item.session_id, item.turn) in said:
            continue
        if per_session[item.session_id] < PER_SESSION and per_session.total() < docs:
            per_session[item.session_id] += 1
            kept.append(item)
    order = {id(item): n for n, item in enumerate(items)}
    return sorted(kept, key=lambda item: order[id(item)])


def kept_lines(quotes: Sequence[str], *, refused: bool = False) -> list[str]:
    """The lines that say what melo did with the message's own words.

    *quotes* are the user's words that became notes, quoted as the recall
    lines quote them (at most :data:`KEPT_NOTES`, each cut at
    :data:`_KEPT_CHARS`, so these lines always fit the block); *refused* says
    a sentence looked like a secret.
    """
    lines: list[str] = []
    if quotes:
        shown = [f"“{snippet(q, _KEPT_CHARS)}”" for q in quotes[:KEPT_NOTES]]
        if len(quotes) > KEPT_NOTES:
            shown.append(f"and {len(quotes) - KEPT_NOTES} more")
        lines.append(KEPT_LINE.format(notes=" ".join(shown)))
    return lines + ([NOT_KEPT_LINE] if refused else [])


def block(
    items: Iterable[Item], kept: Sequence[str] = (), chars: int = BLOCK_CHARS
) -> tuple[str, list[Item]]:
    """The recall block a user message carries, and the items that fit in it.

    The *kept* lines (:func:`kept_lines`) come first, with or without recall.
    Items are added while the block stays within *chars*
    (:data:`BLOCK_CHARS` unless the settings say otherwise).
    """
    lines = [BLOCK_HEADER, *kept]
    used: list[Item] = []
    size = sum(len(line) + 1 for line in lines) - 1
    for item in items:
        line = f"- {item.prompt}"
        if size + len(line) + 1 > chars:
            continue
        lines.append(line)
        used.append(item)
        size += len(line) + 1
    return ("\n".join(lines) if used or kept else ""), used
