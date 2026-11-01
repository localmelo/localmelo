"""Hybrid retrieval over records and the conversation index, and how an item reads.

BM25 always runs, over words and CJK characters and bigrams; with an embedding
model, cosine similarity over the vectors joins it, and reciprocal rank fusion
(k = :data:`RRF_K`) merges the two lists (:func:`ranked`; :func:`pick` keeps the
best few). The ``llm`` retriever asks :func:`ranked` for loose BM25 candidates
and lets the chat model choose among them. An item takes part in a list only
above that list's bar: most of the question's content words for BM25
(:data:`MIN_COVERAGE`), :data:`MIN_COSINE` for vectors, so most turns recall
nothing; both bars are developer settings. Documents (not
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
With ``[memory] recall_form = "list"`` a turn is not shown as an excerpt but as one
line, its id, date, chat, title and similarity (:meth:`Item.listed`), under a line
that tells the model to open the turns it needs with ``open_memory``; the list is
not bound by the chats' cap or the block's room (:func:`block`, :func:`pick`), and
:func:`whole_turn` and :func:`cut_at_sentence` give a turn the model opens.
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np

from localmelo.melo import clock
from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.melo.memory.bm25 import BM25Index, tokenize
from localmelo.melo.memory.hippo.policy import STOPWORDS, normalise
from localmelo.melo.memory.notes import split_sentences

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
# What a turn's two parts may take together in the block (the words that say who
# spoke are extra, so a turn line stays about as long as it was), and the least
# melo's reply keeps when the user's words are long.
TURN_CHARS: Final = _LINE_CHARS - 40
_REPLY_CHARS: Final = 60
# With ``recall_excerpt = "match"``, the share of a turn's room the user's words
# keep when they are long; the rest is melo's reply's.
_USER_SHARE: Final = 0.4
_TITLE_CHARS: Final = 40
# The list form (``[memory] recall_form = "list"``): the most turns it lists and the
# room one opened turn takes, as the settings default them; the longest title of a
# line; the line above the list; and the mark where an opened turn is cut.
LIST_MAX: Final = 20
OPEN_CHARS: Final = 4_000
LIST_TITLE_CHARS: Final = 80
LIST_GUIDE: Final = (
    "Older chat turns that may hold the answer, closest first. Open the ones you "
    "need with open_memory before you answer from them; a title is not the content."
)
CUT_MARK: Final = "… (cut)"
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


class Shown(NamedTuple):
    """How much of a recalled turn the model reads, and which part of it.

    *chars* is the room one turn takes (:data:`TURN_CHARS`), in the recall block
    and in the memory search tool's list; text that is no turn takes *chars* plus
    the 40 characters that say who spoke. With *terms*, the question's content
    terms (``[memory] recall_excerpt = "match"``), each part of a turn shows the
    sentences that hold most of them rather than its head (:func:`turn_said`).
    The defaults are the head of every turn, as it always was.
    """

    chars: int = TURN_CHARS
    terms: frozenset[str] = frozenset()

    @classmethod
    def of(cls, chars: int = TURN_CHARS, question: str | None = None) -> Shown:
        """*chars* of room, and the best matching sentences of *question* if given."""
        return cls(chars, frozenset(gate_terms(question)) if question else frozenset())


HEADS: Final = Shown()  # the head of every turn, in :data:`TURN_CHARS`
WHOLE: Final = Shown(10**9)  # every turn whole: no room limit


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
    when a newer one replaced it: :data:`OLDER` or :data:`OLDER_TURN`. A
    document's ``chunk`` is 0 for the head of a turn and 1 and up for the later
    pieces of a long one (:mod:`.index`). ``similarity`` is the cosine of a
    turn to the message when a search matched it by meaning (the list form
    shows it); it is set on a copy for one search, never stored.
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
    chunk: int = 0
    similarity: float | None = None

    @property
    def is_record(self) -> bool:
        return self.key.startswith("r:")

    @property
    def source(self) -> str:
        said = ("you said it · " if self.is_record else "") + self.where
        return f"{said} · {self.older}" if self.older else said

    @property
    def prompt(self) -> str:
        """The item as the model reads it, with where it came from."""
        return self.line()

    def line(
        self, now: float | None = None, *, timed: bool = False, show: Shown = HEADS
    ) -> str:
        """The item as the model reads it, with where and when it came from.

        A record reads as the user's words when it has them; without, a note in
        the second person says it is written to the user ("you" is the user),
        and one in the first person says it is the user's own. A turn says who
        spoke (:func:`turn_said`). The date has the year when it is not *now*'s
        (the pinned or real time by default), and the time of day with *timed*
        (:func:`times`). *show* says how much of a turn is shown (:class:`Shown`).
        """
        where = self._dated(clock.now() if now is None else now, timed)
        older = f" ({self.older})" if self.older else ""
        if not self.is_record:
            return f"{turn_said(self.text, show)} ({where}){older}"
        if self.quote:
            said = f"the user said “{self.quote}”"
        elif _FIRST_PERSON.match(self.text):
            said = f"the user's note “{self.text}”"
        else:
            said = f"a note to the user “{self.text}”"
        return f"{said} ({where}){older}"

    def _dated(self, now: float, timed: bool) -> str:
        """:attr:`where` with its date as :func:`day` writes it for *now*; a
        ``where`` that does not end in the item's own day is left as it is."""
        chat, date = self._chat_and_date(now, timed)
        return f"{chat}, {date}" if date else chat

    def _chat_and_date(self, now: float, timed: bool) -> tuple[str, str]:
        """The chat of :attr:`where` and its date as :func:`day` writes it for *now*;
        no date (``""``) when ``where`` does not end in the item's own day."""
        plain = day(self.created_at)
        if not self.where.endswith(f", {plain}"):
            return self.where, ""
        return (
            self.where[: -len(plain) - 2],
            day(self.created_at, now, timed=timed),
        )

    def listed(self, now: float | None = None, *, timed: bool = False) -> str:
        """The turn as one line of the list form: ``[t12] Oct 30 · Main chat ·
        “Where do I plant the tomatoes?” · 0.71``.

        ``t12`` is the document's id (what ``open_memory`` takes), the date and
        chat read as :meth:`line` writes them (the year when it is not *now*'s, the
        time of day with *timed*), the title is :func:`listed_title`, and the
        similarity is the cosine to the message with two decimals when a search
        matched the turn by meaning, else ``words``. A turn a newer one replaced
        keeps its older mark, as :meth:`line` writes it.
        """
        chat, date = self._chat_and_date(clock.now() if now is None else now, timed)
        date = date or day(self.created_at)
        meaning = "words" if self.similarity is None else f"{self.similarity:.2f}"
        older = f" ({self.older})" if self.older else ""
        return (
            f"[t{self.key[2:]}] {date} · {chat} · “{listed_title(self)}” · "
            f"{meaning}{older}"
        )

    def opened_heading(self, now: float | None = None) -> str:
        """The first line of a turn opened in full: its id, date and chat, and the
        older mark when a newer one replaced it."""
        chat, date = self._chat_and_date(clock.now() if now is None else now, False)
        older = f" ({self.older})" if self.older else ""
        return f"[t{self.key[2:]}] {date or day(self.created_at)} · {chat}{older}"

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


def day(at: float, now: float | None = None, *, timed: bool = False) -> str:
    """The day of *at*: ``Oct 30``. Given *now*, the year follows when *at* is not
    in *now*'s year (``May 20, 2023``); with *timed*, the time of day ends it
    (``Oct 30, 14:05``)."""
    local = time.localtime(at)
    text = f"{time.strftime('%b', local)} {local.tm_mday}"
    if now is not None and local.tm_year != time.localtime(now).tm_year:
        text += f", {local.tm_year}"
    return f"{text}, {time.strftime('%H:%M', local)}" if timed else text


def times(items: Iterable[Item]) -> set[str]:
    """The keys of the *items* that are from the same day as another of them:
    their dates show the time too, so a reader can tell which is newer."""
    items = list(items)
    days = Counter(time.localtime(item.created_at)[:3] for item in items)
    return {item.key for item in items if days[time.localtime(item.created_at)[:3]] > 1}


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


def turn_said(text: str, show: Shown = HEADS) -> str:
    """A turn's document text as the model reads it: ``the user said “…”; melo
    replied “…”``, so a past reply of melo's is never read as the user's own
    statement. Both parts are bounded together (``show.chars``, :data:`TURN_CHARS`
    by default); the user's words come first and stay whole when they fit, and
    melo's reply keeps at least :data:`_REPLY_CHARS` of the room. Text that is no
    turn (a summary, a task's question) is a plain :func:`snippet`.

    With ``show.terms`` a part that does not fit shows the window that holds the
    most of them instead (:func:`_matching`).
    """
    plain = show.chars + _LINE_CHARS - TURN_CHARS
    body = text.partition("\n")[2] if text.startswith("Task: ") else text
    if not body.startswith("User: "):
        return snippet(text, plain)
    typed, marker, answer = body[len("User: ") :].partition("\nmelo: ")
    if not marker:
        return snippet(text, plain)
    typed, answer = " ".join(typed.split()), " ".join(answer.split())
    if len(typed) + len(answer) > show.chars:
        picked = _matching(typed, answer, show) if show.terms else None
        if picked is not None:
            typed, answer = picked
        else:
            typed = snippet(typed, show.chars - _REPLY_CHARS if answer else show.chars)
            answer = snippet(answer, show.chars - len(typed)) if answer else ""
    said = [f"the user said “{typed}”"] if typed else []
    return "; ".join(said + ([f"melo replied “{answer}”"] if answer else []))


def _matching(typed: str, answer: str, show: Shown) -> tuple[str, str] | None:
    """The two parts of a turn too long for *show*, each as the window that holds
    the most of ``show.terms``; ``None`` when neither part holds one.

    The user's words stay whole when they fit in :data:`_USER_SHARE` of the room
    and melo's reply has the rest; a part that is short leaves its room to the
    other. A part with no term shows its head.
    """
    room = show.chars
    own = int(room * _USER_SHARE)
    if not answer:
        shares = (room, 0)
    elif not typed:
        shares = (0, room)
    elif len(typed) <= own:
        shares = (len(typed), room - len(typed))
    elif len(answer) <= room - own:
        shares = (room - len(answer), len(answer))
    else:
        shares = (own, room - own)
    first, found_first = _window(typed, show.terms, shares[0])
    second, found_second = _window(answer, show.terms, shares[1])
    return (first, second) if found_first or found_second else None


def _stems(text: str) -> frozenset[str]:
    """The stemmed terms of *text*, as :func:`gate_terms` stems a question's."""
    return frozenset(stem(term) for term in tokenize(normalise(text)))


def _spans(text: str, limit: int) -> list[tuple[int, int]]:
    """The sentences of flat *text* as (start, end) offsets; a sentence longer than
    *limit* is cut into pieces at spaces (anywhere, for text without them)."""
    spans: list[tuple[int, int]] = []
    for start, sentence in split_sentences(text):
        end = start + len(sentence)
        while end - start > limit:
            cut = text.rfind(" ", start + 1, start + limit + 1)
            cut = cut if cut > start else start + limit
            spans.append((start, cut))
            start = cut
            while start < end and text[start] == " ":
                start += 1
        if start < end:
            spans.append((start, end))
    return spans


def _window(text: str, terms: frozenset[str], room: int) -> tuple[str, bool]:
    """*text* if it fits in *room* characters, else the sentence that holds the most
    of *terms* grown to its neighbours (the better one first) while the window, with
    a ``…`` where text is left out, fits; and whether a term was found. With no term
    in the text, its head."""
    if len(text) <= room:
        return text, False
    spans = _spans(text, room - 2)  # a window may need a … at each end
    scores = [len(terms & _stems(text[start:end])) for start, end in spans]
    if not spans or not max(scores):
        return snippet(text, room), False
    last = len(spans) - 1

    def ends(low: int, high: int) -> tuple[int, int]:
        return (0 if low == 0 else spans[low][0]), (
            len(text) if high == last else spans[high][1]
        )

    def size(low: int, high: int) -> int:
        begin, end = ends(low, high)
        return end - begin + (begin > 0) + (end < len(text))

    low = high = max(range(len(spans)), key=lambda n: (scores[n], -n))
    grown = True
    while grown:
        grown = False
        for n in sorted(
            (n for n in (high + 1, low - 1) if 0 <= n <= last),
            key=lambda n: -scores[n],
        ):
            wider = (low, n) if n > high else (n, high)
            if size(*wider) <= room:
                low, high = wider
                grown = True
                break
    begin, end = ends(low, high)
    return ("…" if begin else "") + text[begin:end] + (
        "…" if end < len(text) else ""
    ), True


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
    min_cosine: float = MIN_COSINE,
) -> list[Item]:
    """The items that pass the bars for *query*, best first.

    *coverage* is the share of the question's content words an item must hold
    to join the BM25 list; a small one lets loose candidates through.
    *min_cosine* is the least similarity a vector needs to join the vector
    list. The turns of *exclude_session* from *keep_from_turn* on (all of them
    without one) are left out (:func:`in_view`).
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
        lists.append([k for c, k in sorted(scored, reverse=True) if c >= min_cosine])
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


def word_cut(text: str, limit: int) -> str:
    """*text* if it fits in *limit* characters, else its head cut at a word, with a
    ``…`` (at most *limit* characters in all); text without a space is cut anywhere."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 1, limit)
    head = text[: cut if cut > 0 else limit - 1].rstrip(" ,;:-")
    return head + "…"


def listed_title(item: Item) -> str:
    """The title of *item*'s line in the list form: the first sentence of the user's
    words in the turn, cut at a word (:data:`LIST_TITLE_CHARS` at most).

    A document knows only its own text: for a later chunk of a long turn that is
    the head of the user's words it repeats (or, when the user's words were longer
    than a chunk, the window of them it starts in), so every chunk of a turn has
    the same title unless the user's words ran on. A turn without the user's words
    takes the first sentence of melo's reply, and a summary or a task's exchange
    (no ``User:`` part) the first sentence of what there is.
    """
    text = item.text
    body = text.partition("\n")[2] if text.startswith("Task: ") else text
    if body.startswith("User: "):
        typed, marker, answer = body[len("User: ") :].partition("\nmelo: ")
        body = typed or answer if marker else typed
    sentences = split_sentences(body)
    first = sentences[0][1] if sentences else body
    return word_cut(" ".join(first.split()), LIST_TITLE_CHARS) or "…"


def meaning_of(
    item: Item, vector: np.ndarray | None, embed_model: str, min_cosine: float
) -> float | None:
    """The cosine of *item* to the question's *vector* when :func:`ranked` took the
    item into its vector list (its vector is of *embed_model* and the cosine is at
    least *min_cosine*); ``None`` when it was matched by words only."""
    if (
        vector is None
        or item.vector is None
        or item.embed_model != embed_model
        or item.vector.shape != vector.shape
    ):
        return None
    cosine = float(np.dot(vector, item.vector))
    return cosine if cosine >= min_cosine else None


_ANSWER_MARK: Final = "\nmelo: "
# How far back from the end of the text a later chunk's start is looked for, and
# the least overlap that counts as one.
_OVERLAP_REACH: Final = 400
_MIN_OVERLAP: Final = 8


def whole_turn(chunks: Sequence[str]) -> str:
    """A turn's document text, ``User: …\nmelo: …``, from its chunks in order.

    A later chunk (:func:`~localmelo.support.memory.index.document_chunks`) repeats
    ``User: `` and a short head of the user's words before melo's reply when it
    starts in the reply, and overlaps the end of the chunk before it: both are
    taken out, so the text reads as it was written. Text the index left out after
    its last chunk is not here.
    """
    if not chunks:
        return ""
    text = chunks[0]
    for chunk in chunks[1:]:
        body = chunk.removeprefix("User: ")
        head, mark, window = body.partition(_ANSWER_MARK)
        typed = " ".join(text.partition(_ANSWER_MARK)[0].partition("User: ")[2].split())
        if mark and typed.startswith(head.removesuffix("…").rstrip()):
            body = window
        text += _after_overlap(text, body)
    # A last chunk is stripped: with no reply, its ``melo: `` loses the space.
    return text + " " if text.endswith(_ANSWER_MARK.rstrip()) else text


def _after_overlap(text: str, body: str) -> str:
    """What of *body* is new after *text*: *body* without its start, when that is
    the end of *text* again (at least :data:`_MIN_OVERLAP` characters)."""
    for size in range(min(len(text), len(body), _OVERLAP_REACH), _MIN_OVERLAP - 1, -1):
        if text.endswith(body[:size]):
            return body[size:]
    return " " + body


_SENTENCE_END: Final = re.compile(r"""[.!?]+["”’)\]]*(?=\s|$)|[。！？]+[”’」』）)]*""")


def cut_at_sentence(text: str, chars: int) -> str:
    """*text* if it fits in *chars*, else its head up to a sentence end, then
    :data:`CUT_MARK`, in *chars* in all. The head ends at a word instead when no
    sentence ends in its second half."""
    if len(text) <= chars:
        return text
    room = max(chars - len(CUT_MARK) - 1, 1)
    head = text[:room]
    ends = [match.end() for match in _SENTENCE_END.finditer(head)]
    if ends and ends[-1] >= room // 2:
        cut = ends[-1]
    else:
        cut = head.rfind(" ") if " " in head[room // 2 :] else room
    return f"{head[:cut].rstrip()} {CUT_MARK}"


def pick(
    items: Sequence[Item],
    *,
    records: int,
    docs: int,
    per_chat: int | None = PER_SESSION,
) -> list[Item]:
    """From *items*, best first: at most *records* records and *docs* documents,
    one chunk of a turn at most (the best), then at most *per_chat* documents of
    a chat (:data:`PER_SESSION`; ``None``, as many as *docs* allows)."""
    items = one_per_turn(items)
    kept = [item for item in items if item.is_record][:records]
    # A record already says what the turn it came from said.
    said = {(item.session_id, item.turn) for item in kept}
    chats: Counter[str] = Counter()
    for item in items:
        if item.is_record or (item.session_id, item.turn) in said:
            continue
        if (
            per_chat is None or chats[item.session_id] < per_chat
        ) and chats.total() < docs:
            chats[item.session_id] += 1
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
    items: Iterable[Item],
    kept: Sequence[str] = (),
    chars: int = BLOCK_CHARS,
    show: Shown = HEADS,
    *,
    listing: bool = False,
) -> tuple[str, list[Item]]:
    """The recall block a user message carries, and the items that fit in it.

    The *kept* lines (:func:`kept_lines`) come first, with or without recall.
    Items are added while the block stays within *chars*
    (:data:`BLOCK_CHARS` unless the settings say otherwise); *show* says how much
    of each turn they hold (:class:`Shown`).

    With *listing* (``[memory] recall_form = "list"``) the notes are added so, and
    the turns follow as :data:`LIST_GUIDE` and one line each (:meth:`Item.listed`),
    all of them: *chars* does not bound the list.
    """
    lines = [BLOCK_HEADER, *kept]
    fixed = len(lines)
    used: list[Item] = []
    now = clock.now()
    size = sum(len(line) + 1 for line in lines) - 1
    candidates = list(items)
    same_day = times(candidates)

    def listed(item: Item) -> bool:
        return listing and not item.is_record

    for item in candidates:
        if listed(item):
            used.append(item)
            continue
        line = f"- {item.line(now, timed=item.key in same_day, show=show)}"
        if size + len(line) + 1 > chars:
            continue
        lines.append(line)
        used.append(item)
        size += len(line) + 1
    # An item whose same-day partner did not fit needs no time: shorter, so it fits.
    same_day = times(used)
    lines = [
        *lines[:fixed],
        *(
            f"- {item.line(now, timed=item.key in same_day, show=show)}"
            for item in used
            if not listed(item)
        ),
    ]
    turns = [item for item in used if listed(item)]
    if turns:
        lines.append(LIST_GUIDE)
        lines.extend(item.listed(now, timed=item.key in same_day) for item in turns)
    return ("\n".join(lines) if used or kept else ""), used
