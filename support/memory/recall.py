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
With ``[memory] excerpt_open = "on"`` an excerpt that shows less than the whole turn
(it was cut, or the line is a piece of a longer turn) starts with the turn's id,
``[t123]``, under a line that says to open such a turn with ``open_memory``
(:data:`CUT_GUIDE`); a line that shows the whole turn has no id (:meth:`Item.excerpt`).
With ``[memory] recall_source = "items"`` the items a chat model encoded from the
chats (key ``m:<id>``, :mod:`.hippo_encode`) take the place of the turns: they are
ranked, picked and listed the same way, an item is shown as its title and text
(:meth:`Item.line`), reads as older when a newer item replaced it, and says which
items it conflicts with.
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
# The vector bar of a question that counts, totals or compares, when ``[memory]
# gather`` is ``wide`` or ``guide`` (``gather_min_cosine``, 0.20-0.55): a category
# question sits further from each item that answers it than a single-fact one does.
GATHER_COSINE: Final = 0.40
HALF_LIFE_DAYS: Final = 30.0
PER_SESSION: Final = 2
# What one turn recalls by itself, and the room its block may take. The last two are
# the shipped defaults of ``[memory] recall_chars`` (the block) and
# ``recall_item_chars`` (one recalled turn): a larger room answered more questions
# from earlier chats (#349). They are not the head :data:`TURN_CHARS` of a turn.
AUTO_RECORDS: Final = 3
AUTO_DOCS: Final = 2
BLOCK_CHARS: Final = 4_000
ITEM_CHARS: Final = 900
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
OLDER_ITEM: Final = "older; replaced by a newer item"
# Index document kinds besides a chat's turns (:mod:`.index`).
SIDE_SUMMARY: Final = "side_summary"
AGENT_EXCHANGE: Final = "agent_exchange"
_KEPT_CHARS: Final = 120
_LINE_CHARS: Final = 240
# What a turn's two parts may take together in the block: as many of the user's
# words as the plain 240-character head ``User: …`` of a turn showed before a turn
# said who spoke (those words are extra). In a larger room melo's reply keeps
# :data:`_REPLY_CHARS` of it; a reply is shown only when :data:`_REPLY_MIN` or more
# characters are left for it.
TURN_CHARS: Final = _LINE_CHARS - len("User: ")
_REPLY_CHARS: Final = 60
_REPLY_MIN: Final = 20
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
# The line above a list of memory items, as LIST_GUIDE is above a list of turns.
ITEM_LIST_GUIDE: Final = (
    "Memory items that may hold the answer, closest first. Open the ones you "
    "need with open_memory before you answer from them; a title is not the content."
)
# With ``[memory] gather = "guide"``: the sentence that follows the list's guide line
# when the message asks to count, total or compare (:func:`asks_to_gather`).
GATHER_GUIDE: Final = (
    "This asks for a count, total or comparison: open every listed item that may "
    "belong, search again with other words for each kind not listed yet, and count "
    "only what the items say."
)
CUT_MARK: Final = "… (cut)"
# With ``[memory] excerpt_open = "on"``: the line under the excerpts when one of them
# starts with its turn's id, because it shows less than the whole turn.
CUT_GUIDE: Final = (
    "A line that starts with an id such as [t123] is cut; open it with open_memory "
    "when the answer may be in the rest."
)
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
# A message that asks to count, total, list or compare across things (the English
# words stand alone: "account" is not "count"; the Chinese ones are plain text).
_GATHER: Final = re.compile(
    r"\b(?:how\s+(?:many|much)|totals?|altogether|combined|in\s+all|number\s+of"
    r"|counts?|counted|counting|all\s+(?:the|my)|lists?|different|differences?"
    r"|each|every|both|compar(?:e|es|ed|ing|ison)|(?:more|less|fewer)\s+than"
    r"|longer|faster)\b|多少|几个|几次|一共|总共|合计|所有|每个|哪些|都有",
    re.IGNORECASE | re.ASCII,
)
# A message that speaks of its own speaker: "I", "my", "we", "our", "我" …
_SPEAKER: Final = re.compile(
    r"\b(?:i|i'm|i've|i'd|my|me|mine|myself|we|our|us)\b|我", re.IGNORECASE
)


class Shown(NamedTuple):
    """How much of a recalled turn the model reads, and which part of it.

    *chars* is the room one turn takes (:data:`TURN_CHARS`), in the recall block
    and in the memory search tool's list; text that is no turn takes *chars* plus
    the 6 characters of ``User: `` (240 by default, the plain head it always
    had). With *terms*, the question's content
    terms (``[memory] recall_excerpt = "match"``), each part of a turn shows the
    sentences that hold most of them rather than its head (:func:`turn_said`).
    With *ids* (``[memory] excerpt_open = "on"``) a turn's line that shows less than
    the whole turn starts with the turn's id (:meth:`Item.excerpt`), the id counted
    in the room. The defaults are the head of every turn, as it always was.
    """

    chars: int = TURN_CHARS
    terms: frozenset[str] = frozenset()
    ids: bool = False

    @classmethod
    def of(
        cls, chars: int = TURN_CHARS, question: str | None = None, ids: bool = False
    ) -> Shown:
        """*chars* of room, and the best matching sentences of *question* if given."""
        terms = frozenset(gate_terms(question)) if question else frozenset()
        return cls(chars, terms, ids)


HEADS: Final = Shown()  # the head of every turn, in :data:`TURN_CHARS`
WHOLE: Final = Shown(10**9)  # every turn whole: no room limit


# The source tags of a record whose quote is the user's own words: what the user
# typed (``typed``, or no tag at all, as every record had before the note gate). A
# record tagged ``pointed``, ``result`` or ``inferred`` never reads as the user's.
USER_SOURCES: Final = ("", "typed")


@dataclass
class Item:
    """One record (key ``r:<id>``), document (``d:<id>``) or memory item
    (``m:<id>``) that can be recalled.

    ``text`` is what the person reads: a record in the second person, or a
    turn. ``quote`` is a record's words as the user typed them, which the
    model reads, so that "you" never stands for the user in its prompt.
    ``where`` names the chat and the day. A record's ``replaces`` are the
    turns (session id, turn) whose facts it replaced; a turn's ``said`` is
    what its typed words state about the user, each folded statement with
    its subject key (``""`` for none). ``older`` is how such a turn reads
    when a newer one replaced it: :data:`OLDER` or :data:`OLDER_TURN`. A
    document's ``chunk`` is 0 for the head of a turn and 1 and up for the later
    pieces of a long one (:mod:`.index`); ``more`` is set on the head of a turn that
    has later pieces, when the corpus is filled (:meth:`ChatMemory._sync`), so a
    line can tell, without a lookup, that it shows only a part of its turn.
    ``similarity`` is the cosine of a
    turn to the message when a search matched it by meaning (the list form
    shows it); it is set on a copy for one search, never stored. ``shown`` is
    the line the model read for the item, as the recall block or the memory
    search result worded it (without a list number): :func:`block` and the search
    tool set it on a copy, once, where the line is built, for ``turn.memory``'s
    entries (:meth:`to_json`); it is never stored and takes no part in a
    comparison.

    A memory item has a ``title`` (``text`` is its text), the ``sources`` it came
    from (``(session id, turn)``; its ``turn`` is the oldest of them, so a chat that
    still shows that turn shows it all), ``replaced_by`` (the key of the item that
    replaced it, ``""`` for none) and ``conflicts`` (the ids of the items it
    disagrees with).
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
    more: bool = False
    similarity: float | None = None
    title: str = ""
    sources: tuple[tuple[str, int], ...] = ()
    replaced_by: str = ""
    conflicts: tuple[int, ...] = ()
    # Where a note came from when it is not the user's own words: ``pointed`` (a
    # page or file the user pointed to; ``where`` then reads ``from example.com,
    # Oct 30``) or ``inferred`` (the model's own conclusion). ``""`` for the rest.
    how: str = ""
    shown: str = field(default="", compare=False)

    @property
    def is_record(self) -> bool:
        return self.key.startswith("r:")

    @property
    def is_memory(self) -> bool:
        """Whether this is a memory item (``m:<id>``)."""
        return self.key.startswith("m:")

    @property
    def is_turn(self) -> bool:
        """Whether this is a chat turn's document (``d:<id>``)."""
        return self.key.startswith("d:")

    @property
    def ref(self) -> str:
        """The id a list line shows: ``t123`` for a turn, ``m12`` for an item."""
        return f"{'m' if self.is_memory else 't'}{self.key[2:]}"

    def _conflict(self) -> str:
        """`` (conflicts with m3, m5)`` for an item that disagrees with others."""
        if not self.conflicts:
            return ""
        return f" (conflicts with {', '.join(f'm{n}' for n in self.conflicts)})"

    @property
    def source(self) -> str:
        if self.is_record and self.how == "pointed":
            said = self.where
        elif self.is_record and self.how == "inferred":
            said = f"melo inferred it · {self.where}"
        else:
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

        A record reads as the user's words when it has them and its source is the
        user's (:data:`USER_SOURCES`: words from a page, a file or a task's result
        are never shown as the user's, quote or not); without, a note in the second
        person says it is written to the user ("you" is the user), and one in the
        first person says it is the user's own; a note a page or file gave reads so
        with that host or file name as its place (``from example.com, Oct 30``), and
        one the model inferred says ``melo inferred``.
        A turn says who spoke (:func:`turn_said`). The date has the year when it is not *now*'s
        (the pinned or real time by default), and the time of day with *timed*
        (:func:`times`). *show* says how much of a turn is shown (:class:`Shown`).
        """
        where = self._dated(clock.now() if now is None else now, timed)
        older = f" ({self.older})" if self.older else ""
        if self.is_memory:
            text = _window(self.text, show.terms, show.chars)[0]
            said = f"“{self.title}”: {text}" if self.title else text
            return f"{said} ({where}){older}{self._conflict()}"
        if not self.is_record:
            return self._turn_line(where, older, show)[0]
        if self.how == "inferred":
            said = f"melo inferred “{self.text}”"
        elif self.quote and self.how in USER_SOURCES:
            said = f"the user said “{self.quote}”"
        elif _FIRST_PERSON.match(self.text):
            said = f"the user's note “{self.text}”"
        else:
            said = f"a note to the user “{self.text}”"
        return f"{said} ({where}){older}"

    def excerpt(
        self, now: float | None = None, *, timed: bool = False, show: Shown = HEADS
    ) -> tuple[str, bool]:
        """:meth:`line`, and whether it starts with the turn's id (``[t123] …``).

        That is only with ``show.ids`` (``[memory] excerpt_open = "on"``), for a
        chat turn whose line shows less than the whole turn: :func:`turn_excerpt`
        cut what the user said or melo replied (or left the reply out), or the
        document is a piece of a longer turn (:attr:`chunk` above 0, or :attr:`more`).
        A note, a memory item and a turn shown whole read exactly as :meth:`line`
        writes them. The id is counted in the room (:func:`turn_excerpt`).
        """
        if not (show.ids and self.is_turn):
            return self.line(now, timed=timed, show=show), False
        where = self._dated(clock.now() if now is None else now, timed)
        older = f" ({self.older})" if self.older else ""
        return self._turn_line(where, older, show)

    def _turn_line(self, where: str, older: str, show: Shown) -> tuple[str, bool]:
        """A turn's line, and whether it carries its id (see :meth:`excerpt`)."""
        said, cut = turn_excerpt(self.text, show)
        if show.ids and (cut or self.chunk or self.more):
            prefix = f"[{self.ref}] "
            said = turn_excerpt(self.text, show, len(prefix))[0]
            return f"{prefix}{said} ({where}){older}", True
        return f"{said} ({where}){older}", False

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
        “Where do I plant the tomatoes?” · 0.71``; an item's has its own title and
        ``m`` id (``[m12] …``) and ends with the items it conflicts with.

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
            f"[{self.ref}] {date} · {chat} · “{listed_title(self)}” · "
            f"{meaning}{older}{self._conflict()}"
        )

    def opened_heading(self, now: float | None = None) -> str:
        """The first line of a turn opened in full: its id, date and chat, and the
        older mark when a newer one replaced it."""
        chat, date = self._chat_and_date(clock.now() if now is None else now, False)
        older = f" ({self.older})" if self.older else ""
        return (
            f"[{self.ref}] {date or day(self.created_at)} · {chat}{older}"
            f"{self._conflict()}"
        )

    def to_json(self) -> dict[str, Any]:
        """The item as a ``turn.memory`` entry. ``text`` is the stored text (a
        turn's head, :func:`snippet`); ``shown``, only when the model was shown
        the item, is the exact line it read, whatever the recall settings made it."""
        data: dict[str, Any] = {
            "kind": (
                "record" if self.is_record else "item" if self.is_memory else "chat"
            ),
            "id": int(self.key[2:]),
            "text": self.text if self.is_record else snippet(self.text),
            "source": self.source,
            "session_id": self.session_id,
            "turn": self.turn,
            "at": self.created_at,
        }
        if self.is_memory:
            data["item"] = int(self.key[2:])
            data["sources"] = [[session, turn] for session, turn in self.sources]
        if self.shown:
            data["shown"] = self.shown
        return data


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
    by default); the user's words come first and stay whole when they fit. Long
    words of the user take the room less :data:`_REPLY_CHARS` for melo's reply,
    but in a room of at least :data:`TURN_CHARS` never fewer than that (as far as
    the plain 240-character head ``User: …`` showed them): a value a person
    states late in a long message is where questions about them look. The reply takes what is left when that
    is at least :data:`_REPLY_MIN` characters. Text that is no turn (a summary, a
    task's question) is a plain :func:`snippet`.

    With ``show.terms`` a part that does not fit shows the window that holds the
    most of them instead (:func:`_matching`).
    """
    return turn_excerpt(text, show)[0]


def turn_excerpt(text: str, show: Shown = HEADS, reserve: int = 0) -> tuple[str, bool]:
    """:func:`turn_said`, and whether it shows less than the whole text: the
    user's words or melo's reply are shortened, or the reply is left out.

    With *reserve*, that many characters of the room are taken for something the
    line carries besides (the turn's id, ``[t123] ``): the two parts and a plain
    text keep their own bounds (the user's room of :data:`TURN_CHARS` too), but
    the room they share is *reserve* shorter. Called with a *reserve* on a text
    that is cut already, so the cut stays.
    """
    plain = show.chars + _LINE_CHARS - TURN_CHARS - reserve
    body = text.partition("\n")[2] if text.startswith("Task: ") else text
    if not body.startswith("User: "):
        return snippet(text, plain), len(" ".join(text.split())) > plain
    typed, marker, answer = body[len("User: ") :].partition("\nmelo: ")
    if not marker:
        return snippet(text, plain), len(" ".join(text.split())) > plain
    typed, answer = " ".join(typed.split()), " ".join(answer.split())
    room = show.chars - reserve
    cut = len(typed) + len(answer) > room
    if cut:
        picked = _matching(typed, answer, show, room) if show.terms else None
        if picked is not None:
            typed, answer = picked
        else:
            user_room = show.chars - _REPLY_CHARS
            if show.chars >= TURN_CHARS:
                user_room = max(user_room, TURN_CHARS)
            typed = snippet(typed, min(user_room, room) if answer else room)
            left = room - len(typed)
            answer = snippet(answer, left) if answer and left >= _REPLY_MIN else ""
    said = [f"the user said “{typed}”"] if typed else []
    return "; ".join(said + ([f"melo replied “{answer}”"] if answer else [])), cut


def _matching(
    typed: str, answer: str, show: Shown, room: int
) -> tuple[str, str] | None:
    """The two parts of a turn too long for *room* characters (``show.chars`` less
    what the line carries besides), each as the window that holds the most of
    ``show.terms``; ``None`` when neither part holds one.

    The user's words stay whole when they fit in :data:`_USER_SHARE` of the room
    and melo's reply has the rest; a part that is short leaves its room to the
    other. A part with no term shows its head.
    """
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


# Words, besides the stopwords and the asking words, that point back at the talk or
# ask to go on and name nothing: "go look it up", "do that", "what about the second
# one?". Only verbs, small words and fillers are here, never a noun a question could
# be about. A single letter (what a contraction leaves, "let's", "don't") is one too.
_FOLLOW_UP_TEXT: Final = (
    "go goes going gone went come comes coming let lets continue continues "
    "continuing proceed resume carry keep start begin finish finished done "
    "complete try tries trying retry again look looks looking check checks see "
    "show find search lookup get got make use doing run open ahead up out over "
    "back forward next then now soon first second third last previous other "
    "another ok okay yes yeah yep yup no nope not sure fine right alright good "
    "great cool nice thanks thank thx please pls hi hello hey well so just also "
    "still too yet already really actually maybe perhaps anyway exactly indeed "
    "very much some all both each every only even ever never more further else "
    "thing things stuff one ones same may might must shall here those these him "
    "us if than because say says said don doesn didn isn aren wasn weren won "
    "wouldn couldn shouldn haven hasn hadn"
)
_FOLLOW_UP_WORDS: Final = frozenset(_FOLLOW_UP_TEXT.split())
# The same for Chinese, as characters: a message whose every character is one of
# these (or a ``_CJK_FILLER``) names nothing. "继续", "去查一下", "那个呢", "然后呢".
_FOLLOW_UP_CHARS: Final = frozenset(
    "去来查找看做试再继续接着然后先又还把给让请帮麻烦好行对嗯哦噢哈呀哇个这那些"
    "一二三四五六七八九十下次遍点有没不要用以想说讲告诉第样别其刚才最另外之前过记得知道"
)
# Words of two characters whose characters are not fillers by themselves.
_FOLLOW_UP_PHRASES: Final = ("开始", "结束", "完成", "可以")
_HAN: Final = re.compile("[\u3400-\u4dbf\u4e00-\u9fff]+")
# "We" ("我们") and the first person as the object of a request ("tell me more", "do
# it for me", "帮我查一下", "告诉我") do not make a message about the person.
_NOT_ABOUT_ME: Final = re.compile(
    r"\b(?:tell|show|let|give|help|send|bring|pass|read|ask|for)\s+me\b"
    r"|我们|(?:帮|给|让|替|陪|为|跟|告诉)我"
)
# The person in a message: "I", "me", "my", "mine", "myself", "我", "我的".
_ABOUT_ME: Final = re.compile(r"\b(?:i|me|my|mine|myself)\b|我")
# What a message holds that is its own subject whatever its words: a number, a
# URL, or quoted text (straight or curly quotes, corner brackets, title marks, code).
_OWN_SUBJECT: Final = re.compile(
    r"\d|https?://|www\.|\"[^\"]*\w[^\"]*\"|“[^”]*\w[^”]*”|「[^」]*\w[^」]*」"
    r"|『[^』]*\w[^』]*』|《[^》]*\w[^》]*》|`[^`]*\w[^`]*`"
)


def is_contentless(text: str) -> bool:
    """Whether *text* names nothing to look up: after the stopwords, the asking
    words and the follow-up words (:data:`_FOLLOW_UP_WORDS`, the Chinese
    :data:`_FOLLOW_UP_CHARS`) no word is left, it holds no number, URL or quoted
    text, and it does not speak of the person ("I", "me", "my", "我", "我的").
    "go look it up", "continue", "do that", "tell me more", "去查一下" and "那个呢"
    are contentless; "what is my name?", "what do you know about me?", "我是谁"
    and "我朋友叫什么名字" are not. "Tell me", "for me", "帮我" and "我们" are
    requests and group words, not a question about the person.

    :func:`gate_terms` is not the measure for this: it counts a Chinese run as its
    characters and bigrams (so "继续" is three terms) and it counts "go", "look" and
    "up". A script this does not know (Cyrillic, kana, Hangul) counts as content,
    so such a message is searched as it is. An empty message is contentless.
    """
    folded = normalise(text)
    if _OWN_SUBJECT.search(folded) or _ABOUT_ME.search(_NOT_ABOUT_ME.sub(" ", folded)):
        return False
    for word in _WORD.findall(folded):
        if (
            len(word) > 1
            and word not in STOPWORDS
            and word not in _ASKING
            and word not in _FOLLOW_UP_WORDS
        ):
            return False
    for run in _HAN.findall(folded):
        for phrase in _FOLLOW_UP_PHRASES:
            run = run.replace(phrase, " ")
        if any(
            char not in _CJK_FILLER and char not in _FOLLOW_UP_CHARS
            for char in run
            if not char.isspace()
        ):
            return False
    other = _WORD.sub(" ", _HAN.sub(" ", folded))
    return not any(char.isalpha() for char in other)


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
        words = stemmed(
            normalise(f"{item.title}\n{item.text}" if item.title else item.text)
        )
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
    turn's chunks (:mod:`.index`) share their session and turn. Records and memory
    items stay."""
    seen: set[tuple[str, int]] = set()
    kept: list[Item] = []
    for item in items:
        if item.is_turn:
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
    known: Mapping[str, Item] | None = None,
) -> list[Item]:
    """*items* with each turn whose fact a newer note or turn *replaced* moved
    after the rest and marked older; the newer one comes in ahead of it if it
    was not found (a turn the chat under way still shows, never: the chat has
    it already, :func:`in_view`).

    A memory item a newer item replaced (:attr:`Item.replaced_by`) is moved and
    marked the same way, the newer one looked up in *known* (the items by key).
    The old fact stays as history, as people keep it, but the newer one wins.
    """
    keys = {item.key for item in items}
    fresh: list[Item] = []
    stale: list[Item] = []
    for item in items:
        if item.is_memory:
            newer = _newest(item, known or {}) if item.replaced_by else None
            if not item.replaced_by:
                fresh.append(item)
                continue
            if (
                newer is not None
                and newer.key not in keys
                and not in_view(newer, exclude_session, keep_from_turn)
            ):
                keys.add(newer.key)
                fresh.append(newer)
            stale.append(dataclasses.replace(item, older=OLDER_ITEM))
            continue
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


def _newest(item: Item, known: Mapping[str, Item]) -> Item | None:
    """The item that replaced *item*, or the one that replaced that, and so on:
    ``None`` when *known* does not hold it."""
    seen = {item.key}
    newer = known.get(item.replaced_by)
    while newer is not None and newer.replaced_by and newer.replaced_by not in seen:
        seen.add(newer.key)
        further = known.get(newer.replaced_by)
        if further is None:
            break
        newer = further
    return newer


def asks_about_past(message: str) -> bool:
    """Whether *message* asks about an older value ("before", "used to", "以前" …)."""
    return _PAST.search(message) is not None


def asks_to_gather(message: str) -> bool:
    """Whether *message* asks to count, total, list or compare across things ("how
    many", "in total", "each", "compared", "一共", "哪些" …) **and** speaks of its
    own speaker ("I", "my", "we", "我" …): an answer about the person's own life
    then needs every item of a kind, not the one that matches best. A generic
    prompt ("how many seconds are in a day", "list the planets") does not."""
    return bool(_GATHER.search(message)) and bool(_SPEAKER.search(message))


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
    if item.is_memory:
        return word_cut(" ".join(item.title.split()), LIST_TITLE_CHARS) or "…"
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
        if item.is_record or (item.is_turn and (item.session_id, item.turn) in said):
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
    guide: str = LIST_GUIDE,
) -> tuple[str, list[Item]]:
    """The recall block a user message carries, and the items that fit in it, each
    a copy that holds the line the model reads (:attr:`Item.shown`).

    The *kept* lines (:func:`kept_lines`) come first, with or without recall.
    Items are added while the block stays within *chars*
    (:data:`BLOCK_CHARS` unless the settings say otherwise); *show* says how much
    of each turn they hold (:class:`Shown`).

    With *listing* (``[memory] recall_form = "list"``) the notes are added so, and
    the turns follow as :data:`LIST_GUIDE` and one line each (:meth:`Item.listed`),
    all of them: *chars* does not bound the list, and *guide* is the line above it.

    With ``show.ids`` (``[memory] excerpt_open = "on"``) a turn whose excerpt shows
    less than the whole turn starts with its id (:meth:`Item.excerpt`), and
    :data:`CUT_GUIDE` follows the excerpts when at least one line has an id. That
    line counts toward *chars* as the others do, so the block stays within it.
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

    guided = False  # the line under the excerpts is counted once, with the first id
    for item in candidates:
        if listed(item):
            used.append(item)
            continue
        said, carries = item.excerpt(now, timed=item.key in same_day, show=show)
        line = f"- {said}"
        extra = len(CUT_GUIDE) + 1 if carries and not guided else 0
        if size + len(line) + 1 + extra > chars:
            continue
        lines.append(line)
        used.append(item)
        size += len(line) + 1 + extra
        guided = guided or carries
    # An item whose same-day partner did not fit needs no time: shorter, so it fits.
    # What is written here is what the model reads; each item keeps its own line.
    same_day = times(used)
    read: dict[int, str] = {}
    lines = lines[:fixed]
    opens = False
    for n, item in enumerate(used):
        if not listed(item):
            read[n], carries = item.excerpt(now, timed=item.key in same_day, show=show)
            opens = opens or carries
            lines.append(f"- {read[n]}")
    if opens:
        lines.append(CUT_GUIDE)
    turns = [n for n, item in enumerate(used) if listed(item)]
    if turns:
        lines.append(guide)
        for n in turns:
            read[n] = used[n].listed(now, timed=used[n].key in same_day)
            lines.append(read[n])
    text = "\n".join(lines) if used or kept else ""
    return text, [
        dataclasses.replace(item, shown=read[n]) for n, item in enumerate(used)
    ]
