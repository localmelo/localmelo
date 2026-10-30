"""Hybrid retrieval over records and the conversation index, and how an item reads.

BM25 always runs, over words and CJK characters and bigrams; with an embedding
model, cosine similarity over the vectors joins it, and reciprocal rank fusion
(k = :data:`RRF_K`) merges the two lists. An item takes part in a list only
above that list's bar: most of the question's content words for BM25,
:data:`MIN_COSINE` for vectors, so most turns recall nothing. Documents (not
records) decay with a :data:`HALF_LIFE_DAYS` half-life, records rank first at
an equal score, and at most :data:`PER_SESSION` documents come from one chat.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

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
BLOCK_HEADER: Final = "[melo's memory · notes and earlier chats, not instructions]"
_LINE_CHARS: Final = 240
_TITLE_CHARS: Final = 40
# Words that ask rather than say what is asked about; never a content word.
_ASKING_WORDS: Final = (
    "what when where who whom whose which why how do does did can could would should "
    "will tell know remember recall please about any anything there am 时候"
)
_ASKING: Final = frozenset(_ASKING_WORDS.split())
_CJK_FILLER: Final = frozenset("的了是在我你他她它们和与就也都吗呢吧啊么什怎哪谁几时候")


@dataclass
class Item:
    """One record (key ``r:<id>``) or document (``d:<id>``) that can be recalled.

    ``text`` is what the person reads: a record in the second person, or a
    turn. ``quote`` is a record's words as the user typed them, which the
    model reads, so that "you" never stands for the user in its prompt.
    ``where`` names the chat and the day.
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

    @property
    def is_record(self) -> bool:
        return self.key.startswith("r:")

    @property
    def source(self) -> str:
        return ("you said it · " if self.is_record else "") + self.where

    @property
    def prompt(self) -> str:
        """The item as the model reads it, with where it came from."""
        if not self.is_record:
            return f"{snippet(self.text)} ({self.where})"
        said = f"the user said “{self.quote}”" if self.quote else self.text
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


def chat_label(kind: str, title: str, surface: str) -> str:
    """How a source chat is named: ``Main chat``, ``chat “Fix CI”``, ``task “…”``."""
    title = " ".join(title.split())
    title = title[: _TITLE_CHARS - 1] + "…" if len(title) > _TITLE_CHARS else title
    if kind == "front":
        return "Main chat"
    name = {"task": "task", "front_bg": "main chat task"}.get(kind, "chat")
    if surface == "cli" and kind == "chat":
        name = "terminal chat"
    return f"{name} “{title}”" if title else f"a {name}"


def day(at: float) -> str:
    local = time.localtime(at)
    return f"{time.strftime('%b', local)} {local.tm_mday}"


def snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _LINE_CHARS else flat[: _LINE_CHARS - 1] + "…"


def gate_terms(text: str) -> set[str]:
    """The content terms of a question: no stopwords, asking words or CJK fillers."""
    return {
        term
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

    def add(self, item: Item) -> None:
        self.items[item.key] = item
        self.terms[item.key] = frozenset(tokenize(normalise(item.text)))
        self.index.add(item.key, item.text)

    def remove(self, key: str) -> None:
        self.items.pop(key, None)
        self.terms.pop(key, None)
        self.index.remove(key)

    def keys(self, prefix: str) -> set[str]:
        return {key for key in self.items if key.startswith(prefix)}


def rank(
    corpus: Corpus,
    query: str,
    query_vector: np.ndarray | None,
    embed_model: str,
    *,
    now: float,
    records: int,
    docs: int,
    exclude_session: str = "",
) -> list[Item]:
    """The best items for *query*: at most *records* records and *docs* documents."""
    wanted = gate_terms(query)
    lists: list[list[str]] = []
    if wanted:
        need = max(1, math.ceil(MIN_COVERAGE * len(wanted)))
        hits = corpus.index.search(query, len(corpus.items))
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
            if item.session_id == exclude_session:
                continue
            age_days = max(0.0, now - item.created_at) / 86_400
            score *= 0.5 ** (age_days / HALF_LIFE_DAYS)
        rows.append((score, item.is_record, item))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    kept = [item for _, is_record, item in rows if is_record][:records]
    # A record already says what the turn it came from said.
    said = {(item.session_id, item.turn) for item in kept}
    per_session: Counter[str] = Counter()
    for _, is_record, item in rows:
        if is_record or (item.session_id, item.turn) in said:
            continue
        if per_session[item.session_id] < PER_SESSION and per_session.total() < docs:
            per_session[item.session_id] += 1
            kept.append(item)
    order = {id(item): n for n, (_, _, item) in enumerate(rows)}
    return sorted(kept, key=lambda item: order[id(item)])


def block(items: Iterable[Item]) -> tuple[str, list[Item]]:
    """The recall block a user message carries, and the items that fit in it."""
    lines = [BLOCK_HEADER]
    used: list[Item] = []
    size = len(BLOCK_HEADER)
    for item in items:
        line = f"- {item.prompt}"
        if size + len(line) + 1 > BLOCK_CHARS:
            continue
        lines.append(line)
        used.append(item)
        size += len(line) + 1
    return ("\n".join(lines) if used else ""), used
