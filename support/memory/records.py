"""Records: what the user said about themselves or their work, one sentence each.

A record is written two ways, and neither asks the user to confirm. When the
user asks ("remember that…", "记住…", "别忘了…", "帮我记一下…"), code keeps the
fact the request carries (:func:`candidates`, the signal lexicon of
:mod:`localmelo.melo.memory.hippo.signals`), in the second person ("You
prefer short answers."). Otherwise the chat's model keeps a note with the
``remember`` tool when it judges something worth keeping; the note's content
words must come from what the user typed in that chat or from a finished
task's result shown there (:func:`unsupported`), never from a page or tool
output alone. A plain statement is not kept by rule. Each record has at most
:data:`MAX_RECORD_CHARS` characters, its topic, the user's quote and where it
was said. Text that looks like a secret is refused, not stored redacted. A new
record of the same subject supersedes the old one ("call me Ray" after "call
me R.", "我的健身教练是小宇" after "…是阿凯"): the old one is kept, not recalled,
and comes back with an undo of the new one. The subject is what a note is
about without its value (:func:`subject`); with an embedding model, a note
close in meaning that differs from an old one in one short span is the same
subject too (:func:`value_apart`). A new record lists the turns whose facts it
replaced (``provenance["replaces"]``), so recall can rank them below it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Final

import numpy as np

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.bm25 import tokenize
from localmelo.melo.memory.hippo.policy import (
    STOPWORDS,
    content_tokens,
    normalise,
    overlap,
)
from localmelo.melo.memory.hippo.signals import classify_user_text
from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.notes import split_sentences
from localmelo.melo.schema import new_record_id
from localmelo.support.memory.recall import stem, unit

TOPICS: Final = ("about", "preferences", "people", "projects", "decisions")
TOPIC_LABELS: Final = {
    "about": "About you",
    "preferences": "Preferences",
    "people": "People",
    "projects": "Projects",
    "decisions": "Decisions",
}
# The long-term kind each topic is stored as.
TOPIC_KINDS: Final = {
    "about": "profile",
    "preferences": "profile",
    "people": "semantic",
    "projects": "episodic",
    "decisions": "semantic",
}
MAX_RECORD_CHARS: Final = 300
# A message longer than this is a task or a paste, not a statement to keep.
MAX_TYPED_CHARS: Final = 2_000
# Content overlap at which a new record replaces an old one.
SUPERSEDE_OVERLAP: Final = 0.6
# With vectors: how close in meaning, and how long the one changed span (the
# value) may be, for a note to replace an old one of other subject words.
NEAR_COSINE: Final = 0.8
VALUE_TOKENS: Final = 8
NOT_KEPT: Final = "it looked like a password or key"
# What a list of notes says when it has none, and how memory starts.
EMPTY: Final = (
    "Nothing remembered yet. melo keeps a note when you tell it something about "
    "yourself in the web console or the app, such as “I prefer short answers”, "
    "or ask it to remember something."
)

_DECISION = re.compile(
    r"\b(we(?:'ve| have)? decided|let's go with|we(?:'ll| will) go with)\b"
    r"|我们决定|决定用|就用",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    r"\b(prefer|like|love|hate|dislike|from now on)\b|喜欢|习惯|偏好|讨厌|以后",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(I|I'm|my|me|mine|we|our|us)\b|我", re.IGNORECASE)
# Subjects a verb names, in the user's words, a record's or the model's
# ("I live in", "You live in", "The user lives in", "我住在", "用户住在").
_WHO = r"\b(?:i|you|user)(?: now| currently)?"
_SUBJECTS: Final = tuple(
    (key, re.compile(pattern))
    for key, pattern in (
        ("timezone", r"\btime ?zone\b|时区"),
        (
            "name",
            r"^(?:please )?call me\b|\bgo(?:es)? by\b"
            rf"|{_WHO}(?:'m| am|'re| are| is) (?:called|named)\b"
            r"|^(?:请)?叫我|希望被(?:叫作|称为)|(?:我|你|用户)的?名字",
        ),
        (
            "home",
            rf"{_WHO} (?:live|lives|reside|resides)\b|(?:我|你|用户)(?:现在)?住在",
        ),
        (
            "job",
            rf"{_WHO} works? (?:at|for)\b"
            r"|(?:我|你|用户)(?:现在)?在[^，,。\s]{1,20}?(?:工作|上班)",
        ),
    )
)
# "my X is Y" / "the user's X is Y" / "我的X是Y": X is the subject.
_OWNED = re.compile(
    r"^(?:my|your|our|(?:the )?user['’]s) ((?:[\w'’-]+ ){0,5}?[\w'’-]+) "
    r"(?:is|are|was|were)\b"
)
_OWNED_ZH = re.compile(
    r"^(?:我|你|用户)的?([^\s，,。是叫姓]{1,12}?)(?:现在|也)?(?:是|叫|姓)"
)
_ARTICLES: Final = frozenset({"a", "an", "the"})
_TOKEN = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
# Request phrases the shared lexicon lacks, read as "remember that":
# "帮我记一下：…", "keep in mind that …", "make a note that …".
_ALSO_ASKS = re.compile(
    r"^(?:please\s+)?(?:keep in mind|make a note)(?:\s+that)?\b[\s,:]*"
    r"|^(?:请)?(?:帮我)?记一下[\s,:：，]*",
    re.IGNORECASE,
)
# Words a note adds when it restates what was said ("The user prefers…",
# "You go by…"); they are never what has to be found in the user's words.
_FRAMING: Final = frozenset(
    [
        "user",
        "users",
        "prefer",
        "prefers",
        "like",
        "likes",
        "love",
        "loves",
        "want",
        "wants",
        "go",
        "goes",
        "call",
        "called",
        "name",
        "named",
        "remember",
        "note",
    ]
)
_CJK_FILLER: Final = frozenset("的了是在我你他她它们和与就也都吗呢吧啊么")
_CJK_FRAMING = re.compile("希望被叫作|希望被称为|用户")
_POSSESSIVE = re.compile(r"(?<=\w)['’]s\b")
_WORD = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
# The never-kept classes: passwords and codes, card and ID numbers.
_SECRET_WORDS = re.compile(
    r"\b(passwords?|passcodes?|passwd|pin|cvv|security code|one-time code|otp)\b"
    r"|密码|口令|验证码",
    re.IGNORECASE,
)
_ID_NUMBER = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4}|\d{17}[\dXx])(?![\dA-Za-z])")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

# The first person in the user's words, in the second person in a record.
_SWAPS: Final = (
    (re.compile(r"^(?:please\s+)?call me\b", re.IGNORECASE), "you go by"),
    (re.compile(r"\b(?:I am|I'm)\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bI was\b", re.IGNORECASE), "you were"),
    (re.compile(r"\b(?:I have|I've)\b", re.IGNORECASE), "you have"),
    (re.compile(r"\b(?:I will|I'll)\b", re.IGNORECASE), "you will"),
    (re.compile(r"\b(?:we are|we're)\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bI\b"), "you"),
    (re.compile(r"\bmyself\b", re.IGNORECASE), "yourself"),
    (re.compile(r"\b(?:mine|ours)\b", re.IGNORECASE), "yours"),
    (re.compile(r"\b(?:my|our)\b", re.IGNORECASE), "your"),
    (re.compile(r"\b(?:me|we)\b", re.IGNORECASE), "you"),
    (re.compile(r"\bus\b"), "you"),
    (re.compile(r"^(?:请)?叫我"), "你希望被叫作"),
    (re.compile("我们"), "你们"),
    (re.compile("我"), "你"),
)


@dataclass(frozen=True)
class Candidate:
    """One record to write: its text, the user's quote, topic and signal."""

    text: str
    quote: str
    topic: str
    admitted_by: str

    @property
    def kind(self) -> str:
        return TOPIC_KINDS[self.topic]


@dataclass(frozen=True)
class Written:
    """A record a turn wrote; ``replaced`` is the id it superseded, if any."""

    id: int
    text: str
    topic: str
    replaced: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "topic": self.topic}


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char) * (2 if index % 2 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def sensitive(text: str) -> bool:
    """Whether *text* holds a secret: a key, a password or code, a card or ID number."""
    if redact_secrets(text)[1] or _SECRET_WORDS.search(text) or _ID_NUMBER.search(text):
        return True
    return any(_luhn(re.sub(r"\D", "", m.group(0))) for m in _CARD.finditer(text))


def second_person(sentence: str) -> str:
    """The user's *sentence* as a record: second person, one sentence, bounded."""
    text = " ".join(sentence.split()).strip(" -—:：,，")
    for pattern, replacement in _SWAPS:
        text = pattern.sub(replacement, text)
    return finished(text)


def finished(text: str) -> str:
    """*text* as a record reads: one line, a capital, an end mark, bounded."""
    text = " ".join(text.split()).strip(" -—:：,，")
    text = text[:1].upper() + text[1:]
    if len(text) > MAX_RECORD_CHARS:
        text = text[: MAX_RECORD_CHARS - 1].rstrip() + "…"
    elif text and text[-1].isascii() and text[-1].isalnum():
        text += "."
    elif text and not re.search(r"[.!。！…)）\"”]$", text):
        text += "。"
    return text


def topic_of(sentence: str, *, request: bool) -> str:
    """The topic of a fact the user stated (*request*: after "remember that")."""
    if _DECISION.search(sentence):
        return "decisions"
    if _PREFERENCE.search(sentence):
        return "preferences"
    if _FIRST_PERSON.search(sentence):
        return "about"
    return "people" if request else "about"


def candidates(typed: str) -> tuple[list[Candidate], bool]:
    """The records the user's *typed* text asks for, and whether one was refused.

    Only a request to remember counts ("remember that…", "记住…", "别忘了…",
    "帮我记一下…"); a plain statement is the model's to keep. A sentence that
    looks like a secret, a request's or a statement's, is never a candidate;
    it only sets the refusal flag, so the turn can say "Not remembered".
    """
    if not typed.strip() or len(typed) > MAX_TYPED_CHARS:
        return [], False
    # A ";" joins two statements as often as it joins two clauses.
    joined = re.sub(r"\s*[;；]\s*", "\n", typed)
    joined = "\n".join(
        _ALSO_ASKS.sub("remember that ", sentence, count=1)
        for _, sentence in split_sentences(joined)
    )
    signals = classify_user_text(joined)
    found = list(signals.request_facts)
    # A request whose words after the marker did not read as a fact alone
    # ("Remember, I'm allergic to peanuts.") keeps the whole sentence.
    found += [f for f in signals.facts if classify_user_text(f).remember]
    # A statement is not kept by rule, but one with a secret still says so.
    refused = any(sensitive(s) for s in signals.facts)
    out: list[Candidate] = []
    for sentence in found:
        if sensitive(sentence):
            refused = True
            continue
        text = second_person(sentence)
        if len(normalise(text)) < 4 or any(c.text == text for c in out):
            continue
        topic = topic_of(sentence, request=True)
        out.append(Candidate(text, sentence[:MAX_RECORD_CHARS], topic, "user_request"))
    return out, refused


def _stem(word: str) -> str:
    """*word* without its endings, twice, so "meetings" finds "meets"."""
    return stem(stem(word))


def note_terms(text: str) -> list[str]:
    """The content words a note must find in the evidence, in order."""
    terms: list[str] = []
    # "the user's name" restates; its "s" is not a word the user typed.
    text = _POSSESSIVE.sub("", _CJK_FRAMING.sub(" ", text))
    for term in tokenize(normalise(text)):
        if term.isascii():
            if term in STOPWORDS or term in _FRAMING:
                continue
            term = _stem(term)
        elif _CJK_FILLER.intersection(term):
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _evidence(texts: Sequence[str]) -> set[str]:
    """The words of *texts*, stemmed; two adjacent words joined ("3 pm"); and
    Chinese pairs with one character between them ("下周三要去" gives "三去")."""
    found: set[str] = set()
    for text in texts:
        folded = normalise(text)
        terms = tokenize(folded)
        found.update(_stem(t) if t.isascii() else t for t in terms)
        words = _WORD.findall(folded)
        found.update(a + b for a, b in zip(words, words[1:], strict=False))
        for run in _CJK_RUN.findall(folded):
            found.update(run[i] + run[i + 2] for i in range(len(run) - 2))
    return found


def unsupported(note: str, evidence: Sequence[str]) -> list[str]:
    """The content words of *note* that none of the *evidence* texts holds.

    A word counts when the evidence has it after stemming, or a word at most
    two letters longer that starts with it ("named" finds "name"); the words a
    note adds to restate a fact ("The user prefers…") need not be found.
    ``[]`` means grounded.
    """
    words = _evidence(evidence)
    ascii_words = [w for w in words if w.isascii()]
    return [
        term
        for term in note_terms(note)
        if term not in words
        and not (
            term.isascii()
            and len(term) >= 3
            and any(w.startswith(term) and len(w) <= len(term) + 2 for w in ascii_words)
        )
    ]


def quote_for(note: str, said: Sequence[str]) -> str:
    """The sentence of *said* that holds most of *note*'s words: the user's quote."""
    terms = set(note_terms(note))
    best, score = "", 0
    for text in said:
        for _, sentence in split_sentences(re.sub(r"\s*[;；]\s*", "\n", text)):
            hits = len(terms - set(unsupported(note, [sentence])))
            if hits > score:
                best, score = sentence, hits
    return best[:MAX_RECORD_CHARS]


def subject(text: str) -> str:
    """What *text* is about without its value; ``""`` when it names none.

    "I live in Lisbon", "You live in Utrecht" and "The user lives in Utrecht"
    are ``home``; "my weekly sync is on Friday" is ``weekly sync``; "我的健身
    教练是小宇" is ``健身教练``. Look-alikes keep apart: "flight to porto" is
    not "flight to faro", "书法课" is not "游泳课".
    """
    folded = normalise(text)
    for key, pattern in _SUBJECTS:
        if pattern.search(folded):
            return key
    if found := _OWNED.match(folded):
        words = _TOKEN.findall(_POSSESSIVE.sub("", found.group(1)))
        return " ".join(stem(w) for w in words if w not in _ARTICLES)
    found = _OWNED_ZH.match(folded)
    return found.group(1) if found else ""


def _weight(tokens: Sequence[str]) -> float:
    """Words count 1, Chinese characters 1/2 (a word is about two)."""
    return sum(1.0 if t.isascii() else 0.5 for t in tokens)


def value_apart(text: str, other: str) -> bool:
    """Whether *text* says what *other* says but one value: one changed span of
    at most :data:`VALUE_TOKENS` words (or Chinese characters), in a frame of
    at least two content words they share that outweighs it ("You go to the
    gym on Mondays/Fridays"; not "你习惯在湖边公园跑步/早餐吃豆浆")."""
    left, right = _TOKEN.findall(normalise(text)), _TOKEN.findall(normalise(other))
    changed = [
        op
        for op in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    if len(changed) != 1:
        return False
    _, start, end, other_start, other_end = changed[0]
    frame = _weight(
        [
            t
            for t in left[:start] + left[end:]
            if t not in STOPWORDS and t not in _CJK_FILLER
        ]
    )
    value = max(_weight(left[start:end]), _weight(right[other_start:other_end]))
    longest = max(end - start, other_end - other_start)
    return longest <= VALUE_TOKENS and frame >= max(2.0, value)


def _near(vector: Sequence[float] | None, model: str, entry: StoredEntry) -> bool:
    """Whether *vector* (of *model*) and *entry*'s are close in meaning."""
    mine, theirs = unit(vector), unit(entry.vector)
    if mine is None or theirs is None or not model or entry.embed_model != model:
        return False
    return mine.shape == theirs.shape and float(np.dot(mine, theirs)) >= NEAR_COSINE


def decide(
    text: str,
    active: Sequence[StoredEntry],
    *,
    kind: str = "",
    vector: Sequence[float] | None = None,
    embed_model: str = "",
) -> tuple[str, int | None]:
    """``("IGNORE", id)`` for a record kept already, ``("UPDATE", id)`` to supersede, or NEW.

    An old record of the same :func:`subject` is replaced, whatever its topic;
    two records that name different subjects never replace each other. Else,
    with vectors, one close in meaning and :func:`value_apart`; else one of
    the same *kind* that shares most content words ("said again in other words").
    """
    folded = normalise(text)
    mine = subject(text)
    tokens = content_tokens(text)
    best: StoredEntry | None = None
    best_score = 0.0
    for entry in active:
        if normalise(entry.text) == folded:
            return "IGNORE", entry.id
        theirs = subject(entry.text)
        if mine and theirs:
            score = 1.0 if mine == theirs else 0.0
        elif _near(vector, embed_model, entry) and value_apart(text, entry.text):
            score = 1.0
        elif not kind or entry.kind == kind:
            score = overlap(tokens, content_tokens(entry.text))
        else:
            score = 0.0
        if score >= SUPERSEDE_OVERLAP and score >= best_score:
            best, best_score = entry, score
    return ("UPDATE", best.id) if best is not None else ("NEW", None)


def _replaces(old: StoredEntry, source: dict[str, Any]) -> list[list[Any]]:
    """The turns whose facts a record replacing *old* replaces: *old*'s and
    those *old* replaced, but never the new record's own turn."""
    turns = [
        [str(t[0]), int(t[1])]
        for t in old.provenance.get("replaces") or []
        if isinstance(t, (list, tuple)) and len(t) == 2
    ]
    turns.append([old.session_id, int(old.provenance.get("turn") or 0)])
    own = [str(source.get("session_id", "")), int(source.get("turn") or 0)]
    return [t for n, t in enumerate(turns) if t != own and t not in turns[:n]]


async def write(
    store: SqliteLongTerm,
    found: Sequence[Candidate],
    vectors: Sequence[list[float] | None],
    *,
    source: dict[str, Any],
    embed_model: str,
) -> list[Written]:
    """Write each candidate as a record of *source* (session, turn, chat, surface)."""
    written: list[Written] = []
    for cand, vector in zip(found, vectors, strict=True):
        active = await store.active_entries()
        action, target = decide(
            cand.text, active, kind=cand.kind, vector=vector, embed_model=embed_model
        )
        if action == "IGNORE":
            continue
        provenance = {
            **source,
            "topic": cand.topic,
            "quote": cand.quote,
            "embed_model": embed_model if vector is not None else "",
        }
        old = next((e for e in active if e.id == target), None)
        if old is not None and (replaces := _replaces(old, source)):
            provenance["replaces"] = replaces
        entry_id, actual = await store.apply(
            text=cand.text,
            kind=cand.kind,
            source="user",
            admitted_by=cand.admitted_by,
            provenance=provenance,
            task_id="",
            session_id=str(source.get("session_id", "")),
            created_at=time.time(),
            embedding=vector,
            action=action,
            target_id=target,
            reason="said again in other words" if target else "new",
            source_ids=(),
            keep_text=False,
            candidate_id=new_record_id(),
            record={
                "topic": cand.topic,
                "quote": cand.quote,
                "embed_model": provenance["embed_model"],
            },
        )
        if entry_id is not None:
            replaced = target if actual == "UPDATE" else None
            written.append(Written(entry_id, cand.text, cand.topic, replaced))
    return written
