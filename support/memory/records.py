"""Records: what the user said about themselves, kept as one sentence each.

A record is written only from the user's own typed words. The signal lexicon
(:mod:`localmelo.melo.memory.hippo.signals`) finds a statement ("I prefer…",
"my … is", "call me…", "我习惯…") or a remember request ("remember that…",
"记住…") in the typed text, never in attachments, tool output, pages or a
worker's report. Each sentence becomes one record in the second person ("You
prefer short answers."), at most :data:`MAX_RECORD_CHARS` characters, with its
topic, the user's quote and where it was said. Text that looks like a secret
is refused, not stored redacted. A new record of the same subject supersedes
the old one ("call me Ray" after "call me R.").
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.hippo.policy import content_tokens, normalise, overlap
from localmelo.melo.memory.hippo.signals import classify_user_text
from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import new_record_id

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
# The single-valued subjects: a new value replaces the old one.
_SINGLE = {
    key: re.compile(pattern, re.IGNORECASE)
    for key, pattern in (
        ("name", r"\byour name is\b|\byou go by\b|你的名字|你希望被叫作"),
        ("home", r"\byou live\b|你住在"),
        ("job", r"\byou work (?:at|for|in)\b|你在\S{1,20}工作"),
        ("timezone", r"\btime ?zone\b|时区"),
    )
}
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

    A sentence that looks like a secret is never a candidate; it only sets
    the refusal flag, so the turn can say "Not remembered".
    """
    if not typed.strip() or len(typed) > MAX_TYPED_CHARS:
        return [], False
    # A ";" joins two statements as often as it joins two clauses.
    signals = classify_user_text(re.sub(r"\s*[;；]\s*", "\n", typed))
    found: list[tuple[str, str]] = [(f, "user_request") for f in signals.request_facts]
    found += [(f, "user_statement") for f in signals.facts]
    decisions = [
        s.strip()
        for s in re.split(r"(?<=[.!?。！？])\s+|\n", typed)
        if _DECISION.search(s) and not s.rstrip().endswith(("?", "？"))
    ]
    found += [(s, "user_statement") for s in decisions]
    out: list[Candidate] = []
    refused = False
    for sentence, signal in found:
        if sensitive(sentence):
            refused = True
            continue
        text = second_person(sentence)
        if len(normalise(text)) < 4 or any(c.text == text for c in out):
            continue
        topic = topic_of(sentence, request=signal == "user_request")
        out.append(Candidate(text, sentence[:MAX_RECORD_CHARS], topic, signal))
    return out, refused


def _single(text: str) -> str:
    return next((key for key, rx in _SINGLE.items() if rx.search(text)), "")


def decide(text: str, active: Sequence[StoredEntry]) -> tuple[str, int | None]:
    """``("IGNORE", id)`` for a record kept already, ``("UPDATE", id)`` to supersede, or NEW."""
    folded = normalise(text)
    single = _single(text)
    tokens = content_tokens(text)
    best: StoredEntry | None = None
    best_score = 0.0
    for entry in active:
        if normalise(entry.text) == folded:
            return "IGNORE", entry.id
        score = 1.0 if single and _single(entry.text) == single else 0.0
        score = max(score, overlap(tokens, content_tokens(entry.text)))
        if score >= SUPERSEDE_OVERLAP and score >= best_score:
            best, best_score = entry, score
    return ("UPDATE", best.id) if best is not None else ("NEW", None)


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
        active = await store.active_entries(cand.kind)
        action, target = decide(cand.text, active)
        if action == "IGNORE":
            continue
        provenance = {
            **source,
            "topic": cand.topic,
            "quote": cand.quote,
            "embed_model": embed_model if vector is not None else "",
        }
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
