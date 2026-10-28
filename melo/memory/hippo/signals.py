"""R6-lite's user signals: a remember request and a stated fact.

Only the user-role text is ever read here. The extractor reads a sentence
of the task query (direct mode) or of the session's user message; it never
sees a tool result, a note, a recall block or an assistant message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from localmelo.melo.memory.notes import split_sentences
from localmelo.melo.schema import MemoryCandidate, new_record_id

REMEMBER_MARKERS: tuple[str, ...] = (
    "remember",
    "don't forget",
    "from now on",
    "keep … in memory",
    "记住",
    "别忘了",
    "以后",
    "记下来",
)

USER_STATEMENT_MARKERS: tuple[str, ...] = (
    "I prefer",
    "I use",
    "I live",
    "my … is",
    "call me",
    "we always",
    "I work at",
    "I'm allergic",
    "我喜欢",
    "我习惯",
    "我住在",
    "我的…是",
    "叫我",
    "我在…工作",
    "我对…过敏",
)

# How a marker with an ellipsis reads in real text.
_MARKER_PATTERNS: dict[str, re.Pattern[str]] = {
    "keep … in memory": re.compile(r"\bkeep\b[^.\n]{0,80}\bin memory\b", re.IGNORECASE),
    "my … is": re.compile(r"\bmy\b[^.\n]{0,80}\bis\b", re.IGNORECASE),
    "我的…是": re.compile(r"我的[^。\n]{0,40}是"),
    "我在…工作": re.compile(r"我在[^。\n]{0,40}工作"),
    "我对…过敏": re.compile(r"我对[^。\n]{0,40}过敏"),
}

# A remember request stores a fact only when the sentence states one.
_FIRST_PERSON = re.compile(r"(?<![a-z0-9_])(i|i'm|my|we|our)(?![a-z0-9_])")


@dataclass(frozen=True)
class UserSignals:
    """What the user's own text asks the store to keep."""

    remember: bool = False
    facts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def states_facts(self) -> bool:
        return bool(self.facts)


def _matches(text: str, marker: str) -> bool:
    pattern = _MARKER_PATTERNS.get(marker)
    if pattern is not None:
        return pattern.search(text) is not None
    if marker.isascii():
        return (
            re.search(
                rf"(?<![a-z0-9_']){re.escape(marker.lower())}(?![a-z0-9_])",
                text.lower(),
            )
            is not None
        )
    return marker in text


def _first_person(text: str) -> bool:
    return _FIRST_PERSON.search(text.lower()) is not None or any(
        marker in text for marker in ("我", "我的", "我们")
    )


def classify_user_text(text: str) -> UserSignals:
    """The remember request and the fact sentences of one user message."""
    remember = any(_matches(text, marker) for marker in REMEMBER_MARKERS)
    facts: list[str] = []
    for _offset, sentence in split_sentences(text):
        stripped = sentence.strip()
        if not stripped:
            continue
        stated = any(_matches(stripped, marker) for marker in USER_STATEMENT_MARKERS)
        if not stated and remember:
            # "remember that my birthday is ..." carries its fact after the
            # marker; a bare "remember to buy milk" carries none.
            stated = any(
                _matches(stripped, marker) for marker in REMEMBER_MARKERS
            ) and _first_person(stripped)
            stated = stated and len(stripped) >= 20
        if stated and stripped not in facts:
            facts.append(stripped)
    return UserSignals(remember=remember, facts=tuple(facts))


def user_candidates(
    text: str,
    *,
    task_id: str,
    session_id: str = "",
    provenance: dict[str, Any] | None = None,
) -> list[MemoryCandidate]:
    """One whole candidate per stated fact, in message order."""
    signals = classify_user_text(text)
    candidates: list[MemoryCandidate] = []
    for fact in signals.facts:
        candidates.append(
            MemoryCandidate(
                candidate_id=new_record_id(),
                source="user",
                text=fact[:300],
                remember_by="rule",
                task_id=task_id,
                session_id=session_id,
                provenance={**(provenance or {}), "task_id": task_id},
                remember=True,
            )
        )
    return candidates


__all__ = [
    "REMEMBER_MARKERS",
    "USER_STATEMENT_MARKERS",
    "UserSignals",
    "classify_user_text",
    "user_candidates",
]
