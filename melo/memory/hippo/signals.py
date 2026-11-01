"""R6-lite's user signals: a remember request and a stated fact.

Only the user-role text is ever read here. The extractor reads a sentence
of the task query (direct mode) or of the session's user message; it never
sees a tool result, a note, a recall block or an assistant message.
"""

from __future__ import annotations

import re
import time
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

# A remember request stores a fact only when the sentence states one. The
# patterns are declarative and stable; a to-do ("remember to …"), a request
# ("I need you to …") and a question ("do you remember …?") are not facts.
_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(my|our)\b[^.\n]{0,80}\b(is|are|was|were)\b", re.IGNORECASE),
    re.compile(
        r"\bI\s+(live|work|prefer|use|like|am allergic|am based|always)\b",
        re.IGNORECASE,
    ),
    re.compile(r"我的[^。\n]{0,40}是"),
    re.compile(r"我(住在|喜欢|习惯|在[^。\n]{0,20}工作|对[^。\n]{0,20}过敏)"),
)
_QUESTION_END = ("?", "？")

# A remember request also states the fact it carries after the marker: "Remember
# that the launch code is NECTAR-42." is one request, and the clause after
# "that" is the fact itself. These predicates keep a bare to-do or a
# pointer ("remember this for later") out of the store: decision 4 prefers
# precision over recall.
_DECLARATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(is|are|was|were|has|have|had|means|equals)\b", re.IGNORECASE),
    re.compile(
        r"\b(contains?|includes?|starts?|ends?|begins?|costs?|opens?|closes?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(lives?|works?|prefers?|likes?|uses?|moved|scheduled|due|located)\b",
        re.IGNORECASE,
    ),
    re.compile(r"[是为有叫在位于等于包含需要开始结束使用存放]"),
)
# The words and punctuation a request may put between its marker and the
# fact: "remember,", "remember that", "记住：", "记住，".
_CLAUSE_LEAD = re.compile(r"^[\s,:;，：；\-—–、]+", re.UNICODE)
_CLAUSE_THAT = re.compile(r"^(?:the\s+fact\s+)?that\b[\s,:;，：；]*", re.IGNORECASE)
# "Remember to buy milk." is a to-do, not a fact.
_INFINITIVE_LEAD = re.compile(r"^(?:to\b|去)", re.IGNORECASE)

# A meta sentence is not a stable fact about the user (review finding 6):
# "my task is …" and "我的任务是…" describe the work, not the person.
_META_NOUNS = frozenset(
    {"task", "question", "request", "goal", "plan", "problem", "issue", "job", "point"}
)
_META_NOUNS_CJK = ("任务", "问题", "请求", "目标", "计划")
_MY_IS = re.compile(r"\bmy\b([^.\n]{0,80}?)\bis\b", re.IGNORECASE)
_MY_IS_CJK = re.compile(r"我的([^。\n]{0,40}?)是")
_MY_MARKERS = ("my … is", "我的…是")


@dataclass(frozen=True)
class UserSignals:
    """What the user's own text asks the store to keep."""

    remember: bool = False
    facts: tuple[str, ...] = field(default_factory=tuple)
    # The facts a remember request carries after its marker; ``admit`` writes
    # them with ``user_request`` beside the run's digest.
    request_facts: tuple[str, ...] = field(default_factory=tuple)

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


def _marker_span(sentence: str, marker: str) -> tuple[int, int] | None:
    """Where *marker* sits in *sentence*, or ``None``."""
    pattern = _MARKER_PATTERNS.get(marker)
    if pattern is not None:
        match = pattern.search(sentence)
        return None if match is None else (match.start(), match.end())
    if marker.isascii():
        match = re.search(
            rf"(?<![a-z0-9_']){re.escape(marker)}(?![a-z0-9_])",
            sentence,
            re.IGNORECASE,
        )
        return None if match is None else (match.start(), match.end())
    start = sentence.find(marker)
    return None if start < 0 else (start, start + len(marker))


def request_clause(sentence: str) -> str:
    """The fact a remember request carries after its earliest marker."""
    spans = [
        span
        for marker in REMEMBER_MARKERS
        if (span := _marker_span(sentence, marker)) is not None
    ]
    if not spans:
        return ""
    start = min(spans)[1]
    clause = _CLAUSE_LEAD.sub("", sentence[start:])
    clause = _CLAUSE_THAT.sub("", clause)
    return clause.strip()


def _is_meta_statement(text: str) -> bool:
    """Whether a ``my … is`` sentence names the work, not the person."""
    for match in _MY_IS.finditer(text):
        if set(re.findall(r"[a-z]+", match.group(1).lower())) & _META_NOUNS:
            return True
    for match in _MY_IS_CJK.finditer(text):
        if any(noun in match.group(1) for noun in _META_NOUNS_CJK):
            return True
    return False


def _declarative(text: str) -> bool:
    return not text.rstrip().endswith(_QUESTION_END)


def _states_a_fact(text: str) -> bool:
    """Whether one sentence states a stable first-person fact."""
    if not _declarative(text) or _is_meta_statement(text):
        return False
    return any(pattern.search(text) for pattern in _FACT_PATTERNS)


def _states_a_statement_marker(text: str) -> bool:
    """Whether *text* carries a statement marker that is not a meta sentence."""
    if _is_meta_statement(text):
        return any(
            _matches(text, marker)
            for marker in USER_STATEMENT_MARKERS
            if marker not in _MY_MARKERS
        )
    return any(_matches(text, marker) for marker in USER_STATEMENT_MARKERS)


def _states_a_declarative_fact(text: str) -> bool:
    """Whether a remember request's clause states a fact, not a to-do.

    The predicate list is wider than :data:`_FACT_PATTERNS`, which is
    first-person only: "the launch code is NECTAR-42" is a third-party fact.
    """
    if not text or not _declarative(text) or _is_meta_statement(text):
        return False
    if _INFINITIVE_LEAD.match(text):
        return False
    return _states_a_fact(text) or any(
        pattern.search(text) for pattern in _DECLARATIVE_PATTERNS
    )


def classify_user_text(text: str) -> UserSignals:
    """The remember request and the fact sentences of one user message.

    Every marker needs a declarative sentence: a question never asks the
    store to remember and never states a fact (review finding 6). ``my … is``
    and ``我的…是`` also refuse a meta noun, because a task, question or
    plan is not a stable fact. A remember request's clause after the marker
    is a ``request_facts`` entry, written with ``user_request`` beside the
    run's digest; a direct statement stays a ``facts`` entry, written at
    once with ``user_statement``.
    """
    remember = False
    facts: list[str] = []
    request_facts: list[str] = []
    for _offset, sentence in split_sentences(text):
        stripped = sentence.strip()
        if not stripped or not _declarative(stripped):
            continue
        sentence_remember = any(
            _matches(stripped, marker) for marker in REMEMBER_MARKERS
        )
        remember = remember or sentence_remember
        if sentence_remember:
            clause = request_clause(stripped)
            if clause and _states_a_declarative_fact(clause):
                if clause not in request_facts:
                    request_facts.append(clause)
                continue
        stated = _states_a_statement_marker(stripped)
        if not stated and sentence_remember:
            # "remember that my birthday is ..." carries its fact after the
            # marker when the clause did not read as one; "remember to buy
            # milk" carries none.
            stated = _states_a_fact(stripped)
        if stated and stripped not in facts:
            facts.append(stripped)
    return UserSignals(
        remember=remember,
        facts=tuple(facts),
        request_facts=tuple(request_facts),
    )


def _user_candidate(
    fact: str,
    *,
    task_id: str,
    session_id: str,
    provenance: dict[str, Any] | None,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=new_record_id(),
        source="user",
        text=fact[:300],
        remember_by="rule",
        task_id=task_id,
        session_id=session_id,
        provenance={
            "source_ids": [],
            **(provenance or {}),
            "task_id": task_id,
        },
        created_at=time.time(),
        remember=True,
    )


def user_candidates(
    text: str,
    *,
    task_id: str,
    session_id: str = "",
    provenance: dict[str, Any] | None = None,
    typed_assertions: bool = False,
) -> list[MemoryCandidate]:
    """One whole candidate per stated fact, in message order."""
    signals = classify_user_text(text)
    facts = list(signals.facts)
    if typed_assertions:
        from localmelo.melo.memory.assertions import parse_assertion

        facts = []
        for _, sentence in split_sentences(text):
            fact = sentence.strip()
            if fact in facts:
                continue
            if fact in signals.facts or (len(fact) <= 300 and parse_assertion(fact)):
                facts.append(fact)
    return [
        _user_candidate(
            fact,
            task_id=task_id,
            session_id=session_id,
            provenance=provenance,
        )
        for fact in facts
    ]


def request_candidates(
    text: str,
    *,
    task_id: str,
    session_id: str = "",
    provenance: dict[str, Any] | None = None,
) -> list[MemoryCandidate]:
    """One candidate per fact a remember request carries, in message order.

    The agent stages these with ``remember`` set and admits them at the end
    of the run with ``user_request``, beside the final digest.
    """
    signals = classify_user_text(text)
    return [
        _user_candidate(
            fact,
            task_id=task_id,
            session_id=session_id,
            provenance=provenance,
        )
        for fact in signals.request_facts
    ]


__all__ = [
    "REMEMBER_MARKERS",
    "USER_STATEMENT_MARKERS",
    "UserSignals",
    "classify_user_text",
    "request_candidates",
    "user_candidates",
]
