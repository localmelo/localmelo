"""Recognising a recurring task: the normalized query, keys and template ids.

A *routine* is a recurring task, identified by its key: the SHA-256 of a
text and a place (:func:`routine_key`). For a task given as a query, the
text is the query normalized by :func:`normalize_query` and the place is
the directory the run's tools work in (:func:`run_place`); for a named
routine, the text is ``routine:<name>`` and the place is empty, so its
runs are one routine in any folder.

A *template* belongs to one routine, one model and one set of the run
options that shape its length (:data:`LENGTH_OPTIONS`); :func:`template_id`
names it. :func:`find_similar` finds a stored routine whose query is close
to a new one, for a hint only. Everything here is pure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.bm25 import BM25Index

DATE: Final = "<date>"
TIME: Final = "<time>"

# The longest query text a routine keeps, in characters.
MAX_QUERY_TEXT: Final = 2000

# Model ids that do not name a model: every model behind one URL would
# share a template, so such a run gets at most level 1.
UNKNOWN_MODELS: Final = frozenset({"", "default"})

# The run options that shape a run's length (keys of RunResult.config).
LENGTH_OPTIONS: Final = (
    "attempt_policy",
    "reflection_gate",
    "max_steps",
    "steps_per_attempt",
    "max_attempts",
)

# Similar matching: the fewest routines a place must hold, the least
# best ÷ self score and the most second ÷ best score.
SIMILAR_MIN_ROUTINES: Final = 5
SIMILAR_SELF_RATIO: Final = 0.6
SIMILAR_SECOND_RATIO: Final = 0.5

_MONTH = (
    r"(?:january|february|march|april|may|june|july|august|september|october"
    r"|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
)
_DAY_SUFFIX = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+\d{4}(?!\d))?"

# Applied in this order, each on the previous one's output. A digit never
# touches a masked number, so "2026-10-261" stays as it is.
_DATE_PATTERNS: Final = tuple(
    re.compile(pattern)
    for pattern in (
        # ISO dates: 2026-10-26, 2026/10/26.
        r"(?<!\d)\d{4}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])(?!\d)",
        # Month-name dates: october 26, oct. 26th, 2026; 26 october 2026;
        # october 2026.
        rf"\b{_MONTH}\.?\s+\d{{1,2}}{_DAY_SUFFIX}(?!\d){_YEAR}",
        rf"(?<!\d)\d{{1,2}}{_DAY_SUFFIX}\s+(?:of\s+)?{_MONTH}\b\.?{_YEAR}",
        rf"\b{_MONTH}\.?\s+\d{{4}}(?!\d)",
        # Day.month.year: 26.10.2026.
        r"(?<![\d.])(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}(?![\d.])",
        # Month/day[/year]: 10/26, 10/26/2026.
        r"(?<![\d/])(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?:/\d{2}(?:\d{2})?)?"
        r"(?![\d/])",
        # Weekday names.
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
        # CJK: 2026年10月26日, 2026年, 10月26日, 星期一, 周一, 礼拜一.
        r"\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?)?",
        r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*[日号]",
        r"(?:星期|礼拜|周)[一二三四五六日天]",
    )
)
# HH:MM[:SS], with an optional am or pm.
_TIME_PATTERN: Final = re.compile(
    r"(?<![\d:])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?![\d:])(?:\s*[ap]\.?m\.?\b)?"
)
_SPACE: Final = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """*query* as the key of a recurring task.

    NFKC, lowercase, dates masked as ``<date>`` and times as ``<time>``,
    whitespace collapsed to one space, trimmed. The dates masked are ISO
    dates, ``10/26``, ``10/26/2026``, ``26.10.2026``, month-name dates,
    weekday names, and the CJK forms ``2026年``, ``10月26日``, ``星期一``
    and ``周一``; the times are ``HH:MM[:SS]``. Other numbers stay, so
    "fix issue 123" and "fix issue 456" are different tasks.
    """
    text = unicodedata.normalize("NFKC", query).lower()
    for pattern in _DATE_PATTERNS:
        text = pattern.sub(DATE, text)
    text = _TIME_PATTERN.sub(TIME, text)
    return _SPACE.sub(" ", text).strip()


def routine_key(text: str, place: str) -> str:
    """The key of a routine: ``sha256(text + "\\0" + place)`` in hex."""
    return hashlib.sha256(f"{text}\0{place}".encode()).hexdigest()


@dataclass(frozen=True)
class RoutineKey:
    """What identifies a routine, with the text a new routine stores.

    ``name`` is the routine's name for a named routine, else ``None``;
    ``query_text`` is the normalized query, redacted and cut to
    :data:`MAX_QUERY_TEXT` characters (empty when there is no query).
    """

    key: str
    place: str
    name: str | None
    query_text: str


def key_for_query(query: str, place: str) -> RoutineKey:
    """The routine of *query* run in *place* (see :func:`run_place`)."""
    normalized = normalize_query(query)
    return RoutineKey(
        key=routine_key(normalized, place),
        place=place,
        name=None,
        query_text=stored_query_text(normalized),
    )


def key_for_routine(name: str, query: str = "") -> RoutineKey:
    """The named routine *name*: one routine in any folder.

    The key's text is ``routine:<name>`` with *name* stripped, and its
    place is empty. *query*, normalized, is the text a new routine stores.
    Raises ``ValueError`` for a blank name.
    """
    stripped = name.strip()
    if not stripped:
        raise ValueError("a routine name must not be blank")
    return RoutineKey(
        key=routine_key(f"routine:{stripped}", ""),
        place="",
        name=stripped,
        query_text=stored_query_text(normalize_query(query)),
    )


def stored_query_text(normalized: str) -> str:
    """*normalized* as a routine keeps it: redacted, then cut."""
    redacted, _ = redact_secrets(normalized)
    return redacted[:MAX_QUERY_TEXT]


def run_place(root: str | None) -> str:
    """The directory a run's tools work in, as a routine's place.

    The executor's workspace root when there is one, else the process's
    working directory, read now; either as ``os.path.realpath`` gives it.
    """
    return os.path.realpath(root if root else os.getcwd())


def known_model(model: str) -> bool:
    """Whether *model* names one model (not ``""`` or ``"default"``)."""
    return model.strip() not in UNKNOWN_MODELS


def length_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """The :data:`LENGTH_OPTIONS` of a run's *config*; a missing one is ``None``."""
    return {name: config.get(name) for name in LENGTH_OPTIONS}


def canonical_json(value: Any) -> str:
    """*value* as JSON with sorted keys and no spaces, for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def template_id(routine_id: str, model: str, options: Mapping[str, Any]) -> str:
    """``<routine_id>:<12 hex digits>``, the digits hashing *model* and *options*."""
    digest = hashlib.sha256(
        f"{model}\0{canonical_json(dict(options))}".encode()
    ).hexdigest()
    return f"{routine_id}:{digest[:12]}"


@dataclass(frozen=True)
class SimilarMatch:
    """A stored routine close to a new query: a hint, never a percent.

    ``ratio`` is the best score ÷ the query's self-score, ``second_ratio``
    the second score ÷ the best (0 without a second).
    """

    routine_id: str
    ratio: float
    second_ratio: float


def find_similar(
    query_text: str, candidates: Sequence[tuple[str, str]]
) -> SimilarMatch | None:
    """The routine in *candidates* that *query_text* is similar to, or ``None``.

    *candidates* are ``(routine_id, query_text)`` pairs: the routines of
    one place. BM25 ranks their texts for *query_text*. The self-score is
    the query's own score with the query added to the index under a
    temporary id, which counts in the number of documents and their
    average length; the routines are scored in that same search. A match
    needs at least :data:`SIMILAR_MIN_ROUTINES` candidates, best ÷ self ≥
    :data:`SIMILAR_SELF_RATIO` and second ÷ best ≤
    :data:`SIMILAR_SECOND_RATIO`.
    """
    if len(candidates) < SIMILAR_MIN_ROUTINES or not query_text.strip():
        return None
    index = BM25Index()
    ids: set[str] = set()
    for routine_id, text in candidates:
        index.add(routine_id, text)
        ids.add(routine_id)
    probe = "\0query"
    while probe in ids:
        probe += "\0"
    index.add(probe, query_text)
    hits = index.search(query_text, top_k=len(ids) + 1)
    self_score = next((score for doc_id, score in hits if doc_id == probe), 0.0)
    others = [score for doc_id, score in hits if doc_id != probe]
    best_id = next((doc_id for doc_id, _ in hits if doc_id != probe), None)
    if self_score <= 0 or best_id is None:
        return None
    best = others[0]
    second = others[1] if len(others) > 1 else 0.0
    ratio = best / self_score
    second_ratio = second / best
    if ratio < SIMILAR_SELF_RATIO or second_ratio > SIMILAR_SECOND_RATIO:
        return None
    return SimilarMatch(routine_id=best_id, ratio=ratio, second_ratio=second_ratio)
