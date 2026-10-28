"""R6-lite's admission policy: classify a fact, compare it, pick one action.

The policy is pure: it takes one item's text, its source and the active
entries of its kind, and returns one decision. It makes no model call and
touches no store. The five actions are NEW, UPDATE, RELATED, CONFLICT and
IGNORE; the thresholds are v1 and are set on synthetic fixtures.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from localmelo.melo.memory.bm25 import tokenize
from localmelo.melo.memory.notes import (
    has_negative_assertion,
    is_meta_claim,
    path_spans,
)

ACTIONS = ("NEW", "UPDATE", "RELATED", "CONFLICT", "IGNORE")

# A short fact is noise: fewer than this many normalised characters.
MIN_FACT_CHARS = 20

# The source a candidate comes from names where it may be written.
PROFILE_PREDICATES: tuple[str, ...] = (
    "live",
    "lives",
    "住在",
    "居住",
    "work",
    "works",
    "工作",
    "上班",
    "prefer",
    "like",
    "喜欢",
    "偏好",
    "name",
    "call me",
    "叫我",
    "名字",
    "timezone",
    "时区",
    "language",
    "语言",
    "allergic",
    "过敏",
)

_PROFILE_CANONICAL: tuple[tuple[str, str], ...] = (
    ("call me", "name"),
    ("住在", "live"),
    ("居住", "live"),
    ("live", "live"),
    ("lives", "live"),
    ("工作", "work"),
    ("上班", "work"),
    ("work", "work"),
    ("works", "work"),
    ("prefer", "prefer"),
    ("like", "prefer"),
    ("喜欢", "prefer"),
    ("偏好", "prefer"),
    ("name", "name"),
    ("叫我", "name"),
    ("名字", "name"),
    ("timezone", "timezone"),
    ("时区", "timezone"),
    ("language", "language"),
    ("语言", "language"),
    ("allergic", "allergy"),
    ("过敏", "allergy"),
)

SEMANTIC_ATTRIBUTES: tuple[str, ...] = (
    "address",
    "lives at",
    "moved to",
    "room",
    "date",
    "time",
    "appointment",
    "phone",
    "email",
    "birthday",
    "price",
    "amount",
    "balance",
    "deadline",
    "地址",
    "住在",
    "搬到",
    "房间",
    "日期",
    "时间",
    "预约",
    "电话",
    "邮箱",
    "生日",
    "价格",
    "金额",
    "余额",
    "截止",
)

CHANGE_MARKERS: tuple[str, ...] = (
    "moved",
    "now",
    "no longer",
    "switched",
    "changed",
    "updated",
    "corrected",
    "instead",
    "搬到",
    "现在",
    "不再",
    "换成",
    "改了",
    "改为",
    "更新",
    "更正",
)

FILE_ONLY_WORDS: frozenset[str] = frozenset(
    {
        "read",
        "wrote",
        "write",
        "file",
        "files",
        "line",
        "lines",
        "path",
        "test",
        "tests",
        "passed",
        "failed",
        "读",
        "写",
        "文件",
        "行",
        "路径",
        "测试",
        "通过",
        "失败",
        "ran",
        "run",
        "import",
        "imports",
        "fixed",
        "fix",
        "code",
    }
)

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "my",
        "me",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your",
        "的",
        "了",
        "是",
        "在",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "和",
        "与",
        "就",
        "也",
        "都",
    }
)

_PUNCTUATION = ".,!?;:，。！？；：、…"


@dataclass(frozen=True)
class PolicyDecision:
    """One action for one item, and the target it compared against."""

    action: str
    kind: str
    subject: tuple[str, ...]
    target_id: int | None = None
    reason: str = ""
    overlap: float = 0.0
    cosine: float = 0.0


@dataclass(frozen=True)
class _Compared:
    entry: Any
    same_subject: bool
    overlap: float
    cosine: float
    change: bool
    contradicts: bool


def normalise(text: str) -> str:
    """NFKC, lower case, collapsed whitespace, no terminal punctuation."""
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("，", ",").replace("。", ".")
    folded = " ".join(folded.lower().split())
    return folded.strip().rstrip(_PUNCTUATION).strip()


def content_tokens(text: str) -> set[str]:
    """The text's BM25 terms without the stopwords."""
    return {token for token in tokenize(normalise(text)) if token not in STOPWORDS}


def _has_phrase(text: str, phrase: str) -> bool:
    """Whether *phrase* is in the normalised *text*, whole for ASCII."""
    if phrase.isascii():
        return (
            re.search(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])", text)
            is not None
        )
    return phrase in text


def _first_marker(text: str, markers: Sequence[str]) -> str:
    found = [marker for marker in markers if _has_phrase(text, marker)]
    if not found:
        return ""
    # The earliest marker in the text wins; a longer one at the same place
    # wins over its prefix.
    return min(found, key=lambda marker: (text.find(marker), -len(marker)))


def classify_kind(
    text: str,
    *,
    source: str,
    task_text: str = "",
    first_person: bool | None = None,
) -> str:
    """Assign one of the four kinds from the text and its source."""
    folded = normalise(text)
    if source == "reflection":
        return "reflection"
    if source == "digest":
        return "episodic"
    if source == "user":
        if first_person is None:
            # "call me" is a profile predicate even though "me" is not in
            # _first_person (T53R finding 7).
            first_person = (
                _first_person(folded)
                or _has_phrase(folded, "call me")
                or "叫我" in folded
            )
        return "profile" if first_person else "semantic"
    if source == "reasoning_note":
        if _first_person(folded):
            return "episodic"
        if _first_marker(folded, SEMANTIC_ATTRIBUTES):
            return "semantic"
        return "episodic"
    return "episodic"


def _first_person(text: str) -> bool:
    return re.search(
        r"(?<![a-z0-9_])(i|i'm|my|we|our)(?![a-z0-9_])", text
    ) is not None or any(marker in text for marker in ("我", "我的", "我们"))


def _profile_subject(text: str) -> tuple[str, ...]:
    for phrase, canonical in _PROFILE_CANONICAL:
        if _has_phrase(text, phrase):
            return (canonical, "user")
    return (normalise(text), "user")


def _semantic_subject(text: str) -> tuple[str, ...]:
    folded = normalise(text)
    marker = _first_marker(folded, SEMANTIC_ATTRIBUTES)
    if not marker:
        return (folded,)
    start = folded.find(marker)
    before = folded[:start].strip(" ,:;")
    capitalised = re.search(r"[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+", text)
    entity = ""
    if capitalised is not None and capitalised.start() < max(start, 0):
        entity = capitalised.group(0)
    if not entity:
        cjk = re.search(r"([\u4e00-\u9fff]{2,6})\s*$", before)
        if cjk is not None:
            entity = cjk.group(1)
    if not entity:
        entity = before.split()[-1] if before.split() else ""
    return (normalise(entity), marker)


def subjects(
    text: str,
    kind: str,
    *,
    task_text: str = "",
    item_key: str = "",
) -> tuple[str, ...]:
    """The comparison subject of one text of *kind*.

    An episodic item's subject is item-local: its candidate id and index
    (T53R finding 4). Two different items of one note therefore both stay,
    while a restatement re-proposes the same candidate with the same item
    order, so its item keeps the subject and is an ``UPDATE``.
    """
    if kind == "profile":
        return _profile_subject(normalise(text))
    if kind == "semantic":
        return _semantic_subject(text)
    if item_key:
        return ("item", item_key)
    return ("task", normalise(task_text))


def overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """``|A ∩ B| / max(1, min(|A|, |B|))`` over the two content sets."""
    first, second = set(left), set(right)
    if not first or not second:
        return 0.0
    return len(first & second) / max(1, min(len(first), len(second)))


def value_tokens(text: str, *, subject: Sequence[str] = ()) -> set[str]:
    """The content words that are neither subject nor change markers."""
    folded = normalise(text)
    markers = {token for marker in CHANGE_MARKERS for token in tokenize(marker)}
    subject_tokens = {token for part in subject for token in tokenize(part)}
    return content_tokens(folded) - markers - subject_tokens


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _is_distinctive(token: str, text: str) -> bool:
    """A number, a capitalised word in *text*, or a CJK run of two."""
    if token.isdigit():
        return True
    if len(token) >= 2 and all("\u4e00" <= char <= "\u9fff" for char in token):
        return True
    return any(
        match.group(0)[:1].isupper()
        for match in re.finditer(re.escape(token), text, re.IGNORECASE)
    )


def _values_contradict(
    left: set[str], right: set[str], left_text: str, right_text: str
) -> bool:
    if not left or not right or left & right:
        return False
    return any(_is_distinctive(token, left_text) for token in left) or any(
        _is_distinctive(token, right_text) for token in right
    )


def _file_only(text: str) -> bool:
    spans = path_spans(text)
    if not spans:
        return False
    stripped = list(text)
    for start, end in spans:
        for index in range(start, end):
            stripped[index] = " "
    remaining = content_tokens("".join(stripped))
    return all(token in FILE_ONLY_WORDS or token.isdigit() for token in remaining)


def _entry_text(entry: Any) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("text") or "")
    return str(getattr(entry, "text", ""))


def _item_key(provenance: Mapping[str, Any]) -> str:
    """``<candidate id>:<item index>`` for an item-local episodic subject."""
    candidate_id = provenance.get("candidate_id")
    item_index = provenance.get("item_index")
    if isinstance(candidate_id, str) and candidate_id and isinstance(item_index, int):
        return f"{candidate_id}:{item_index}"
    return ""


def _entry_subject(entry: Any, kind: str) -> tuple[str, ...]:
    provenance = (
        entry.get("provenance")
        if isinstance(entry, Mapping)
        else getattr(entry, "provenance", {})
    )
    task_text = ""
    item_key = ""
    if isinstance(provenance, Mapping):
        task_text = str(provenance.get("task_text") or "")
        item_key = _item_key(provenance)
    return subjects(_entry_text(entry), kind, task_text=task_text, item_key=item_key)


def _entry_vector(entry: Any) -> Sequence[float] | None:
    value = (
        entry.get("vector")
        if isinstance(entry, Mapping)
        else getattr(entry, "vector", None)
    )
    if value is None:
        return None
    return tuple(float(item) for item in value)


def _entry_id(entry: Any) -> int | None:
    value = (
        entry.get("id") if isinstance(entry, Mapping) else getattr(entry, "id", None)
    )
    return int(value) if value is not None else None


def _entry_created(entry: Any) -> float:
    value = (
        entry.get("created_at")
        if isinstance(entry, Mapping)
        else getattr(entry, "created_at", 0.0)
    )
    return float(value or 0.0)


def _meta_filter_applies(
    text: str,
    *,
    source: str,
    admitted_by: str,
    provenance: Mapping[str, Any],
) -> bool:
    """Whether the meta filter may drop *text* at this boundary.

    A digest admitted by ``outcome`` for a completed run is never filtered:
    the end of a completed task is exactly the episode that signal exists
    to keep. A digest with a ``task:`` line is content anyway (see
    :func:`~localmelo.melo.memory.notes.is_meta_claim`).
    """
    if (
        admitted_by == "outcome"
        and source == "digest"
        and str(provenance.get("final_status") or "") == "completed"
    ):
        return False
    return is_meta_claim(text)


def decide(
    text: str,
    *,
    kind: str,
    source: str,
    provenance: Mapping[str, Any],
    active: Sequence[Any],
    embedding: Sequence[float] | None = None,
    admitted_by: str = "",
) -> PolicyDecision:
    """Compare one item with the active entries of *kind* and pick an action."""
    folded = normalise(text)
    task_text = str(provenance.get("task_text") or "")
    item_key = _item_key(provenance)
    subject = subjects(text, kind, task_text=task_text, item_key=item_key)
    content = content_tokens(text)
    if _meta_filter_applies(
        text, source=source, admitted_by=admitted_by, provenance=provenance
    ):
        return PolicyDecision(
            action="IGNORE",
            kind=kind,
            subject=subject,
            reason="meta claim about the agent's own work",
        )
    if len(folded) < MIN_FACT_CHARS:
        return PolicyDecision(
            action="IGNORE",
            kind=kind,
            subject=subject,
            reason="short fact",
        )
    if _file_only(text):
        return PolicyDecision(
            action="IGNORE",
            kind=kind,
            subject=subject,
            reason="task-file-only fact",
        )
    values = value_tokens(text, subject=subject)

    compared: list[_Compared] = []
    for entry in active:
        other_text = _entry_text(entry)
        other_subject = _entry_subject(entry, kind)
        same = other_subject == subject
        ratio = overlap(content, content_tokens(other_text))
        cosine = _cosine(embedding, _entry_vector(entry))
        other_folded = normalise(other_text)
        change = any(
            _has_phrase(folded, marker) or _has_phrase(other_folded, marker)
            for marker in CHANGE_MARKERS
        )
        other_values = value_tokens(other_text, subject=other_subject)
        compared.append(
            _Compared(
                entry=entry,
                same_subject=same,
                overlap=ratio,
                cosine=cosine,
                change=change,
                contradicts=same
                and _values_contradict(values, other_values, text, other_text)
                and not change,
            )
        )

    target: _Compared | None = None
    if compared:
        compared.sort(
            key=lambda item: (
                item.same_subject,
                item.overlap,
                item.cosine,
                _entry_created(item.entry),
                _entry_id(item.entry) or 0,
            ),
            reverse=True,
        )
        target = compared[0]
    target_id = None if target is None else _entry_id(target.entry)

    if target is not None:
        other_text = _entry_text(target.entry)
        opposite = has_negative_assertion(text) != has_negative_assertion(other_text)
        if opposite and (
            target.same_subject or target.overlap >= 0.35 or target.cosine >= 0.75
        ):
            return PolicyDecision(
                action="CONFLICT",
                kind=kind,
                subject=subject,
                target_id=target_id,
                reason=f"opposite polarity to entry {target_id}",
                overlap=target.overlap,
                cosine=target.cosine,
            )
        if not opposite and (
            normalise(other_text) == folded
            or target.overlap >= 0.9
            or target.cosine >= 0.95
        ):
            return PolicyDecision(
                action="IGNORE",
                kind=kind,
                subject=subject,
                target_id=target_id,
                reason=f"overlap {target.overlap:.2f}; duplicate",
                overlap=target.overlap,
                cosine=target.cosine,
            )
        if target.same_subject:
            if target.contradicts:
                return PolicyDecision(
                    action="CONFLICT",
                    kind=kind,
                    subject=subject,
                    target_id=target_id,
                    reason=f"same subject; contradicts entry {target_id}",
                    overlap=target.overlap,
                    cosine=target.cosine,
                )
            return PolicyDecision(
                action="UPDATE",
                kind=kind,
                subject=subject,
                target_id=target_id,
                reason=f"same subject; update of entry {target_id}",
                overlap=target.overlap,
                cosine=target.cosine,
            )
        if target.overlap >= 0.35 or target.cosine >= 0.75:
            return PolicyDecision(
                action="RELATED",
                kind=kind,
                subject=subject,
                target_id=target_id,
                reason=f"overlap {target.overlap:.2f}; related to entry {target_id}",
                overlap=target.overlap,
                cosine=target.cosine,
            )
    return PolicyDecision(
        action="NEW",
        kind=kind,
        subject=subject,
        target_id=None,
        reason="new",
    )


class HippoPolicy:
    """The five actions, as a small object the store's callers can hold."""

    def decide(
        self,
        text: str,
        *,
        kind: str,
        source: str,
        provenance: Mapping[str, Any],
        active: Sequence[Any],
        embedding: Sequence[float] | None = None,
        admitted_by: str = "",
    ) -> PolicyDecision:
        return decide(
            text,
            kind=kind,
            source=source,
            provenance=provenance,
            active=active,
            embedding=embedding,
            admitted_by=admitted_by,
        )


__all__ = [
    "ACTIONS",
    "CHANGE_MARKERS",
    "FILE_ONLY_WORDS",
    "HippoPolicy",
    "MIN_FACT_CHARS",
    "PROFILE_PREDICATES",
    "PolicyDecision",
    "SEMANTIC_ATTRIBUTES",
    "classify_kind",
    "content_tokens",
    "decide",
    "normalise",
    "overlap",
    "subjects",
    "value_tokens",
]
