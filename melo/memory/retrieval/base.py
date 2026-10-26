"""The retrieval interface: documents, hits, results and the retriever protocol.

A retriever ranks documents (:class:`Doc`) for a query and answers with a
:class:`RetrievalResult` of at most *k* hits (:class:`Hit`). Every
retriever follows the same rules in ``search``, which
:func:`search_candidates` checks:

- ``k < 1``, or two docs with one ``doc_id``, raises :class:`ValueError`;
- a blank query, or no doc left after ``exclude``, gives no hits and sends
  no request;
- hits are at most ``k``, ranked from 1, and never name a doc outside
  ``docs`` or inside ``exclude``; ids in ``exclude`` that are not in
  ``docs`` are ignored;
- docs never change, so a retriever may cache by ``doc_id``, as long as it
  checks that the cached text is still the doc's text and indexes the doc
  again when it is not.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeGuard, runtime_checkable

from localmelo.melo.schema import Message

DOC_KINDS = ("note", "draft", "thought", "digest", "reflection", "long")
FALLBACK_REASONS = ("timeout", "parse", "provider_error")
RETRIEVER_NAMES = ("bm25", "llm", "embedding")


@dataclass(frozen=True)
class Doc:
    """One document a retriever can return."""

    doc_id: str  # non-empty, no whitespace
    kind: str  # one of DOC_KINDS
    text: str
    title: str = ""  # one line; the llm catalog falls back to the text's first line
    # task_id, session_id, step, attempt, stage, source, created_at, ...
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.doc_id, str)
            or not self.doc_id
            or any(char.isspace() for char in self.doc_id)
        ):
            raise ValueError(
                "doc_id must be a non-empty string without whitespace, "
                f"got {self.doc_id!r}"
            )
        if self.kind not in DOC_KINDS:
            raise ValueError(
                f"unknown doc kind {self.kind!r}; the kinds are "
                f"{', '.join(DOC_KINDS)}"
            )


@dataclass(frozen=True)
class Hit:
    """One document a search returned."""

    doc_id: str
    score: float
    rank: int  # 1-based


@dataclass
class RetrievalResult:
    """What one search returned, and how it got there."""

    hits: list[Hit]
    retriever: str  # the one that answered: "bm25" after a fallback
    ms: int  # wall time of the whole search, fallback included, rounded
    fallback: str = ""  # "" or one of FALLBACK_REASONS
    # The response's integer usage keys; {} when no model answered.
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Ranks documents for a query (see the module docstring for the rules)."""

    name: str

    async def search(
        self,
        query: str,
        docs: Sequence[Doc],
        *,
        k: int,
        exclude: AbstractSet[str] = frozenset(),
    ) -> RetrievalResult: ...


# One chat request without tools: the messages in, the assistant's reply out.
Send = Callable[[list[Message]], Awaitable[Message]]


def search_candidates(
    query: str, docs: Sequence[Doc], *, k: int, exclude: AbstractSet[str]
) -> list[Doc]:
    """Check the arguments of a search and return the docs it ranks.

    Raises :class:`ValueError` when *k* is under 1 or two of *docs* share
    a ``doc_id``. Returns the docs not in *exclude*, in order; ``[]`` for
    a blank *query*, so that the search sends no request.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k!r}")
    seen: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            raise ValueError(f"two docs have the doc_id {doc.doc_id!r}")
        seen.add(doc.doc_id)
    if not query.strip():
        return []
    return [doc for doc in docs if doc.doc_id not in exclude]


def ranked(scored: Iterable[tuple[str, float]]) -> list[Hit]:
    """Hits for ``(doc_id, score)`` pairs that are already in rank order."""
    return [
        Hit(doc_id=doc_id, score=score, rank=rank)
        for rank, (doc_id, score) in enumerate(scored, start=1)
    ]


def elapsed_ms(start: float) -> int:
    """Whole milliseconds since *start*, a :func:`time.perf_counter` value."""
    return round((time.perf_counter() - start) * 1000)


def int_step(meta: Mapping[str, Any]) -> int | None:
    """``meta["step"]`` when it is an int (a bool is not one), else ``None``."""
    step = meta.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    return None


def is_number(value: object) -> TypeGuard[int | float]:
    """Whether *value* is an int or a float (a bool is neither)."""
    return isinstance(value, int | float) and not isinstance(value, bool)
