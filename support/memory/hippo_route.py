"""Hippo's routing: each new memory item is compared with the items kept (#428).

A new item is routed by :func:`localmelo.melo.memory.hippo.policy.decide` against
the active items of its kind, by rules and no model call, to one of five actions:

- ``NEW``: the item is stored.
- ``UPDATE``: it is stored and the item it updates is marked replaced by it (that
  one stays, and reads as older).
- ``RELATED``: it is stored and the two are linked both ways.
- ``CONFLICT``: it is stored and both are marked in conflict; both stay recallable.
- ``IGNORE``: it is not stored; the route names the item it duplicates.

:class:`Router` keeps the active items in memory and updates them as it stores, so a
run does not read the whole table again for each item. The policy sorts the items it
is given by the same-subject flag, the content overlap and the cosine, and acts on
the first; so :class:`Router` hands it only the few that sort first by those same
numbers, which gives the decision it would give for all of them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from localmelo.melo.memory.hippo.policy import (
    PolicyDecision,
    content_tokens,
    decide,
    overlap,
    subjects,
)
from localmelo.support.memory.index import ConversationIndex, MemoryItem

SOURCE: Final = "ingest"
# How many of the best-matching active items are handed to the policy.
POOL: Final = 8


@dataclass(frozen=True)
class Route:
    """What became of a new item: the action, the id it was stored under (``None``
    for ``IGNORE``), the item it was compared with, and why."""

    action: str
    item: int | None
    target: int | None
    reason: str

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Entry:
    """An active item in the shape the policy reads: ``text``, ``vector``, ``id``,
    ``created_at`` and a ``provenance`` that gives it an item-local subject."""

    id: int
    text: str
    created_at: float
    vector: tuple[float, ...] | None
    provenance: Mapping[str, Any]
    subject: tuple[str, ...]
    tokens: frozenset[str]
    unit: np.ndarray | None


def _unit(vector: Sequence[float] | None) -> np.ndarray | None:
    if not vector:
        return None
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else None


def _entry(item: MemoryItem, model: str) -> Entry:
    # An item's episodic subject is its own (the policy then never replaces one
    # episode with another); profile and semantic subjects come from its text.
    provenance = {"candidate_id": f"m{item.id}", "item_index": 0}
    vector = item.vector if item.embed_model == model else None
    return Entry(
        id=item.id,
        text=item.text,
        created_at=item.created_at,
        vector=vector,
        provenance=provenance,
        subject=subjects(item.text, item.kind, item_key=f"m{item.id}:0"),
        tokens=frozenset(content_tokens(item.text)),
        unit=_unit(vector),
    )


class Router:
    """Routes new items of one run against the active items of ``index``."""

    def __init__(self, index: ConversationIndex) -> None:
        self._index = index
        self._active: dict[str, dict[int, Entry]] | None = None
        self._model = ""
        self._count = 0

    async def _load(self, model: str) -> dict[str, dict[int, Entry]]:
        if self._active is None or model != self._model:
            self._model = model
            self._active = {}
            for item in await self._index.active_items():
                self._active.setdefault(item.kind, {})[item.id] = _entry(item, model)
        return self._active

    def _pool(self, item: MemoryItem, provenance: Mapping[str, Any]) -> list[Entry]:
        """The :data:`POOL` active items of the item's kind that the policy ranks
        first for it (its own order: same subject, overlap, cosine, newest, id)."""
        entries = list((self._active or {}).get(item.kind, {}).values())
        if len(entries) <= POOL:
            return entries
        subject = subjects(
            item.text,
            item.kind,
            item_key=f"{provenance['candidate_id']}:{provenance['item_index']}",
        )
        tokens = frozenset(content_tokens(item.text))
        mine = _unit(item.vector if item.embed_model == self._model else None)

        def rank(entry: Entry) -> tuple[bool, float, float, float, int]:
            cosine = (
                float(np.dot(mine, entry.unit))
                if mine is not None
                and entry.unit is not None
                and mine.shape == entry.unit.shape
                else 0.0
            )
            return (
                entry.subject == subject,
                overlap(tokens, entry.tokens),
                cosine,
                entry.created_at,
                entry.id,
            )

        return sorted(entries, key=rank, reverse=True)[:POOL]

    async def route(self, item: MemoryItem) -> Route:
        """Compare *item* with the active items of its kind and act on the
        policy's decision (see the module's list). Stores the item unless it is
        ignored."""
        active = await self._load(item.embed_model)
        self._count += 1
        provenance = {"candidate_id": "new", "item_index": self._count}
        decision: PolicyDecision = decide(
            item.text,
            kind=item.kind,
            source=SOURCE,
            provenance=provenance,
            active=self._pool(item, provenance),
            embedding=item.vector,
        )
        action, target = decision.action, decision.target_id
        if action == "IGNORE" or (action != "NEW" and target is None):
            return Route("IGNORE", None, target, decision.reason)
        new_id = await self._index.add_item(item)
        stored = dataclasses.replace(item, id=new_id)
        kind = active.setdefault(item.kind, {})
        if action == "UPDATE" and target is not None:
            await self._index.mark_replaced(target, new_id)
            kind.pop(target, None)
        elif action == "RELATED" and target is not None:
            await self._index.link_items(new_id, target, conflict=False)
        elif action == "CONFLICT" and target is not None:
            await self._index.link_items(new_id, target, conflict=True)
            kind.pop(target, None)  # a conflict is no longer an active item
            return Route(action, new_id, target, decision.reason)
        kind[new_id] = _entry(stored, self._model)
        return Route(action, new_id, target, decision.reason)
