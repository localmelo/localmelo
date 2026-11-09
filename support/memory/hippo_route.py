"""Hippo's routing: each new memory item is compared with the items kept (#428).

A new item is routed against the active items of every kind to one of five actions:

- ``NEW``: the item is stored.
- ``UPDATE``: it is stored and the item it updates is marked replaced by it (that
  one stays, and reads as older). An item older than the one it would update is
  stored as already replaced by it instead: the newer fact stays current, whatever
  order the chats are encoded in.
- ``RELATED``: it is stored and the two are linked both ways.
- ``CONFLICT``: it is stored and both are marked in conflict; both stay recallable.
- ``IGNORE``: it is not stored; the route names the item it duplicates, and that item
  gains its sources and, when the repeat is newer, its time
  (:meth:`~.index.ConversationIndex.add_sources`).
- ``REFUSE``: its title or text holds a secret (:func:`~.index.holds_secret`); it is
  not stored and nothing is compared. :meth:`Router.route` is the only caller of
  :meth:`~.index.ConversationIndex.add_item`, the only writer of the items table, so
  no item that holds a secret reaches it by any path.

Code settles only what is certain, by rules and no model call
(:func:`localmelo.melo.memory.hippo.policy.decide` with ``certain_only``): a meta
claim, a short fact, a fact about files only, an opposite polarity of the same
text (two texts that differ by more than their negation may be about different
things), an identical text. Whether two notes say the same thing, or the newer one
says it changed, is a question of meaning, so :meth:`Router.shortlist` finds the
active items that may say it by content, across kinds, and a judge
(:mod:`.hippo_judge`) reads them. Its clear answer is the :class:`Judged` that
:meth:`Router.route` acts on: ``same`` is an ``IGNORE`` and ``changes`` an
``UPDATE``. With none, a same-subject item is only ``RELATED``: no item replaces
another on a word they share.

:class:`Router` keeps the active items in one map, whatever their kind, and updates
it as it stores, so a run does not read the whole table again for each item. The
policy sorts the items it is given by the same-subject flag, the content overlap and
the cosine, and acts on the first; so :class:`Router` hands it only the few that sort
first by those same numbers, which gives the decision it would give for all of them.
An item whose text is identical is looked up in the whole map first, whatever its kind
or subject, and is the only one handed over: a repeat is never missed for an item that
only shares a subject with it. What a write rests on is read again from the table,
under the same hold of the lock the write is made in (:meth:`Router.confirm` and
:meth:`Router.route`), because an item may be forgotten or deleted while a judge is
asked; when one of them is out of step, the whole map is read again, since a forget
can also make an item active again.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from localmelo.melo.memory.hippo.policy import (
    LINK_COSINE,
    LINK_OVERLAP,
    PolicyDecision,
    content_tokens,
    decide,
    normalise,
    overlap,
    subjects,
)
from localmelo.support.memory.index import ConversationIndex, MemoryItem, holds_secret

# The action of an item that holds a secret (:class:`Route`).
REFUSE: Final = "REFUSE"
# How many of the best-matching active items are handed to the policy.
POOL: Final = 8
# How many active items one new item is judged against.
SHORTLIST: Final = 3


@dataclass(frozen=True)
class Route:
    """What became of a new item: the action, the id it was stored under (``None``
    for ``IGNORE`` and ``REFUSE``), the item it was compared with, and why."""

    action: str
    item: int | None
    target: int | None
    reason: str

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Entry:
    """An active item in the shape the policy reads: ``text``, ``vector``, ``id``,
    ``created_at``, its ``kind`` and a ``provenance`` that gives it an item-local
    subject. ``sources`` are the turns it cites; ``folded`` is its normalised text."""

    id: int
    text: str
    kind: str
    created_at: float
    sources: tuple[tuple[str, int], ...]
    vector: tuple[float, ...] | None
    provenance: Mapping[str, Any]
    subject: tuple[str, ...]
    tokens: frozenset[str]
    unit: np.ndarray | None
    folded: str


@dataclass(frozen=True)
class Judged:
    """A judge's clear answer for one new item (:mod:`.hippo_judge`): ``same`` or
    ``changes`` for the active item *target*, read among the *shortlist* as each
    item stood when the judge was asked."""

    verdict: str
    target: int
    shortlist: tuple[Entry, ...]


def _unit(vector: Sequence[float] | None) -> np.ndarray | None:
    if not vector:
        return None
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else None


def _cosine(mine: np.ndarray | None, entry: Entry) -> float:
    if mine is None or entry.unit is None or mine.shape != entry.unit.shape:
        return 0.0
    return float(np.dot(mine, entry.unit))


def _entry(item: MemoryItem, model: str) -> Entry:
    # An item's episodic subject is its own (the policy then never replaces one
    # episode with another); profile and semantic subjects come from its text.
    provenance = {"candidate_id": f"m{item.id}", "item_index": 0}
    vector = item.vector if item.embed_model == model else None
    return Entry(
        id=item.id,
        text=item.text,
        kind=item.kind,
        created_at=item.created_at,
        sources=item.sources,
        vector=vector,
        provenance=provenance,
        subject=subjects(item.text, item.kind, item_key=f"m{item.id}:0"),
        tokens=frozenset(content_tokens(item.text)),
        unit=_unit(vector),
        folded=normalise(item.text),
    )


class Router:
    """Routes new items of one run against the active items of ``index``."""

    def __init__(self, index: ConversationIndex) -> None:
        self._index = index
        self._active: dict[int, Entry] | None = None
        self._model = ""
        self._count = 0

    async def load(self, model: str) -> dict[int, Entry]:
        """Read the active items of every kind, with their vectors of embedding
        model *model*. Read once: again only for another model, an item with no
        vector (*model* ``""``) compares by words and keeps what is loaded."""
        if self._active is None or (model and model != self._model):
            self._model = model
            self._active = await self._read()
        return self._active

    async def _read(self) -> dict[int, Entry]:
        return {
            item.id: _entry(item, self._model)
            for item in await self._index.active_items()
        }

    def _provenance(self) -> dict[str, Any]:
        """What gives the item being routed a subject of its own: never the
        subject of a stored item."""
        return {"candidate_id": "new", "item_index": self._count + 1}

    def _pool(self, item: MemoryItem, provenance: Mapping[str, Any]) -> list[Entry]:
        """The active items the policy is given for the item: the one with an
        identical text if there is one (of the item's kind first, then the newest),
        whatever its subject, else the :data:`POOL` that the policy ranks first (its
        own order: same subject and kind, overlap, cosine, newest, id)."""
        entries = list((self._active or {}).values())
        folded = normalise(item.text)
        same = [entry for entry in entries if entry.folded == folded]
        if same:
            return [max(same, key=lambda e: (e.kind == item.kind, e.created_at, e.id))]
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
            return (
                entry.kind == item.kind and entry.subject == subject,
                overlap(tokens, entry.tokens),
                _cosine(mine, entry),
                entry.created_at,
                entry.id,
            )

        return sorted(entries, key=rank, reverse=True)[:POOL]

    def _rules(self, item: MemoryItem, provenance: Mapping[str, Any]) -> PolicyDecision:
        return decide(
            item.text,
            kind=item.kind,
            provenance=provenance,
            active=self._pool(item, provenance),
            embedding=item.vector,
            certain_only=True,
        )

    async def shortlist(self, item: MemoryItem) -> tuple[Entry, ...]:
        """The active items a judge reads for *item*, best match first: at most
        :data:`SHORTLIST` of those close enough to link (the overlap or the cosine
        of the policy's link zone), of any kind, ranked by overlap and then cosine,
        not by subject, so a few items that share a verb with it cannot crowd out
        the one that says the same. None when the rules settle the item without a
        model (a secret, a meta claim, a short fact, a fact about files only, an
        opposite polarity of the same text, an identical text)."""
        if holds_secret(item.title, item.text):
            return ()
        active = await self.load(item.embed_model)
        if self._rules(item, self._provenance()).settled:
            return ()
        tokens = frozenset(content_tokens(item.text))
        mine = _unit(item.vector if item.embed_model == self._model else None)
        near: list[tuple[float, float, float, int, Entry]] = []
        for entry in active.values():
            ratio, cosine = overlap(tokens, entry.tokens), _cosine(mine, entry)
            if ratio >= LINK_OVERLAP or cosine >= LINK_COSINE:
                near.append((ratio, cosine, entry.created_at, entry.id, entry))
        near.sort(key=lambda found: found[:4], reverse=True)
        return tuple(found[4] for found in near[:SHORTLIST])

    async def _refresh(self, entries: Iterable[Entry]) -> bool:
        """Whether each of *entries* is still stored, active and as it was (its id,
        time and sources), read from the table. When one is not, the whole map is read
        again, in place: it did not change only there (a forget that deletes an item
        makes the one it replaced active again)."""
        wanted = {entry.id: entry for entry in entries}
        rows = {row.id: row for row in await self._index.fetch_items(wanted)}
        for item_id, entry in wanted.items():
            row = rows.get(item_id)
            if (
                row is None
                or row.status != "active"
                or row.created_at != entry.created_at
                or set(row.sources) != set(entry.sources)
            ):
                assert self._active is not None
                fresh = await self._read()
                self._active.clear()
                self._active.update(fresh)
                return False
        return True

    async def confirm(self, judged: Judged | None) -> Judged | None:
        """*judged* if every item of its shortlist still exists, active and as it
        was when the judge read it; otherwise ``None``, and the item is routed by
        the rules alone. Call it under the hold of the lock that :meth:`route`
        runs in: nothing may change between the two."""
        if judged is None or await self._refresh(judged.shortlist):
            return judged
        return None

    async def route(self, item: MemoryItem, judged: Judged | None = None) -> Route:
        """Compare *item* with the active items and act on the policy's decision
        (see the module's list), or on *judged* (confirmed, by :meth:`confirm`, in the
        same hold of the lock) unless the rules were certain. Stores the item unless it
        is ignored or refused. The item the action rests on is read again first: one
        that is gone is left out, so a repeat whose target was deleted is stored."""
        if holds_secret(item.title, item.text):
            return Route(REFUSE, None, None, "holds a secret")
        active = await self.load(item.embed_model)
        self._count += 1
        provenance = {"candidate_id": "new", "item_index": self._count}
        picked = None if judged is None else active.get(judged.target)
        if picked is not None and not await self._refresh([picked]):
            picked = None  # it changed since the judge read it: the rules decide
        while True:
            decision = self._rules(item, provenance)
            target = (
                None if decision.target_id is None else active.get(decision.target_id)
            )
            if target is None or await self._refresh([target]):
                break
        action, target_id, reason = decision.action, decision.target_id, decision.reason
        if judged is not None and not decision.settled and picked is not None:
            same = judged.verdict == "same"
            action, target_id = ("IGNORE" if same else "UPDATE"), judged.target
            reason = (
                f"judged {'the same as' if same else 'a change of'} entry {target_id}"
            )
        if action == "IGNORE" or (action != "NEW" and target_id is None):
            if target_id is not None:
                # The item repeats one kept: that one now cites this item's turns too,
                # so they stay behind it, whatever replaces it, and come back to
                # recall if it is forgotten; and it is as new as the repeat.
                await self._index.add_sources(target_id, item.sources, item.created_at)
                entry = active.get(target_id)
                if entry is not None:
                    active[target_id] = dataclasses.replace(
                        entry,
                        created_at=max(entry.created_at, item.created_at),
                        sources=tuple(dict.fromkeys((*entry.sources, *item.sources))),
                    )
            return Route("IGNORE", None, target_id, reason)
        replace = target_id if action == "UPDATE" else None
        if replace is not None and item.created_at < active[replace].created_at:
            # Newer always wins, whatever order the chats are encoded in: an older
            # fact is kept, already replaced by the newer one it would have replaced.
            item = dataclasses.replace(item, status="replaced", replaced_by=replace)
            reason += f"; older than entry {replace}, kept as replaced"
            replace = None
        # The item and its update or link are one write: a crash between them cannot
        # leave two current facts, or a replaced one with no replacement.
        new_id = await self._index.add_item(
            item,
            replace=replace,
            link=target_id if action in ("RELATED", "CONFLICT") else None,
            conflict=action == "CONFLICT",
        )
        if replace is not None:
            active.pop(replace, None)
        elif action == "CONFLICT" and target_id is not None:
            active.pop(target_id, None)  # a conflict is no longer an active item
            return Route(action, new_id, target_id, reason)
        if item.status == "active":
            active[new_id] = _entry(dataclasses.replace(item, id=new_id), self._model)
        return Route(action, new_id, target_id, reason)
