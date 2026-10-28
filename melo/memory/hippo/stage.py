"""The staged candidates of one Agent: RAM only, capped, replace by id."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

from localmelo.melo.schema import MemoryCandidate

STAGE_CAP = 300


class StagedCandidates:
    """An ordered set of candidates, capped at :data:`STAGE_CAP`.

    ``propose`` replaces a known id's candidate (the restatement rule) and
    drops the oldest when a new id would exceed the cap. It never stages a
    candidate the store already wrote: a re-proposal's text changed, so it
    is a new write. The stage lives in RAM and dies with the Agent.
    """

    def __init__(self, cap: int = STAGE_CAP) -> None:
        self._cap = cap
        self._items: OrderedDict[str, MemoryCandidate] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def propose(self, candidate: MemoryCandidate) -> None:
        existing = self._items.get(candidate.candidate_id)
        if existing is not None:
            if existing.processed and existing.text == candidate.text:
                # An unchanged restatement is not a second write.
                self._items.move_to_end(candidate.candidate_id)
                return
            self._items[candidate.candidate_id] = replace(
                candidate, entry_id=None, processed=False
            )
            self._items.move_to_end(candidate.candidate_id)
            return
        self._items[candidate.candidate_id] = replace(
            candidate, entry_id=None, processed=False
        )
        while len(self._items) > self._cap:
            self._items.popitem(last=False)

    def get(self, candidate_id: str) -> MemoryCandidate | None:
        return self._items.get(candidate_id)

    def all(self) -> list[MemoryCandidate]:
        return list(self._items.values())

    def update(self, candidate: MemoryCandidate) -> None:
        """Store the staged copy's bookkeeping after a decision."""
        if candidate.candidate_id in self._items:
            self._items[candidate.candidate_id] = candidate

    def drop_task(self, task_id: str) -> None:
        for candidate_id, candidate in list(self._items.items()):
            if candidate.task_id == task_id:
                del self._items[candidate_id]

    def clear(self) -> None:
        self._items.clear()
