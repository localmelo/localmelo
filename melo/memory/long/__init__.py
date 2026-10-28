from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Entry:
    text: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class LongTerm:
    def __init__(self) -> None:
        self._entries: list[Entry] = []

    async def add(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        self._entries.append(Entry(text=text, vector=vec, metadata=metadata or {}))

    async def count_entries(self) -> int:
        """The number of entries held (see SqliteLongTerm.count_entries)."""
        return len(self._entries)

    async def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._entries:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores: list[tuple[int, float]] = []
        for i, entry in enumerate(self._entries):
            # A vector from another embedding model has another dimension;
            # it cannot be compared, so it is left out rather than crashing.
            if entry.vector.shape != q.shape:
                continue
            e_norm = np.linalg.norm(entry.vector)
            if e_norm == 0:
                continue
            sim = float(np.dot(q, entry.vector / e_norm))
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            (self._entries[i].text, sim, self._entries[i].metadata)
            for i, sim in scores[:top_k]
        ]

    async def forget(self, task_id: str) -> None:
        """Forget the entries whose metadata names *task_id* as their task.

        An entry's provenance is its ``metadata`` object: it names a task
        when ``metadata["task_id"] == task_id``. The reflections that
        ``Hippo.promote_reflections`` writes carry that key; a step summary
        carries only ``"step_id"``, so a per-task forget leaves it and only
        :meth:`forget_all` removes it.
        """
        self._entries = [
            entry for entry in self._entries if entry.metadata.get("task_id") != task_id
        ]

    async def forget_all(self) -> None:
        """Forget every long-term entry, whatever its provenance."""
        self._entries.clear()
