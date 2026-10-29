from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from localmelo.melo.memory.assertions import decide_assertion
from localmelo.melo.memory.bm25 import BM25Index

# The typed store's closed sets (R6-lite).
KINDS = ("semantic", "episodic", "profile", "reflection")
SOURCES = (
    "reasoning_note",
    "digest",
    "reflection",
    "user",
    "sleep",
    "step_summary",
)
STATUSES = ("active", "superseded", "conflicted")
ACTIONS = ("NEW", "UPDATE", "RELATED", "CONFLICT", "IGNORE")
SIGNALS = ("rehearsal", "user_request", "user_statement", "outcome", "sleep")


class LongTermSchemaError(Exception):
    """The file was written by a newer version than this one understands."""


@dataclass(frozen=True)
class StoredEntry:
    """One typed long-term entry, as the store exposes it."""

    id: int
    text: str
    kind: str
    source: str
    status: str
    admitted_by: str
    provenance: dict[str, Any]
    task_id: str
    session_id: str
    uses: int
    created_at: float
    last_used_at: float
    supersedes: int | None
    links: tuple[int, ...]
    embed_model: str
    embed_dim: int
    vector: tuple[float, ...] | None


def _history_order(entry: StoredEntry) -> tuple[float, int]:
    assertion = entry.provenance.get("assertion")
    observed = assertion.get("observed_at") if isinstance(assertion, dict) else None
    if (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and observed > 0
    ):
        return float(observed), entry.id
    return entry.created_at, entry.id


@dataclass
class _Entry:
    """The mutable in-memory twin of one store row."""

    id: int
    text: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "episodic"
    source: str = "step_summary"
    status: str = "active"
    admitted_by: str = ""
    supersedes: int | None = None
    links: tuple[int, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    session_id: str = ""
    uses: int = 0
    created_at: float = 0.0
    last_used_at: float = 0.0
    embed_model: str = ""
    embed_dim: int = 0


def _meta(entry: _Entry) -> dict[str, Any]:
    """The legacy metadata merged with the typed columns, as search returns it."""
    provenance = copy.deepcopy(entry.provenance)
    origin = provenance.get("origin_task_ids")
    if isinstance(origin, list | tuple) and all(isinstance(i, str) for i in origin):
        origin_task_ids = list(origin)
    elif entry.task_id:
        origin_task_ids = [entry.task_id]
    else:
        origin_task_ids = []
    return {
        **entry.metadata,
        "id": entry.id,
        "kind": entry.kind,
        "source": entry.source,
        "status": entry.status,
        "admitted_by": entry.admitted_by,
        "provenance": provenance,
        "task_id": entry.task_id,
        "session_id": entry.session_id,
        "uses": entry.uses,
        "created_at": entry.created_at,
        "last_used_at": entry.last_used_at,
        "supersedes": entry.supersedes,
        "links": list(entry.links),
        "embed_model": entry.embed_model,
        "embed_dim": entry.embed_dim,
        "origin_task_ids": origin_task_ids,
    }


def _stored(entry: _Entry) -> StoredEntry:
    vector: tuple[float, ...] | None = (
        None if entry.vector.size == 0 else tuple(float(v) for v in entry.vector)
    )
    return StoredEntry(
        id=entry.id,
        text=entry.text,
        kind=entry.kind,
        source=entry.source,
        status=entry.status,
        admitted_by=entry.admitted_by,
        provenance=copy.deepcopy(entry.provenance),
        task_id=entry.task_id,
        session_id=entry.session_id,
        uses=entry.uses,
        created_at=entry.created_at,
        last_used_at=entry.last_used_at,
        supersedes=entry.supersedes,
        links=tuple(entry.links),
        embed_model=entry.embed_model,
        embed_dim=entry.embed_dim,
        vector=vector,
    )


def _tie_key(entry: _Entry, score: float) -> tuple[float, bool, float]:
    """The documented equal-score order: rehearsed, then newest."""
    return (
        score,
        entry.admitted_by == "rehearsal",
        entry.created_at if entry.admitted_by else 0.0,
    )


class LongTerm:
    """In-memory, typed long-term memory (the tests' default store)."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._decisions: list[dict[str, Any]] = []
        self._next_id = 1
        self._index: BM25Index | None = None
        self._indexed: dict[str, _Entry] = {}

    async def add(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = dict(metadata or {})
        entry = _Entry(
            id=self._next_id,
            text=text,
            vector=np.asarray(embedding, dtype=np.float32),
            metadata=meta,
            task_id=str(meta.get("task_id") or ""),
            session_id=str(meta.get("session_id") or ""),
            created_at=time.time(),
            embed_dim=len(embedding),
        )
        self._next_id += 1
        self._entries.append(entry)
        self._index = None

    async def apply(
        self,
        *,
        text: str,
        kind: str,
        source: str,
        admitted_by: str,
        provenance: dict[str, Any],
        task_id: str,
        session_id: str,
        created_at: float,
        embedding: list[float] | None,
        action: str,
        target_id: int | None,
        reason: str,
        source_ids: Any,
        keep_text: bool,
        candidate_id: str,
        typed_assertion: bool = False,
        decision_details: dict[str, Any] | None = None,
    ) -> tuple[int | None, str]:
        """Write one decision in one step; returns ``(entry_id, action)``."""
        for name, value, allowed in (
            ("kind", kind, KINDS),
            ("source", source, SOURCES),
            ("admitted_by", admitted_by, SIGNALS),
            ("action", action, ACTIONS),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown long-term {name}: {value!r}")
        assertion = provenance.get("assertion")
        if typed_assertion and isinstance(assertion, dict) and task_id:
            rows = [
                (entry.id, entry.provenance["assertion"], entry.created_at)
                for entry in self._entries
                if entry.status == "active"
                and entry.kind == kind
                and entry.task_id == task_id
                and isinstance(entry.provenance.get("assertion"), dict)
            ]
            action, target_id, reason = decide_assertion(assertion, rows, created_at)
        target = self._find(target_id) if target_id is not None else None
        if action == "UPDATE" and (target is None or target.status != "active"):
            # The SQLite twin clears the target on this downgrade; the two
            # stores must answer the same decision row (review finding 12).
            action = "NEW"
            target = None
            target_id = None
        if action == "IGNORE":
            if target is not None:
                target.uses += 1
                target.last_used_at = time.time()
            self._decisions.append(
                self._decision_row(
                    len(self._decisions) + 1,
                    candidate_id=candidate_id,
                    action=action,
                    kind=kind,
                    admitted_by=admitted_by,
                    entry_id=None,
                    target_id=target_id,
                    reason=reason,
                    source_ids=source_ids,
                    text=text,
                    keep_text=keep_text,
                    task_id=task_id,
                )
            )
            self._index = None
            if decision_details is not None:
                decision_details.update(target_id=target_id, reason=reason)
            return None, action
        now = created_at or time.time()
        vector = (
            np.asarray(embedding, dtype=np.float32)
            if embedding is not None
            else np.empty(0, dtype=np.float32)
        )
        links: tuple[int, ...] = ()
        supersedes: int | None = None
        if action == "UPDATE" and target is not None:
            supersedes = target.id
            links = (target.id,)
            target.status = "superseded"
        elif action in ("CONFLICT", "RELATED") and target is not None:
            links = (target.id,)
        meta: dict[str, Any] = {
            "task_id": task_id,
            "session_id": session_id,
            "created_at": now,
            "origin_task_ids": list(provenance.get("origin_task_ids") or ())
            or ([task_id] if task_id else []),
        }
        if kind == "reflection":
            meta["type"] = "reflection"
        entry = _Entry(
            id=self._next_id,
            text=text,
            vector=vector,
            metadata=meta,
            kind=kind,
            source=source,
            status="active",
            admitted_by=admitted_by,
            supersedes=supersedes,
            links=links,
            provenance=copy.deepcopy(provenance),
            task_id=task_id,
            session_id=session_id,
            created_at=now,
            embed_dim=int(vector.size),
        )
        self._next_id += 1
        self._entries.append(entry)
        self._decisions.append(
            self._decision_row(
                len(self._decisions) + 1,
                candidate_id=candidate_id,
                action=action,
                kind=kind,
                admitted_by=admitted_by,
                entry_id=entry.id,
                target_id=target_id,
                reason=reason,
                source_ids=source_ids,
                text=text,
                keep_text=keep_text,
                task_id=task_id,
            )
        )
        self._index = None
        if decision_details is not None:
            decision_details.update(target_id=target_id, reason=reason)
        return entry.id, action

    @staticmethod
    def _decision_row(
        seq: int,
        *,
        candidate_id: str,
        action: str,
        kind: str,
        admitted_by: str,
        entry_id: int | None,
        target_id: int | None,
        reason: str,
        source_ids: Any,
        text: str,
        keep_text: bool,
        task_id: str,
    ) -> dict[str, Any]:
        import hashlib

        return {
            "seq": seq,
            "candidate_id": candidate_id,
            "action": action,
            "kind": kind,
            "admitted_by": admitted_by,
            "entry_id": entry_id,
            "target_id": target_id,
            "reason": reason,
            "source_ids": list(source_ids),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text if keep_text else None,
            "timestamp": time.time(),
            "task_id": task_id,
        }

    def _find(self, entry_id: int | None) -> _Entry | None:
        if entry_id is None:
            return None
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def _active(self, kinds: Any = None, exclude: Any = ()) -> list[_Entry]:
        excluded = set(exclude)
        return [
            entry
            for entry in self._entries
            if entry.status == "active"
            and (not kinds or entry.kind in kinds)
            and entry.task_id not in excluded
        ]

    async def count_entries(self) -> int:
        """The number of entries held (see SqliteLongTerm.count_entries)."""
        return len(self._entries)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        kinds: Any = None,
        exclude_task_ids: Any = (),
        task_key: str = "",
    ) -> list[tuple[str, float, dict[str, Any]]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        scored: list[tuple[_Entry, float]] = []
        for entry in self._active(kinds, exclude_task_ids):
            if entry.vector.size != q.shape[0]:
                continue
            norm = np.linalg.norm(entry.vector)
            if norm == 0:
                continue
            scored.append((entry, float(np.dot(q, entry.vector / norm))))
        if task_key:
            scored = [
                item
                for item in scored
                if item[0].provenance.get("task_key") == task_key
            ]
        scored.sort(key=lambda item: _tie_key(item[0], item[1]), reverse=True)
        return [(entry.text, score, _meta(entry)) for entry, score in scored[:top_k]]

    async def search_embedding(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Alias of :meth:`search`, the name the typed API uses."""
        return await self.search(*args, **kwargs)

    async def search_text(
        self,
        query: str,
        top_k: int = 5,
        *,
        kinds: Any = None,
        exclude_task_ids: Any = (),
        task_key: str = "",
        include_history: bool = False,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if include_history:
            excluded = set(exclude_task_ids)
            rows = [
                entry
                for entry in self._entries
                if entry.status in ("active", "superseded")
                and (not kinds or entry.kind in kinds)
                and entry.task_id not in excluded
            ]
        else:
            rows = self._active(kinds, exclude_task_ids)
        if not rows or not query.strip():
            return []
        index = BM25Index()
        by_id: dict[str, _Entry] = {}
        for entry in rows:
            index.add(str(entry.id), entry.text)
            by_id[str(entry.id)] = entry
        scored = [
            (by_id[doc_id], score)
            for doc_id, score in index.search(query, len(rows))
            if score > 0
        ]
        if task_key:
            scored = [
                item
                for item in scored
                if item[0].provenance.get("task_key") == task_key
            ]
        scored.sort(key=lambda item: _tie_key(item[0], item[1]), reverse=True)
        return [(entry.text, score, _meta(entry)) for entry, score in scored[:top_k]]

    async def active_entries(self, kind: str | None = None) -> list[StoredEntry]:
        rows = [
            entry
            for entry in self._entries
            if entry.status == "active" and (kind is None or entry.kind == kind)
        ]
        rows.sort(key=lambda entry: (entry.created_at, entry.id))
        return [_stored(entry) for entry in rows]

    async def history_entries(self, task_id: str) -> list[StoredEntry]:
        """Return a task's entries, including superseded rows, newest first."""
        if not task_id:
            return []
        rows = [_stored(entry) for entry in self._entries if entry.task_id == task_id]
        rows.sort(key=_history_order, reverse=True)
        return rows

    async def decisions(self) -> list[dict[str, Any]]:
        return list(self._decisions)

    async def forget(self, task_id: str) -> None:
        self._entries = [e for e in self._entries if e.task_id != task_id]
        self._decisions = [d for d in self._decisions if d.get("task_id") != task_id]
        self._index = None

    async def forget_all(self) -> None:
        self._entries.clear()
        self._decisions.clear()
        self._index = None
