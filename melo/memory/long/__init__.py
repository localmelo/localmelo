from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

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
# EDIT and DELETE are the user's own changes to a record.
ACTIONS = ("NEW", "UPDATE", "RELATED", "CONFLICT", "IGNORE", "EDIT", "DELETE")
# The signals a run admits by; its counters (RunResult ``admitted``) have one each.
SIGNALS = ("rehearsal", "user_request", "user_statement", "outcome", "sleep")
# What the store also accepts: the user's own tap on a proposal, or edit, and a
# line brought in from a file another assistant keeps ("import").
STORE_SIGNALS = (*SIGNALS, "user_confirm", "user_edit", "import")
# A decision row keeps at most this many characters of its reason.
MAX_REASON_CHARS = 500


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


def _typed_meta(entry: _Entry) -> dict[str, Any]:
    """The legacy metadata merged with the typed columns, as search returns it.

    The provenance is used as it stands, so the caller passes a copy it owns.
    """
    provenance = entry.provenance
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


def _meta(entry: _Entry) -> dict[str, Any]:
    """:func:`_typed_meta` of *entry*, with a copy of its provenance."""
    return _typed_meta(replace(entry, provenance=copy.deepcopy(entry.provenance)))


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


def _tie_key(
    score: float, admitted_by: str, created_at: float
) -> tuple[float, bool, float]:
    """The documented equal-score order: rehearsed, then newest.

    The sort is stable, so the final tie keeps the rows' own order (id
    ascending): a legacy file's equal scores stay in the order it wrote.
    """
    return (
        score,
        admitted_by == "rehearsal",
        float(created_at or 0.0) if admitted_by else 0.0,
    )


def _by_tie_key(item: tuple[_Entry, float]) -> tuple[float, bool, float]:
    entry, score = item
    return _tie_key(score, entry.admitted_by, entry.created_at)


def check_write(kind: str, source: str, admitted_by: str, action: str) -> None:
    """Refuse a value outside the closed sets, before any store is touched."""
    for name, value, allowed in (
        ("kind", kind, KINDS),
        ("source", source, SOURCES),
        ("admitted_by", admitted_by, STORE_SIGNALS),
        ("action", action, ACTIONS),
    ):
        if value not in allowed:
            raise ValueError(f"Unknown long-term {name}: {value!r}")


@dataclass(frozen=True)
class WritePlan:
    """What a store writes for one decision; see :func:`plan_write`.

    ``action`` and ``target_id`` are the final ones, as the decision row keeps
    them. ``links`` and ``supersedes`` name rows that exist. ``metadata`` and
    ``created_at`` are for the new entry; an ``IGNORE`` writes no entry.
    """

    action: str
    target_id: int | None
    reason: str
    created_at: float
    links: tuple[int, ...]
    supersedes: int | None
    metadata: dict[str, Any]


def plan_write(
    *,
    kind: str,
    action: str,
    target_id: int | None,
    target_status: str | None,
    reason: str,
    provenance: dict[str, Any],
    task_id: str,
    session_id: str,
    created_at: float,
    now: float,
) -> WritePlan:
    """Decide what one write does. Pure: both stores call it and keep their I/O.

    The caller has run :func:`check_write`. *target_status* is the status of
    the row *target_id* names, ``None`` when the store has no such row, and
    *now* is the store's clock.

    - An ``UPDATE`` whose target is missing or not active becomes a ``NEW``
      and drops its target, so the decision row says what happened.
    - An ``UPDATE`` supersedes its target and links to it; a ``CONFLICT`` or
      a ``RELATED`` links to its target only when that row exists. A missing
      target stays in the decision row, with no link.
    - The decision row keeps at most ``MAX_REASON_CHARS`` of the reason.
    """
    if action == "UPDATE" and target_status != "active":
        action, target_id = "NEW", None
    exists = target_id is not None and target_status is not None
    linked = exists and action in ("UPDATE", "RELATED", "CONFLICT")
    created = created_at or now
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "session_id": session_id,
        "created_at": created,
        "origin_task_ids": list(provenance.get("origin_task_ids") or ())
        or ([task_id] if task_id else []),
    }
    if kind == "reflection":
        metadata["type"] = "reflection"
    return WritePlan(
        action=action,
        target_id=target_id,
        reason=reason[:MAX_REASON_CHARS],
        created_at=created,
        links=(target_id,) if linked and target_id is not None else (),
        supersedes=target_id if linked and action == "UPDATE" else None,
        metadata=metadata,
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
        record: dict[str, str] | None = None,
    ) -> tuple[int | None, str]:
        """Write one decision in one step; returns ``(entry_id, action)``.

        *record* is for a records file's columns; this store keeps none.
        """
        check_write(kind, source, admitted_by, action)
        now = time.time()
        target = self._find(target_id)
        plan = plan_write(
            kind=kind,
            action=action,
            target_id=target_id,
            target_status=None if target is None else target.status,
            reason=reason,
            provenance=provenance,
            task_id=task_id,
            session_id=session_id,
            created_at=created_at,
            now=now,
        )
        entry_id: int | None = None
        if plan.action == "IGNORE":
            if target is not None:
                target.uses += 1
                target.last_used_at = now
        else:
            vector = (
                np.asarray(embedding, dtype=np.float32)
                if embedding is not None
                else np.empty(0, dtype=np.float32)
            )
            entry = _Entry(
                id=self._next_id,
                text=text,
                vector=vector,
                metadata=plan.metadata,
                kind=kind,
                source=source,
                status="active",
                admitted_by=admitted_by,
                supersedes=plan.supersedes,
                links=plan.links,
                provenance=copy.deepcopy(provenance),
                task_id=task_id,
                session_id=session_id,
                created_at=plan.created_at,
                embed_dim=int(vector.size),
            )
            self._next_id += 1
            self._entries.append(entry)
            entry_id = entry.id
            if target is not None and plan.supersedes is not None:
                target.status = "superseded"
        self._decisions.append(
            self._decision_row(
                len(self._decisions) + 1,
                candidate_id=candidate_id,
                action=plan.action,
                kind=kind,
                admitted_by=admitted_by,
                entry_id=entry_id,
                target_id=plan.target_id,
                reason=plan.reason,
                source_ids=source_ids,
                text=text,
                keep_text=keep_text,
                task_id=task_id,
            )
        )
        self._index = None
        return entry_id, plan.action

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
        scored.sort(key=_by_tie_key, reverse=True)
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
    ) -> list[tuple[str, float, dict[str, Any]]]:
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
        scored.sort(key=_by_tie_key, reverse=True)
        return [(entry.text, score, _meta(entry)) for entry, score in scored[:top_k]]

    async def active_entries(self, kind: str | None = None) -> list[StoredEntry]:
        rows = [
            entry
            for entry in self._entries
            if entry.status == "active" and (kind is None or entry.kind == kind)
        ]
        rows.sort(key=lambda entry: (entry.created_at, entry.id))
        return [_stored(entry) for entry in rows]

    async def decisions(self) -> list[dict[str, Any]]:
        return list(self._decisions)

    async def delete_entry(self, entry_id: int, *, restore: bool = False) -> bool:
        """Delete one entry (see :meth:`SqliteLongTerm.delete_entry`)."""
        entry = self._find(entry_id)
        if entry is None:
            return False
        self._entries.remove(entry)
        for row in self._decisions:
            if entry_id in (row.get("entry_id"), row.get("target_id")):
                row["text"] = None
                row["text_sha256"] = ""
        previous = self._find(entry.supersedes)
        if restore and previous is not None and previous.status == "superseded":
            previous.status = "active"
        self._decisions.append(
            self._decision_row(
                len(self._decisions) + 1,
                candidate_id="",
                action="DELETE",
                kind=entry.kind,
                admitted_by="user_edit",
                entry_id=None,
                target_id=entry_id,
                reason="deleted by the user",
                source_ids=(),
                text="",
                keep_text=False,
                task_id="",
            )
        )
        self._decisions[-1]["text_sha256"] = ""
        self._index = None
        return True

    async def forget(self, task_id: str) -> None:
        self._entries = [e for e in self._entries if e.task_id != task_id]
        self._decisions = [d for d in self._decisions if d.get("task_id") != task_id]
        self._index = None

    async def forget_all(self) -> None:
        self._entries.clear()
        self._decisions.clear()
        self._index = None
