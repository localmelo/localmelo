"""The procedural store: routines, episodes and template versions.

:class:`SqliteProcedures` keeps them in ``procedures.db``, through the
aiosqlite helpers that ``SqliteHistory`` uses, with ``busy_timeout`` 500
ms; :class:`MemoryProcedures` keeps them in RAM for the life of an agent.
Both run the same logic:

- :meth:`Procedures.lookup` finds, at a run's start, the template a
  display may rest on (:class:`~.template.TemplateView`);
- :meth:`Procedures.learn`, at a run's end, records the run's episode and,
  for a completed run, appends the template's next version, all in one
  ``BEGIN IMMEDIATE`` transaction (section 5.6);
- :meth:`Procedures.add_candidate` and :meth:`Procedures.decide_candidate`
  are a sleep job's interface (section 8);
- :meth:`Procedures.export_rows` and :meth:`Procedures.forget` serve
  ``melo-procedures``.

The tables are append-only: a routine or episode row never changes, and a
template row changes only its status (``active`` → ``superseded``,
``candidate`` → ``active`` or ``rejected``), which triggers in the file
enforce. At most one version of a template is ``active`` (a unique
index). Rows leave only through :meth:`Procedures.forget`.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Protocol

import aiosqlite

from localmelo.melo.memory._sqlite import create_private_file, prepare_connection
from localmelo.melo.memory.procedures.normalize import (
    RoutineKey,
    find_similar,
    known_model,
    template_id,
)
from localmelo.melo.memory.procedures.template import (
    COMPLETED,
    DEFAULT_PARAMS,
    Episode,
    Params,
    SkeletonItem,
    Stats,
    Template,
    TemplateView,
    build_version,
    decide_level,
    view_of,
)
from localmelo.melo.memory.procedures.trace import Segment, Trace, trace_from_segments
from localmelo.melo.memory.procedures.tracker import shadow_bar_error

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 1
BUSY_TIMEOUT_MS: Final = 500
# A sleep candidate is activated only when its replay caps at most this
# share of steps.
CANDIDATE_CAP_SHARE: Final = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routines (
  routine_id TEXT PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  name TEXT,
  query_text TEXT NOT NULL DEFAULT '',
  place TEXT NOT NULL DEFAULT '',
  similar_to TEXT,
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
  task_id TEXT PRIMARY KEY,
  routine_id TEXT NOT NULL, template_id TEXT NOT NULL, model TEXT NOT NULL,
  harness_version TEXT NOT NULL, match TEXT NOT NULL, level INTEGER NOT NULL,
  template_version INTEGER, status TEXT NOT NULL, steps INTEGER NOT NULL,
  attempts INTEGER NOT NULL, n_tool INTEGER NOT NULL, duration_s REAL NOT NULL,
  first_pass_step INTEGER, cycles INTEGER NOT NULL, segments_json TEXT NOT NULL,
  shadow_bar_error REAL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS templates (
  template_id TEXT NOT NULL, version INTEGER NOT NULL,
  routine_id TEXT NOT NULL, model TEXT NOT NULL, options_json TEXT NOT NULL,
  parent_version INTEGER, status TEXT NOT NULL, source TEXT NOT NULL,
  admitted_by TEXT NOT NULL, source_ids TEXT NOT NULL,
  skeleton_json TEXT NOT NULL, stats_json TEXT NOT NULL,
  params_json TEXT NOT NULL, eval_json TEXT, harness_version TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (template_id, version));
CREATE INDEX IF NOT EXISTS episodes_template ON episodes(template_id, created_at);
CREATE INDEX IF NOT EXISTS templates_status ON templates(template_id, status);
CREATE INDEX IF NOT EXISTS routines_place ON routines(place);
CREATE UNIQUE INDEX IF NOT EXISTS templates_one_active
  ON templates(template_id) WHERE status = 'active';
CREATE TRIGGER IF NOT EXISTS routines_append_only BEFORE UPDATE ON routines
BEGIN SELECT RAISE(ABORT, 'routines rows never change'); END;
CREATE TRIGGER IF NOT EXISTS episodes_append_only BEFORE UPDATE ON episodes
BEGIN SELECT RAISE(ABORT, 'episodes rows never change'); END;
CREATE TRIGGER IF NOT EXISTS templates_append_only BEFORE UPDATE ON templates
WHEN NEW.template_id IS NOT OLD.template_id OR NEW.version IS NOT OLD.version
  OR NEW.routine_id IS NOT OLD.routine_id OR NEW.model IS NOT OLD.model
  OR NEW.options_json IS NOT OLD.options_json
  OR NEW.parent_version IS NOT OLD.parent_version
  OR NEW.source IS NOT OLD.source OR NEW.admitted_by IS NOT OLD.admitted_by
  OR NEW.source_ids IS NOT OLD.source_ids
  OR NEW.skeleton_json IS NOT OLD.skeleton_json
  OR NEW.stats_json IS NOT OLD.stats_json OR NEW.params_json IS NOT OLD.params_json
  OR NEW.harness_version IS NOT OLD.harness_version
  OR NEW.created_at IS NOT OLD.created_at
  OR (NEW.eval_json IS NOT OLD.eval_json
      AND NOT (OLD.status = 'candidate' AND NEW.status = 'rejected'))
  OR NOT ((OLD.status = 'active' AND NEW.status = 'superseded')
          OR (OLD.status = 'candidate' AND NEW.status IN ('active', 'rejected')))
BEGIN SELECT RAISE(ABORT, 'a template row changes only its status'); END;
"""

TABLES: Final = ("routines", "episodes", "templates")
_ROUTINE_COLUMNS: Final = (
    "routine_id",
    "key",
    "name",
    "query_text",
    "place",
    "similar_to",
    "created_at",
)
_EPISODE_COLUMNS: Final = (
    "task_id",
    "routine_id",
    "template_id",
    "model",
    "harness_version",
    "match",
    "level",
    "template_version",
    "status",
    "steps",
    "attempts",
    "n_tool",
    "duration_s",
    "first_pass_step",
    "cycles",
    "segments_json",
    "shadow_bar_error",
    "created_at",
)
_TEMPLATE_COLUMNS: Final = (
    "template_id",
    "version",
    "routine_id",
    "model",
    "options_json",
    "parent_version",
    "status",
    "source",
    "admitted_by",
    "source_ids",
    "skeleton_json",
    "stats_json",
    "params_json",
    "eval_json",
    "harness_version",
    "created_at",
)
COLUMNS: Final = {
    "routines": _ROUTINE_COLUMNS,
    "episodes": _EPISODE_COLUMNS,
    "templates": _TEMPLATE_COLUMNS,
}
# The keys a sleep candidate's evaluation must hold.
EVAL_KEYS: Final = (
    "heldout_task_ids",
    "bar_error",
    "parent_bar_error",
    "cap_share",
    "longer_rate",
    "reason",
)


class ProceduresSchemaError(RuntimeError):
    """``procedures.db`` has a newer schema version than this code reads."""


def harness_version() -> str:
    """The installed localmelo's version, or ``"unknown"`` without metadata."""
    try:
        return importlib.metadata.version("localmelo")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class Routine:
    """A recurring task: a row of ``routines``."""

    routine_id: str
    key: str
    name: str | None
    query_text: str
    place: str
    created_at: float
    similar_to: str | None = None

    @property
    def title(self) -> str:
        """The routine's name, else the start of its query."""
        return self.name if self.name else self.query_text[:60]

    def routine_key(self) -> RoutineKey:
        return RoutineKey(
            key=self.key, place=self.place, name=self.name, query_text=self.query_text
        )


@dataclass(frozen=True)
class RunFacts:
    """What :meth:`Procedures.learn` keeps of a finished run.

    ``segments``, ``first_pass_step``, ``cycles`` and ``n_tool`` describe
    its calls (:meth:`from_trace` reads them from its trace); ``trace``
    replays the run for its shadow score, rebuilt from the segments when
    ``None``. ``created_at`` defaults to the store's clock.
    """

    task_id: str
    key: RoutineKey
    model: str
    options: Mapping[str, Any]
    status: str
    steps: int
    attempts: int = 1
    duration_s: float = 0.0
    segments: tuple[Segment, ...] = ()
    first_pass_step: int | None = None
    cycles: int = 0
    n_tool: int = 0
    trace: Trace | None = None
    harness_version: str = "unknown"
    created_at: float | None = None

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        task_id: str,
        key: RoutineKey,
        model: str,
        options: Mapping[str, Any],
        status: str,
        steps: int,
        attempts: int,
        duration_s: float,
        harness_version: str = "unknown",
    ) -> RunFacts:
        return cls(
            task_id=task_id,
            key=key,
            model=model,
            options=dict(options),
            status=status,
            steps=steps,
            attempts=attempts,
            duration_s=duration_s,
            segments=tuple(trace.segments()),
            first_pass_step=trace.first_pass_step,
            cycles=trace.cycles,
            n_tool=len(trace.calls),
            trace=trace,
            harness_version=harness_version,
        )


@dataclass(frozen=True)
class LearnResult:
    """What a run's end recorded: its episode and the new version, if any."""

    routine: Routine
    episode: Episode
    version: Template | None


# ── Rows ────────────────────────────────────────────────────────────────────


def routine_row(routine: Routine) -> dict[str, Any]:
    return {
        "routine_id": routine.routine_id,
        "key": routine.key,
        "name": routine.name,
        "query_text": routine.query_text,
        "place": routine.place,
        "similar_to": routine.similar_to,
        "created_at": routine.created_at,
    }


def routine_from_row(row: Mapping[str, Any]) -> Routine:
    return Routine(
        routine_id=str(row["routine_id"]),
        key=str(row["key"]),
        name=None if row.get("name") is None else str(row["name"]),
        query_text=str(row.get("query_text") or ""),
        place=str(row.get("place") or ""),
        similar_to=None if row.get("similar_to") is None else str(row["similar_to"]),
        created_at=float(row["created_at"]),
    )


def episode_row(episode: Episode) -> dict[str, Any]:
    return {
        "task_id": episode.task_id,
        "routine_id": episode.routine_id,
        "template_id": episode.template_id,
        "model": episode.model,
        "harness_version": episode.harness_version,
        "match": episode.match,
        "level": episode.level,
        "template_version": episode.template_version,
        "status": episode.status,
        "steps": episode.steps,
        "attempts": episode.attempts,
        "n_tool": episode.n_tool,
        "duration_s": episode.duration_s,
        "first_pass_step": episode.first_pass_step,
        "cycles": episode.cycles,
        "segments_json": json.dumps(
            [segment.as_dict() for segment in episode.segments], ensure_ascii=False
        ),
        "shadow_bar_error": episode.shadow_bar_error,
        "created_at": episode.created_at,
    }


def segments_from_json(
    text: str | None, *, strict: bool = False
) -> tuple[Segment, ...]:
    """The segments of a ``segments_json`` value.

    Malformed entries are skipped, or raise ``ValueError`` when *strict*.
    """
    try:
        entries = json.loads(text or "[]")
    except ValueError:
        if strict:
            raise ValueError("segments_json is not JSON") from None
        return ()
    if not isinstance(entries, list):
        if strict:
            raise ValueError("segments_json is not a list")
        return ()
    segments: list[Segment] = []
    for entry in entries:
        try:
            if not isinstance(entry, Mapping):
                raise ValueError("a segment is not an object")
            segments.append(Segment.from_dict(entry))
        except ValueError:
            if strict:
                raise
    return tuple(segments)


def episode_from_row(row: Mapping[str, Any], *, strict: bool = False) -> Episode:
    """An :class:`Episode` from a row; ``KeyError``/``ValueError`` when malformed."""

    def optional_int(name: str) -> int | None:
        value = row.get(name)
        return None if value is None else int(value)

    shadow = row.get("shadow_bar_error")
    return Episode(
        task_id=str(row["task_id"]),
        routine_id=str(row["routine_id"]),
        template_id=str(row["template_id"]),
        model=str(row.get("model") or ""),
        harness_version=str(row.get("harness_version") or "unknown"),
        match=str(row.get("match") or "new"),
        level=int(row.get("level") or 0),
        template_version=optional_int("template_version"),
        status=str(row["status"]),
        steps=int(row["steps"]),
        attempts=int(row.get("attempts") or 1),
        n_tool=int(row.get("n_tool") or 0),
        duration_s=float(row.get("duration_s") or 0.0),
        first_pass_step=optional_int("first_pass_step"),
        cycles=int(row.get("cycles") or 0),
        segments=segments_from_json(row.get("segments_json"), strict=strict),
        shadow_bar_error=None if shadow is None else float(shadow),
        created_at=float(row["created_at"]),
    )


def template_row(template: Template) -> dict[str, Any]:
    return {
        "template_id": template.template_id,
        "version": template.version,
        "routine_id": template.routine_id,
        "model": template.model,
        "options_json": json.dumps(dict(template.options), sort_keys=True),
        "parent_version": template.parent_version,
        "status": template.status,
        "source": template.source,
        "admitted_by": template.admitted_by,
        "source_ids": json.dumps(list(template.source_ids)),
        "skeleton_json": json.dumps(
            {
                "medoid": template.medoid,
                "items": [item.as_dict() for item in template.skeleton],
                "source": template.skeleton_source,
            },
            ensure_ascii=False,
        ),
        "stats_json": json.dumps(template.stats.as_dict()),
        "params_json": json.dumps(template.params.as_dict()),
        "eval_json": None if template.eval is None else json.dumps(dict(template.eval)),
        "harness_version": template.harness_version,
        "created_at": template.created_at,
    }


def template_from_row(row: Mapping[str, Any]) -> Template:
    skeleton = json.loads(row.get("skeleton_json") or "{}")
    eval_json = row.get("eval_json")
    parent = row.get("parent_version")
    return Template(
        template_id=str(row["template_id"]),
        version=int(row["version"]),
        routine_id=str(row["routine_id"]),
        model=str(row.get("model") or ""),
        options=json.loads(row.get("options_json") or "{}"),
        status=str(row["status"]),
        source=str(row.get("source") or "online"),
        admitted_by=str(row.get("admitted_by") or "harness"),
        source_ids=tuple(json.loads(row.get("source_ids") or "[]")),
        skeleton=tuple(
            SkeletonItem.from_dict(item) for item in skeleton.get("items", [])
        ),
        medoid=skeleton.get("medoid"),
        skeleton_source=str(skeleton.get("source") or "online"),
        stats=Stats.from_dict(json.loads(row.get("stats_json") or "{}")),
        params=Params.from_dict(json.loads(row.get("params_json") or "{}")),
        eval=None if eval_json is None else json.loads(eval_json),
        parent_version=None if parent is None else int(parent),
        harness_version=str(row.get("harness_version") or "unknown"),
        created_at=float(row["created_at"]),
    )


# ── The logic, over a transaction ───────────────────────────────────────────


class _Tx(Protocol):
    """The reads and writes the logic needs, inside one transaction."""

    async def routine_by_key(self, key: str) -> Routine | None: ...
    async def routines_in(self, place: str) -> list[tuple[str, str]]: ...
    async def routine(self, routine_id: str) -> Routine | None: ...
    async def insert_routine(self, routine: Routine) -> None: ...
    async def has_episode(self, task_id: str) -> bool: ...
    async def insert_episode(self, episode: Episode) -> None: ...
    async def window(
        self, template_id: str, statuses: frozenset[str], limit: int
    ) -> list[Episode]: ...
    async def first_episode(self, template_id: str) -> str | None: ...
    async def model_scores(self, model: str, limit: int) -> list[float]: ...
    async def active(self, template_id: str) -> Template | None: ...
    async def template(self, template_id: str, version: int) -> Template | None: ...
    async def max_version(self, template_id: str) -> int: ...
    async def insert_template(self, template: Template) -> None: ...
    async def set_status(
        self,
        template_id: str,
        version: int,
        status: str,
        eval: Mapping[str, Any] | None = None,
    ) -> None: ...
    async def rows(self, table: str) -> list[dict[str, Any]]: ...
    async def delete(self, routine_id: str | None) -> int: ...


async def _lookup(
    tx: _Tx, key: RoutineKey, model: str, options: Mapping[str, Any], params: Params
) -> TemplateView | None:
    routine = await tx.routine_by_key(key.key)
    title = None
    if routine is not None:
        match = "routine" if routine.name is not None else "exact"
    elif key.name is None:
        similar = find_similar(key.query_text, await tx.routines_in(key.place))
        if similar is None:
            return None
        routine = await tx.routine(similar.routine_id)
        if routine is None:
            return None
        match, title = "similar", routine.title
    else:
        return None
    active = await tx.active(template_id(routine.routine_id, model, options))
    if active is None:
        return None
    scores = await tx.model_scores(model, params.demotion_window)
    return view_of(active, match=match, model=model, model_scores=scores, title=title)


async def _learn(
    tx: _Tx,
    facts: RunFacts,
    params: Params,
    now: float,
    view: TemplateView | None,
) -> LearnResult:
    if await tx.has_episode(facts.task_id):
        raise ValueError(f"task {facts.task_id} already has an episode")
    routine = await tx.routine_by_key(facts.key.key)
    similar_routine: Routine | None = None
    if routine is None:
        similar_to = None
        if facts.key.name is None:
            similar = find_similar(
                facts.key.query_text, await tx.routines_in(facts.key.place)
            )
            if similar is not None:
                similar_to = similar.routine_id
                similar_routine = await tx.routine(similar.routine_id)
        routine = Routine(
            routine_id=uuid.uuid4().hex,
            key=facts.key.key,
            name=facts.key.name,
            query_text=facts.key.query_text,
            place=facts.key.place,
            similar_to=similar_to,
            created_at=now,
        )
        await tx.insert_routine(routine)
        match = "similar" if similar_to is not None else "new"
    else:
        match = "routine" if routine.name is not None else "exact"
    tid = template_id(routine.routine_id, facts.model, facts.options)
    active = await tx.active(tid)
    statuses = params.statistics_statuses
    window = await tx.window(tid, statuses, params.window - 1)
    first = await tx.first_episode(tid)

    version_shown: int | None
    if view is not None:
        level, version_shown = view.level, view.version
    elif active is not None:
        scores = await tx.model_scores(facts.model, params.demotion_window)
        level, _ = decide_level(
            active, match=match, model=facts.model, model_scores=scores
        )
        version_shown = active.version
    elif similar_routine is not None:
        borrowed = await tx.active(
            template_id(similar_routine.routine_id, facts.model, facts.options)
        )
        level, version_shown = (0, None) if borrowed is None else (1, None)
    else:
        level, version_shown = 0, None

    shadow = None
    if (
        active is not None
        and active.stats.gate
        and match in ("exact", "routine")
        and known_model(facts.model)
    ):
        trace = facts.trace or trace_from_segments(
            facts.segments, facts.steps, first_pass_step=facts.first_pass_step
        )
        scored = view_of(active, match=match, model=facts.model, model_scores=())
        shadow = shadow_bar_error(scored, trace, facts.steps)

    episode = Episode(
        task_id=facts.task_id,
        routine_id=routine.routine_id,
        template_id=tid,
        model=facts.model,
        harness_version=facts.harness_version,
        match=match,
        level=level,
        template_version=version_shown,
        status=facts.status,
        steps=facts.steps,
        attempts=max(1, facts.attempts),
        n_tool=facts.n_tool,
        duration_s=max(0.0, facts.duration_s),
        first_pass_step=facts.first_pass_step,
        cycles=facts.cycles,
        segments=tuple(facts.segments),
        shadow_bar_error=shadow,
        created_at=now,
    )
    await tx.insert_episode(episode)

    new_version = None
    if facts.status in statuses and facts.steps > 0:
        episodes = [*window, episode][-params.window :]
        if any(e.status == COMPLETED for e in episodes):
            new_version = build_version(
                template_id=tid,
                routine_id=routine.routine_id,
                model=facts.model,
                options=facts.options,
                window=episodes,
                first_task_id=first if first is not None else episode.task_id,
                parent=active,
                params=params,
                harness_version=facts.harness_version,
                created_at=now,
            )
            new_version = replace(new_version, version=await tx.max_version(tid) + 1)
            if active is not None:
                await tx.set_status(tid, active.version, "superseded")
            await tx.insert_template(new_version)
    return LearnResult(routine=routine, episode=episode, version=new_version)


async def _add_candidate(tx: _Tx, candidate: Template, now: float) -> Template:
    missing = [key for key in EVAL_KEYS if key not in (candidate.eval or {})]
    if missing:
        raise ValueError(f"a candidate's evaluation lacks {', '.join(missing)}")
    if candidate.parent_version is None:
        raise ValueError("a candidate names the version it replaces")
    if await tx.template(candidate.template_id, candidate.parent_version) is None:
        raise ValueError(
            f"template {candidate.template_id} has no version {candidate.parent_version}"
        )
    row = replace(
        candidate,
        version=await tx.max_version(candidate.template_id) + 1,
        status="candidate",
        source="sleep",
        admitted_by="sleep",
        skeleton_source="sleep",
        created_at=now,
    )
    await tx.insert_template(row)
    return row


async def _decide_candidate(tx: _Tx, tid: str, version: int) -> str:
    candidate = await tx.template(tid, version)
    if candidate is None or candidate.status != "candidate":
        raise ValueError(f"template {tid} has no candidate version {version}")
    evaluation = dict(candidate.eval or {})
    active = await tx.active(tid)
    if active is None or active.version != candidate.parent_version:
        await tx.set_status(tid, version, "rejected", {**evaluation, "reason": "stale"})
        return "rejected"
    try:
        better = float(evaluation["bar_error"]) < float(evaluation["parent_bar_error"])
        capped = float(evaluation["cap_share"]) <= CANDIDATE_CAP_SHARE
    except (KeyError, TypeError, ValueError):
        better = capped = False
    if not (better and capped):
        await tx.set_status(tid, version, "rejected", evaluation)
        return "rejected"
    await tx.set_status(tid, active.version, "superseded")
    await tx.set_status(tid, version, "active")
    return "active"


# ── The stores ──────────────────────────────────────────────────────────────


class Procedures(abc.ABC):
    """A procedural store; see the module docstring.

    *params* drive the learner (window, statistics, drop, gate) and the
    tracker; *clock* gives each new row's ``created_at``.
    """

    def __init__(
        self,
        *,
        params: Params = DEFAULT_PARAMS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.params = params
        self._clock = clock

    @abc.abstractmethod
    def _session(self, write: bool) -> contextlib.AbstractAsyncContextManager[_Tx]:
        """A transaction: rolled back on any exception, committed otherwise."""

    async def lookup(
        self, key: RoutineKey, *, model: str, options: Mapping[str, Any]
    ) -> TemplateView | None:
        """The template a run of *key* on *model* with *options* rests on.

        ``None`` when there is none: level 0. A routine found by its key
        gives the active version of its template for this model and these
        options; a new key in a place with enough routines may borrow a
        similar routine's template, at level 1.
        """
        async with self._session(False) as tx:
            return await _lookup(tx, key, model, options, self.params)

    async def learn(
        self, facts: RunFacts, *, view: TemplateView | None = None
    ) -> LearnResult:
        """Record a finished run: its routine, its episode, maybe a version.

        One transaction. *view* is what the run's lookup found, when it
        made one; its level and version are what the episode records,
        else the level the run would have shown. Raises ``ValueError`` when
        the task already has an episode.
        """
        now = facts.created_at if facts.created_at is not None else self._clock()
        async with self._session(True) as tx:
            return await _learn(tx, facts, self.params, now, view)

    async def add_candidate(self, candidate: Template) -> Template:
        """Append a sleep job's *candidate* version; it returns the row stored.

        The row gets the next version number and ``status`` ``candidate``,
        ``source`` and ``admitted_by`` ``sleep``. ``candidate.eval`` must
        hold :data:`EVAL_KEYS` and ``parent_version`` an existing version.
        """
        async with self._session(True) as tx:
            return await _add_candidate(tx, candidate, self._clock())

    async def decide_candidate(self, template_id_: str, version: int) -> str:
        """Activate or reject a candidate: ``"active"`` or ``"rejected"``.

        In one transaction: a candidate whose parent is no longer the
        active version is rejected with ``eval_json.reason`` ``stale``;
        one whose replay error is not below its parent's, or that caps more
        than :data:`CANDIDATE_CAP_SHARE` of steps, is rejected; otherwise
        the active version is superseded and the candidate becomes active.
        """
        async with self._session(True) as tx:
            return await _decide_candidate(tx, template_id_, version)

    async def export_rows(self) -> list[dict[str, Any]]:
        """Every row, as ``{"table": <name>, <column>: <value>, ...}``.

        Routines, then episodes, then templates, each in the order they
        were written; JSON columns stay JSON text.
        """
        async with self._session(False) as tx:
            rows: list[dict[str, Any]] = []
            for table in TABLES:
                rows.extend({"table": table, **row} for row in await tx.rows(table))
            return rows

    async def forget(self, routine_id: str | None = None) -> int:
        """Delete a routine with its episodes and templates, or everything.

        Returns the number of rows deleted.
        """
        async with self._session(True) as tx:
            return await tx.delete(routine_id)

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release the store's resources."""


class _MemoryState:
    def __init__(self) -> None:
        self.routines: dict[str, Routine] = {}
        self.episodes: list[Episode] = []
        self.templates: dict[tuple[str, int], Template] = {}


class _MemoryTx:
    """:class:`_Tx` over a :class:`_MemoryState`."""

    def __init__(self, state: _MemoryState) -> None:
        self.state = state

    async def routine_by_key(self, key: str) -> Routine | None:
        return next((r for r in self.state.routines.values() if r.key == key), None)

    async def routines_in(self, place: str) -> list[tuple[str, str]]:
        return [
            (r.routine_id, r.query_text)
            for r in self.state.routines.values()
            if r.place == place
        ]

    async def routine(self, routine_id: str) -> Routine | None:
        return self.state.routines.get(routine_id)

    async def insert_routine(self, routine: Routine) -> None:
        if routine.routine_id in self.state.routines or await self.routine_by_key(
            routine.key
        ):
            raise ValueError(f"routine {routine.routine_id} exists")
        self.state.routines[routine.routine_id] = routine

    async def has_episode(self, task_id: str) -> bool:
        return any(e.task_id == task_id for e in self.state.episodes)

    async def insert_episode(self, episode: Episode) -> None:
        if await self.has_episode(episode.task_id):
            raise ValueError(f"task {episode.task_id} already has an episode")
        self.state.episodes.append(episode)

    def _ordered(self) -> list[Episode]:
        indexed = list(enumerate(self.state.episodes))
        indexed.sort(key=lambda item: (item[1].created_at, item[0]))
        return [episode for _, episode in indexed]

    async def window(
        self, template_id: str, statuses: frozenset[str], limit: int
    ) -> list[Episode]:
        if limit <= 0:
            return []
        chosen = [
            e
            for e in self._ordered()
            if e.template_id == template_id and e.status in statuses and e.steps > 0
        ]
        return chosen[-limit:]

    async def first_episode(self, template_id: str) -> str | None:
        return next(
            (e.task_id for e in self._ordered() if e.template_id == template_id), None
        )

    async def model_scores(self, model: str, limit: int) -> list[float]:
        scores = [
            e.shadow_bar_error
            for e in self._ordered()
            if e.model == model and e.shadow_bar_error is not None
        ]
        return scores[-limit:]

    async def active(self, template_id: str) -> Template | None:
        return next(
            (
                t
                for (tid, _), t in self.state.templates.items()
                if tid == template_id and t.status == "active"
            ),
            None,
        )

    async def template(self, template_id: str, version: int) -> Template | None:
        return self.state.templates.get((template_id, version))

    async def max_version(self, template_id: str) -> int:
        return max(
            (v for tid, v in self.state.templates if tid == template_id), default=0
        )

    async def insert_template(self, template: Template) -> None:
        key = (template.template_id, template.version)
        if key in self.state.templates:
            raise ValueError(f"template {key} exists")
        if template.status == "active" and await self.active(template.template_id):
            raise ValueError(f"template {template.template_id} has an active version")
        self.state.templates[key] = template

    async def set_status(
        self,
        template_id: str,
        version: int,
        status: str,
        eval: Mapping[str, Any] | None = None,
    ) -> None:
        current = self.state.templates[(template_id, version)]
        allowed = {("active", "superseded"), ("candidate", "active")}
        allowed.add(("candidate", "rejected"))
        if (current.status, status) not in allowed:
            raise ValueError("a template row changes only its status")
        if eval is not None and (current.status, status) != ("candidate", "rejected"):
            raise ValueError("a template row changes only its status")
        if status == "active" and await self.active(template_id):
            raise ValueError(f"template {template_id} has an active version")
        self.state.templates[(template_id, version)] = replace(
            current, status=status, eval=current.eval if eval is None else dict(eval)
        )

    async def rows(self, table: str) -> list[dict[str, Any]]:
        if table == "routines":
            return [routine_row(r) for r in self.state.routines.values()]
        if table == "episodes":
            return [episode_row(e) for e in self.state.episodes]
        return [template_row(t) for t in self.state.templates.values()]

    async def delete(self, routine_id: str | None) -> int:
        state = self.state

        def doomed(owner: str) -> bool:
            return routine_id is None or owner == routine_id

        before = len(state.routines) + len(state.episodes) + len(state.templates)
        state.routines = {k: r for k, r in state.routines.items() if not doomed(k)}
        state.episodes = [e for e in state.episodes if not doomed(e.routine_id)]
        state.templates = {
            k: t for k, t in state.templates.items() if not doomed(t.routine_id)
        }
        return before - len(state.routines) - len(state.episodes) - len(state.templates)


class MemoryProcedures(Procedures):
    """A procedural store in RAM, for the life of one agent.

    The same logic as :class:`SqliteProcedures`; a failed operation leaves
    the store as it was.
    """

    def __init__(
        self,
        *,
        params: Params = DEFAULT_PARAMS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(params=params, clock=clock)
        self._state = _MemoryState()
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def _session(self, write: bool) -> AsyncIterator[_Tx]:
        async with self._lock:
            state = self._state
            saved = (dict(state.routines), list(state.episodes), dict(state.templates))
            try:
                yield _MemoryTx(state)
            except BaseException:
                state.routines, state.episodes, state.templates = saved
                raise

    async def aclose(self) -> None:
        """Nothing to release: the rows stay in RAM until the store is dropped."""


class _SqlTx:
    """:class:`_Tx` over an aiosqlite connection."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def _all(self, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self.conn.execute(sql, tuple(args)) as cursor:
            names = [column[0] for column in cursor.description or ()]
            return [
                dict(zip(names, row, strict=True)) for row in await cursor.fetchall()
            ]

    async def _one(self, sql: str, args: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = await self._all(sql, args)
        return rows[0] if rows else None

    async def _insert(self, table: str, row: Mapping[str, Any]) -> None:
        columns = COLUMNS[table]
        await self.conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            tuple(row[column] for column in columns),
        )

    async def routine_by_key(self, key: str) -> Routine | None:
        row = await self._one("SELECT * FROM routines WHERE key = ?", (key,))
        return None if row is None else routine_from_row(row)

    async def routines_in(self, place: str) -> list[tuple[str, str]]:
        rows = await self._all(
            "SELECT routine_id, query_text FROM routines WHERE place = ? "
            "ORDER BY created_at, rowid",
            (place,),
        )
        return [(row["routine_id"], row["query_text"]) for row in rows]

    async def routine(self, routine_id: str) -> Routine | None:
        row = await self._one(
            "SELECT * FROM routines WHERE routine_id = ?", (routine_id,)
        )
        return None if row is None else routine_from_row(row)

    async def insert_routine(self, routine: Routine) -> None:
        await self._insert("routines", routine_row(routine))

    async def has_episode(self, task_id: str) -> bool:
        return (
            await self._one(
                "SELECT 1 AS found FROM episodes WHERE task_id = ?", (task_id,)
            )
        ) is not None

    async def insert_episode(self, episode: Episode) -> None:
        await self._insert("episodes", episode_row(episode))

    async def window(
        self, template_id: str, statuses: frozenset[str], limit: int
    ) -> list[Episode]:
        if limit <= 0:
            return []
        marks = ", ".join("?" for _ in statuses)
        rows = await self._all(
            f"SELECT * FROM episodes WHERE template_id = ? AND status IN ({marks}) "
            "AND steps > 0 ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (template_id, *sorted(statuses), limit),
        )
        return [episode_from_row(row) for row in reversed(rows)]

    async def first_episode(self, template_id: str) -> str | None:
        row = await self._one(
            "SELECT task_id FROM episodes WHERE template_id = ? "
            "ORDER BY created_at, rowid LIMIT 1",
            (template_id,),
        )
        return None if row is None else str(row["task_id"])

    async def model_scores(self, model: str, limit: int) -> list[float]:
        rows = await self._all(
            "SELECT shadow_bar_error FROM episodes WHERE model = ? "
            "AND shadow_bar_error IS NOT NULL ORDER BY created_at DESC, rowid DESC "
            "LIMIT ?",
            (model, limit),
        )
        return [float(row["shadow_bar_error"]) for row in reversed(rows)]

    async def active(self, template_id: str) -> Template | None:
        row = await self._one(
            "SELECT * FROM templates WHERE template_id = ? AND status = 'active'",
            (template_id,),
        )
        return None if row is None else template_from_row(row)

    async def template(self, template_id: str, version: int) -> Template | None:
        row = await self._one(
            "SELECT * FROM templates WHERE template_id = ? AND version = ?",
            (template_id, version),
        )
        return None if row is None else template_from_row(row)

    async def max_version(self, template_id: str) -> int:
        row = await self._one(
            "SELECT COALESCE(MAX(version), 0) AS top FROM templates "
            "WHERE template_id = ?",
            (template_id,),
        )
        return 0 if row is None else int(row["top"])

    async def insert_template(self, template: Template) -> None:
        await self._insert("templates", template_row(template))

    async def set_status(
        self,
        template_id: str,
        version: int,
        status: str,
        eval: Mapping[str, Any] | None = None,
    ) -> None:
        if eval is None:
            await self.conn.execute(
                "UPDATE templates SET status = ? WHERE template_id = ? AND version = ?",
                (status, template_id, version),
            )
        else:
            await self.conn.execute(
                "UPDATE templates SET status = ?, eval_json = ? "
                "WHERE template_id = ? AND version = ?",
                (status, json.dumps(dict(eval)), template_id, version),
            )

    async def rows(self, table: str) -> list[dict[str, Any]]:
        columns = ", ".join(COLUMNS[table])
        return await self._all(f"SELECT {columns} FROM {table} ORDER BY rowid")

    async def delete(self, routine_id: str | None) -> int:
        total = 0
        for table in TABLES:
            if routine_id is None:
                cursor = await self.conn.execute(f"DELETE FROM {table}")
            else:
                cursor = await self.conn.execute(
                    f"DELETE FROM {table} WHERE routine_id = ?", (routine_id,)
                )
            total += max(0, cursor.rowcount)
        return total


class SqliteProcedures(Procedures):
    """The procedural store in the SQLite file *db_path* (``procedures.db``).

    The connection opens at the first call: a missing file is created with
    mode 0600 unless *create* is false (then ``FileNotFoundError``), the
    PRAGMAs of :mod:`localmelo.melo.memory._sqlite` apply, then
    ``busy_timeout`` :data:`BUSY_TIMEOUT_MS`, and the schema is brought to
    :data:`SCHEMA_VERSION` (``PRAGMA user_version``); a newer file raises
    :class:`ProceduresSchemaError`. Writes run in ``BEGIN IMMEDIATE``
    transactions, so two processes that learn from one template at once
    serialize. A call that is cancelled, such as by a timeout, drops the
    connection without waiting for it (its worker thread may still be
    inside SQLite), and the next call opens a new one.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        params: Params = DEFAULT_PARAMS,
        clock: Callable[[], float] = time.time,
        create: bool = True,
    ) -> None:
        super().__init__(params=params, clock=clock)
        self._db_path = str(db_path)
        self._create = create
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._closing: set[asyncio.Future[Any]] = set()

    @property
    def path(self) -> str:
        return self._db_path

    async def _open(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        if self._db_path != ":memory:":
            if not self._create and not os.path.exists(self._db_path):
                raise FileNotFoundError(f"{self._db_path} does not exist")
            create_private_file(self._db_path)
        conn = await aiosqlite.connect(self._db_path)
        try:
            await prepare_connection(conn, _SCHEMA)
            await conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            await self._migrate(conn)
        except asyncio.CancelledError:
            self._drop(conn)
            raise
        except BaseException:
            with contextlib.suppress(Exception):
                await conn.close()
            raise
        self._conn = conn
        return conn

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        try:
            await conn.execute("BEGIN IMMEDIATE")
            async with conn.execute("PRAGMA user_version") as cursor:
                row = await cursor.fetchone()
            version = int(row[0]) if row else 0
            if version > SCHEMA_VERSION:
                raise ProceduresSchemaError(
                    f"{self._db_path} has schema version {version}; this "
                    f"localmelo reads version {SCHEMA_VERSION}"
                )
            if version < SCHEMA_VERSION:
                await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            await conn.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                await conn.rollback()
            raise

    def _drop(self, conn: aiosqlite.Connection) -> None:
        """Close *conn* in the background; a stuck SQLite call delays only that."""
        if self._conn is conn:
            self._conn = None
        closing = asyncio.ensure_future(conn.close())
        self._closing.add(closing)

        def done(future: asyncio.Future[Any]) -> None:
            self._closing.discard(future)
            if not future.cancelled():
                future.exception()  # retrieved, so never reported

        closing.add_done_callback(done)

    @contextlib.asynccontextmanager
    async def _session(self, write: bool) -> AsyncIterator[_Tx]:
        async with self._lock:
            conn = await self._open()
            try:
                if write:
                    await conn.execute("BEGIN IMMEDIATE")
                yield _SqlTx(conn)
                if write:
                    await conn.commit()
            except asyncio.CancelledError:
                self._drop(conn)
                raise
            except BaseException:
                if write:
                    with contextlib.suppress(Exception):
                        await conn.rollback()
                raise

    async def forget(self, routine_id: str | None = None) -> int:
        """As :meth:`Procedures.forget`, overwriting what it deletes.

        ``secure_delete`` zeroes the deleted rows' content, and a WAL
        checkpoint moves the change into the database file.
        """
        async with self._lock:
            conn = await self._open()
            await conn.execute("PRAGMA secure_delete = ON")
        deleted = await super().forget(routine_id)
        async with self._lock:
            conn = await self._open()
            await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return deleted

    async def aclose(self) -> None:
        """Close the connection, and wait for any closing in the background."""
        async with self._lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                await conn.close()
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)
