"""The importer: bench proxy ``calls.jsonl`` files and localmelo's ``history.db``.

Replays and the validation window need real calls (design 8.1), so this
module turns the bench proxy's records into traces (spec Requirements
53-55) and reads tool times from a ``history.db`` (Requirements 56-58).
Every bad line or row is dropped and counted, never fatal, except a file
of another schema version, which is rejected whole. History is opened
read-only and only ``duration_ms`` leaves a tool result's JSON.

``ttfb_s`` and ``ttft_s`` are never read (design 8.1; decision D22). The
bench's answer (2026-10-26): for a non-streamed call ``ttfb_s`` is
min(``duration_s``, about 2.0 s), because oMLX 0.7.1rc1 sends the whole
body, or a keepalive space after 2 s, so it carries no time to first
token; only a streamed call's ``ttft_s`` does.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localmelo.support.scheduler.costs import RATES_OMLX_OQ5, Rates, alone_service_s
from localmelo.support.scheduler.model import CallKind
from localmelo.support.scheduler.sim.trace import (
    CHAT_KINDS,
    ServiceSource,
    Status,
    TaskTrace,
    TraceCall,
)

# A call within this many seconds after a failed one is its retry:
# localmelo retries after 1 s and 4 s (design 3.6).
RETRY_WINDOW_S = 5.0

# The proxy record schema this importer reads.
CALLS_SCHEMA_VERSION = 2

# The first bytes of every SQLite database file.
SQLITE_HEADER = b"SQLite format 3\x00"

# The steps columns the history SELECT reads (spec Requirement 57).
STEPS_COLUMNS = (
    "task_id",
    "step_id",
    "seq",
    "timestamp",
    "tool_call_json",
    "tool_result_json",
)

# The only statements that read rows; no text column reaches Python
# (spec Requirement 57).
_STEPS_SQL = """SELECT task_id, step_id, seq, timestamp,
       tool_call_json IS NULL AS thought_only,
       CASE WHEN json_valid(tool_result_json)
            THEN json_extract(tool_result_json, '$.duration_ms') END AS duration_ms,
       (tool_result_json IS NOT NULL AND NOT json_valid(tool_result_json)) AS bad_json
FROM steps ORDER BY task_id, seq, step_id"""
_TASKS_SQL = "SELECT task_id, status FROM tasks ORDER BY task_id"


class CallsFormatError(ValueError):
    """A ``calls.jsonl`` of another schema version, raised under ``strict``."""


class HistoryFormatError(ValueError):
    """A file that is not SQLite, or (under ``strict``) has no usable steps table."""


class HistoryReadError(OSError):
    """A history database that SQLite cannot read read-only; the message says why."""


@dataclass(frozen=True)
class ImportReport:
    """What an import read, kept and dropped (spec Requirement 55).

    For ``import_calls()``, ``lines == records_kept + Σ dropped`` over the
    files not rejected, and the distinct tasks among kept records number
    ``tasks + tasks_without_chat`` plus ``tasks_overlapping`` when those
    are left out.
    """

    files: int
    files_rejected: int
    lines: int
    records_kept: int
    dropped: tuple[tuple[str, int], ...]
    tasks: int
    tasks_overlapping: int = 0
    tasks_without_chat: int = 0
    wall_time_tasks: int = 0
    service_sources: tuple[tuple[str, int], ...] = ()
    has_llm_calls: bool = False
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoryStep:
    """One ``steps`` row: ``tool_s`` 0.0 for a thought, ``None`` when unknown."""

    seq: int
    end_wall_s: float
    tool_s: float | None


@dataclass(frozen=True)
class HistoryTask:
    """One task of a ``history.db``; ``status`` is ``None`` without a tasks row."""

    task_key: str
    status: str | None
    steps: tuple[HistoryStep, ...]


# ── Values ──


def _number(v: Any) -> float | None:
    """*v* as a finite float, else ``None`` (booleans are not numbers)."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def _seconds(v: Any) -> float | None:
    f = _number(v)
    return f if f is not None and f >= 0 else None


def _count(v: Any) -> int | None:
    """A token count: a non-negative int that is not a bool, else ``None``."""
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _obj(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


# ── calls.jsonl ──


class _NonFiniteError(Exception):
    pass


def _reject(name: str) -> object:
    raise _NonFiniteError(name)


def _parse(line: bytes) -> dict[str, Any] | str:
    """The record of *line*, or the reason it is dropped."""
    try:
        value = json.loads(line.decode("utf-8"), parse_constant=_reject)
    except _NonFiniteError:
        return "nan_or_infinity"
    except (ValueError, RecursionError):  # bad UTF-8 is a ValueError too
        return "not_json"
    return value if isinstance(value, dict) else "not_object"


def _kind(r: dict[str, Any]) -> CallKind | None:
    """The call kind of a model call record, ``None`` for any other record."""
    endpoint = r.get("endpoint")
    if endpoint in ("chat_completions", "messages"):
        # localmelo's reflection request has no tools, so the proxy says aux.
        return "reflect" if r.get("role") == "aux" else "plan"
    if (endpoint, r.get("method"), r.get("path")) == (
        "other",
        "POST",
        "/v1/embeddings",
    ):
        return "embed"
    return None


def _version_ok(r: dict[str, Any]) -> bool:
    v = r.get("schema_version")
    return isinstance(v, int) and not isinstance(v, bool) and v == CALLS_SCHEMA_VERSION


def _check(r: dict[str, Any], harnesses: Sequence[str]) -> str | None:
    """The first record-level drop reason, in the lead's order (decision 6)."""
    if not isinstance(r.get("run_id"), str) or not r["run_id"]:
        return "no_run_id"
    if _count(r.get("call_index")) is None:
        return "bad_call_index"
    if _kind(r) is None:
        return "not_a_model_call"
    if r.get("harness") not in harnesses:
        return "harness"
    return None


def _status(r: dict[str, Any]) -> Status:
    if r.get("handled_locally") is True:
        return "refused"
    streamed = r.get("stream") is True
    if (
        r.get("status") == 200
        and r.get("error") is None
        and (not streamed or r.get("stream_complete") is True)
    ):
        return "ok"
    return "failed"  # client_disconnected alone is no failure (proxy.md)


def _times(
    recs: list[dict[str, Any]], drops: Counter[str]
) -> tuple[list[tuple[dict[str, Any], float, float, float]], bool]:
    """Records kept with their start, end and duration on one clock, and whether wall.

    Mono times when every record has both, else wall times when every
    record has both, else mono times with the other records dropped.
    """
    mono = [(_number(r.get("mono_start")), _number(r.get("mono_end"))) for r in recs]
    wall = [(_number(r.get("ts_start")), _number(r.get("ts_end"))) for r in recs]
    has_mono = all(a is not None and b is not None for a, b in mono)
    use_wall = not has_mono and all(a is not None and b is not None for a, b in wall)
    kept = []
    for r, (start, end) in zip(recs, wall if use_wall else mono, strict=True):
        if start is None or end is None:
            drops["bad_times"] += 1
        elif end < start:
            drops["end_before_start"] += 1
        elif r.get("duration_s") is None:
            kept.append((r, start, end, end - start))
        elif (duration := _seconds(r["duration_s"])) is None:
            drops["bad_duration"] += 1
        else:
            kept.append((r, start, end, duration))
    return kept, use_wall


def _trace_calls(
    key: str, kept: list[tuple[dict[str, Any], float, float, float]], rates: Rates
) -> list[TraceCall]:
    """The calls of one task, sorted by ``(start_s, call_index)``."""
    base = min(start for _, start, _, _ in kept)
    rows = sorted(kept, key=lambda k: (k[1], k[0]["call_index"]))
    out: list[TraceCall] = []
    for r, start, end, duration in rows:
        kind, status = _kind(r) or "plan", _status(r)
        usage, raw = _obj(r.get("usage")), _obj(r.get("usage_raw"))
        prompt = _count(usage.get("prompt_tokens"))
        completion = _count(usage.get("completion_tokens"))
        cached = _count(usage.get("cached_tokens"))
        if cached is None:
            cached = _count(_obj(raw.get("prompt_tokens_details")).get("cached_tokens"))
        s_prompt = _seconds(raw.get("prompt_eval_duration"))
        s_gen = _seconds(raw.get("generation_duration"))
        service: float | None
        source: ServiceSource
        residual = None
        if s_prompt is not None and s_gen is not None:
            service, source = s_prompt + s_gen, "server"
            residual = duration - s_prompt - s_gen
        elif prompt is not None and completion is not None:
            service = alone_service_s(prompt, cached, completion, rates)
            source = "tokens"
        else:
            service, source = duration, "wall"
        prev = out[-1] if out else None
        start_s, end_s = start - base, end - base
        attempt = 1
        if (
            prev is not None
            and prev.status == "failed"
            and start_s - prev.end_s <= RETRY_WINDOW_S
        ):
            attempt = prev.attempt + 1
        unknown = prompt is None or completion is None
        out.append(
            TraceCall(
                task_key=key,
                call_index=r["call_index"],
                kind=kind,
                attempt=attempt,
                status=status,
                start_s=start_s,
                end_s=end_s,
                wall_start_s=_number(r.get("ts_start")),
                duration_s=duration,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=cached,
                server_ttft_s=_seconds(raw.get("time_to_first_token")),
                server_prompt_s=s_prompt,
                server_generation_s=s_gen,
                service_s=service,
                service_source=source,
                residual_s=residual,
                gap_before_s=0.0 if prev is None else start_s - prev.end_s,
                tokens_unknown=kind in CHAT_KINDS and status == "ok" and unknown,
            )
        )
    return out


def import_calls(
    root: Path,
    *,
    harnesses: Sequence[str] = ("localmelo",),
    include_overlapping: bool = False,
    strict: bool = False,
    rates: Rates = RATES_OMLX_OQ5,
) -> tuple[tuple[TaskTrace, ...], ImportReport]:
    """The tasks of every ``calls.jsonl`` under *root*, sorted by ``task_key``.

    Files are read in order of their relative POSIX path. A file with a
    record whose ``schema_version`` is not 2 (or missing) is rejected
    whole, or raises ``CallsFormatError`` with *strict*. *rates* serve
    only the ``"tokens"`` service fallback (``RATES_REC_FLINT`` for rec).
    """
    root = Path(root)
    files = sorted(
        (p for p in root.rglob("calls.jsonl") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    drops: Counter[str] = Counter()
    errors: list[str] = []
    lines = rejected = 0
    by_task: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        parsed = [_parse(x) for x in path.read_bytes().split(b"\n") if x.strip()]
        bad = next(
            (r for r in parsed if isinstance(r, dict) and not _version_ok(r)), None
        )
        if bad is not None:
            msg = f"{rel}: schema_version {bad.get('schema_version')}, expected 2"
            if strict:
                raise CallsFormatError(msg)
            rejected += 1
            errors.append(msg)
            continue
        lines += len(parsed)
        for r in parsed:
            reason = r if isinstance(r, str) else _check(r, harnesses)
            if reason is not None:
                drops[reason] += 1
            elif isinstance(r, dict):
                by_task.setdefault(f"{rel}#{r['run_id']}", (rel, []))[1].append(r)

    traces: list[TaskTrace] = []
    kept_n = wall_tasks = overlapping_n = no_chat = 0
    sources: Counter[str] = Counter()
    for key in sorted(by_task):
        source, recs = by_task[key]
        kept, use_wall = _times(recs, drops)
        if not kept:
            continue
        kept_n += len(kept)
        wall_tasks += int(use_wall)
        calls = _trace_calls(key, kept, rates)
        if not any(c.kind in CHAT_KINDS for c in calls):
            no_chat += 1
            continue
        overlapping = any(c.gap_before_s < 0 for c in calls)
        overlapping_n += int(overlapping)
        if overlapping and not include_overlapping:
            continue
        sources.update(c.service_source for c in calls)
        harness = str(kept[0][0].get("harness"))
        traces.append(TaskTrace(key, source, harness, tuple(calls), overlapping))
    report = ImportReport(
        files=len(files),
        files_rejected=rejected,
        lines=lines,
        records_kept=kept_n,
        dropped=tuple(sorted(drops.items())),
        tasks=len(traces),
        tasks_overlapping=overlapping_n,
        tasks_without_chat=no_chat,
        wall_time_tasks=wall_tasks,
        service_sources=tuple(sorted(sources.items())),
        errors=tuple(sorted(errors)),
    )
    return tuple(traces), report


# ── history.db ──


def _connect_read_only(path: Path, immutable: bool) -> sqlite3.Connection:
    """A read-only connection to *path* that never writes the database.

    The twin of ``load_history_calls()`` in
    ``melo/memory/retrieval/corpus.py`` (R3); kept apart because S1
    imports nothing from ``melo/`` (spec Requirement 8), and the two may
    merge in a follow-up. ``immutable=1`` reads the main file alone and
    creates no ``-wal`` or ``-shm``; with a ``-wal`` file, plain
    ``mode=ro`` reads the rows it holds. No pragma is applied:
    ``journal_mode=WAL`` would be a write.
    """
    uri = path.as_uri() + "?mode=ro" + ("&immutable=1" if immutable else "")
    return sqlite3.connect(uri, uri=True)


def _read(
    path: Path, immutable: bool
) -> tuple[set[str], set[str], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Tables, steps columns, step rows and task rows, in one pass on one connection."""
    try:
        conn = _connect_read_only(path, immutable)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                )
            }
            cols: dict[str, set[str]] = {
                name: {row[1] for row in conn.execute(f"PRAGMA table_info({name})")}
                for name in ("steps", "tasks")
                if name in tables
            }
            steps = cols.get("steps", set())
            rows = (
                conn.execute(_STEPS_SQL).fetchall()
                if steps >= set(STEPS_COLUMNS)
                else []
            )
            task_rows = (
                conn.execute(_TASKS_SQL).fetchall()
                if cols.get("tasks", set()) >= {"task_id", "status"}
                else []
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        cause = str(exc)
        if "json_valid" in cause:
            cause = "this SQLite has no JSON functions (json_valid)"
        elif Path(f"{path}-wal").exists() and not os.access(path.parent, os.W_OK):
            cause = "a -wal file needs a -shm file, and the directory is read-only"
        raise HistoryReadError(f"{path}: cannot read the database: {cause}") from None
    return tables, steps, rows, task_rows


def import_history(
    path: Path, *, label: str | None = None, strict: bool = False
) -> tuple[tuple[HistoryTask, ...], ImportReport]:
    """The tasks of the ``history.db`` at *path* with their tool times, read-only.

    ``task_key`` is ``f"{label or file name}#{task_id}"``. Raises
    ``HistoryFormatError`` for a file that is not SQLite, and for a
    missing ``steps`` table or column only with *strict* (else the
    import is empty with an error entry); ``HistoryReadError`` when SQLite
    cannot read the file read-only. A ``duration_ms`` stored as a JSON
    boolean reaches Python as 1 or 0 through ``json_extract``, so it reads
    as a time; a boolean that reaches Python as one reads as unknown.
    A ``timestamp`` that is not a finite number counts as NULL, and a
    ``seq`` that is not an integer reads as 0.
    """
    database = Path(path).resolve()
    try:
        with database.open("rb") as f:
            header = f.read(len(SQLITE_HEADER))
    except OSError as exc:
        raise HistoryReadError(f"{database}: cannot read: {exc}") from None
    if header != SQLITE_HEADER:
        raise HistoryFormatError(f"{database}: not a SQLite database")
    immutable = not any(
        Path(f"{database}{suffix}").exists() for suffix in ("-wal", "-journal")
    )
    tables, steps, rows, task_rows = _read(database, immutable)
    if immutable and Path(f"{database}-wal").exists():  # a writer appeared
        tables, steps, rows, task_rows = _read(database, False)

    errors: list[str] = []
    missing = [c for c in STEPS_COLUMNS if c not in steps]
    if missing:
        msg = f"{database}: no usable steps table (missing: {', '.join(missing)})"
        if strict:
            raise HistoryFormatError(msg)
        errors.append(msg)
    status = {tid: s if isinstance(s, str) else None for tid, s in task_rows}
    name = label or database.name
    drops: Counter[str] = Counter()
    by_task: dict[Any, list[HistoryStep]] = {}
    for task_id, _step_id, seq, ts, thought_only, duration_ms, bad_json in rows:
        steps_of = by_task.setdefault(task_id, [])
        end = _number(ts)
        if bad_json:
            drops["bad_json"] += 1
            continue
        if end is None:
            drops["null_timestamp"] += 1
            continue
        tool: float | None = 0.0
        if not thought_only:
            ms = _seconds(duration_ms)  # from time.perf_counter() around the tool
            tool = None if ms is None else ms / 1000
        steps_of.append(HistoryStep(seq if isinstance(seq, int) else 0, end, tool))
    tasks = tuple(
        HistoryTask(f"{name}#{tid}", status.get(tid), tuple(by_task[tid]))
        for tid in by_task
    )
    report = ImportReport(
        files=1,
        files_rejected=1 if missing else 0,
        lines=len(rows),
        records_kept=sum(len(t.steps) for t in tasks),
        dropped=tuple(sorted(drops.items())),
        tasks=len(tasks),
        has_llm_calls="llm_calls" in tables,
        errors=tuple(errors),
    )
    return tasks, report


def tool_time_pool(tasks: Sequence[HistoryTask]) -> tuple[float, ...]:
    """The known tool times above 0, sorted: the pool of ``synthetic_templates()``."""
    return tuple(
        sorted(
            s.tool_s
            for t in tasks
            for s in t.steps
            if s.tool_s is not None and s.tool_s > 0
        )
    )
