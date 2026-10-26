"""The trace format, schema 1: recorded model calls per task, and task templates.

A trace is what a replay needs from a run: one ``TraceCall`` per model
call with its kind, status, times, tokens and the gap before it (spec
Requirement 53). S1 owns the format and S2's call log writes it (decision
D38), so one reader serves the importer, the simulator and the validation
window. Lines are canonical JSON, so a trace written twice is the same
file, and ``read_trace()`` of a written trace gives back the same value.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

from localmelo.support.scheduler.model import CallKind, SessionKind

# The schema name and version every trace line carries.
SCHEMA = "localmelo.sched.trace"
VERSION = 1

# The prompt of a template call whose count is unknown and has no previous
# call to copy: the default prompt_prior_tokens (design 3.7, mean prompt).
PROMPT_PRIOR_TOKENS = 3235

# Call kinds that are chat calls (a planning or reflection request).
CHAT_KINDS = ("plan", "reflect")

Status = Literal["ok", "failed", "refused"]
ServiceSource = Literal["server", "tokens", "wall", "unknown"]

# Allowed values of the trace fields that are enumerations.
_CHOICES: dict[str, tuple[str, ...]] = {
    "kind": get_args(CallKind),
    "status": get_args(Status),
    "service_source": get_args(ServiceSource),
    "session_kind": get_args(SessionKind),
}


class TraceFormatError(ValueError):
    """A trace line that breaks schema 1; the message names the file and line."""


@dataclass(frozen=True)
class TraceCall:
    """One recorded model call (spec Requirement 53).

    Times are seconds from the task's first start; ``None`` is unknown,
    never 0. ``gap_before_s`` is the start minus the previous call's end
    (0 for the first, negative when calls overlap). The last five fields
    are filled only by S2's call log.
    """

    task_key: str
    call_index: int
    kind: CallKind
    attempt: int
    status: Status
    start_s: float
    end_s: float
    wall_start_s: float | None
    duration_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    server_ttft_s: float | None
    server_prompt_s: float | None
    server_generation_s: float | None
    service_s: float | None
    service_source: ServiceSource
    residual_s: float | None
    gap_before_s: float
    tokens_unknown: bool
    session_kind: SessionKind = "task"
    level: int | None = None
    submit_s: float | None = None
    admit_s: float | None = None
    inflight_at_submit: int | None = None


@dataclass(frozen=True)
class TaskTrace:
    """The calls of one task, sorted by ``(start_s, call_index)``.

    ``source`` is the relative path of the file the calls came from.
    """

    task_key: str
    source: str
    harness: str
    calls: tuple[TraceCall, ...]
    overlapping: bool


@dataclass(frozen=True)
class TemplateCall:
    """One call of a template: what the simulator submits and charges."""

    kind: CallKind
    attempt: int
    prompt_tokens: int
    completion_tokens: int
    warm_cached_tokens: int | None
    gap_before_s: float
    status: Literal["ok", "failed"]


@dataclass(frozen=True)
class TaskTemplate:
    """A task's calls in order, for replay."""

    name: str
    calls: tuple[TemplateCall, ...]


# ── Reading and writing ──


def _dumps(obj: dict[str, Any]) -> str:
    return json.dumps(
        obj, sort_keys=True, allow_nan=False, separators=(",", ":"), ensure_ascii=False
    )


def write_trace(path: Path, traces: Sequence[TaskTrace]) -> None:
    """Write *traces* to *path*: one canonical line per call, by ``(task_key, call_index)``."""
    rows: list[tuple[str, int, str]] = []
    for t in traces:
        extra = {"source": t.source, "harness": t.harness, "overlapping": t.overlapping}
        for c in t.calls:
            line = {**asdict(c), **extra, "schema": SCHEMA, "version": VERSION}
            rows.append((c.task_key, c.call_index, _dumps(line)))
    rows.sort(key=lambda r: (r[0], r[1]))
    Path(path).write_text("".join(line + "\n" for _, _, line in rows), encoding="utf-8")


def _non_finite(name: str) -> object:
    raise ValueError(f"{name} is not a number")


def _field(name: str, type_: str, value: Any, where: str) -> Any:
    """*value* checked against the field's declared type; ints read as floats."""
    if value is None and type_.endswith("| None"):
        return None
    base = type_.split(" |")[0]
    ok = False
    if name in _CHOICES:
        ok = value in _CHOICES[name]
    elif base == "float":
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        value = float(value) if ok else value
        ok = ok and math.isfinite(value)
    elif base == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif base == "bool":
        ok = isinstance(value, bool)
    else:
        ok = isinstance(value, str)
    if not ok:
        raise TraceFormatError(f"{where}: {name} {value!r} is not a {base}")
    return value


def read_trace(path: Path) -> tuple[TaskTrace, ...]:
    """The tasks of the trace file *path*, sorted by ``task_key``.

    Raises ``TraceFormatError`` naming the line for a line that is not a
    JSON object of schema 1, a field of the wrong type, ``source``,
    ``harness`` or ``overlapping`` that differ within one task, or a
    repeated ``(task_key, call_index)``.
    """
    tasks: dict[str, tuple[tuple[str, str, bool], list[TraceCall]]] = {}
    seen: set[tuple[str, int]] = set()
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    for n, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        where = f"{path}:{n}"
        try:
            obj = json.loads(text, parse_constant=_non_finite)
        except ValueError as exc:
            raise TraceFormatError(f"{where}: not JSON: {exc}") from None
        if not isinstance(obj, dict):
            raise TraceFormatError(f"{where}: not a JSON object")
        if obj.get("schema") != SCHEMA or obj.get("version") != VERSION:
            raise TraceFormatError(f"{where}: not {SCHEMA} version {VERSION}")
        values: dict[str, Any] = {}
        for f in fields(TraceCall):
            if f.name in obj:
                values[f.name] = _field(f.name, str(f.type), obj[f.name], where)
        try:
            call = TraceCall(**values)
        except TypeError as exc:
            raise TraceFormatError(f"{where}: {exc}") from None
        task = (
            _field("source", "str", obj.get("source"), where),
            _field("harness", "str", obj.get("harness"), where),
            _field("overlapping", "bool", obj.get("overlapping"), where),
        )
        if (call.task_key, call.call_index) in seen:
            raise TraceFormatError(f"{where}: repeated call {call.call_index}")
        seen.add((call.task_key, call.call_index))
        entry = tasks.setdefault(call.task_key, (task, []))
        if entry[0] != task:
            raise TraceFormatError(f"{where}: source, harness or overlapping differ")
        entry[1].append(call)
    out: list[TaskTrace] = []
    for key in sorted(tasks):
        (source, harness, overlapping), calls = tasks[key]
        calls.sort(key=lambda c: (c.start_s, c.call_index))
        out.append(TaskTrace(key, source, harness, tuple(calls), overlapping))
    return tuple(out)


# ── Templates ──


def templates_from_traces(traces: Sequence[TaskTrace]) -> tuple[TaskTemplate, ...]:
    """A template per usable trace (spec Requirement 53).

    Overlapping traces and traces with an ``ok`` chat call whose tokens
    are unknown are skipped. Refused calls are dropped (they used no
    server time), and their gap and duration join the next call's gap.
    Negative gaps floor at 0. An unknown chat prompt takes the previous
    chat call's, else ``PROMPT_PRIOR_TOKENS``; any other unknown count is 0.
    """
    out: list[TaskTemplate] = []
    for t in traces:
        if t.overlapping or any(
            c.status == "ok"
            and c.kind in CHAT_KINDS
            and (c.prompt_tokens is None or c.completion_tokens is None)
            for c in t.calls
        ):
            continue
        calls: list[TemplateCall] = []
        carry, prev_prompt = 0.0, PROMPT_PRIOR_TOKENS
        for c in t.calls:
            gap = max(0.0, c.gap_before_s)
            if c.status == "refused":
                carry += gap + c.duration_s
                continue
            prompt = c.prompt_tokens
            if c.kind in CHAT_KINDS:
                prompt = prev_prompt if prompt is None else prompt
                prev_prompt = prompt
            calls.append(
                TemplateCall(
                    c.kind,
                    c.attempt,
                    prompt or 0,
                    c.completion_tokens or 0,
                    c.cached_tokens,
                    carry + gap,
                    "ok" if c.status == "ok" else "failed",
                )
            )
            carry = 0.0
        if calls:
            out.append(TaskTemplate(t.task_key, tuple(calls)))
    return tuple(out)
