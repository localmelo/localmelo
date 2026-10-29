"""One snapshot of what a session's requests, context, steps and tasks stand at.

:class:`UsageSnapshot` is fed the events the session already receives
(``assistant.message`` and ``reflection`` usage, ``context_budget``,
``run.started``, ``run.progress``, ``run.finished``); it makes no request and
polls no server. :func:`usage_lines` prints it for ``/status``: each segment's
value, or in a few words why it has none (#176, decisions 1, 2 and 5).

The cache segment is the share of prompt tokens the server reported as cached.
A server that sends no ``cached_tokens`` shows ``--``, never 0%. A local
server's prefix cache is evicted by memory, not by time, so it shows the idle
time since the last request and no countdown.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from localmelo.support.progress_line import run_elapsed
from localmelo.support.sessions.display import format_tokens


@dataclass(frozen=True)
class TaskCounts:
    """The scheduler's queue as a session that can reach a task store sees it."""

    running: int
    queued: int
    held: int
    top_level: int | None = None


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


@dataclass
class UsageSnapshot:
    """What the events of one session say about cache, context and steps."""

    requests: int = 0
    # The last request's prompt and cached tokens; cached is None when the
    # server did not report it.
    last_prompt: int | None = None
    last_cached: int | None = None
    # Prompt and cached tokens summed over the requests that reported both.
    reported_prompt: int = 0
    reported_cached: int = 0
    reported_requests: int = 0
    last_request_ts: float | None = None
    context: Mapping[str, Any] | None = None
    progress: Mapping[str, Any] | None = None
    run_active: bool = False

    def observe(self, event: Mapping[str, Any]) -> None:
        """Take one stored event (``{"type", "ts", "data", ...}``) into account."""
        kind = event.get("type")
        data = event.get("data")
        if not isinstance(data, Mapping):
            return
        if kind in ("assistant.message", "reflection"):
            self._request(data.get("usage"), event.get("ts"))
        elif kind == "context_budget":
            if _count(data.get("measured_prompt_tokens")) is not None:
                self.context = dict(data)
        elif kind == "run.started":
            self.run_active = True
        elif kind == "run.progress":
            self.progress = dict(data)
        elif kind == "run.finished":
            self.run_active = False

    def _request(self, usage: object, ts: object) -> None:
        if not isinstance(usage, Mapping):
            return
        prompt = _count(usage.get("prompt_tokens"))
        if prompt is None or prompt <= 0:
            return
        cached = _count(usage.get("cached_tokens"))
        self.requests += 1
        self.last_prompt = prompt
        self.last_cached = None if cached is None else min(max(cached, 0), prompt)
        if isinstance(ts, int | float) and not isinstance(ts, bool):
            self.last_request_ts = float(ts)
        if self.last_cached is not None:
            self.reported_prompt += prompt
            self.reported_cached += self.last_cached
            self.reported_requests += 1


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%"


def _cache_text(snap: UsageSnapshot, *, local: bool, now: float) -> str:
    if not snap.requests or snap.last_prompt is None:
        return "no requests yet"
    if snap.last_cached is None:
        parts = ["last request --: the server did not report cached tokens"]
    else:
        parts = [
            f"last request {_percent(snap.last_cached, snap.last_prompt)} cached "
            f"({format_tokens(snap.last_cached)} of "
            f"{format_tokens(snap.last_prompt)} prompt tokens)"
        ]
    if snap.reported_prompt:
        n = snap.reported_requests
        parts.append(
            f"session {_percent(snap.reported_cached, snap.reported_prompt)} "
            f"over {n} request{'' if n == 1 else 's'}"
        )
    if local:
        if snap.last_request_ts is not None:
            idle = run_elapsed(now - snap.last_request_ts)
            parts.append(f"idle {idle}")
    else:
        parts.append("cache lifetime not tracked yet")
    return " · ".join(parts)


def _measured(snap: UsageSnapshot) -> int | None:
    data = snap.context
    return None if data is None else _count(data.get("measured_prompt_tokens"))


def _context_text(snap: UsageSnapshot) -> str:
    data = snap.context
    measured = _measured(snap)
    if data is None or measured is None:
        return "no context budget measured yet"
    limit = _count(data.get("input_limit"))
    if not limit or limit <= 0:
        return f"{format_tokens(measured)} prompt tokens · no input limit known"
    return (
        f"{format_tokens(measured)} of {format_tokens(limit)} input tokens "
        f"({_percent(measured, limit)}) at the last measured request"
    )


def usage_lines(
    snap: UsageSnapshot,
    *,
    local: bool,
    now: float,
    steps: Callable[[Mapping[str, Any]], list[str]],
    tasks: TaskCounts | None = None,
    dot: str = "·",
    hide_empty: bool = False,
) -> list[str]:
    """The ``/status`` rows of *snap*: cache, context, steps, tasks.

    With *hide_empty*, a row with nothing to say yet ("no ... yet") is left out.

    *steps* formats a ``run.progress`` payload as one fact per line (the
    caller's ``/progress`` formatter); *now* is ``time.time()``.
    """
    if snap.progress is None:
        step_rows = [
            "no progress facts yet" if not snap.run_active else "run has no facts yet"
        ]
    else:
        state = "running" if snap.run_active else "last run"
        step_rows = steps(snap.progress)
        step_rows[0] = f"{state} {dot} {step_rows[0]}"
    if tasks is None:
        task_text = "no task store in this session"
    else:
        task_text = f"{tasks.running} running {dot} {tasks.queued} queued {dot} " + (
            f"{tasks.held} held"
        )
        if tasks.top_level is not None:
            task_text += f" {dot} top priority P{tasks.top_level}"
    rows = [
        (
            "cache",
            _cache_text(snap, local=local, now=now),
            not snap.requests or snap.last_prompt is None,
        ),
        ("context", _context_text(snap), _measured(snap) is None),
        ("steps", step_rows[0], snap.progress is None),
        ("tasks", task_text, tasks is None),
    ]
    lines: list[str] = []
    for name, text, empty in rows:
        if hide_empty and empty:
            continue
        lines.append(f"{name:<11}{text}")
        if name == "steps":
            lines += ["           " + row for row in step_rows[1:]]
    return lines
