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

The status row's segments are kept as words (:func:`status_parts`: a key of the
console's catalog and its parameters, as ``said`` stores them), so the console
can write them in its reader's language. The terminal's own segments
(:func:`status_segments`) are the same words rendered in English.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from localmelo.support.progress_line import run_elapsed
from localmelo.support.sessions.display import format_tokens
from localmelo.support.words import t_in


@dataclass(frozen=True)
class TaskCounts:
    """The scheduler's queue as a session that can reach a task store sees it."""

    running: int
    queued: int
    held: int
    top_level: int | None = None
    # The held tasks that wait for the person (an approval or an answer), not
    # for a free model slot or a higher-priority task.
    waiting: int = 0


def _words(key: str, /, **params: Any) -> dict[str, Any]:
    """One segment of the status row as words: ``{"key", "params"}`` of the console's catalog."""
    return {"key": key, "params": params}


def _english(part: Mapping[str, Any]) -> str:
    """A segment's words in English, as the terminal prints them."""
    return t_in("en", part["key"], **part["params"])


def waits_for_you(count: int) -> str:
    """The sentence for *count* tasks that wait for the person."""
    return _english(_words("status.waits", n=count))


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
    # The session's tokens sent (prompt) and received (completion).
    sent: int = 0
    received: int = 0
    # The model the next requests go to (the session sets it), and per model
    # the prompt, cached and completion tokens, which configured prices turn
    # into a cost.
    model: str = ""
    by_model: dict[str, list[int]] = field(default_factory=dict)

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
        completion = max(_count(usage.get("completion_tokens")) or 0, 0)
        self.requests += 1
        self.sent += prompt
        self.received += completion
        tally = self.by_model.setdefault(self.model, [0, 0, 0])
        tally[0] += prompt
        tally[1] += min(max(cached or 0, 0), prompt)
        tally[2] += completion
        self.last_prompt = prompt
        self.last_cached = None if cached is None else min(max(cached, 0), prompt)
        if isinstance(ts, int | float) and not isinstance(ts, bool):
            self.last_request_ts = float(ts)
        if self.last_cached is not None:
            self.reported_prompt += prompt
            self.reported_cached += self.last_cached
            self.reported_requests += 1

    def to_dict(
        self, *, local: bool, now: float, tasks: TaskCounts | None = None
    ) -> dict[str, Any]:
        """The snapshot as the console's status route serves it.

        ``segments`` is exactly :func:`status_segments`, so a surface that only
        draws them agrees with the terminal. The numbers beside it are for a
        surface that styles them: a share is ``None`` when the server reported
        no cached tokens (drawn ``--``, never 0%), and ``idle_s`` is set for a
        local server only.
        """
        context = self.context or {}
        measured = _count(context.get("measured_prompt_tokens"))
        progress = self.progress or {}
        phase = progress.get("phase")
        idle = None
        if local and self.last_request_ts is not None:
            idle = max(0.0, now - self.last_request_ts)
        return {
            "local": local,
            "segments": status_segments(self, local=local, now=now, tasks=tasks),
            "parts": status_parts(self, local=local, now=now, tasks=tasks),
            "cache": {
                "requests": self.requests,
                "last_percent": _share(self.last_cached, self.last_prompt),
                "session_percent": _share(self.reported_cached, self.reported_prompt),
                "idle_s": idle,
            },
            "context": (
                {"measured": measured, "limit": _count(context.get("input_limit"))}
                if measured
                else None
            ),
            "steps": (
                None
                if self.progress is None
                else {
                    "active": self.run_active,
                    "step": _count(progress.get("step")) or 0,
                    "attempt": _count(progress.get("attempt")) or 1,
                    "phase": phase if isinstance(phase, str) else "",
                }
            ),
            "tasks": None if tasks is None else asdict(tasks),
        }


def _share(part: int | None, whole: int | None) -> int | None:
    if part is None or not whole:
        return None
    return int(_percent(part, whole)[:-1])


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


def context_summary(snap: UsageSnapshot, last_prompt: int = 0) -> str:
    """How full the context is, in words: ``8% used (2.6k of 32k tokens)``.

    Without a measured request it falls back on *last_prompt*, the prompt of the
    last request (what the status bar's ``ctx`` shows), and without that says so.
    """
    measured = _measured(snap)
    limit = _count((snap.context or {}).get("input_limit"))
    if measured and limit:
        used = f"{format_tokens(measured)} of {format_tokens(limit)} tokens"
        return f"{_percent(measured, limit)} used ({used})"
    if measured or last_prompt:
        return f"{format_tokens(measured or last_prompt)} tokens used"
    return "not measured yet"


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


def _idle_text(seconds: float) -> str:
    """``3m`` or ``1h02m``: a minute-grained idle time, cut."""
    minutes = int(seconds) // 60
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def status_parts(
    snap: UsageSnapshot,
    *,
    local: bool,
    now: float,
    tasks: TaskCounts | None = None,
    steps: bool = True,
    words: bool = False,
) -> list[dict[str, Any]]:
    """The status row's segments as words, left to right: cache, ctx, steps, tasks, waiting.

    Each is ``{"key", "params"}`` of the console's catalog (``status.*``); see
    :func:`status_segments` for what they say. A segment with no data is left
    out, and so are the steps with *steps* false.
    """
    parts: list[dict[str, Any]] = []
    if snap.requests:
        idle = ""
        if local and snap.last_request_ts is not None:
            gap = now - snap.last_request_ts
            if gap >= 60:
                idle = _idle_text(gap)
        if snap.last_cached is None or not snap.last_prompt:
            if not words:
                parts.append(
                    _words("status.cacheNoneIdle", idle=idle)
                    if idle
                    else _words("status.cacheNone")
                )
        else:
            share = _percent(snap.last_cached, snap.last_prompt)
            if words:
                key = "status.cachedIdle" if idle else "status.cached"
            else:
                key = "status.cacheIdle" if idle else "status.cache"
            parts.append(_words(key, share=share, idle=idle))
    context = snap.context or {}
    measured = _count(context.get("measured_prompt_tokens"))
    if measured:
        limit = _count(context.get("input_limit"))
        if words:
            parts.append(_context_part(measured, limit))
        elif limit:
            parts.append(
                _words(
                    "status.ctx",
                    used=format_tokens(measured),
                    limit=format_tokens(limit),
                )
            )
        else:
            parts.append(_words("status.ctxOpen", used=format_tokens(measured)))
    if steps and snap.run_active and snap.progress is not None:
        facts: dict[str, Any] = {"step": _count(snap.progress.get("step")) or 0}
        attempt = _count(snap.progress.get("attempt")) or 1
        if attempt >= 2:
            facts["attempt"] = attempt
        phase = snap.progress.get("phase")
        if isinstance(phase, str) and phase:
            facts["phase"] = phase
        key = {
            (False, False): "status.step",
            (True, False): "status.stepAttempt",
            (False, True): "status.stepPhase",
            (True, True): "status.stepAttemptPhase",
        }[("attempt" in facts, "phase" in facts)]
        parts.append(_words(key, **facts))
    if tasks is not None:
        parts.append(
            _words(
                "status.tasksHeld" if tasks.held else "status.tasks",
                running=tasks.running,
                queued=tasks.queued,
                held=tasks.held,
            )
        )
        # The counts are for developers. A task that waits for the person is
        # not: it gets a sentence of its own, which every mode draws.
        if tasks.waiting > 0:
            parts.append(_words("status.waits", n=tasks.waiting))
    return parts


def status_segments(
    snap: UsageSnapshot,
    *,
    local: bool,
    now: float,
    tasks: TaskCounts | None = None,
    steps: bool = True,
    words: bool = False,
) -> list[str]:
    """The status row's segments in English, left to right: cache, ctx, steps, tasks, waiting.

    A segment with no data is left out, and so are the steps with *steps*
    false. Steps are the progress spec's level-0
    facts (never a percent or ``n/max``) and show only while a run is active.
    The idle time is as of the last redraw (the row redraws on events only),
    so it shows from one minute on.

    With *words* (the terminal's default mode) the cache and the context read
    as words, ``83% cached`` and ``context 8% used``, and the cache waits until
    the server has reported a value instead of showing ``cache --``. The
    console's status route keeps the compact segments, and sends them as
    :func:`status_parts` too, for its reader's language.
    """
    return [
        _english(part)
        for part in status_parts(
            snap, local=local, now=now, tasks=tasks, steps=steps, words=words
        )
    ]


def _context_part(measured: int, limit: int | None) -> dict[str, Any]:
    if limit:
        return _words("status.context", share=_percent(measured, limit))
    return _words("status.contextTokens", tokens=format_tokens(measured))


def context_words(measured: int, limit: int | None = None) -> str:
    """The status bar's context in words: ``context 8% used``, or the tokens."""
    return _english(_context_part(measured, limit))


@dataclass(frozen=True)
class BarPart:
    """One piece of the terminal's status bar.

    *role* says how it is drawn: ``""`` muted like the bar, ``figure`` in the
    body's ink, or ``ok``, ``warn``, ``error``, ``live`` and ``key``. A
    *detail* gives way before any figure on a narrow terminal; a *glued*
    piece follows the one before it after a space instead of a dot.
    """

    text: str
    role: str = ""
    detail: bool = False
    glued: bool = False


def session_cost(
    snap: UsageSnapshot, prices: Mapping[str, Mapping[str, float]]
) -> float | None:
    """The session's cost in dollars from ``[pricing.<model>]``; None unpriced.

    Prices are per million tokens: ``input``, ``output`` and ``cache_read``
    (``input`` when absent) for the prompt tokens the server reported cached.
    A model used without a price makes the whole cost unknown.
    """
    total = 0.0
    for model, (prompt, cached, completion) in snap.by_model.items():
        price = prices.get(model)
        if price is None:
            return None
        read = price.get("cache_read", price["input"])
        total += (prompt - cached) * price["input"] + cached * read
        total += completion * price["output"]
    return total / 1e6 if snap.by_model else None


def meter(percent: int, ascii: bool = False) -> str:
    """``▮▮▯▯▯``: *percent* in five cells; any use shows at least one."""
    full = min(5, max(1 if percent > 0 else 0, round(percent / 20)))
    on, off = ("#", "-") if ascii else ("▮", "▯")
    return on * full + off * (5 - full)


def fill_role(percent: int) -> str:
    """How a filling figure is drawn: muted, then ``warn`` at 70%, ``error`` at 90%."""
    return "error" if percent >= 90 else "warn" if percent >= 70 else ""


def _clock(seconds: float) -> str:
    """``45s``, ``3m`` or ``1h02m``: a time left, cut."""
    if seconds < 60:
        return f"{int(seconds)}s"
    return _idle_text(seconds)


def clock_running(
    snap: UsageSnapshot,
    *,
    local: bool,
    now: float,
    limits: tuple[int, float | None] | None = None,
) -> bool:
    """Whether the bar shows a clock: the idle time (from a minute on) or a reset."""
    if local:
        # A second early, so the idle time appears on time.
        return snap.last_request_ts is not None and now - snap.last_request_ts >= 59
    return limits is not None and limits[1] is not None and limits[1] > now


def bar_usage(
    snap: UsageSnapshot,
    *,
    local: bool,
    now: float,
    prices: Mapping[str, Mapping[str, float]] | None = None,
    limits: tuple[int, float | None] | None = None,
    last_prompt: int = 0,
    ascii: bool = False,
) -> list[BarPart]:
    """The status bar's usage zone (#282), from what the session already received.

    A cloud backend says the session's cost from configured prices (its
    tokens when a model has none), the context and the tightest rate limit
    the last response reported, with its reset; a local one says the tokens
    sent and received, the context, the prompt cache's share of the last
    request (the session's average after it) and the idle time since the
    last request. A percentage gets a meter only where it is real.
    """
    parts: list[BarPart] = []
    cost = None if local or not snap.requests else session_cost(snap, prices or {})
    if cost is not None:
        parts.append(
            BarPart("<$0.01" if 0 < cost < 0.005 else f"${cost:,.2f}", "figure")
        )
    elif snap.requests:
        up, down = ("in ", "out ") if ascii else ("↑ ", "↓ ")
        sent, received = format_tokens(snap.sent), format_tokens(snap.received)
        parts.append(BarPart(f"{up}{sent} {down}{received}"))
    measured = _measured(snap) or 0
    limit = _count((snap.context or {}).get("input_limit")) if measured else None
    if measured and limit:
        share = int(_percent(measured, limit)[:-1])
        used = f"ctx {format_tokens(measured)}/{format_tokens(limit)}"
        parts.append(BarPart(used, fill_role(share)))
        parts.append(BarPart(meter(share, ascii), fill_role(share), True, True))
    elif measured or last_prompt:
        parts.append(BarPart(f"ctx {format_tokens(measured or last_prompt)}"))
    if local:
        share_now = _share(snap.last_cached, snap.last_prompt)
        if share_now is not None:
            poor = share_now < 20 and snap.reported_requests >= 2
            parts.append(
                BarPart(
                    f"cache {share_now}%",
                    "ok" if share_now >= 50 else "warn" if poor else "",
                )
            )
            if snap.reported_requests >= 2:
                avg = _share(snap.reported_cached, snap.reported_prompt)
                parts.append(BarPart(f"(avg {avg}%)", detail=True, glued=True))
        if snap.last_request_ts is not None and now - snap.last_request_ts >= 60:
            idle = _idle_text(now - snap.last_request_ts)
            parts.append(BarPart(f"idle {idle}", detail=True))
    elif limits is not None:
        percent, reset = limits
        role = fill_role(percent)
        parts.append(BarPart(f"limit {percent}%", role))
        parts.append(BarPart(meter(percent, ascii), role, True, True))
        if reset is not None and reset > now:
            left = _clock(reset - now)
            parts.append(BarPart(f"reset {left}" if ascii else f"↻{left}", glued=True))
    return parts
