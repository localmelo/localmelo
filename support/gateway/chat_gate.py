"""The live gate on the web workspace's model calls (S2).

Every chat and embedding request a workspace chat makes is submitted to one
in-process :class:`ChatGate`, which either sends it at once or holds it until
the scheduling policy in ``localmelo.support.scheduler`` admits it. The gate
writes one schema-1 trace line per call, at the call's end.

``policy="off"`` (the default) is observation only: no decision, no future
and no hold, just the pass-through and the line (R8). ``fcfs`` and ``class``
are the two enforcing arms; they use S1's ``fifo`` and ``class`` presets with
the live overrides of the S2 spec (section 4.2).

The gate lives here, not in ``support/scheduler/``, because S1's determinism
tests forbid a clock read and a non-stdlib import under that package
(decision D1). It imports no ``fastapi`` or ``uvicorn``, so it is testable
without the gateway extra, and it never imports ``playground`` (R1).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from localmelo.melo.contracts.providers import (
    LLM_REQUEST_ATTEMPT,
    LLM_REQUEST_KIND,
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.memory._sqlite import thinking_retention_days
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.scheduler import (
    Call,
    CallEnded,
    CallSubmitted,
    Clock,
    Decision,
    Dispatch,
    Machine,
    Params,
    State,
    Task,
    TaskAdded,
    TaskUpdated,
    Typing,
    advance,
    apply_dispatch,
    decide,
    dispatch,
    params_for_arm,
)
from localmelo.support.scheduler.costs import Rates, charge_s
from localmelo.support.scheduler.model import (
    CallKind,
    Event,
    LevelSource,
    Outcome,
    SessionKind,
    is_terminal,
)
from localmelo.support.scheduler.sim.trace import SCHEMA, VERSION, dumps

logger = logging.getLogger("localmelo.gateway")

# The live policies. ``fcfs`` is S1's ``fifo`` under the design's name.
Policy = Literal["off", "fcfs", "class"]

# The call kinds a workspace chat can send; anything else is read as a plan
# (design 1.1, line 133), so a provider built outside Chat is still held.
_KINDS: tuple[str, ...] = ("plan", "reflect", "retrieve", "estimate", "embed")

# A workspace chat call is a planning or reflection request.
_CHAT_KINDS: tuple[str, ...] = ("plan", "reflect")

# The trace's ``service_source`` for each vocabulary ``charge_s()`` returns
# (R20). S1 will align its own strings later (decision D9).
_SOURCE: dict[str, str] = {
    "server": "server",
    "tokens": "tokens",
    "wall_share": "wall",
    "none": "unknown",
}


def _trace_source(source: str) -> str:
    """Map ``charge_s()``'s source to the trace's, never to an unknown string."""
    return _SOURCE.get(source, "unknown")


# The default directory of the per-run call logs.
DEFAULT_CALL_LOG_DIR = Path("~/.cache/localmelo/scheduler").expanduser()

# A run directory's name: UTC yyyymmddThhmmss.
_RUN_DIR = re.compile(r"^\d{8}T\d{6}$")

# Trace constants (the log is S1's format, written by S2; decision D38).
_TRACE_SOURCE = "gateway"
_TRACE_HARNESS = "localmelo"

# The largest plausible token count; larger values are treated as unknown,
# as S1's clean_state does, so a bad usage cannot overflow charge_s.
_MAX_TOKENS = 10**9

# Alarms already logged, once per process (R10).
_WARNED_ALARMS: set[str] = set()


def _default_clock() -> Clock:
    """The live clock: monotonic seconds and wall seconds, both in µs."""
    return Clock(
        mono_us=round(time.monotonic() * 1_000_000),
        wall_us=round(time.time() * 1_000_000),
    )


def _int_usage(usage: object) -> dict[str, int] | None:
    """*usage* as a dict of ints, else ``None`` (R18's copy rule)."""
    if not isinstance(usage, dict):
        return None
    if not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in usage.items()
    ):
        return None
    return dict(usage)


def _usage_tokens(
    usage: dict[str, int] | None,
) -> tuple[int | None, int | None, int | None]:
    """``(prompt, completion, cached)`` of a usage dict, ``None`` when absent."""
    if not isinstance(usage, dict):
        return None, None, None
    out: list[int | None] = []
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        value = usage.get(key)
        out.append(
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= _MAX_TOKENS
            else None
        )
    return out[0], out[1], out[2]


# ── The call log ──


class CallLog:
    """One JSONL file per gateway run: ``<dir>/runs/<UTC>/calls.jsonl`` (R19).

    The directory is 0700 and the file 0600; each line is one call in S1's
    schema-1 trace format, written when the call ends. A directory that
    cannot be created, or a write that fails, logs one warning and disables
    the file: the chat never fails because of the log.
    """

    def __init__(
        self,
        directory: Path | str | None,
        *,
        retention_days: float,
    ) -> None:
        self._fd: int | None = None
        self._path: Path | None = None
        self._warned = False
        if directory is None:
            return
        try:
            runs = Path(directory).expanduser() / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(runs, 0o700)
            self._prune(runs, retention_days)
            name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            run = runs / name
            run.mkdir(mode=0o700, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(run, 0o700)
            path = run / "calls.jsonl"
            self._fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)
            self._path = path
        except OSError as exc:
            logger.warning("Call log disabled: cannot create %s: %s", directory, exc)
            self._fd = None

    @property
    def path(self) -> Path | None:
        """The file this log writes, or ``None`` when it is disabled."""
        return self._path

    def _prune(self, runs: Path, retention_days: float) -> None:
        """Remove run directories older than *retention_days*, and nothing else."""
        cutoff = time.time() - retention_days * 86400.0
        try:
            children = list(runs.iterdir())
        except OSError:
            return
        for child in children:
            if not _RUN_DIR.match(child.name):
                continue
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue

    def write(self, values: dict[str, Any]) -> None:
        """Append one canonical schema-1 line; a failure warns once and disables."""
        if self._fd is None:
            return
        try:
            line = dumps({**values, "schema": SCHEMA, "version": VERSION})
            os.write(self._fd, line.encode("utf-8") + b"\n")
        except (OSError, TypeError, ValueError) as exc:
            if not self._warned:
                logger.warning("Call log write failed, disabling it: %s", exc)
                self._warned = True
            self._close_fd()
            self._path = None

    def _close_fd(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    def close(self) -> None:
        """Close the file; a second close is a no-op."""
        self._close_fd()


# ── One call, as the gate tracks it ──


@dataclass
class CallMeta:
    """What the log and the hold clock need about one call.

    ``submit_mono_us`` and ``admitted_mono_us`` are the gate's clock;
    ``first_submit_mono_us`` is the session's first submit, the origin of
    the trace's relative times.
    """

    session_id: str
    call_id: str
    call_index: int
    kind: str
    attempt: int
    level: int
    session_kind: str
    submit_mono_us: int
    submit_wall_s: float
    first_submit_mono_us: int
    inflight_at_submit: int
    prev_end_mono_us: int | None
    admitted_mono_us: int | None = None


@dataclass
class GateCall:
    """One call at the gate: its metadata and the future admission resolves."""

    meta: CallMeta
    future: asyncio.Future[None] | None = None
    done: bool = False


@dataclass(frozen=True)
class SessionClient:
    """A chat's handle on the gate: one session id and the gate it belongs to."""

    gate: ChatGate
    session_id: str


# ── The gate ──


class ChatGate:
    """One in-process gate for a playground's chats (R1).

    With ``policy="off"`` every method is a recorded pass-through; with
    ``fcfs`` or ``class`` it keeps S1's ``State`` and runs one synchronous
    transition (``advance``, ``dispatch``, ``apply_dispatch``) per submit,
    end, focus and level change. It creates no asyncio task and no timer.
    """

    def __init__(
        self,
        *,
        policy: str = "off",
        n_srv: int = 8,
        r_f: int = 1,
        n_fg: int = 1,
        call_log_dir: Path | str | None = None,
        clock: Callable[[], Clock] | None = None,
        retention_days: float | None = None,
        decide_fn: Callable[[State, Clock, Params], Decision] | None = None,
    ) -> None:
        if policy not in ("off", "fcfs", "class"):
            raise ValueError(
                f"policy {policy!r} is not one of ('off', 'fcfs', 'class')"
            )
        self.policy = policy
        self.n_srv = n_srv
        self.r_f = r_f
        self.n_fg = n_fg
        self._params: Params | None = None
        if policy != "off":
            self._params = params_for_arm(
                "fifo" if policy == "fcfs" else "class",
                n_srv=n_srv,
                r_f=r_f,
                n_fg=n_fg,
                aging=False,
                inheritance=False,
                idle_gate=False,
                battery_gate_enforce=False,
                p0_meaning="conversation",
            )
        days = (
            thinking_retention_days()
            if retention_days is None and call_log_dir is not None
            else (retention_days or 0.0)
        )
        self._log = CallLog(call_log_dir, retention_days=days)
        self._clock = clock if clock is not None else _default_clock
        self._decide_fn = decide_fn if decide_fn is not None else decide
        self._rates = (
            Rates(self._params.prefill_tok_s, self._params.decode_tok_s)
            if self._params is not None
            else Rates()
        )

        self._state = State(machine=Machine())
        self._calls: dict[str, GateCall] = {}
        self._levels: dict[str, tuple[int, str]] = {}
        self._counts: dict[str, int] = {}
        self._first_submit: dict[str, int] = {}
        self._last_end: dict[str, int] = {}
        self._turn_start: dict[str, int] = {}
        self._held_total_us: dict[str, int] = {}
        self._hold_start: dict[str, int] = {}
        self._held_reason: dict[str, str] = {}
        self._reason_by_task: dict[str, str] = {}
        self._focus: str | None = None
        self._closed = False

        # Set on every hold start and hold end; the run's wait loop clears
        # it in the same synchronous segment in which it reads the hold.
        self._wait_event = asyncio.Event()
        # Every distinct alarm this gate saw (tests read it).
        self.alarms: list[str] = []

    # ── Small accessors ──

    @property
    def wait_event(self) -> asyncio.Event:
        """Set whenever a hold starts or ends."""
        return self._wait_event

    def now(self) -> int:
        """The clock's monotonic instant, in microseconds."""
        return self._clock().mono_us

    @property
    def call_log_path(self) -> Path | None:
        """This run's call log, or ``None`` when the file is disabled."""
        return self._log.path

    def client(self, session_id: str) -> SessionClient:
        """A handle that tags this chat's calls with its session id."""
        return SessionClient(self, session_id)

    def level(self, session_id: str) -> int:
        """The chat's stored level, P1-P4 (default P2)."""
        return self._levels.get(session_id, (2, "computed"))[0]

    def level_source(self, session_id: str) -> str:
        """``"user"`` once the chip set the level, else ``"computed"``."""
        return self._levels.get(session_id, (2, "computed"))[1]

    def held(self, session_id: str) -> str:
        """The reason code of the chat's pending call, or ``""``."""
        return self._held_reason.get(session_id, "")

    def held_since(self, session_id: str) -> int | None:
        """When the chat's current hold began, in µs, or ``None`` if not held."""
        return self._hold_start.get(session_id)

    def held_seconds(self, session_id: str) -> float:
        """Completed holds of this turn, in seconds."""
        return self._held_total_us.get(session_id, 0) / 1_000_000

    def _task(self, session_id: str) -> Task | None:
        return next((t for t in self._state.tasks if t.task_id == session_id), None)

    def _inflight(self) -> int:
        return sum(
            1
            for c in self._calls.values()
            if not c.done and c.meta.admitted_mono_us is not None
        )

    # ── Submission and release ──

    def submit(
        self,
        session_id: str,
        *,
        kind: str = "plan",
        attempt: int = 1,
        prompt_tokens: int | None = None,
    ) -> GateCall:
        """Record a call and, under an enforcing arm, submit it to the policy.

        Returns at once: the caller awaits :meth:`acquire`, which resolves
        when the policy admits the call. Under ``off`` there is no future.
        """
        now = self._clock()
        call_kind = cast(CallKind, kind if kind in _KINDS else "plan")
        index = self._counts.get(session_id, 0)
        self._counts[session_id] = index + 1
        first = self._first_submit.setdefault(session_id, now.mono_us)
        level = self.level(session_id)
        source = self.level_source(session_id)
        planning = call_kind in _CHAT_KINDS
        # Only a planning call can be a lane call: an embedding or a
        # retrieval is never held (R14), and S1's lane path would hold it.
        session_kind = "front" if planning and session_id == self._focus else "chat"
        meta = CallMeta(
            session_id=session_id,
            call_id=f"{session_id}:{index}",
            call_index=index,
            kind=call_kind,
            attempt=max(1, attempt) if planning else 1,
            level=level,
            session_kind=session_kind,
            submit_mono_us=now.mono_us,
            submit_wall_s=now.wall_us / 1_000_000,
            first_submit_mono_us=first,
            inflight_at_submit=self._inflight(),
            prev_end_mono_us=self._last_end.get(session_id),
        )
        call = GateCall(meta)
        self._calls[meta.call_id] = call
        if self.policy == "off" or self._closed:
            # Observation only (or already shut down): the call is never
            # pending, so admit_s is submit_s and nothing can hang on it.
            meta.admitted_mono_us = now.mono_us
            return call

        call.future = asyncio.get_running_loop().create_future()
        events: list[Event] = []
        if planning:
            # Only a chat call opens (or reopens) the chat's task; an
            # embedding or a retrieval is taskless in S1 (section 4.2).
            current = self._task(session_id)
            if current is None:
                events.append(
                    TaskAdded(
                        Task(
                            task_id=session_id,
                            kind="chat",
                            origin="user",
                            level=level,
                            level_source=cast(LevelSource, source),
                            phase="running",
                            created_mono_us=now.mono_us,
                            created_wall_us=now.wall_us,
                        )
                    )
                )
            elif is_terminal(current.phase):
                events.append(TaskUpdated(session_id, (("phase", "running"),)))
        events.append(
            CallSubmitted(
                Call(
                    call_id=meta.call_id,
                    # R11: a lane call is taskless; only a background plan or
                    # reflect call belongs to the chat's task.
                    task_id=(
                        session_id if planning and session_kind != "front" else None
                    ),
                    session_kind=cast(SessionKind, session_kind),
                    kind=call_kind,
                    attempt=meta.attempt,
                    submit_mono_us=now.mono_us,
                    prompt_tokens=prompt_tokens,
                )
            )
        )
        self._transition(events)
        return call

    async def acquire(self, call: GateCall) -> None:
        """Wait until the policy admits *call*; under ``off`` it is a no-op."""
        if call.future is None:
            return
        await call.future

    def finish(
        self,
        call: GateCall,
        outcome: str = "ok",
        usage: dict[str, int] | None = None,
    ) -> None:
        """Remove *call*, book its end and write its one log line.

        *outcome* is ``"ok"``, ``"failed"`` (it was sent and then failed or
        was cancelled) or ``"refused"`` (it never left the gate). Never
        raises into the chat.
        """
        if call.done:
            return
        call.done = True
        meta = call.meta
        self._calls.pop(meta.call_id, None)
        now = self._clock()
        prompt, completion, cached = _usage_tokens(usage)
        try:
            charge, source = self._charge(
                meta, outcome, prompt, completion, cached, now
            )
        except Exception:  # an odd usage value must not leak the call's slot
            logger.warning("Call charge failed; logging it as unknown", exc_info=True)
            charge, source = 0.0, "unknown"
        if self.policy != "off":
            events: list[Event] = [
                CallEnded(
                    call_id=meta.call_id,
                    outcome=cast(
                        Outcome,
                        (
                            "ok"
                            if outcome == "ok"
                            else ("aborted" if outcome == "refused" else "failed")
                        ),
                    ),
                    charge_us=round(max(0.0, charge) * 1_000_000),
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                )
            ]
            if self._task(meta.session_id) is not None and not any(
                c.meta.session_id == meta.session_id for c in self._calls.values()
            ):
                # The chat's last call ends: mark it done in the same
                # transition, so an idle chat is never a reason of its own.
                events.append(TaskUpdated(meta.session_id, (("phase", "done"),)))
            self._transition(events)
        self._last_end[meta.session_id] = now.mono_us
        try:
            self._write_line(
                meta, outcome, now, charge, source, prompt, completion, cached
            )
        except Exception:  # the log may never fail a chat (R19)
            logger.warning("Call log line could not be written", exc_info=True)

    # ── Events ──

    def wake(self) -> None:
        """Run one transition with no event, so the clock's effect is decided."""
        self._transition(())

    def focus(self, session_id: str) -> None:
        """Make *session_id* the lane's chat and send its typing heartbeat."""
        self._focus = session_id
        if self.policy == "off" or self._closed:
            return
        self._transition((Typing(),))

    def presence(self, session_id: str) -> None:
        """A presence heartbeat: the chat is focused and being typed in."""
        self.focus(session_id)

    def set_level(self, session_id: str, level: int) -> int:
        """Set the chat's level, P1-P4; re-dispatch if its task exists."""
        if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 4:
            raise ValueError("level must be an integer from 1 to 4")
        self._levels[session_id] = (level, "user")
        if (
            self.policy != "off"
            and not self._closed
            and self._task(session_id) is not None
        ):
            self._transition(
                (
                    TaskUpdated(
                        session_id,
                        (("level", level), ("level_source", "user")),
                    ),
                )
            )
        return level

    def turn_start(self, session_id: str) -> int:
        """Reset the turn's hold clock; return its start instant in µs."""
        if self.policy == "off":
            self._turn_start[session_id] = 0
            self._held_total_us.pop(session_id, None)
            self._hold_start.pop(session_id, None)
            self._held_reason.pop(session_id, None)
            return 0
        start = self.now()
        self._turn_start[session_id] = start
        self._held_total_us[session_id] = 0
        self._hold_start.pop(session_id, None)
        self._held_reason.pop(session_id, None)
        return start

    def forget(self, session_id: str) -> None:
        """Drop a closed chat's bookkeeping once it has no live call.

        The policy's ``State`` has no event that removes a task, and a
        finished task would otherwise be cleaned and scanned on every later
        transition of a long-running gateway. Closing is only allowed while
        the chat is idle, so its task is terminal and its calls are gone.
        """
        for mapping in (
            self._levels,
            self._counts,
            self._first_submit,
            self._last_end,
            self._turn_start,
            self._held_total_us,
            self._hold_start,
            self._held_reason,
            self._reason_by_task,
        ):
            mapping.pop(session_id, None)
        if self._closed or self.policy == "off":
            return
        if any(c.meta.session_id == session_id for c in self._calls.values()):
            return
        self._state = replace(
            self._state,
            tasks=tuple(t for t in self._state.tasks if t.task_id != session_id),
            calls=tuple(c for c in self._state.calls if c.task_id != session_id),
            recent_admits=tuple(
                x for x in self._state.recent_admits if x != session_id
            ),
        )

    async def close(self) -> None:
        """Cancel pending futures and close the log; a second close is a no-op."""
        if self._closed:
            return
        self._closed = True
        now = self._clock()
        for call in list(self._calls.values()):
            if not call.done and call.meta.admitted_mono_us is None:
                # A call still waiting at shutdown never left the gate; it
                # still gets its one line (R19), as a refused call.
                try:
                    self._write_line(
                        call.meta, "refused", now, 0.0, "unknown", None, None, None
                    )
                except Exception:
                    logger.warning("Call log line could not be written", exc_info=True)
            call.done = True
            if call.future is not None and not call.future.done():
                call.future.cancel()
        self._calls.clear()
        self._hold_start.clear()
        self._held_reason.clear()
        self._log.close()
        self._wait_event.set()

    # ── The one transition ──

    def _transition(self, events: Sequence[Event]) -> None:
        """Advance, decide, apply and release, with no await between (R10)."""
        if self.policy == "off" or self._closed or self._params is None:
            return
        now = self._clock()
        try:
            state = advance(self._state, now, events, self._params)
        except Exception as exc:
            self._failed(now, exc)
            return
        self._state = state
        try:
            result = dispatch(state, now, self._params, decide_fn=self._decide_fn)
        except Exception as exc:  # dispatch already falls back; this is a bug
            self._failed(now, exc)
            return
        try:
            self._state = apply_dispatch(state, now, result, self._params)
        except Exception as exc:
            self._failed(now, exc)
            return
        try:
            self._apply(result, now)
        except Exception as exc:  # releasing a future or booking a hold failed
            self._failed(now, exc)

    def _failed(self, now: Clock, exc: BaseException) -> None:
        """A transition raised: alarm, admit everything pending, keep going (R10)."""
        self._alarm(f"sched_transition_failed:{type(exc).__name__}")
        for call in self._calls.values():
            if call.done:
                continue
            if call.meta.admitted_mono_us is None:
                call.meta.admitted_mono_us = now.mono_us
            if call.future is not None and not call.future.done():
                call.future.set_result(None)
        self._update_holds(None, now)

    def _apply(self, result: Dispatch, now: Clock) -> None:
        """Resolve admitted futures, record alarms and book the holds."""
        decided = result.decision
        for call_id in (*result.lane_admit, *decided.admit, *decided.admit_unheld):
            call = self._calls.get(call_id)
            if call is None or call.done:
                continue
            if call.meta.admitted_mono_us is None:
                call.meta.admitted_mono_us = now.mono_us
            if call.future is not None and not call.future.done():
                call.future.set_result(None)
        for alarm in decided.alarms:
            self._alarm(alarm)
        for call_id, reason in result.lane_answer:
            # S2 has no code answer for a lane call: alarm, then admit it.
            self._alarm(f"lane_answer:{reason}")
            call = self._calls.get(call_id)
            if call is None or call.done:
                continue
            if call.meta.admitted_mono_us is None:
                call.meta.admitted_mono_us = now.mono_us
            if call.future is not None and not call.future.done():
                call.future.set_result(None)
        self._reason_by_task = {tid: reason for tid, reason, _ in decided.reasons}
        self._update_holds(decided, now)

    def _held_for(
        self, session_id: str, waiting: list[CallMeta], decided: Decision | None
    ) -> str:
        """The reason code of the chat's pending call (R16)."""
        if any(meta.session_kind == "front" for meta in waiting):
            return "Resources" if self._inflight() >= self.n_srv else "Priority"
        if decided is not None:
            return self._reason_by_task.get(session_id, "Priority")
        return "Priority"

    def _update_holds(self, decided: Decision | None, now: Clock) -> None:
        """Book hold starts and ends; set ``wait_event`` on each change."""
        by_session: dict[str, list[CallMeta]] = {}
        for call in self._calls.values():
            if not call.done:
                by_session.setdefault(call.meta.session_id, []).append(call.meta)
        for session_id in set(by_session) | set(self._hold_start):
            waiting = [
                meta
                for meta in by_session.get(session_id, [])
                if meta.admitted_mono_us is None
            ]
            if waiting:
                if session_id not in self._hold_start:
                    self._hold_start[session_id] = now.mono_us
                    self._wait_event.set()
                self._held_reason[session_id] = self._held_for(
                    session_id, waiting, decided
                )
            elif session_id in self._hold_start:
                self._held_total_us[session_id] = self._held_total_us.get(
                    session_id, 0
                ) + max(0, now.mono_us - self._hold_start.pop(session_id))
                self._held_reason.pop(session_id, None)
                self._wait_event.set()

    def _alarm(self, alarm: str) -> None:
        """Record *alarm* once per gate; log its first occurrence per process."""
        if alarm in self.alarms:
            return
        self.alarms.append(alarm)
        if alarm not in _WARNED_ALARMS:
            _WARNED_ALARMS.add(alarm)
            logger.warning("scheduler alarm: %s", alarm)

    # ── The line ──

    def _charge(
        self,
        meta: CallMeta,
        outcome: str,
        prompt: int | None,
        completion: int | None,
        cached: int | None,
        now: Clock,
    ) -> tuple[float, str]:
        """The call's charge in seconds and its trace ``service_source`` (R20)."""
        admitted = meta.admitted_mono_us
        duration = (
            max(0.0, (now.mono_us - admitted) / 1_000_000)
            if admitted is not None
            else None
        )
        wall = None if outcome == "refused" else duration
        charge, source = charge_s(
            "ok" if outcome == "ok" else outcome,
            prompt,
            cached,
            completion,
            None,
            None,
            wall,
            1.0,
            self._rates,
        )
        return charge, _trace_source(source)

    def _write_line(
        self,
        meta: CallMeta,
        outcome: str,
        now: Clock,
        charge: float,
        source: str,
        prompt: int | None,
        completion: int | None,
        cached: int | None,
    ) -> None:
        """Write one schema-1 line for the finished call (section 4.7)."""
        # A refused call never left the gate, whatever a caller did to it.
        admitted = None if outcome == "refused" else meta.admitted_mono_us
        start = admitted if admitted is not None else meta.submit_mono_us
        first = meta.first_submit_mono_us
        previous = meta.prev_end_mono_us
        self._log.write(
            {
                "task_key": meta.session_id,
                "call_index": meta.call_index,
                "kind": meta.kind,
                "attempt": meta.attempt,
                "status": outcome,
                "start_s": (start - first) / 1_000_000,
                "end_s": (now.mono_us - first) / 1_000_000,
                "wall_start_s": float(meta.submit_wall_s),
                "duration_s": (
                    0.0 if admitted is None else (now.mono_us - admitted) / 1_000_000
                ),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_tokens": cached,
                "server_ttft_s": None,
                "server_prompt_s": None,
                "server_generation_s": None,
                "service_s": float(max(0.0, charge)),
                "service_source": source,
                "residual_s": None,
                "gap_before_s": (
                    0.0 if previous is None else (start - previous) / 1_000_000
                ),
                "tokens_unknown": prompt is None or completion is None,
                "session_kind": meta.session_kind,
                "level": meta.level,
                "submit_s": (meta.submit_mono_us - first) / 1_000_000,
                "admit_s": None if admitted is None else (admitted - first) / 1_000_000,
                "inflight_at_submit": meta.inflight_at_submit,
                "source": _TRACE_SOURCE,
                "harness": _TRACE_HARNESS,
                "overlapping": False,
            }
        )


# ── The wrappers ──


class GatedLLM(BaseLLMProvider):
    """A chat provider that submits each request to *client*'s gate (R18).

    It forwards attributes other than the three the base class documents
    through ``__getattr__``; ``model``, ``last_reasoning_key`` and
    ``last_usage`` are copied from the inner provider right after each
    successful ``chat()``, with no await in between.
    """

    def __init__(self, inner: BaseLLMProvider, client: SessionClient) -> None:
        self.inner = inner
        self.client = client
        self.model = inner.model if isinstance(inner.model, str) else ""
        self.last_reasoning_key = (
            inner.last_reasoning_key
            if isinstance(inner.last_reasoning_key, str)
            else ""
        )
        self.last_usage = _int_usage(inner.last_usage)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        gate = self.client.gate
        raw = LLM_REQUEST_KIND.get()
        kind = raw if raw in _KINDS else "plan"
        call = gate.submit(
            self.client.session_id,
            kind=kind,
            attempt=LLM_REQUEST_ATTEMPT.get(),
        )
        try:
            await gate.acquire(call)
        except BaseException:
            gate.finish(call, "refused")
            raise
        try:
            message = await self.inner.chat(messages, tools=tools)
        except BaseException:
            gate.finish(call, "failed")
            raise
        self._copy()
        gate.finish(call, "ok", self.last_usage)
        return message

    def _copy(self) -> None:
        """Copy the three documented attributes from the inner provider (R18)."""
        self.model = self.inner.model if isinstance(self.inner.model, str) else ""
        self.last_reasoning_key = (
            self.inner.last_reasoning_key
            if isinstance(self.inner.last_reasoning_key, str)
            else ""
        )
        self.last_usage = _int_usage(self.inner.last_usage)

    async def close(self) -> None:
        await self.inner.close()


class GatedEmbedding(BaseEmbeddingProvider):
    """An embedding provider that submits each call to the gate.

    Embeddings carry no task id, so the policy admits them through
    ``Decision.admit_unheld``: they are never held, but they count in
    flight for the lane and the limit (R14).
    """

    def __init__(self, inner: BaseEmbeddingProvider, client: SessionClient) -> None:
        self.inner = inner
        self.client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        gate = self.client.gate
        call = gate.submit(self.client.session_id, kind="embed")
        try:
            await gate.acquire(call)
        except BaseException:
            gate.finish(call, "refused")
            raise
        try:
            vectors = await self.inner.embed(texts)
        except BaseException:
            gate.finish(call, "failed")
            raise
        gate.finish(call, "ok")
        return vectors

    async def close(self) -> None:
        await self.inner.close()
