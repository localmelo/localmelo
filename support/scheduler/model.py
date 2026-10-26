"""Types of the scheduling policy: tasks, calls, the machine and decisions.

Every record is a frozen dataclass and every enumeration a ``Literal``, so
a ``State`` is a plain value that ``decide()`` reads and never changes.
Instants and durations are integer microseconds (``*_us``): waits and
service then add up exactly, and replays compare bit for bit (design 3.3).
``clean_state()`` maps odd values (NaN, inf, negatives, oversized
integers, out-of-range levels) to NULL or a default with an alarm, which
keeps ``decide()`` total (design 3.3, "decide() is guarded and total").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

# ── Constants ──

# Levels P0-P4; P5 is Later, a state that is never scheduled (design 2.1).
P0, P1, P2, P3, P4, P5 = 0, 1, 2, 3, 4, 5

# Display names of the levels, indexed by level.
LEVEL_NAMES = ("P0", "P1", "P2", "P3", "P4", "P5")

# Floor of the estimated charge χ̂ of a class: 1 s keeps `credit ≥ χ̂`
# false at zero credit without letting a full bucket pay for more than
# about two admissions (decision D8).
CHI_HAT_FLOOR_US = 1_000_000

# Largest integer that converts to a float exactly; larger values are odd.
MAX_INT = 2**53

# Largest plausible token count; larger values are odd.
MAX_TOKENS = 10**9

# Length of State.recent_admits, the cache proxy's window of admissions.
RECENT_ADMITS_MAX = 16

Origin = Literal["user", "user_rule", "front", "harness", "agent", "channel", "sleep"]
LevelSource = Literal["user", "user_rule", "computed"]
Phase = Literal[
    "queued",
    "running",
    "held",
    "waiting",
    "parked",
    "interrupted",
    "done",
    "failed",
    "cancelled",
    "expired",
]
SessionKind = Literal["front", "front_bg", "task", "chat"]
CallKind = Literal["plan", "reflect", "retrieve", "estimate", "embed"]
Outcome = Literal["ok", "failed", "timeout", "aborted"]
Arm = Literal["fifo", "class", "las2"]
Reason = Literal[
    "Priority",
    "Resources",
    "Reservation",
    "Battery",
    "Thermal",
    "Memory",
    "Dependency",
    "Input",
    "Approval",
    "Budget",
    "IdleOnly",
    "Paused",
]

# The twelve reason codes of design 3.3, step 9, in the design's order.
REASONS: tuple[Reason, ...] = get_args(Reason)

# Phases after which a task is never scheduled again.
TERMINAL_PHASES = frozenset({"done", "failed", "cancelled", "expired"})

# Phases in which admitting a call makes a worker resident ("starting").
STARTING_PHASES = frozenset({"queued", "parked"})


def us(seconds: float) -> int:
    """Microseconds of a parameter given in seconds, rounded to the nearest."""
    return round(seconds * 1_000_000)


# ── Records ──


@dataclass(frozen=True)
class Task:
    """A worker task or chat as the scheduler sees it (spec Requirement 13).

    Fields marked (A) belong to ``advance()`` and ``apply_dispatch()``; the
    caller changes the others through ``TaskUpdated``. ``None`` instants
    and estimates mean unknown, never zero.
    """

    task_id: str
    root_id: str = ""
    kind: Literal["task", "chat", "front_bg"] = "task"
    origin: Origin = "user"
    level: int = 2
    level_source: LevelSource = "computed"
    phase: Phase = "queued"
    wait_reason: Literal["", "input", "approval", "dependency", "lease"] = ""
    paused: bool = False
    user_rank: float | None = None
    deadline_wall_us: int | None = None
    blocked_now: bool = False
    size: Literal["", "small", "medium", "large"] = ""
    remaining_s_est: float | None = None
    tools_left_s_est: float | None = None
    budget_exhausted: bool = False
    created_mono_us: int = 0
    created_wall_us: int = 0
    mention_mono_us: int | None = None
    started_mono_us: int | None = None  # (A)
    last_activity_wall_us: int | None = None  # (A)
    service_us: int = 0  # (A)
    promo_service_us: int = 0  # (A)
    wait_promo_us: int = 0  # (A)
    wait_total_us: int = 0  # (A)
    wait_held_us: int = 0  # (A)
    calls_done: int = 0  # (A)
    completion_tokens_done: int = 0  # (A)
    last_prompt_tokens: int | None = None  # (A)
    last_call_end_mono_us: int | None = None  # (A)
    aged: bool = False  # (A)
    level_eff: int | None = None  # (A)
    last_reason: str = ""  # (A)
    last_detail: str = ""  # (A)


# Task fields that only advance() and apply_dispatch() may change.
_ACCOUNTING = (
    "started_mono_us last_activity_wall_us service_us promo_service_us wait_promo_us"
    " wait_total_us wait_held_us calls_done completion_tokens_done last_prompt_tokens"
    " last_call_end_mono_us aged level_eff last_reason last_detail"
)
ACCOUNTING_FIELDS = frozenset(_ACCOUNTING.split())


@dataclass(frozen=True)
class Call:
    """A model call at the gate or in flight (spec Requirement 14).

    ``session_kind == "front"`` marks a lane call. ``admitted_mono_us`` is
    ``None`` while the call is pending.
    """

    call_id: str
    task_id: str | None = None
    session_kind: SessionKind = "task"
    kind: CallKind = "plan"
    attempt: int = 1
    submit_mono_us: int = 0
    prompt_tokens: int | None = None
    completion_p99_tokens: int | None = None
    admitted_mono_us: int | None = None  # (A)
    floor_class: int | None = None  # (A)
    level_at_admit: int | None = None  # (A)


@dataclass(frozen=True)
class Dep:
    """An edge: *waiter* waits on *holder* (spec Requirement 15)."""

    waiter: str
    holder: str
    kind: Literal["start_after", "result", "lease"] = "start_after"
    confirmed: bool = False


@dataclass(frozen=True)
class Machine:
    """A snapshot of the machine, as the caller's monitors debounced it."""

    lock: Literal["none", "self", "foreign"] = "none"
    lock_mem_gib: float = 0.0
    on_ac: bool = True
    battery_pct: float | None = None
    low_power: bool = False
    thermal: Literal["nominal", "fair", "serious", "critical"] = "nominal"
    memory_pressure: Literal["normal", "warning", "critical"] = "normal"
    server_up: bool = True
    model: Literal["loaded", "draining", "unloaded", "loading"] = "loaded"
    away_since_mono_us: int | None = None


@dataclass(frozen=True)
class Clock:
    """An instant on both clocks: waits use mono, deadlines and expiry wall."""

    mono_us: int
    wall_us: int


@dataclass(frozen=True)
class State:
    """Everything ``decide()`` reads (spec Requirement 18).

    Pairs hold P2 at index 0 and P3 at index 1.
    """

    clock: Clock = field(default_factory=lambda: Clock(0, 0))
    tasks: tuple[Task, ...] = ()
    deps: tuple[Dep, ...] = ()
    calls: tuple[Call, ...] = ()
    machine: Machine = field(default_factory=Machine)
    battery_band: int = 4
    last_typing_mono_us: int | None = None
    last_front_end_mono_us: int | None = None
    credit_us: tuple[int, int] = (0, 0)
    floor_open: tuple[bool, bool] = (False, False)
    chi_hat_us: tuple[int, int] = (0, 0)
    busy_above_p4_mono_us: int | None = None
    recent_admits: tuple[str, ...] = ()
    clock_anomalies: int = 0


# ── Events ──


@dataclass(frozen=True)
class TaskAdded:
    """A new task; a duplicate ``task_id`` is a caller bug."""

    task: Task


@dataclass(frozen=True)
class TaskUpdated:
    """Changes to the caller's fields of one task, as (name, value) pairs."""

    task_id: str
    changes: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class DepsSet:
    """Replaces every dependency edge."""

    deps: tuple[Dep, ...]


@dataclass(frozen=True)
class CallSubmitted:
    """A new pending call."""

    call: Call


@dataclass(frozen=True)
class CallEnded:
    """A call ended or was withdrawn; an unknown ``call_id`` is ignored."""

    call_id: str
    outcome: Outcome
    charge_us: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class Typing:
    """A text-free typing heartbeat."""


@dataclass(frozen=True)
class MachineChanged:
    """A new machine snapshot."""

    machine: Machine


@dataclass(frozen=True)
class Tick:
    """The periodic tick; it only advances the clock."""


Event = (
    TaskAdded
    | TaskUpdated
    | DepsSet
    | CallSubmitted
    | CallEnded
    | Typing
    | MachineChanged
    | Tick
)


# ── Outputs ──


@dataclass(frozen=True)
class Decision:
    """What ``decide()`` returns (spec Requirement 20); no field holds a float."""

    admit: tuple[str, ...] = ()
    admit_unheld: tuple[str, ...] = ()
    floor: tuple[tuple[str, int], ...] = ()
    reasons: tuple[tuple[str, str, str], ...] = ()
    levels: tuple[tuple[str, int], ...] = ()
    aged: tuple[str, ...] = ()
    yields: tuple[str, ...] = ()
    parks: tuple[tuple[str, str], ...] = ()
    expire: tuple[str, ...] = ()
    asks: tuple[tuple[str, str], ...] = ()
    floor_open: tuple[bool, bool] = (False, False)
    class_runnable: tuple[bool, bool] = (False, False)
    chi_hat_us: tuple[int, int] = (CHI_HAT_FLOOR_US, CHI_HAT_FLOOR_US)
    battery_band: int = 4
    idle_window: bool = False
    focus: bool = False
    limit_nonlane: int = 1
    alarms: tuple[str, ...] = ()
    fallback: bool = False


@dataclass(frozen=True)
class Dispatch:
    """The lane's admission or code answer, plus the decision for task calls."""

    lane_admit: tuple[str, ...]
    lane_answer: tuple[tuple[str, str], ...]
    decision: Decision


# ── Lookup helpers ──


def is_terminal(phase: str) -> bool:
    """Whether *phase* ends the task (done, failed, cancelled, expired)."""
    return phase in TERMINAL_PHASES


def is_lane(c: Call) -> bool:
    """Whether *c* is a lane call (the front's session)."""
    return c.session_kind == "front"


def is_planning(c: Call) -> bool:
    """Whether *c* is a non-lane plan or reflect call, a safe point."""
    return c.session_kind != "front" and c.kind in ("plan", "reflect")


def is_holdable(c: Call) -> bool:
    """Whether *c* is a planning call at its first attempt."""
    return is_planning(c) and c.attempt <= 1


def is_retry(c: Call) -> bool:
    """Whether *c* is a planning call past its first attempt."""
    return is_planning(c) and c.attempt > 1


def root_of(t: Task, by_id: dict[str, Task]) -> Task:
    """The task charged for *t*'s calls: its ``root_id``, else *t* itself."""
    if t.root_id and t.root_id in by_id:
        return by_id[t.root_id]
    return t


# ── Odd values (spec Requirement 24) ──

# Field classes per record, separated by "|": non-optional ints, optional
# ints, non-optional token counts, optional token counts, optional floats,
# non-optional floats.
_TASK_FIELDS = (
    "created_mono_us created_wall_us service_us promo_service_us wait_promo_us"
    " wait_total_us wait_held_us calls_done | deadline_wall_us mention_mono_us"
    " started_mono_us last_activity_wall_us last_call_end_mono_us level_eff"
    " | completion_tokens_done | last_prompt_tokens"
    " | user_rank remaining_s_est tools_left_s_est |"
)
_CALL_FIELDS = (
    "attempt submit_mono_us | admitted_mono_us floor_class level_at_admit"
    " | | prompt_tokens completion_p99_tokens | |"
)
_MACHINE_FIELDS = "| away_since_mono_us | | | battery_pct | lock_mem_gib"
_STATE_FIELDS = (
    "| last_typing_mono_us last_front_end_mono_us busy_above_p4_mono_us | | | |"
)


# Optional instants whose presence carries meaning: an admitted call with an
# oversized admission time is still in flight, so it counts as admitted at 0
# rather than NULL, which would make it pending and admit it twice.
_PRESENCE = frozenset({"admitted_mono_us"})


def _finite(v: float) -> bool:
    try:
        return math.isfinite(v)
    except OverflowError:  # an int too large for a float
        return False


def _classes(classes: str) -> tuple[tuple[str, ...], ...]:
    """The six field classes of *classes*, parsed once per string."""
    if classes not in _PARSED:
        _PARSED[classes] = tuple(tuple(part.split()) for part in classes.split("|"))
    return _PARSED[classes]


# The parsed field classes, by their string: every decision cleans every
# record, so parsing once keeps the simulator's steps cheap.
_PARSED: dict[str, tuple[tuple[str, ...], ...]] = {}


def _clean(rec: Any, owner: str, classes: str, out: list[str]) -> Any:
    """*rec* with its odd values replaced, appending one alarm per field."""
    ints, opt_ints, tokens, opt_tokens, opt_floats, floats = _classes(classes)
    changes: dict[str, Any] = {}
    for names, cap, null in (
        (ints, MAX_INT, 0),
        (opt_ints, MAX_INT, None),
        (tokens, MAX_TOKENS, 0),
        (opt_tokens, MAX_TOKENS, None),
    ):
        for name in names:
            v = getattr(rec, name)
            if v is not None and v > cap:
                changes[name] = 0 if name in _PRESENCE else null
            elif v is not None and v < 0:
                changes[name] = 0
    for name in opt_floats + floats:
        v = getattr(rec, name)
        if v is not None and not _finite(v):
            changes[name] = None if name in opt_floats else 0.0
    if isinstance(rec, Task) and not 0 <= rec.level <= 5:
        changes["level"] = 2  # never clamp: a corrupt -1 must not become P0
    if not changes:
        return rec
    out.extend(f"bad_value:{owner}:{name}" for name in changes)
    return replace(rec, **changes)


def _clean_pair(pair: tuple[int, int], name: str, out: list[str]) -> tuple[int, int]:
    if all(0 <= v <= MAX_INT for v in pair):
        return pair
    out.append(f"bad_value:state:{name}")
    return (
        pair[0] if 0 <= pair[0] <= MAX_INT else 0,
        pair[1] if 0 <= pair[1] <= MAX_INT else 0,
    )


def clean_state(state: State) -> tuple[State, tuple[str, ...]]:
    """*state* with odd values made safe, sorted and deduplicated, plus alarms.

    Duplicate task or call ids keep their first record (alarm
    ``duplicate_id``). A battery band outside 0-4 counts as 4, its default.
    """
    alarms: list[str] = []
    tasks: dict[str, Task] = {}
    for t in state.tasks:
        if t.task_id in tasks:
            alarms.append(f"duplicate_id:{t.task_id}")
        else:
            tasks[t.task_id] = _clean(t, t.task_id, _TASK_FIELDS, alarms)
    calls: dict[str, Call] = {}
    for c in state.calls:
        if c.call_id in calls:
            alarms.append(f"duplicate_id:{c.call_id}")
        else:
            calls[c.call_id] = _clean(c, c.call_id, _CALL_FIELDS, alarms)
    band = state.battery_band
    if not 0 <= band <= 4:
        alarms.append("bad_value:state:battery_band")
        band = 4
    cleaned = replace(
        _clean(state, "state", _STATE_FIELDS, alarms),
        tasks=tuple(tasks[k] for k in sorted(tasks)),
        calls=tuple(calls[k] for k in sorted(calls)),
        deps=tuple(
            sorted(state.deps, key=lambda d: (d.waiter, d.holder, d.kind, d.confirmed))
        ),
        machine=_clean(state.machine, "machine", _MACHINE_FIELDS, alarms),
        battery_band=band,
        credit_us=_clean_pair(state.credit_us, "credit_us", alarms),
        chi_hat_us=_clean_pair(state.chi_hat_us, "chi_hat_us", alarms),
    )
    return cleaned, tuple(alarms)
