"""The stage a task is at, decided once for the terminal, the gateway and the page.

:func:`task_stage` reads a task's plain facts (the task JSON the gateway sends,
or a dict built from its record) and returns a :class:`Stage`: who is awaited,
what is needed, how it ended. It is one table of ordered rows, the first that
matches wins, so which of two true facts a person sees is read in one place.
A stage names its words by catalog key (:attr:`Stage.label`); the wording
stays in the catalogs. A combination no row matches (an unknown state too)
reads as the running stage: never an empty one, and an open approval or
question always wins over it.

The reasons a task waits after a restart live here too: each stored sentence
(kept as it is for scripts and old rows) has a key the page words in the
person's language (``tasks.why.<key>``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

# Why a task waits for the person after a restart, in plain words: what
# happened and what to do. Stored as they are; the page words them by key.
FENCED_REASON: Final = (
    "localmelo could not read this task's record when it started: "
    "check its work, then resume."
)
# A run a stop cut off, written by the stop itself: the next start decides it.
STOPPED_REASON: Final = "localmelo stopped; this task continues when it opens again"
# Something of the run may still be going, or its record is not complete: a
# process of the run may still be running, the record is unknown, or more than
# one call had not returned.
IN_FLIGHT_REASON: Final = (
    "localmelo stopped before it could tell whether this task's last step "
    "finished: check its work, then resume."
)
# A call that can change something had not returned, and its effect is not one
# melo can look at afterwards (a command, or one that reaches outside), or the
# row does not say.
CALL_REASON: Final = (
    "A step melo cannot check by itself, such as a command or one that reaches "
    "outside this computer, was running when localmelo stopped: check whether it "
    "finished, then resume."
)
# The person asked for a pause that had not settled when localmelo stopped.
PAUSED_REASON: Final = (
    "You paused this task before localmelo stopped: "
    "resume it when you want it to go on."
)
# The run was cut off by a crash the time before too, since it last ended done or failed.
CRASHED_REASON: Final = (
    "localmelo closed unexpectedly twice while this task was working: "
    "check its work, then resume."
)
# The task's folder could not be used at start (a disk not mounted yet).
WORKSPACE_REASON: Final = (
    "This task's folder was missing when localmelo started: "
    "resume it once the folder is back."
)
# A stored reason's key: what the person reads, ``tasks.why.<key>``. One key
# per thing to do; the record and the crash say "check its work" in words of
# their own, so they keep theirs.
WHY: Final[Mapping[str, str]] = {
    STOPPED_REASON: "restart",
    PAUSED_REASON: "paused",
    IN_FLIGHT_REASON: "unsure",
    CALL_REASON: "call",
    CRASHED_REASON: "crashed",
    WORKSPACE_REASON: "folder",
    FENCED_REASON: "record",
}

FINISHED: Final = frozenset({"done", "failed", "cancelled", "expired"})
ENDINGS: Final = frozenset({"ready", "draft", "blocked", "failed", "dropped", "moot"})
# The scheduler's reasons to hold a model call (``activity.held.<code>``), by
# what the task waits for: a slot, the model another job uses, or a time.
HOLDS: Final[Mapping[str, str]] = {
    "Priority": "slot",
    "Resources": "slot",
    "Memory": "slot",
    "Reservation": "model",
    "Battery": "time",
    "Thermal": "time",
    "IdleOnly": "time",
}
# A hold that waits for the person: an approval, or an answer.
PERSON_HOLDS: Final = frozenset({"approval", "input"})


@dataclass(frozen=True)
class Stage:
    """Where a task is: its stage, who it waits for, what it needs, how it ended.

    ``key`` is one of ``running``, ``stopping``, ``approval``, ``answer``,
    ``asked`` (its chat's answer), ``model``, ``slot``, ``queued``,
    ``paused``, ``resume`` (the person's Resume after a restart),
    ``went_on`` (it ended and its work goes on in a newer task, #571) and
    ``finished``. ``awaited`` is ``nobody``, ``person``, ``model``, ``slot`` or
    ``time``; ``need`` is ``""``, ``approval``, ``answer``, ``resume``,
    ``continue``, ``review`` or, for a slot, the gate's code. ``ending`` is how
    a finished task ended (``ready``, ``draft``, ``blocked``, ``failed``,
    ``dropped``, ``moot``, or ``""`` for a done one with no ending); ``why`` is the key of the reason it waits.
    """

    key: str
    awaited: str = "nobody"
    need: str = ""
    ending: str = ""
    why: str = ""

    @property
    def label(self) -> str:
        """The catalog key of this stage's words."""
        if self.key == "finished":
            return f"ending.state.{self.ending}" if self.ending else "panel.state.done"
        if self.key == "slot":
            return f"activity.held.{self.need}"
        return LABELS[self.key]

    def as_json(self) -> dict[str, str]:
        return {**asdict(self), "label": self.label}


LABELS: Final[Mapping[str, str]] = {
    "running": "panel.state.running",
    "stopping": "panel.head.stopping",
    "approval": "panel.state.waiting",
    "answer": "ask.waitingOnYou",
    "asked": "ask.waitingOn",
    "model": "panel.state.held",
    "queued": "panel.state.queued",
    "paused": "panel.state.paused",
    "resume": "panel.head.waiting",
    "went_on": "panel.state.wentOn",
}


def _finished(task: Mapping[str, Any]) -> Stage:
    if task.get("out_of_steps") is True:
        return Stage("finished", "person", "continue", "dropped")
    ending = task.get("ending")
    name = ending.get("state") if isinstance(ending, Mapping) else None
    if name == "dropped" and ending.get("proposed") == "dropped":  # type: ignore[union-attr]
        name = "moot"  # a worker dropped its task: not the person's Stop
    if name not in ENDINGS:
        # No ending recorded (an older row): the state says it, a done one as done.
        state = task.get("state")
        name = {"failed": "failed", "done": ""}.get(str(state), "dropped")
    if name == "blocked":
        return Stage("finished", "person", "review", name)
    return Stage("finished", ending=name)


def _asked(task: Mapping[str, Any]) -> Stage | None:
    ask = task.get("ask")
    if not isinstance(ask, Mapping):
        return None
    if ask.get("on") == "person":
        return Stage("answer", "person", "answer")
    return Stage("asked", "model", "answer")


def _held(task: Mapping[str, Any]) -> Stage:
    reason = str(task.get("wait_reason") or "")
    if reason in PERSON_HOLDS:
        return Stage("approval", "person", "approval")
    if reason in HOLDS:
        return Stage("slot", HOLDS[reason], reason)
    return Stage("model", "model")


def _queued(task: Mapping[str, Any]) -> Stage:
    reason = str(task.get("wait_reason") or "")
    return (
        Stage("slot", HOLDS[reason], reason)
        if reason in HOLDS
        else Stage("queued", "slot")
    )


# The table: (when, stage), in order; the first row whose test holds decides.
Row = tuple[
    Callable[[Mapping[str, Any], bool], bool],
    Callable[[Mapping[str, Any]], Stage | None],
]
ROWS: Final[tuple[Row, ...]] = (
    # Ended, with its work going on in a newer task: that one says how it goes,
    # so this one needs nobody, however it ended.
    (
        lambda task, stopping: task.get("state") in FINISHED
        and bool(task.get("successor")),
        lambda task: Stage("went_on"),
    ),
    # Finished, over everything: how it ended.
    (lambda task, stopping: task.get("state") in FINISHED, _finished),
    # A Stop in progress, over every wait.
    (lambda task, stopping: stopping, lambda task: Stage("stopping")),
    # An open approval or question, over running and every hold.
    (
        lambda task, stopping: bool(task.get("pending_approvals")),
        lambda task: Stage("approval", "person", "approval"),
    ),
    (lambda task, stopping: isinstance(task.get("ask"), Mapping), _asked),
    (lambda task, stopping: task.get("state") == "held", _held),
    (lambda task, stopping: task.get("state") in ("queued", "waiting"), _queued),
    (
        lambda task, stopping: task.get("state") == "parked",
        lambda task: Stage("paused", "person", "resume"),
    ),
    # Cut off by a stop, it continues by itself at the next start: paused, not
    # waiting for the person.
    (
        lambda task, stopping: task.get("state") == "interrupted"
        and task.get("wait_reason") == STOPPED_REASON,
        lambda task: Stage("paused", why="restart"),
    ),
    (
        lambda task, stopping: task.get("state") == "interrupted",
        lambda task: Stage(
            "resume", "person", "resume", why=WHY.get(str(task.get("wait_reason")), "")
        ),
    ),
)


def task_stage(task: Mapping[str, Any], *, stopping: bool = False) -> Stage:
    """The stage of the task whose plain facts are *task*; *stopping*: a Stop runs.

    Reads ``state``, ``successor``, ``out_of_steps``, ``ending``,
    ``pending_approvals``, ``ask`` and ``wait_reason``, as the task JSON has them. Anything else,
    running included, is the plain running stage.
    """
    for when, stage in ROWS:
        if when(task, stopping):
            found = stage(task)
            if found is not None:
                return found
    return Stage("running")


__all__ = [
    "CALL_REASON",
    "CRASHED_REASON",
    "FENCED_REASON",
    "IN_FLIGHT_REASON",
    "PAUSED_REASON",
    "STOPPED_REASON",
    "WHY",
    "WORKSPACE_REASON",
    "Stage",
    "task_stage",
]
