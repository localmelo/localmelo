"""The scheduling policy core: which model calls go to the server, and why not.

localmelo shares one local model server between the user's conversation
(the lane), worker tasks and chats. This package decides, as a pure and
total function of an in-memory state, which pending calls are admitted,
in what order, and the one reason each task waits; ``dispatch()`` keeps
the lane open whatever ``decide()`` does. It reads no clock and does no
I/O, so the trace-driven simulator and the live service run the same
code.

S2 wires the web workspace's chats: ``support/gateway/chat_gate.py`` runs
``dispatch()`` for every plan, reflect and embed call a chat sends, with
``[scheduler] policy`` (default ``off``, which only records). The CLI, the
legacy route and direct mode are still ungated, and only tests import
this package directly. The design is ``.melo/scheduler_design.md``
(sections 2-3) in the workspace, outside the repository.
"""

from __future__ import annotations

from localmelo.support.scheduler.accounting import advance, apply_dispatch
from localmelo.support.scheduler.machine_gates import evaluate_gates
from localmelo.support.scheduler.model import (
    REASONS,
    Call,
    CallEnded,
    CallSubmitted,
    Clock,
    Decision,
    Dep,
    DepsSet,
    Dispatch,
    Machine,
    MachineChanged,
    State,
    Task,
    TaskAdded,
    TaskUpdated,
    Tick,
    Typing,
)
from localmelo.support.scheduler.params import ARMS, Params, params_for_arm
from localmelo.support.scheduler.policy import (
    admit_lane,
    decide,
    decision_json,
    dispatch,
    fallback_fcfs,
)

__all__ = [
    "Task",
    "Call",
    "Dep",
    "Machine",
    "Clock",
    "State",
    "Decision",
    "Dispatch",
    "Params",
    "TaskAdded",
    "TaskUpdated",
    "DepsSet",
    "CallSubmitted",
    "CallEnded",
    "Typing",
    "MachineChanged",
    "Tick",
    "ARMS",
    "REASONS",
    "params_for_arm",
    "decide",
    "admit_lane",
    "dispatch",
    "fallback_fcfs",
    "advance",
    "apply_dispatch",
    "decision_json",
    "evaluate_gates",
]
