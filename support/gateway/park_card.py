"""The card a parked or cut-off task goes on with, from committed events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from localmelo.melo.agent._checkpoint import CUE_LINE_CHARS
from localmelo.melo.memory.procedures.mask import mask_credentials
from localmelo.melo.schema import GOAL_CHARS
from localmelo.support.gateway.handoff import work_standing
from localmelo.support.gateway.task_endings import paired_calls
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.sessions.store import StoredEvent

# The longest card the stores keep: a stop writes no more than this.
MAX_CARD_CHARS = 2_000
# How every card begins: a turn whose text starts so is melo's card, not words
# the person typed.
CARD_START = "Where stopped: "
HEAD = (
    f"{CARD_START}the task stopped before it finished; where its work stands, "
    "from melo's records:"
)
CAUTIONS = (
    "Cautions: Work may have been interrupted; check for side effects before "
    "repeating any action. Recorded reports are not proof of completion."
)


def build_park_card(
    spec: TaskSpec | Mapping[str, Any], events: Iterable[StoredEvent]
) -> str:
    """Where the task of *spec* stands, from its stored *events*, as its next turn reads it.

    The lines of its :class:`~localmelo.melo.schema.WorkCheckpoint`
    (:func:`~localmelo.support.gateway.handoff.work_standing`: the request's
    words, the files written, every call with a failure or an unknown effect
    named as such, the last work checkpoint's notes and the model's newest
    words after it), under the line every card starts with and its
    cautions, within :data:`MAX_CARD_CHARS`. No model call; only events the
    store kept count, and a credential is masked wherever it shows.
    """
    if isinstance(spec, TaskSpec):
        spec = spec.to_dict()
    if not isinstance(spec, Mapping):
        spec = {}
    kept = [
        e
        for e in events or ()
        if isinstance(e, StoredEvent)
        and e.id is not None
        and isinstance(e.event, dict)
        and isinstance(e.event.get("data"), dict)
    ]
    goal = str(spec.get("user_words") or spec.get("objective") or "")
    # Half of the room the goal leaves for the model's notes, half for the calls.
    room = MAX_CARD_CHARS - len(HEAD) - len(CAUTIONS) - min(len(goal), GOAL_CHARS)
    standing = work_standing(
        paired_calls(kept),
        kept,
        goal=goal,
        folder=str(spec.get("workspace") or "") or None,
        chars=max(room // 2, 0),
        lines=max(room // 2 // CUE_LINE_CHARS, 1),
    )
    card = mask_credentials("\n".join([HEAD, CAUTIONS, *standing.lines()]))
    return card if len(card) <= MAX_CARD_CHARS else card[: MAX_CARD_CHARS - 1] + "…"
