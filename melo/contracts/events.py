"""Contract for the events a run reports to an optional sink.

``Agent.run_task(query, on_event=sink)`` calls *sink* once per
:class:`AgentEvent`, synchronously and in order. :data:`EVENT_FIELDS` names
the ``data`` keys of every event type; within :data:`EVENT_SCHEMA_VERSION`
types and keys are only ever added, so a consumer ignores the ones it does
not know.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from localmelo.melo.schema import ImagePart

EVENT_SCHEMA_VERSION: Final = 1

# The data keys of each event type.
EVENT_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "run.started": frozenset({"query", "max_steps", "workspace_root", "approval"}),
        "inbox.accepted": frozenset({"offset", "text", "step", "attempt"}),
        "step.started": frozenset({"step", "attempt"}),
        "assistant.message": frozenset(
            {
                "step",
                "content",
                "thinking",
                "tool_calls",
                "usage",
                "prefix",
                "duration_s",
            }
        ),
        "tool.started": frozenset({"step", "call_id", "name", "arguments", "preview"}),
        "tool.finished": frozenset(
            {
                "step",
                "call_id",
                "name",
                "status",
                "error_category",
                "output",
                "output_chars",
                "error",
                "duration_s",
                "images",
                "approval",
                "exit_code",
            }
        ),
        "note": frozenset(
            {
                "step",
                "attempt",
                "note_id",
                "kind",
                "source",
                "text",
                "chars",
                "drafts",
                "remember",
                "stage",
            }
        ),
        "memory": frozenset(
            {"candidate_id", "action", "kind", "admitted_by", "entry_id"}
        ),
        "recall": frozenset(
            {
                "step",
                "attempt",
                "trigger",
                "retriever",
                "query",
                "hits",
                "chars",
                "ms",
                "fallback",
                "placement",
            }
        ),
        "notice": frozenset({"kind", "message", "step", "attempt", "detail"}),
        "reflection.started": frozenset({"attempt", "failure_type"}),
        "reflection": frozenset(
            {
                "attempt",
                "action",
                "continue_run",
                "flags",
                "gate",
                "utility",
                "stop_reason",
                "rationale",
                "summary",
                "best_effort_result",
                "requests",
                "readable",
                "usage",
                "prefix",
                "duration_s",
            }
        ),
        "run.progress": frozenset(
            {
                "level",
                "state",
                "step",
                "attempt",
                "limits",
                "phase",
                "cycle",
                "tests",
                "items",
                "fraction",
                "expected_steps",
                "past",
                "off_reason",
                "basis",
            }
        ),
        "context_budget": frozenset(
            {
                "kind",
                "model",
                "attempt",
                "proxy_version",
                "source",
                "estimate",
                "measured_prompt_tokens",
                "input_limit",
                "over_limit",
                "image_allowance",
                "pruned",
            }
        ),
        "run.finished": frozenset({"result"}),
    }
)
# Keys that a stored event from an older release may lack. Events the agent
# emits now always carry every key of EVENT_FIELDS; a session store accepts
# the older shape too.
OPTIONAL_EVENT_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "assistant.message": frozenset({"prefix"}),
        "reflection": frozenset({"prefix", "flags", "gate", "utility", "stop_reason"}),
    }
)
EVENT_TYPES: Final = frozenset(EVENT_FIELDS)

# Characters a string in an event's data keeps (see json_safe_payload).
EVENT_TEXT_LIMIT: Final = 100_000

# Levels of lists and dicts that an event's data keeps (see json_safe_payload).
_DEPTH_LIMIT: Final = 100


@dataclass(frozen=True)
class AgentEvent:
    """One event of a run.

    ``type`` is one of :data:`EVENT_TYPES`, and ``data`` has exactly the
    keys :data:`EVENT_FIELDS` names for it, with JSON-safe values (see
    :func:`json_safe_payload`). ``run_id`` is the run's
    ``RunResult.task_id``; ``seq`` counts the run's events from 1, with no
    gaps; ``ts`` is the ``time.time()`` at which the event was emitted.

    ``images`` holds the full image parts of a ``tool.finished`` event, for
    consumers in the same process. :meth:`to_dict` leaves them out:
    ``data["images"]`` names them by digest.
    """

    type: str
    run_id: str
    seq: int
    ts: float
    data: Mapping[str, Any]
    images: tuple[ImagePart, ...] = field(default=(), repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """The event as a JSON object, with a shallow copy of ``data``."""
        return {
            "v": EVENT_SCHEMA_VERSION,
            "type": self.type,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": self.ts,
            "data": dict(self.data),
        }


# Called once per event, synchronously and in order; the return value is
# ignored. A sink must not block: slow work belongs on a queue.
EventSink = Callable[[AgentEvent], object]


def json_safe_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of *data* that ``json.dumps(..., allow_nan=False)`` accepts.

    The copy holds only ``str``, ``int``, finite ``float``, ``bool`` and
    ``None``, in lists and ``str``-keyed dicts nested at most 100 levels
    deep, the values of *data* being on level 1:

    - a tuple becomes a list, and a mapping's key that is not a ``str``
      becomes ``str(key)``;
    - a non-finite float becomes ``"nan"``, ``"inf"`` or ``"-inf"``;
    - any other value becomes ``str(value)``, and so does a list, tuple or
      mapping on a level past 100: data nested past the recursion limit,
      as a model's tool-call arguments can be, still gives a payload, and
      so does a list that holds itself;
    - a string longer than :data:`EVENT_TEXT_LIMIT` keeps its first
      ``EVENT_TEXT_LIMIT`` characters, followed by ``"… [+N chars]"``, where
      ``N`` counts the characters left out.

    *data* itself is not changed, and the copy shares no list or dict with
    it.
    """
    return {_key(key): _json_safe(value, 1) for key, value in data.items()}


def _json_safe(value: Any, depth: int) -> Any:
    """*value*, which is on level *depth* of the payload, as JSON-safe data."""
    # bool before int: True is an int too.
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, str):
        return _clip(str.__str__(value))
    if isinstance(value, Mapping | list | tuple) and depth > _DEPTH_LIMIT:
        # Copying it could exhaust the stack: its text, as for any other value.
        return _clip(_text(value))
    if isinstance(value, Mapping):
        return {_key(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item, depth + 1) for item in value]
    return _clip(_text(value))


def _key(key: Any) -> str:
    return _clip(str.__str__(key) if isinstance(key, str) else _text(key))


def _text(value: object) -> str:
    """``str(value)``, or its default repr when ``__str__`` raises."""
    try:
        return str(value)
    except Exception:
        return object.__repr__(value)


def _clip(text: str) -> str:
    if len(text) <= EVENT_TEXT_LIMIT:
        return text
    return f"{text[:EVENT_TEXT_LIMIT]}… [+{len(text) - EVENT_TEXT_LIMIT} chars]"
