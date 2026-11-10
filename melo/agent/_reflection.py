"""Reading, fitting and gating a reflection.

A reflection reply is a JSON object with a valid ``recommended_action``:
:func:`_read_reflection` finds it (and mends one common slip), the
``_coerce_*`` helpers turn its values into the fields of a
:class:`~localmelo.melo.schema.ReflectionEntry` and a
:class:`~localmelo.melo.schema.ReflectionDecision`
(:func:`_reflection_from_object`), and :func:`_fit_reflection` and
:func:`_cap_reflection_lists` keep an entry within its budget. The gate helpers
(:func:`_gate_stop_reason`, :func:`_stop_reason`) say why a run does not go on
after a reflection. None of them needs the Agent.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from localmelo.melo.agent._json_repair import error_line, rejected_objects, repairs
from localmelo.melo.schema import (
    Message,
    ReflectionDecision,
    ReflectionEntry,
    WorkCheckpoint,
    shorten,
    strip_code_fences,
)


def _coerce_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _coerce_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value] if value else []
    if value is None:
        return []
    return [str(value)]


# The decisions a reflection may recommend. An object without one of them is no
# reflection (see _find_reflection).
REFLECTION_ACTIONS = ("continue", "stop", "decompose", "done")


def _json_values(text: str) -> Iterator[object]:
    """Yield the JSON values of a reflection reply, in the order they are read.

    A Markdown code fence around the reply is stripped first. When the
    rest is JSON as a whole, that value is the only one. Otherwise each
    balanced ``{...}`` in it that parses as a JSON object is one, so text
    around the objects does not matter. After a ``{`` that does not
    start one, the search goes on from where the decoder gave up, and
    after an object, from its end, so it reads the reply about once and
    never starts inside a string of the text it already read, nor inside
    an object it yielded. Nesting deeper than the decoder's recursion
    limit ends the search.
    """
    body = strip_code_fences(text)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        pass
    else:
        yield value
        return
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
            continue
        except RecursionError:
            return
        yield value
        start = body.find("{", end)


def _decode_reflection(text: str) -> object:
    """Return the first JSON value of a reflection reply; ``ValueError`` if none.

    See :func:`_json_values` for how the reply is read: the reply as a
    whole when it is JSON, else the first ``{...}`` object in it.
    """
    for value in _json_values(text):
        return value
    raise ValueError("No JSON object in the reflection response")


def _holds_reflection_object(text: str) -> bool:
    """Whether a reflection reply holds a JSON object (see :func:`_decode_reflection`)."""
    try:
        return isinstance(_decode_reflection(_coerce_str(text)), dict)
    except ValueError:
        return False


def _find_reflection(text: str) -> dict[str, Any] | None:
    """The reflection a reply holds, or ``None``.

    That is the first JSON value of the reply (see :func:`_json_values`)
    that is an object whose ``recommended_action``, as a string, is one
    of :data:`REFLECTION_ACTIONS`. An object without one, such as the
    arguments of a tool call written out as text, is passed over for the
    objects after it, but an object nested inside it is not looked at.
    JSON that is not an object holds no reflection. This is the strict
    reading; the loop reads replies with :func:`_read_reflection`.
    """
    try:
        for value in _json_values(_coerce_str(text)):
            if not isinstance(value, dict):
                continue
            if _coerce_str(value.get("recommended_action")) in REFLECTION_ACTIONS:
                return value
    except ValueError:  # a number the decoder cannot convert ends the search
        pass
    return None


# The most ``{...}`` objects of one reply that _read_reflection tries to repair.
REPAIR_TRIES = 8


def _read_reflection(text: str) -> tuple[dict[str, Any], str | None] | None:
    """The reflection a reply holds, and the repair it took.

    That is the reflection of :func:`_find_reflection` with ``None`` as the
    repair. A reply that holds none gets one more reading with a single
    conservative repair of a ``{...}`` the decoder rejects:
    ``"unmatched_closer"``, ``"trailing_comma"``, ``"control_char"``,
    ``"invalid_escape"``, ``"unclosed_array"``, ``"single_quotes"`` or
    ``"key_quote"`` (see
    :func:`~localmelo.melo.agent._json_repair.repairs`). The repaired text is
    read as any reply is, so a repair counts only when it gives an object
    with a valid ``recommended_action``, under the order and the nesting rule
    of :func:`_find_reflection`. A reply that needs two repairs holds no
    reflection, and nothing that closes a string or an object is inserted but
    one bracket, so truncated text stays unreadable. A reply that reads as it
    stands is never repaired.
    """
    found = _find_reflection(text)
    if found is not None:
        return found, None
    text = _coerce_str(text)
    if "recommended_action" not in text:
        return None
    body = strip_code_fences(text)
    for start, error in itertools.islice(rejected_objects(body), REPAIR_TRIES):
        for repair, fixed in repairs(body, start, error):
            found = _find_reflection(fixed)
            if found is not None:
                return found, repair
    return None


def _json_error_line(text: str) -> str:
    """The line that a retry adds when a reply has JSON the decoder rejects, else ``""``.

    It quotes the error of the first such ``{...}`` and the text around it
    (see :func:`~localmelo.melo.agent._json_repair.error_line`); the position
    counts the characters of the reply, a code fence around it included.
    """
    text = _coerce_str(text)
    body = strip_code_fences(text)
    first = next(rejected_objects(body), None)
    if first is None:
        return ""
    return error_line(body, first[1], max(text.find(body), 0))


def _unreadable_reason(reply: Message) -> str:
    """Why *reply* holds no reflection: ``tool_call``, ``invalid`` or ``no_json``.

    ``tool_call``: the reply has tool calls (and no reflection in its text);
    ``invalid``: its text has a JSON object that no reflection is read from;
    ``no_json``: its text has no JSON object.
    """
    if reply.tool_calls:
        return "tool_call"
    if _holds_reflection_object(reply.content):
        return "invalid"
    return "no_json"


def _serialize_reflections(entries: list[ReflectionEntry]) -> list[Message]:
    """Convert reflection entries into compact system Messages for retrieval context."""
    if not entries:
        return []
    parts: list[str] = []
    for r in entries:
        lines = [f"[reflection] Attempt {r.attempt_id}: {r.summary}"]
        if r.recommended_avoids:
            lines.append(f"  Avoid: {'; '.join(r.recommended_avoids)}")
        if r.next_promising_directions:
            lines.append(f"  Try: {'; '.join(r.next_promising_directions)}")
        if r.useful_evidence:
            lines.append(f"  Evidence: {'; '.join(r.useful_evidence)}")
        if r.failed_hypotheses:
            lines.append(f"  Failed: {'; '.join(r.failed_hypotheses)}")
        parts.append("\n".join(lines))
    return [Message(role="system", content="\n".join(parts))]


# The lists of a reflection entry that _fit_reflection empties, in this
# order and each from its end, until the entry fits its budget.
REFLECTION_TRIM_ORDER = (
    "failed_hypotheses",
    "useful_evidence",
    "recommended_avoids",
    "next_promising_directions",
)


def _reflection_size(entry: ReflectionEntry) -> int:
    """Characters of *entry* as :func:`_serialize_reflections` writes it."""
    return len(_serialize_reflections([entry])[0].content)


_Trimmed = TypeVar("_Trimmed", ReflectionEntry, WorkCheckpoint)


def _trimmed(
    entry: _Trimmed,
    limit: int,
    size: Callable[[_Trimmed], int],
    lists: Sequence[str],
    texts: Sequence[str],
) -> _Trimmed:
    """*entry*, trimmed until *size* of it is at most *limit*.

    Items are dropped from the end of each of *lists* in turn, one at a time and
    only until it fits; then each of *texts* in turn is shortened at a word
    boundary and ends with "…" (see :func:`shorten`). *entry* itself is not
    changed: one that fits is returned as it is, any other as a trimmed copy.
    """
    if size(entry) <= limit:
        return entry
    fitted = dataclasses.replace(entry)
    for name in lists:
        items = list(getattr(entry, name))  # a copy: entry keeps its own
        setattr(fitted, name, items)
        while items:
            items.pop()
            if size(fitted) <= limit:
                return fitted
    for name in texts:
        over = size(fitted) - limit
        if over <= 0:
            break
        text = getattr(fitted, name)
        setattr(fitted, name, shorten(text, max(len(text) - over, 0)))
    return fitted


def _fit_reflection(entry: ReflectionEntry, limit: int) -> ReflectionEntry:
    """*entry*, trimmed until its serialized form fits in *limit* characters.

    The serialized form is the one :func:`_serialize_reflections` writes.
    Items are dropped from the end of each list of
    :data:`REFLECTION_TRIM_ORDER` in turn (``failed_hypotheses`` first,
    ``next_promising_directions`` last), one at a time and only until the
    entry fits. When the lists are empty and it still does not fit, the
    summary is shortened at a word boundary and ends with "…" (see
    :func:`shorten`). *entry* itself is not changed: one that fits is
    returned as it is, any other as a trimmed copy.
    """
    return _trimmed(entry, limit, _reflection_size, REFLECTION_TRIM_ORDER, ("summary",))


# The lists of a reflection that are capped, the most items each keeps and the
# most characters of an item.
REFLECTION_CAPPED_LISTS = ("useful_evidence", "failed_hypotheses", "recommended_avoids")


REFLECTION_LIST_ITEMS = 8


REFLECTION_ITEM_CHARS = 300


def _cap_reflection_lists(entry: ReflectionEntry) -> ReflectionEntry:
    """*entry* with each of :data:`REFLECTION_CAPPED_LISTS` kept to its cap.

    A list keeps its first :data:`REFLECTION_LIST_ITEMS` items, each cut to
    :data:`REFLECTION_ITEM_CHARS` characters at a word boundary (see
    :func:`shorten`). *entry* itself is not changed.
    """
    capped = dataclasses.replace(entry)
    for name in REFLECTION_CAPPED_LISTS:
        items = getattr(entry, name)[:REFLECTION_LIST_ITEMS]
        setattr(capped, name, [shorten(item, REFLECTION_ITEM_CHARS) for item in items])
    return capped


def _gate_stop_reason(
    decision: ReflectionDecision, *, patient: bool = False
) -> str | None:
    """The first check of the continuation gate that *decision* fails.

    The codes are the ones in ``docs/reference/events.md``:
    ``action_not_continue``, ``not_feasible``, ``not_concrete`` and
    ``not_novel``. ``None`` when every check holds, so that the next attempt
    may start. With *patient* true (the run's one allowance is unspent), a
    decision that is not novel passes that check.
    """
    if decision.recommended_action != "continue":
        return "action_not_continue"
    if not decision.task_still_feasible:
        return "not_feasible"
    if not decision.next_step_is_concrete:
        return "not_concrete"
    if not decision.next_step_is_novel and not patient:
        return "not_novel"
    return None


# The stop reasons of the budget boundaries, by the error that the run
# ends with there.
_OUT_OF_REASONS = {
    "Max attempts reached": "no_attempt_left",
    "Max steps reached": "step_budget_used",
}


def _stop_reason(
    decision: ReflectionDecision | None,
    out_of: str,
    *,
    brief: bool = False,
    patient: bool = False,
) -> str | None:
    """Why the run does not go on after a reflection, ``None`` if it does.

    *decision* is ``None`` for a reply that holds no reflection, and
    *out_of* is the boundary's error (``""`` when another attempt can
    start). The one place that decides: ``continue_run`` is whether this
    is ``None``. In a *brief* run (``AgentOptions.brief_probe``) a
    ``continue`` goes on whatever the gate's flags say. *patient* is
    :func:`_gate_stop_reason`'s.
    """
    if decision is None:
        return "unreadable_reply" if out_of else None
    if decision.recommended_action == "done":
        return "action_not_continue"
    if out_of:
        return _OUT_OF_REASONS[out_of]
    if brief and decision.recommended_action == "continue":
        return None
    return _gate_stop_reason(decision, patient=patient)


def _reflection_from_object(
    data: dict[str, Any],
    attempt_id: int,
    failure_type: str,
) -> tuple[ReflectionEntry, ReflectionDecision]:
    """The reflection entry and decision that the JSON object *data* gives.

    Each value is coerced to its field's type, and a missing one takes a
    default. A
    ``recommended_action`` that is not one of
    :data:`REFLECTION_ACTIONS` reads ``stop``.
    """
    entry = ReflectionEntry(
        attempt_id=attempt_id,
        summary=_coerce_str(data.get("summary")),
        failed_hypotheses=_coerce_str_list(data.get("failed_hypotheses")),
        disproven_actions=_coerce_str_list(data.get("disproven_actions")),
        useful_evidence=_coerce_str_list(data.get("useful_evidence")),
        unresolved_questions=_coerce_str_list(data.get("unresolved_questions")),
        recommended_avoids=_coerce_str_list(data.get("recommended_avoids")),
        next_promising_directions=_coerce_str_list(
            data.get("next_promising_directions")
        ),
        failure_type=failure_type or _coerce_str(data.get("failure_type")),
        confidence=_coerce_float(data.get("confidence")),
    )

    action = _coerce_str(data.get("recommended_action")) or "stop"
    if action not in REFLECTION_ACTIONS:
        action = "stop"

    decision = ReflectionDecision(
        task_still_feasible=_coerce_bool(data.get("task_still_feasible", True)),
        next_step_is_concrete=_coerce_bool(data.get("next_step_is_concrete")),
        next_step_is_novel=_coerce_bool(data.get("next_step_is_novel")),
        recommended_action=action,
        rationale=_coerce_str(data.get("rationale")),
        best_effort_result=_coerce_str(data.get("best_effort_result")),
        tried_memory=entry,
    )
    return entry, decision


def _reflection_summary(content: str) -> str:
    """The summary (else rationale) of a readable reflection reply, or ""."""
    found = _read_reflection(content)
    if found is not None:
        data: object = found[0]
    else:
        try:
            data = _decode_reflection(_coerce_str(content))
        except ValueError:
            return ""
    if not isinstance(data, dict):
        return ""
    for key in ("summary", "rationale"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""
