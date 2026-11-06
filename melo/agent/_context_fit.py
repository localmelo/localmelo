"""Deterministic, request-local context fitting; no history is changed."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Sequence

from localmelo.melo.memory.notes import RECALL_END, RECALL_START
from localmelo.melo.memory.short import group_turns
from localmelo.melo.schema import (
    REFLECTIONS_HEADER,
    Message,
    ToolDef,
    starts_with_reply_label,
)


class ContextFitError(Exception):
    """The pinned request cannot meet the requested input budget."""


class PromptCapError(ContextFitError):
    """``Checker.pre_plan`` refused a request still over the prompt cap after the fit."""


_EXIT = re.compile(r"\[exit code (-?\d+)\]", re.IGNORECASE)
_FAILURE = re.compile(r"\b(error|errors|failed|failure|traceback)\b", re.IGNORECASE)
_PREVIEW_CHARS = 256
_PREVIEW = re.compile(r"\n\[tool output preview: \d+ characters omitted\]")


def _recall_suffix(text: str) -> tuple[str, str]:
    """Separate one complete trailing recall block from its tool output."""
    start = text.rfind(RECALL_START)
    if start < 0 or not text.rstrip().endswith(RECALL_END):
        return text, ""
    if start and text[start - 1] != "\n":
        return text, ""
    return text[:start], text[start:]


def _failed(message: Message) -> bool:
    result = message.tool_result
    text = message.content
    return bool(
        (result and result.error)
        or _FAILURE.search(text)
        or any(int(match.group(1)) != 0 for match in _EXIT.finditer(text))
    )


def _preview(text: str) -> str | None:
    """*text* cut to a preview that keeps its exit codes, or None when not shorter."""
    if len(text) <= _PREVIEW_CHARS or _PREVIEW.search(text):
        return None
    statuses = list(_EXIT.finditer(text))
    cut = _PREVIEW_CHARS
    for status in statuses:
        if status.start() < cut < status.end():
            cut = status.start()
            break
    kept_statuses = [m.group(0) for m in statuses if m.start() >= cut]
    preview = text[:cut]
    omitted = len(text) - len(preview) - sum(map(len, kept_statuses))
    replacement = f"{preview}\n[tool output preview: {omitted} characters omitted]"
    if kept_statuses:
        replacement += "\n" + " ".join(kept_statuses)
    return replacement if len(replacement) < len(text) else None


def _complete_tool_unit(unit: list[Message]) -> bool:
    if not unit[0].tool_calls:
        return False
    calls = [call.id for call in unit[0].tool_calls]
    results = [m.tool_call_id for m in unit if m.role == "tool"]
    return len(set(calls)) == len(calls) and sorted(calls) == sorted(results)


def _unit_key(unit: list[Message]) -> tuple[object, ...]:
    """What identifies a unit across requests: its first message, by value."""
    first = unit[0]
    calls = tuple(
        (call.id, call.tool_name, repr(call.arguments)) for call in first.tool_calls
    )
    return (first.role, first.content, calls, len(unit))


def _message_key(message: Message) -> tuple[object, ...]:
    """What identifies a message across requests, as it came in: by value."""
    calls = tuple(call.id for call in message.tool_calls)
    return (message.role, message.tool_call_id, message.content, calls)


class FitCut:
    """What a fit cut, held across a run's requests of one kind.

    *key* is the oldest unit it kept after its last drop, and *level* how far
    the drops had to go (see ``fit_request``); *shortened* holds the steps it
    shortened a message with (a preview, cleared thinking, removed recall),
    each with the message's key as it came in.
    """

    def __init__(self) -> None:
        self.key: tuple[object, ...] | None = None
        self.level = 0
        self.shortened: set[tuple[object, ...]] = set()

    @property
    def active(self) -> bool:
        """Whether a request must be fitted to repeat this cut."""
        return self.key is not None or bool(self.shortened)


def _removable(unit: list[Message]) -> bool:
    first = unit[0]
    if first.tool_calls:
        return _complete_tool_unit(unit)
    return (
        first.role == "assistant"
        and starts_with_reply_label(first.content)
        and len(unit) == 2
    )


def fit_request(
    messages: list[Message],
    tools: list[ToolDef] | None,
    *,
    limit: int,
    estimate: Callable[[list[Message], list[ToolDef] | None], int],
    requires_replay: bool,
    force_smaller_than: int | None = None,
    drop_target: float = 1.0,
    hold: FitCut | None = None,
    history: Sequence[Message] = (),
) -> tuple[list[Message], dict[str, int]]:
    """Fit copies using the caller's assembled-request estimate.

    Counts are ``tool_results``, ``thinking`` and ``units``. A retry must
    additionally estimate strictly below ``force_smaller_than``. Once the fit
    must act, it shortens (old results to previews, old thinking, old recall)
    and then drops whole units, oldest first, down to *drop_target* of
    *limit*, so the next requests fit as they are. The drops go in three
    levels, each only when the one before cannot reach the target: plain old
    turns; then images and failure evidence too, but the latest failing
    check; then any unit but the system prompt, the task's own message, the
    newest reflections block and the last unit, so the floor is those alone.
    Only when they do not fit does the fit fail.

    *history* names, by identity, messages of *messages* from the turns
    before the current task (a chat's earlier tasks): a unit made only of
    them may go whole, whatever it holds.

    With a *hold*, the held cut is repeated first: the units that its unit
    follows are dropped (those the fit may drop) and the messages it shortened
    are shortened the same way, so the request is byte for byte the last one
    up to where it grew; the units dropped before it are the ones its level
    allowed. Only when that is over *limit* does the fit cut again, down to
    *drop_target*. A held unit or message no longer in the request is
    forgotten. The cut then holds what this fit did.
    """
    copies = copy.deepcopy(messages)
    earlier = {id(m) for m in history}
    past = {id(c) for c, m in zip(copies, messages, strict=True) if id(m) in earlier}
    units = group_turns(copies, empty_prefix=True)
    counts = {"tool_results": 0, "thinking": 0, "units": 0}
    original = list(units)

    def before_task(unit: list[Message]) -> bool:
        return all(id(m) in past for m in unit)

    def flatten() -> list[Message]:
        return [message for unit in units for message in unit]

    def fits(share: float = 1.0) -> bool:
        size = estimate(flatten(), tools)
        return size <= limit * share and (
            force_smaller_than is None or size < force_smaller_than
        )

    # Protect the newest complete structural unit and the task's latest
    # failing check, as well as all malformed or unfinished protocol units.
    complete = [unit for unit in units if _removable(unit)]
    newest = complete[-1] if complete else None
    failing = next(
        (
            unit
            for unit in reversed(units)
            if not before_task(unit)
            and any(m.role == "tool" and _failed(m) for m in unit)
        ),
        None,
    )

    # What the last level keeps: the task's own message (the first user turn
    # after the earlier tasks) and the newest block of reflections.
    task = next(
        (unit for unit in units if not before_task(unit) and unit[0].role == "user"),
        None,
    )
    lesson = next(
        (
            unit
            for unit in reversed(units)
            if unit[0].role == "user" and unit[0].content.startswith(REFLECTIONS_HEADER)
        ),
        None,
    )

    def droppable(unit: list[Message], level: int = 0) -> bool:
        if unit is units[-1]:
            return False
        if level >= 2:
            return before_task(unit) or (
                unit is not task
                and unit is not lesson
                and all(m.role != "system" for m in unit)
            )
        if unit is newest:
            return False
        if before_task(unit):
            return True
        if unit is failing or not _removable(unit):
            return False
        if level == 0 and any(m.role == "tool" and _failed(m) for m in unit):
            return False
        if level == 0 and any(
            m.images or (m.tool_result and m.tool_result.images) for m in unit
        ):
            return False
        # A user message can resemble a recall block; keep it unless the unit
        # is the agent's own empty-reply/nudge pair.
        return not (
            any(m.role == "user" for m in unit)
            and not starts_with_reply_label(unit[0].content)
        )

    def held() -> tuple[object, ...] | None:
        """The oldest unit kept after the last unit this fit dropped."""
        kept = {id(unit) for unit in units}
        dropped = [i for i, unit in enumerate(original) if id(unit) not in kept]
        if not dropped:
            return None
        return next(
            _unit_key(unit) for unit in original[dropped[-1] + 1 :] if id(unit) in kept
        )

    # The steps that shorten one message, keyed by it as it came in.
    keys = {id(m): _message_key(m) for unit in units for m in unit}
    done: set[tuple[object, ...]] = set()

    level = 0

    def finish() -> tuple[list[Message], dict[str, int]]:
        if hold is not None:
            if (key := held()) is not None:
                hold.key = key
                hold.level = level
            hold.shortened = done
        return flatten(), counts

    if hold is not None and hold.key is not None:
        start = next((i for i, u in enumerate(units) if _unit_key(u) == hold.key), None)
        if start is None:
            hold.key = None
        else:
            level = hold.level
            before = [unit for unit in units[:start] if droppable(unit, level)]
            units = [unit for unit in units if not any(unit is d for d in before)]
            counts["units"] += len(before)

    def steps() -> list[tuple[str, list[Message], Message]]:
        """The shortening steps in order: previews, then thinking, then recall."""
        found = [
            ("preview", unit, m)
            for unit in units
            if _complete_tool_unit(unit)
            for m in unit
            if m.role == "tool"
        ]
        # Thinking on in-flight or malformed calls stays pinned, and so does the
        # newest unit's; so does recall there, a whole evictable item elsewhere,
        # even when attached to a tool result.
        later = [
            unit
            for unit in units
            if unit is not newest
            and unit is not units[-1]
            and not (unit[0].tool_calls and not _complete_tool_unit(unit))
        ]
        if not requires_replay:
            found += [
                ("thinking", u, m) for u in later for m in u if m.role == "assistant"
            ]
        found += [("recall", u, m) for u in later for m in u]
        return found

    def shorten(step: str, unit: list[Message], message: Message) -> bool:
        if step == "preview":
            text, recall = _recall_suffix(message.content)
            preview = None if _failed(message) else _preview(text)
            if preview is None:
                return False
            message.content = preview + ("\n" + recall if recall else "")
            counts["tool_results"] += 1
        elif step == "thinking":
            if not message.thinking:
                return False
            message.thinking = ""
            counts["thinking"] += 1
        else:
            body, recall = _recall_suffix(message.content)
            if not recall or message.images:
                return False
            if message.role == "user" and not body:
                unit.remove(message)
                if not unit:
                    units.remove(unit)
            elif message.role == "tool":
                message.content = body.rstrip("\n")
            else:
                return False
            counts["units"] += 1
        done.add((step, keys[id(message)]))
        return True

    if hold is not None and hold.shortened:
        for step, unit, message in steps():
            if (step, keys[id(message)]) in hold.shortened:
                shorten(step, unit, message)

    if fits():
        return finish()

    for step, unit, message in steps():
        if (step, keys[id(message)]) in done or not shorten(step, unit, message):
            continue
        if fits(drop_target):
            return finish()

    fitting: (
        tuple[list[Message], dict[str, int], tuple[object, ...] | None, int] | None
    ) = None
    # Each level goes only when the cut cannot reach its target, or fit at
    # all, without it; the cut holds the deepest level it used.
    for depth in range(3):
        for unit in list(units):
            if not droppable(unit, depth):
                continue
            units = [candidate for candidate in units if candidate is not unit]
            counts["units"] += 1
            level = max(level, depth)
            if fits(drop_target):
                return finish()
            if fitting is None and fits():
                fitting = (flatten(), dict(counts), held(), level)
    if fitting is not None:
        if hold is not None:
            if fitting[2] is not None:
                hold.key = fitting[2]
                hold.level = fitting[3]
            hold.shortened = done
        return fitting[0], fitting[1]

    raise ContextFitError(
        "Request cannot fit without removing pinned context; increase the input "
        "budget or reduce the task, schemas, images, or current tool output."
    )
