"""Deterministic, request-local context fitting; no history is changed."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable

from localmelo.melo.memory.notes import RECALL_END, RECALL_START
from localmelo.melo.memory.short import group_turns
from localmelo.melo.schema import EMPTY_REPLY_LABEL, Message, ToolDef


class ContextFitError(Exception):
    """The pinned request cannot meet the requested input budget."""


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


def _complete_tool_unit(unit: list[Message]) -> bool:
    if not unit[0].tool_calls:
        return False
    calls = [call.id for call in unit[0].tool_calls]
    results = [m.tool_call_id for m in unit if m.role == "tool"]
    return len(set(calls)) == len(calls) and sorted(calls) == sorted(results)


def _removable(unit: list[Message]) -> bool:
    first = unit[0]
    if first.tool_calls:
        return _complete_tool_unit(unit)
    return (
        first.role == "assistant"
        and first.content.startswith(EMPTY_REPLY_LABEL)
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
) -> tuple[list[Message], dict[str, int]]:
    """Fit copies using the caller's assembled-request estimate.

    Counts are ``tool_results``, ``thinking`` and ``units``. A retry must
    additionally estimate strictly below ``force_smaller_than``. Failure is
    conservative: malformed/in-flight pairs, images and failure evidence stay.
    """
    units = group_turns(copy.deepcopy(messages), empty_prefix=True)
    counts = {"tool_results": 0, "thinking": 0, "units": 0}

    def flatten() -> list[Message]:
        return [message for unit in units for message in unit]

    def fits() -> bool:
        size = estimate(flatten(), tools)
        return size <= limit and (
            force_smaller_than is None or size < force_smaller_than
        )

    if fits():
        return flatten(), counts

    # Protect the newest complete structural unit and latest failing check,
    # as well as all malformed or unfinished protocol units.
    complete = [unit for unit in units if _removable(unit)]
    newest = complete[-1] if complete else None
    failing = next(
        (
            unit
            for unit in reversed(units)
            if any(m.role == "tool" and _failed(m) for m in unit)
        ),
        None,
    )

    for unit in units:
        if not _complete_tool_unit(unit):
            continue
        for message in unit:
            if message.role != "tool" or _failed(message):
                continue
            text, recall = _recall_suffix(message.content)
            if len(text) <= _PREVIEW_CHARS or _PREVIEW.search(text):
                continue
            statuses = list(_EXIT.finditer(text))
            cut = _PREVIEW_CHARS
            for status in statuses:
                if status.start() < cut < status.end():
                    cut = status.start()
                    break
            kept_statuses = [m.group(0) for m in statuses if m.start() >= cut]
            preview = text[:cut]
            omitted = len(text) - len(preview) - sum(map(len, kept_statuses))
            replacement = (
                f"{preview}\n[tool output preview: {omitted} characters omitted]"
            )
            if kept_statuses:
                replacement += "\n" + " ".join(kept_statuses)
            if len(replacement) >= len(text):
                continue
            message.content = replacement + ("\n" + recall if recall else "")
            counts["tool_results"] += 1
            if fits():
                return flatten(), counts

    if not requires_replay:
        for unit in units:
            # Thinking on in-flight or malformed calls stays pinned.
            if unit is newest or unit is units[-1]:
                continue
            if unit[0].tool_calls and not _complete_tool_unit(unit):
                continue
            for message in unit:
                if message.role == "assistant" and message.thinking:
                    message.thinking = ""
                    counts["thinking"] += 1
                    if fits():
                        return flatten(), counts

    # Recall is a whole evictable item, even when attached to a tool result.
    # Keep the current unit's recall and every incomplete tool unit intact.
    for unit in list(units):
        if unit is newest or unit is units[-1]:
            continue
        if unit[0].tool_calls and not _complete_tool_unit(unit):
            continue
        for message in list(unit):
            body, recall = _recall_suffix(message.content)
            if not recall or message.images:
                continue
            if message.role == "user" and not body:
                unit.remove(message)
                if not unit:
                    units.remove(unit)
            elif message.role == "tool":
                message.content = body.rstrip("\n")
            else:
                continue
            counts["units"] += 1
            if fits():
                return flatten(), counts

    for unit in list(units):
        if (
            unit is newest
            or unit is failing
            or unit is units[-1]
            or not _removable(unit)
        ):
            continue
        if any(m.role == "tool" and _failed(m) for m in unit):
            continue
        if any(m.images or (m.tool_result and m.tool_result.images) for m in unit):
            continue
        # A user message can resemble a recall block; keep it unless the unit
        # is the agent's own empty-reply/nudge pair.
        if any(m.role == "user" for m in unit) and not unit[0].content.startswith(
            EMPTY_REPLY_LABEL
        ):
            continue
        units = [candidate for candidate in units if candidate is not unit]
        counts["units"] += 1
        if fits():
            return flatten(), counts

    raise ContextFitError(
        "Request cannot fit without removing pinned context; increase the input "
        "budget or reduce the task, schemas, images, or current tool output."
    )
