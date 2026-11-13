"""Memory aids of a ``--brief`` run: the pushes and notes that keep it writing.

Each applies only to a brief run (``AgentOptions.brief_probe``): a push after
steps of reading that changed nothing, a note of the tool output that left
the window, and the pointers a reflection hands to the next attempt. What left
the window is read here for every run (:func:`dropped_calls`): a brief run's note
lists it call by call, and a work checkpoint lists it source by source
(:func:`covered_lines`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.procedures.mask import redact_and_mask

if TYPE_CHECKING:
    from localmelo.melo.schema import Message, ReflectionEntry

# Planning steps in a row without a change before the step-level push.
STEP_PUSH_STEPS = 6
STEP_PUSH = (
    f"You have read for {STEP_PUSH_STEPS} steps without changing anything. Write "
    "the part of the task you can already write now (the smallest item first), "
    "then continue."
)
# Dropped tool results the note remembers, and the key argument's length.
DROPPED_KEPT = 8
ARG_LIMIT = 80
# Added to the reflection request of a brief run (the frozen
# REFLECTION_PROMPT stays as it is).
REFLECTION_ADDENDUM = (
    "In useful_evidence, give exact file paths with line numbers; make the first "
    "of next_promising_directions the first command or edit to run next."
)
START_WITH_LIMIT = 300

_KEY_ARGS = ("path", "file_path", "command", "code", "query", "pattern", "url")
_SHELL_ARGS = ("command", "code")


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _key_arg(arguments: dict[str, object]) -> tuple[str, str]:
    """A call's key argument and its value, whole; ``("", "")`` without one."""
    for key in (*_KEY_ARGS, *arguments):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    return "", ""


def _describe(name: str, arguments: dict[str, object]) -> str:
    """The tool name and its key argument, redacted whole and then shortened, so
    that no cut leaves a piece of a secret no pattern matches."""
    key, value = _key_arg(arguments)
    if not key:
        return name
    arg = _shorten(redact_and_mask(value), ARG_LIMIT)
    return f"{name} `{arg}`" if key in _SHELL_ARGS else f"{name} {arg}"


# The tools whose results end with an exit status the cue keeps.
_COMMANDS = frozenset({"shell_exec", "python_exec"})
_EXIT = re.compile(r"\[exit code (-?\d+)\]")
# Lines of the cue list a work checkpoint shows at most: past it, the oldest
# sources are counted per tool on one line.
COVERED_LINES = 30


@dataclass(frozen=True)
class Dropped:
    """One tool result that left the context: its call and what came of it.

    *target* is the call's key argument whole, *source* the tool and that
    argument shortened and redacted, for display; *part*
    the part of a page the call read (``None`` for a tool that reads no parts),
    *status* ``exit N`` for a command, ``failed`` for a call with an error,
    ``effect unknown`` for a call whose result was never kept, else ``""``.
    """

    call_id: str
    tool: str
    target: str
    source: str
    chars: int
    part: int | None
    status: str


def cue(
    call_id: str,
    name: str,
    arguments: Mapping[str, object],
    *,
    outcome: str = "",
    exit_code: int | None = None,
    chars: int = 0,
) -> Dropped:
    """The entry of one call whose result left the context (or was never kept).

    *outcome* is ``failed`` for a call with an error, ``unknown`` for one whose
    result was never kept (it may have done part of its work), else ``""``.
    """
    given = dict(arguments)
    part = given.get("part")
    status = {"failed": "failed", "unknown": "effect unknown"}.get(outcome, "")
    if not status and name in _COMMANDS:
        status = f"exit {exit_code or 0}"
    if type(part) is not int:
        part = 1 if name == "web_fetch" or "part" in given else None
    return Dropped(
        call_id, name, _key_arg(given)[1], _describe(name, given), chars, part, status
    )


def dropped_calls(messages: Sequence[Message]) -> list[Dropped]:
    """One entry per tool result among the dropped *messages*, in their order."""
    calls = {call.id: call for msg in messages for call in msg.tool_calls if call.id}
    found = []
    for msg in messages:
        if msg.role == "tool":
            call = calls.get(msg.tool_call_id)
            exits = _EXIT.findall(msg.content)
            failed = msg.tool_result is not None and msg.tool_result.error
            found.append(
                cue(
                    msg.tool_call_id,
                    call.tool_name if call is not None else "tool",
                    call.arguments if call is not None else {},
                    outcome="failed" if failed else "",
                    exit_code=int(exits[-1]) if exits else None,
                    chars=len(msg.content),
                )
            )
    return found


def dropped_notes(messages: Sequence[Message]) -> list[str]:
    """One line per tool result among the dropped *messages* (a brief run's note)."""
    return [f"{d.source} ({d.chars:,} chars)" for d in dropped_calls(messages)]


def _spans(parts: Sequence[int]) -> str:
    """``1-3, 7`` for the parts 1, 2, 3 and 7."""
    spans: list[list[int]] = []
    for part in sorted(set(parts)):
        if spans and part == spans[-1][1] + 1:
            spans[-1][1] = part
        else:
            spans.append([part, part])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def covered_lines(dropped: Sequence[Dropped], limit: int = COVERED_LINES) -> list[str]:
    """The cue list of a work checkpoint: one line per source, by their newest
    call, oldest first (a source called again moves to the end).

    A source is a tool and its key argument, whole (two long addresses that
    share their start are two sources), shown shortened; its line names the parts read of a
    page (``web_fetch <url> parts 1-62``) and the last status of a command
    (``shell_exec `ls` (exit 0)``). Past *limit* lines, the oldest sources
    become one line that counts them per tool.
    """
    order: dict[tuple[str, str], list[Dropped]] = {}
    for entry in dropped:
        key = (entry.tool, entry.target)
        order[key] = [*order.pop(key, []), entry]
    lines = []
    for entries in order.values():
        line = entries[0].source
        parts = [e.part for e in entries if e.part is not None]
        if parts:
            line += f" part{'s' if len(set(parts)) > 1 else ''} {_spans(parts)}"
        if entries[-1].status:
            line += f" ({entries[-1].status})"
        lines.append(line)
    if len(lines) <= limit:
        return lines
    older = list(order.values())[: len(lines) - limit + 1]
    counts: dict[str, int] = {}
    for entries in older:
        counts[entries[0].tool] = counts.get(entries[0].tool, 0) + 1
    summary = "earlier: " + ", ".join(
        f"{count} {tool} source{'s' if count > 1 else ''}"
        for tool, count in counts.items()
    )
    return [summary, *lines[len(older) :]]


def render_dropped(notes: Sequence[str], window: Sequence[Message] = ()) -> str:
    """The tail line of a planning request; "" when nothing was dropped.

    A note whose tool and argument the *window* holds again (the model read
    the file a second time) is left out: that output is in the context.
    """
    present = {note.rsplit(" (", 1)[0] for note in dropped_notes(window)}
    notes = [note for note in notes if note.rsplit(" (", 1)[0] not in present]
    if not notes:
        return ""
    return (
        "Earlier tool output no longer in your context: "
        + "; ".join(notes)
        + ". Read again what you need before you edit."
    )


def start_with(entry: ReflectionEntry | None) -> str:
    """The "Start with:" line for a reflection's first direction, or ""."""
    if entry is None or not entry.next_promising_directions:
        return ""
    direction = redact_secrets(entry.next_promising_directions[0].strip())[0]
    if not direction:
        return ""
    return f"Start with: {_shorten(direction, START_WITH_LIMIT)}"


# A file's identity for "was it edited": size and mtime, -1 when it is gone.
FileStamp = tuple[int, int]


def edited_files(
    before: Mapping[str, FileStamp], after: Mapping[str, FileStamp]
) -> set[str]:
    """The files the run edited: listed now and not as they were at its start."""
    return {name for name, stamp in after.items() if before.get(name) != stamp}
