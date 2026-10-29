"""Memory aids of a ``--brief`` run: the pushes and notes that keep it writing.

Each applies only to a brief run (``AgentOptions.brief_probe``): a push after
steps of reading that changed nothing, a note of the tool output that left
the window, and the pointers a reflection hands to the next attempt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from localmelo.melo.checker.redaction import redact_secrets

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


def _describe(name: str, arguments: dict[str, object]) -> str:
    """The tool name and its key argument, shortened and redacted."""
    keys = [*_KEY_ARGS, *arguments]
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            arg = redact_secrets(_shorten(value, ARG_LIMIT))[0]
            return f"{name} `{arg}`" if key in _SHELL_ARGS else f"{name} {arg}"
    return name


def dropped_notes(messages: Sequence[Message]) -> list[str]:
    """One line per tool result among the dropped *messages*."""
    calls = {call.id: call for msg in messages for call in msg.tool_calls if call.id}
    notes = []
    for msg in messages:
        if msg.role != "tool":
            continue
        call = calls.get(msg.tool_call_id)
        name = call.tool_name if call is not None else "tool"
        notes.append(
            f"{_describe(name, call.arguments if call is not None else {})} "
            f"({len(msg.content):,} chars)"
        )
    return notes


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
