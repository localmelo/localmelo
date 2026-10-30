"""Memory aids of a ``--brief`` run: the pushes and notes that keep it writing.

Each applies only to a brief run (``AgentOptions.brief_probe``): a push after
steps of reading that changed nothing, a note of the tool output that left
the window, the pointers a reflection hands to the next attempt, and (with
``AgentOptions.brief_items``) the numbered items of the brief that no edit has
touched yet.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
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


# The brief's numbered items (``AgentOptions.brief_items``): how many are kept
# and how long each line stays.
MAX_ITEMS = 12
ITEM_LIMIT = 120
MIN_ITEMS = 2
CHANGED, NOT_YET, NO_FILE = "changed", "not yet", "no file named"
ITEMS_HEADER = "Items in the brief with no edit yet:"

_ITEM_LINE = re.compile(r"(?:(\d+)[.)]|- \[[ xX]\])\s+(\S.*)")
_FENCE = re.compile(r"\s*(```|~~~)")
_PATH_TOKEN = re.compile(r"[\w@%+~./-]+")
_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
_NOT_PATHS = frozenset({"e.g", "i.e", "etc", "vs", "approx"})
_STATUS_TIMEOUT_S = 10

# A file's identity for "was it edited": size and mtime, -1 when it is gone.
FileStamp = tuple[int, int]


def parse_items(brief: str) -> list[str]:
    """The top-level numbered or checkbox lines of *brief*, shortened.

    Lines in a fenced code block and indented (nested) lines are ignored; each
    item keeps its first line only, at most ``MAX_ITEMS`` of them. A checkbox
    is numbered by its place; fewer than ``MIN_ITEMS`` items give ``[]``.
    """
    items: list[str] = []
    fenced = False
    for line in brief.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        found = _ITEM_LINE.fullmatch(line.rstrip()) if not fenced else None
        if found is None:
            continue
        number = found.group(1) or str(len(items) + 1)
        items.append(_shorten(f"{number}. {found.group(2)}", ITEM_LIMIT))
        if len(items) == MAX_ITEMS:
            break
    return items if len(items) >= MIN_ITEMS else []


def file_tokens(item: str) -> list[str]:
    """The path-like words of *item*: they hold a ``/`` or end in an extension."""
    tokens = []
    for word in _PATH_TOKEN.findall(item.split(" ", 1)[-1]):
        word = word.strip(".,:;)").removeprefix("./")
        if word.lower() in _NOT_PATHS or not word.strip("./"):
            continue
        if "/" in word or _EXTENSION.search(word):
            tokens.append(word)
    return tokens


def workspace_stamps(workspace: str) -> dict[str, FileStamp] | None:
    """The files ``git status`` lists in *workspace* with their stamps, or None.

    ``None`` outside a git tree. A listed file that is gone (deleted) stamps
    ``(-1, -1)``.
    """
    try:
        out = subprocess.run(
            ["git", "-C", workspace, "status", "--porcelain", "-z", "-uall"],
            capture_output=True,
            check=True,
            timeout=_STATUS_TIMEOUT_S,
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return None
    stamps: dict[str, FileStamp] = {}
    entries = iter(out.split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        name = entry[3:]
        if entry[0] in "RC":  # a rename lists its old name next
            next(entries, None)
        try:
            info = os.stat(os.path.join(workspace, name))
            stamps[name] = (info.st_size, info.st_mtime_ns)
        except OSError:
            stamps[name] = (-1, -1)
    return stamps


def edited_files(
    before: Mapping[str, FileStamp], after: Mapping[str, FileStamp]
) -> set[str]:
    """The files the run edited: listed now and not as they were at its start."""
    return {name for name, stamp in after.items() if before.get(name) != stamp}


def mark_items(items: Sequence[str], edited: set[str]) -> list[str]:
    """``changed``, ``not yet`` or ``no file named`` for each of *items*.

    The mark speaks of edits only: an item marked ``changed`` has an edited
    file among those it names, which does not say the item is done.
    """
    marks = []
    for item in items:
        tokens = file_tokens(item)
        if not tokens:
            marks.append(NO_FILE)
        elif any(_names(token, name) for token in tokens for name in edited):
            marks.append(CHANGED)
        else:
            marks.append(NOT_YET)
    return marks


def _names(token: str, path: str) -> bool:
    if token.endswith("/"):
        return path.startswith(token) or f"/{token}" in f"/{path}"
    return path == token or path.endswith(f"/{token}")


def render_items(items: Sequence[str], marks: Sequence[str]) -> str:
    """The reminder block; "" when no item is marked ``not yet``."""
    pending = [
        (item, mark) for item, mark in zip(items, marks, strict=True) if mark == NOT_YET
    ]
    if not pending:
        return ""
    numbers = ", ".join(item.split(".", 1)[0] for item, _ in pending)
    lines = [f"{ITEMS_HEADER} {numbers}", *(item for item, _ in pending)]
    return "\n".join(lines)
