"""The slash commands of the terminal UI: the table, parsing, and /help.

A draft is a command when it starts with ``/`` and a name of lowercase
letters and dashes, then whitespace or its end; ``/etc/hosts is broken``
is a message. Each command says whether it runs during a turn. The
commands themselves are :class:`~localmelo.support.cli.session.CliSession`
methods.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from rich.console import Group, RenderableType
from rich.text import Text

from localmelo.support.cli.text import pad_cells


@dataclass(frozen=True)
class Command:
    name: str
    aliases: tuple[str, ...]
    args: str
    summary: str
    # Runs during a turn: yes, no, or only without arguments ("show").
    during_turn: bool | Literal["show"]


COMMANDS: Final[tuple[Command, ...]] = (
    Command("help", (), "[command]", "Show commands and keys", True),
    Command("new", ("clear",), "", "Start a new session (also /clear)", False),
    Command("model", (), "[name]", "Show, list or switch the model (saved)", False),
    Command("backend", (), "[url]", "Switch to another model server (saved)", False),
    Command("exit", ("quit",), "", "Quit (also /quit, ctrl+d)", True),
)

KEYS: Final = (
    "enter send (queues during a turn) · alt+enter, ctrl+j or \\ enter: new line",
    "↑↓ history · ctrl+r search · tab complete",
    "esc stop the turn · ctrl+c clear, stop or quit · ctrl+d quit",
)

_COMMAND: Final = re.compile(r"/([a-z][a-z-]*)(?:\s+(.*))?\Z", re.DOTALL)


def find(name: str) -> Command | None:
    for command in COMMANDS:
        if name == command.name or name in command.aliases:
            return command
    return None


def parse_command(text: str) -> tuple[Command | None, str, str] | None:
    """``(command, name, arguments)`` for a command; None for a message.

    An unknown name gives ``(None, name, arguments)``.
    """
    match = _COMMAND.match(text.strip())
    if match is None:
        return None
    name, arguments = match.group(1), (match.group(2) or "").strip()
    return find(name), name, arguments


def help_text(topic: str = "") -> RenderableType:
    """``/help``: every command and the keys, or one command's details."""
    if topic:
        command = find(topic.lstrip("/"))
        if command is None:
            return Text(
                f"Unknown command /{topic.lstrip('/')}. /help lists the commands.",
                style="error",
            )
        during = {True: "yes", False: "no", "show": "only without arguments"}[
            command.during_turn
        ]
        usage = f"/{command.name} {command.args}".rstrip()
        return Text(f"{usage}\n  {command.summary}\n  During a turn: {during}")
    lines: list[Text] = [Text("Commands", style="bold")]
    for command in COMMANDS:
        usage = f"/{command.name} {command.args}".rstrip()
        line = Text("  " + pad_cells(usage, 20) + "  ")
        line.append(command.summary, style="muted")
        lines.append(line)
    lines.append(Text("Input", style="bold"))
    lines += [Text("  " + keys, style="muted") for keys in KEYS]
    return Group(*lines)


def names(commands: Sequence[Command] = COMMANDS) -> list[str]:
    """Every name a command answers to, aliases included."""
    return [name for command in commands for name in (command.name, *command.aliases)]
