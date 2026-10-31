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

from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import UNICODE, Symbols, in_symbols
from localmelo.support.cli.views import hang


@dataclass(frozen=True)
class Command:
    name: str
    aliases: tuple[str, ...]
    args: str
    summary: str
    # Runs during a turn: yes, no, or only without arguments ("show").
    during_turn: bool | Literal["show"]
    keys: str = ""  # a key that does the same, named after the other spellings

    def line(self, spelling: str = "") -> str:
        """The summary for *spelling* (the name by default), naming the others.

        ``/exit``'s row says ``(also /quit, ctrl+d)``, and ``/quit``'s ``(also /exit, ctrl+d)``.
        """
        spelling = spelling or self.name
        others = [f"/{name}" for name in (self.name, *self.aliases) if name != spelling]
        others += [self.keys] if self.keys else []
        return f"{self.summary} (also {', '.join(others)})" if others else self.summary


# The everyday commands come first: they head the menu, whose first screen is eight
# rows. A summary has at most 60 cells and says its argument where one is needed, so
# the menu, whose name column is as wide as the longest name, cuts none at 80 columns.
COMMANDS: Final[tuple[Command, ...]] = (
    Command("model", (), "[name]", "Pick a model, or /model NAME (saved)", False),
    Command("new", (), "", "Start a new chat; /clear also clears the screen", False),
    Command("status", (), "[all]", "Check the model; /status all shows more", True),
    Command("help", (), "[command]", "Show commands; /help all lists every one", True),
    Command("exit", ("quit",), "", "Quit", True, keys="ctrl+d"),
    Command("undo", (), "", "Forget the last turn (files stay changed)", False),
    Command("copy", (), "[turn]", "Copy a stored answer to the clipboard", True),
    Command("retry", (), "", "Run the last turn again", False),
    Command("clear", (), "", "Start a new chat and clear the screen", False),
    Command("backend", (), "[url]", "Switch to another model server (saved)", False),
    Command("attach", (), "PATH", "Stage a text file: /attach PATH", False),
    Command("attachments", (), "", "List staged text files", False),
    Command("detach", (), "N|all", "Remove a staged file: /detach N or all", False),
    Command("sessions", (), "[filter]", "Pick a chat to resume", False),
    Command("resume", (), "[id|title]", "Resume a chat", False),
    Command("title", (), "[text]", "Show or set the chat title", True),
    Command("usage", (), "", "Show recorded tokens, tools and time", True),
    Command("memory", (), "[window]", "See and change the notes melo keeps", True),
    Command("verbose", (), "[on|off]", "Toggle verbose rendering", True),
    Command("thinking", (), "[on|off]", "Toggle thinking excerpts", True),
    Command("stream", (), "[on|off]", "Toggle live streamed replies", True),
    Command("expand", (), "[turn.call]", "Show a tool call's full output", True),
    Command(
        "approvals",
        (),
        "[ask|auto|deny|forget [tool]]",
        "Show or switch approvals: ask, auto, deny, forget",
        "show",
    ),
    Command("progress", (), "", "Show this turn's progress facts", True),
    Command("browser", (), "[on|off]", "Show or switch web browsing in Safari", "show"),
    Command("chats", (), "", "Pick the main chat or a side chat", False),
    Command("side", (), "[title]", "Start a side chat in this folder", False),
    Command("main", (), "", "Go back to the main chat", False),
    Command("task", (), "TEXT", "Start a task from the main chat: /task TEXT", False),
    Command("mcp", (), "", "Show the MCP servers", True),
)

# What the main chat leaves to the terminal's own chat: the commands that act
# on its Agent or its file (-X cli.chat=local opens that chat).
LOCAL_ONLY: Final = frozenset(
    {
        "model",
        "backend",
        "undo",
        "copy",
        "attach",
        "attachments",
        "detach",
        "sessions",
        "resume",
        "title",
        "usage",
        "approvals",
        "browser",
        "mcp",
    }
)

# What /help shows without "all": three groups of the commands a new user needs.
HELP_GROUPS: Final = (
    ("Conversation", ("new", "retry", "undo", "copy")),
    ("Model", ("model", "status")),
    ("Help", ("help", "exit")),
)
# The same in the main chat.
MAIN_HELP_GROUPS: Final = (
    ("Chats", ("chats", "side", "main")),
    ("Turns", ("task", "retry")),
    ("Model", ("status",)),
    ("Help", ("help", "exit")),
)

KEYS: Final = (
    "enter send (queues during a turn) · alt+enter, ctrl+j or \\ enter: new line",
    "{arrows} history · ctrl+r search · tab complete",
    "esc stop the turn · ctrl+c clear, stop or quit · ctrl+d quit",
    "! command runs it in your shell, not through the model (! alone: details)",
)

_COMMAND: Final = re.compile(r"/([a-z][a-z-]*)(?:\s+(.*))?\Z", re.DOTALL)


def find(name: str) -> Command | None:
    for command in COMMANDS:
        if name == command.name or name in command.aliases:
            return command
    return None


def parse_command(text: str) -> tuple[Command | None, str, str] | None:
    """``(command, name, arguments)`` for a command; None for a message.

    An unknown name gives ``(None, name, arguments)``. A draft that starts
    with a space or a tab is a message, so a message can start like a
    command.
    """
    match = _COMMAND.match(text.rstrip())
    if match is None:
        return None
    name, arguments = match.group(1), (match.group(2) or "").strip()
    return find(name), name, arguments


def help_text(
    topic: str = "",
    symbols: Symbols = UNICODE,
    *,
    everything: bool = False,
    width: int = 80,
    main_chat: bool = False,
) -> RenderableType:
    """``/help``: the everyday commands, ``/help all`` every command and the keys, or one's details.

    *everything* (developer mode) makes the bare ``/help`` the long one. A row too
    long for *width* continues under its description. In the *main_chat*, the
    commands of the terminal's own chat are left out.
    """
    if topic == "all":
        everything, topic = True, ""
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
        return Text(f"{usage}\n  {command.line()}\n  During a turn: {during}")
    groups = list(MAIN_HELP_GROUPS if main_chat else HELP_GROUPS)
    if everything:
        named = {name for _, names in groups for name in names}
        if main_chat:
            named |= LOCAL_ONLY
        groups.append(("More", tuple(c.name for c in COMMANDS if c.name not in named)))
    column = max(cells(f"/{name}") for _, names in groups for name in names) + 2
    lines: list[Text] = []
    for title, names in groups:
        lines.append(Text(title, style="bold"))
        for name in names:
            command = find(name)
            assert command is not None
            row = Text(command.line())
            lines.append(hang(f"  /{name}", row, width, 2 + column))
    keys = [in_symbols(k.format(arrows=symbols.arrows), symbols) for k in KEYS]
    if everything:
        lines.append(Text("Input", style="bold"))
    else:
        keys = [keys[-1], "/help all lists every command and key."]
    lines += [hang("", Text(key), width, 2) for key in keys]
    return Group(*lines)


def names(commands: Sequence[Command] = COMMANDS) -> list[str]:
    """Every name a command answers to, aliases included."""
    return [name for command in commands for name in (command.name, *command.aliases)]
