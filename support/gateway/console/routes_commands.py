"""``GET /api/commands``: the terminal's slash commands, for the console's "/" menu.

The list is the terminal's own table (``support/cli/commands.py``), so the
two never drift: a command added there has no page action until
:data:`PAGE_ACTIONS` names one, and a test fails until it does. Each
command comes with its words in both languages (the English is the
terminal's summary, the Chinese is ``commands.<name>.help`` in the catalog),
what the page does with it (``action``) and whether it runs during a turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from localmelo.support.gateway.i18n import t_in

if TYPE_CHECKING:
    from . import Console

# What the page does for each command: one id per behaviour, which
# ``static/slash-actions.mjs`` answers. A command added to the terminal
# (``/answer`` of the task questions, say) needs its line here.
PAGE_ACTIONS: Final[dict[str, str]] = {
    "model": "model",
    "backend": "backend",
    "new": "new-chat",
    "clear": "new-chat",
    "side": "new-chat",
    "status": "status",
    "usage": "usage",
    "progress": "progress",
    "mcp": "mcp",
    "help": "help",
    "exit": "exit",
    "undo": "undo",
    "retry": "retry",
    "copy": "copy",
    "title": "title",
    "memory": "memory",
    "approvals": "approvals",
    "browser": "browser",
    "attach": "attach",
    "attachments": "attachments",
    "detach": "detach",
    "verbose": "view",
    "thinking": "view",
    "stream": "view",
    "expand": "expand",
    "chats": "chats",
    "sessions": "chats",
    "resume": "resume",
    "main": "main",
    "task": "send",
}


def commands_json() -> dict[str, Any]:
    """The whole ``GET /api/commands`` body."""
    # Loaded when first asked for: the terminal's libraries stay out of a gateway start.
    from localmelo.support.cli import commands

    everyday: list[str] = []
    for _, names in (*commands.MAIN_HELP_GROUPS, *commands.HELP_GROUPS):
        everyday += [name for name in names if name not in everyday]
    return {
        "commands": [
            {
                "name": command.name,
                "aliases": list(command.aliases),
                "args": command.args,
                "help": {
                    "en": command.summary,
                    "zh-Hans": t_in("zh-Hans", f"commands.{command.name}.help"),
                },
                "during_turn": command.during_turn,
                "keys": command.keys,
                "action": PAGE_ACTIONS.get(command.name, ""),
            }
            for command in commands.COMMANDS
        ],
        "everyday": everyday,
    }


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the commands route on *app*."""

    @app.get("/api/commands")
    async def api_commands() -> JSONResponse:
        return JSONResponse(content=commands_json())


__all__ = ["PAGE_ACTIONS", "add_routes", "commands_json"]
