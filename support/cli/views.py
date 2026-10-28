"""Static outputs of the terminal UI: the backend's description and the banner."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from rich.console import Group, RenderableType
from rich.text import Text

from localmelo.melo.schema import ToolInfo
from localmelo.support.cli.text import cells, pad_cells, wrap_words
from localmelo.support.cli.theme import Symbols
from localmelo.support.sessions.display import clean_output, home_label

# Narrower than this, the banner's rows print without the box.
BOX_MIN_COLUMNS: Final = 50
BOX_MAX_COLUMNS: Final = 80
_LABEL_CELLS: Final = 11

LOCAL_BACKENDS: Final = frozenset({"mlc", "ollama", "vllm", "sglang"})

# What each source of the backend reads in the banner's second model row.
SOURCE_TEXT: Final = {
    "--base-url": "from --base-url (not saved)",
    "found on this machine": "found on this machine",
    "saved": "saved as the default",
    "unsaved": "not saved; melo asks again next time",
}


@dataclass(frozen=True)
class BackendInfo:
    """The model a session talks to, and where it came from.

    *key* is the config's backend key, or ``openai-compatible`` for a URL
    that is not the config's; *source* is ``config`` or a key of
    :data:`SOURCE_TEXT`. *listed* is the number of models the server
    listed when melo chose the first of them, else 0. *key_env* names a
    cloud vendor's key variable.
    """

    key: str
    label: str
    model: str
    base_url: str
    embedding: str = ""
    source: str = "config"
    listed: int = 0
    key_env: str = ""

    @property
    def local(self) -> bool:
        """A local server or a URL: one that is listed and sent no key."""
        return self.key in LOCAL_BACKENDS or self.key == "openai-compatible"


def tools_text(tools: Sequence[ToolInfo]) -> str:
    """``4 built-in (3 ask first)``, with the other tools after it."""
    builtin = [tool for tool in tools if tool.source == "builtin"]
    asking = sum(tool.requires_approval for tool in builtin)
    text = (
        f"{len(builtin)} built-in ({asking} ask first)"
        if asking
        else (f"{len(builtin)} built-in")
    )
    others = len(tools) - len(builtin)
    return f"{text} · {others} other" if others else text


def memory_text(backend: BackendInfo, *, memory: bool) -> str:
    if not memory:
        return "long-term memory off (--memory off)"
    if not backend.embedding:
        if backend.key == "openai-compatible":
            return "no long-term memory · --embedding-base-url adds one"
        return "no long-term memory · melo --reconfigure adds one"
    return f"long-term memory · {backend.embedding}"


def model_rows(backend: BackendInfo) -> list[str]:
    """The banner's model row, and a second one when the model is not the config's."""
    where = clean_output(backend.base_url or backend.label)
    first = " · ".join(
        clean_output(part)
        for part in (backend.model, backend.label, backend.base_url)
        if part
    )
    if backend.source == "config":
        return [first or where]
    second = SOURCE_TEXT.get(backend.source, backend.source)
    if backend.listed > 1:
        second += f" · first of {backend.listed}; /model lists them"
    return [first, second]


def workspace_warning(workspace: str, symbols: Symbols) -> str | None:
    """The warning for a workspace that is the home folder or ``/``."""
    home = os.path.realpath(os.path.expanduser("~"))
    if workspace in (home, os.sep):
        where = "~" if workspace == home else "/"
        return (
            f"{symbols.warn} The workspace is {where}: file tools can reach "
            "every file in it."
        )
    return None


def banner(
    *,
    version: str,
    backend: BackendInfo,
    workspace: str,
    tools: Sequence[ToolInfo],
    memory: bool,
    yolo: bool,
    width: int,
    symbols: Symbols,
) -> RenderableType:
    """The startup banner: a box of rows, its warnings and the key hints."""
    rows: list[tuple[str, str, str]] = []
    for number, text in enumerate(model_rows(backend)):
        rows.append(
            ("model" if number == 0 else "", text, "" if number == 0 else "muted")
        )
    rows += [
        ("workspace", clean_output(home_label(workspace)), ""),
        ("tools", tools_text(tools), ""),
        ("memory", memory_text(backend, memory=memory), ""),
        ("session", "new", ""),
    ]
    # The rows join their parts with " · ": the symbol set's dot.
    rows = [
        (label, text.replace(" · ", f" {symbols.dot} "), style)
        for label, text, style in rows
    ]
    lines: list[Text] = []
    title = f" melo {version} "
    if width >= BOX_MIN_COLUMNS and symbols.rule == "─":
        inner = min(width, BOX_MAX_COLUMNS) - 2
        lines.append(Text("╭─" + title + "─" * max(0, inner - 1 - cells(title)) + "╮"))
        room = inner - 2 - _LABEL_CELLS
        for label, text, style in rows:
            # A long value wraps under itself, inside the box.
            for number, value in enumerate(wrap_words(text, room)):
                line = Text(
                    "│ " + pad_cells(label if number == 0 else "", _LABEL_CELLS)
                )
                line.append(value, style=style or None)
                line.append(" " * (room - cells(value)) + " │")
                lines.append(line)
        lines.append(Text("╰" + "─" * inner + "╯"))
    else:
        lines.append(Text(title.strip(), style="bold"))
        room = max(8, width - _LABEL_CELLS)
        for label, text, style in rows:
            for number, value in enumerate(wrap_words(text, room)):
                line = Text(pad_cells(label if number == 0 else "", _LABEL_CELLS))
                line.append(value, style=style or None)
                lines.append(line)
    if yolo:
        lines.append(
            Text(
                f"{symbols.warn} YOLO: tools run without asking.",
                style="warn",
            )
        )
    warning = workspace_warning(workspace, symbols)
    if warning:
        lines.append(Text(warning, style="warn"))
    dot = symbols.dot
    lines.append(Text(f"  / commands {dot} esc stop {dot} ctrl+d quit", style="muted"))
    lines.append(Text(""))
    return Group(*lines)
