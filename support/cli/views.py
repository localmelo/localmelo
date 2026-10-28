"""Static outputs of the terminal UI: the backend's description and the banner."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from rich.console import Group, RenderableType
from rich.text import Text

from localmelo.melo.schema import (
    INTERRUPT_LABEL,
    NO_ANSWER_LABEL,
    ToolInfo,
    split_thinking,
)
from localmelo.support.cli.text import (
    cells,
    cut_cells,
    expand_tabs,
    pad_cells,
    wrap_words,
)
from localmelo.support.cli.theme import Symbols, in_symbols
from localmelo.support.sessions.attachments import parse_user_text
from localmelo.support.sessions.display import (
    clean_output,
    home_label,
    resume_note,
    session_age,
    status_text,
)
from localmelo.support.sessions.runner import ResumeInfo
from localmelo.support.sessions.store import SessionInfo, TurnInfo

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
    # The model id can come from a server's listing; never draw its escapes.
    return f"long-term memory · {clean_output(backend.embedding)}"


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


def resume_recap(
    info: SessionInfo,
    turns: Sequence[TurnInfo],
    loaded: ResumeInfo,
    *,
    width: int,
    symbols: Symbols,
) -> Text:
    """A short, plain recap from stored redacted copies, never a live event replay."""
    finished = [t for t in turns if not t.undone and t.status != "running"]
    lines = [
        f'◆ Resumed "{info.title}" · {info.turns} turn{"s" if info.turns != 1 else ""} · last active {session_age(info.updated_at, time.time())}',
    ]
    lines.extend(
        "  " + line
        for line in wrap_words(
            resume_note(loaded.mode, loaded.turns, loaded.checkpoint_turns),
            max(1, width - 2),
        )
    )
    if len(finished) > 3:
        lines.append(f"  … {len(finished) - 3} earlier turns")
    for turn in finished[-3:]:
        typed, attachments = parse_user_text(turn.user_text)
        typed = turn.typed_text or typed
        lines.extend("  › " + text for text in clean_output(typed).splitlines()[:2])
        lines.extend(
            f"    └ attached {a.name} ({a.chars:,} chars)" for a in attachments
        )
        answer = split_thinking(turn.answer)[1]
        mark = symbols.assistant
        if not answer.strip() or answer in (
            INTERRUPT_LABEL,
            NO_ANSWER_LABEL.format(status=turn.status),
        ):
            mark = symbols.stopped if turn.status == "cancelled" else symbols.failed
            answer = status_text(turn.status, turn.cause).title + " (no answer)"
        lines.extend(
            f"  {mark} " + text for text in clean_output(answer).splitlines()[:4]
        )
    return Text(
        "\n".join(
            cut_cells(
                expand_tabs(clean_output(in_symbols(line, symbols))), max(1, width)
            )
            for line in lines
        )
    )
