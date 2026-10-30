"""Styles and symbols of the terminal UI.

Styles are semantic and use the 16 ANSI colors only, so light and dark
terminal themes both work; ``NO_COLOR`` removes the colors and keeps bold,
dim and reverse. A warning is yellow only on a background ``COLORFGBG``
reports as dark and bold elsewhere, and code blocks use rich's light theme
on a light one. Each symbol has an ASCII stand-in, used when stdout's
encoding is not UTF-8 or ``LOCALMELO_ASCII=1``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from prompt_toolkit.styles import Style
from rich.theme import Theme

ASCII_ENV: Final = "LOCALMELO_ASCII"

# The look of each style, in rich's words and in prompt_toolkit's.
_STYLES: Final[Mapping[str, tuple[str, str]]] = {
    "accent": ("cyan", "ansicyan"),
    "muted": ("dim", "dim"),
    "ok": ("green", "ansigreen"),
    "error": ("red", "ansired"),
    "warn": ("yellow", "ansiyellow"),
    "blocked": ("yellow", "ansiyellow"),
    "tool": ("bold", "bold"),
    "shell": ("magenta", "ansimagenta"),
    "yolo": ("white on red", "ansiwhite bg:ansired"),
}

# Yellow is unreadable on some light backgrounds: elsewhere than a known dark
# one, bold in the default color draws the attention.
_BOLD: Final = {"warn": ("bold", "bold"), "blocked": ("bold", "bold")}

Background = Literal["dark", "light", "unknown"]


def background(environ: Mapping[str, str] | None = None) -> Background:
    """The terminal's background as ``COLORFGBG`` reports it, else ``unknown``.

    The variable is ``FG;BG`` or ``FG;default;BG`` in ANSI color numbers: 0-6
    and 8 are dark, 7 and 9-15 light. Nothing else is read and no terminal is
    queried.
    """
    environ = os.environ if environ is None else environ
    _, separator, last = environ.get("COLORFGBG", "").rpartition(";")
    if not separator or not last.strip().isdigit():
        return "unknown"
    number = int(last)
    if number in (7, *range(9, 16)):
        return "light"
    return "dark" if number <= 8 else "unknown"


def _styles(bg: Background) -> Mapping[str, tuple[str, str]]:
    return _STYLES if bg == "dark" else {**_STYLES, **_BOLD}


def rich_theme(bg: Background) -> Theme:
    return Theme(
        {
            **{name: rich for name, (rich, _) in _styles(bg).items()},
            # Inline code: bold cyan, without the default black background.
            "markdown.code": "bold cyan",
        }
    )


def pt_style(bg: Background) -> Style:
    return Style.from_dict(
        {
            **{name: pt for name, (_, pt) in _styles(bg).items()},
            "rule": "dim",
            "prompt": "ansicyan bold",
            "placeholder": "dim",
            "status": "",
            "hint": "",  # says what to do or why a key did nothing: not dim
            "question": "bold",
        }
    )


def code_theme(bg: Background) -> str:
    """rich's ANSI theme for code blocks: ``ansi_dark``'s bright colors need a dark one."""
    return "ansi_light" if bg == "light" else "ansi_dark"


# Read once: the background does not change under a running process.
BACKGROUND: Final = background()
RICH_THEME: Final = rich_theme(BACKGROUND)
PT_STYLE: Final = pt_style(BACKGROUND)
CODE_THEME: Final = code_theme(BACKGROUND)


@dataclass(frozen=True)
class Symbols:
    """One field per symbol the UI draws."""

    prompt: str
    assistant: str
    ok: str
    failed: str
    blocked: str
    not_run: str
    connector: str
    thought: str
    retry: str
    warn: str
    stopped: str
    session: str
    image: str
    selection: str
    arrows: str
    queued: str
    output_tokens: str  # a format with one {} for the count
    minus: str
    dot: str
    ellipsis: str
    rule: str
    spinner: tuple[str, ...]


UNICODE: Final = Symbols(
    prompt="›",
    assistant="●",
    ok="✓",
    failed="✗",
    blocked="⊘",
    not_run="◌",
    connector="└",
    thought="✻",
    retry="↻",
    warn="⚠",
    stopped="■",
    session="◆",
    image="▪",
    selection="❯",
    arrows="↑↓",
    queued="↳",
    output_tokens="↓ {} tokens",
    minus="−",
    dot="·",
    ellipsis="…",
    rule="─",
    spinner=("·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"),
)

ASCII: Final = Symbols(
    prompt=">",
    assistant="*",
    ok="+",
    failed="x",
    blocked="!",
    not_run="-",
    connector="|",
    thought="*",
    retry="~",
    warn="!",
    stopped="#",
    session="=",
    image="#",
    selection=">",
    arrows="up/down",
    queued=">>",
    output_tokens="{} tokens out",
    minus="-",
    dot="-",
    ellipsis="...",
    rule="-",
    spinner=("-", "\\", "|", "/"),
)


def _ascii_table() -> dict[int, str]:
    table: dict[int, str] = {ord("↑"): "up", ord("↓"): "down"}
    for name in (
        "prompt",
        "assistant",
        "ok",
        "failed",
        "blocked",
        "not_run",
        "connector",
        "thought",
        "retry",
        "warn",
        "stopped",
        "session",
        "image",
        "selection",
        "queued",
        "minus",
        "dot",
        "ellipsis",
        "rule",
    ):
        table[ord(getattr(UNICODE, name))] = getattr(ASCII, name)
    return table


_TO_ASCII: Final = _ascii_table()


def in_symbols(text: str, symbols: Symbols) -> str:
    """*text*, melo's own words, with its symbols in *symbols*' set.

    With :data:`ASCII`, each Unicode symbol becomes its stand-in, and the
    arrows become words: a terminal that is not UTF-8 would show ``?``.
    """
    return text if symbols is not ASCII else text.translate(_TO_ASCII)


def symbols_for(
    encoding: str | None, environ: Mapping[str, str] | None = None
) -> Symbols:
    """:data:`UNICODE`, or :data:`ASCII` for another encoding or ``LOCALMELO_ASCII=1``."""
    environ = os.environ if environ is None else environ
    if environ.get(ASCII_ENV) == "1":
        return ASCII
    name = (encoding or "").lower().replace("-", "").replace("_", "")
    return UNICODE if name == "utf8" else ASCII


def wants_no_color(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``NO_COLOR`` is set to a value that is not empty."""
    environ = os.environ if environ is None else environ
    return bool(environ.get("NO_COLOR"))
