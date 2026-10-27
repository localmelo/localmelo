"""Styles and symbols of the terminal UI.

Styles are semantic and use the 16 ANSI colors only, so light and dark
terminal themes both work; ``NO_COLOR`` removes the colors and keeps bold,
dim and reverse. Each symbol has an ASCII stand-in, used when stdout's
encoding is not UTF-8 or ``LOCALMELO_ASCII=1``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

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

RICH_THEME: Final = Theme(
    {
        **{name: rich for name, (rich, _) in _STYLES.items()},
        # Inline code: bold cyan, without the default black background.
        "markdown.code": "bold cyan",
    }
)

PT_STYLE: Final = Style.from_dict(
    {
        **{name: pt for name, (_, pt) in _STYLES.items()},
        "rule": "dim",
        "prompt": "ansicyan bold",
        "placeholder": "dim",
        "status": "",
        "hint": "dim",
        "question": "bold",
    }
)


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
