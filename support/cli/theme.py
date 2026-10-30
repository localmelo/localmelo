"""Styles and symbols of the terminal UI.

Styles are roles (accent, muted, ok, error, a tool's kind, ...). A
:class:`Palette` draws each role at the terminal's colour depth and
background: from the role table's colours when the background's polarity is
known at 256 colours or more, else from the 16 named ANSI colours, which the
user's own theme tunes, and with attributes only under ``NO_COLOR``. Fills
(bands, chips, panels) need the terminal's exact background and foreground,
which the interactive UI asks it for at start-up (:mod:`termcolors`). Each
symbol has an ASCII stand-in, used when stdout's encoding is not UTF-8 or
``LOCALMELO_ASCII=1``.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final, Literal, NamedTuple

from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import DynamicStyle, Style
from rich.style import Style as RichStyle
from rich.theme import Theme

from localmelo.support.cli.termcolors import (
    XTERM,
    Depth,
    TerminalColors,
    blend,
    color_depth,
    contrast,
    fit,
    is_dark,
    nearest256,
)

ASCII_ENV: Final = "LOCALMELO_ASCII"

Background = Literal["dark", "light", "unknown"]


def background(environ: Mapping[str, str] | None = None) -> Background:
    """The terminal's background as ``COLORFGBG`` reports it, else ``unknown``.

    The variable is ``FG;BG`` or ``FG;default;BG`` in ANSI color numbers: 0-6
    and 8 are dark, 7 and 9-15 light.
    """
    environ = os.environ if environ is None else environ
    _, separator, last = environ.get("COLORFGBG", "").rpartition(";")
    if not separator or not last.strip().isdigit():
        return "unknown"
    number = int(last)
    if number in (7, *range(9, 16)):
        return "light"
    return "dark" if number <= 8 else "unknown"


# Contrast targets: text, and marks and rules beside text.
TEXT: Final = 4.5
MARK: Final = 3.0
# The typical backgrounds of each polarity; every role keeps its target on each.
TYPICAL: Final[Mapping[str, tuple[str, ...]]] = {
    "dark": ("#000000", "#181a1f", "#1e1e1e", "#282c34", "#002b36"),
    "light": ("#ffffff", "#fafafa", "#fdf6e3"),
}


class Hue(NamedTuple):
    dark: str
    light: str
    dark256: int  # the xterm index kept at 256 colours
    light256: int
    target: float


# The role table: Melo's fur marks the answer, moss is the accent, and each
# tool kind and outcome has its own hue (the terminal's own text is the body).
HUES: Final[Mapping[str, Hue]] = {
    "accent": Hue("#b5cf9f", "#3f5a36", 151, 22, TEXT),
    "answer": Hue("#d98b3b", "#c17626", 172, 166, MARK),
    "heading": Hue("#d98b3b", "#24425f", 172, 17, TEXT),
    "brand": Hue("#d98b3b", "#9a560c", 172, 94, TEXT),
    "muted": Hue("#91978a", "#6d7167", 246, 242, TEXT),
    "rule": Hue("#767d6c", "#82877a", 243, 102, MARK),
    "ok": Hue("#81e7d3", "#037f6b", 116, 29, MARK),
    "ok_text": Hue("#81e7d3", "#037f6b", 116, 23, TEXT),
    "warn": Hue("#e9bc54", "#9b6610", 214, 130, MARK),
    "warn_text": Hue("#e9bc54", "#9b6610", 214, 94, TEXT),
    "error": Hue("#f6a296", "#a21829", 217, 124, TEXT),
    "write": Hue("#ceacf7", "#8252a9", 183, 97, TEXT),
    "run": Hue("#61b1cf", "#015b77", 74, 24, TEXT),
    "network": Hue("#7496e3", "#183d86", 105, 19, TEXT),
    "memory": Hue("#f1cbdf", "#9f5b7f", 225, 96, TEXT),
    "path": Hue("#c9a774", "#80602d", 180, 94, TEXT),
    "code_name": Hue("#9db6cf", "#24425f", 110, 17, TEXT),
}
# Each role in the 16 named colours, on a known dark background and on a
# light or unknown one (where yellow may be unreadable, bold draws the eye),
# and with NO_COLOR.
LOW: Final[Mapping[str, tuple[str, str, str]]] = {
    "accent": ("green", "green", "bold"),
    "answer": ("yellow", "bold", "bold"),
    "heading": ("bold yellow", "bold", "bold underline"),
    "brand": ("bold yellow", "bold", "bold"),
    "muted": ("dim", "dim", "dim"),
    "rule": ("dim", "dim", "dim"),
    "ok": ("green", "green", ""),
    "ok_text": ("green", "green", ""),
    "warn": ("yellow", "bold", "bold"),
    "warn_text": ("yellow", "bold", "bold"),
    "error": ("red", "red", "bold"),
    "write": ("magenta", "magenta", "bold"),
    "run": ("cyan", "blue", "bold"),
    "network": ("blue", "blue", ""),
    "memory": ("magenta", "magenta", ""),
    "path": ("yellow", "", ""),
    "code_name": ("blue", "blue", ""),
}
_BOLD_ROLES: Final = frozenset({"heading", "brand"})
# Fills, blended over the exact background: (colour, opacity) dark and light.
FILLS: Final[Mapping[str, tuple[tuple[str, float], tuple[str, float]]]] = {
    "raised": (("#eef0e9", 0.075), ("#1d1f1b", 0.05)),
    "selection": (("#b5cf9f", 0.14), ("#3f5a36", 0.11)),
    "diff_add": (("#81e7d3", 0.13), ("#037f6b", 0.10)),
    "diff_remove": (("#f6a296", 0.13), ("#a21829", 0.08)),
}
# Code's tokens and the roles that draw them (numbers stay the body's ink:
# orange is the answer's alone).
SYNTAX: Final[Mapping[str, str]] = {
    "keyword": "accent",
    "string": "path",
    "name": "code_name",
}
# The style names the UI uses, and the role that draws each.
NAMES: Final[Mapping[str, str]] = {
    **{role: role for role in HUES},
    "blocked": "warn",
    "kind_read": "muted",
    "kind_write": "write",
    "kind_run": "run",
    "kind_network": "network",
    "kind_memory": "memory",
    "thinking": "muted",
    "intermediate": "muted",
    "file_new": "ok_text",
    "file_mod": "warn_text",
    "file_del": "error",
}


class Colour(NamedTuple):
    """``#rrggbb`` or an ANSI colour name, and the xterm index drawn at 256 colours."""

    value: str
    xterm: int | None = None

    def rich(self) -> str:
        return f"color({self.xterm})" if self.xterm is not None else self.value

    def pt(self) -> str:
        if self.xterm is not None:
            return XTERM[self.xterm]
        return self.value if self.value.startswith("#") else "ansi" + self.value


@dataclass(frozen=True)
class Look:
    """How one role draws: a colour, a fill and attributes."""

    fg: Colour | None = None
    bg: Colour | None = None
    bold: bool = False
    dim: bool = False
    underline: bool = False
    italic: bool = False

    @classmethod
    def of(cls, words: str) -> Look:
        """A look from rich-style words: ``bold yellow``, ``dim``, ``bold underline``."""
        parts = words.split()
        flags = ("bold", "dim", "underline", "italic")
        colours = [word for word in parts if word not in flags]
        return cls(
            Colour(colours[0]) if colours else None,
            bold="bold" in parts,
            dim="dim" in parts,
            underline="underline" in parts,
        )

    def rich(self) -> RichStyle:
        return RichStyle(
            color=self.fg.rich() if self.fg else None,
            bgcolor=self.bg.rich() if self.bg else None,
            bold=self.bold or None,
            dim=self.dim or None,
            underline=self.underline or None,
            italic=self.italic or None,
        )

    def pt(self) -> str:
        words = [self.fg.pt()] if self.fg else []
        words += [f"bg:{self.bg.pt()}"] if self.bg else []
        flags = ("bold", "dim", "underline", "italic")
        return " ".join(words + [flag for flag in flags if getattr(self, flag)])


@dataclass(frozen=True)
class Palette:
    """The roles drawn at a colour depth on a background (exact when the terminal said)."""

    depth: Depth = "16"
    polarity: Background = "unknown"
    background: str | None = None
    foreground: str | None = None

    @property
    def hues(self) -> bool:
        """Whether roles draw from the role table rather than the 16 named colours."""
        return self.depth in ("256", "truecolor") and self.polarity != "unknown"

    @functools.cached_property
    def fills(self) -> Mapping[str, Colour]:
        """Each fill over the exact background; none unless the terminal's text keeps 4.5:1 on all."""
        if not (self.hues and self.background and self.foreground):
            return {}
        side = 0 if self.polarity == "dark" else 1
        blended = {
            name: blend(pair[side][0], self.background, pair[side][1])
            for name, pair in FILLS.items()
        }
        if any(contrast(self.foreground, fill) < TEXT for fill in blended.values()):
            return {}
        return {name: self._colour(fill) for name, fill in blended.items()}

    def _colour(
        self, value: str, grounds: tuple[str, ...] = (), target: float = 0.0
    ) -> Colour:
        if self.depth != "256":
            return Colour(value)
        return Colour(value, nearest256(value, grounds, target))

    def look(self, name: str) -> Look:
        """How the style *name* (a role, or a name of :data:`NAMES`) draws here."""
        return self._looks[NAMES.get(name, name)]

    @functools.cached_property
    def _looks(self) -> Mapping[str, Look]:
        looks: dict[str, Look] = {}
        for role, hue in HUES.items():
            dark, light, none = LOW[role]
            if self.depth == "none":
                looks[role] = Look.of(none)
            elif not self.hues:
                looks[role] = Look.of(dark if self.polarity == "dark" else light)
            else:
                looks[role] = Look(self._hue(hue), bold=role in _BOLD_ROLES)
        return looks

    def _hue(self, hue: Hue) -> Colour:
        """A role's colour; an unusual exact background moves it until it keeps its target."""
        dark = self.polarity == "dark"
        value, index = (hue.dark, hue.dark256) if dark else (hue.light, hue.light256)
        if self.background is not None:
            moved = fit(value, [self.background], hue.target)
            if moved != value:
                return self._colour(moved, (self.background,), hue.target)
        return Colour(value, index if self.depth == "256" else None)

    def on_raised(self, name: str, fill: str = "raised") -> Look:
        """*name*'s look on the *fill*: its colour moved, hue kept, until it reads at 4.5:1."""
        look, raised = self.look(name), self.fills.get(fill)
        fg = look.fg
        if raised is None or fg is None or not fg.value.startswith("#"):
            return look
        painted = XTERM[raised.xterm] if raised.xterm is not None else raised.value
        moved = fit(fg.value, [painted], TEXT)
        if self.depth == "256":
            drawn = XTERM[fg.xterm] if fg.xterm is not None else moved
            if contrast(drawn, painted) < TEXT:
                fg = Colour(moved, nearest256(moved, [painted], TEXT))
        else:
            fg = Colour(moved)
        return replace(look, fg=fg)

    @functools.cached_property
    def rich_theme(self) -> Theme:
        raised = self.fills.get("raised")
        fill = RichStyle(bgcolor=raised.rich()) if raised else RichStyle()
        answer, rule = self.look("answer").rich(), self.look("rule").rich()
        accent, muted = self.look("accent").rich(), self.look("muted").rich()
        return Theme(
            {
                **{name: self.look(name).rich() for name in NAMES},
                "shell": "magenta",
                "yolo": "white on red",
                # The user's band, inline-code chips and code panels; with no
                # fill, the band is gone, inline code is bold and code has a gutter.
                "raised": fill,
                "user": self.on_raised("accent").rich(),
                "markdown.code": fill if raised else "bold",
                # Code: keywords moss, strings gold, names navy, comments muted.
                **{
                    f"syntax.{part}": self.on_raised(role).rich()
                    for part, role in SYNTAX.items()
                },
                "syntax.comment": self.on_raised("muted").rich()
                + RichStyle(italic=True),
                # Markdown in any message; an answer adds its heading and bullets.
                "markdown.heading": "bold",
                "markdown.item.bullet": "bold",
                "markdown.item.number": "",
                "markdown.answer.heading": self.look("heading").rich(),
                "markdown.answer.bullet": answer,
                "markdown.link": accent + RichStyle(underline=True),
                "markdown.link_url": accent,
                "markdown.block_quote": muted,
                "markdown.hr": rule,
                "markdown.table.border": rule,
                "markdown.table.header": "bold",
                **{f"markdown.h{level}": "none" for level in range(1, 8)},
                # A slash command named in what melo says (Start it, then /retry).
                "command": accent + RichStyle(bold=True),
            }
        )

    def band(self, fill: str, role: str) -> str:
        """A row on the *fill* band in *role*'s colour; bold *role* without fills."""
        colour = self.fills.get(fill)
        if self.depth == "none":
            return "reverse" if fill == "selection" else ""
        look = self.on_raised(role, fill).pt()
        if colour is None:
            return f"{look} bold" if fill == "selection" else look
        return f"{look} bg:{colour.pt()}" + (" bold" if fill == "selection" else "")

    @functools.cached_property
    def pt_style(self) -> Style:
        accent, muted = self.look("accent"), self.look("muted")
        return Style.from_dict(
            {
                **{name: self.look(name).pt() for name in NAMES},
                "tool": "bold",
                "shell": "ansimagenta",
                "yolo": "ansiwhite bg:ansired",
                "prompt": f"{accent.pt()} bold".strip(),
                "placeholder": muted.pt(),
                "status": muted.pt(),
                "hint": "",  # says what to do or why a key did nothing: not dim
                "question": "bold",
                # Lists and panels: the chosen row's band, a frame, its title,
                # the keys a footer names, and the diff's rows.
                "chosen": self.band("selection", "accent"),
                "diff_add": self.band("diff_add", "ok_text"),
                "diff_remove": self.band("diff_remove", "error"),
                "frame": self.look("warn").pt(),
                "title": "bold",
                "key": "",
                "command": f"{accent.pt()} bold".strip(),
                # The status bar: the model bold, the folder a path, zones apart.
                "status_model": "bold",
                "status_path": self.look("path").pt(),
                "status_sep": self.look("rule").pt(),
                "status_run": "",
                # The bar's usage and work (#282): a figure in the body's ink,
                # a state by kind (cache good, filling, held work, running work).
                "status_figure": "",
                "status_ok": self.look("ok_text").pt(),
                "status_warn": self.look("warn_text").pt(),
                "status_error": self.look("error").pt(),
                "status_live": accent.pt(),
                # A key's words beside it: muted, but never dim (dim reads at
                # 2.4:1 on a light terminal), so plain at 16 colours.
                "faint": muted.pt() if self.hues else "",
                # The completion list. prompt_toolkit applies a more precise
                # class after a less precise one, so each of its own classes is
                # reset here: no block, names in the body's ink, descriptions
                # muted, the chosen row as in a picker, a slim bar in the rule
                # colour.
                "completion-menu": "noinherit",
                "completion-menu.completion.current": (
                    f"noinherit {self.band('selection', 'accent')}"
                ),
                "completion-menu.meta.completion": f"noinherit {muted.pt()}",
                "completion-menu.meta.completion.current": (
                    f"noinherit {self.band('selection', 'muted')} nobold"
                ),
                "completion-menu.multi-column-meta": f"noinherit {muted.pt()}",
                "scrollbar.background": "noinherit",
                "scrollbar.button": f"noinherit {self.look('rule').pt()}",
            }
        )

    @property
    def rich_system(self) -> Literal["standard", "256", "truecolor"]:
        if self.depth == "truecolor" or self.depth == "256":
            return self.depth
        return "standard"

    @property
    def pt_depth(self) -> ColorDepth:
        return {
            "none": ColorDepth.DEPTH_1_BIT,
            "16": ColorDepth.DEPTH_4_BIT,
            "256": ColorDepth.DEPTH_8_BIT,
            "truecolor": ColorDepth.DEPTH_24_BIT,
        }[self.depth]


def detect(
    *,
    colors: str = "auto",
    bg: str = "auto",
    environ: Mapping[str, str] | None = None,
    ask: Callable[[], TerminalColors | None] | None = None,
) -> Palette:
    """The palette for this terminal; the first source of the background that answers wins.

    *colors* and *bg* are the ``[cli] colors`` and ``[cli] background``
    settings. A background setting gives the polarity alone; else *ask* (the
    terminal's own answer, interactive UI only) gives the exact background
    and foreground; else ``COLORFGBG`` gives the polarity.
    """
    environ = os.environ if environ is None else environ
    depth = color_depth(environ, colors)
    if depth == "none":
        return Palette("none")
    if bg in ("dark", "light"):
        return Palette(depth, bg)  # type: ignore[arg-type]
    told = ask() if ask is not None else None
    if told is not None and told.background is not None:
        polarity: Background = "dark" if is_dark(told.background) else "light"
        return Palette(depth, polarity, told.background, told.foreground)
    return Palette(depth, background(environ))


_active = Palette(color_depth(os.environ), background())


def active() -> Palette:
    """The palette the UI draws with: the environment's until :func:`activate`."""
    return _active


def activate(palette: Palette) -> None:
    """Draw with *palette* from now on (once, at start-up)."""
    global _active
    _active = palette


# For prompt_toolkit: follows the active palette.
PT_STYLE: Final = DynamicStyle(lambda: _active.pt_style)


def pt_color_depth(no_color: bool) -> ColorDepth:
    """prompt_toolkit's colour depth: one bit under ``NO_COLOR``, else the palette's."""
    return ColorDepth.DEPTH_1_BIT if no_color else _active.pt_depth


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
    # A tool's kind, by its permission tier (memory is reserved).
    kind_read: str
    kind_write: str
    kind_run: str
    kind_network: str
    kind_memory: str
    # A changed file's state: new, modified (deleted is reserved: minus).
    file_new: str
    file_modified: str
    gutter: str  # beside code when there is no panel
    bar: str  # a scrollbar's thumb
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
    kind_read="→",
    kind_write="±",
    kind_run="$",
    kind_network="@",
    kind_memory="◇",
    file_new="+",
    file_modified="~",
    gutter="│",
    bar="┃",
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
    kind_read=">",
    kind_write="~",
    kind_run="$",
    kind_network="@",
    kind_memory="&",
    file_new="+",
    file_modified="~",
    gutter="|",
    bar="|",
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
        "kind_read",
        "kind_write",
        "kind_memory",
        "gutter",
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
