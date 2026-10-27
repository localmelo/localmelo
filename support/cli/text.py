"""Text measures of the terminal UI, in terminal cells.

A CJK or fullwidth character takes two cells. Every measure and cut here
uses prompt_toolkit's ``get_cwidth`` (wcwidth), never ``len()``, and a
wide character that does not fit whole is left out, never split.
"""

from __future__ import annotations

import unicodedata
from typing import Final

from prompt_toolkit.layout.screen import Char
from prompt_toolkit.utils import get_cwidth

# Rows the input box grows to before it scrolls.
INPUT_MAX_ROWS: Final = 10

TAB_CELLS: Final = 4

# Characters that reorder or hide text, shown escaped in arguments: every
# format character (bidi controls, zero-width ones, tags), the line and
# paragraph separators, and the invisible fillers.
_HIDING_CATEGORIES: Final = frozenset({"Cf", "Zl", "Zp"})
_FILLERS: Final = frozenset({0x034F, 0x115F, 0x1160, 0x3164, 0xFFA0})

_NAMED_ESCAPES: Final = {"\r": "\\r", "\t": "\\t", "\x1b": "\\x1b"}


def cells(text: str) -> int:
    """The terminal cells *text* takes."""
    return get_cwidth(text)


def take_cells(text: str, width: int) -> str:
    """The longest start of *text* that fits *width* cells."""
    out: list[str] = []
    used = 0
    for char in text:
        size = get_cwidth(char)
        if used + size > width:
            break
        out.append(char)
        used += size
    return "".join(out)


def cut_cells(text: str, width: int, ellipsis: str = "…") -> str:
    """*text* cut to *width* cells, ending with *ellipsis* when it was cut."""
    if width <= 0:
        return ""
    if cells(text) <= width:
        return text
    room = width - cells(ellipsis)
    if room < 0:
        return take_cells(ellipsis, width)
    return take_cells(text, room) + ellipsis


def cut_middle(text: str, width: int, ellipsis: str = "…") -> str:
    """*text*, a path, cut from the middle so its last part stays: ``~/…/project``."""
    if cells(text) <= width:
        return text
    head, sep, last = text.rpartition("/")
    tail = sep + last
    room = width - cells(ellipsis) - cells(tail)
    if not sep or room < 1:
        return cut_cells(text, width, ellipsis)
    return take_cells(head, room) + ellipsis + tail


def wrap_cells(line: str, width: int) -> list[str]:
    """*line* in rows of at most *width* cells, every character kept.

    No whitespace is collapsed or dropped, unlike ``textwrap``: what a
    command will run must show whole.
    """
    rows: list[str] = []
    while line:
        row = take_cells(line, max(width, 2))
        if not row:  # a character wider than the row: it takes one alone
            row = line[0]
        rows.append(row)
        line = line[len(row) :]
    return rows or [""]


def wrap_words(line: str, width: int) -> list[str]:
    """*line* in rows of at most *width* cells, broken after a space where one fits.

    Every character is kept, as in :func:`wrap_cells`; a word longer than
    a row is broken by cells.
    """
    rows: list[str] = []
    while cells(line) > width:
        row = take_cells(line, max(width, 2)) or line[0]
        space = row.rfind(" ") + 1
        if 0 < space < len(row) and line[len(row)] != " ":
            row = row[:space]
        rows.append(row)
        line = line[len(row) :]
    return rows + [line] if line or not rows else rows


def pad_cells(text: str, width: int) -> str:
    """*text* with spaces after it up to *width* cells."""
    return text + " " * max(0, width - cells(text))


def expand_tabs(text: str) -> str:
    """*text* with each tab expanded to the next stop of :data:`TAB_CELLS`."""
    return text.expandtabs(TAB_CELLS)


def shown(text: str) -> str:
    """*text* with each control character spelled as prompt_toolkit draws it.

    A pasted ``\\x1a`` reads ``^Z`` and U+0085 ``<85>``, so the cells
    counted for a line are the cells it takes on the screen.
    """
    mappings = Char.display_mappings
    return "".join(
        mappings.get(char, char) if char != "\xa0" else char for char in text
    )


def drawn_cells(char: str) -> int:
    """The cells prompt_toolkit draws *char* in: a tab is ``^I``, U+0085 ``<85>``."""
    shown = Char.display_mappings.get(char)
    return get_cwidth(char if shown is None else shown)


def input_rows(
    text: str,
    prompt_cells: int,
    prefix_cells: int,
    columns: int,
    *,
    limit: int = INPUT_MAX_ROWS,
) -> int:
    """The rows the input box needs for *text*, at most *limit*.

    The draft is walked as prompt_toolkit draws it: the prompt takes
    *prompt_cells* on the first row, and every other row, a new line or a
    wrapped one, starts with *prefix_cells*; a control character takes the
    cells of how it is drawn (``^I``); a character that does not fit
    the rest of a row, a wide one included, starts the next row; and each
    line ends with one cell for the cursor. prompt_toolkit's own count
    divides a line's width by the columns, which misses the cell a wide
    character leaves empty at a row's end.
    """
    if columns <= prefix_cells + 1:
        return limit
    rows = 0
    for number, line in enumerate(text.split("\n")):
        x = prefix_cells
        rows += 1
        if number == 0:
            # A prompt wider than the terminal wraps too; one that fills
            # its last row exactly puts the draft on the next.
            full = (max(prompt_cells, 1) - 1) // columns
            rows += full
            x = prompt_cells - full * columns
        for char in line + " ":
            size = drawn_cells(char)
            if x + size > columns:
                rows += 1
                x = prefix_cells
            x += size
        if rows >= limit:
            return limit
    return rows


def _escaped(char: str) -> str | None:
    code = ord(char)
    if char in _NAMED_ESCAPES:
        return _NAMED_ESCAPES[char]
    if char != "\n" and (code < 0x20 or 0x7F <= code <= 0x9F):
        return f"\\x{code:02x}"
    if code in _FILLERS or unicodedata.category(char) in _HIDING_CATEGORIES:
        return f"\\u{{{code:04x}}}"
    return None


def escape_arguments(text: str) -> tuple[str, bool]:
    """*text* with every control and hiding character shown as an escape.

    Nothing is dropped: C0 and C1 controls but ``\\n`` read ``\\r``,
    ``\\t``, ``\\x1b`` or ``\\xNN``, and the characters that reorder or
    hide text (format characters, Unicode's tags among them, the line and
    paragraph separators, and the invisible fillers) read ``\\u{...}``. The
    second value says whether anything was escaped. A command that runs
    must be shown whole: ``sh #\\rls`` would otherwise read ``ls``.
    """
    out: list[str] = []
    changed = False
    for char in text:
        escaped = _escaped(char)
        if escaped is None:
            out.append(char)
        else:
            out.append(escaped)
            changed = True
    return "".join(out), changed
