"""Melo's head beside the welcome: rows of half blocks from the pet's face picture.

The pixels are not drawn by hand: ``scripts/make_terminal_mark.py`` computes
them by downsampling ``support/gateway/static/moves/face-brand.png`` and writes the
PIXELS block below (``tests/cli/test_mark.py`` checks that it still matches
the picture). Each terminal cell holds two pixels, the upper as
the foreground of ``▀`` and the lower as its background (a cell of one colour
is a coloured space, which fills the cell's line gap too). The mark shows
only where it reads: 256 colours or more, Unicode, and 80 columns or more.
"""

from __future__ import annotations

from typing import Final

from rich.style import Style
from rich.text import Text

from localmelo.support.cli.theme import ASCII, Palette, Symbols

# Written by scripts/make_terminal_mark.py: ``.`` empty, ``o`` fur, ``c``
# cream, ``b`` shade, ``k`` the eyes, nose and mouth.
PIXELS: Final = (
    ".cc......cc.",
    ".coco..ococ.",
    ".cocoooococ.",
    ".cooooooooc.",
    "ccooccccoocc",
    "cccckcckcccc",
    "ccccckkccccc",
    "ccccckkccccc",
    ".cccccccccc.",
    ".occcccccco.",
)
# The mark's own colours (artwork, not roles), on a dark and a light terminal.
COLOURS: Final = {
    "dark": {"o": "#f09e51", "c": "#faecdd", "b": "#92583f", "k": "#341e15"},
    "light": {"o": "#d98a3e", "c": "#f3e2cc", "b": "#7e4a33", "k": "#341e15"},
}
# Narrower, the welcome is its rows alone: three rows of half blocks lose
# the ears, and five leave the rows too little room.
MIN_COLUMNS: Final = 80
# The mark's cells, and the gap between it and the welcome's rows.
CELLS: Final = len(PIXELS[0])
GAP: Final = 3


def shows(palette: Palette, symbols: Symbols, width: int) -> bool:
    """Whether the mark is drawn: 256 colours or more, Unicode, 80 columns or more.

    ``NO_COLOR`` gives a palette with no colours, so no mark either.
    """
    return (
        palette.depth in ("256", "truecolor")
        and symbols is not ASCII
        and width >= MIN_COLUMNS
    )


def rows(polarity: str) -> list[Text]:
    """The mark's five rows of :data:`CELLS` cells, in *polarity*'s colours."""
    colours = COLOURS["light" if polarity == "light" else "dark"]
    out: list[Text] = []
    for top_row, bottom_row in zip(PIXELS[::2], PIXELS[1::2], strict=True):
        row = Text(no_wrap=True)
        for top, bottom in zip(top_row, bottom_row, strict=True):
            upper, lower = colours.get(top), colours.get(bottom)
            if upper is None and lower is None:
                row.append(" ")
            elif upper is None:
                row.append("▄", Style(color=lower))
            elif lower is None:
                row.append("▀", Style(color=upper))
            elif upper == lower:
                # A coloured space fills the whole cell, its line gap too.
                row.append(" ", Style(bgcolor=upper))
            else:
                row.append("▀", Style(color=upper, bgcolor=lower))
        out.append(row)
    return out
