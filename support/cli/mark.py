"""Melo's face beside the welcome: three rows of half blocks from the logo.

The pixels are not drawn by hand: ``scripts/make_terminal_mark.py`` computes
them by downsampling ``docs/assets/icon_small.png`` and writes the PIXELS
block below (``tests/cli/test_mark.py`` checks that it still matches the
logo). Each terminal cell holds two pixels, the upper as the foreground of
``▀`` and the lower as its background. The mark shows only where it reads:
256 colours or more, Unicode, and 60 columns or more.
"""

from __future__ import annotations

from typing import Final

from rich.style import Style
from rich.text import Text

from localmelo.support.cli.theme import ASCII, Palette, Symbols

# Written by scripts/make_terminal_mark.py: ``.`` empty, ``o`` fur, ``c``
# cream, ``k`` the eyes, nose and mouth.
PIXELS: Final = (
    "...oooo...",
    ".oooooooo.",
    "oookcckooo",
    "cccckkcccc",
    ".cckkkkcc.",
    "..cccccc..",
)
# The mark's own colours (artwork, not roles), on a dark and a light terminal.
COLOURS: Final = {
    "dark": {"o": "#d4893a", "c": "#f3e9d2", "k": "#182836"},
    "light": {"o": "#c17626", "c": "#efe1c2", "k": "#182836"},
}
MIN_COLUMNS: Final = 60
# The mark's cells, and the gap between it and the welcome's rows.
CELLS: Final = len(PIXELS[0])
GAP: Final = 3


def shows(palette: Palette, symbols: Symbols, width: int) -> bool:
    """Whether the mark is drawn: 256 colours or more, Unicode, 60 columns or more.

    ``NO_COLOR`` gives a palette with no colours, so no mark either.
    """
    return (
        palette.depth in ("256", "truecolor")
        and symbols is not ASCII
        and width >= MIN_COLUMNS
    )


def rows(polarity: str) -> list[Text]:
    """The mark's three rows of :data:`CELLS` cells, in *polarity*'s colours."""
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
            elif lower is None or upper == lower:
                row.append("▀" if lower is None else "█", Style(color=upper))
            else:
                row.append("▀", Style(color=upper, bgcolor=lower))
        out.append(row)
    return out
