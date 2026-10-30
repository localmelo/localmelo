"""Melo's face beside the welcome: computed from the logo, shown only where it reads."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from rich.console import Console

from localmelo.support.cli import mark
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE, Palette
from localmelo.support.cli.views import BackendInfo, welcome

ROOT = Path(__file__).resolve().parents[2]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_terminal_mark", ROOT / "scripts/make_terminal_mark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pixels_are_the_logo_downsampled() -> None:
    # Run scripts/make_terminal_mark.py when the logo changes.
    assert _script().pixels() == mark.PIXELS


def test_the_png_reader_matches_a_known_pixel() -> None:
    image = _script().read_png(ROOT / "docs/assets/icon_small.png")
    assert (len(image), len(image[0])) == (64, 64)
    # The page around the round logo is its paper colour, a warm white.
    red, green, blue = image[0][0]
    assert min(red, green, blue) > 220 and red >= blue


def test_the_mark_is_three_rows_of_its_cells() -> None:
    for polarity in ("dark", "light", "unknown"):
        rows = mark.rows(polarity)
        assert len(rows) == 3
        assert all(row.cell_len == mark.CELLS for row in rows)
    # Each cell holds two pixels: an empty pair is a space.
    assert mark.rows("dark")[0].plain[0] == " "


@pytest.mark.parametrize(
    ("palette", "symbols", "width", "shown"),
    [
        (Palette("truecolor", "dark"), UNICODE, 60, True),
        (Palette("256", "light"), UNICODE, 80, True),
        (Palette("truecolor", "dark"), UNICODE, 59, False),  # under 60 columns
        (Palette("16", "dark"), UNICODE, 80, False),  # 16 colours
        (Palette("none"), UNICODE, 80, False),  # NO_COLOR
        (Palette("truecolor", "dark"), ASCII, 80, False),  # ASCII
    ],
)
def test_the_mark_shows_only_where_it_reads(
    palette: Palette, symbols: object, width: int, shown: bool
) -> None:
    assert mark.shows(palette, symbols, width) is shown  # type: ignore[arg-type]


def _welcome(width: int, palette: Palette | None) -> list[str]:
    out = RecordingOut(width)
    out.print(
        welcome(
            version="0.1.0",
            backend=BackendInfo(
                "ollama",
                "Ollama",
                "qwen3-8b",
                "http://127.0.0.1:9/v1",
                source="saved",
            ),
            workspace="/tmp/demo-project",
            yolo=False,
            width=width,
            symbols=UNICODE,
            palette=palette,
        )
    )
    return out.text().split("\n")


def test_the_rows_stand_beside_the_mark_and_keep_the_width() -> None:
    lines = [line for line in _welcome(60, Palette("truecolor", "dark")) if line]
    offset = mark.CELLS + mark.GAP
    assert lines[0][offset:].startswith("melo 0.1.0")
    assert lines[1][offset:] == "/tmp/demo-project"
    assert lines[2][offset:].startswith("Try: What is this project?")
    assert all(cells(line) <= 60 for line in lines)
    # The rows were fit to what the mark leaves: the commands part went first.
    assert "/ commands" not in lines[2]


def test_no_palette_or_a_narrow_terminal_gives_the_rows_alone() -> None:
    for width, palette in ((80, None), (59, Palette("truecolor", "dark"))):
        lines = [line for line in _welcome(width, palette) if line]
        assert lines[0].startswith("melo 0.1.0"), (width, lines)


def test_the_mark_paints_its_own_colours() -> None:
    console = Console(color_system="truecolor", force_terminal=True, width=80)
    with console.capture() as captured:
        for row in mark.rows("dark"):
            console.print(row)
    ansi = captured.get()
    assert "38;2;212;137;58" in ansi  # the fur, #d4893a
    assert "48;2;243;233;210" in ansi or "38;2;243;233;210" in ansi  # the cream
