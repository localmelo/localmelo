"""Melo's head beside the welcome: computed from the pet's picture, shown where it reads."""

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


def test_the_pixels_are_the_pet_picture_downsampled() -> None:
    # Run scripts/make_terminal_mark.py when the picture changes.
    assert _script().pixels() == mark.PIXELS


def test_the_head_has_two_upright_ears_eyes_and_a_nose() -> None:
    top, inner = mark.PIXELS[0], mark.PIXELS[1]
    # Two ears, apart, each with its inner ear below; a gap between them.
    assert top.strip(".").count("..") >= 1 and top == top[::-1]
    assert inner.count("b") == 2
    dark = [row for row in mark.PIXELS if "k" in row]
    assert len(dark) >= 2  # the eyes, then the nose under them
    assert all(row == row[::-1] for row in mark.PIXELS)


def test_the_png_reader_reads_rgb_and_alpha() -> None:
    image = _script().read_png(ROOT / "docs/assets/icon_small.png")
    assert (len(image), len(image[0])) == (64, 64)
    # An RGB picture is opaque; the logo's page is a warm white.
    red, green, blue, alpha = image[0][0]
    assert min(red, green, blue) > 220 and red >= blue and alpha == 255
    face = _script().read_png(ROOT / "macos/art/akita/face/idle@2x.png")
    assert (len(face), len(face[0])) == (96, 96)
    assert face[0][0][3] == 0  # the face's corner is transparent


def test_the_mark_is_five_rows_of_its_cells() -> None:
    for polarity in ("dark", "light", "unknown"):
        rows = mark.rows(polarity)
        assert len(rows) == len(mark.PIXELS) // 2 == 5
        assert all(row.cell_len == mark.CELLS for row in rows)
    # Each cell holds two pixels: an empty pair is a space with no colour,
    # a pair of one colour a space on that colour.
    first = mark.rows("dark")[0]
    assert first.plain[0] == " " and not first.get_style_at_offset(Console(), 0)
    middle = mark.rows("dark")[1]
    style = middle.get_style_at_offset(Console(), 5)
    assert middle.plain[5] == " " and style.bgcolor is not None
    assert style.bgcolor.name == "#d4893a"


@pytest.mark.parametrize(
    ("palette", "symbols", "width", "shown"),
    [
        (Palette("truecolor", "dark"), UNICODE, 80, True),
        (Palette("256", "light"), UNICODE, 120, True),
        (Palette("truecolor", "dark"), UNICODE, 79, False),  # under 80 columns
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


def test_the_rows_stand_beside_the_middle_of_the_head_and_keep_the_width() -> None:
    lines = _welcome(80, Palette("truecolor", "dark"))[1:6]  # a blank row first
    offset = mark.CELLS + mark.GAP
    assert lines[0][offset:] == "" and lines[4][offset:] == ""
    assert lines[1][offset:].startswith("melo 0.1.0")
    assert lines[2][offset:] == "/tmp/demo-project"
    assert lines[3][offset:].startswith("Try: What is this project?")
    assert all(cells(line) <= 80 for line in lines)


def test_no_palette_or_a_narrower_terminal_gives_the_rows_alone() -> None:
    for width, palette in ((80, None), (79, Palette("truecolor", "dark"))):
        lines = [line for line in _welcome(width, palette) if line]
        assert lines[0].startswith("melo 0.1.0"), (width, lines)
        assert len(lines) == 3


def test_the_mark_paints_its_own_colours() -> None:
    console = Console(color_system="truecolor", force_terminal=True, width=80)
    with console.capture() as captured:
        for row in mark.rows("dark"):
            console.print(row)
    ansi = captured.get()
    assert "2;212;137;58" in ansi  # the fur, #d4893a
    assert "2;243;233;210" in ansi  # the cream
    assert "2;24;40;54" in ansi  # the eyes and nose, #182836
    assert "2;143;82;32" in ansi  # the inner ears, #8f5220
