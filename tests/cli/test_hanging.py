"""Output that hangs under its line behind a quiet connector (support/cli/hanging.py)."""

from __future__ import annotations

import asyncio
import io
import re

from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from localmelo.support.cli import hanging
from localmelo.support.cli.hanging import Hang, Hung
from localmelo.support.cli.output import RecordingOut, TerminalOut
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE, active


def _draw(*bodies: RenderableType, width: int = 40, color: bool = False) -> str:
    """What *bodies* print as, trailing spaces kept (they are what a terminal gets)."""
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=width,
        force_terminal=color,
        color_system="standard" if color else None,
        markup=False,
        highlight=False,
        theme=active().rich_theme,
    )
    for body in bodies:
        console.print(body)
    return buffer.getvalue()


def _terminal(no_color: bool) -> tuple[TerminalOut, io.StringIO]:
    stream = io.StringIO()
    output = Vt100_Output(stream, lambda: Size(rows=24, columns=40), term="xterm")
    return TerminalOut(output, no_color=no_color), stream


# ── The shape ──


def test_one_line_hangs_under_the_connector() -> None:
    assert _draw(Hung(Text("Approvals: auto"), UNICODE)) == "  └ Approvals: auto\n"


def test_later_lines_line_up_under_the_first() -> None:
    drawn = _draw(Hung(Text("one\ntwo\nthree"), UNICODE))
    assert drawn == "  └ one\n    two\n    three\n"


def test_blank_lines_stay_blank_with_no_trailing_spaces() -> None:
    assert _draw(Hung(Text("one\n\ntwo"), UNICODE)) == "  └ one\n\n    two\n"


def test_a_long_line_wraps_in_the_width_less_the_indent_and_continues_under_itself() -> (
    None
):
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo"
    drawn = _draw(Hung(Text(words), UNICODE), width=24)
    rows = drawn.splitlines()
    assert len(rows) > 2
    assert rows[0].startswith("  └ ") and all(r.startswith("    ") for r in rows[1:])
    assert all(cells(row) <= 24 for row in rows)
    assert " ".join(row.strip("└ ") for row in rows).split() == words.split()


def test_wide_characters_wrap_by_cells() -> None:
    rows = _draw(Hung(Text("你好世界" * 8), UNICODE), width=20).splitlines()
    assert len(rows) > 2 and all(cells(row) <= 20 for row in rows)
    assert rows[0].startswith("  └ ") and all(r.startswith("    ") for r in rows[1:])


def test_a_table_is_laid_out_in_the_hung_width() -> None:
    table = Table.grid(padding=(0, 1))
    table.add_column()
    table.add_column(justify="right")
    table.add_row("model", "x" * 30)
    rows = _draw(Hung(table, UNICODE), width=30).splitlines()
    assert all(cells(row) <= 30 for row in rows)
    assert rows[0].startswith("  └ model")


def test_ascii_symbols_draw_the_ascii_connector() -> None:
    assert _draw(Hung(Text("one\ntwo"), ASCII)) == "  | one\n    two\n"


def test_pieces_of_one_hang_draw_the_connector_once_on_the_first_text() -> None:
    hang = Hang(UNICODE)
    drawn = _draw(hang.hung(Text("")), hang.hung(Text("a")), hang.hung(Text("b\nc")))
    assert drawn == "\n  └ a\n    b\n    c\n"


# ── Colour ──


def test_the_connector_is_muted_and_the_text_keeps_its_own_style() -> None:
    drawn = _draw(Hung(Text("careful", style="warn"), UNICODE), color=True)
    muted = _draw(Text("x", style="muted"), color=True).split("x")[0]
    warn = _draw(Text("x", style="warn"), color=True).split("x")[0]
    assert muted and warn and muted != warn
    assert drawn.startswith(muted + "  └ ")
    assert warn + "careful" in drawn
    assert muted + "careful" not in drawn


def test_without_colour_the_connector_has_no_hue_and_the_layout_is_the_same() -> None:
    color = r"\x1b\[[0-9;]*(?:3\d|9\d|4\d|10\d)m"
    for no_color in (True, False):
        out, _ = _terminal(no_color)
        drawn = out.render(Hung(Text("careful", style="error"), UNICODE))
        assert bool(re.search(color, drawn)) is not no_color, drawn
        assert re.sub(r"\x1b\[[0-9;]*m", "", drawn) == "  └ careful\n"


# ── Under a command ──


def test_what_a_command_prints_hangs_as_one_body() -> None:
    out = RecordingOut(40)
    with hanging.under_command(UNICODE):
        out.print(Text("one"))
        out.print(Text(""), Text("two\nthree"))
        out.print(Text("four"))
    assert out.text() == "  └ one\n\n    two\n    three\n    four\n"


def test_a_blank_first_print_leaves_the_connector_for_the_first_text() -> None:
    out = RecordingOut(40)
    with hanging.under_command(UNICODE):
        out.print(Text(""))
        out.print(Text("one"))
    assert out.text() == "\n  └ one\n"


def test_a_body_that_is_hung_already_is_not_hung_again() -> None:
    out = RecordingOut(40)
    with hanging.under_command(UNICODE):
        out.print(Hung(Text("one"), UNICODE))
    assert out.text() == "  └ one\n"


def test_each_command_draws_its_own_connector() -> None:
    out = RecordingOut(40)
    for word in ("one", "two"):
        with hanging.under_command(UNICODE):
            out.print(Text(word))
    assert out.text() == "  └ one\n  └ two\n"


def test_the_width_is_narrower_by_the_indent_only_while_a_command_prints() -> None:
    out = RecordingOut(40)
    assert out.width == 40
    with hanging.under_command(UNICODE):
        assert out.width == 36
        with hanging.full_width():
            assert out.width == 40
    assert out.width == 40


def test_a_command_prints_a_full_width_row_on_request() -> None:
    out = RecordingOut(20)
    with hanging.under_command(UNICODE):
        out.print(Text("one"))
        with hanging.full_width():
            out.print(Text("─" * 20))
        out.print(Text("two"))
    assert out.text().splitlines() == ["  └ one", "─" * 20, "    two"]


def test_nothing_hangs_outside_a_command() -> None:
    out = RecordingOut(40)
    out.print(Text("one\ntwo"))
    assert out.text() == "one\ntwo\n"
    with hanging.under_command(UNICODE):
        pass
    out.print(Text("three"))
    assert out.text() == "one\ntwo\nthree\n"


def test_the_terminal_hangs_what_it_prints_and_asks_for_the_narrow_width() -> None:
    out, stream = _terminal(no_color=True)
    with hanging.under_command(UNICODE):
        assert out.width == 36
        out.print(Text("one\ntwo"))
    out.print(Text("three"))
    written = re.sub(r"\x1b\[[0-9;]*m", "", stream.getvalue().replace("\r\n", "\n"))
    assert written == "  └ one\n    two\nthree\n"


async def test_a_task_a_command_starts_prints_at_the_left_edge() -> None:
    out = RecordingOut(40)

    async def say(word: str) -> None:
        out.print(Text(word))

    with hanging.under_command(UNICODE):
        await hanging.spawn_detached(say("detached"))
        await asyncio.create_task(say("inherited"))
    assert out.text().splitlines() == ["detached", "  └ inherited"]
