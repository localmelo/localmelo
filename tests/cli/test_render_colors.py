"""Which role each part of the record takes: a tool's kind and outcome, files, the answer.

Role-span tests check the style name a renderer gives each piece of text; the
palette (tests/cli/test_theme.py) decides how a role looks. A few tests draw
through a palette to check the two meet.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console, RenderableType
from rich.style import Style
from rich.text import Text

from localmelo.support.cli import edits
from localmelo.support.cli.approvals import Snapshot
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.termcolors import XTERM, contrast
from localmelo.support.cli.theme import ASCII, Colour, Palette

from .test_render import Turn, Writes, _block, _header_text, _message


def _roles(text: Text) -> dict[str, str]:
    """Each span's text and its style name."""
    return {text.plain[span.start : span.end]: str(span.style) for span in text.spans}


def _hex(style: Style) -> str | None:
    return style.color.name if style.color is not None else None


def _drawn(palette: Palette, renderable: RenderableType, part: str) -> Style:
    """The style *part* is drawn with through *palette*."""
    console = Console(
        file=io.StringIO(),
        width=80,
        force_terminal=True,
        color_system=palette.rich_system,
        theme=palette.rich_theme,
    )
    for line in console.render_lines(renderable, pad=False):
        plain = "".join(segment.text for segment in line)
        if part in plain:
            at, here = plain.index(part), 0
            for segment in line:
                if here <= at < here + len(segment.text):
                    return segment.style or Style()
                here += len(segment.text)
    raise AssertionError(f"{part!r} not drawn")


# ── A tool's kind ──


@pytest.mark.parametrize(
    ("name", "glyph", "role"),
    [
        ("file_read", "→", "kind_read"),
        ("list_files", "→", "kind_read"),
        ("file_write", "±", "kind_write"),
        ("shell_exec", "$", "kind_run"),
        ("python_exec", "$", "kind_run"),
        ("some_mcp_tool", "±", "kind_write"),  # not known: the cautious default
    ],
)
def test_a_tool_row_shows_its_kind_glyph_and_name_in_the_kind(
    name: str, glyph: str, role: str
) -> None:
    header = _header_text(Turn(), name=name, duration_s=0.1)
    assert header.plain.startswith(f"✓ {glyph} {name} · ")
    assert _roles(header)[f"{glyph} {name}"] == role


def test_the_kind_comes_from_the_sessions_tiers() -> None:
    turn = Turn()
    calls: list[int] = []

    def tiers() -> dict[str, str]:
        calls.append(1)
        return {"fetch_page": "network", "search_notes": "read", "deploy": "exec"}

    turn.renderer.tool_tiers = tiers
    for name, glyph, role in (
        ("fetch_page", "@", "kind_network"),
        ("search_notes", "→", "kind_read"),
        ("deploy", "$", "kind_run"),
    ):
        header = _header_text(turn, name=name)
        assert _roles(header)[f"{glyph} {name}"] == role
    # Read once, then again only for a name it does not know.
    assert len(calls) == 1
    _header_text(turn, name="new_tool")
    assert len(calls) == 2


def test_the_kind_glyphs_have_ascii_stand_ins() -> None:
    turn = Turn(symbols=ASCII)
    turn.renderer.tool_tiers = lambda: {"fetch_page": "network"}
    rows = [
        _header_text(turn, name=name).plain.split(" - ")[0]
        for name in ("file_read", "file_write", "shell_exec", "fetch_page")
    ]
    assert rows == [
        "+ > file_read",
        "+ ~ file_write",
        "+ $ shell_exec",
        "+ @ fetch_page",
    ]


# ── A call's outcome ──


@pytest.mark.parametrize(
    ("data", "mark", "role"),
    [
        ({}, "✓", "ok"),
        ({"exit_code": 0}, "✓", "ok"),
        ({"exit_code": 2}, "✗", "error"),
        ({"status": "error", "error": "boom"}, "✗", "error"),
        ({"status": "timeout", "error": "slow"}, "✗", "error"),
        ({"status": "blocked", "error": "no"}, "⊘", "blocked"),
    ],
)
def test_the_mark_takes_the_outcome(data: dict[str, Any], mark: str, role: str) -> None:
    header = _header_text(Turn(), duration_s=0.1, **data)
    assert header.plain[0] == mark
    assert _roles(header)[mark] == role
    exit_code = data.get("exit_code")
    if exit_code:
        assert _roles(header)[f"exit {exit_code}"] == "error"


def test_a_call_that_did_not_run_is_dim_as_a_whole_glyph_too() -> None:
    header = _header_text(Turn(), status="cancelled", duration_s=1.0)
    assert header.style == "muted" and header.plain.startswith("◌ $ shell_exec")


def test_a_failed_first_line_is_in_the_error_role_and_a_success_is_not() -> None:
    for data, first in (({"exit_code": 1}, "error"), ({}, "")):
        body = _block(Turn(), output="F..\nmore", **data)[-1]
        assert isinstance(body, Text)
        assert _roles(body).get("F..", "") == first


# ── Files ──


def test_a_write_body_colors_its_change() -> None:
    renderer = Turn().renderer
    new = Snapshot("r1", edits.Target("/w/n.md", exists=False), "a\n", 1, 0)
    old = Snapshot("r2", edits.Target("/w/o.md", exists=True, text="y\n"), "x\n", 1, 1)
    new_body, old_body = renderer._change_body(new), renderer._change_body(old)
    assert new_body.plain == "  └ new file, 1 line"
    assert old_body.plain == "  └ +1 −1 lines"
    new_roles, old_roles = _roles(new_body), _roles(old_body)
    assert new_roles["new file"] == "file_new" and new_roles[", 1 line"] == "muted"
    assert old_roles["+1"] == "ok_text" and old_roles["−1"] == "error"
    assert old_roles[" lines"] == "muted"


def test_the_changed_line_marks_each_state_and_colors_its_counts() -> None:
    writes = Writes().write("SUMMARY.md", "a\nb\nc\n")
    writes.write("todo.md", "a\nc\n", old="a\nb\n")
    line = writes.turn.renderer.changed_line()
    assert line is not None
    assert line.plain == "  Changed + SUMMARY.md (+3), ~ todo.md (+1 −1)"
    roles = _roles(line)
    assert roles["+ "] == "file_new" and roles["SUMMARY.md"] == "file_new"
    assert roles["~ "] == "file_mod" and roles["todo.md"] == "file_mod"
    assert roles["+3"] == "ok_text" and roles["−1"] == "error"
    assert roles["Changed "] == "muted" and roles[" ("] == "muted"


def test_a_file_whose_state_is_unknown_has_no_mark() -> None:
    # --yolo: nothing read the file first, so new or modified is not known.
    line = Writes().write("hello.py", "x\n", asked=False).turn.renderer.changed_line()
    assert line is not None and line.plain == "  Changed hello.py"
    assert "hello.py" not in _roles(line)


def test_the_file_states_have_ascii_stand_ins() -> None:
    writes = Writes(symbols=ASCII).write("a.md", "x\n").write("b.md", "y\n", old="z\n")
    line = writes.turn.renderer.changed_line()
    assert line is not None and line.plain == "  Changed + a.md (+1), ~ b.md (+1 -1)"


# ── Drawn through a palette ──


def test_kinds_and_outcomes_draw_in_their_colors_at_truecolor() -> None:
    turn = Turn()
    write = _header_text(turn, name="file_write", duration_s=0.1)
    failed = _header_text(turn, name="shell_exec", exit_code=1)
    read = _header_text(turn, name="file_read")
    dark = Palette("truecolor", "dark")
    assert _hex(_drawn(dark, write, "file_write")) == "#ceacf7"
    assert _hex(_drawn(dark, failed, "shell_exec")) == "#61b1cf"
    assert _hex(_drawn(dark, failed, "exit 1")) == "#f6a296"
    assert _hex(_drawn(dark, failed, "✗")) == "#f6a296"
    assert _hex(_drawn(dark, read, "file_read")) == "#91978a"  # it fades
    light = Palette("truecolor", "light")
    assert _hex(_drawn(light, write, "file_write")) == "#8252a9"


def test_no_color_keeps_the_kinds_apart_by_glyph_and_weight() -> None:
    turn = Turn()
    none = Palette("none")
    write = _drawn(none, _header_text(turn, name="file_write"), "file_write")
    read = _drawn(none, _header_text(turn, name="file_read"), "file_read")
    failed = _drawn(none, _header_text(turn, exit_code=1), "exit 1")
    assert write.color is None and write.bold
    assert read.color is None and read.dim
    assert failed.color is None and failed.bold


# ── The answer, the user and code ──

ONE_DARK = Palette("truecolor", "dark", "#282c34", "#abb2bf")
SOLARIZED = Palette("truecolor", "dark", "#002b36", "#839496")
ANSWER = (
    "## Done\n\nI wrote `app.py`.\n\n- one\n\n```python\ndef f():\n    return 1\n```"
)


class _Kept(RecordingOut):
    """A recording that keeps each renderable, to draw it through a palette."""

    def __init__(self, width: int = 80) -> None:
        super().__init__(width)
        self.kept: list[RenderableType] = []

    def print(self, *renderables: RenderableType) -> None:
        self.kept.extend(renderables)
        super().print(*renderables)


def _turn_with_answer(answer: str = ANSWER) -> _Kept:
    out = _Kept()
    turn = Turn()
    turn.out = out
    turn.renderer.out = out
    turn.renderer.begin_turn("Write app.py.")
    call = [{"id": "c1", "name": "file_read", "preview": "a"}]
    _message(turn, "## Looking first\n\n- a", thinking="hm", tool_calls=call)
    turn.tool("file_read", "a", output="x")
    turn.finish(result=answer, n_tool=1)
    return out


def _segments(palette: Palette, out: _Kept) -> list[tuple[str, Style]]:
    console = Console(
        file=io.StringIO(),
        width=out.width,
        force_terminal=True,
        color_system=palette.rich_system,
        theme=palette.rich_theme,
    )
    return [
        (segment.text, segment.style or Style())
        for renderable in out.kept
        for line in console.render_lines(renderable, pad=False)
        for segment in line
    ]


def _first(segments: list[tuple[str, Style]], text: str, after: str = "") -> Style:
    seen = not after
    for part, style in segments:
        seen = seen or after in part
        if seen and text in part:
            return style
    raise AssertionError(f"{text!r} not drawn")


def test_the_answer_mark_heading_and_bullets_are_fur_and_nothing_else_is() -> None:
    drawn = _segments(Palette("truecolor", "dark"), _turn_with_answer())
    fur = "#d98b3b"
    assert _hex(_first(drawn, "Done")) == fur and _first(drawn, "Done").bold
    assert _hex(_first(drawn, "•", after="Done")) == fur
    # The intermediate message: a muted mark, a bold heading, plain bullets.
    marks = [style for text, style in drawn if text == "●"]
    assert _hex(marks[0]) == "#91978a" and _hex(marks[-1]) == fur
    assert _hex(_first(drawn, "Looking first")) is None
    assert _first(drawn, "Looking first").bold
    assert _hex(_first(drawn, "•")) is None
    # Thinking fades with the record.
    assert _hex(_first(drawn, "Thought for")) == "#91978a"
    # The only orange is the answer's.
    orange = {text.strip() for text, style in drawn if _hex(style) == fur}
    assert orange <= {"●", "Done", "•", ""}, orange


@pytest.mark.parametrize(
    ("palette", "mark", "heading", "bold", "underline"),
    [
        (Palette("truecolor", "light"), "#c17626", "#24425f", True, None),
        (Palette("16", "dark"), "yellow", "yellow", True, None),
        (Palette("16", "light"), None, None, True, None),
        (Palette("none"), None, None, True, True),
    ],
)
def test_the_answer_at_each_depth(
    palette: Palette,
    mark: str | None,
    heading: str | None,
    bold: bool,
    underline: bool | None,
) -> None:
    drawn = _segments(palette, _turn_with_answer())
    assert _hex([style for text, style in drawn if text == "●"][-1]) == mark
    done = _first(drawn, "Done")
    assert (_hex(done), done.bold, done.underline) == (heading, bold, underline)


def test_the_user_band_needs_a_fill() -> None:
    out = _Kept()
    renderer = Turn().renderer
    renderer.out = out
    renderer.echo("Write it.")
    [band] = out.kept
    console = Console(
        file=io.StringIO(), width=40, force_terminal=True, theme=ONE_DARK.rich_theme
    )
    [line] = console.render_lines(band)
    assert sum(len(segment.text) for segment in line) == 40  # across the row
    raised = ONE_DARK.fills["raised"].value
    assert all(
        segment.style and segment.style.bgcolor and segment.style.bgcolor.name == raised
        for segment in line
    )
    assert _hex(line[0].style or Style()) == "#b5cf9f"
    # Without the exact background: the mark alone, no band.
    for palette in (Palette("truecolor", "dark"), SOLARIZED, Palette("16", "dark")):
        console = Console(
            file=io.StringIO(), width=40, force_terminal=True, theme=palette.rich_theme
        )
        [line] = console.render_lines(band, pad=False)
        assert not any(s.style and s.style.bgcolor for s in line)


def test_inline_code_is_a_chip_on_the_fill_and_bold_without_one() -> None:
    out = _turn_with_answer("Use `app.py` now.")
    chip = _first(_segments(ONE_DARK, out), "app.py", after="Use")
    assert chip.bgcolor is not None and chip.bgcolor.name == "#373b42"
    assert not chip.bold and chip.color is None
    for palette in (SOLARIZED, Palette("16", "light"), Palette("none")):
        plain = _first(_segments(palette, out), "app.py", after="Use")
        assert plain.bold and plain.bgcolor is None, palette


def test_code_is_a_raised_panel_with_a_fill_and_has_a_gutter_without() -> None:
    out = _turn_with_answer()
    panel = _segments(ONE_DARK, out)
    keyword = _first(panel, "def")
    assert _hex(keyword) == "#b5cf9f"
    assert keyword.bgcolor is not None and keyword.bgcolor.name == "#373b42"
    assert not any(text.startswith("│") for text, _ in panel)
    for palette in (SOLARIZED, Palette("16", "dark"), Palette("none")):
        drawn = _segments(palette, out)
        gutters = [style for text, style in drawn if text == "│ "]
        assert len(gutters) == 2 and all(style.bgcolor is None for style in gutters)


@pytest.mark.parametrize(
    "palette",
    [
        ONE_DARK,
        Palette("truecolor", "dark", "#1e1e1e", "#d4d4d4"),
        Palette("truecolor", "light", "#ffffff", "#383a42"),
        Palette("256", "dark", "#181a1f", "#c9d1d9"),
        Palette("256", "light", "#ffffff", "#24292f"),
    ],
)
def test_text_on_the_raised_fill_is_moved_until_it_reads(palette: Palette) -> None:
    raised = palette.fills["raised"]
    painted = XTERM[raised.xterm] if raised.xterm is not None else raised.value
    for role in ("accent", "path", "code_name", "muted"):
        fg = palette.on_raised(role).fg
        assert fg is not None
        drawn = XTERM[fg.xterm] if fg.xterm is not None else fg.value
        assert contrast(drawn, painted) >= 4.5, (role, drawn, painted)


def test_the_guard_keeps_the_hue_and_moves_only_what_needs_it() -> None:
    # One Dark: comments move #91978a -> #a0a699, the accent already reads.
    assert ONE_DARK.on_raised("muted").fg == Colour("#a0a699")
    assert ONE_DARK.on_raised("accent").fg == Colour("#b5cf9f")
    # No fill, no move.
    assert SOLARIZED.on_raised("muted") == SOLARIZED.look("muted")
