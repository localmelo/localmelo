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
from localmelo.support.cli.theme import ASCII, Palette

from .test_render import Turn, Writes, _block, _header_text


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
