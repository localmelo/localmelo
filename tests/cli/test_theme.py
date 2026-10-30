"""The terminal's colors: the role table, its contrast, the depth and background sources."""

from __future__ import annotations

import subprocess
import sys

import pytest
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.output.vt100 import _256ColorCache

from localmelo.support.cli import theme
from localmelo.support.cli.termcolors import (
    XTERM,
    TerminalColors,
    color_depth,
    contrast,
    rgb,
)
from localmelo.support.cli.theme import HUES, TYPICAL, Palette, background, detect

RGB = tuple[int, int, int]

# The 16-color palettes of a typical dark and light terminal theme.
PALETTES: dict[str, dict[str, RGB]] = {
    "dark": {
        "bg": (24, 26, 31),
        "fg": (201, 209, 217),
        "red": (255, 123, 114),
        "yellow": (227, 179, 65),
    },
    "light": {
        "bg": (255, 255, 255),
        "fg": (36, 41, 47),
        "red": (207, 34, 46),
        "yellow": (154, 103, 0),
    },
}
# The palettes a background may be: an unknown one could be either.
MAY_BE = {"dark": ["dark"], "light": ["light"], "unknown": ["dark", "light"]}
MIN_CONTRAST = 4.5  # WCAG AA for body text


def _luminance(rgb: RGB) -> float:
    r, g, b = (
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in (value / 255 for value in rgb)
    )
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first: RGB, second: RGB) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _drawn(style: str, palette: dict[str, RGB]) -> RGB:
    """The color rich's *style* draws on *palette*: dim fades 60% toward the background."""
    color = next((palette[w] for w in style.split() if w in palette), palette["fg"])
    if "dim" not in style.split():
        return color
    r, g, b = (
        round(c * 0.4 + e * 0.6) for c, e in zip(color, palette["bg"], strict=True)
    )
    return (r, g, b)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("15;0", "dark"),
        ("12;8", "dark"),
        ("0;15", "light"),
        ("0;7", "light"),
        ("0;default;15", "light"),
        ("default;default", "unknown"),
        ("15", "unknown"),
        ("0;99", "unknown"),
    ],
)
def test_colorfgbg_names_the_background(value: str, expected: str) -> None:
    assert background({"COLORFGBG": value}) == expected
    assert background({}) == "unknown"


def _sixteen(bg: str) -> Palette:
    return Palette("16", bg)  # type: ignore[arg-type]


@pytest.mark.parametrize("bg", ["dark", "light", "unknown"])
def test_warnings_are_yellow_only_on_a_known_dark_background(bg: str) -> None:
    # 16 colors: the user's own theme tunes the named colors.
    palette = _sixteen(bg)
    theme, style = palette.rich_theme, palette.pt_style
    yellow = bg == "dark"
    for role in ("warn", "blocked"):
        attrs = style.get_attrs_for_style_str(f"class:{role}")
        assert (str(theme.styles[role]) == "yellow") is yellow
        assert (attrs.color == "ansiyellow") is yellow and attrs.bold is not yellow
    # A hint says what to do: only meta is dim.
    assert not style.get_attrs_for_style_str("class:hint").dim
    assert style.get_attrs_for_style_str("class:muted").dim


@pytest.mark.parametrize("bg", ["dark", "light", "unknown"])
def test_text_error_and_warn_read_at_4_5_to_1_on_every_palette_they_may_meet(
    bg: str,
) -> None:
    theme = _sixteen(bg).rich_theme
    for name in MAY_BE[bg]:
        palette = PALETTES[name]
        for role in ("", "error", "warn"):
            style = str(theme.styles[role]) if role else ""
            ratio = _contrast(_drawn(style, palette), palette["bg"])
            assert ratio >= MIN_CONTRAST, (bg, name, role or "text", ratio)
        # Why no sentence a person must read is `muted`.
        assert _contrast(_drawn("dim", palette), palette["bg"]) < MIN_CONTRAST


@pytest.mark.parametrize(
    ("env", "seen"),
    [
        ({"COLORFGBG": "15;0"}, "yellow 16 dark"),
        ({"COLORFGBG": "0;15"}, "bold 16 light"),
        ({"COLORFGBG": "15;0", "COLORTERM": "truecolor"}, "#e9bc54 truecolor dark"),
        ({"TERM": "xterm-256color", "COLORFGBG": "0;15"}, "color(130) 256 light"),
        ({}, "bold 16 unknown"),
        ({"NO_COLOR": "1", "COLORTERM": "truecolor"}, "bold none unknown"),
    ],
)
def test_the_process_environment_picks_the_palette(
    env: dict[str, str], seen: str
) -> None:
    # Until the UI asks the terminal, the environment alone decides.
    script = (
        "from localmelo.support.cli import theme; p = theme.active(); "
        "print(p.rich_theme.styles['warn'], p.depth, p.polarity)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.split() == seen.split()


# ── The role table ──


@pytest.mark.parametrize("polarity", ["dark", "light"])
@pytest.mark.parametrize("depth", ["truecolor", "256"])
def test_every_role_keeps_its_contrast_on_every_typical_background(
    polarity: str, depth: str
) -> None:
    # Text 4.5:1, marks and rules 3:1; at 256 colors, the index the role keeps.
    problems = []
    for role, hue in HUES.items():
        dark = polarity == "dark"
        drawn = (
            (hue.dark if dark else hue.light)
            if depth == "truecolor"
            else (XTERM[hue.dark256 if dark else hue.light256])
        )
        for ground in TYPICAL[polarity]:
            if (ratio := contrast(drawn, ground)) < hue.target:
                problems.append(f"{role} {drawn} on {ground}: {ratio:.2f}")
    assert not problems


def test_at_256_colors_both_libraries_land_on_the_chosen_index() -> None:
    palette = Palette("256", "dark")
    cache = _256ColorCache()
    for role, hue in HUES.items():
        look = palette.look(role)
        assert look.fg is not None
        color = look.rich().color
        assert color is not None and color.number == hue.dark256, role
        # prompt_toolkit maps the hex of the index to the index itself.
        assert cache[rgb(look.fg.pt())] == hue.dark256, role


def test_each_depth_draws_with_its_own_colors() -> None:
    dark = {depth: Palette(depth, "dark") for depth in ("16", "256", "truecolor")}
    assert str(dark["truecolor"].look("accent").rich()) == "#b5cf9f"
    assert str(dark["256"].look("accent").rich()) == "color(151)"
    assert str(dark["16"].look("accent").rich()) == "green"
    assert dark["truecolor"].rich_system == "truecolor"
    assert dark["16"].rich_system == "standard"
    assert dark["256"].pt_depth == ColorDepth.DEPTH_8_BIT
    # A known depth without a known polarity uses the named colors.
    assert str(Palette("truecolor").look("accent").rich()) == "green"
    # Muted is a real color with hues, not SGR dim.
    assert not dark["truecolor"].look("muted").rich().dim


def test_no_color_keeps_attributes_only() -> None:
    palette = Palette("none")
    for name in theme.NAMES:
        look = palette.look(name)
        assert look.fg is None and look.bg is None, name
    assert palette.look("heading").underline and palette.look("heading").bold
    assert palette.look("error").bold and palette.look("muted").dim
    assert palette.pt_depth == ColorDepth.DEPTH_1_BIT
    assert theme.pt_color_depth(True) == ColorDepth.DEPTH_1_BIT


# ── Depth and background ──


@pytest.mark.parametrize(
    ("env", "setting", "depth"),
    [
        ({"NO_COLOR": "1", "COLORTERM": "truecolor"}, "truecolor", "none"),
        ({"TERM": "dumb"}, "256", "none"),
        ({"NO_COLOR": ""}, "auto", "16"),
        ({"TERM": "xterm-256color"}, "16", "16"),
        ({}, "truecolor", "truecolor"),
        ({"COLORTERM": "24bit", "TERM": "xterm"}, "auto", "truecolor"),
        ({"COLORTERM": "TrueColor"}, "auto", "truecolor"),
        ({"TERM": "screen-256color"}, "auto", "256"),
        ({"TERM": "xterm"}, "auto", "16"),
        ({}, "bogus", "16"),
    ],
)
def test_the_color_depth_first_rule_wins(
    env: dict[str, str], setting: str, depth: str
) -> None:
    assert color_depth(env, setting) == depth


def test_the_background_sources_in_order() -> None:
    env = {"COLORTERM": "truecolor", "COLORFGBG": "0;15"}
    asked: list[bool] = []

    def ask(colors: TerminalColors | None) -> object:
        def answer() -> TerminalColors | None:
            asked.append(True)
            return colors

        return answer

    black = TerminalColors("#000000", "#ffffff")
    # The setting wins, and the terminal is not asked.
    assert detect(bg="dark", environ=env, ask=ask(black)) == Palette(  # type: ignore[arg-type]
        "truecolor", "dark"
    )
    assert not asked
    # The terminal's own answer beats COLORFGBG, and keeps the exact colors.
    told = detect(environ=env, ask=ask(black))  # type: ignore[arg-type]
    assert told == Palette("truecolor", "dark", "#000000", "#ffffff")
    # Silence (or no background in the answer) falls back to COLORFGBG.
    for silence in (None, TerminalColors(None, "#ffffff")):
        assert detect(environ=env, ask=ask(silence)) == Palette(  # type: ignore[arg-type]
            "truecolor", "light"
        )
    # NO_COLOR: nothing is asked at all.
    asked.clear()
    assert detect(environ={"NO_COLOR": "1"}, ask=ask(black)).depth == "none"  # type: ignore[arg-type]
    assert not asked


def test_fills_need_the_exact_background_and_text_that_keeps_4_5_to_1() -> None:
    assert not Palette("truecolor", "dark").fills  # polarity only
    assert not Palette("truecolor", "dark", "#282c34").fills  # no foreground
    assert not Palette("16", "dark", "#282c34", "#abb2bf").fills
    one_dark = Palette("truecolor", "dark", "#282c34", "#abb2bf").fills
    assert set(one_dark) == {"raised", "selection", "diff_add", "diff_remove"}
    assert one_dark["raised"].value == "#373b42"
    # Solarized dark's text falls under 4.5:1 on the raised band: no fills.
    assert not Palette("truecolor", "dark", "#002b36", "#839496").fills
    # At 256 colors a fill is the nearest index.
    white = Palette("256", "light", "#ffffff", "#000000").fills
    assert white["raised"].xterm == 255


def test_an_unusual_exact_background_moves_a_role_until_it_reads() -> None:
    grey = Palette("truecolor", "dark", "#4a4a4a", "#ffffff")
    muted = grey.look("muted").fg
    assert muted is not None and muted.value != HUES["muted"].dark
    assert contrast(muted.value, "#4a4a4a") >= 4.5
    # A typical one keeps the table's color.
    assert Palette("truecolor", "dark", "#1e1e1e").look("muted").fg == theme.Colour(
        HUES["muted"].dark
    )


# ── Bands: the chosen row and a diff's rows ──

BANDS = (("selection", "accent"), ("diff_add", "ok_text"), ("diff_remove", "error"))
FOREGROUND = {"dark": "#e6e6e6", "light": "#1f1f1f"}


@pytest.mark.parametrize("depth", ["truecolor", "256"])
@pytest.mark.parametrize("polarity", ["dark", "light"])
def test_text_on_a_band_keeps_4_5_to_1_on_every_typical_background(
    depth: str, polarity: str
) -> None:
    from localmelo.support.cli.termcolors import XTERM, contrast

    for ground in TYPICAL[polarity]:
        palette = Palette(depth, polarity, ground, FOREGROUND[polarity])  # type: ignore[arg-type]
        assert palette.fills, ground
        for fill, role in BANDS:
            colour, fg = palette.fills[fill], palette.on_raised(role, fill).fg
            assert fg is not None
            drawn = XTERM[fg.xterm] if fg.xterm is not None else fg.value
            painted = XTERM[colour.xterm] if colour.xterm is not None else colour.value
            assert contrast(drawn, painted) >= theme.TEXT, (ground, fill)


def test_the_chosen_row_without_fills_is_bold_and_reverse_with_no_color() -> None:
    def attrs(palette: Palette, name: str) -> object:
        return palette.pt_style.get_attrs_for_style_str(f"class:{name}")

    banded = attrs(Palette("truecolor", "dark", "#181a1f", "#e6e6e6"), "chosen")
    assert banded.bgcolor and banded.bold and not banded.reverse  # type: ignore[attr-defined]
    plain = attrs(Palette("truecolor", "dark"), "chosen")  # the background unknown
    assert not plain.bgcolor and plain.bold and plain.color  # type: ignore[attr-defined]
    assert attrs(Palette("none"), "chosen").reverse  # type: ignore[attr-defined]
    # A key's words are muted with a palette, plain at 16 colours: never dim.
    assert attrs(Palette("256", "dark"), "faint").color  # type: ignore[attr-defined]
    assert not attrs(Palette("16", "light"), "faint").dim  # type: ignore[attr-defined]
    assert attrs(Palette("truecolor", "dark"), "frame").color == "e9bc54"  # type: ignore[attr-defined]


# ── The completion list: prompt_toolkit's own classes, drawn from the roles ──

MENU_PALETTES = {
    "dark-truecolor": Palette("truecolor", "dark", "#181a1f", "#e6e6e6"),
    "light-truecolor": Palette("truecolor", "light", "#ffffff", "#1f1f1f"),
    "dark-256": Palette("256", "dark", "#181a1f", "#e6e6e6"),
    "light-256": Palette("256", "light", "#fafafa", "#1f1f1f"),
    "polarity-only": Palette("256", "dark"),
    "16-colours": Palette("16"),
    "no-color": Palette("none"),
}


def _menu_attrs(palette: Palette, name: str) -> object:
    """What the app draws for *name*: prompt_toolkit's defaults merged under ours."""
    from prompt_toolkit.styles import default_ui_style, merge_styles

    style = merge_styles([default_ui_style(), palette.pt_style])
    return style.get_attrs_for_style_str(f"class:completion-menu class:{name}")


@pytest.mark.parametrize("palette", MENU_PALETTES.values(), ids=MENU_PALETTES)
def test_the_completion_list_has_no_block_and_names_in_the_body_ink(
    palette: Palette,
) -> None:
    for name in ("completion-menu", "completion-menu.completion"):
        plain = _menu_attrs(palette, name)
        assert not (plain.bgcolor or plain.color or plain.reverse), name  # type: ignore[attr-defined]
        assert not (plain.bold or plain.dim), name  # type: ignore[attr-defined]
    for name in ("scrollbar.background", "completion-menu.multi-column-meta"):
        assert not _menu_attrs(palette, name).bgcolor, name  # type: ignore[attr-defined]


@pytest.mark.parametrize("palette", MENU_PALETTES.values(), ids=MENU_PALETTES)
def test_the_chosen_row_of_the_list_is_the_pickers_band(palette: Palette) -> None:
    chosen = _menu_attrs(palette, "chosen")
    row = _menu_attrs(palette, "completion-menu.completion.current")
    for field in ("color", "bgcolor", "bold", "reverse"):
        assert getattr(row, field) == getattr(chosen, field), field
    note = _menu_attrs(palette, "completion-menu.meta.completion.current")
    assert note.bgcolor == chosen.bgcolor and note.reverse == chosen.reverse  # type: ignore[attr-defined]
    assert not note.bold  # type: ignore[attr-defined]
    if palette.fills:  # a band: the exact background is known
        assert row.bgcolor and not row.reverse  # type: ignore[attr-defined]
    elif palette.depth == "none":
        assert row.reverse  # type: ignore[attr-defined]
    else:  # no band: the choice is bold in the accent, and the mark carries it
        assert row.bold and not row.bgcolor and not row.reverse  # type: ignore[attr-defined]


@pytest.mark.parametrize("palette", MENU_PALETTES.values(), ids=MENU_PALETTES)
def test_descriptions_are_muted_and_the_bar_is_in_the_rule_colour(
    palette: Palette,
) -> None:
    muted, rule = _menu_attrs(palette, "muted"), _menu_attrs(palette, "rule")
    for name in (
        "completion-menu.meta.completion",
        "completion-menu.multi-column-meta",
    ):
        note = _menu_attrs(palette, name)
        assert (note.color, note.dim) == (muted.color, muted.dim), name  # type: ignore[attr-defined]
        assert not note.bgcolor, name  # type: ignore[attr-defined]
    thumb = _menu_attrs(palette, "scrollbar.button")
    assert (thumb.color, thumb.dim) == (rule.color, rule.dim)  # type: ignore[attr-defined]
    assert not thumb.bgcolor  # type: ignore[attr-defined]


def test_the_completion_list_uses_the_fur_tokens_per_theme() -> None:
    dark, light = MENU_PALETTES["dark-truecolor"], MENU_PALETTES["light-truecolor"]
    for palette, tokens in ((dark, "dark"), (light, "light")):
        fill = palette.fills["selection"].value.lstrip("#")
        row = _menu_attrs(palette, "completion-menu.completion.current")
        assert row.bgcolor == fill  # type: ignore[attr-defined]
        assert row.color == getattr(HUES["accent"], tokens)[1:]  # type: ignore[attr-defined]
        note = _menu_attrs(palette, "completion-menu.meta.completion")
        assert note.color == getattr(HUES["muted"], tokens)[1:]  # type: ignore[attr-defined]
        thumb = _menu_attrs(palette, "scrollbar.button")
        assert thumb.color == getattr(HUES["rule"], tokens)[1:]  # type: ignore[attr-defined]
