"""The terminal's colors by background: what COLORFGBG says and what each role draws."""

from __future__ import annotations

import subprocess
import sys

import pytest

from localmelo.support.cli.theme import background, code_theme, pt_style, rich_theme

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


@pytest.mark.parametrize("bg", ["dark", "light", "unknown"])
def test_warnings_are_yellow_only_on_a_known_dark_background(bg: str) -> None:
    theme, style = rich_theme(bg), pt_style(bg)  # type: ignore[arg-type]
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
    theme = rich_theme(bg)  # type: ignore[arg-type]
    for name in MAY_BE[bg]:
        palette = PALETTES[name]
        for role in ("", "error", "warn"):
            style = str(theme.styles[role]) if role else ""
            ratio = _contrast(_drawn(style, palette), palette["bg"])
            assert ratio >= MIN_CONTRAST, (bg, name, role or "text", ratio)
        # Why no sentence a person must read is `muted`.
        assert _contrast(_drawn("dim", palette), palette["bg"]) < MIN_CONTRAST


@pytest.mark.parametrize(
    ("value", "warn", "code"),
    [
        ("15;0", "yellow", "ansi_dark"),
        ("0;15", "bold", "ansi_light"),
        ("", "bold", "ansi_dark"),
    ],
)
def test_the_process_environment_picks_the_theme(
    value: str, warn: str, code: str
) -> None:
    assert code_theme("light") == "ansi_light" and code_theme("unknown") == "ansi_dark"
    script = (
        "from localmelo.support.cli import theme; "
        "print(theme.RICH_THEME.styles['warn'], theme.CODE_THEME)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", script],
        env={"COLORFGBG": value},
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.split() == [warn, code]
