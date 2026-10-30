"""The completion list below the input: the picker's mark, a slim bar only on overflow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from prompt_toolkit.completion import WordCompleter

from localmelo.support.cli.completion import LEAD, SlashCompleter, _Scrollbar
from localmelo.support.cli.theme import ASCII, UNICODE, Symbols
from localmelo.tests.cli.ui_helpers import Driver, FakeController, running_app

DOWN = "\x1b[B"


def _menu(driver: Driver) -> list[str]:
    """The list's rows: between the input's second rule and the status bar.

    A blank row sets the bar apart from the list: it is not one of the rows.
    """
    screen = driver.screen()
    rules = [i for i, line in enumerate(screen) if line.startswith("──────")]
    status = max(i for i, line in enumerate(screen) if line.strip())
    assert not screen[status - 1].strip(), screen  # the bar never reads as a row
    return screen[rules[1] + 1 : status - 1]


async def _opened(app: Driver, typed: str) -> list[str]:
    app.type(typed)
    await app.wait(lambda: app.ui.buffer.complete_state is not None)
    await app.settle()
    return _menu(app)


async def test_the_list_keeps_its_rows_and_aligns_names_and_descriptions() -> None:
    async with running_app(FakeController(), completer=SlashCompleter()) as app:
        rows = await _opened(app, "/")
        assert len(rows) == 8  # COMPLETION_ROWS, as before
        assert all(row.startswith(" " * LEAD + "/") for row in rows)
        names = [row[LEAD:].split("  ")[0] for row in rows]
        assert names[0] == "/model" and names[-1] == "/retry"
        # Every description starts in the column after the widest name.
        starts = {row.index(row[LEAD:].split("  ", 1)[1].lstrip()) for row in rows}
        assert len(starts) == 1, rows


async def test_the_chosen_row_has_the_pickers_mark_and_the_names_do_not_move() -> None:
    async with running_app(FakeController(), completer=SlashCompleter()) as app:
        rows = await _opened(app, "/")
        assert not any(UNICODE.selection in row for row in rows)
        app.type(DOWN * 2)
        await app.wait(lambda: any(UNICODE.selection in row for row in _menu(app)))
        rows = _menu(app)
        marked = [row for row in rows if UNICODE.selection in row]
        assert len(marked) == 1 and marked[0].startswith(f"{UNICODE.selection} /new")
        # The mark takes the lead the other rows leave blank.
        assert {row.index("/") for row in rows} == {LEAD}


async def test_the_bar_shows_only_when_there_are_more_rows_than_fit() -> None:
    async with running_app(FakeController(), completer=SlashCompleter()) as app:
        rows = await _opened(app, "/")
        assert any(row.endswith(UNICODE.bar) for row in rows), rows
        app.type("mo")
        await app.wait(lambda: len(_menu(app)) == 1)
        assert UNICODE.bar not in _menu(app)[0]


async def test_arguments_use_the_same_list() -> None:
    completer = WordCompleter(["/model one", "/model two"], sentence=True)
    async with running_app(FakeController(), completer=completer) as app:
        assert await _opened(app, "/model t") == [" " * LEAD + "/model two"]


@pytest.mark.parametrize("symbols", [UNICODE, ASCII])
def test_the_bar_is_a_cell_of_the_symbol_set_sized_and_placed_by_the_scroll(
    symbols: Symbols,
) -> None:
    bar = _Scrollbar(symbols.bar)

    def draw(total: int, rows: int, scroll: int) -> list[str]:
        info = SimpleNamespace(
            content_height=total, window_height=rows, vertical_scroll=scroll
        )
        fragments = bar.create_margin(info, 1, rows)  # type: ignore[arg-type]
        return "".join(part[1] for part in fragments).split("\n")

    mark = symbols.bar
    assert draw(8, 8, 0) == [""]  # everything fits: no bar
    assert draw(3, 8, 0) == [""]
    assert draw(32, 8, 0) == [mark, mark] + [" "] * 6
    assert draw(32, 8, 24) == [" "] * 6 + [mark, mark]  # the last page
    assert draw(32, 8, 12) == [" "] * 3 + [mark, mark] + [" "] * 3
    assert draw(1000, 8, 0).count(mark) == 1  # never thinner than a cell
