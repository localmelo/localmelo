"""The recap a resumed chat prints: plain text, one mark per answer, and no internals."""

from __future__ import annotations

import time

import pytest

from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE, Symbols
from localmelo.support.cli.views import resume_recap
from localmelo.support.sessions.runner import ResumeInfo
from localmelo.support.sessions.store import SessionInfo, TurnInfo

ANSWER = (
    "This folder is a small Python project:\n"
    "\n"
    "- `app.py` summarizes the readings in `data.csv`\n"
    "- **tests/test_app.py** checks the summary\n"
    "- `todo.md` asks for a `--json` flag next\n"
    "\n"
    "```python\n"
    "from app import summarize\n"
    "```\n"
)


def _info(turns: int = 1) -> SessionInfo:
    return SessionInfo(
        id="1b476a472877",
        kind="chat",
        title="What is this project?",
        created_at=0.0,
        updated_at=time.time(),
        model="stub-model",
        workspace="/demo",
        meta={},
        busy=False,
        turns=turns,
    )


def _turn(
    number: int, typed: str, answer: str, status: str = "completed", cause: str = "run"
) -> TurnInfo:
    return TurnInfo(
        id=f"turn-{number}",
        session_id="1b476a472877",
        turn=number,
        run_id="run",
        user_text=typed,
        status=status,
        cause=cause,
        answer=answer,
        result=None,
        holder="cli",
        undone=False,
        started_at=0.0,
        finished_at=1.0,
        client_request_id=None,
        typed_text=typed,
        attachments=(),
    )


def _loaded(mode: str = "pairs", turns: int = 1, checkpoint: int = 0) -> ResumeInfo:
    return ResumeInfo(
        mode=mode,  # type: ignore[arg-type]
        fell_back=False,
        turns=turns,
        checkpoint_turns=checkpoint,
        messages=2 * turns,
        chars=100,
    )


def _lines(
    turns: list[TurnInfo],
    *,
    width: int = 80,
    symbols: Symbols = UNICODE,
    developer: bool = False,
    loaded: ResumeInfo | None = None,
) -> list[str]:
    recap = resume_recap(
        _info(len(turns)),
        turns,
        loaded or _loaded(turns=len(turns)),
        width=width,
        symbols=symbols,
        developer=developer,
    )
    return recap.plain.split("\n")


def test_an_answer_gets_one_mark_and_reads_as_plain_text() -> None:
    assert _lines([_turn(1, "What is this project?", ANSWER)]) == [
        '◆ Resumed "What is this project?" · 1 turn · last active just now',
        "  1 turn is back in the model's memory.",
        "  › What is this project?",
        "  ● This folder is a small Python project:",
        "    - app.py summarizes the readings in data.csv",
        "    - tests/test_app.py checks the summary",
        "    - todo.md asks for a --json flag next",
    ]


def test_blank_lines_never_carry_a_mark_and_a_heading_loses_its_hashes() -> None:
    answer = "\n\n## Summary\n\n\nIt *works*, and `snake_case` and 2 * 3 stay.\n"
    rows = _lines([_turn(1, "q", answer)])[3:]
    assert rows == ["  ● Summary", "    It works, and snake_case and 2 * 3 stay."]


def test_markers_inside_words_and_code_are_left_alone() -> None:
    answer = "Run `a **b** c` in __init__.py or my_file_name.py with 2*3*4\n"
    rows = _lines([_turn(1, "q", answer)])[3:]
    assert rows == ["  ● Run a **b** c in __init__.py or my_file_name.py with 2*3*4"]


def test_a_link_keeps_its_text_and_a_quote_its_words() -> None:
    answer = "See [the guide](https://example.com/guide).\n> a quoted line\n"
    assert _lines([_turn(1, "q", answer)])[3:] == [
        "  ● See the guide.",
        "    a quoted line",
    ]


def test_an_answer_shows_four_lines_at_most() -> None:
    answer = "\n".join(f"line {n}" for n in range(1, 9))
    rows = _lines([_turn(1, "q", answer)])[3:]
    assert rows == ["  ● line 1", "    line 2", "    line 3", "    line 4"]


def test_each_of_the_last_three_answers_has_its_own_mark() -> None:
    turns = [_turn(n, f"question {n}", f"answer {n}\nmore {n}") for n in range(1, 5)]
    rows = _lines(turns)
    assert rows[2] == "  … 1 earlier turn"
    assert [row for row in rows if "●" in row] == [
        "  ● answer 2",
        "  ● answer 3",
        "  ● answer 4",
    ]
    assert rows[rows.index("  ● answer 2") + 1] == "    more 2"


def test_a_turn_without_an_answer_says_how_it_ended() -> None:
    rows = _lines([_turn(1, "q", "", status="failed", cause="provider")])
    assert rows[3].startswith("  ✗ ") and rows[3].endswith("(no answer)")
    rows = _lines([_turn(1, "q", "", status="cancelled", cause="user_stop")])
    assert rows[3].startswith("  ■ ") and rows[3].endswith("(no answer)")


@pytest.mark.parametrize(
    ("mode", "turns", "checkpoint", "default", "developer"),
    [
        (
            "pairs",
            1,
            0,
            "1 turn is back in the model's memory.",
            "1 turn is back in the model's memory; its tool calls are not.",
        ),
        (
            "pairs",
            6,
            0,
            "6 turns are back in the model's memory.",
            "6 turns are back in the model's memory; their tool calls are not.",
        ),
        (
            "window",
            6,
            4,
            "6 turns are back in the model's memory.",
            "4 turns are back in the model's memory, with their tool steps; "
            "2 later turns without them.",
        ),
    ],
)
def test_the_note_names_tool_calls_in_developer_mode_only(
    mode: str, turns: int, checkpoint: int, default: str, developer: str
) -> None:
    loaded = _loaded(mode, turns, checkpoint)
    stored = [_turn(1, "q", "a")]

    def note(developer: bool) -> str:
        rows = _lines(stored, loaded=loaded, developer=developer)
        return " ".join(row.strip() for row in rows[1 : rows.index("  › q")])

    assert note(False) == default
    assert note(True) == developer


@pytest.mark.parametrize("width", [80, 44])
def test_no_row_is_wider_than_the_terminal_and_a_cut_row_says_so(width: int) -> None:
    long = "word " * 40
    rows = _lines([_turn(1, "q " + long, long + "\nshort")], width=width)
    assert all(cells(row) <= width for row in rows)
    assert rows[2].endswith("…") and rows[3].endswith("…")
    assert rows[4] == "    short"


def test_ascii_draws_ascii_marks_and_an_ascii_ellipsis() -> None:
    long = "word " * 40
    rows = _lines([_turn(1, "q", long)], width=44, symbols=ASCII)
    assert all(row.isascii() and cells(row) <= 44 for row in rows)
    assert rows[3].startswith("  * ") and rows[3].endswith("...")
