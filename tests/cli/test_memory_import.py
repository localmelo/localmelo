"""Bringing in other assistants' notes from the terminal: the choice, the first run's offer, `/memory import`.

The home is the synthetic one of the memory tests (invented files of several assistants), and ``Path.home``
points at it, so nothing here reads the real one. Typed answers stand for a piped stdin; the arrow-key list
of a terminal is the one ``/memory import`` draws in melo's own screen.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support import command_words, config, memory_cli, onboard
from localmelo.support.cli import launch, memory_import
from localmelo.support.cli.launch import Choice, run_interactive
from localmelo.support.cli.session_tools import SessionTools
from localmelo.support.memory import ChatMemory
from localmelo.tests.cli.test_memory_notes import Session
from localmelo.tests.cli.ui_helpers import ENTER, ESC, running_app
from localmelo.tests.memory.import_home import SECRET, build

DOWN = "\x1b[B"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A home of invented assistant files; the notes and the config live under it."""
    found = build(tmp_path.resolve())
    monkeypatch.setenv("HOME", str(found))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: found))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(config, "CONFIG_DIR", str(found / "cfg"))
    monkeypatch.setattr(config, "CONFIG_PATH", str(found / "cfg" / "config.toml"))
    yield found


def typed(monkeypatch: pytest.MonkeyPatch, answers: str) -> None:
    """Stdin is a pipe that holds *answers* (one per line); not a terminal, so lists are typed numbers."""
    monkeypatch.setattr("sys.stdin", io.StringIO(answers))


async def read_notes() -> list[str]:
    memory = ChatMemory(config.load)
    try:
        return [entry.text for entry in await memory.records()]
    finally:
        await memory.aclose()


def notes() -> list[str]:
    return asyncio.run(read_notes())


def pending() -> bool:
    return ChatMemory(config.load).import_pending()


# ── The words of the list ──


def test_the_rows_name_the_notes_and_files_and_the_files_show_paths_with_samples_and_what_was_left_out() -> (
    None
):
    plan = asyncio.run(memory_import._scan())
    assert plan is not None
    sources = memory_import.usable(plan)
    rows = memory_import.menu(sources)
    assert rows[1:] == ["Show what they say", "Not now"]
    notes_found = sum(source["items_kept"] for source in sources)
    assert rows[0] == f"Bring in {notes_found} notes from {len(sources)} files"
    assert memory_import.menu(sources[:1])[0].endswith("from 1 file")
    shown = memory_import.describe(plan)
    assert "~/.claude/CLAUDE.md · " in shown, "home is written as ~"
    assert (
        "“Prefer tabs in Makefiles.”" in shown
        and "“Xu prefers concise replies.”" in shown
    )
    assert (
        "left out: secrets, other people's details, how the other tool works" in shown
    )
    assert str(Path.home()) not in shown
    assert SECRET not in shown and "hunter2" not in shown and "LEAK" not in shown
    assert "claude" not in memory_import.TITLE.lower()


# ── Piped stdin keeps typed answers; no answer is "Not now" ──


def test_a_typed_answer_shows_what_the_files_say_then_brings_the_notes_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    typed(monkeypatch, "2\n1\n")
    assert memory_import.offer() == 0
    out = capsys.readouterr().out
    assert (
        out.count("[1] Bring in ") == 2
    ), "the list is asked again after the files are shown"
    assert out.index("~/.claude/CLAUDE.md") < out.index("Brought in ")
    assert "melo memory import --undo " in out
    assert "Use uv for new Python projects." in notes()
    assert not pending(), "bringing notes in counts as the offer made"


def test_not_now_and_no_answer_bring_nothing_in_and_the_offer_counts_as_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed(monkeypatch, "3\n")
    assert memory_import.offer() == 0
    assert notes() == [] and not pending()
    # Nobody at the other end of the pipe: end of input is not consent.
    memory_dir = Path(ChatMemory(config.load).directory) / "import_offer.json"
    memory_dir.unlink()
    typed(monkeypatch, "")
    assert memory_import.offer() == 0
    assert notes() == [] and not pending()


def test_the_command_brings_in_and_takes_back_a_batch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    typed(monkeypatch, "1\n")
    assert memory_cli.main(["import"]) == 0
    out = capsys.readouterr().out
    batch = out.split("melo memory import --undo ")[1].split()[0].rstrip(" .")
    assert len(notes()) > 0
    assert memory_cli.main(["import", "--undo", batch]) == 0
    assert "Took back " in capsys.readouterr().out and notes() == []
    assert memory_cli.main(["import", "-u", batch]) == 1
    assert "no notes are left from the import" in capsys.readouterr().out


def test_with_nothing_to_bring_in_or_memory_off_the_command_says_so_and_asks_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    ask = mock.MagicMock()
    monkeypatch.setattr(onboard, "_ask_choice", ask)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert memory_import.offer() == 0
    assert capsys.readouterr().out == memory_import.NONE + "\n"
    cfg = config.load()
    cfg.memory.across_chats = False
    config.save(cfg)
    assert memory_import.offer() == 0
    assert capsys.readouterr().out == memory_import.OFF + "\n"
    ask.assert_not_called()


def test_files_whose_notes_are_all_known_say_nothing_is_new(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    typed(monkeypatch, "1\n")
    memory_import.offer()
    capsys.readouterr()
    ask = mock.MagicMock(side_effect=AssertionError("asked"))
    monkeypatch.setattr(onboard, "_ask_choice", ask)
    assert memory_import.offer() == 0
    assert capsys.readouterr().out == memory_import.KNOWN + "\n"


def test_the_command_word_names_the_verb() -> None:
    assert "import" in memory_cli.VERBS
    with pytest.raises(SystemExit):
        command_words.expand(["memory", "-h"])


# ── The first run ──


def test_the_first_run_offers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    typed(monkeypatch, "3\n")
    memory_import.first_run()
    assert not pending() and notes() == []
    ask = mock.MagicMock(side_effect=AssertionError("the offer came back"))
    monkeypatch.setattr(onboard, "_ask_choice", ask)
    memory_import.first_run()
    ask.assert_not_called()


def test_the_first_run_says_nothing_when_there_is_nothing_to_offer_or_memory_is_off(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    ask = mock.MagicMock(side_effect=AssertionError("asked"))
    monkeypatch.setattr(onboard, "_ask_choice", ask)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    memory_import.first_run()
    assert (
        capsys.readouterr().out == "" and pending()
    ), "an empty scan leaves the offer open"
    cfg = config.load()
    cfg.memory.across_chats = False
    config.save(cfg)
    memory_import.first_run()
    assert capsys.readouterr().out == ""
    ask.assert_not_called()


def test_a_first_run_whose_files_hold_nothing_new_shows_no_list_and_leaves_the_offer_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every line is a note already: no list, and a tool installed later can still be offered."""
    typed(monkeypatch, "1\n")
    memory_import.offer()
    (Path(ChatMemory(config.load).directory) / "import_offer.json").unlink()
    capsys.readouterr()
    ask = mock.MagicMock(side_effect=AssertionError("asked"))
    monkeypatch.setattr(onboard, "_ask_choice", ask)
    memory_import.first_run()
    ask.assert_not_called()
    assert capsys.readouterr().out == "" and pending()


def test_a_failing_offer_never_stops_melo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_import, "offer", mock.MagicMock(side_effect=OSError("disk"))
    )
    memory_import.first_run()


def _interactive(monkeypatch: pytest.MonkeyPatch, *, terminal: bool) -> mock.MagicMock:
    """Run the interactive entry up to the UI with the first run spied; the spy."""
    spy = mock.MagicMock()
    monkeypatch.setattr(memory_import, "first_run", spy)
    monkeypatch.setattr(launch, "ui_terminal", lambda: terminal)
    monkeypatch.setattr(
        launch,
        "choose_backend",
        lambda args: Choice(backend=mock.MagicMock(), cfg=None),
    )
    monkeypatch.setattr(
        launch,
        "configure_logging",
        mock.MagicMock(side_effect=FileExistsError(17, "x")),
    )
    args = argparse.Namespace(reconfigure=False, workspace=None, base_url=None)
    run_interactive(args, parser=argparse.ArgumentParser(), options=AgentOptions())
    return spy


def test_only_the_interactive_ui_at_a_terminal_makes_the_first_run_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _interactive(monkeypatch, terminal=True).assert_called_once_with()
    _interactive(monkeypatch, terminal=False).assert_not_called()


def test_a_direct_run_never_reaches_the_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    """`melo --base-url … "task"` (what a benchmark runs) is not the interactive UI."""
    import localmelo.__main__ as cli

    spy = mock.MagicMock()
    monkeypatch.setattr(memory_import, "first_run", spy)
    monkeypatch.setattr(memory_import, "offer", spy)
    agent = mock.MagicMock()
    agent.run_task = mock.AsyncMock(
        return_value=mock.MagicMock(status="completed", result="42")
    )
    agent.close = mock.AsyncMock()
    with (
        mock.patch("sys.argv", ["melo", "--base-url", "http://127.0.0.1:1/v1", ""]),
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=agent),
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    spy.assert_not_called()


# ── /memory import in a chat ──


async def test_slash_memory_import_is_a_list_that_shows_the_files_then_brings_the_notes_in() -> (
    None
):
    async with running_app() as driver:
        session = Session(driver)
        task = asyncio.create_task(memory_import.bring_in(session))  # type: ignore[arg-type]
        await driver.wait(lambda: "melo found notes on this Mac" in driver.text())
        assert "Bring in " in driver.text() and "Show what they say" in driver.text()
        driver.type(DOWN + ENTER)
        await driver.wait(lambda: "~/.claude/CLAUDE.md" in "".join(session.out.lines))
        await driver.wait(lambda: "Show what they say" in driver.text())
        assert await read_notes() == [], "showing writes nothing"
        driver.type(ENTER)
        await asyncio.wait_for(task, 5)
    assert "Brought in " in "".join(session.out.lines)
    assert "Use uv for new Python projects." in await read_notes() and not pending()


async def test_esc_in_the_list_is_not_now() -> None:
    async with running_app() as driver:
        task = asyncio.create_task(memory_import.bring_in(Session(driver)))  # type: ignore[arg-type]
        await driver.wait(lambda: "Not now" in driver.text())
        driver.type(ESC)
        await asyncio.wait_for(task, 5)
    assert await read_notes() == [] and not pending()


async def test_slash_memory_import_is_a_command_of_the_chat(tmp_path: Path) -> None:
    seen: list[Any] = []

    async def bring(session: Any) -> None:
        seen.append(session)

    session = mock.MagicMock()
    session.state = "idle"
    with mock.patch.object(memory_import, "bring_in", bring):
        await SessionTools(session).run("memory", "import")
        session.state = "running"
        with pytest.raises(ValueError, match="opens when the turn ends"):
            await SessionTools(session).run("memory", "import")
        with pytest.raises(ValueError, match="/memory import"):
            await SessionTools(session).run("memory", "what")
    assert seen == [session]
