"""``/memory`` in the terminal: the notes as a list, then what to do with one.

The lists are the terminal's pickers: ↑↓ and Enter, a number key, Esc; there is no
yes/no. Deleting and editing change the files the web console reads, and the
last row of the first list forgets everything after a list that keeps first.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.cli.memory_notes import manage
from localmelo.support.cli.theme import UNICODE
from localmelo.support.memory import ChatMemory
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.cli.ui_helpers import ENTER, ESC, Driver, running_app

DOWN = "\x1b[B"
HOLDER = "test-host:1:notes"


@dataclass
class Printed:
    lines: list[str] = field(default_factory=list)

    def print(self, text: Any) -> None:
        self.lines.append(str(text))


class Session:
    """What ``manage`` asks of a terminal session: its app, output, text and symbols."""

    def __init__(self, driver: Driver) -> None:
        self.app = driver.ui
        self.out = Printed()
        self.renderer = type("Renderer", (), {"symbols": UNICODE})()

    def _text(self, text: str) -> str:
        return text


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A home of the test's own: the notes and the config live under it."""
    from localmelo.support import config

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml"))
    yield tmp_path


async def seed(home: Path) -> ChatMemory:
    """Two notes, written as the console writes them, in the home's memory folder."""
    store = SessionStore(str(home / "sessions.db"), writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: None)
    try:
        for text in ("I prefer short answers.", "My dentist is Lee Park."):
            turn = await store.begin_turn(
                None, text, holder=HOLDER, new_session=NewSession()
            )
            await memory.before_turn(store, turn, text)
    finally:
        await store.aclose()
    return memory


@pytest.fixture()
async def notes(home: Path) -> AsyncIterator[ChatMemory]:
    memory = await seed(home)
    try:
        yield memory
    finally:
        await memory.aclose()


async def test_the_notes_are_a_list_newest_first_with_a_row_that_forgets(
    notes: ChatMemory,
) -> None:
    async with running_app() as driver:
        task = asyncio.create_task(manage(Session(driver)))  # type: ignore[arg-type]
        await driver.wait(lambda: "What melo remembers · 2 notes" in driver.text())
        text = driver.text()
        assert text.index("Your dentist is Lee Park.") < text.index("You prefer short")
        assert (
            "(About you · from chat “My dentist is Lee Pa" in text
        ), "the origin takes the room the note leaves"
        assert "Forget everything…" in text and "enter open" in text
        driver.type("tea")  # no note matches: the forget row goes too
        await driver.wait(lambda: "No note matches 'tea'." in driver.text())
        driver.type(ESC)
        await asyncio.wait_for(task, 3)
        assert len(await notes.records()) == 2


async def test_a_note_is_edited_from_the_list(notes: ChatMemory) -> None:
    async with running_app() as driver:
        session = Session(driver)
        task = asyncio.create_task(manage(session))  # type: ignore[arg-type]
        await driver.wait(lambda: "Your dentist is Lee Park." in driver.text())
        driver.type(ENTER)
        await driver.wait(
            lambda: "Back" in driver.text() and "Edit it" in driver.text()
        )
        driver.type(DOWN + ENTER)
        await driver.wait(
            lambda: "New wording [Your dentist is Lee Park.]" in driver.text()
        )
        driver.type("Your dentist is Kim Soo." + ENTER)
        await driver.wait(
            lambda: "Saved: Your dentist is Kim Soo." in "".join(session.out.lines)
        )
        driver.type(ESC)
        await asyncio.wait_for(task, 3)
    assert [e.text for e in await notes.records()] == [
        "Your dentist is Kim Soo.",
        "You prefer short answers.",
    ]


async def test_a_note_is_deleted_from_the_list_and_back_keeps_it(
    notes: ChatMemory,
) -> None:
    async with running_app() as driver:
        session = Session(driver)
        task = asyncio.create_task(manage(session))  # type: ignore[arg-type]
        await driver.wait(lambda: "Your dentist is Lee Park." in driver.text())
        driver.type(ENTER)  # the first answer is Back
        await driver.wait(lambda: "Delete it" in driver.text())
        driver.type(ENTER)
        await driver.wait(lambda: "What melo remembers · 2 notes" in driver.text())
        assert len(await notes.records()) == 2
        driver.type(ENTER)
        await driver.wait(lambda: "Delete it" in driver.text())
        driver.type("3")  # a number key chooses the row at once
        await driver.wait(
            lambda: "Forgot: Your dentist is Lee Park." in "".join(session.out.lines)
        )
        await driver.wait(lambda: "What melo remembers · 1 note" in driver.text())
        driver.type(ESC)
        await asyncio.wait_for(task, 3)
    assert [e.text for e in await notes.records()] == ["You prefer short answers."]


async def test_forgetting_everything_names_what_goes_and_keeps_first(
    notes: ChatMemory,
) -> None:
    async with running_app() as driver:
        session = Session(driver)
        task = asyncio.create_task(manage(session))  # type: ignore[arg-type]
        await driver.wait(lambda: "Forget everything…" in driver.text())
        driver.type(DOWN + DOWN + ENTER)  # the last row
        await driver.wait(
            lambda: "Forget 2 notes and melo's search index of 0 chat turns?"
            in driver.text()
        )
        assert (
            "1. Keep everything" in driver.text()
            and "Your chats stay." in driver.text()
        )
        driver.type(ENTER)  # keep is the first answer
        await driver.wait(lambda: "What melo remembers · 2 notes" in driver.text())
        assert len(await notes.records()) == 2
        driver.type(DOWN + DOWN + ENTER)
        await driver.wait(lambda: "Forget 2 notes" in driver.text())
        driver.type("2")
        await asyncio.wait_for(task, 3)  # nothing left: it says so and ends
    assert await notes.records() == []
    assert "Forgot everything melo remembered." in session.out.lines
    assert session.out.lines[-1].startswith("Nothing remembered yet.")


async def test_with_no_notes_it_says_how_memory_starts(home: Path) -> None:
    async with running_app() as driver:
        session = Session(driver)
        await asyncio.wait_for(manage(session), 3)  # type: ignore[arg-type]
    assert session.out.lines[0].startswith("Nothing remembered yet. melo keeps a note")
