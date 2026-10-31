"""The main chat works in the default folder as it is now; a side chat keeps its own."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from localmelo.melo.schema import Message
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.errors import ApiError

from .conftest import FakeAgent, FakeBuilder, open_session


def _home() -> Path:
    return Path(os.environ["HOME"])


async def test_the_main_chat_follows_a_changed_default(
    console: Console, builder: FakeBuilder
) -> None:
    first, second = _home() / "first", _home() / "second"
    console.cfg.gateway.workspace = str(first)
    main = console.main_id
    assert main is not None
    entry = await open_session(console.sessions, main)
    agent = cast(FakeAgent, entry.agent)
    assert agent.workspace_root == str(first.resolve())
    agent.messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]

    # The default changes while the main chat's Agent is open.
    console.cfg.gateway.workspace = str(second)
    entry = await open_session(console.sessions, main)

    moved = cast(FakeAgent, entry.agent)
    assert moved is not agent and agent.closed
    assert moved.workspace_root == str(second.resolve())
    assert second.is_dir()
    assert [m.content for m in moved.conversation()] == ["hi", "hello"]
    info = await console.store.get_session(main)
    assert info is not None and info.workspace == str(second.resolve())


async def test_a_stored_folder_does_not_hold_the_main_chat(console: Console) -> None:
    main = console.main_id
    assert main is not None
    old = _home() / "old"
    old.mkdir()
    await console.store.update_session(main, workspace=str(old))
    console.cfg.gateway.workspace = str(_home() / "now")

    entry = await open_session(console.sessions, main)

    assert cast(FakeAgent, entry.agent).workspace_root == str(
        (_home() / "now").resolve()
    )


async def test_the_main_chat_folder_is_not_changed_as_a_chat(
    console: Console, tmp_path: Path
) -> None:
    main = console.main_id
    assert main is not None
    with pytest.raises(ApiError) as refused:
        await console.sessions.change_workspace(main, str(tmp_path))
    assert (refused.value.status_code, refused.value.code) == (409, "main_chat_folder")


async def test_a_side_chat_keeps_its_folder_when_the_default_changes(
    console: Console,
) -> None:
    chosen = _home() / "project"
    chosen.mkdir()
    entry, _ = await console.sessions.create(str(chosen))
    console.cfg.gateway.workspace = str(_home() / "elsewhere")

    await console.sessions.keep(entry.session_id)
    reopened = await open_session(console.sessions, entry.session_id)

    assert cast(FakeAgent, reopened.agent).workspace_root == str(chosen.resolve())
    info = await console.store.get_session(entry.session_id)
    assert info is not None and info.workspace == str(chosen.resolve())


async def test_a_task_refused_its_folder_says_why_in_its_card_and_row(
    console: Console, tmp_path: Path
) -> None:
    from localmelo.support.gateway.console.activity import FOLDER_REFUSED, _task_rows
    from localmelo.support.gateway.console.cards import CARD_EVENT

    # Outside the home folder: a task may not work there, and works nowhere else.
    console.cfg.gateway.workspace = str(tmp_path / "outside")
    (tmp_path / "outside").mkdir()
    task, _ = await console.workers.create({"objective": "Sort the photos"})
    assert task["state"] == "failed"

    await console.cards.flush()
    assert console.main_id is not None
    [card] = await console.store.events_of_type(console.main_id, CARD_EVENT)
    data = card.event["data"]
    assert data["source_id"] == task["task_id"] and data["state"] == "failed"
    assert data["line"] == (
        f"{FOLDER_REFUSED}: A user task's workspace must be strictly inside "
        "your home folder"
    )
    [row] = await _task_rows(console, 0.0, {})
    assert row["result"] == FOLDER_REFUSED
    assert row["folder"].endswith("outside")
