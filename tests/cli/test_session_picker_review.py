"""Selection snapshots reject stale results and keep lookup out of completion."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from localmelo.support.cli.completion import SlashCompleter
from localmelo.support.cli.picker import SessionPicker
from localmelo.support.cli.theme import UNICODE

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed


@pytest.mark.parametrize("outcome", ["rows", "error"])
async def test_cancel_insensitive_old_query_cannot_replace_newer_state(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    sid = await seed(setup)
    info = await setup.store.get_session(sid)
    assert info is not None
    old, new = replace(info, title="old"), replace(info, title="new")
    entered, release = asyncio.Event(), asyncio.Event()

    async def query(**kwargs: Any) -> Any:
        if kwargs["query"] == "":
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()  # emulate a driver which settles despite cancel
            if outcome == "error":
                raise RuntimeError("stale query failed")
            return [old]
        return [new]

    monkeypatch.setattr(setup.store, "list_sessions", query)
    picker = SessionPicker(
        setup.store,
        setup.session.agent.workspace_root or "",
        width=lambda: 100,
        invalidate=lambda: None,
        symbols=UNICODE,
    )
    picker.refresh(delay=0)
    stale = picker._load
    assert stale is not None
    await entered.wait()
    picker.buffer.text = "new"
    picker.refresh(delay=0)
    assert picker._load is not None
    await picker._load
    release.set()
    await stale
    assert picker.rows == [new]
    assert not picker.answer.done()
    picker.close()


@pytest.mark.parametrize("fixed", [False, True])
async def test_picker_caps_database_and_resolved_choice_lists_at_fifty(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
    fixed: bool,
) -> None:
    sid = await seed(setup)
    info = await setup.store.get_session(sid)
    assert info is not None
    rows = [
        replace(info, id=f"{i:012x}", title=f"chat {i}", busy=(i == 0))
        for i in range(60)
    ]

    async def query(**kwargs: Any) -> Any:
        assert not fixed, "fixed choices must not query the database"
        assert kwargs["limit"] == 50 and kwargs["min_turns"] == 1
        return rows[: kwargs["limit"]]

    monkeypatch.setattr(setup.store, "list_sessions", query)
    picker = SessionPicker(
        setup.store,
        setup.session.agent.workspace_root or "",
        width=lambda: 100,
        invalidate=lambda: None,
        symbols=UNICODE,
        choices=rows if fixed else None,
    )
    picker.refresh(delay=0)
    assert picker._load is not None
    await picker._load
    assert len(picker.rows) == 50
    assert "running elsewhere" in picker._rows()
    if fixed:
        assert "matching sessions" in picker._header()
    picker.close()


async def test_resume_completion_uses_only_the_cached_snapshot(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup, "Cached chat")
    await setup.session.navigation.refresh_refs()

    async def forbidden(**kwargs: Any) -> Any:
        raise AssertionError("completion must not query SQLite")

    monkeypatch.setattr(setup.store, "list_sessions", forbidden)
    completer = SlashCompleter(sessions=lambda: setup.session.navigation.refs)
    values = [
        c.text for c in completer.get_completions(Document("/resume "), CompleteEvent())
    ]
    assert values == [sid, "Cached chat"]


async def test_sessions_command_passes_its_filter_to_the_picker(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    async def choose(*args: Any, **kwargs: Any) -> None:
        seen.append(kwargs["query"])

    monkeypatch.setattr(setup.app, "choose_session", choose, raising=False)
    setup.session.enter("/sessions fix tests")
    await setup.idle()
    assert seen == ["fix tests"]
