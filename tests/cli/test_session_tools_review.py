"""Regressions reproduced from the independent session-command review."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from localmelo.melo.schema import SHORT_TERM_MAX, Message
from localmelo.support.cli.session_views import memory_view
from localmelo.support.cli.theme import ASCII

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed
from .test_session_views import render


@pytest.mark.parametrize("switch", ["new", "resume"])
async def test_stale_title_read_cannot_overwrite_a_different_session(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
    switch: str,
) -> None:
    original = await seed(setup, "Old session title")
    target = await seed(setup, "Target session title")
    setup.session.enter(f"/resume {original}")
    await setup.idle()
    get = setup.store.get_session
    reading, release = asyncio.Event(), asyncio.Event()
    intercepted = False

    async def delayed_read(sid: str) -> Any:
        nonlocal intercepted
        info = await get(sid)
        if sid == original and not intercepted:
            intercepted = True
            reading.set()
            await release.wait()
        return info

    monkeypatch.setattr(setup.store, "get_session", delayed_read)
    task = setup.session.spawn(setup.session.session_tools.title(""))
    await asyncio.wait_for(reading.wait(), 3)
    setup.session.enter("/new" if switch == "new" else f"/resume {target}")
    await setup.idle()
    expected = "" if switch == "new" else "Target session title"
    assert setup.session.title == expected
    release.set()
    await task
    assert setup.session.title == expected


async def test_rename_refreshes_resume_completion_before_starting_a_new_chat(
    setup: Setup,
) -> None:
    sid = await seed(setup, "underlying task")
    assert await setup.store.rename_session(sid, "Initial custom title")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    assert "Initial custom title" in setup.session.navigation.refs
    await setup.session.session_tools.title("New custom title")
    setup.session.enter("/new")
    await setup.idle()
    assert "New custom title" in setup.session.navigation.refs
    assert "Initial custom title" not in setup.session.navigation.refs
    assert await setup.session.navigation.resolve("Initial custom title") == []
    assert (await setup.session.navigation.resolve("New custom title"))[0].id == sid


def test_ascii_memory_clipping_keeps_ascii_markers_and_cell_width(setup: Setup) -> None:
    text = render(
        memory_view(
            [Message("user", "x" * 100)],
            setup.factory.backend,
            window_max=SHORT_TERM_MAX,
            memory=True,
            loaded=None,
            width=40,
            symbols=ASCII,
        )
    )
    row = next(line for line in text.splitlines() if line.startswith("  > "))
    assert row.isascii()
    assert len(row) <= 40
    assert row.endswith(ASCII.ellipsis)
