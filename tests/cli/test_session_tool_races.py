"""Session commands remain safe across cancellation and concurrent CLI work."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from localmelo.support.cli.sessions import BUSY, DELETED

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed
from .test_session_tools import wait_for


async def test_undo_busy_keeps_its_draft_and_uses_the_shared_message(
    setup: Setup,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    await setup.store.begin_turn(sid, "another window", holder="other")
    setup.session.enter("/undo")
    await setup.idle()
    assert BUSY in " ".join(setup.out.text().split())
    assert setup.app.buffer.text == "/undo"
    assert not (await setup.store.list_turns(sid))[0].undone


async def test_deleted_undo_uses_the_shared_message_and_keeps_its_draft(
    setup: Setup,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    await setup.store.delete_session(sid)
    setup.session.enter("/undo")
    await setup.idle()
    assert DELETED in " ".join(setup.out.text().split())
    assert setup.app.buffer.text == "/undo"


async def test_cancelled_undo_still_restores_typed_history(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = await seed(setup, "full input with an attachment", typed="typed input")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    assert setup.session.runner is not None
    undo = setup.session.runner.undo
    committed, release = asyncio.Event(), asyncio.Event()

    async def delayed_undo() -> Any:
        turn = await undo()
        committed.set()
        await release.wait()
        return turn

    monkeypatch.setattr(setup.session.runner, "undo", delayed_undo)
    setup.app.buffer.text = "keep this draft"
    setup.session.enter("/undo")
    await asyncio.wait_for(committed.wait(), 3)
    assert setup.session.stop()
    release.set()
    await setup.idle()
    assert (await setup.store.list_turns(sid))[0].undone
    assert "typed input" in setup.app.buffer.history.get_strings()
    assert setup.app.buffer.text == "keep this draft"
    assert setup.session.navigation.mark.undone == 1


async def test_new_and_resume_clear_a_pending_title(setup: Setup) -> None:
    await setup.session.session_tools.title("pending for an unsaved chat")
    setup.session.enter("/new")
    await setup.idle()
    assert setup.session.session_tools.pending_title is None
    assert setup.session.title == ""
    await setup.session.session_tools.title("another unsaved title")
    sid = await seed(setup, "saved target")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    assert setup.session.session_tools.pending_title is None
    assert setup.session.title == "saved target"


async def test_title_during_first_turn_is_applied_after_lazy_creation(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    begin = setup.store.begin_turn
    creating, release = asyncio.Event(), asyncio.Event()

    async def delayed_begin(*args: Any, **kwargs: Any) -> Any:
        creating.set()
        await release.wait()
        return await begin(*args, **kwargs)

    monkeypatch.setattr(setup.store, "begin_turn", delayed_begin)
    setup.session.enter("first request")
    await asyncio.wait_for(creating.wait(), 3)
    await setup.session.session_tools.title("Chosen while starting")
    release.set()
    await setup.idle()
    [info] = await setup.store.list_sessions()
    assert info.title == "Chosen while starting"
    assert setup.session.title == info.title
    assert setup.session.session_tools.pending_title is None


async def test_old_title_write_cannot_replace_new_session_state(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    rename = setup.store.rename_session
    writing, release = asyncio.Event(), asyncio.Event()

    async def delayed_rename(*args: Any, **kwargs: Any) -> bool:
        writing.set()
        await release.wait()
        return await rename(*args, **kwargs)

    monkeypatch.setattr(setup.store, "rename_session", delayed_rename)
    task = setup.session.spawn(setup.session.session_tools.title("old chat title"))
    await asyncio.wait_for(writing.wait(), 3)
    setup.session.enter("/new")
    await setup.idle()
    await setup.session.session_tools.title("new chat title")
    release.set()
    await task
    assert setup.session.title == "new chat title"
    assert setup.session.session_tools.pending_title == "new chat title"


async def test_title_write_is_refused_during_a_session_switch(setup: Setup) -> None:
    setup.session._state = "switching"
    assert setup.session.enter("/title must not rename the old chat") == (
        "Wait for the chat switch to finish before changing its title."
    )
    assert setup.session.session_tools.pending_title is None
    setup.session._state = "idle"


async def test_undo_is_refused_during_a_turn(setup: Setup) -> None:
    setup.session._state = "running"
    hint = setup.session.enter("/undo")
    assert hint is not None and "not available" in hint
    assert await setup.store.list_sessions() == []
    setup.session._state = "idle"


async def test_memory_can_run_during_a_model_turn(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = setup.session.agent._llm
    chat = llm.chat
    responding, release = asyncio.Event(), asyncio.Event()

    async def delayed_chat(*args: Any, **kwargs: Any) -> Any:
        responding.set()
        await release.wait()
        return await chat(*args, **kwargs)

    monkeypatch.setattr(llm, "chat", delayed_chat)
    setup.session.enter("still answering")
    await asyncio.wait_for(responding.wait(), 3)
    before = setup.session.agent.conversation()
    setup.session.enter("/memory window")
    await wait_for(lambda: "Long-term memory:" in setup.out.text())
    assert setup.session.state == "running"
    assert setup.session.agent.conversation() == before
    # The list of notes opens between turns.
    setup.session.enter("/memory")
    await wait_for(lambda: "/memory opens when the turn ends" in setup.out.text())
    assert setup.session.state == "running"
    release.set()
    await setup.idle()
