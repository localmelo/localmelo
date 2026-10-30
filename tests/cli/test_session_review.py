"""Review regressions: startup identity, typed errors and post-load rollback."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.sessions import BUSY, DELETED
from localmelo.support.cli.views import banner
from localmelo.support.sessions.store import SessionInfo, SessionStoreError

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub
from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed
from .ui_helpers import backend


def test_banner_names_the_selected_resumed_session() -> None:
    from localmelo.support.cli.theme import UNICODE

    out = RecordingOut(100)
    out.print(
        banner(
            version="test",
            backend=backend(),
            workspace="/demo",
            tools=[],
            memory=False,
            yolo=False,
            width=100,
            symbols=UNICODE,
            session_title="Chosen chat",
            developer=True,
        )
    )
    assert 'session    "Chosen chat" (resumed)' in out.text()
    assert "session    new" not in out.text()


async def test_resume_failure_after_load_rolls_back_conversation_runner_and_draft(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = await seed(setup, "original context")
    target = await seed(setup, "target context")
    setup.session.enter(f"/resume {first}")
    await setup.idle()
    previous = setup.session.agent.conversation()
    runner = setup.session.runner
    setup.app.buffer.text = "unsent draft"
    loaded: list[bool] = []

    async def failing_metadata(info: SessionInfo) -> None:
        assert info.id == target
        assert setup.session.agent.conversation() != previous
        loaded.append(True)
        raise SessionStoreError("metadata unavailable after load")

    monkeypatch.setattr(setup.session.navigation, "metadata", failing_metadata)
    setup.session.enter(f"/resume {target}")
    await setup.idle()
    assert loaded == [True]
    assert setup.session.runner is runner
    assert setup.session.agent.conversation() == previous
    assert setup.app.buffer.text == f"/resume {target}\nunsent draft"


async def test_resume_retains_display_preferences_but_clears_call_state(
    setup: Setup,
) -> None:
    sid = await seed(setup)
    renderer = setup.session.renderer
    renderer.verbose, renderer.thinking = True, True  # not the default mode's
    renderer.progress = {"level": 1}
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    assert renderer.verbose and renderer.thinking
    assert renderer.progress is None and renderer.tool_outputs == []
    await setup.session.new()
    assert not renderer.verbose and not renderer.thinking


@pytest.mark.parametrize("kind,message", [("deleted", DELETED), ("busy", BUSY)])
def test_pre_turn_store_errors_are_human_sentences(kind: str, message: str) -> None:
    from localmelo.support.cli.render import TurnRenderer
    from localmelo.support.cli.theme import UNICODE
    from localmelo.support.sessions.store import SessionBusyError, UnknownSessionError

    out = RecordingOut(140)
    renderer = TurnRenderer(out, symbols=UNICODE, backend=backend)
    renderer.begin_turn("unsent request")
    error = (
        UnknownSessionError(message) if kind == "deleted" else SessionBusyError(message)
    )
    renderer.end_turn(None, error)
    assert f"✗ {message}" in " ".join(out.text().split())
    assert type(error).__name__ not in out.text()
    assert "The turn could not run" not in out.text()


@pytest.mark.parametrize("flag", ["-c", "-r"])
def test_startup_banner_uses_the_resolved_title(
    melo: Callable[..., Melo], flag: str
) -> None:
    first = melo()
    first.see("Ask melo")
    first.type("hello\r")
    first.see("Worked for")
    assert first.quit() == 0
    resumed = melo("--developer", flag, *(["hello"] if flag == "-r" else []))
    resumed.see('Resumed "hello"')
    assert 'session    "hello" (resumed)' in resumed.text()
    assert "session    new" not in resumed.text()
    assert resumed.quit() == 0


def test_startup_picker_precedes_the_banner(melo: Callable[..., Melo]) -> None:
    first = melo()
    first.see("Ask melo")
    first.type("hello\r")
    first.see("Worked for")
    assert first.quit() == 0
    picker = melo("--developer", "-r")
    picker.see("Resume a chat")
    assert "╭─ melo" not in picker.text()
    picker.type("\r")
    picker.see('Resumed "hello"')
    assert 'session    "hello" (resumed)' in picker.text()
    assert picker.quit() == 0
