"""Independent-review regressions for undo, retry targeting and feedback."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from localmelo.support.cli.text import cells
from localmelo.tests.sessions.helpers import answer

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import scripted, seed


async def two_turns(setup: Setup) -> str:
    scripted(setup).replies.extend(
        [answer("replacement"), answer("second replacement")]
    )
    sid = await seed(setup, "first request")
    turn = await setup.store.begin_turn(
        sid,
        "second request\n\nfull stored attachment",
        title="Second typed request.",
        holder="other",
    )
    await setup.store.finish_turn(
        sid,
        turn.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="second answer",
    )
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    return sid


async def test_retry_after_undo_reruns_the_undone_turn_without_undoing_the_previous(
    setup: Setup,
) -> None:
    sid = await two_turns(setup)
    setup.session.enter("/undo")
    await setup.idle()
    setup.session.enter("/retry")
    await setup.idle()
    turns = await setup.store.list_turns(sid)
    assert [(turn.turn, turn.undone) for turn in turns] == [
        (1, False),
        (2, True),
        (3, False),
    ]
    assert turns[2].user_text == turns[1].user_text
    assert turns[2].typed_text == "Second typed request."
    text = "\n".join(message.content for message in scripted(setup).requests[-1])
    assert "first request" in text and "second request" in text
    assert "Retrying turn 2:" in setup.out.text()
    assert setup.session.navigation.mark.undone == 1
    assert "changed in another window" not in setup.out.text()


async def test_retry_can_undo_the_replacement_after_retrying_an_undone_turn(
    setup: Setup,
) -> None:
    sid = await two_turns(setup)
    for command in ("/undo", "/retry", "/retry"):
        setup.session.enter(command)
        await setup.idle()
    turns = await setup.store.list_turns(sid)
    assert [(turn.turn, turn.undone) for turn in turns] == [
        (1, False),
        (2, True),
        (3, True),
        (4, False),
    ]
    assert turns[3].user_text == turns[1].user_text
    assert setup.session.navigation.mark.undone == 2


async def test_cancelled_committed_undo_reports_success_and_refreshes_refs(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup, "typed request.")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    assert setup.session.runner is not None
    undo = setup.session.runner.undo
    refresh = setup.session.navigation.refresh_refs
    committed, release = asyncio.Event(), asyncio.Event()
    refreshed = []

    async def delayed_undo() -> Any:
        result = await undo()
        committed.set()
        await release.wait()
        return result

    async def tracked_refresh() -> None:
        await refresh()
        refreshed.append(True)

    monkeypatch.setattr(setup.session.runner, "undo", delayed_undo)
    monkeypatch.setattr(setup.session.navigation, "refresh_refs", tracked_refresh)
    setup.session.enter("/undo")
    await asyncio.wait_for(committed.wait(), 3)
    hint = setup.session.enter("next message while undo settles")
    assert hint is not None and "esc" not in hint.lower() and "undo" in hint.lower()
    assert setup.session.stop()
    await asyncio.sleep(0)
    assert setup.session.stop()
    release.set()
    await setup.idle()
    text = " ".join(setup.out.text().split())
    assert "Undid turn 1: typed request. Files the tools changed stay changed." in text
    assert text.count("Undid turn 1:") == 1
    assert refreshed == [True]
    assert setup.session.navigation.mark.undone == 1
    assert (await setup.store.list_turns(sid))[0].undone


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("Punctuated.\nsecond line", "Punctuated."),
        ("Question?\nsecond line", "Question?"),
        ("Plain first line\nsecond line", "Plain first line."),
    ],
)
async def test_undo_reports_only_the_first_line_with_one_sentence_ending(
    setup: Setup,
    text: str,
    first: str,
) -> None:
    sid = await seed(setup, text)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    before = setup.out.text()
    setup.session.enter("/undo")
    await setup.idle()
    tail = " ".join(setup.out.text()[len(before) :].split())
    assert f"Undid turn 1: {first} Files the tools changed stay changed." in tail
    assert "second line" not in tail


async def test_undo_clips_wide_first_lines_at_a_cell_boundary(setup: Setup) -> None:
    sid = await seed(setup, "界" * 80 + "\nsecond line")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    before = setup.out.text()
    setup.session.enter("/undo")
    await setup.idle()
    tail = " ".join(setup.out.text()[len(before) :].split())
    prefix = tail.split("Undid turn 1: ", 1)[1].split(" Files", 1)[0]
    assert cells(prefix) <= 60
    assert prefix.endswith(setup.session.renderer.symbols.ellipsis)
    assert "second line" not in tail
