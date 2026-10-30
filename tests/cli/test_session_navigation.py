"""Resume, retry, durable expansion and cross-window changes on real stores."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.cli.sessions import DELETED
from localmelo.support.sessions.store import (
    SessionStore,
    process_holder,
)
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, core_event, tool_call

from .test_session import Setup
from .test_session import setup as setup


def scripted(env: Setup) -> ScriptedLLM:
    llm = env.session.agent._llm
    assert isinstance(llm, ScriptedLLM)
    return llm


async def seed(
    env: Setup,
    text: str = "earlier question",
    *,
    typed: str | None = None,
    workspace: str | None = None,
    kind: str = "chat",
) -> str:
    info = await env.store.create_session(
        workspace=workspace or env.session.agent.workspace_root,
        model="old",
        kind=kind,
    )
    turn = await env.store.begin_turn(info.id, text, title=typed, holder="other")
    await env.store.finish_turn(
        info.id,
        turn.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="stored **answer**",
    )
    return info.id


async def test_resume_keeps_workspace_and_uses_stored_typed_text(setup: Setup) -> None:
    text = (
        'typed question\n\n<attachment name="notes.txt" chars="4">\nbody\n</attachment>'
    )
    sid = await seed(setup, text, typed="typed question", workspace="/other")
    async with setup.store.component_transaction("legacy_fixture") as conn:
        await conn.execute(
            "UPDATE turns SET attachments = NULL WHERE session_id = ?", (sid,)
        )
    setup.session.approvals._policy._granted.add("file_write")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    session = setup.session
    assert session.runner is not None and session.runner.session_id == sid
    assert not session.approvals.allowed_tools
    assert session.agent.workspace_root == str(setup.factory.root)
    info = await setup.store.get_session(sid)
    assert info is not None and info.workspace == session.agent.workspace_root
    assert info.model == setup.factory.backend.model
    assert "Resumed" in setup.out.text()
    assert "attached notes.txt (4 chars)" in setup.out.text()
    assert "1 turn is back in the model's memory." in setup.out.text()
    assert "tool calls are not" not in setup.out.text()  # developer mode says it
    assert "● stored answer" in setup.out.text()  # the Markdown shows as text
    assert "last worked in /other" in setup.out.text()
    session.enter("/retry")
    await setup.idle()
    rows = await setup.store.list_turns(sid)
    assert rows[0].undone
    assert rows[-1].user_text == text
    assert rows[-1].typed_text == "typed question"
    setup.out.console.export_text()
    session.enter("/new")
    await setup.idle()
    session.enter(f"/resume {sid}")
    await setup.idle()
    assert (
        "typed question" in setup.out.text()
        and "<attachment name=" not in setup.out.text()
    )
    assert "changed in another window" not in setup.out.text()


async def test_resolution_filters_exact_nonchat_and_empty_ids(setup: Setup) -> None:
    task = await seed(setup, "hidden task", kind="task")
    empty = await setup.store.create_session()
    assert await setup.session.navigation.resolve(task) == []
    assert await setup.session.navigation.resolve(empty.id) == []
    for ref in (task, empty.id, "no such session"):
        setup.session.enter(f"/resume {ref}")
        await setup.idle()
        assert "No chat matches" in setup.out.text()
        assert setup.session.runner is None


async def test_foreign_turn_and_undo_reload_before_next_send(
    setup: Setup, tmp_path: Path
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    other = SessionStore(tmp_path / "sessions.db", writer="cli")
    try:
        await other.undo_last_turn(sid)
        turn = await other.begin_turn(sid, "foreign correction", holder="other")
        await other.finish_turn(
            sid,
            turn.turn,
            holder="other",
            status="completed",
            cause="run",
            result=None,
            answer="foreign answer",
        )
        setup.session.enter("next question")
        await setup.idle()
        text = "\n".join(m.content for m in scripted(setup).requests[-1])
        assert "foreign correction" in text
        assert "earlier question" not in text
        assert setup.out.text().count("changed in another window") == 1
    finally:
        await other.aclose()


async def test_deleted_between_precheck_and_begin_keeps_draft(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    begin = setup.store.begin_turn

    async def deleting(*args: Any, **kwargs: Any) -> Any:
        await setup.store.delete_session(sid)
        return await begin(*args, **kwargs)

    monkeypatch.setattr(setup.store, "begin_turn", deleting)
    setup.session.enter("kept request")
    await setup.idle()
    assert setup.app.buffer.text == "kept request"
    assert DELETED in " ".join(setup.out.text().split())
    assert not scripted(setup).requests


async def test_a_live_other_holder_rejects_send_without_losing_draft(
    setup: Setup,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    holder = process_holder("cli").rsplit(":", 1)[0] + ":ffffffff"
    turn = await setup.store.begin_turn(sid, "elsewhere", holder=holder)
    try:
        setup.session.enter("try after the other window")
        await setup.idle()
        assert setup.app.buffer.text == "try after the other window"
        assert "running a turn in another melo window" in " ".join(
            setup.out.text().split()
        )
    finally:
        await setup.store.finish_turn(
            sid,
            turn.turn,
            holder=holder,
            status="completed",
            cause="run",
            result=None,
            answer="done",
        )


async def test_failed_resume_does_not_replace_conversation_or_runner(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = await seed(setup, "original")
    second = await seed(setup, "deleted target")
    setup.session.enter(f"/resume {first}")
    await setup.idle()
    runner = setup.session.runner
    previous = setup.session.agent.conversation()
    read = setup.store.list_turns

    async def deleting(sid: str, **kwargs: Any) -> Any:
        if sid == second:
            await setup.store.delete_session(second)
        return await read(sid, **kwargs)

    monkeypatch.setattr(setup.store, "list_turns", deleting)
    setup.session.enter(f"/resume {second}")
    await setup.idle()
    assert setup.session.runner is runner
    assert setup.session.agent.conversation() == previous
    assert DELETED in " ".join(setup.out.text().split())


async def test_saved_hint_is_computed_before_store_close(setup: Setup) -> None:
    assert await setup.session.navigation.saved_hint() == ""
    setup.session.enter("save this")
    await setup.idle()
    await setup.session.shutdown()
    assert "Resume it with: melo -r" in setup.session.saved_resume_hint


async def test_the_exit_hint_says_melo_c_for_the_newest_chat_of_this_folder(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(setup.factory.root)
    setup.session.enter("save this")
    await setup.idle()
    await setup.session.shutdown()
    assert (
        setup.session.saved_resume_hint
        == '◆ Saved "save this". Continue it with: melo -c'
    )


async def test_the_exit_hint_names_the_id_when_melo_c_would_not_find_the_chat(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup.session.enter("save this")
    await setup.idle()
    assert setup.session.runner is not None and setup.session.runner.session_id
    sid = setup.session.runner.session_id
    named = f'◆ Saved "save this". Resume it with: melo -r {sid}'
    # In another folder, `melo -c` would look there.
    monkeypatch.chdir(tmp_path)
    assert await setup.session.navigation.saved_hint() == named
    # Here, but another chat of this folder is newer.
    monkeypatch.chdir(setup.factory.root)
    assert (
        await setup.session.navigation.saved_hint()
        == '◆ Saved "save this". Continue it with: melo -c'
    )
    await asyncio.sleep(0.02)
    await seed(setup, "written in another window")
    assert await setup.session.navigation.saved_hint() == named


async def test_own_turns_do_not_trigger_spurious_reload(setup: Setup) -> None:
    scripted(setup).replies.extend([answer("second answer")])
    for text in ("one", "two"):
        setup.session.enter(text)
        await setup.idle()
    assert "changed in another window" not in setup.out.text()


async def test_persisted_expand_reads_past_a_thousand_events(setup: Setup) -> None:
    info = await setup.store.create_session(
        workspace=setup.session.agent.workspace_root
    )
    turn = await setup.store.begin_turn(info.id, "read", holder="other")
    events = [core_event("step.started", seq=i + 1) for i in range(1001)]
    events.append(
        core_event("tool.finished", seq=1002, output="past the first event page")
    )
    for start in range(0, len(events), 200):
        await setup.store.append_events(
            info.id, turn.turn, events[start : start + 200], holder="other"
        )
    await setup.store.finish_turn(
        info.id,
        turn.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="done",
    )
    setup.session.enter(f"/resume {info.id}")
    await setup.idle()
    assert setup.session.renderer.tool_outputs == []
    setup.session.enter("/expand 1.1")
    async with asyncio.timeout(3):
        while "past the first event page" not in setup.out.text():
            await asyncio.sleep(0.01)
    assert setup.session.renderer.tool_outputs == []  # no fake live events replayed


async def test_new_calls_after_resume_use_durable_turn_numbers(setup: Setup) -> None:
    sid = await seed(setup)
    scripted(setup).replies.clear()
    scripted(setup).replies.extend(
        [
            tool_call("file_read", {"path": "notes.txt"}),
            answer("done"),
        ]
    )
    (setup.factory.root / "notes.txt").write_text("content", encoding="utf-8")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    setup.session.enter("read notes")
    await setup.idle()
    assert setup.session.renderer.tool_outputs[0].ref == "2.1"


@pytest.mark.parametrize("continue_session,ref", [(True, None), (False, "")])
async def test_empty_startup_creates_no_session(
    setup: Setup, continue_session: bool, ref: str | None
) -> None:
    await setup.session.navigation.startup(continue_session, ref)
    assert setup.session.runner is None
    assert await setup.store.list_sessions() == []
    assert "No earlier chat" in setup.out.text() or "No chats yet" in setup.out.text()


async def test_missing_startup_ref_exits_two(
    setup: Setup, capsys: pytest.CaptureFixture[str]
) -> None:
    await setup.session.navigation.startup(False, "not-found")
    assert setup.session.exit_code == 2
    assert setup.app.exited
    assert (
        capsys.readouterr().err
        == "melo: no chat matches 'not-found' (melo -r lists them)\n"
    )


@pytest.mark.parametrize(
    "checkpoint,tail", [(False, False), (True, False), (True, True)]
)
async def test_window_resume_plumbing_and_fallback_notes(
    setup: Setup, checkpoint: bool, tail: bool
) -> None:
    if checkpoint:
        setup.session.enter("first question")
        await setup.idle()
        assert setup.session.runner is not None
        sid = setup.session.runner.session_id
        assert sid is not None
    else:
        sid = await seed(setup)
    if tail:
        turn = await setup.store.begin_turn(sid, "tail question", holder="other")
        await setup.store.finish_turn(
            sid,
            turn.turn,
            holder="other",
            status="completed",
            cause="run",
            result=None,
            answer="tail answer",
        )
    setup.session.navigation.mode = "window"
    setup.session.developer = True  # the note names what became of the tool steps
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    runner = setup.session.runner
    assert runner is not None and runner.last_resume is not None
    loaded = runner.last_resume
    assert loaded.fell_back == (not checkpoint)
    assert loaded.mode == ("window" if checkpoint else "pairs")
    assert loaded.checkpoint_turns == (1 if checkpoint else 0)
    assert loaded.turns == (2 if tail else 1)
    text = " ".join(setup.out.text().split())
    if tail:
        assert "1 later turn without them" in text
    elif checkpoint:
        assert "with their tool steps" in text
    else:
        assert "its tool calls are not" in text


async def test_query_fallback_opens_picker_even_for_one_match(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = await seed(setup, "Unique subject to find")
    seen: list[str] = []

    async def choose(store: SessionStore, workspace: str, **kwargs: Any) -> Any:
        choices = kwargs["choices"]
        seen.extend(row.id for row in choices)
        return choices[0]

    monkeypatch.setattr(setup.app, "choose_session", choose, raising=False)
    setup.session.enter("/resume subject")
    await setup.idle()
    assert seen == [sid]
    assert setup.session.runner is not None and setup.session.runner.session_id == sid
