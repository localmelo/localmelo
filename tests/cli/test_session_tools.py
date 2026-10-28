"""Session editing and inspection commands use stored turns, not rendered output."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from localmelo.melo.schema import Message, RunResult, ToolCall
from localmelo.support.cli import commands

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed


async def wait_for(check: Any) -> None:
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0.01)


async def test_pending_title_is_saved_with_the_first_message(setup: Setup) -> None:
    setup.session.enter("/title A deliberate title")
    await wait_for(lambda: "Title: A deliberate title" in setup.out.text())
    assert await setup.store.list_sessions() == []
    setup.session.enter("the actual typed task")
    await setup.idle()
    [info] = await setup.store.list_sessions()
    assert info.title == "A deliberate title"
    [turn] = await setup.store.list_turns(info.id)
    assert turn.typed_text == "the actual typed task"
    assert setup.session.status_bar().title == info.title


async def test_title_validation_keeps_the_command_draft(setup: Setup) -> None:
    text = "/title " + "x" * 201
    setup.session.enter(text)
    await wait_for(lambda: "A title has 1 to 200 characters." in setup.out.text())
    assert setup.app.buffer.text == text
    assert await setup.store.list_sessions() == []


async def test_undo_restores_memory_and_history_but_not_files(setup: Setup) -> None:
    sid = await seed(setup, "earlier request")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    path = setup.factory.root / "changed.txt"
    path.write_text("files stay changed", encoding="utf-8")
    setup.session.enter("/undo")
    await setup.idle()
    assert (await setup.store.list_turns(sid))[0].undone
    assert setup.session.agent.conversation() == []
    assert path.read_text(encoding="utf-8") == "files stay changed"
    assert "Files the tools changed stay changed." in " ".join(setup.out.text().split())
    assert "earlier request" in setup.app.buffer.history.get_strings()
    assert setup.session.navigation.mark.undone == 1


async def test_usage_marks_undone_work_and_leaves_missing_results_unknown(
    setup: Setup,
) -> None:
    sid = await seed(setup)
    await setup.store.undo_last_turn(sid)
    turn = await setup.store.begin_turn(sid, "recorded work", holder="other")
    result = RunResult(
        task_id="recorded",
        status="completed",
        steps=3,
        n_tool=2,
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        duration_s=1.2,
    )
    await setup.store.finish_turn(
        sid,
        turn.turn,
        holder="other",
        status="completed",
        cause="run",
        result=result.to_dict(),
        answer="answer",
    )
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    setup.session.enter("/usage")
    await wait_for(lambda: "Totals" in setup.out.text())
    text = setup.out.text()
    assert "undone" in text and "?" in text
    assert "100" in text and "20" in text and "1.2s" in text
    assert "Totals include undone turns" in text


async def test_memory_reads_a_copy_with_correct_turn_and_tool_counts(
    setup: Setup,
) -> None:
    messages = [
        Message("user", "question"),
        Message(
            "assistant",
            tool_calls=[
                ToolCall(tool_name="file_read", arguments={"path": "a"}, id="call")
            ],
        ),
        Message("tool", "tool result", tool_call_id="call"),
        Message("assistant", "answer"),
    ]
    setup.session.agent.load_conversation(messages)
    original = setup.session.agent.conversation()
    setup.session.enter("/memory")
    await wait_for(lambda: "Long-term memory:" in setup.out.text())
    text = " ".join(setup.out.text().split())
    assert "4 of 20 messages" in text
    assert "1 turn, 1 with tool steps" in text
    assert "file_read" in text and "tool result" in text
    assert setup.session.agent.conversation() == original


async def test_copy_uses_the_stored_redacted_markdown(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support.cli import terminal

    sid = await seed(setup)
    secret = "sk-" + "f00dfeed12" * 3
    answer = "## Result\n\n```python\nprint('ok')\n```\n" + secret
    turn = await setup.store.begin_turn(sid, "copy a result", holder="other")
    await setup.store.finish_turn(
        sid,
        turn.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="<think>private reasoning</think>\n" + answer,
    )
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    copied: list[str] = []

    async def copy_text(text: str, control: Any) -> str:
        copied.append(text)
        return "pbcopy"

    monkeypatch.setattr(terminal, "copy_text", copy_text)
    setup.session.enter("/copy")
    await wait_for(lambda: "Copied the answer of turn 2" in setup.out.text())
    assert copied == [answer.replace(secret, "[redacted]")]
    assert "private reasoning" not in copied[0]
    assert secret not in copied[0]


def test_session_commands_are_registered_with_their_busy_rules() -> None:
    for name in ("title", "usage", "memory", "copy"):
        command = commands.find(name)
        assert command is not None and command.during_turn is True
    undo = commands.find("undo")
    assert undo is not None and undo.during_turn is False
