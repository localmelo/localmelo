"""Verify additional completed-review findings at the CLI boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from localmelo.melo.schema import RunResult

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed


@pytest.mark.parametrize("command", ["/new", "/undo"])
async def test_cancelling_before_a_command_starts_releases_switching(
    setup: Setup,
    command: str,
) -> None:
    setup.session.enter(command)
    task = setup.session._work
    assert task is not None
    assert setup.session.stop()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert setup.session.state == "idle"
    assert setup.session._work is None


async def test_invalid_utf8_title_changes_no_state(setup: Setup) -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        await setup.session.session_tools.title("invalid \udcff title")
    assert setup.session.session_tools.pending_title is None
    assert setup.session.title == ""
    setup.session.enter("a valid message")
    await setup.idle()
    assert (await setup.rows())[0].status == "completed"


async def test_second_stop_during_undo_settlement_keeps_typed_history(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup, "full input", typed="typed undo input")
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
    setup.session.enter("/undo")
    await asyncio.wait_for(committed.wait(), 3)
    assert setup.session.stop()
    await asyncio.sleep(0)
    assert setup.session.stop()
    release.set()
    await setup.idle()
    assert "typed undo input" in setup.app.buffer.history.get_strings()
    assert setup.session.navigation.mark.undone == 1
    assert (await setup.store.list_turns(sid))[0].undone


@pytest.mark.parametrize(
    "prefix", ["", "<think>reasoning</think>", "reasoning</think>"]
)
async def test_copy_preserves_indented_code_and_trailing_newline(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    from localmelo.support.cli import terminal

    code = "    print(1)\n    print(2)\n"
    sid = await seed(setup)
    turn = await setup.store.begin_turn(sid, "copy code", holder="other")
    await setup.store.finish_turn(
        sid,
        turn.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer=prefix + code,
    )
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    copied: list[str] = []

    async def fake_copy(text: str, control: Any) -> str:
        copied.append(text)
        return "pbcopy"

    monkeypatch.setattr(terminal, "copy_text", fake_copy)
    await setup.session.session_tools.copy("")
    assert copied == [code]


async def test_narrow_usage_does_not_ellipsize_recorded_counters(setup: Setup) -> None:
    sid = await seed(setup)
    turn = await setup.store.begin_turn(sid, "measured work", holder="other")
    result = RunResult(
        task_id="measured",
        status="completed",
        steps=123456,
        n_tool=98765,
        usage={"prompt_tokens": 234567, "completion_tokens": 345678},
        duration_s=12.3,
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
    setup.out._width = 40
    setup.out.console.width = 40
    await setup.session.session_tools.run("usage", "")
    text = setup.out.text()
    assert all(str(n) in text for n in (123456, 98765, 234567, 345678))


async def test_explicit_copy_reads_only_the_selected_turn(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.cli import terminal

    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    read = setup.store.list_turns
    reads: list[dict[str, Any]] = []

    async def bounded(*args: Any, **kwargs: Any) -> Any:
        reads.append(kwargs)
        assert kwargs.get("limit") == 1
        assert kwargs.get("before") == 2
        return await read(*args, **kwargs)

    async def fake_copy(text: str, control: Any) -> str:
        assert text == "stored **answer**"
        return "pbcopy"

    monkeypatch.setattr(setup.store, "list_turns", bounded)
    monkeypatch.setattr(terminal, "copy_text", fake_copy)
    await setup.session.session_tools.copy("1")
    assert reads == [{"limit": 1, "before": 2}]


async def test_default_copy_pages_past_undone_turns(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from localmelo.support.cli import terminal

    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    [original] = await setup.store.list_turns(sid)
    reads: list[dict[str, Any]] = []

    async def pages(*args: Any, **kwargs: Any) -> Any:
        reads.append(kwargs)
        assert kwargs.get("limit") == 50
        if kwargs.get("before") is None:
            return [replace(original, turn=i, undone=True) for i in range(2, 52)]
        assert kwargs.get("before") == 2
        return [original]

    async def fake_copy(text: str, control: Any) -> str:
        assert text == original.answer
        return "pbcopy"

    monkeypatch.setattr(setup.store, "list_turns", pages)
    monkeypatch.setattr(terminal, "copy_text", fake_copy)
    await setup.session.session_tools.copy("")
    assert reads == [{"limit": 50, "before": None}, {"limit": 50, "before": 2}]


@pytest.mark.parametrize("reference", ["9" * 19, str(2**63 - 1)])
async def test_large_copy_reference_is_missing_not_a_sqlite_overflow(
    setup: Setup,
    reference: str,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    with pytest.raises(ValueError, match="has no answer"):
        await setup.session.session_tools.copy(reference)


@pytest.mark.parametrize(
    ("text", "thinking", "answer"),
    [
        ("  code\n", "", "  code\n"),
        ("<think> why </think>  code\n", "why", "  code\n"),
        ("why</think>  code\n", "why", "  code\n"),
        ("<think>unfinished", "unfinished", ""),
    ],
)
def test_reasoning_split_preserves_whitespace_only_when_requested(
    text: str,
    thinking: str,
    answer: str,
) -> None:
    from localmelo.melo.schema import split_thinking

    assert split_thinking(text, strip_answer=False) == (thinking, answer)
    assert split_thinking(text) == (thinking, answer.strip())
