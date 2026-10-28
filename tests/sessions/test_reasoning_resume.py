"""Reasoning after a real pairs/window resume, with in-process providers only.

Requests below are the Message arguments the Agent passes to its provider.
Legacy deliberately leaves thinking there; its wire encoding is the provider's
contract, tested separately. Notes are RAM-only, unlike text in a checkpoint.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from localmelo.melo.agent.options import ReasoningMode
from localmelo.melo.schema import Message
from localmelo.support.sessions.runner import ResumeMode, SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.support.sessions.window import decode_window

from .helpers import (
    DAY,
    Clock,
    ScriptedLLM,
    answer,
    rows,
    run_result,
    started_agent,
    tool_call,
)

MODES: tuple[ReasoningMode, ...] = (
    "legacy",
    "drop",
    "replay",
    "compress",
    "compress+replay",
)
REPLAY_MODES = ("replay", "compress+replay")
COMPRESS_MODES = ("compress", "compress+replay")
# Hand-written test data, not a credential read from a user's environment.
SYNTHETIC_KEY = "sk-" + "abc123feed" * 3
OLD_THINKING = f"Read the grocery note. Test credential: {SYNTHETIC_KEY}."
REDACTED_THINKING = "Read the grocery note. Test credential: [redacted]."
OLD_QUERY = "What do my grocery notes say?"
OLD_ANSWER = "The grocery note says buy milk."
FOLLOW_UP = "Remind me of the grocery item."
HOLDER = "testhost:1:aaaaaaaa"


@pytest.mark.parametrize("resume_mode", ["pairs", "window"])
@pytest.mark.parametrize("mode", MODES)
async def test_resume_uses_current_reasoning_mode_not_recorded_outcome(
    tmp_path: Path, mode: ReasoningMode, resume_mode: ResumeMode
) -> None:
    """A real old turn records the opposite replay policy to the new process."""
    recorded_mode: ReasoningMode = "drop" if mode in REPLAY_MODES else "replay"
    clock = Clock(time.time())
    path = tmp_path / "sessions.db"
    root = tmp_path / "ws"
    async with AsyncExitStack() as cleanup:
        store = SessionStore(path, writer="gateway", clock=clock)
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM(
            [
                replace(
                    tool_call("file_read", {"path": "notes.txt"}),
                    content="Read the grocery note.",
                    thinking=OLD_THINKING,
                ),
                answer(OLD_ANSWER),
            ]
        )
        llm.reasoning_field = "reasoning_content"
        live = await started_agent(
            root,
            llm,
            reasoning=recorded_mode,
            replay_scope="window",
            retriever="off",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(live.close)
        writer = SessionRunner(live, store, None, new_session=NewSession())
        cleanup.push_async_callback(writer.close)
        original = await asyncio.wait_for(writer.run_turn(OLD_QUERY), timeout=5)
        assert original.status == "completed"
        assert original.config["reasoning"] == recorded_mode
        assert [m.thinking for m in live.conversation() if m.thinking] == [OLD_THINKING]
        sid = writer.session_id
        assert sid is not None
        checkpoint = await store.get_window(sid)
        assert checkpoint is not None
        assert SYNTHETIC_KEY not in checkpoint.data
        decoded = decode_window(checkpoint.data)
        assert decoded is not None
        assert [m.thinking for m in decoded if m.thinking] == [REDACTED_THINKING]

    # Reopen the file and use a fresh Agent, not the recording Agent's RAM.
    async with AsyncExitStack() as cleanup:
        store = SessionStore(path, writer="gateway", clock=clock)
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM([answer("Milk, as before.")])
        llm.reasoning_field = "reasoning_content"
        current = await started_agent(
            root,
            llm,
            reasoning=mode,
            replay_scope="window",
            retriever="off",
            note_writer="heuristic",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(current.close)
        runner = SessionRunner(current, store, sid, resume_mode=resume_mode)
        cleanup.push_async_callback(runner.close)
        assert await runner.resume() == 1
        info = runner.last_resume
        assert info is not None
        assert (info.mode, info.fell_back, info.checkpoint_turns) == (
            resume_mode,
            False,
            1 if resume_mode == "window" else 0,
        )
        restored = current.conversation()
        if resume_mode == "pairs":
            assert [(m.role, m.content) for m in restored] == [
                ("user", OLD_QUERY),
                ("assistant", OLD_ANSWER),
            ]
            assert all(not m.thinking and not m.tool_calls for m in restored)
        else:
            assert [m.role for m in restored] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
            assert [m.thinking for m in restored if m.thinking] == [REDACTED_THINKING]
            assert restored[1].tool_calls[0].id == restored[2].tool_call_id
        assert current.notes() == []
        assert current.options.reasoning == mode
        [stored] = await store.list_turns(sid)
        assert stored.result is not None
        assert stored.result["config"]["reasoning"] == recorded_mode != mode

        outcome = await asyncio.wait_for(runner.run_turn(FOLLOW_UP), timeout=5)
        assert outcome.status == "completed"
        [request] = llm.requests
        assert request[-1].content == FOLLOW_UP
        assert any(m.role == "user" and m.content == OLD_QUERY for m in request)
        assert any(m.role == "assistant" and m.content == OLD_ANSWER for m in request)
        expected_thinking = (
            [REDACTED_THINKING]
            if resume_mode == "window" and mode not in ("drop", "compress")
            else []
        )
        assert [m.thinking for m in request if m.thinking] == expected_thinking
        assert SYNTHETIC_KEY not in repr(request)
        if resume_mode == "window":
            [old_step] = [m for m in request if m.tool_calls]
            assert old_step.tool_calls[0].arguments == {"path": "notes.txt"}
            [result] = [m for m in request if m.role == "tool"]
            assert result.tool_call_id == old_step.tool_calls[0].id
            assert "buy milk" in result.content
            # Drop is a send transform, not destruction of the resumed window.
            assert [m.thinking for m in current.conversation() if m.thinking] == [
                REDACTED_THINKING
            ]
        if mode == "legacy":
            assert "reasoning" not in outcome.config
            assert outcome.memory_stats is None
        else:
            assert outcome.config["reasoning"] == mode
            assert outcome.memory_stats is not None
            assert outcome.memory_stats["replayed_chars"] == (
                len(REDACTED_THINKING)
                if resume_mode == "window" and mode in REPLAY_MODES
                else 0
            )
        assert bool(current.notes()) == (mode in COMPRESS_MODES)
        turns = await store.list_turns(sid)
        assert turns[0].result is not None
        assert turns[0].result["config"]["reasoning"] == recorded_mode


@pytest.mark.parametrize("mode", MODES)
async def test_window_resume_obeys_thinking_and_checkpoint_retention(
    tmp_path: Path, mode: ReasoningMode
) -> None:
    """Startup trims old thinking in a young checkpoint, then expires the row."""
    clock = Clock(time.time())
    path = tmp_path / "sessions.db"
    old_stamp = clock() - 29 * DAY
    recent_stamp = clock() - DAY
    old_thinking = "An expired grocery draft."
    checkpoint_messages: list[Message] = []
    async with AsyncExitStack() as cleanup:
        store = SessionStore(
            path, writer="gateway", clock=clock, thinking_retention_days=30
        )
        cleanup.push_async_callback(store.aclose)
        session = await store.create_session()
        for query, text, thinking, stamp in (
            ("Older question", "Older answer", old_thinking, old_stamp),
            ("Recent question", "Recent answer", OLD_THINKING, recent_stamp),
        ):
            turn = await store.begin_turn(session.id, query, holder=HOLDER)
            checkpoint_messages.extend(
                [
                    Message("user", query, timestamp=stamp),
                    Message(
                        "assistant",
                        (
                            f"<think>{thinking}</think>{text}"
                            if stamp == old_stamp
                            else text
                        ),
                        thinking=thinking,
                        timestamp=stamp,
                    ),
                ]
            )
            await store.finish_turn(
                session.id,
                turn.turn,
                holder=HOLDER,
                status="completed",
                cause="run",
                result=run_result(result=text, config={"reasoning": "replay"}),
                answer=text,
                window=checkpoint_messages,
            )
        saved = await store.get_window(session.id)
        assert saved is not None
        messages = decode_window(saved.data)
        assert messages is not None
        assert [m.thinking for m in messages if m.thinking] == [
            old_thinking,
            REDACTED_THINKING,
        ]

    # The checkpoint row is two days old, but its oldest thinking is now 31.
    clock.advance(2 * DAY)
    async with AsyncExitStack() as cleanup:
        store = SessionStore(
            path, writer="gateway", clock=clock, thinking_retention_days=30
        )
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM([answer("First follow-up."), answer("After expiry.")])
        llm.reasoning_field = "reasoning_content"
        current = await started_agent(
            tmp_path / "ws",
            llm,
            reasoning=mode,
            replay_scope="window",
            retriever="off",
            note_writer="heuristic",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(current.close)
        runner = SessionRunner(current, store, session.id, resume_mode="window")
        cleanup.push_async_callback(runner.close)
        assert await runner.resume() == 2
        assert runner.last_resume is not None
        assert (runner.last_resume.mode, runner.last_resume.checkpoint_turns) == (
            "window",
            2,
        )
        restored = current.conversation()
        assert [m.thinking for m in restored if m.thinking] == [REDACTED_THINKING]
        assert restored[1].content == "Older answer"
        assert old_thinking not in repr(restored)
        assert restored[3].content == "Recent answer"

        outcome = await asyncio.wait_for(runner.run_turn(FOLLOW_UP), timeout=5)
        assert outcome.status == "completed"
        [request] = llm.requests
        assert [m.thinking for m in request if m.thinking] == (
            [] if mode in ("drop", "compress") else [REDACTED_THINKING]
        )
        assert old_thinking not in repr(request)
        assert SYNTHETIC_KEY not in repr(request)

        clock.advance(31 * DAY)
        await store.apply_retention()
        assert await store.get_window(session.id) is None
        assert await runner.resume() == 3
        assert runner.last_resume is not None
        assert (
            runner.last_resume.mode,
            runner.last_resume.fell_back,
            runner.last_resume.checkpoint_turns,
        ) == ("pairs", True, 0)
        assert all(not m.thinking for m in current.conversation())
        assert all("<think>" not in m.content for m in current.conversation())
        outcome = await asyncio.wait_for(runner.run_turn("After retention?"), timeout=5)
        assert outcome.status == "completed"
        assert len(llm.requests) == 2
        assert all(not m.thinking for m in llm.requests[-1])
        assert all("<think>" not in m.content for m in llm.requests[-1])


@pytest.mark.parametrize("mode", ["drop", "compress"])
async def test_non_replay_resume_does_not_send_legacy_inline_thinking(
    tmp_path: Path, mode: ReasoningMode
) -> None:
    """Changing modes must also drop a legacy checkpoint's inline copy."""
    path = tmp_path / "sessions.db"
    root = tmp_path / "ws"
    thinking = "Private synthetic grocery deliberation."
    async with AsyncExitStack() as cleanup:
        store = SessionStore(path, writer="gateway")
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM(
            [
                replace(
                    tool_call("file_read", {"path": "notes.txt"}),
                    content=f"<think>{thinking}</think>Read the grocery note.",
                    thinking=thinking,
                ),
                answer(OLD_ANSWER),
            ]
        )
        live = await started_agent(
            root,
            llm,
            reasoning="legacy",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(live.close)
        writer = SessionRunner(live, store, None, new_session=NewSession())
        cleanup.push_async_callback(writer.close)
        outcome = await asyncio.wait_for(writer.run_turn(OLD_QUERY), timeout=5)
        assert outcome.status == "completed"
        assert "reasoning" not in outcome.config
        sid = writer.session_id
        assert sid is not None

    async with AsyncExitStack() as cleanup:
        store = SessionStore(path, writer="gateway")
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM([answer("Milk, as before.")])
        current = await started_agent(
            root,
            llm,
            reasoning=mode,
            replay_scope="window",
            retriever="off",
            note_writer="heuristic",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(current.close)
        runner = SessionRunner(current, store, sid, resume_mode="window")
        cleanup.push_async_callback(runner.close)
        assert await runner.resume() == 1
        assert any(m.thinking == thinking for m in current.conversation())
        outcome = await asyncio.wait_for(runner.run_turn(FOLLOW_UP), timeout=5)
        assert outcome.status == "completed"
        assert outcome.config["reasoning"] == mode
        [request] = llm.requests
        assert all(not m.thinking for m in request)
        assert all(thinking not in m.content for m in request)
        assert all("<think>" not in m.content for m in request)


@pytest.mark.parametrize("resume_mode", ["pairs", "window"])
@pytest.mark.parametrize("mode", ["compress", "compress+replay"])
async def test_resume_keeps_checkpoint_note_text_not_the_ram_note_store(
    tmp_path: Path, mode: ReasoningMode, resume_mode: ResumeMode
) -> None:
    """turns.note is a digest, not a serialized NoteStore to restore in R2b."""
    path = tmp_path / "sessions.db"
    root = tmp_path / "ws"
    model_note = "<note>plan: read notes.txt for the grocery reminder</note>"
    async with AsyncExitStack() as cleanup:
        store = SessionStore(path, writer="gateway")
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM(
            [
                replace(
                    tool_call("file_read", {"path": "notes.txt"}),
                    content=model_note,
                    thinking="Read the saved grocery reminder.",
                ),
                answer(OLD_ANSWER),
            ]
        )
        llm.reasoning_field = "reasoning_content"
        live = await started_agent(
            root,
            llm,
            reasoning=mode,
            replay_scope="window",
            retriever="off",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(live.close)
        writer = SessionRunner(live, store, None, new_session=NewSession())
        cleanup.push_async_callback(writer.close)
        outcome = await asyncio.wait_for(writer.run_turn(OLD_QUERY), timeout=5)
        assert outcome.status == "completed"
        assert live.notes()
        assert any(entry.kind == "digest" for entry in live.notes())
        sid = writer.session_id
        assert sid is not None
        [(digest,)] = rows(
            path, "SELECT note FROM turns WHERE session_id = ? AND turn = 1", sid
        )
        assert digest.startswith(f"task: {OLD_QUERY}")
        assert "outcome: completed" in digest

    async with AsyncExitStack() as cleanup:
        store = SessionStore(path, writer="gateway")
        cleanup.push_async_callback(store.aclose)
        llm = ScriptedLLM([answer("Milk, as before.")])
        llm.reasoning_field = "reasoning_content"
        current = await started_agent(
            root,
            llm,
            reasoning=mode,
            replay_scope="window",
            retriever="off",
            note_writer="heuristic",
            attempt_policy="none",
            max_steps=3,
        )
        cleanup.push_async_callback(current.close)
        runner = SessionRunner(current, store, sid, resume_mode=resume_mode)
        cleanup.push_async_callback(runner.close)
        assert current.notes() == []
        assert await runner.resume() == 1
        assert current.notes() == []
        assert (model_note in "\n".join(m.content for m in current.conversation())) == (
            resume_mode == "window"
        )
        assert rows(
            path, "SELECT note FROM turns WHERE session_id = ? AND turn = 1", sid
        ) == [(digest,)]

        outcome = await asyncio.wait_for(runner.run_turn(FOLLOW_UP), timeout=5)
        assert outcome.status == "completed"
        [request] = llm.requests
        sent_text = "\n".join(m.content for m in request)
        assert (model_note in sent_text) == (resume_mode == "window")
        assert digest not in sent_text
