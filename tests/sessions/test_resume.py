"""resume(): the pairs arm, the window arm, their fallbacks and their counts."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import NO_ANSWER_LABEL, SESSION_TEXT_CHARS, clip_session_text
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.attachments import TextAttachment, compose_user_text
from localmelo.support.sessions.runner import RESUME_BUDGET_CHARS, SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore

from .helpers import (
    DAY,
    Clock,
    ScriptedLLM,
    answer,
    started_agent,
    tool_call,
    workspace,
)

HOLDER = "testhost:1:aaaaaaaa"
SECRET = "sk-" + "c0ffee1234" * 3


async def stored_session(store: SessionStore, turns: list[tuple[str, str, str]]) -> str:
    """A session whose turns are (user text, status, answer), written directly."""
    session = await store.create_session()
    for text, status, reply in turns:
        turn = await store.begin_turn(session.id, text, holder=HOLDER)
        if status == "running":
            continue
        await store.finish_turn(
            session.id,
            turn.turn,
            holder=HOLDER,
            status=status,
            cause="run",
            result=None,
            answer=reply,
        )
    return session.id


async def runner_for(
    tmp_path: Path,
    store: SessionStore,
    session_id: str,
    mode: str = "pairs",
    **options: Any,
) -> tuple[SessionRunner, Agent, ScriptedLLM]:
    llm = ScriptedLLM([answer("next answer")])
    agent = await started_agent(tmp_path / "ws", llm, **options)
    runner = SessionRunner(agent, store, session_id, resume_mode=mode)  # type: ignore[arg-type]
    return runner, agent, llm


def roles_alternate(messages: list[Any]) -> bool:
    roles = [m.role for m in messages if m.role in ("user", "assistant")]
    return all(a != b for a, b in zip(roles, roles[1:], strict=False))


# ── pairs ───────────────────────────────────────────────────────────────────


async def test_attachment_snapshot_replayed_without_clipping(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    session = await store.create_session()
    runner, agent, llm = await runner_for(tmp_path, store, session.id)
    typed = 'Look\n\n<attachment name="fake" chars="1">\nx\n</attachment>'
    files = (TextAttachment("notes.txt", "x" * 9000 + "</attachment> </ATTACHMENT>"),)
    await runner.run_turn(typed, attachments=files)
    turn = (await store.list_turns(session.id))[0]
    expected = compose_user_text(typed, files)
    assert "<\\/ATTACHMENT>" in expected
    assert turn.typed_text == typed and turn.attachments == files
    assert turn.user_text == expected
    assert llm.requests[0][-1].content == expected
    await agent.close()
    runner2, agent2, _ = await runner_for(tmp_path, store, session.id)
    assert await runner2.resume() == 1
    assert agent2.conversation()[0].content == expected
    await agent2.close()
    sid = await stored_session(store, [("earlier", "completed", "answer")])
    large_runner, large_agent, _ = await runner_for(tmp_path, store, sid)
    await large_runner.run_turn(
        "large", attachments=(TextAttachment("large", "x" * 40000),)
    )
    await large_agent.close()
    resumed, agent3, _ = await runner_for(tmp_path, store, sid)
    assert await resumed.resume() == 1
    assert agent3.conversation()[0].content == "earlier"
    await agent3.close()
    await store.aclose()


async def test_pairs_stay_within_the_budget(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    long = "x" * (SESSION_TEXT_CHARS + 500)
    sid = await stored_session(
        store, [(f"{i} {long}", "completed", long) for i in range(5)]
    )
    runner, agent, _ = await runner_for(tmp_path, store, sid)
    clipped = len(clip_session_text("0 " + long)) + len(clip_session_text(long))
    fits = RESUME_BUDGET_CHARS // clipped
    assert 0 < fits < 5
    assert await runner.resume() == fits
    window = agent.conversation()
    assert len(window) == 2 * fits
    assert window[-2].content.startswith("4 ")  # the newest turns
    assert sum(len(m.content) for m in window) <= RESUME_BUDGET_CHARS
    assert runner.last_resume is not None
    assert runner.last_resume.chars == sum(len(m.content) for m in window)
    await agent.close()
    await store.aclose()


async def test_pairs_skip_running_and_undone_turns_and_label_a_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    sid = await stored_session(
        store,
        [
            ("first", "completed", "one"),
            ("second", "crashed", "[no answer: the run ended with status crashed]"),
            ("third", "completed", "three"),
        ],
    )
    await store.undo_last_turn(sid)
    await store.begin_turn(sid, "fourth, still running", holder=HOLDER)
    runner, agent, llm = await runner_for(tmp_path, store, sid)
    caplog.set_level(logging.WARNING)
    assert await runner.resume() == 2
    assert [(m.role, m.content) for m in agent.conversation()] == [
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
        ("assistant", NO_ANSWER_LABEL.format(status="crashed")),
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    info = runner.last_resume
    assert info is not None
    assert (info.mode, info.fell_back, info.turns, info.checkpoint_turns) == (
        "pairs",
        False,
        2,
        0,
    )
    assert info.messages == 4
    # The next request is valid: the roles alternate and it ends with the query.
    await store.finish_turn(
        sid, 4, holder=HOLDER, status="completed", cause="run", result=None, answer="4"
    )
    await runner.run_turn("fifth")
    request = llm.requests[-1]
    assert roles_alternate(request) and request[-1].content == "fifth"
    await agent.close()
    await store.aclose()


# ── window: the requests after a resume equal the live ones ─────────────────


def recording_transport(
    replies: list[dict[str, Any]], bodies: list[bytes]
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json=replies.pop(0))

    return httpx.MockTransport(handle)


def openai_reply(
    content: str = "", calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"]),
                },
            }
            for call in calls
        ]
    return {"choices": [{"index": 0, "message": message}]}


def ollama_reply(
    content: str = "", calls: list[dict[str, Any]] | None = None, thinking: str = ""
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if calls:
        message["tool_calls"] = [
            {"function": {"name": call["name"], "arguments": call["arguments"]}}
            for call in calls
        ]
    return {"message": message, "done": True}


PROVIDERS: dict[
    str, tuple[Callable[..., BaseLLMProvider], Callable[..., dict[str, Any]]]
] = {
    "openai": (
        lambda transport: OpenAICompatLLM(
            "http://model.test/v1", "m", transport=transport
        ),
        openai_reply,
    ),
    "ollama": (
        lambda transport: OllamaNativeChat(
            "http://model.test", "m", transport=transport
        ),
        ollama_reply,
    ),
}


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
async def test_the_first_request_after_a_window_resume_equals_the_live_one(
    tmp_path: Path, provider: str
) -> None:
    build, reply = PROVIDERS[provider]
    root = workspace(tmp_path / "ws")
    call = {"id": "call_abc", "name": "file_read", "arguments": {"path": "notes.txt"}}
    thinking = "I should read the notes." if provider == "ollama" else ""
    live_bodies: list[bytes] = []
    live_replies = [
        (
            reply("", [call], thinking=thinking)
            if provider == "ollama"
            else reply("", [call])
        ),
        reply(f"The notes say buy milk; your key is {SECRET}."),
        reply("Milk, as before."),
        reply("next"),
    ]
    live = Agent(
        llm=build(recording_transport(live_replies, live_bodies)),
        options=AgentOptions(workspace_root=root, record_answers=True),
    )
    await live.start()
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    runner = SessionRunner(live, store, None, new_session=NewSession())
    await runner.run_turn("What do my notes say?")
    await runner.run_turn("Remind me again")
    assert runner.session_id is not None
    await store.aclose()
    copy = tmp_path / "copy.db"
    shutil.copy(path, copy)  # the state the restart finds
    # The live process goes on with its working memory.
    store = SessionStore(path, writer="gateway")
    runner = SessionRunner(live, store, runner.session_id)
    await runner.run_turn("And now?")
    live_next = live_bodies[3]
    # A restarted process resumes from the copy, with the window arm.
    resumed_bodies: list[bytes] = []
    resumed = Agent(
        llm=build(recording_transport([reply("next")], resumed_bodies)),
        options=AgentOptions(workspace_root=root, record_answers=True),
    )
    await resumed.start()
    restarted = SessionStore(copy, writer="gateway")
    again = SessionRunner(resumed, restarted, runner.session_id, resume_mode="window")
    assert await again.resume() == 2
    assert again.last_resume is not None and again.last_resume.checkpoint_turns == 2
    await again.run_turn("And now?")
    # Byte for byte, apart from the secret the checkpoint redacted (the
    # tool_result output it leaves out is in no request).
    assert SECRET.encode() in live_next
    assert resumed_bodies[0] == live_next.replace(SECRET.encode(), b"[redacted]")
    for agent in (live, resumed):
        await agent.close()
    await store.aclose()
    await restarted.aclose()


# ── window: tail, fallbacks and counts ──────────────────────────────────────


async def window_session(
    tmp_path: Path, store: SessionStore, turns: int, replies: list[Any] | None = None
) -> tuple[str, ScriptedLLM, Agent]:
    """A session whose first *turns* turns ran through a runner, with checkpoints."""
    llm = ScriptedLLM(replies or [answer(f"answer {i}") for i in range(turns)])
    agent = await started_agent(tmp_path / "live", llm)
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    for index in range(turns):
        await runner.run_turn(f"question {index}")
    assert runner.session_id is not None
    return runner.session_id, llm, agent


async def test_the_window_arm_adds_a_crashed_turn_as_a_pair(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    sid, _, live = await window_session(
        tmp_path,
        store,
        1,
        [tool_call("file_read", {"path": "notes.txt"}), answer("milk")],
    )
    await store.begin_turn(sid, "crashing question", holder=HOLDER)
    await store.aclose()
    store = SessionStore(path, writer="gateway")  # recovery ends the turn crashed
    runner, agent, _ = await runner_for(tmp_path, store, sid, "window")
    caplog.set_level(logging.WARNING)
    assert await runner.resume() == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [(m.role, bool(m.tool_calls)) for m in agent.conversation()] == [
        ("user", False),
        ("assistant", True),
        ("tool", False),
        ("assistant", False),
        ("user", False),
        ("assistant", False),
    ]
    assert agent.conversation()[-1].content == NO_ANSWER_LABEL.format(status="crashed")
    info = runner.last_resume
    assert info is not None
    assert (info.mode, info.fell_back, info.turns, info.checkpoint_turns) == (
        "window",
        False,
        2,
        1,
    )
    assert info.chars == sum(len(m.content) for m in agent.conversation())
    await live.close()
    await agent.close()
    await store.aclose()


@pytest.mark.parametrize("case", ["none", "version", "corrupt", "undone", "retention"])
async def test_without_a_usable_checkpoint_the_window_arm_falls_back(
    tmp_path: Path, case: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = SessionStore(path, writer="gateway", clock=clock)
    if case == "none":
        sid = await stored_session(
            store, [("q1", "completed", "a1"), ("q2", "completed", "a2")]
        )
    else:
        sid, _, live = await window_session(tmp_path, store, 3)
        await live.close()
    if case in ("version", "corrupt"):
        await store.aclose()
        conn = sqlite3.connect(path)
        if case == "version":
            conn.execute("UPDATE windows SET version = 2")
        else:
            conn.execute("UPDATE windows SET data = '{not json'")
        conn.commit()
        conn.close()
        store = SessionStore(path, writer="gateway", clock=clock)
    elif case == "undone":
        await store.undo_last_turn(sid)
    elif case == "retention":
        clock.advance(31 * DAY)
        await store.apply_retention()
    assert (await store.get_window(sid) is None) == (
        case in ("none", "undone", "retention")
    )
    runner, agent, _ = await runner_for(tmp_path, store, sid, "window")
    caplog.set_level(logging.WARNING)
    turns = await runner.resume()
    info = runner.last_resume
    assert info is not None
    assert (info.mode, info.fell_back, info.checkpoint_turns) == ("pairs", True, 0)
    assert turns == info.turns == (2 if case in ("none", "undone") else 3)
    assert all(not m.tool_calls for m in agent.conversation())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    await agent.close()
    await store.aclose()


async def test_a_checkpoint_of_four_turns_and_a_tail_of_two(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="cli")
    sid, _, live = await window_session(tmp_path, store, 4)
    for index in (4, 5):  # turns another terminal ran: no checkpoint of ours
        turn = await store.begin_turn(sid, f"question {index}", holder=HOLDER)
        await store.finish_turn(
            sid,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer=f"answer {index}",
        )
    runner, agent, _ = await runner_for(tmp_path, store, sid, "window")
    caplog.set_level(logging.WARNING)
    assert await runner.resume() == 6
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    info = runner.last_resume
    assert info is not None and (info.turns, info.checkpoint_turns) == (6, 4)
    assert info.messages == 12
    assert [m.content for m in agent.conversation()][-2:] == ["question 5", "answer 5"]
    await live.close()
    await agent.close()
    await store.aclose()


async def test_an_evicted_checkpoint_counts_only_what_stayed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="cli")
    sid, _, live = await window_session(tmp_path, store, 10)  # 20 messages
    for index in (10, 11):
        turn = await store.begin_turn(sid, f"question {index}", holder=HOLDER)
        await store.finish_turn(
            sid,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer=f"answer {index}",
        )
    runner, agent, _ = await runner_for(
        tmp_path, store, sid, "window", window_eviction="slide", window_messages=20
    )
    assert await runner.resume() == 10  # 24 loaded, the 4 oldest evicted
    info = runner.last_resume
    assert info is not None
    assert (info.turns, info.checkpoint_turns) == (10, 8)
    assert info.turns - info.checkpoint_turns == 2  # the tail
    assert agent.conversation()[0].content == "question 2"
    await live.close()
    await agent.close()
    await store.aclose()


async def test_resume_without_a_session_loads_nothing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    agent = await started_agent(tmp_path / "ws", ScriptedLLM())
    runner = SessionRunner(
        agent, store, None, new_session=NewSession(), resume_mode="window"
    )
    assert await runner.resume() == 0
    assert runner.last_resume is not None and runner.last_resume.messages == 0
    with pytest.raises(ValueError):
        await runner.resume(max_turns=-1)
    await agent.close()
    await store.aclose()


async def test_an_evicted_checkpoint_counts_only_what_stayed_under_chunk(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="cli")
    sid, _, live = await window_session(tmp_path, store, 10)  # 20 messages
    for index in (10, 11):
        turn = await store.begin_turn(sid, f"question {index}", holder=HOLDER)
        await store.finish_turn(
            sid,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer=f"answer {index}",
        )
    # The default window eviction drops whole units at once, so fewer stay than
    # under slide; the count is still the questions that stayed.
    runner, agent, _ = await runner_for(
        tmp_path, store, sid, "window", window_messages=20
    )
    stayed = await runner.resume()
    assert 0 < stayed < 10
    questions = [m.content for m in agent.conversation() if m.role == "user"]
    assert len(questions) == stayed and questions[-1] == "question 11"
    info = runner.last_resume
    assert info is not None and info.turns == stayed
    await live.close()
    await agent.close()
    await store.aclose()
