"""R4's placement of notes in the window, the window API and tool hints."""

from __future__ import annotations

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.memory.notes import ReasoningNote
from localmelo.melo.memory.short import group_turns
from localmelo.melo.memory.short.note_store import NoteStoreEntry
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    Message,
    ToolDef,
)

from .test_compress import FakeLLM, _agent, _answer, _call_reply, _options


def _note(text: str = "plan: read list_ops.py") -> ReasoningNote:
    return ReasoningNote(note_id=text, plan=text)


def test_an_empty_label_that_carries_a_note_keeps_its_nudge_unit() -> None:
    label = Message(role="assistant", content=f"{EMPTY_REPLY_LABEL}\nplan: think again")
    nudge = Message(role="user", content=EMPTY_REPLY_NUDGE)
    assert group_turns([label, nudge]) == [[label], [nudge]]
    assert group_turns([label, nudge], empty_prefix=True) == [[label, nudge]]
    exact = Message(role="assistant", content=EMPTY_REPLY_LABEL)
    assert group_turns([exact, nudge]) == [[exact, nudge]]


def test_window_eviction_leaves_the_note_store_intact() -> None:
    agent = _agent(FakeLLM(_answer()), _options(reasoning="compress"))
    agent.hippo.working.notes.add("note", "plan: x", note=_note("plan: x"))
    agent.hippo.working.clear()
    assert len(agent.hippo.working.get_window()) == 0
    assert [entry.alias for entry in agent.notes()] == ["n1"]


def test_notes_returns_deep_copies() -> None:
    agent = _agent(
        FakeLLM(_answer("The test passed because the fix works.")),
        _options(reasoning="compress"),
    )
    agent.hippo.working.notes.add("note", "plan: x", note=_note())
    copied = agent.notes()
    copied[0].text = "changed"
    assert copied[0].note is not None
    copied[0].note.plan = "changed"
    assert agent.notes()[0].text == "plan: x"
    kept_note = agent.notes()[0].note
    assert kept_note is not None
    assert kept_note.plan == "plan: read list_ops.py"


def test_load_notes_replaces_the_store_and_keeps_aliases() -> None:
    agent = _agent(FakeLLM(_answer()), _options(reasoning="compress"))
    first = _agent(FakeLLM(_answer()), _options(reasoning="compress"))
    first.hippo.working.notes.add("note", "one", note=_note("one"))
    first.hippo.working.notes.add("note", "two", note=_note("two"))
    loaded = first.notes()
    assert agent.load_notes(loaded) == 2
    assert [entry.alias for entry in agent.notes()] == ["n1", "n2"]
    added = agent.hippo.working.notes.add("note", "three", note=_note("three"))
    assert added is not None and added.alias == "n3"
    assert agent.load_notes([]) == 0
    assert agent.notes() == []


def test_load_notes_refuses_wrong_types_and_runs() -> None:
    agent = _agent(FakeLLM(_answer()), _options(reasoning="compress"))
    with pytest.raises(TypeError, match="NoteStoreEntry"):
        agent.load_notes([Message(role="user", content="x")])  # type: ignore[list-item]
    agent._active_runs = 1
    try:
        with pytest.raises(RuntimeError, match="while a run is active"):
            agent.load_notes([])
    finally:
        agent._active_runs = 0


@pytest.mark.asyncio
async def test_a_loaded_note_survives_a_run_in_the_store() -> None:
    server = FakeLLM(_call_reply(), _answer())
    agent = _agent(server, _options(reasoning="compress"))
    entry = NoteStoreEntry(
        alias="n1",
        kind="note",
        text="plan: the loaded fact",
        note=_note("plan: the loaded fact"),
    )
    assert agent.load_notes([entry]) == 1
    await agent.run_task("q")
    await agent.close()
    # R4 has no retrieval: the loaded note stays in the store, and a later
    # release's recall reads it from there, not from the window.
    assert any(stored.text == "plan: the loaded fact" for stored in agent.notes())


@pytest.mark.asyncio
async def test_compress_replay_keeps_all_thinking_and_sends_only_k() -> None:
    server = FakeLLM(
        _call_reply(thinking="first thought"),
        _call_reply(thinking="second thought"),
        _answer(thinking="answer thought"),
    )
    agent = _agent(
        server,
        _options(reasoning="compress+replay", replay_turns=1, max_steps=3),
    )
    await agent.run_task("q")
    await agent.close()
    window_thinking = [
        message.thinking
        for message in agent.conversation()
        if message.role == "assistant"
    ]
    # The answer ends the run and is not kept in the window (record_answers
    # is off), so the two tool turns are what the window holds.
    assert window_thinking == ["first thought", "second thought"]
    # The third request carries only the newest in-scope turn's thinking.
    sent = [
        message.thinking
        for message in server.requests[2]
        if message.role == "assistant"
    ]
    assert sent == ["", "second thought"]
    # A compress run's window carries no thinking at all.
    compress = FakeLLM(_call_reply(thinking="hidden"), _answer())
    agent = _agent(compress, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    assert [
        message.thinking
        for message in agent.conversation()
        if message.role == "assistant"
    ] == [""]


@pytest.mark.asyncio
async def test_a_note_naming_a_tool_does_not_offer_it() -> None:
    def _secret() -> ToolDef:
        return ToolDef(
            name="secret_tool",
            description="does a secret thing",
            parameters={"type": "object", "properties": {}},
        )

    def _build(mode: str) -> tuple[Agent, FakeLLM]:
        server = FakeLLM(_answer())
        agent = Agent(
            llm=server,
            options=_options(
                reasoning=mode,
                builtins=frozenset(),
                tool_exposure="retrieve",
            ),
        )
        agent.hippo.register_tool(_secret())
        agent.load_conversation(
            [
                Message(
                    role="assistant",
                    content="<note>found: secret_tool would help here</note>",
                )
            ]
        )
        agent.load_notes(
            [
                NoteStoreEntry(
                    alias="n1",
                    kind="note",
                    text="secret_tool would help here",
                    note=_note("secret_tool would help here"),
                )
            ]
        )
        return agent, server

    compress, compress_server = _build("compress")
    await compress.run_task("q")
    await compress.close()
    # The query alone resolves nothing; the note would have hinted the tool.
    drop, drop_server = _build("drop")
    await drop.run_task("q")
    await drop.close()
    assert "secret_tool" in drop_server.offers[0]
    assert "secret_tool" not in compress_server.offers[0]
