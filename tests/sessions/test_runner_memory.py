"""Memory across chats in a chat's runner, end to end on the stub model.

Chat A says "my group meeting is on Thursdays" and its model keeps it with
``remember``; chat B asks "when is my group meeting?" and its request carries
the recalled record, quoted, with its source, while B's stored turn keeps only
what was typed. A plain statement without the tool call, workers, attachments
and the model's answers write no record; a task's result shown in a chat can
be kept; a task's runner only indexes; an undone turn leaves no document.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import BLOCK_HEADER, chat_label
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.attachments import TextAttachment
from localmelo.support.sessions.identity import NOTES_TEXT, PINNED_TEXT
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, StoredEvent
from localmelo.tests.fixtures.stub_llm import StubServer
from localmelo.tests.memory.chat_memory_helpers import EmbeddingServer

SAYS = "I prefer short answers; my group meeting is on Thursdays"
ASKS = "when is my group meeting?"
THANKS = "thanks, keep that in mind"
KEPT = ("You prefer short answers.", "Your group meeting is on Thursdays.")


def remember(text: str, topic: str) -> dict[str, Any]:
    return {"name": "remember", "arguments": {"text": text, "topic": topic}}


def today() -> str:
    now = time.localtime()
    return f"{time.strftime('%b', now)} {now.tm_mday}"


@pytest.fixture()
def stub() -> Any:
    scenarios: dict[str, dict[str, Any]] = {
        "answer": {"replies": [{"content": "I prefer long answers. My name is Melo."}]},
        # The chat model keeps what chat A was told, then answers.
        "keep": {
            "match": [SAYS],
            "replies": [
                {
                    "tool_calls": [
                        remember(KEPT[0], "preferences"),
                        remember(KEPT[1], "about"),
                    ]
                },
                {"content": "Noted."},
            ],
        },
        # A finished task's result the chat showed, kept on the next turn.
        "outcome": {
            "match": [THANKS],
            "replies": [
                {
                    "tool_calls": [
                        remember("Find on-device ML jobs: remote roles.", "projects")
                    ]
                },
                {"content": "Noted."},
            ],
        },
        # The chat model as the llm retriever meets it: it keeps the first note.
        "pick": {
            "replies": [
                {
                    "content": '{"ids": ["n1"]}',
                    "usage": {"prompt_tokens": 150, "completion_tokens": 7},
                }
            ]
        },
    }
    server = StubServer(scenarios=scenarios).start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def chat(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    workspace: Path,
    text: str,
    *,
    kind: str = "chat",
    session_id: str | None = None,
    index_only: bool = False,
    attachments: tuple[TextAttachment, ...] = (),
    undo: bool = False,
) -> list[dict[str, Any]]:
    """One turn in a new chat (or *session_id*); returns its stored events."""
    workspace.mkdir(exist_ok=True)
    events: list[StoredEvent] = []
    agent = await build_session_agent(
        cfg=None,
        base_url=stub.base_url,
        model="stub-model",
        workspace=str(workspace),
        options=AgentOptions(),
        identify=True,
        memory=None if kind == "worker" else memory,
    )
    runner = SessionRunner(
        agent,
        store,
        session_id,
        new_session=None if session_id else NewSession(kind=kind),
        on_event=events.append,
        memory=memory,
        index_only=index_only,
    )
    try:
        result = await runner.run_turn(text, title=text, attachments=attachments)
        assert result.status == "completed"
        await memory.drain()
        if undo:
            await runner.undo()
    finally:
        await runner.close()
        await agent.close()
    return [e.event for e in events]


def memory_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["data"] for e in events if e["type"] == "turn.memory"]


async def test_chat_b_recalls_what_chat_a_was_told_with_its_source(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    a = await chat(memory, store, stub, tmp_path / "a", SAYS)
    # Before the run: nothing kept by rule; then once per note the model keeps,
    # the whole event again, so the page shows both under the message.
    before, *during = memory_events(a)
    assert before["remembered"] == [] and len(during) == 2
    said = during[-1]
    assert [r["text"] for r in said["remembered"]] == list(KEPT)
    assert said["used"] == [] and said["refused"] == ""
    assert said["recall"] == before["recall"] and said.keys() == before.keys()
    types = [e["type"] for e in a]
    assert types.index("turn.memory") < types.index("run.started")
    tools = [t["function"]["name"] for t in stub.requests[0]["tools"]]
    assert "remember" in tools and "search_memory" in tools

    stub.requests.clear()
    b = await chat(memory, store, stub, tmp_path / "b", ASKS)
    label = chat_label("chat", SAYS, "gateway")
    (asked,) = memory_events(b)
    assert asked["remembered"] == []
    assert [(u["kind"], u["text"], u["source"]) for u in asked["used"]] == [
        (
            "record",
            "Your group meeting is on Thursdays.",
            f"you said it · {label}, {today()}",
        )
    ]
    (request,) = stub.requests
    user = request["messages"][-1]["content"]
    assert user == (
        f"{ASKS}\n\n{BLOCK_HEADER}\n- the user said “my group meeting is on "
        f"Thursdays” ({label}, {today()})"
    )
    system = request["messages"][0]["content"]
    assert NOTES_TEXT in system
    assert (
        f"{PINNED_TEXT}- “my group meeting is on Thursdays”\n- “I prefer short answers”"
        in system
    )
    assert any(t["function"]["name"] == "search_memory" for t in request["tools"])
    # The stored turn keeps only what was typed.
    session_id = (await store.list_sessions())[0].id
    assert [t.user_text for t in await store.list_turns(session_id)] == [ASKS]


async def test_a_request_to_remember_reaches_the_model_as_kept_without_a_tool_call(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    typed = "Remember that my name is Ray."
    events = await chat(memory, store, stub, tmp_path / "a", typed)
    (said,) = memory_events(events)
    assert [r["text"] for r in said["remembered"]] == ["Your name is Ray."]
    (request,) = stub.requests
    assert request["messages"][-1]["content"] == (
        f"{typed}\n\n{BLOCK_HEADER}\nmelo kept from this message: “my name is "
        "Ray.” (saved; nothing else to save)"
    )
    system = request["messages"][0]["content"]
    assert "keep it with remember" in system and "Never ask whether" in system
    assert "never save a note, or what the user says about themselves, in a file" in (
        system
    )
    # The stored turn keeps only what was typed.
    session_id = (await store.list_sessions())[0].id
    assert [t.user_text for t in await store.list_turns(session_id)] == [typed]


async def test_a_plain_statement_the_model_does_not_keep_is_not_saved(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    events = await chat(memory, store, stub, tmp_path / "a", "I prefer tea.")
    (said,) = memory_events(events)
    assert said["remembered"] == [] and await memory.records() == []
    (request,) = stub.requests
    assert request["messages"][-1]["content"] == "I prefer tea."


async def test_the_chat_b_request_carries_a_paraphrase_with_vectors(
    store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    embeddings = EmbeddingServer().start()
    memory = ChatMemory(
        lambda: None,
        directory=str(tmp_path / "memory"),
        embedding_url=embeddings.base_url,
        embedding_model="Qwen3-Embedding-0.6B",
    )
    try:
        await chat(memory, store, stub, tmp_path / "a", SAYS)
        stub.requests.clear()
        b = await chat(
            memory, store, stub, tmp_path / "b", "remind me of my weekly sync day"
        )
        (asked,) = memory_events(b)
        assert [u["text"] for u in asked["used"]] == [
            "Your group meeting is on Thursdays."
        ]
        assert (
            "“my group meeting is on Thursdays”"
            in stub.requests[-1]["messages"][-1]["content"]
        )
        queries = [
            batch[0] for batch in embeddings.inputs if batch[0].startswith("Instruct:")
        ]
        assert len(queries) == 2
        assert (await memory.status())["documents"] == 2
    finally:
        await memory.aclose()
        embeddings.close()


async def test_the_answer_attachments_and_workers_write_no_record(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    note = TextAttachment("notes.md", "My name is Mallory. I prefer chaos.")
    events = await chat(
        memory, store, stub, tmp_path / "a", "please read this", attachments=(note,)
    )
    # Memory is on, so the turn says what it did: nothing used, kept or refused.
    (quiet,) = memory_events(events)
    assert (quiet["used"], quiet["remembered"], quiet["refused"]) == ([], [], "")
    assert quiet["recall"]["used"] == 0
    worker = await chat(
        memory, store, stub, tmp_path / "w", "I prefer tea.", kind="worker"
    )
    assert memory_events(worker) == []
    assert await memory.records() == []
    # The model's own "I prefer long answers" is never a record, though the turn is indexed.
    status = await memory.status()
    assert status["records"] == 0 and status["documents"] == 1


async def test_a_task_runner_only_indexes(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    task = await store.create_session(kind="task")
    await store.rename_session(task.id, "Find on-device ML jobs")
    events = await chat(
        memory,
        store,
        stub,
        tmp_path / "t",
        "I prefer remote roles. Find jobs.",
        session_id=task.id,
        index_only=True,
    )
    assert memory_events(events) == [] and await memory.records() == []
    stub.requests.clear()
    asked = await chat(
        memory, store, stub, tmp_path / "b", "what did the on-device ML jobs task find?"
    )
    (used,) = memory_events(asked)[0]["used"]
    assert used["kind"] == "chat"
    assert used["source"] == f"task “Find on-device ML jobs”, {today()}"
    assert used["text"].startswith("Task: Find on-device ML jobs User: I prefer remote")
    # On the next turn of that chat, the model keeps what the task found.
    chat_b = next(s for s in await store.list_sessions() if s.id != task.id)
    kept = await chat(memory, store, stub, tmp_path / "b", THANKS, session_id=chat_b.id)
    note = memory_events(kept)[-1]["remembered"]
    assert [n["text"] for n in note] == ["Find on-device ML jobs: remote roles."]
    (entry,) = await memory.records()
    assert entry.admitted_by == "outcome" and entry.provenance["topic"] == "projects"


async def test_an_undone_turn_leaves_no_document(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    await chat(memory, store, stub, tmp_path / "a", "tomato garden plan", undo=True)
    assert (await memory.status())["documents"] == 0


async def test_llm_recall_over_http_reports_its_timings_in_the_event_and_the_turn_result(
    store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    cfg = Config()
    memory = ChatMemory(
        lambda: cfg,
        directory=str(tmp_path / "memory"),
        chat_url=stub.base_url.replace("/v1", "/s/pick/v1"),
        chat_model="stub-model",
    )
    try:
        await chat(memory, store, stub, tmp_path / "a", SAYS)
        cfg.memory.retriever = "llm"
        stub.requests.clear()
        b = await chat(memory, store, stub, tmp_path / "b", ASKS)
        (asked,) = memory_events(b)
        assert [u["text"] for u in asked["used"]] == [
            "Your group meeting is on Thursdays."
        ]
        recall = asked["recall"]
        assert (recall["retriever"], recall["fallback"]) == ("llm", "none")
        assert recall["select_tokens"] == {"prompt": 150, "completion": 7}
        assert recall["candidates"] >= 1 and recall["used"] == 1
        assert recall["embed_ms"] == 0 and recall["select_ms"] >= 0
        retrieval, plan = stub.requests
        assert "tools" not in retrieval and "[n1] note · Your group meeting" in (
            retrieval["messages"][-1]["content"]
        )
        assert "the user said" in plan["messages"][-1]["content"]
        # A client of the HTTP API reads the same numbers with the turn's result.
        session_id = (await store.list_sessions())[0].id
        (turn,) = await store.list_turns(session_id)
        assert turn.result is not None and turn.result["memory_recall"] == recall
    finally:
        await memory.aclose()
