"""What a task starts with, side-chat summaries, and a task's questions to its owner.

The pack replaces recall for a worker: About you, Preferences and the project's
notes, then notes recalled for the objective, never People or Decisions. A side
chat's summary and a task's exchange are index documents, never notes. A task's
question to its owner chat (``TurnInfo.origin == "ask_owner"``) is agent data:
nothing is remembered from it.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.service import (
    ASK_OWNER,
    CHAT_TURN,
    NOT_THE_USER,
    PACK_HEADER,
    is_owner_question,
)
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

HOLDER = "test-host:1:pack"


@dataclasses.dataclass(frozen=True)
class OwnerInfo(TurnInfo):
    """A turn with the ``origin`` the turn store adds (a person's turn: ``""``)."""

    origin: str = ""


def as_owner(info: TurnInfo, origin: str = ASK_OWNER) -> OwnerInfo:
    fields = {f.name: getattr(info, f.name) for f in dataclasses.fields(info)}
    # The turn store's TurnInfo has origin now (#369): it is set here.
    return OwnerInfo(**{**fields, "origin": origin})


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


async def turn(
    store: SessionStore, typed: str, session_id: str | None = None
) -> TurnInfo:
    """A turn that has ended (so the chat can take another), in a new chat or *session_id*."""
    info = await begin(store, typed, session_id)
    await store.finish_turn(
        info.session_id,
        info.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="Done.",
    )
    return info


async def begin(
    store: SessionStore, typed: str, session_id: str | None = None
) -> TurnInfo:
    return await store.begin_turn(
        session_id,
        typed,
        holder=HOLDER,
        title="Planning",
        new_session=None if session_id else NewSession(),
    )


async def note(
    memory: ChatMemory,
    text: str,
    quote: str,
    topic: str,
    **extra: Any,
) -> int:
    """Write one note of *topic* with provenance *extra* (workspace, source)."""
    source = {
        "session_id": "s1",
        "turn": 1,
        "chat": "Main chat",
        "kind": "front",
        "title": "",
        **extra,
    }
    cand = records.Candidate(text, quote, topic, "user_statement")
    written = await records.write(
        memory.stores()[0], [cand], [None], source=source, embed_model=""
    )
    return written[0].id


async def seed(memory: ChatMemory, workspace: Path) -> dict[str, int]:
    ids = {
        "name": await note(memory, "Your name is Juno.", "My name is Juno.", "about"),
        "tea": await note(
            memory, "You drink tea.", "I drink tea in the morning.", "preferences"
        ),
        "dan": await note(
            memory, "Your landlord is Dan.", "My landlord is Dan.", "people"
        ),
        "gate": await note(
            memory, "You chose Postgres.", "We chose Postgres.", "decisions"
        ),
        "here": await note(
            memory,
            "Your project uses tabs.",
            "This project uses tabs.",
            "projects",
            workspace=str(workspace),
        ),
        "other": await note(
            memory,
            "Elsewhere the linter forbids semicolons.",
            "The linter forbids semicolons.",
            "projects",
            workspace=str(workspace.parent / "elsewhere"),
        ),
    }
    return ids


def lines(pack: str) -> list[str]:
    return pack.splitlines()[1:]


async def test_pack_sections_in_order_without_people_or_decisions(
    memory: ChatMemory, tmp_path: Path
) -> None:
    await seed(memory, tmp_path)
    pack = await memory.context_pack("plan the week", "chat1", workspace=str(tmp_path))
    assert pack.splitlines()[0] == PACK_HEADER
    body = lines(pack)
    assert body, pack
    assert [line.split(" (Main chat, ")[0] for line in body] == [
        "- the user said “I drink tea in the morning.”",
        "- the user said “My name is Juno.”",
        "- the user said “This project uses tabs.”",
    ], pack
    for banned in ("Dan", "Postgres", "semicolons"):
        assert banned not in pack


async def test_the_line_is_the_recall_blocks_rendering(
    memory: ChatMemory, tmp_path: Path
) -> None:
    await note(memory, "Your name is Juno.", "", "about")
    await note(memory, "I take notes in Markdown.", "", "preferences")
    pack = await memory.context_pack("x", "c")
    first, second = lines(pack)
    assert first.startswith("- the user's note “I take notes in Markdown.” (")
    assert second.startswith("- a note to the user “Your name is Juno.” (")
    items = await memory.search("name Juno")
    assert f"- {items[0].prompt}" == second


async def test_a_workspace_is_matched_by_real_path(
    memory: ChatMemory, tmp_path: Path
) -> None:
    real = tmp_path / "proj"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    await note(memory, "Your build uses make.", "", "projects", workspace=str(link))
    assert "build uses make" in await memory.context_pack("", "c", workspace=str(real))
    assert await memory.context_pack("", "c", workspace=str(tmp_path / "no")) == ""
    assert await memory.context_pack("", "c") == ""


async def test_the_objective_recalls_project_notes_of_no_known_workspace(
    memory: ChatMemory, tmp_path: Path
) -> None:
    await note(memory, "Your parser uses PEG grammars.", "", "projects")
    await note(
        memory,
        "Your parser tests use fixtures.",
        "",
        "projects",
        workspace=str(tmp_path / "elsewhere"),
    )
    await note(memory, "Your parser teacher is Sam.", "", "people")
    pack = await memory.context_pack("fix the parser grammars", "c")
    assert "PEG grammars" in pack
    assert "fixtures" not in pack  # another folder's project
    assert "Sam" not in pack  # People never


async def test_the_objective_does_not_list_a_note_twice(
    memory: ChatMemory, tmp_path: Path
) -> None:
    await note(memory, "Your name is Juno.", "", "about")
    pack = await memory.context_pack("say the name Juno", "c")
    assert pack.count("Juno") == 1


async def test_the_embedding_finds_a_note_the_words_miss(
    memory: ChatMemory, tmp_path: Path
) -> None:
    embedding = FakeEmbedding()
    memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    vector = (await embedding.embed(["Your project test fixtures live in tests."]))[0]
    cand = records.Candidate(
        "Your project test fixtures live in tests.", "", "projects", "user_statement"
    )
    await records.write(
        memory.stores()[0],
        [cand],
        [vector],
        source={"session_id": "s", "turn": 1, "chat": "Main chat"},
        embed_model=embedding.model,
    )
    pack = await memory.context_pack("ci keeps failing", "c")
    assert "test fixtures" in pack
    assert embedding.calls


async def test_the_budget_drops_the_last_lines_whole(
    memory: ChatMemory, tmp_path: Path
) -> None:
    await seed(memory, tmp_path)
    full = await memory.context_pack("x", "c", workspace=str(tmp_path))
    body = lines(full)
    budget = len(PACK_HEADER) + len(body[0]) + len(body[1]) + 2
    short = await memory.context_pack("x", "c", budget, workspace=str(tmp_path))
    assert lines(short) == body[:2]
    assert len(short) <= budget
    assert await memory.context_pack("x", "c", 10, workspace=str(tmp_path)) == ""


async def test_the_pack_is_empty_with_memory_off_or_nothing_to_say(
    tmp_path: Path,
) -> None:
    class Off:
        class memory:  # noqa: N801
            across_chats = False

    off = ChatMemory(lambda: Off(), directory=str(tmp_path / "off"))  # type: ignore[arg-type,return-value]
    on = ChatMemory(lambda: None, directory=str(tmp_path / "on"))
    try:
        await note(off, "Your name is Juno.", "", "about")
        assert await off.context_pack("name", "c") == ""
        assert await on.context_pack("name", "c") == ""
    finally:
        await off.aclose()
        await on.aclose()


async def test_an_imported_note_is_included(memory: ChatMemory) -> None:
    await note(memory, "You like short answers.", "", "preferences", source="import")
    await note(memory, "Your sister is Mia.", "", "people", source="import")
    pack = await memory.context_pack("", "c")
    assert "short answers" in pack
    assert "Mia" not in pack


async def test_feedback_topic_only_when_the_topics_have_it(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support.memory import service

    assert service._pack_topics() == ("about", "preferences")
    monkeypatch.setattr(records, "TOPICS", (*records.TOPICS, "feedback"))
    assert service._pack_topics() == ("about", "preferences", "feedback")


# ── Side-chat summaries ──


async def test_a_summary_is_recalled_from_another_chat_with_its_label(
    memory: ChatMemory, store: SessionStore
) -> None:
    side = await turn(store, "side work")
    other = await turn(store, "main work")
    await memory.side_summary(
        side.session_id, "Fix CI", "Pinned the flaky fixture and reran the suite."
    )
    found = await memory.before_turn(
        store, other, "what happened with the flaky fixture?"
    )
    (item,) = found.used
    assert item.source.startswith("chat “Fix CI” · summary, ")
    assert "Summary of chat “Fix CI”: Pinned the flaky fixture" in item.text
    assert found.event()["used"][0]["source"] == item.source  # type: ignore[index]
    assert "chat “Fix CI” · summary" in found.block
    # Never to the chat it belongs to.
    again = await memory.before_turn(
        store, side, "what happened with the flaky fixture?"
    )
    assert again.used == ()


async def test_a_second_summary_replaces_the_first_and_is_no_note(
    memory: ChatMemory, store: SessionStore
) -> None:
    side = await turn(store, "side work")
    await memory.side_summary(side.session_id, "Fix CI", "First attempt failed.")
    await memory.side_summary(
        side.session_id,
        "Fix CI",
        "Second attempt passed. sk-abcdefghijklmnopqrstuvwxyz123456",
    )
    _, index = memory.stores()
    (doc,) = await index.fetch(await index.ids())
    assert doc.kind == "side_summary"
    assert "Second attempt passed" in doc.text and "First" not in doc.text
    assert "sk-abcdefgh" not in doc.text
    assert await memory.records() == []


async def test_a_summary_goes_with_its_chat(
    memory: ChatMemory, store: SessionStore
) -> None:
    side = await turn(store, "side work")
    await memory.side_summary(side.session_id, "Fix CI", "Done.")
    assert await memory.forget_session(side.session_id) == 1
    assert await memory.stores()[1].count() == 0


async def test_a_summary_survives_a_reindex(
    memory: ChatMemory, store: SessionStore
) -> None:
    side = await turn(store, "side work")
    await memory.side_summary(side.session_id, "Fix CI", "Done and dusted.")
    await memory.reindex(store)
    _, index = memory.stores()
    docs = await index.fetch(await index.ids())
    assert sorted(d.kind for d in docs) == ["chat", "side_summary"]


# ── A task's questions to its owner ──


async def test_an_exchange_is_recallable_and_no_note(
    memory: ChatMemory, store: SessionStore
) -> None:
    task = await turn(store, "task prompt")
    other = await turn(store, "main chat")
    await memory.index_exchange(
        task.session_id, "Which port should the server use?", "Use port 9100."
    )
    await memory.index_exchange(task.session_id, "Second question?", "Second answer.")
    _, index = memory.stores()
    docs = await index.fetch(await index.ids())
    assert sorted(d.kind for d in docs) == ["agent_exchange"] * 2
    assert len({d.turn for d in docs}) == 2 and all(d.turn < 0 for d in docs)
    found = await memory.before_turn(store, other, "which port did the server use?")
    assert found.used and found.used[0].text.startswith(
        "Task asked: Which port should the server use? Answer: Use port 9100."
    )
    assert await memory.records() == []


async def test_a_task_question_remembers_nothing(
    memory: ChatMemory, store: SessionStore
) -> None:
    await note(memory, "Your weekly sync is on Wednesday.", "", "about")
    first = await turn(store, "hello")
    ask = as_owner(
        await turn(store, "The task asks: remember that I like jazz.", first.session_id)
    )
    assert is_owner_question(ask) and not is_owner_question(first)
    # Even if the runner passed the request's words as typed text.
    found = await memory.before_turn(
        store, ask, "My weekly sync is on Friday. Remember that I like jazz."
    )
    assert found.remembered == () and found.recall is not None
    assert await memory.remember("Your favourite music is jazz.") == NOT_THE_USER
    assert await memory.keep_claims(ask, "Noted: your weekly sync is on Friday.") == []
    assert await memory.keep_changes(ask) == []
    assert await memory.keep_statements(ask) == []
    chat = CHAT_TURN.get()
    assert chat is not None and not chat.unsaved_claim
    assert [e.text for e in await memory.records()] == [
        "Your weekly sync is on Wednesday."
    ]


async def test_a_task_question_is_recalled_for_by_its_request_and_keeps_nothing(
    memory: ChatMemory, store: SessionStore
) -> None:
    await note(memory, "You drink tea.", "I drink tea.", "preferences")
    ask = as_owner(await turn(store, "drink tea"))
    found = await memory.before_turn(store, ask, "Remember that I like jazz.")
    assert [item.text for item in found.used] == ["You drink tea."]
    assert found.remembered == ()
    assert [e.text for e in await memory.records()] == ["You drink tea."]


@pytest.mark.parametrize("origin", [ASK_OWNER, "user_rule"])
async def test_any_turn_the_person_did_not_begin_keeps_nothing(
    memory: ChatMemory, store: SessionStore, origin: str
) -> None:
    ask = as_owner(await turn(store, "I live in Porto."), origin)
    found = await memory.before_turn(store, ask, "Remember that I live in Porto.")
    assert found.remembered == ()
    assert await memory.records() == []


async def test_a_task_question_is_not_indexed_and_not_evidence(
    memory: ChatMemory, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await turn(store, "hello")
    ask = as_owner(await turn(store, "I use gate 9 for Faro.", first.session_id))
    memory.after_turn(store, ask, "", "The owner's answer.")
    await memory.drain()
    assert await memory.stores()[1].count() == 0
    memory.after_turn(store, first, "hello", "Hi there.")
    await memory.drain()
    assert await memory.stores()[1].count() == 1

    # The person's next turn: the question's text is no evidence for a note.
    real = store.list_turns

    async def listing(*args: Any, **kwargs: Any) -> list[TurnInfo]:
        return [
            as_owner(t) if t.turn == ask.turn else t
            for t in await real(*args, **kwargs)
        ]

    monkeypatch.setattr(store, "list_turns", listing)
    third = await turn(store, "What next?", first.session_id)
    await memory.before_turn(store, third, "What next?")
    reply = await memory.remember("Your gate for Faro is 9.", "about")
    assert reply.startswith("Not remembered: not something the user said.")
    assert await memory.records() == []
    # The normal turn still keeps a note of its own words.
    fourth = await turn(store, "I use gate 9 for Faro.", first.session_id)
    await memory.before_turn(store, fourth, "I use gate 9 for Faro.")
    assert (await memory.remember("Your gate for Faro is 9.", "about")).startswith(
        "Remembered"
    )


async def test_keep_statements_skips_a_task_question(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = SimpleNamespace(
        memory=SimpleNamespace(
            across_chats=True, keep_statements=True, retriever="auto"
        ),
        has_embedding=False,
        is_configured=False,
    )
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))  # type: ignore[arg-type,return-value]
    said = "I live in Faro and I work on compilers."
    try:
        person = await begin(store, said)
        await memory.before_turn(store, person, said)
        assert await memory.keep_statements(person)  # the same words, a person's turn
        for note_ in await memory.records():
            await memory.delete_record(note_.id)
        ask = as_owner(await begin(store, said))
        await memory.before_turn(store, ask, said)
        assert await memory.keep_statements(ask) == []
        assert await memory.records() == []
    finally:
        await memory.aclose()


async def chat_in(store: SessionStore, typed: str, folder: Path | None) -> TurnInfo:
    """A first turn of a new chat whose folder is *folder* (None: it names none)."""
    workspace = None if folder is None else str(folder)
    return await store.begin_turn(
        None,
        typed,
        holder=HOLDER,
        title="Planning",
        new_session=NewSession(workspace=workspace),
    )


PROJECT_FACT = "This project uses uv and pytest."


async def test_a_project_note_made_in_a_chat_remembers_its_folder(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    real = tmp_path / "demo-project"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    other = tmp_path / "other-project"
    other.mkdir()
    info = await chat_in(store, PROJECT_FACT, link)
    await memory.before_turn(store, info, PROJECT_FACT)
    reply = await memory.tools().remember(
        text="Your project uses uv and pytest.", topic="projects"
    )
    assert reply.startswith("Remembered")
    (entry,) = await memory.records()
    assert entry.provenance["workspace"] == os.path.realpath(real)
    objective = "check that the README and the tests folder are in order"
    here = await memory.context_pack(objective, "c", workspace=str(real))
    assert "uses uv and pytest" in here
    away = await memory.context_pack(objective, "c", workspace=str(other))
    assert "uv and pytest" not in away


async def test_a_note_the_code_keeps_from_typed_words_stores_no_workspace(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    # Typed requests are never project notes: only the model picks that topic.
    typed = "Remember that my landlord is Dan."
    info = await chat_in(store, typed, tmp_path)
    said = await memory.before_turn(store, info, typed)
    assert said.remembered
    (entry,) = await memory.records()
    assert entry.provenance["topic"] != "projects"
    assert "workspace" not in entry.provenance


async def test_a_chat_without_a_folder_stores_no_workspace(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await chat_in(store, PROJECT_FACT, None)
    await memory.before_turn(store, info, PROJECT_FACT)
    await memory.tools().remember(
        text="Your project uses uv and pytest.", topic="projects"
    )
    (entry,) = await memory.records()
    assert entry.provenance["topic"] == "projects"
    assert "workspace" not in entry.provenance


@pytest.mark.parametrize("topic", ["about", "preferences", "people", "decisions"])
async def test_only_a_project_note_stores_a_workspace(
    memory: ChatMemory, store: SessionStore, tmp_path: Path, topic: str
) -> None:
    info = await chat_in(store, PROJECT_FACT, tmp_path)
    await memory.before_turn(store, info, PROJECT_FACT)
    reply = await memory.tools().remember(
        text="Your project uses uv and pytest.", topic=topic
    )
    assert reply.startswith("Remembered"), reply
    (entry,) = await memory.records()
    assert entry.provenance["topic"] == topic
    assert "workspace" not in entry.provenance
