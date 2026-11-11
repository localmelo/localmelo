"""Memory reads a stored turn's typed words through the session store's one reader.

A turn's text can hold more than what the person typed: a file's text and name, the
stand-in for a message that was only files, a chat's title. The store's
``typed_words`` is the one reader of the person's own words, and memory takes them
from it everywhere: the document the runner hands memory when a turn ends (live),
the one a rebuild makes from the stored row, the note gate's evidence. For each kind
of row the live document equals the rebuilt one and holds only the person's words.

Each turn runs through the session runner on a scripted in-process model, in a
temporary store and memory folder; the text is invented.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.melo.contracts.attachments import words_before_files
from localmelo.support.memory import ChatMemory, sleep
from localmelo.support.memory.recall import AGENT_EXCHANGE
from localmelo.support.memory.service import ChatTurn, TurnMemory
from localmelo.support.sessions.attachments import (
    TextAttachment,
    compose_user_text,
    stand_in,
)
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    TurnInfo,
    typed_words,
)
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, started_agent

HOLDER = "test-host:1:typed-words"
FILE = TextAttachment("plan-notes.txt", "Beds: three rows of tomatoes by the fence.")
# What a document holds for a turn: an optional task title, the words, the reply.
DOCUMENT = re.compile(
    r"(?:Task: [^\n]*\n)?User:[ ]?(?P<words>.*?)(?:\nmelo: .*)?", re.DOTALL
)


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


async def run_turn(
    memory: ChatMemory,
    store: SessionStore,
    workspace: Path,
    text: str,
    *,
    kind: str = "chat",
    index_only: bool = False,
    **begin: Any,
) -> TurnInfo:
    """One completed turn in a new session, through the runner's own path.

    *begin* is what ``SessionRunner.start_turn`` takes (a title, files, words given
    apart); *index_only* is a task's runner. Returns the turn as the store holds it.
    """
    agent = await started_agent(workspace, ScriptedLLM([answer("Noted.")]))
    runner = SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(kind=kind),
        memory=memory,
        index_only=index_only,
    )
    try:
        await runner.start_turn(text, **begin)
        result = await runner.wait()
        assert result is not None and result.status == "completed"
        await memory.drain()
        assert runner.session_id is not None
        return (await store.list_turns(runner.session_id))[-1]
    finally:
        await runner.close()
        await agent.close()


async def documents(memory: ChatMemory, info: TurnInfo) -> list[str]:
    """The text of each piece the index holds for *info*'s turn."""
    _, index = memory.stores()
    found = await index.turn_documents([(info.session_id, info.turn)])
    return [doc.text for doc in found]


def words_of(document: str) -> str:
    """What the document holds as the person's words."""
    found = DOCUMENT.fullmatch(document)
    assert found is not None, document
    return found["words"].strip()


@dataclass(frozen=True)
class Row:
    """A kind of stored turn: how its session's runner begins it, and what the
    person typed in it."""

    text: str
    words: str
    kind: str = "chat"
    index_only: bool = False
    begin: dict[str, Any] = field(default_factory=dict)


# A retry begins its turn with no title and no words given apart, so a retried row
# has blank typed text, and a retried row with files keeps its words and its files.
# A continued task's card holds the new message and its file in its text.
CARD = compose_user_text("Tidy the garden notes. Also check the beds.", (FILE,))
ROWS = {
    "words and a file": Row(
        "Please read the garden notes.",
        "Please read the garden notes.",
        begin={"attachments": (FILE,), "title": "Please read the garden notes."},
    ),
    "a file only": Row(stand_in(0, 1), "", begin={"attachments": (FILE,)}),
    "a plain row retried with blank typed text": Row(
        "Sam keeps the lab keys.", "Sam keeps the lab keys."
    ),
    "a retried row with a file": Row(
        "Please read the garden notes.",
        "Please read the garden notes.",
        begin={"attachments": (FILE,)},
    ),
    "a task begun with a title": Row(
        "Tidy the garden notes folder.",
        "Tidy the garden notes folder.",
        kind="task",
        index_only=True,
        begin={"title": "Garden notes"},
    ),
    "a task begun with the persons words apart": Row(
        "Write a report of the beds for the garden club, in plain steps.",
        "make the garden report",
        kind="task",
        index_only=True,
        begin={"title": "Garden report", "words": "make the garden report"},
    ),
    "a titled chat turn": Row(
        "I live in Lisbon.", "I live in Lisbon.", begin={"title": "Trip plans"}
    ),
    "a continued task whose card holds a file, indexed by a runner that only indexes": Row(
        CARD,
        "Tidy the garden notes. Also check the beds.",
        kind="task",
        index_only=True,
        begin={"words": words_before_files(CARD)},
    ),
}


@pytest.mark.parametrize("row", list(ROWS))
async def test_the_live_document_equals_the_rebuilt_one_and_holds_only_the_persons_words(
    row: str, memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    case = ROWS[row]
    info = await run_turn(
        memory,
        store,
        tmp_path / "ws",
        case.text,
        kind=case.kind,
        index_only=case.index_only,
        **case.begin,
    )
    live = await documents(memory, info)
    assert len(live) == 1
    # The store's reader says what the person typed; the document holds that.
    assert typed_words(info) == case.words
    assert words_of(live[0]) == case.words
    # No file's text or name, no stand-in, and no title as the person's words.
    title = case.begin.get("title", "")
    for left_out in (FILE.text, FILE.name, "[1 file]", title):
        if left_out and left_out != case.words:
            assert left_out not in words_of(live[0]), left_out
    # A rebuild from the stored row gives the same document.
    await memory.reindex(store)
    assert await documents(memory, info) == live


async def test_an_answer_with_a_file_given_to_a_waiting_task_is_kept_as_its_words_only(
    memory: ChatMemory, store: SessionStore
) -> None:
    asks = pytest.importorskip("localmelo.support.gateway.console.asks")
    echoes: list[dict[str, Any]] = []

    async def echo(data: dict[str, Any], *rest: Any) -> dict[str, Any]:
        echoes.append(data)
        return {"id": len(echoes), **data}

    console = SimpleNamespace(
        memory=memory,
        store=store,
        front_for=lambda chat_id: SimpleNamespace(echo=echo),
    )
    owner = asks.OwnerAsks(cast(Any, console))
    owner._asks["task-1"] = asks.Ask(
        "ask-1",
        "task-1",
        "Garden notes",
        "chat-1",
        "melo",
        {"problem": "Which folder holds the notes?"},
        asyncio.get_running_loop().create_future(),
    )
    reply = compose_user_text("Use the second folder.", (FILE,))
    assert await owner.reply("task-1", reply, "ask-1") is not None
    await owner.settle()
    _, index = memory.stores()
    (held,) = await index.fetch(await index.ids())
    assert held.kind == AGENT_EXCHANGE
    assert held.text == (
        "Task asked: Which folder holds the notes? Answer: Use the second folder."
    )
    assert FILE.text not in held.text and FILE.name not in held.text


async def stored(
    store: SessionStore,
    chat: str | None,
    text: str,
    *,
    event: bool = False,
    **begin: Any,
) -> TurnInfo:
    """A finished turn of *text* in the chat *chat* (a new one for ``None``), as a
    session file holds it; *event* gives it the ``turn.memory`` event of a turn that
    ran with memory on."""
    info = await store.begin_turn(
        chat,
        text,
        holder=HOLDER,
        new_session=None if chat else NewSession(),
        **begin,
    )
    if event:
        data = {"turn": info.turn, "turn_id": info.id, "used": [], "remembered": []}
        data |= {"refused": "", "recall": {}, "unsaved_claim": False}
        await store.append_events(
            info.session_id,
            info.turn,
            [{"type": "turn.memory", "data": data}],
            holder=HOLDER,
        )
    return await store.finish_turn(
        info.session_id,
        info.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="Noted.",
    )


# A row from before files were stored apart: blocks in its text, and a title.
LEGACY: dict[str, Any] = {
    "text": compose_user_text("I keep bees.", (FILE,)),
    "title": "Hobbies",
}
# A message that was only files: the stand-in is not words.
ONLY_FILES: dict[str, Any] = {
    "text": compose_user_text(stand_in(0, 1), (FILE,)),
    "attachments": (FILE,),
    "typed_text": stand_in(0, 1),
}
# A row with its files stored apart.
APART: dict[str, Any] = {
    "text": compose_user_text("My bike is blue.", (FILE,)),
    "attachments": (FILE,),
    "typed_text": "My bike is blue.",
}
TITLED: dict[str, Any] = {"text": "I live in Lisbon.", "title": "Trip plans"}


async def test_a_files_text_and_a_title_are_not_what_counts_as_typed_for_a_later_note(
    memory: ChatMemory, store: SessionStore
) -> None:
    """What ``_said`` offers a later note as typed in the chat's earlier turns."""
    chat: str | None = None
    for row in (LEGACY, ONLY_FILES, APART, TITLED):
        chat = (await stored(store, chat, **row)).session_id
    assert chat is not None
    later = await store.begin_turn(chat, "What do I keep?", holder=HOLDER)
    said = await memory._said(ChatTurn(store, later, later.user_text, TurnMemory()))
    assert said == ["I keep bees.", "", "My bike is blue.", "I live in Lisbon."]
    joined = "\n".join(said)
    assert FILE.text not in joined and FILE.name not in joined
    assert "Hobbies" not in joined and "Trip plans" not in joined


async def test_a_rebuild_holds_only_the_words_of_a_row_stored_before_files_were_apart(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await stored(store, None, **LEGACY)
    await memory.reindex(store)
    (document,) = await documents(memory, info)
    assert words_of(document) == "I keep bees."
    assert FILE.text not in document and FILE.name not in document


async def test_a_sleep_run_reads_the_typed_words_the_store_does(
    store: SessionStore,
) -> None:
    chat: str | None = None
    for row in (LEGACY, ONLY_FILES, APART, TITLED):
        chat = (await stored(store, chat, event=True, **row)).session_id
    turns = await sleep.sleep_turns(store, 0)
    assert [turn.typed for turn in turns] == [
        "I keep bees.",
        "",
        "My bike is blue.",
        "I live in Lisbon.",
    ]
    # What a turn follows is the words of the one before it, the same way.
    assert [turn.previous for turn in turns] == [
        "",
        "I keep bees.",
        "",
        "My bike is blue.",
    ]
