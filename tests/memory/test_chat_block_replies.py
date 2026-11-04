"""A reply about a blocked step is not memory: the turn keeps the person's words only.

A turn that ended on a step melo was not allowed to take (the executor's denial, or
a tool's refused question with one of Auto's notes) is indexed as ``User: <words>``:
no ``melo:`` line, so a later chat never recalls a reply that reports rules that have
changed since. A block that the turn went on from (a later call ran) leaves the
answer in, and so does any other failure. The live index and a rebuild agree, and
a turn that did not end on a block is indexed exactly as it was.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    BLOCK_NOTE,
    BLOCK_STOP,
    STOPPED_NOTE,
    block_note,
)
from localmelo.support.memory import ChatMemory, blocked
from localmelo.support.memory.index import document_chunks, document_text, typed_of
from localmelo.support.memory.service import (
    ADDED_MARK,
    ended_on_block,
    turn_document,
    turn_words,
)
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.sessions.helpers import core_event

HOLDER = "test-host:1:blocked"
TYPED = "I authorize you to submit the form."
ANSWER = "Auto blocked this step; type a permission naming both actions."
NOTE = block_note(None, words="sends a form")
STOP = block_note(None, second=True, words="sends a form")
DENIED = ("blocked", "approval_denied")
TOOL_ERROR = ("error", "tool_error")
FINE = ("success", "none")


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


def _call(
    seq: int,
    kind: tuple[str, str],
    words: str = "",
    *,
    field: str = "error",
) -> dict[str, Any]:
    """A ``tool.finished`` event of *kind* (status, error category) with *words* in
    its error (where a failed call keeps its text) or in its output."""
    status, category = kind
    data: dict[str, Any] = {"error": "", "output": "", field: words}
    return core_event(
        "tool.finished",
        seq=seq,
        status=status,
        error_category=category,
        call_id=f"call_{seq}",
        **data,
    )


async def _turn(
    store: SessionStore,
    calls: tuple[tuple[tuple[str, str], str], ...] = (),
    text: str = TYPED,
    *,
    session_id: str | None = None,
    kind: str = "chat",
    title: str | None = None,
    answer: str = ANSWER,
    read: tuple[str, ...] = (),
    field: str = "error",
) -> TurnInfo:
    """A finished turn whose tool calls ended as *calls* (kind and text), in order."""
    info = await store.begin_turn(
        session_id,
        text,
        holder=HOLDER,
        title=title,
        new_session=NewSession(kind=kind) if session_id is None else None,
    )
    events = [
        core_event("inbox.accepted", seq=n, offset=n, text=note)
        for n, note in enumerate(read, 1)
    ]
    events += [
        _call(len(events) + n, kind_, words, field=field)
        for n, (kind_, words) in enumerate(calls, 1)
    ]
    if events:
        await store.append_events(info.session_id, info.turn, events, holder=HOLDER)
    await store.finish_turn(
        info.session_id,
        info.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )
    return (await store.list_turns(info.session_id))[-1]


async def _documents(memory: ChatMemory) -> list[tuple[str, int, int, str]]:
    _, index = memory.stores()
    docs = await index.fetch(await index.ids())
    return sorted((d.session_id, d.turn, d.chunk, d.text) for d in docs)


async def _texts(memory: ChatMemory, info: TurnInfo) -> list[str]:
    return [t for sid, _, _, t in await _documents(memory) if sid == info.session_id]


async def _live(memory: ChatMemory, store: SessionStore, info: TurnInfo) -> list[str]:
    """The documents of *info* as the live index writes them."""
    typed = info.typed_text if info.attachments else info.user_text
    memory.after_turn(store, info, typed, info.answer)
    await memory.drain()
    return await _texts(memory, info)


WORDS_ONLY = f"User: {TYPED}"


async def test_an_executor_denial_last_leaves_the_persons_words_only(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, ((DENIED, "The user denied this call: no."),))
    assert await _live(memory, store, info) == [WORDS_ONLY]


async def test_an_autos_block_note_last_leaves_the_persons_words_only(
    memory: ChatMemory, store: SessionStore
) -> None:
    for number, words in enumerate((NOTE, STOP, STOPPED_NOTE), 1):
        info = await _turn(store, ((TOOL_ERROR, words),), session_id=None)
        assert await _live(memory, store, info) == [WORDS_ONLY], number
        assert ANSWER not in "".join(await _texts(memory, info))


async def test_the_note_may_be_in_the_output_or_behind_a_tools_own_lead(
    memory: ChatMemory, store: SessionStore
) -> None:
    lead = "melo did not click or choose on example.com. "
    for words, field in (
        (NOTE, "output"),
        (lead + NOTE, "error"),
        (lead + STOP, "error"),
    ):
        info = await _turn(store, ((TOOL_ERROR, words),), field=field)
        assert await _live(memory, store, info) == [WORDS_ONLY], words


async def test_a_historical_prefix_last_leaves_the_persons_words_only(
    memory: ChatMemory, store: SessionStore
) -> None:
    for prefix in blocked.HISTORICAL_PREFIXES:
        info = await _turn(store, ((TOOL_ERROR, f"{prefix} It installs matplotlib."),))
        assert await _live(memory, store, info) == [WORDS_ONLY], prefix


async def test_a_block_the_turn_went_on_from_keeps_the_answer(
    memory: ChatMemory, store: SessionStore
) -> None:
    for first in ((DENIED, "The user denied this call."), (TOOL_ERROR, NOTE)):
        info = await _turn(
            store, (first, (FINE, "ran another way")), answer="Done another way."
        )
        assert await _live(memory, store, info) == [
            document_text(TYPED, "Done another way.")
        ]
    # A block and then a call that failed for another reason is not the end either.
    info = await _turn(store, ((TOOL_ERROR, NOTE), (TOOL_ERROR, "no such file")))
    assert await _live(memory, store, info) == [document_text(TYPED, ANSWER)]


async def test_a_tool_error_with_other_text_keeps_the_answer(
    memory: ChatMemory, store: SessionStore
) -> None:
    for words in ("no such file", "", "Auto mode is a setting; the page says so"):
        info = await _turn(store, ((TOOL_ERROR, words),), answer="It failed.")
        assert await _live(memory, store, info) == [document_text(TYPED, "It failed.")]
    # Text that only holds a block note is no block: it must start the text.
    info = await _turn(store, ((TOOL_ERROR, f"error: {NOTE}"),), answer="Nope.")
    assert await _live(memory, store, info) == [document_text(TYPED, "Nope.")]
    # A call that succeeded and quotes a note (a page about melo) is no block.
    info = await _turn(store, ((FINE, NOTE),), answer="Quoted.")
    assert await _live(memory, store, info) == [document_text(TYPED, "Quoted.")]


async def test_a_plain_turn_is_indexed_exactly_as_it_was(
    memory: ChatMemory, store: SessionStore
) -> None:
    plain = await _turn(store, answer="Hi there.")
    ran = await _turn(store, ((FINE, "output"),), answer="It ran.")
    assert await _live(memory, store, plain) == ["User: " + TYPED + "\nmelo: Hi there."]
    assert await _live(memory, store, ran) == [document_text(TYPED, "It ran.")]
    assert document_text(TYPED, "It ran.") == f"User: {TYPED}\nmelo: It ran."
    assert await turn_document(store, plain, TYPED, "Hi there.") == (TYPED, "Hi there.")
    assert await turn_document(store, ran, TYPED, "It ran.") == (TYPED, "It ran.")


async def test_the_words_of_a_blocked_turn_keep_the_notes_the_run_read(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, ((TOOL_ERROR, NOTE),), read=("and use the second form",))
    assert await _live(memory, store, info) == [
        f"User: {TYPED}\n\n{ADDED_MARK}\nand use the second form"
    ]


async def test_a_task_keeps_its_title_and_no_reply_line(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(
        store,
        ((DENIED, "The approval policy denied this call."),),
        "Send the form.",
        kind="task",
        title="Send form",
    )
    assert await _live(memory, store, info) == ["Task: Send form\nUser: Send the form."]


async def test_a_blocked_turn_without_words_holds_nothing(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, ((DENIED, "The user denied this call."),))
    assert await turn_document(store, info, "", ANSWER) is None
    assert await turn_document(store, info, "  \n", ANSWER) is None
    memory.after_turn(store, info, "", ANSWER)
    await memory.drain()
    assert await _documents(memory) == []
    # An empty answer is still no turn to index, as before.
    memory.after_turn(store, info, TYPED, "  ")
    await memory.drain()
    assert await _documents(memory) == []


async def test_a_long_blocked_turn_has_no_reply_line_in_any_chunk(
    memory: ChatMemory, store: SessionStore
) -> None:
    long_words = "Please file the form with these details. " * 120
    info = await _turn(store, ((TOOL_ERROR, NOTE),), long_words.strip())
    texts = await _live(memory, store, info)
    assert len(texts) > 1
    assert all(
        t.startswith("User: ") and "melo:" not in t and t == t.rstrip() for t in texts
    )
    assert document_chunks(long_words.strip(), None) == texts


async def test_the_live_index_and_a_rebuild_give_the_same_documents(
    memory: ChatMemory, store: SessionStore
) -> None:
    lead = "melo did not open example.com. "
    first = await _turn(store, ((DENIED, "The user denied this call."),))
    sid = first.session_id
    turns = [
        first,
        await _turn(store, ((TOOL_ERROR, NOTE),), session_id=sid),
        await _turn(store, ((TOOL_ERROR, lead + STOP),), session_id=sid, field="error"),
        await _turn(store, ((TOOL_ERROR, STOPPED_NOTE),), session_id=sid),
        await _turn(store, ((TOOL_ERROR, NOTE),), session_id=sid, field="output"),
        await _turn(store, ((TOOL_ERROR, NOTE), (FINE, "ok")), session_id=sid),
        await _turn(store, ((TOOL_ERROR, "no such file"),), session_id=sid),
        await _turn(store, ((TOOL_ERROR, NOTE),), session_id=sid, read=("one more",)),
        await _turn(store, (), session_id=sid, answer="Hello."),
        await _turn(store, ((TOOL_ERROR, NOTE),), "Long. " * 600, session_id=sid),
        await _turn(
            store, ((TOOL_ERROR, NOTE),), session_id=None, kind="task", title="T"
        ),
    ]
    for info in turns:
        typed = info.typed_text if info.attachments else info.user_text
        memory.after_turn(store, info, typed, info.answer)
    await memory.drain()
    live = await _documents(memory)
    # Three turns went on or failed otherwise or had no call: their replies stay.
    assert sum("\nmelo: " in t for *_, t in live) == 3
    assert sum("\nmelo: " not in t for *_, t in live) >= 8  # the blocked ones
    _, indexed = await memory.reindex(store)
    assert indexed == len(turns)
    assert await _documents(memory) == live


def test_ended_on_block_follows_the_last_call_only() -> None:
    denied = ("blocked", "approval_denied", "x")
    noted = ("error", "tool_error", NOTE)
    fine = ("success", "none", "ran")
    other = ("error", "tool_error", "no such file")
    assert not blocked.ended_on_block([])
    assert blocked.ended_on_block([denied])
    assert blocked.ended_on_block([fine, noted])
    assert blocked.ended_on_block([noted, denied])
    assert not blocked.ended_on_block([denied, fine])
    assert not blocked.ended_on_block([noted, fine, other])
    assert not blocked.ended_on_block([noted, other])
    # A call that never ran for another reason is no block.
    assert not blocked.ended_on_block([("cancelled", "cancelled", "")])
    assert not blocked.ended_on_block([("timeout", "timeout", NOTE)])


async def test_the_store_read_is_the_last_result_of_the_turn(
    store: SessionStore,
) -> None:
    blockedturn = await _turn(store, ((FINE, "ok"), (TOOL_ERROR, NOTE)))
    went_on = await _turn(
        store, ((TOOL_ERROR, NOTE), (FINE, "ok")), session_id=blockedturn.session_id
    )
    none = await _turn(store, (), session_id=blockedturn.session_id)
    assert await ended_on_block(store, blockedturn)
    assert not await ended_on_block(store, went_on)
    assert not await ended_on_block(store, none)
    assert await turn_words(store, blockedturn, TYPED) == TYPED


def test_the_prefixes_are_the_text_before_the_first_field_of_the_templates() -> None:
    assert blocked.derive_prefixes((BLOCK_NOTE, BLOCK_STOP)) == (
        BLOCK_NOTE.split("{", 1)[0].strip(),
        BLOCK_STOP.split("{", 1)[0].strip(),
    )
    assert blocked.CURRENT_PREFIXES == (
        "Auto mode blocked this step, so it did not run.",
        "Auto mode blocked this step too:",
        "Auto mode stopped this turn after blocking a step, so this step did not run.",
    )
    for template in (BLOCK_NOTE, BLOCK_STOP, STOPPED_NOTE):
        assert any(template.startswith(p) for p in blocked.CURRENT_PREFIXES)
        assert "{" not in "".join(blocked.CURRENT_PREFIXES)
    # A template that starts with a field has no stable text, and none repeats.
    assert blocked.derive_prefixes(("{who} blocked it", "A {x}", "A {y}")) == ("A",)


def test_every_note_the_runtime_builds_starts_with_a_known_prefix() -> None:
    review = pytest.importorskip("localmelo.melo.contracts.review")
    high = review.ActionReview(
        "high", ("send_out",), ("Sends data to another site.",), by="rules"
    )
    notes = [
        block_note(None),
        block_note(None, words="reads a key"),
        block_note(high),
        block_note(high, task=True),
        block_note(high, folder="project"),
        block_note(high, changed=True),
        block_note(None, second=True),
        block_note(high, second=True),
        STOPPED_NOTE,
    ]
    for note in notes:
        assert note.startswith(blocked.BLOCK_PREFIXES), note
        assert blocked.is_block(("error", "tool_error", note))
        assert blocked.is_block(
            ("error", "tool_error", f"melo did not read a.example. {note}")
        )


def test_the_historical_prefixes_are_kept_when_a_template_is_reworded() -> None:
    reworded = blocked.derive_prefixes(("Auto stopped this step. {it}",))
    prefixes = (*reworded, *blocked.HISTORICAL_PREFIXES)
    old_note = (
        blocked.HISTORICAL_PREFIXES[0] + " It installs matplotlib.\nIn your answer"
    )
    assert blocked.starts_with_block(old_note, prefixes)
    assert blocked.starts_with_block(
        "Auto stopped this step. It sends a form.", prefixes
    )
    assert not blocked.starts_with_block(old_note, reworded)
    assert set(blocked.HISTORICAL_PREFIXES) <= set(blocked.BLOCK_PREFIXES)
    assert set(blocked.CURRENT_PREFIXES) <= set(blocked.BLOCK_PREFIXES)


def test_the_words_alone_read_back_as_the_users_and_no_reply() -> None:
    from localmelo.support.memory import records
    from localmelo.support.memory.digest import _reply_of

    text = document_text("My dentist is Okafor.", None)
    assert text == "User: My dentist is Okafor."
    assert typed_of(text) == "My dentist is Okafor."
    assert _reply_of(text) == ""
    assert records.said(typed_of(text)) == records.said("My dentist is Okafor.")
    assert document_text("Hi", "") == "User: Hi\nmelo: "  # an empty answer, as before
