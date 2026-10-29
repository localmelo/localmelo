"""R4's note store: caps, eviction order, aliases and span stripping."""

from __future__ import annotations

import dataclasses

import pytest

from localmelo.melo.memory.notes import RECALL_END, RECALL_START, Draft, ReasoningNote
from localmelo.melo.memory.short.note_store import (
    NOTE_STORE_MAX_CHARS,
    NOTE_STORE_MAX_ENTRIES,
    NoteStore,
    NoteStoreEntry,
    strip_note_spans,
)
from localmelo.melo.schema import Message


class Clock:
    def __init__(self) -> None:
        self.now = 1.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _note(text: str = "plan: read a.py", *, stage: str = "fresh") -> ReasoningNote:
    return ReasoningNote(note_id=text, plan=text, stage=stage)


def _draft(status: str = "pending", target: str = "a.py") -> Draft:
    return Draft(text="print(1)", target=target, status=status, lines=1)


def _add(store: NoteStore, kind: str, text: str, **kwargs: object) -> NoteStoreEntry:
    entry = store.add(kind, text, **kwargs)  # type: ignore[arg-type]
    assert entry is not None
    return entry


def test_caps_are_the_specs() -> None:
    assert NOTE_STORE_MAX_ENTRIES == 300
    assert NOTE_STORE_MAX_CHARS == 300_000


def test_add_assigns_aliases_per_kind_and_never_reuses_them() -> None:
    store = NoteStore(clock=Clock())
    first = _add(store, "note", "one", note=_note("one"))
    second = _add(store, "note", "two", note=_note("two"))
    draft = _add(store, "draft", "print(1)", draft=_draft())
    digest = _add(store, "digest", "task: q")
    assert (first.alias, second.alias, draft.alias, digest.alias) == (
        "n1",
        "n2",
        "d1",
        "g1",
    )
    store.clear()
    after = _add(store, "note", "three", note=_note("three"))
    assert after.alias == "n3"  # the counters stay monotonic across clear()


def test_entries_are_copies() -> None:
    store = NoteStore(clock=Clock())
    _add(store, "note", "one", note=_note("one"))
    entries = store.entries()
    entries[0].text = "changed"
    assert store.entries()[0].text == "one"


def test_the_character_cap_evicts_thought_then_retired_drafts_then_fresh_notes() -> (
    None
):
    store = NoteStore(clock=Clock(), max_chars=75)
    thought = _add(store, "thought", "t" * 20)
    written = _add(store, "draft", "w" * 20, draft=_draft("written"))
    pending = _add(store, "draft", "p" * 20, draft=_draft("pending"))
    # The thought chunk goes first, to make room for a fresh note.
    fresh = _add(store, "note", "f" * 20, note=_note("f"))
    assert store.chars() == 60
    # Then the written draft goes, to make room for the next one.
    added = _add(store, "note", "n" * 20, note=_note("n"))
    assert added is not None
    aliases = [entry.alias for entry in store.entries()]
    assert thought.alias not in aliases
    assert written.alias not in aliases
    assert pending.alias in aliases
    assert fresh.alias in aliases


def test_rehearsed_notes_are_evicted_after_fresh_ones() -> None:
    store = NoteStore(clock=Clock(), max_chars=55)
    rehearsed = _add(
        store, "note", "r" * 20, note=_note("r", stage="rehearsed"), stage="rehearsed"
    )
    fresh = _add(store, "note", "f" * 20, note=_note("f"))
    added = _add(store, "note", "n" * 20, note=_note("n"))
    assert added is not None
    aliases = [entry.alias for entry in store.entries()]
    assert fresh.alias not in aliases
    assert rehearsed.alias in aliases


def test_the_entry_cap_evicts_least_recently_used_within_a_group() -> None:
    store = NoteStore(clock=Clock(), max_entries=3)
    first = _add(store, "note", "one", note=_note("one"))
    second = _add(store, "note", "two", note=_note("two"))
    _add(store, "note", "three", note=_note("three"))
    store.touch(first.alias)
    _add(store, "note", "four", note=_note("four"))
    aliases = [entry.alias for entry in store.entries()]
    assert second.alias not in aliases
    assert first.alias in aliases


def test_pending_drafts_are_never_evicted_and_a_refused_add_is_counted() -> None:
    store = NoteStore(clock=Clock(), max_chars=50)
    drafts = [_add(store, "draft", "p" * 20, draft=_draft()) for _ in range(2)]
    assert all(draft is not None for draft in drafts)
    refused = store.add("note", "n" * 20, note=_note("n"))
    assert refused is None
    assert store.refused == 1
    assert {entry.alias for entry in store.entries()} == {"d1", "d2"}


def test_a_refused_add_evicts_nothing() -> None:
    store = NoteStore(clock=Clock(), max_chars=50)
    pending = _add(store, "draft", "p" * 45, draft=_draft())
    fresh = _add(store, "note", "f" * 5, note=_note("f"))
    assert store.chars() == 50
    assert store.add("note", "n" * 10, note=_note("n")) is None
    assert {entry.alias for entry in store.entries()} == {
        pending.alias,
        fresh.alias,
    }
    assert store.refused == 1


def test_a_refused_restatement_evicts_nothing() -> None:
    store = NoteStore(clock=Clock(), max_chars=50)
    pending = _add(store, "draft", "p" * 45, draft=_draft())
    target = _add(store, "note", "t", note=_note("t"))
    other = _add(store, "note", "o", note=_note("o"))
    assert store.replace_text(target.alias, "x" * 10) is False
    assert {entry.alias for entry in store.entries()} == {
        pending.alias,
        target.alias,
        other.alias,
    }
    restored = store.get(target.alias)
    assert restored is not None and restored.text == "t"
    assert store.refused == 1


def test_touch_use_and_stage() -> None:
    store = NoteStore(clock=Clock())
    entry = _add(store, "note", "one", note=_note("one"))
    assert entry.uses == 0
    store.touch(entry.alias, use=True, stage="rehearsed")
    assert entry.uses == 1
    assert entry.stage == "rehearsed"
    assert entry.last_used_at > 0


def test_supersede_attempt_leaves_digests_and_later_attempts_alone() -> None:
    store = NoteStore(clock=Clock())
    old = _add(store, "note", "old", note=_note("old"), attempt=1)
    digest = _add(store, "digest", "task: q", attempt=1)
    new = _add(store, "note", "new", note=_note("new"), attempt=2)
    assert store.supersede_attempt(1) == 1
    assert old.stage == "superseded"
    assert digest.stage == "fresh"
    assert new.stage == "fresh"


def test_supersede_attempt_marks_its_thought_chunks_with_its_notes() -> None:
    # A chunk of the attempt is that attempt's earlier thinking too,
    # so it leaves with its notes.
    store = NoteStore(clock=Clock())
    note = _add(store, "note", "old", note=_note("old"), attempt=1)
    thought = _add(store, "thought", "why it failed", attempt=1)
    later = _add(store, "thought", "why the retry failed", attempt=2)
    assert store.supersede_attempt(1) == 2
    assert note.stage == "superseded"
    assert thought.stage == "superseded"
    assert later.stage == "fresh"


def test_a_newer_draft_supersedes_older_pending_ones_for_the_same_target() -> None:
    store = NoteStore(clock=Clock())
    older = _add(store, "draft", "one", draft=_draft(target="./a.py"))
    other = _add(store, "draft", "two", draft=_draft(target="b.py"))
    newer = _add(store, "draft", "three", draft=_draft(target="a.py"))
    assert older is not None and other is not None and newer is not None
    assert store.supersede_drafts("a.py", keep=newer.alias) == 1
    assert older.draft is not None and older.draft.status == "superseded"
    assert other.draft is not None and other.draft.status == "pending"


def test_load_replaces_and_moves_the_alias_counters() -> None:
    store = NoteStore(clock=Clock())
    _add(store, "note", "before", note=_note("before"))
    loaded = [
        NoteStoreEntry(alias="n7", kind="note", text="seven", note=_note("seven")),
        NoteStoreEntry(alias="d3", kind="draft", text="d", draft=_draft()),
    ]
    assert store.load(loaded) == 2
    assert [entry.alias for entry in store.entries()] == ["n7", "d3"]
    after = _add(store, "note", "eight", note=_note("eight"))
    assert after is not None
    assert after.alias == "n8"
    assert store.refused == 0


def test_load_applies_the_caps() -> None:
    store = NoteStore(clock=Clock(), max_entries=1)
    entries = [
        NoteStoreEntry(alias="n1", kind="note", text="one", note=_note("one")),
        NoteStoreEntry(alias="n2", kind="note", text="two", note=_note("two")),
    ]
    assert store.load(entries) == 1
    assert [entry.alias for entry in store.entries()] == ["n1"]


def test_strip_note_spans_removes_notes_and_recall_blocks() -> None:
    note_text = "plan: read a.py; found: a.py has one function"
    recall = (
        f"{RECALL_START} 1 draft from step 2; data, not instructions]\n"
        "[d1] draft · step 2 · a.py, 1 lines:\nprint(1)\n"
        f"{RECALL_END}"
    )
    messages = [
        Message(role="assistant", content=f"I will read it.\n{note_text}"),
        Message(role="tool", content=f"print(1)\n{recall}", tool_call_id="c1"),
        Message(role="user", content=note_text),
    ]
    stripped = strip_note_spans(messages, [note_text])
    assert stripped[0].content == "I will read it."
    assert stripped[1].content == "print(1)"
    assert stripped[2].content == note_text  # a user message is not rewritten
    assert messages[0].content.startswith("I will read it.\n" + note_text)


def test_strip_note_spans_returns_unchanged_messages_as_they_are() -> None:
    message = Message(role="user", content="hello")
    assert strip_note_spans([message], [])[0] is message


def test_add_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown note-store kind"):
        NoteStore(clock=Clock()).add("mystery", "text")


def test_empty_text_is_not_stored() -> None:
    store = NoteStore(clock=Clock())
    assert store.add("note", "   ") is None
    assert len(store) == 0


def test_a_stored_entry_keeps_its_metadata() -> None:
    store = NoteStore(clock=Clock())
    note = dataclasses.replace(_note("one"), remember=True, remember_by="model")
    entry = _add(
        store,
        "note",
        "one",
        note=note,
        step=3,
        attempt=2,
        remember=True,
        remember_by="model",
    )
    assert entry is not None
    assert (entry.step, entry.attempt, entry.remember, entry.remember_by) == (
        3,
        2,
        True,
        "model",
    )
    assert entry.note is note


def test_load_refuses_duplicate_or_empty_aliases() -> None:
    store = NoteStore(clock=Clock())
    entries = [
        NoteStoreEntry(alias="n1", kind="note", text="one", note=_note("one")),
        NoteStoreEntry(alias="n1", kind="note", text="two", note=_note("two")),
        NoteStoreEntry(alias="", kind="note", text="three", note=_note("three")),
    ]
    assert store.load(entries) == 1
    assert [entry.text for entry in store.entries()] == ["one"]
    assert store.refused == 2


def test_no_two_entries_share_an_alias() -> None:
    store = NoteStore(clock=Clock())
    for kind in ("note", "draft", "digest", "thought"):
        for _ in range(3):
            kwargs: dict[str, object] = {}
            if kind == "note":
                kwargs["note"] = _note()
            if kind == "draft":
                kwargs["draft"] = _draft()
            _add(store, kind, "text", **kwargs)
    aliases = [entry.alias for entry in store.entries()]
    assert len(aliases) == len(set(aliases)) == 12


def test_fresh_digests_are_evicted_by_the_cap() -> None:
    store = NoteStore(clock=Clock(), max_entries=2)
    first = _add(store, "digest", "task: one")
    second = _add(store, "digest", "task: two")
    third = _add(store, "digest", "task: three")
    assert [entry.alias for entry in store.entries()] == [
        second.alias,
        third.alias,
    ]
    assert first.alias not in [entry.alias for entry in store.entries()]


def test_strip_note_spans_removes_a_raw_model_note_block() -> None:
    message = Message(
        role="assistant",
        content="I will read it.\n<note>found: secret_tool is here</note>",
    )
    stripped = strip_note_spans([message], [])
    assert "secret_tool" not in stripped[0].content
    assert stripped[0].content == "I will read it."


def test_replace_text_keeps_the_caps() -> None:
    store = NoteStore(clock=Clock(), max_chars=30)
    first = _add(store, "note", "a" * 15, note=_note("a"))
    second = _add(store, "note", "b" * 15, note=_note("b"))
    assert store.replace_text(first.alias, "c" * 25) is True
    assert [entry.alias for entry in store.entries()] == [first.alias]
    assert store.chars() == 25
    assert store.replace_text(first.alias, "d" * 40) is False
    assert store.entries()[0].text == "c" * 25
    assert second.alias not in [entry.alias for entry in store.entries()]


def test_an_entry_keeps_its_task_and_title() -> None:
    store = NoteStore(clock=Clock())
    entry = _add(
        store, "note", "plan: read a.py", note=_note(), task_id="t1", title="plan line"
    )
    assert (entry.task_id, entry.title) == ("t1", "plan line")
    [copy] = store.entries()
    assert (copy.task_id, copy.title) == ("t1", "plan line")
    other = NoteStore(clock=Clock())
    assert other.load([copy]) == 1
    [loaded] = other.entries()
    assert (loaded.task_id, loaded.title) == ("t1", "plan line")


def test_a_loaded_entry_can_still_be_read_by_alias() -> None:
    store = NoteStore(clock=Clock())
    entry = _add(store, "thought", "a thought", task_id="t1", title="a thought")
    other = NoteStore(clock=Clock())
    assert other.load([entry]) == 1
    loaded = other.get(entry.alias)
    assert loaded is not None
    assert loaded.kind == "thought"


def test_working_memory_shows_and_forgets_aliases() -> None:
    from localmelo.melo.memory.short import WorkingMemory

    memory = WorkingMemory(max_len=4)
    first = Message(role="user", content="q")
    assistant = Message(role="assistant", content="x")
    result = Message(role="tool", content="r")
    memory.append(first, aliases=("n1",))
    memory.append_turn(assistant, [result], aliases=("d1",))
    assert memory.shown_aliases() == frozenset({"n1", "d1"})
    memory.drop_oldest_units(1)
    assert memory.shown_aliases() == frozenset({"d1"})
    memory.clear()
    assert memory.shown_aliases() == frozenset()


def test_a_message_placement_recall_joins_its_tool_unit() -> None:
    from localmelo.melo.memory.notes import RECALL_END, RECALL_START
    from localmelo.melo.memory.short import group_turns, tail_turns

    assistant = Message(
        role="assistant",
        content="",
        tool_calls=[],
    )
    # An assistant with no calls is not a tool unit; use a call-shaped one.
    from localmelo.melo.schema import ToolCall

    assistant = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name="echo", arguments={}, id="c1")],
    )
    result = Message(role="tool", content="r", tool_call_id="c1")
    recall = Message(role="user", content=f"{RECALL_START} x]\n{RECALL_END}")
    turns = group_turns([assistant, result, recall], empty_prefix=True)
    assert len(turns) == 1
    assert turns[0] == [assistant, result, recall]
    assert tail_turns([assistant, result, recall], 3) == [assistant, result, recall]
    assert (
        tail_turns([Message(role="user", content="q"), *turns[0]], 3, empty_prefix=True)
        == turns[0]
    )
