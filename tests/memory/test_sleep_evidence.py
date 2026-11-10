"""Sleep keeps evidence and inactive history for every operation, without training."""

from __future__ import annotations

from dataclasses import replace

import pytest

from localmelo.support.memory import ChatMemory, sleep
from localmelo.support.sessions.store import SessionStore

from . import test_sleep as rules

memory = rules.memory
chats = rules.chats


async def test_mixed_run_reports_evidence_and_retains_rows_through_undo_redo(
    memory: ChatMemory,
) -> None:
    typed = [
        "I live in Utrecht.",
        "My name is Juno.",
        "My advisor is Dr. Okafor.",
        "I enjoy kitesurfing.",
    ]
    turns = [
        sleep.SleepTurn(
            "s", i, rules.LONG_AGO + 100 + i, rules.LONG_AGO + 101 + i, text
        )
        for i, text in enumerate(typed, 1)
    ]
    await rules.put(memory, "You live in Lisbon.")
    await rules.put(
        memory, "You live in Utrecht.", rules.LONG_AGO + 5, session="s", turn=1
    )
    await rules.put(memory, "You go by Pip.")
    before = {n.id: n.text for n in await memory.records()}
    report = await sleep.run(
        *memory.stores(), turns, variant="rules", now=rules.LONG_AGO + 200
    )
    assert not report.aborted
    assert {c.op for c in report.changes} == {"O1", "O2", "O3"}
    payload = report.to_json()
    assert payload["run_id"] == report.run_id and payload["variant"] == "rules"
    for change in payload["changes"]:
        session, number = change["source_turn"]
        source = next(t for t in turns if (t.session_id, t.turn) == (session, number))
        assert change["quote"] and change["quote"] in source.typed
        assert change["before_texts"] == [before[i] for i in change["before_ids"]]
    all_ids = {i for i, _, _ in rules.rows(memory)}
    before_ids = {i for change in report.changes for i in change.before_ids}
    assert before_ids <= all_ids
    await memory.undo_sleep(report.run_id)
    assert {n.id: n.text for n in await memory.records()} == before
    assert {i for i, _, _ in rules.rows(memory)} == all_ids
    # The undo sticks: the same pairs stay apart and the same notes are not written.
    assert not (await sleep.run(*memory.stores(), turns, variant="rules")).changes
    # Said again in new turns, the user's words are new evidence.
    said_again = [replace(t, turn=t.turn + 100) for t in turns]
    redone = await sleep.run(*memory.stores(), said_again, variant="rules")
    assert redone.changes
    assert all_ids <= {i for i, _, _ in rules.rows(memory)}
    assert before_ids <= {i for i, _, _ in rules.rows(memory)}


async def test_undo_keeps_a_created_row_later_used_as_a_before_id(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await rules.chat(chats, "I live in Lisbon.")
    first = await memory.sleep("rules", chats=chats)
    created = first.changes[0].after_id
    assert created is not None
    await rules.chat(chats, "I live in Utrecht.")
    second = await memory.sleep("rules", chats=chats)
    assert [c.before_ids for c in second.changes] == [[created]]
    await memory.undo_sleep(second.run_id)
    await memory.undo_sleep(first.run_id)
    assert await memory.records() == []
    retained = await memory.stores()[0].entry(created)
    assert retained is not None and retained.text == "You live in Lisbon."
    await rules.chat(chats, "I live in Lisbon.")
    assert (await memory.sleep("rules", chats=chats)).changes
    assert await memory.stores()[0].entry(created) is not None


async def test_undone_rows_are_marked_and_excluded_from_every_active_surface(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await rules.chat(chats, "My advisor is Dr. Okafor.")
    report = await memory.sleep("rules", chats=chats)
    created = report.changes[0].after_id
    assert await memory.search("who is my advisor?", docs_k=0)  # warm recall cache
    await memory.undo_sleep(report.run_id)
    store, _ = memory.stores()
    assert any(
        d["target_id"] == created and d["reason"] == f"undone: sleep {report.run_id}"
        for d in await store.decisions()
    )
    assert (created, "Your advisor is Dr. Okafor.", "superseded") in rules.rows(memory)
    assert await memory.search("who is my advisor?", docs_k=0) == []
    assert await memory.tools().search("who is my advisor?", scope="records") == (
        "Nothing in melo's memory matches that."
    )
    assert await memory.records() == []  # the Memory tab's active-list source
    assert (await memory.status())["records"] == 0


@pytest.mark.parametrize("mode", ["one", "all", "session", "chat_notes"])
async def test_user_forgetting_removes_retained_undo_rows(
    memory: ChatMemory, chats: SessionStore, mode: str
) -> None:
    source = await rules.chat(chats, "My advisor is Dr. Okafor.")
    report = await memory.sleep("rules", chats=chats)
    created = report.changes[0].after_id
    assert created is not None
    await memory.undo_sleep(report.run_id)
    assert await memory.stores()[0].entry(created) is not None
    same_chat = await rules.put(
        memory, "Your dentist is Lee Park.", session=source.session_id
    )
    other_chat = await rules.put(memory, "You enjoy kitesurfing.", session="other")
    if mode == "one":
        assert await memory.delete_record(created)
    elif mode == "all":
        await memory.forget_all()
    elif mode == "session":
        await memory.forget_session(source.session_id)
    else:
        await memory.forget_chat_notes(source.session_id)
    assert await memory.stores()[0].entry(created) is None
    remaining = {n.id for n in await memory.records()}
    assert (same_chat in remaining) == (mode in {"one", "session"})
    assert (other_chat in remaining) == (mode != "all")


async def test_forgetting_an_active_note_also_erases_undone_descendants(
    memory: ChatMemory, chats: SessionStore
) -> None:
    old = await rules.put(memory, "You go by Pip.")
    await rules.chat(chats, "My name is Juno.")
    report = await memory.sleep("rules", chats=chats)
    new = report.changes[0].after_id
    assert new is not None
    await memory.undo_sleep(report.run_id)
    assert await memory.delete_record(old)
    store, _ = memory.stores()
    assert await store.entry(old) is None and await store.entry(new) is None


@pytest.mark.parametrize("secret", [False, True])
async def test_duplicate_fallback_quote_is_bounded_and_secret_free(
    memory: ChatMemory, secret: bool
) -> None:
    await rules.put(memory, "You live in Lisbon.")
    await rules.put(
        memory, "You live in Utrecht.", rules.LONG_AGO + 5, session="s", turn=1
    )
    typed = "I live in Utrecht. " + ("My password is hunter2." if secret else "x" * 400)
    report = await sleep.run(
        *memory.stores(), [sleep.SleepTurn("s", 1, 1, 2, typed)], variant="rules"
    )
    assert [c.op for c in report.changes] == ["O1"]
    assert report.to_json()["changes"][0]["quote"] == (
        "I live in Utrecht." if secret else typed[:300]
    )
