"""A different fact about the same person stays; a new value of one fact replaces.

``records.subject`` read only the user's own subjects ("my/your X is …"), so a
note about a third party ("Tom Briggs's office is in room 9C.") had none, and
``records.decide`` fell back to "same kind and 60% shared content words": it
superseded "Your manager is Tom Briggs." (the benchmark's cell r1__M2, where the
later "Who is my manager?" then failed). A third party's possessive is a subject
too ("Name's X is …", "NAME的X是/在/叫 …"), keyed by its owner, and the fallback
refuses a note that names a subject the other's text does not hold.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.melo.memory.long import StoredEntry
from localmelo.support.memory import ChatMemory, records
from localmelo.support.sessions.store import NewSession, SessionStore

HOLDER = "test-host:1:subjects"
MANAGER = "Your manager is Tom Briggs."
OFFICE = "Tom Briggs's office is in room 9C."


def entry(entry_id: int, text: str, kind: str = "semantic") -> StoredEntry:
    return StoredEntry(
        id=entry_id,
        text=text,
        kind=kind,
        source="user",
        status="active",
        admitted_by="user_statement",
        provenance={},
        task_id="",
        session_id="",
        uses=0,
        created_at=0.0,
        last_used_at=0.0,
        supersedes=None,
        links=(),
        embed_model="",
        embed_dim=0,
        vector=None,
    )


@pytest.mark.parametrize(
    ("text", "subject"),
    [
        (OFFICE, "tom briggs's office"),
        (
            "Dr. Yuki Tanaka's phone number is 555-0230.",
            "dr. yuki tanaka's phone number",
        ),
        ("Prof. Duarte's office is in the Atrium Wing.", "prof. duarte's office"),
        ("Marco's studio is on Main Street.", "marco's studio"),
        ("马可的工作室在中山路88号", "马可的工作室"),
        ("马可的生日是三月九号", "马可的生日"),
        ("周教授的助教叫小林", "周教授的助教"),
        # The user's own subjects are unchanged: no owner in the key.
        (MANAGER, "manager"),
        ("My weekly sync is on Friday.", "weekly sync"),
        ("我的健身教练是小宇", "健身教练"),
        # Not a third party's possessive: a time word, a contraction, a weekday.
        ("Today's meeting is on Friday.", ""),
        ("That's why I left.", ""),
        ("Friday's standup is at nine.", ""),
        # A third party's sentence is never the user's timezone or home.
        ("Tom's time zone differs from mine.", ""),
    ],
)
def test_a_third_partys_possessive_is_a_subject_keyed_by_its_owner(
    text: str, subject: str
) -> None:
    assert records.subject(text) == subject


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # The real pair (cell r1__M2, notes 9 and 10), and the other way round.
        (MANAGER, OFFICE),
        (OFFICE, MANAGER),
        ("马可的工作室在中山路88号", "马可的生日是三月九号"),
        ("Tom Briggs's office is in room 9C.", "Ana Li's office is in room 9C."),
        (
            "Tom Briggs's office is in room 9C.",
            "Tom Briggs's phone number is 555-0147.",
        ),
        # A note that names no subject, and names none of the other's words.
        (MANAGER, "Tom Briggs lives on Elm Street."),
    ],
)
def test_a_different_fact_about_the_same_person_is_a_new_note(
    old: str, new: str
) -> None:
    assert records.decide(new, [entry(9, old)], kind="semantic") == ("NEW", None)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (MANAGER, "Your manager is Ana Li."),
        (OFFICE, "Tom Briggs's office is in room 12A."),
        ("马可的工作室在中山路88号", "马可的工作室在上海路5号"),
        ("马可的生日是三月九号", "马可的生日是四月二号"),
        # The same fact in other words keeps its subject's words.
        (MANAGER, "Tom Briggs is now your manager."),
    ],
)
def test_the_same_owner_and_attribute_with_a_new_value_supersedes(
    old: str, new: str
) -> None:
    assert records.decide(new, [entry(9, old)], kind="semantic") == ("UPDATE", 9)


def test_a_repeat_is_ignored_and_a_new_value_replaces_the_note_of_its_subject() -> None:
    active = [entry(9, MANAGER), entry(10, OFFICE)]
    assert records.decide(OFFICE, active, kind="semantic") == ("IGNORE", 10)
    new = "Tom Briggs's office is in room 12A."
    assert records.decide(new, active, kind="semantic") == ("UPDATE", 10)
    assert records.decide("Your manager is Ana Li.", active, kind="semantic") == (
        "UPDATE",
        9,
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


async def test_the_model_keeping_both_facts_leaves_the_manager_note_active(
    memory: ChatMemory, store: SessionStore
) -> None:
    for typed, note in (("My manager is Tom Briggs.", MANAGER), (OFFICE, OFFICE)):
        turn = await store.begin_turn(
            None, typed, holder=HOLDER, title=typed, new_session=NewSession()
        )
        await memory.before_turn(store, turn, typed)
        assert (await memory.remember(note, "people")).startswith("Remembered")
    assert sorted(e.text for e in await memory.records()) == sorted([MANAGER, OFFICE])
