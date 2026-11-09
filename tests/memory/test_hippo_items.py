"""Memory items in the conversation index, and Hippo's routing of a new item.

The ``items`` table is created with the others and holds what a chat model encoded
from chats; deleting a chat, or one turn an item names, takes the item with it, and
``forget_all`` clears them. A turn is hidden from recall only while a stored item
cites it. A new item is routed against the active items of every kind: NEW, UPDATE (the
target is marked replaced, unless it is the newer fact; only a judge's clear answer
gives one), RELATED (linked both ways), CONFLICT (both marked, both stay) or IGNORE
(not stored), and is stored in one transaction with its update or link. Synthetic
items, no model: a judge's answer is built by hand.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from localmelo.melo.memory.hippo.policy import decide
from localmelo.support.memory import hippo_encode, hippo_judge, hippo_route, note_judge
from localmelo.support.memory.hippo_encode import Cache
from localmelo.support.memory.hippo_route import (
    POOL,
    REFUSE,
    SHORTLIST,
    Judged,
    Router,
)
from localmelo.support.memory.index import (
    ConversationIndex,
    Document,
    MemoryItem,
    document_chunks,
)
from localmelo.tests.memory.hippo_helpers import Held, StubEncoder, relate, verdict

MODEL = "fixed-embed"


@pytest.fixture()
async def index(tmp_path: Path) -> AsyncIterator[ConversationIndex]:
    index = ConversationIndex(str(tmp_path / "index.db"))
    try:
        yield index
    finally:
        await index.aclose()


def make(
    text: str,
    *,
    session: str = "s1",
    turns: tuple[int, ...] = (1,),
    kind: str = "episodic",
    title: str = "",
    at: float = 1000.0,
    vector: tuple[float, ...] | None = None,
) -> MemoryItem:
    return MemoryItem(
        session_id=session,
        title=title or text[:30],
        text=text,
        kind=kind,
        created_at=at,
        sources=tuple((session, turn) for turn in turns),
        vector=vector,
        embed_model=MODEL if vector else "",
        encoder="1:stub",
    )


async def add_turn(
    index: ConversationIndex,
    session: str,
    turn: int,
    *,
    kind: str = "chat",
    answer: str = "Answer.",
) -> None:
    await index.add_many(
        [
            Document(
                session_id=session,
                turn=turn,
                surface="gateway",
                kind=kind,
                title=f"Chat {session}",
                text=text,
                created_at=1000.0 + turn,
                chunk=n,
            )
            for n, text in enumerate(document_chunks(f"Question {turn}?", answer))
        ]
    )


# ── The store ──


async def test_an_item_is_stored_and_read_back_whole(index: ConversationIndex) -> None:
    item = MemoryItem(
        session_id="s1",
        title="Home",
        text="The user said they live in Pittsburgh.",
        kind="profile",
        created_at=1234.5,
        sources=(("s1", 2), ("s1", 5)),
        vector=(0.25, 0.5, 0.75),
        embed_model=MODEL,
        encoder="1:stub-encoder",
    )
    number = await index.add_item(item)
    assert number == 1 and await index.add_item(make("Another item, long enough.")) == 2
    (read,) = await index.fetch_items([1])
    assert read == MemoryItem(**{**item.__dict__, "id": 1})
    assert (read.status, read.replaced_by, read.related, read.conflicts) == (
        "active",
        None,
        (),
        (),
    )
    assert await index.count_items() == 2
    assert [i.id for i in await index.fetch_items([2, 1, 99])] == [1, 2]
    states = await index.item_states()
    assert set(states) == {1, 2} and states[1].sources == (("s1", 2), ("s1", 5))


async def test_the_table_is_made_on_open_beside_an_older_index(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT "
        "NOT NULL, turn INTEGER NOT NULL, surface TEXT NOT NULL DEFAULT '', kind TEXT "
        "NOT NULL DEFAULT 'chat', title TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, "
        "created_at REAL NOT NULL, vector BLOB, embed_model TEXT NOT NULL DEFAULT '', "
        "chunk INTEGER NOT NULL DEFAULT 0, UNIQUE (session_id, turn, chunk));"
        "INSERT INTO documents (session_id, turn, text, created_at) "
        "VALUES ('s1', 1, 'User: hi\nmelo: hello', 5.0);"
    )
    old.commit()
    old.close()
    index = ConversationIndex(str(path))
    try:
        assert await index.count() == 1 and await index.count_items() == 0
        # No mark is no migration, and no item cites the turn: it is recalled.
        assert await index.encoded_turn("s1") is None
        assert await index.ids(uncited=True) == await index.ids()
        assert await index.add_item(make("An item for an older index.")) == 1
        assert await index.ids(uncited=True) == set()  # the item cites turn 1
        await index.mark_encoded("s1", 1)
        assert await index.encoded_turn("s1") == 1
    finally:
        await index.aclose()
    columns = [
        row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(items)")
    ]
    assert columns[:2] == ["id", "session_id"]
    assert {"title", "text", "kind", "created_at", "sources", "status"} <= set(columns)
    assert {"replaced_by", "related", "vector", "embed_model", "encoder"} <= set(
        columns
    )


async def test_active_items_are_listed_by_kind_and_leave_the_replaced_and_conflicting(
    index: ConversationIndex,
) -> None:
    a = await index.add_item(make("Profile item one.", kind="profile"))
    b = await index.add_item(make("Profile item two.", kind="profile"))
    c = await index.add_item(make("Episodic item.", kind="episodic"))
    d = await index.add_item(make("Semantic item.", kind="semantic"))
    assert [i.id for i in await index.active_items()] == [a, b, c, d]
    assert [i.id for i in await index.active_items("profile")] == [a, b]
    await relate(index, replace=(a, b))
    await relate(index, link=(c, d), conflict=True)
    assert [i.id for i in await index.active_items()] == [b]
    assert (await index.active_items("episodic")) == []


async def test_links_are_both_ways_and_a_replaced_item_keeps_its_replacer(
    index: ConversationIndex,
) -> None:
    a = await index.add_item(make("First item text."))
    b = await index.add_item(make("Second item text."))
    c = await index.add_item(make("Third item text."))
    await relate(index, link=(a, b))
    await relate(index, link=(a, b))  # twice is once
    await relate(index, replace=(c, a))
    first, second, third = await index.fetch_items([a, b, c])
    assert (first.related, second.related, third.related) == ((b,), (a,), ())
    assert (first.status, second.status) == ("active", "active")
    assert (third.status, third.replaced_by) == ("replaced", a)
    await relate(index, link=(a, c), conflict=True)  # a conflict with a replaced one
    first, _, third = await index.fetch_items([a, b, c])
    assert first.conflicts == (c,) and first.status == "conflict"
    assert third.status == "replaced"


async def test_a_conflict_marks_both_and_replacing_one_ends_it(
    index: ConversationIndex,
) -> None:
    a = await index.add_item(make("The user said they like spicy food."))
    b = await index.add_item(make("The user said they never like spicy food."))
    await relate(index, link=(b, a), conflict=True)
    first, second = await index.fetch_items([a, b])
    assert (first.status, first.conflicts) == ("conflict", (b,))
    assert (second.status, second.conflicts) == ("conflict", (a,))
    await relate(index, replace=(a, b))
    first, second = await index.fetch_items([a, b])
    assert (first.status, first.conflicts, first.replaced_by) == ("replaced", (), b)
    assert (second.status, second.conflicts) == ("active", ())


# ── Deleting ──


async def uncited(index: ConversationIndex) -> set[tuple[str, int]]:
    """The ``(chat, turn)`` of every turn no item cites: what recall reads as a turn."""
    docs = await index.fetch(await index.ids(uncited=True))
    return {(doc.session_id, doc.turn) for doc in docs}


async def test_deleting_a_chat_deletes_every_item_that_names_it(
    index: ConversationIndex,
) -> None:
    for session in ("s1", "s2"):
        for turn in (1, 2, 3):
            await add_turn(index, session, turn)
    own = await index.add_item(make("Only in the first chat.", turns=(1, 2)))
    shared = await index.add_item(
        MemoryItem(
            "s2",
            "Shared",
            "Said in both chats.",
            "episodic",
            1.0,
            sources=(("s1", 3), ("s2", 1)),
        )
    )
    other = await index.add_item(make("Only in the second.", session="s2", turns=(2,)))
    assert await index.delete("s1") == 3
    # The shared item rested on a turn that is gone: it goes whole.
    assert [i.id for i in await index.fetch_items([own, shared, other])] == [other]
    assert await index.count() == 3
    # Its other turn has no item now: it is recalled as a turn again.
    assert await uncited(index) == {("s2", 1), ("s2", 3)}


async def test_deleting_one_turn_deletes_the_items_that_name_it_and_frees_their_other_turns(
    index: ConversationIndex,
) -> None:
    for turn in (1, 2, 3):
        await add_turn(index, "s1", turn)
    both = await index.add_item(make("Said in two turns.", turns=(1, 2)))
    one = await index.add_item(make("Said in one turn.", turns=(2,)))
    elsewhere = await index.add_item(make("Said in the third.", turns=(3,)))
    assert await uncited(index) == set()
    assert await index.delete("s1", 2) == 1
    # A text that rested on two turns cannot be shown to rest on the one that is
    # left: both items go, and turn 1 comes back to recall.
    assert [i.id for i in await index.fetch_items([both, one, elsewhere])] == [
        elsewhere
    ]
    assert await uncited(index) == {("s1", 1)}


async def test_deleting_an_item_makes_what_it_replaced_current_and_ends_a_conflict(
    index: ConversationIndex,
) -> None:
    await add_turn(index, "s1", 1)
    await add_turn(index, "s2", 1)
    old = await index.add_item(
        make("The user said they live in Pittsburgh.", session="s1")
    )
    new = await index.add_item(
        make("The user said they now live in Utrecht.", session="s2")
    )
    await relate(index, replace=(old, new))
    liked = await index.add_item(
        make("The user said they like spicy food.", session="s1")
    )
    never = await index.add_item(
        make("The user said they never like spicy food.", session="s2")
    )
    await relate(index, link=(liked, never), conflict=True)
    related = await index.add_item(make("A related item text here.", session="s2"))
    await relate(index, link=(liked, related))
    await index.delete("s2")
    survivors = {i.id: i for i in await index.fetch_items([old, new, liked, never])}
    assert set(survivors) == {old, liked}
    assert (survivors[old].status, survivors[old].replaced_by) == ("active", None)
    assert (survivors[liked].status, survivors[liked].conflicts) == ("active", ())
    assert survivors[liked].related == ()


async def test_deleting_the_middle_of_a_replacement_chain_leaves_the_end_replacing_the_first(
    index: ConversationIndex,
) -> None:
    for turn in (1, 2, 3, 4):
        await add_turn(index, "s1", turn)
    a = await index.add_item(make("The user lived in Pittsburgh.", turns=(1,), at=1.0))
    b = await index.add_item(make("The user lived in Utrecht.", turns=(2, 3), at=2.0))
    c = await index.add_item(make("The user lives in Lisbon.", turns=(4,), at=3.0))
    await relate(index, replace=(a, b))
    await relate(index, replace=(b, c))
    await index.delete("s1", 3)  # one of the two turns B rests on: B goes whole
    first, last = await index.fetch_items([a, c])
    assert [i.id for i in await index.fetch_items([a, b, c])] == [a, c]
    # A is still older than C: it is not current again.
    assert (first.status, first.replaced_by) == ("replaced", c)
    assert (last.status, last.replaced_by) == ("active", None)


async def test_deleting_the_one_two_older_items_were_replaced_by_keeps_the_newer_of_them(
    index: ConversationIndex,
) -> None:
    for turn in (1, 2, 3):
        await add_turn(index, "s1", turn)
    router = Router(index)
    # The newest fact is routed first, then two older ones: both are stored already
    # replaced by it.
    for text, at, turn in (
        ("The user said they now live in Lisbon.", 3.0, 3),
        (UTRECHT, 2.0, 2),
        (PITTSBURGH, 1.0, 1),
    ):
        await route(
            router,
            text,
            "changes" if turn < 3 else None,
            kind="profile",
            at=at,
            turns=(turn,),
        )
    lisbon, utrecht, pittsburgh = await index.fetch_items([1, 2, 3])
    assert (utrecht.replaced_by, pittsburgh.replaced_by) == (1, 1)
    await index.delete("s1", 3)
    utrecht, pittsburgh = await index.fetch_items([2, 3])
    # With Lisbon forgotten the newest of the two is current, and the other is older
    # than it, not a second current fact.
    assert (utrecht.status, utrecht.replaced_by) == ("active", None)
    assert (pittsburgh.status, pittsburgh.replaced_by) == ("replaced", 2)
    assert [i.id for i in await index.active_items("profile")] == [2]


async def test_deleting_one_item_frees_its_turns_and_an_undo_puts_it_back_as_it_was(
    index: ConversationIndex,
) -> None:
    for turn in (1, 2, 3):
        await add_turn(index, "s1", turn)
    held = await index.add_item(
        make("Said in two turns.", turns=(1, 2), vector=(0.25, 0.5))
    )
    other = await index.add_item(make("Said in the third.", turns=(3,)))
    assert await uncited(index) == set()
    gone = await index.delete_item(held)
    assert gone is not None and (gone.id, gone.text) == (held, "Said in two turns.")
    assert [i.id for i in await index.fetch_items([held, other])] == [other]
    assert await uncited(index) == {("s1", 1), ("s1", 2)}
    assert await index.delete_item(held) is None, "there is nothing left to delete"
    assert await index.restore_item(gone)
    assert await index.fetch_items([held]) == [gone], "back whole, under its own id"
    assert await uncited(index) == set()
    assert not await index.restore_item(gone), "its id is in use again"
    assert await index.add_item(make("Written after the undo.")) > other


async def test_deleting_a_replaced_or_conflicting_item_undoes_its_links_and_an_undo_restores_them(
    index: ConversationIndex,
) -> None:
    old = await index.add_item(make("The user said they live in Pittsburgh."))
    new = await index.add_item(make("The user said they now live in Utrecht."))
    await relate(index, replace=(old, new))
    liked = await index.add_item(make("The user said they like spicy food."))
    never = await index.add_item(make("The user said they never like spicy food."))
    await relate(index, link=(liked, never), conflict=True)
    near = await index.add_item(make("The user said they cook at home."))
    await relate(index, link=(old, near))

    async def read(
        *ids: int,
    ) -> list[tuple[str, int | None, tuple[int, ...], tuple[int, ...]]]:
        return [
            (i.status, i.replaced_by, i.related, i.conflicts)
            for i in await index.fetch_items(ids)
        ]

    # A replaced item goes: the newer one that replaced it stays, links to it are dropped.
    gone_old = await index.delete_item(old)
    assert await read(new, near) == [
        ("active", None, (), ()),
        ("active", None, (), ()),
    ]
    # A conflicting item goes: its partner is no longer in conflict.
    gone_never = await index.delete_item(never)
    assert await read(liked) == [("active", None, (), ())]
    # An undo puts the links back, both ways.
    assert gone_old is not None and await index.restore_item(gone_old)
    assert gone_never is not None and await index.restore_item(gone_never)
    assert await read(old, liked, never, near) == [  # in id order
        ("replaced", new, (near,), ()),
        ("conflict", None, (), (never,)),
        ("conflict", None, (), (liked,)),
        ("active", None, (old,), ()),
    ]
    # A partner deleted in between is not linked to: the item comes back on its own.
    gone_never = await index.delete_item(never)
    await index.delete_item(liked)
    assert gone_never is not None and await index.restore_item(gone_never)
    assert await read(never) == [("active", None, (), ())]


async def test_deleting_the_item_that_replaced_another_leaves_the_older_one_replaced(
    index: ConversationIndex,
) -> None:
    """The person called the newer fact wrong or unwanted; that does not make the older
    one true again, so it stays history (never current, never listed as active)."""
    old = await index.add_item(make("The user said they live in Pittsburgh."))
    new = await index.add_item(make("The user said they now live in Utrecht."))
    await relate(index, replace=(old, new))
    gone = await index.delete_item(new)
    (older,) = await index.fetch_items([old])
    assert (older.status, older.replaced_by) == ("replaced", new)
    assert await index.active_items() == []
    # A chat that goes later does not bring it back either: its replacer is not in that set.
    await add_turn(index, "s2", 1)
    other = await index.add_item(make("An item of another chat.", session="s2"))
    await index.delete("s2")
    assert (await index.fetch_items([old]))[0].status == "replaced" and other
    # An undo makes the pointer good again.
    assert gone is not None and await index.restore_item(gone)
    assert (await index.fetch_items([old, new]))[0].replaced_by == new


async def test_deleting_the_middle_of_a_chain_by_hand_keeps_the_order_and_an_undo_keeps_it_older(
    index: ConversationIndex,
) -> None:
    a = await index.add_item(make("The user lived in Pittsburgh.", at=1.0))
    b = await index.add_item(make("The user lived in Utrecht.", at=2.0))
    c = await index.add_item(make("The user lives in Lisbon.", at=3.0))
    await relate(index, replace=(a, b))
    await relate(index, replace=(b, c))
    gone = await index.delete_item(b)
    first, last = await index.fetch_items([a, c])
    assert (first.status, first.replaced_by) == ("replaced", c), "still older than C"
    assert (last.status, last.replaced_by) == ("active", None)
    assert gone is not None and await index.restore_item(gone)
    assert [(i.status, i.replaced_by) for i in await index.fetch_items([a, b, c])] == [
        ("replaced", c),
        ("replaced", c),
        ("active", None),
    ]


async def test_the_list_holds_every_status_newest_first_without_vectors(
    index: ConversationIndex,
) -> None:
    a = await index.add_item(make("Oldest item.", at=10.0, vector=(0.5, 0.5)))
    b = await index.add_item(make("Newest item.", at=30.0))
    c = await index.add_item(make("Middle item.", at=20.0))
    await relate(index, replace=(a, b))
    listed = await index.all_items()
    assert [i.id for i in listed] == [b, c, a]
    assert [i.status for i in listed] == ["active", "active", "replaced"]
    assert listed[2].vector is None and listed[2].text == "Oldest item."


async def test_forgetting_everything_clears_the_items_and_a_rebuild_keeps_them(
    index: ConversationIndex,
) -> None:
    await add_turn(index, "s1", 1)
    await index.add_item(make("An item that stays through a rebuild."))
    assert await index.forget_all(mark=False) == 1
    assert await index.count_items() == 1
    assert await index.forget_all() == 0
    assert await index.count_items() == 0
    assert await index.forgotten_at() > 0
    # The numbers start again from where the table left off, and never repeat a
    # live item's.
    assert await index.add_item(make("A new item after forgetting.")) >= 1


async def test_a_turn_is_hidden_from_recall_only_while_an_item_cites_it(
    index: ConversationIndex,
) -> None:
    # A summary sits at a turn below 1: no item cites it, so it is always recalled.
    for session, turns in (("s1", (1, 2, 3, -1)), ("s2", (1, 2))):
        for turn in turns:
            await add_turn(index, session, turn)
    every = await uncited(index)
    assert len(every) == 6  # no item: every turn is recalled
    await index.add_item(make("One turn.", turns=(1,)))
    assert await uncited(index) == every - {("s1", 1)}
    # A replaced item still stands for its turns: its old fact does not come back
    # as a turn.
    old = await index.add_item(make("Old fact.", turns=(2,)))
    new = await index.add_item(make("New fact.", turns=(3,)))
    await relate(index, replace=(old, new))
    assert await uncited(index) == {("s1", -1), ("s2", 1), ("s2", 2)}
    # A mark says a turn was read, not that an item stands for it.
    await index.mark_encoded("s2", 2)
    assert await uncited(index) == {("s1", -1), ("s2", 1), ("s2", 2)}


async def test_a_mark_only_moves_forward(index: ConversationIndex) -> None:
    await index.mark_encoded("s1", 2)
    await index.mark_encoded("s1", 1)
    assert await index.encoded_turn("s1") == 2
    await index.mark_encoded("s1", 3)
    assert await index.encoded_turn("s1") == 3


async def test_forgetting_clears_the_marks_but_one_turn_and_a_rebuild_do_not(
    index: ConversationIndex,
) -> None:
    for session in ("s1", "s2"):
        for turn in (1, 2):
            await add_turn(index, session, turn)
        await index.mark_encoded(session, 1)
    await index.delete("s1", 2)
    assert await index.encoded_turn("s1") == 1
    await index.forget_all(mark=False)
    assert [await index.encoded_turn(s) for s in ("s1", "s2")] == [1, 1]
    await index.delete("s1")
    assert [await index.encoded_turn(s) for s in ("s1", "s2")] == [None, 1]
    await index.forget_all()
    assert await index.encoded_turn("s2") is None


async def test_the_encoded_share_is_the_turns_at_or_below_their_chats_mark(
    index: ConversationIndex,
) -> None:
    assert await index.encoded_share() == 0.0  # no turns
    for session in ("s1", "s2"):
        for turn in (1, 2, 3, 4):
            await add_turn(index, session, turn)
    # A chunked turn counts once, a turn below 1 (a summary, an exchange) not at all,
    # and a task's turns are indexed and never encoded, so they do not count either.
    await add_turn(index, "s3", 1, answer="word " * 1500)
    assert len(await index.turn_documents([("s3", 1)])) > 1
    for turn in (-2, -1, 0):
        await add_turn(index, "s1", turn)
    for turn in (1, 2, 3):
        await add_turn(index, "t1", turn, kind="task")
        await add_turn(index, "t2", turn, kind="front_bg")
    assert await index.encoded_share() == 0.0  # nothing read yet: 9 turns, no mark
    await index.mark_encoded("s1", 2)
    assert await index.encoded_share() == 2 / 9
    # A mark past the last turn, a mark on a chat with no turns and a mark on a task
    # change nothing but the turns they pass; no item cites any of these turns.
    await index.mark_encoded("s2", 9)
    await index.mark_encoded("nobody", 5)
    await index.mark_encoded("t1", 3)
    assert await index.count_items() == 0
    assert await index.encoded_share() == 6 / 9
    await index.mark_encoded("s3", 1)
    assert await index.encoded_share() == 7 / 9
    await index.forget_all()
    assert await index.encoded_share() == 0.0
    await add_turn(index, "s1", 1)
    assert await index.encoded_share() == 0.0  # the marks went with the rest


async def test_the_item_encoders_are_listed_once_each_whatever_the_status(
    index: ConversationIndex,
) -> None:
    assert await index.item_encoders() == set()
    old = await index.add_item(make("An old fact."))
    new = await index.add_item(make("A new fact."))
    await index.add_item(replace(make("One more."), encoder=""))
    await relate(index, replace=(old, new))
    assert await index.item_encoders() == {"1:stub", ""}


# ── Routing ──

PITTSBURGH = "The user said they live in Pittsburgh."
UTRECHT = "The user said they now live in Utrecht."


async def route(  # type: ignore[no-untyped-def]
    router: Router, text: str, judge: str | tuple[str, int] | None = None, **kw: object
):
    """Route a new item. With *judge*, what a judge says of its best match (``same`` or
    ``changes``), or of the item whose id it names (``("changes", 1)``); without, the
    rules alone."""
    item = make(text, **kw)  # type: ignore[arg-type]
    if judge is None:
        return await router.route(item)
    near = await router.shortlist(item)
    says, target = (judge, near[0].id) if isinstance(judge, str) else judge
    return await router.route(item, Judged(says, target, near))


async def test_a_new_fact_is_stored_as_new(index: ConversationIndex) -> None:
    router = Router(index)
    first = await route(router, PITTSBURGH, kind="profile")
    assert (first.action, first.item, first.target, first.reason) == (
        "NEW",
        1,
        None,
        "new",
    )
    second = await route(
        router,
        "The user adopted a grey cat named Pixel on 2023-04-02.",
        kind="episodic",
    )
    assert (second.action, second.item) == ("NEW", 2)
    assert [i.status for i in await index.fetch_items([1, 2])] == ["active", "active"]


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("The user said their card number is 4111 1111 1111 1111.", ""),
        ("The user said their password is private.", ""),
        ("The user said they have a card.", "Card 4111 1111 1111 1111"),
    ],
    ids=["value in the text", "words only", "title only"],
)
async def test_an_item_that_holds_a_secret_is_refused_and_stores_nothing(
    index: ConversationIndex, text: str, title: str
) -> None:
    router = Router(index)
    refused = await route(router, text, title=title)
    assert (refused.action, refused.item, refused.target) == (REFUSE, None, None)
    assert refused.reason == "holds a secret" and await index.count_items() == 0
    # Nothing was taken from the router's run: the next item is the first stored.
    clean = await route(router, PITTSBURGH, kind="profile")
    assert (clean.action, clean.item) == ("NEW", 1)


def test_the_router_is_the_only_way_into_the_items_table() -> None:
    """``add_item`` is the one statement that writes an item's title and text, and
    the router, which refuses an item that holds a secret first, is its only caller
    outside the tests; no statement changes a stored item's title, text or kind (an
    undo of a delete puts back the row the router stored)."""
    root = Path(hippo_route.__file__).parents[2]
    writers: set[str] = set()
    callers: set[str] = set()
    for folder in ("support", "melo"):
        for path in (root / folder).rglob("*.py"):
            code = path.read_text(encoding="utf-8")
            if re.search(r"(?:INSERT|REPLACE)\b[^;]*?\bINTO\s+items\b", code, re.I):
                writers.add(path.name)
            if ".add_item(" in code:
                callers.add(path.name)
            assert not re.search(r"UPDATE\s+items\s+SET\s+(?:title|text|kind)\b", code)
    assert writers == {"index.py"} and callers == {"hippo_route.py"}


async def test_an_update_stores_the_new_item_and_marks_the_old_one_replaced(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    old = await route(router, PITTSBURGH, kind="profile", at=1.0)
    new = await route(router, UTRECHT, "changes", kind="profile", at=2.0)
    assert (new.action, new.item, new.target) == ("UPDATE", 2, old.item)
    assert new.reason == "judged a change of entry 1"
    first, second = await index.fetch_items([1, 2])
    assert (first.status, first.replaced_by) == ("replaced", 2)
    assert (second.status, second.replaced_by) == ("active", None)
    # The replaced item is no longer compared with, the new one is.
    assert [i.id for i in await index.active_items("profile")] == [2]
    third = await route(
        router, "The user said they now live in Lisbon.", "changes", kind="profile"
    )
    assert (third.action, third.target) == ("UPDATE", 2)


@pytest.mark.parametrize("newer_first", [False, True], ids=["in order", "reversed"])
async def test_the_newer_fact_stays_current_whatever_order_they_are_routed_in(
    index: ConversationIndex, newer_first: bool
) -> None:
    router = Router(index)
    facts = [(PITTSBURGH, 1.0), (UTRECHT, 2.0)]
    for n, (text, at) in enumerate(reversed(facts) if newer_first else facts):
        await route(router, text, "changes" if n else None, kind="profile", at=at)
    # An older statement that arrives late is kept, but already replaced.
    by_text = {i.text: i for i in await index.fetch_items([1, 2])}
    older, newer = by_text[PITTSBURGH], by_text[UTRECHT]
    assert (older.status, older.replaced_by) == ("replaced", newer.id)
    assert (newer.status, newer.replaced_by) == ("active", None)
    assert [i.id for i in await index.active_items("profile")] == [newer.id]
    # The router keeps comparing with the newer one only.
    third = await route(
        router, "The user said they now live in Lisbon.", "changes", kind="profile"
    )
    assert (third.action, third.target) == ("UPDATE", newer.id)


async def test_an_update_and_its_item_are_stored_together_or_not_at_all(
    index: ConversationIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = Router(index)
    await route(router, PITTSBURGH, kind="profile", at=1.0)

    async def fail(*_: object) -> None:
        raise RuntimeError("the gateway stopped between the two writes")

    with monkeypatch.context() as patch:
        patch.setattr(ConversationIndex, "_replace", fail)
        with pytest.raises(RuntimeError):
            await route(router, UTRECHT, "changes", kind="profile", at=2.0)
    # Neither the new item nor the replacement was kept.
    assert await index.count_items() == 1
    (old,) = await index.fetch_items([1])
    assert (old.status, old.replaced_by) == ("active", None)
    # A retry is not mistaken for a duplicate: it completes the update.
    again = await route(router, UTRECHT, "changes", kind="profile", at=2.0)
    assert (again.action, again.item, again.target) == ("UPDATE", 2, 1)
    assert [i.status for i in await index.fetch_items([1, 2])] == ["replaced", "active"]


async def test_a_related_item_is_stored_and_linked_both_ways(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(
        router,
        "The user booked a hotel in the Alps for the hiking trip on 2023-06-01.",
    )
    related = await route(
        router, "The user went hiking in the Alps with Maria on 2023-05-20."
    )
    assert (related.action, related.item, related.target) == ("RELATED", 2, 1)
    assert related.reason == "overlap 0.50; related to entry 1"
    first, second = await index.fetch_items([1, 2])
    assert (first.related, second.related) == ((2,), (1,))
    assert (first.status, second.status) == ("active", "active")


async def test_a_conflict_stores_the_new_item_and_marks_both_but_both_stay(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(router, "The user said they eat spicy food every day.", kind="profile")
    conflict = await route(
        router, "The user said they never eat spicy food every day.", kind="profile"
    )
    assert (conflict.action, conflict.item, conflict.target) == ("CONFLICT", 2, 1)
    assert conflict.reason == "opposite polarity to entry 1"
    first, second = await index.fetch_items([1, 2])
    assert (first.status, first.conflicts) == ("conflict", (2,))
    assert (second.status, second.conflicts) == ("conflict", (1,))
    assert await index.count_items() == 2
    # Neither is active any more, so a third statement meets neither.
    assert (await index.active_items()) == []


async def test_a_duplicate_is_ignored_and_not_stored(index: ConversationIndex) -> None:
    router = Router(index)
    text = "melo recommended the book Dune by Frank Herbert for the trip."
    await route(router, text, kind="semantic")
    ignored = await route(router, text, kind="semantic", session="s2", turns=(4,))
    assert (ignored.action, ignored.item, ignored.target) == ("IGNORE", None, 1)
    assert ignored.reason == "exact duplicate"
    assert await index.count_items() == 1
    # The item it repeats now cites the repeat's turn too; once, however often.
    await route(router, text, kind="semantic", session="s2", turns=(4,))
    (kept,) = await index.fetch_items([1])
    assert kept.sources == (("s1", 1), ("s2", 4))
    short = await route(router, "Hi", kind="profile", session="s3", turns=(7,))
    assert (short.action, short.item, short.target) == ("IGNORE", None, None)
    assert short.reason == "short fact" and await index.count_items() == 1
    # A short fact repeats no item: no item's sources change.
    assert [i.sources for i in await index.fetch_items([1])] == [kept.sources]
    assert ignored.to_json() == {
        "action": "IGNORE",
        "item": None,
        "target": 1,
        "reason": "exact duplicate",
    }


@pytest.mark.parametrize("new_run", [False, True], ids=["same run", "later run"])
async def test_a_fact_repeated_later_is_not_older_than_an_update_dated_between(
    index: ConversationIndex, new_run: bool
) -> None:
    monday, tuesday, wednesday = 1000.0, 2000.0, 3000.0
    router = Router(index)
    await route(router, PITTSBURGH, kind="profile", at=monday)
    repeat = await route(
        router, PITTSBURGH, kind="profile", at=wednesday, session="s2", turns=(4,)
    )
    assert (repeat.action, repeat.target) == ("IGNORE", 1)
    (kept,) = await index.fetch_items([1])
    assert kept.created_at == wednesday  # the item's time is its newest source's
    # A change dated Tuesday is encoded last: it is older than Wednesday's repeat of
    # the first fact, so that fact stays current.
    late = await route(
        Router(index) if new_run else router,
        UTRECHT,
        "changes",
        kind="profile",
        at=tuesday,
        session="s3",
        turns=(2,),
    )
    assert (late.action, late.item, late.target) == ("UPDATE", 2, 1)
    first, second = await index.fetch_items([1, 2])
    assert (first.status, first.replaced_by) == ("active", None)
    assert (second.status, second.replaced_by) == ("replaced", 1)


async def test_an_episode_never_replaces_another_episode(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(router, "The user went hiking in the Alps with Maria on 2023-05-20.")
    later = await route(
        router, "The user went hiking in the Alps with Maria on 2023-05-21."
    )
    assert later.action == "RELATED"  # not UPDATE: each episode is its own subject


async def test_items_of_every_kind_are_compared_and_a_vector_can_relate_them(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(
        router,
        "The user adopted a grey cat named Pixel on 2023-04-02.",
        vector=(0.8, 0.6),
    )
    far = await route(
        router,
        "The user went hiking in the Alps with Maria on 2023-05-20.",
        vector=(0.0, 1.0),
    )
    assert far.action == "NEW"
    near = await route(
        router,
        "The user paddled a kayak down the Allegheny river last summer.",
        vector=(1.0, 0.0),
    )
    assert (near.action, near.target) == ("RELATED", 1)
    assert "overlap" in near.reason
    other_kind = await route(
        router,
        "The user said they adopted a grey cat named Pixel on 2023-04-02.",
        kind="semantic",
        vector=(0.8, 0.6),
    )
    # Items are compared by content, whatever their kind: the same fact told as a
    # semantic item is linked to the episodic one, and replaces nothing on its own.
    assert (other_kind.action, other_kind.target) == ("RELATED", 1)


async def test_a_vector_of_another_model_is_not_compared(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(router, "The user owns a grey cat named Pixel.", vector=(1.0, 0.0))
    other = make("Quarterly invoices are paid on Fridays.", vector=(1.0, 0.0))
    other = MemoryItem(**{**other.__dict__, "embed_model": "another-model"})
    result = await Router(index).route(other)
    assert result.action == "NEW"  # same direction, but not the same space


async def test_the_pool_gives_the_decision_the_policy_gives_for_all_of_them(
    index: ConversationIndex,
) -> None:
    """With many active items only the best-sorted few reach the policy, which acts
    on the best-sorted one: its decision is the same as for all of them."""
    router = Router(index)
    cities = [
        "Utrecht",
        "Lisbon",
        "Oslo",
        "Porto",
        "Bergen",
        "Malmo",
        "Graz",
        "Turin",
        "Seville",
        "Lyon",
        "Zurich",
        "Ghent",
    ]
    for n, city in enumerate(cities):  # a distinct subject each: a profile has one
        await route(
            router,
            f"The user said their pet {city}fish is called Number{n} since 2020.",
            kind="semantic",
            at=float(n),
        )
    assert len(await index.active_items()) > POOL
    new = make(
        "The user said their pet Bergenfish is called Number4 since 2020 and loves it.",
        kind="semantic",
        at=50.0,
    )
    everything = await index.active_items()
    expected = decide(
        new.text,
        kind="semantic",
        provenance={"candidate_id": "new", "item_index": 99},
        active=[_Entry(i.id, i.text, i.kind, i.created_at) for i in everything],
        certain_only=True,
    )
    result = await router.route(new)
    assert result.action == expected.action
    assert result.target == expected.target_id == 5


class _Entry:
    """An active item as the policy reads it, with no vector and an item-local subject."""

    def __init__(self, id: int, text: str, kind: str, created_at: float) -> None:
        self.id, self.text, self.kind, self.created_at = id, text, kind, created_at
        self.vector = None
        self.provenance = {"candidate_id": f"m{id}", "item_index": 0}


# ── What the rules settle and what a judge decides ──

CHOIR = "The user's choir rehearsal is on Mondays at 19:00."
CHOIR_AGAIN = "The user said their choir rehearsal is on Mondays at 19:00."
CHOIR_MOVED = "The user said the choir rehearsal moved to Wednesdays."


async def test_a_change_written_under_another_kind_replaces_the_old_fact(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    old = await route(router, CHOIR, kind="profile", at=1.0)
    new = await route(router, CHOIR_MOVED, "changes", kind="semantic", at=2.0)
    assert (new.action, new.item, new.target) == ("UPDATE", 2, old.item)
    first, second = await index.fetch_items([1, 2])
    assert (first.status, first.replaced_by) == ("replaced", 2)
    assert (second.status, second.kind) == ("active", "semantic")
    # A third chat that repeats the change, under a third kind, joins the new item.
    third = await route(
        router,
        "The user said the choir now rehearses on Wednesdays.",
        "same",
        kind="episodic",
        at=3.0,
        session="s2",
        turns=(4,),
    )
    assert (third.action, third.item, third.target) == ("IGNORE", None, 2)
    assert [i.id for i in await index.active_items()] == [2]
    (kept,) = await index.fetch_items([2])
    assert kept.sources == (("s1", 1), ("s2", 4))


async def test_a_reworded_repeat_joins_the_old_item_whatever_its_kind(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(router, CHOIR, kind="profile", at=1.0, turns=(1,))
    again = await route(
        router, CHOIR_AGAIN, "same", kind="semantic", at=5.0, session="s2", turns=(4,)
    )
    assert (again.action, again.item, again.target) == ("IGNORE", None, 1)
    (kept,) = await index.fetch_items([1])
    assert kept.sources == (("s1", 1), ("s2", 4)) and kept.created_at == 5.0
    assert await index.count_items() == 1
    # With no judge, the same repeat is stored and linked, not dropped.
    plain = await route(router, CHOIR_AGAIN, kind="semantic", at=6.0, turns=(5,))
    assert (plain.action, plain.item, plain.target) == ("RELATED", 2, 1)


async def test_two_facts_that_share_a_verb_are_linked_not_replaced(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(router, "The user lives with three rabbits.", kind="profile", at=1.0)
    other = await route(
        router, "The user lives in a small harbour town.", kind="profile", at=2.0
    )
    assert (other.action, other.target) == ("RELATED", 1)
    first, second = await index.fetch_items([1, 2])
    assert (first.status, second.status) == ("active", "active")
    assert (first.related, second.related) == ((2,), (1,))


async def test_an_item_with_every_word_of_an_old_one_and_a_detail_is_not_dropped(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(router, "The user plays tennis on Saturday mornings.", at=1.0)
    more = await route(
        router,
        "The user plays tennis on Saturday mornings at the club with a friend.",
        at=2.0,
    )
    assert (more.action, more.item, more.target) == ("RELATED", 2, 1)
    assert await index.count_items() == 2


async def test_the_shortlist_is_by_content_across_kinds_and_not_by_subject(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    await route(
        router,
        "The user bought the old stone cottage by the river bend in 2019.",
        kind="episodic",
        at=1.0,
    )
    for n, text in enumerate(
        (
            "The user lives in a rented cottage near the harbour.",
            "The user lives in a rented flat by the river.",
            "The user lives in a rented house near the river.",
            "The user lives in a rented cottage by the sea.",
        )
    ):  # four profile items that share the verb, and so the subject, of the new one
        await route(router, text, kind="profile", at=2.0 + n)
    new = make(
        "The user lives in the old stone cottage by the river bend.",
        kind="profile",
        at=10.0,
    )
    near = await router.shortlist(new)
    assert len(near) == SHORTLIST == 3
    # The best match by words comes first, though it is of another kind.
    assert near[0].id == 1 and near[0].kind == "episodic"
    assert [e.kind for e in near[1:]] == ["profile", "profile"]


async def test_nothing_is_shortlisted_when_the_rules_settle_the_item_or_none_is_close(
    index: ConversationIndex,
) -> None:
    router = Router(index)
    spicy = "The user said they eat spicy food every day."
    await route(router, spicy, kind="profile")
    for text in (
        "Hi",  # a short fact
        spicy,  # the same text
        "The user said they never eat spicy food every day.",  # the opposite
        "The user said their card number is 4111 1111 1111 1111.",  # a secret
        "The user adopted a grey cat named Pixel on 2023-04-02.",  # nothing close
    ):
        assert await router.shortlist(make(text, kind="profile")) == (), text
    near = await router.shortlist(
        make("The user said they eat spicy food on most days.", kind="profile")
    )
    assert [e.id for e in near] == [1]


@pytest.mark.parametrize("changed", ["deleted", "replaced", "sources", "time"])
async def test_a_verdict_is_dropped_when_an_item_it_rested_on_changed(
    index: ConversationIndex, changed: str
) -> None:
    router = Router(index)
    await route(router, PITTSBURGH, kind="profile", at=1.0)
    await route(router, "The user adopted a grey cat named Pixel on 2023-04-02.")
    new = make(UTRECHT, kind="profile", at=2.0)
    near = await router.shortlist(new)
    judged = Judged("changes", near[0].id, near)
    assert await router.confirm(judged) is judged  # nothing changed yet
    assert await router.confirm(None) is None
    if changed == "deleted":
        await index.delete_item(1)
    elif changed == "replaced":
        await relate(index, replace=(1, 2))
    elif changed == "sources":
        await index.add_sources(1, [("s9", 3)])
    else:
        await index.add_sources(1, [], 99.0)
    assert await router.confirm(judged) is None
    # Routed by the rules alone: stored, never lost.
    stored = await router.route(new)
    assert stored.item is not None and stored.action in ("NEW", "RELATED")


@pytest.mark.parametrize("says", ["same", "changes"])
async def test_a_repeat_whose_target_is_gone_is_stored_as_a_new_item(
    index: ConversationIndex, says: str
) -> None:
    router = Router(index)
    await route(router, PITTSBURGH, kind="profile", at=1.0)
    await index.delete_item(1)  # behind the router's back
    again = await route(
        router, PITTSBURGH, kind="profile", at=2.0, session="s2", turns=(4,)
    )
    assert (again.action, again.item) == ("NEW", 2)  # not an IGNORE of a gone item
    # A judge's answer about an item that has since gone is not acted on either.
    item = make(UTRECHT, kind="profile", at=3.0, session="s3", turns=(7,))
    near = await router.shortlist(item)
    await index.delete_item(2)
    stored = await router.route(item, Judged(says, near[0].id, near))
    assert (stored.action, stored.item) == ("NEW", 3)
    assert [i.id for i in await index.active_items()] == [3]


async def test_a_repeat_of_an_item_that_became_active_again_is_found_after_a_refresh(
    index: ConversationIndex,
) -> None:
    for session in ("s1", "s2"):
        await add_turn(index, session, 1)
    router = Router(index)
    await route(router, PITTSBURGH, kind="profile", at=1.0)  # item 1, chat s1
    await route(  # item 2, chat s2: replaces item 1
        router, UTRECHT, "changes", kind="profile", at=2.0, session="s2"
    )
    assert [i.id for i in await index.active_items()] == [2]
    repeat = make(PITTSBURGH, kind="profile", at=3.0, session="s3", turns=(7,))
    near = await router.shortlist(repeat)
    assert [e.id for e in near] == [2]
    # Chat s2 is forgotten while the repeat is judged: item 1, which item 2 replaced,
    # is current again, though the router never loaded it as such.
    await index.delete("s2")
    assert [i.id for i in await index.active_items()] == [1]
    judged = await router.confirm(Judged("changes", 2, near))
    assert judged is None
    stored = await router.route(repeat, judged)
    assert (stored.action, stored.item, stored.target) == ("IGNORE", None, 1)
    assert await index.count_items() == 1  # not stored again beside item 1
    (kept,) = await index.fetch_items([1])
    assert kept.sources == (("s1", 1), ("s3", 7))


@pytest.mark.parametrize(
    "same_subject", [1, POOL + 1], ids=["few", "more than the pool"]
)
async def test_an_identical_text_is_found_whatever_its_kind_and_the_items_that_share_a_subject(
    index: ConversationIndex, same_subject: int
) -> None:
    router = Router(index)
    text = "The user lives in the old stone cottage by the river bend."
    await route(router, text, kind="semantic", at=1.0)  # item 1, another kind
    places = ("flat", "house", "barn", "loft", "villa", "hut", "lodge", "mill", "farm")
    for n, place in enumerate(places[:same_subject]):
        await route(
            router, f"The user lives in a rented {place} number {n}.", kind="profile"
        )
    assert len(await index.active_items()) == same_subject + 1
    again = await route(router, text, kind="profile", at=50.0, session="s2", turns=(4,))
    # The profile items share its verb and its subject, and rank before the identical
    # item of another kind; the identical text is looked up in the whole map first.
    assert (again.action, again.item, again.target) == ("IGNORE", None, 1)
    assert again.reason == "exact duplicate"
    assert await index.count_items() == same_subject + 1
    (kept,) = await index.fetch_items([1])
    assert kept.sources == (("s1", 1), ("s2", 4))
    # With the shortlist too: nothing is asked of a judge for an identical text.
    assert await router.shortlist(make(text, kind="profile", at=60.0)) == ()


@pytest.mark.parametrize(
    "repeat_first", [True, False], ids=["repeat first", "change first"]
)
async def test_a_shorter_repeat_not_called_the_same_leaves_the_old_time_alone(
    index: ConversationIndex, repeat_first: bool
) -> None:
    """Folding a shorter repeat into the old item would move its time to the repeat's,
    and a real change dated between them, encoded later, would then be rejected as
    older. Answered ``other``, the repeat is stored and linked, and the change still
    replaces the old item, whichever of the two is encoded first."""
    router = Router(index)
    full = "The user's choir rehearsal is on Mondays at 19:00 in the church hall."
    short = "The user has a choir rehearsal on Mondays."
    first = await route(router, full, kind="profile", at=1000.0)

    async def repeat() -> None:
        route_ = await route(router, short, kind="profile", at=3000.0, session="s2")
        assert route_.action == "RELATED"
        (kept,) = await index.fetch_items([first.item or 0])
        assert kept.created_at == 1000.0 and len(kept.sources) == 1

    async def change() -> None:
        moved = await route(
            router,
            CHOIR_MOVED,
            ("changes", first.item or 0),
            kind="semantic",
            at=2000.0,
            session="s3",
        )
        assert (moved.action, moved.target) == ("UPDATE", first.item)

    for step in (repeat, change) if repeat_first else (change, repeat):
        await step()
    (old,) = await index.fetch_items([first.item or 0])
    assert (old.status, old.replaced_by) == ("replaced", 3 if repeat_first else 2)


# ── The judge ──


async def seeded(index: ConversationIndex) -> Router:
    router = Router(index)
    await route(router, PITTSBURGH, kind="profile", at=1.0)
    return router


async def judge_once(
    router: Router,
    provider: StubEncoder,
    typed: tuple[str, ...] = ("I live in Utrecht now.",),
    *,
    text: str = UTRECHT,
    cache: Cache | None = None,
    counts: hippo_judge.Counts | None = None,
    stats: hippo_encode.Stats | None = None,
) -> Judged | None:
    return await hippo_judge.judge(
        router,
        provider,
        None,
        make(text, kind="profile", at=2.0),
        typed,
        cache=cache or Cache(""),
        stats=stats or hippo_encode.Stats(),
        counts=counts or hippo_judge.Counts(),
    )


async def test_a_clear_changes_the_typed_words_support_is_acted_on_and_counted(
    index: ConversationIndex,
) -> None:
    router = await seeded(index)
    provider = StubEncoder(lambda _: verdict("changes", 1))
    counts, stats = hippo_judge.Counts(), hippo_encode.Stats()
    got = await judge_once(router, provider, counts=counts, stats=stats)
    assert got is not None and (got.verdict, got.target) == ("changes", 1)
    assert counts.to_json() == {
        "judged": 1,
        "same": 0,
        "changes": 1,
        "unsupported": 0,
        "other": 0,
        "failed": 0,
        "fallbacks": 0,
    }
    assert stats.calls == 1
    # One request through the quiet path: the texts and the typed words, as data.
    (prompt,) = provider.prompts
    request = json.loads(prompt)
    assert set(request) == {"new", "old", "typed"}
    assert request["new"]["text"] == UTRECHT
    assert [(o["item"], o["text"]) for o in request["old"]] == [(1, PITTSBURGH)]
    assert request["typed"] == ["I live in Utrecht now."]
    assert provider.systems == [hippo_judge.SYSTEM_PROMPT]
    assert provider.bodies[0]["temperature"] == 0


async def test_a_changes_the_typed_words_do_not_support_only_counts(
    index: ConversationIndex,
) -> None:
    router = await seeded(index)
    counts = hippo_judge.Counts()
    provider = StubEncoder(lambda _: verdict("changes", 1))
    got = await judge_once(router, provider, ("Where is that again?",), counts=counts)
    assert got is None
    assert (counts.judged, counts.changes, counts.unsupported) == (1, 1, 1)
    assert counts.failed == counts.same == counts.other == 0


@pytest.mark.parametrize("answer", ["no judge", "unsupported", "other", "failure"])
async def test_a_different_value_for_the_same_subject_is_linked_never_conflicted(
    index: ConversationIndex, answer: str
) -> None:
    """Only an opposite polarity is a conflict without a model: a different value may
    be a change, and the old fact, which may well be true, is not marked for it."""
    router = Router(index)
    await route(router, "Your timezone is Lisbon.", kind="profile", at=1.0)
    new = make("Your timezone is Madrid.", kind="profile", at=2.0)
    judged = None
    if answer != "no judge":
        provider = StubEncoder(
            {
                "unsupported": lambda _: verdict("changes", 1),
                "other": lambda _: verdict(),
                "failure": lambda _: "RAISE",
            }[answer]
        )
        judged = await hippo_judge.judge(
            router,
            provider,
            None,
            new,
            ("Where is that again?",),
            cache=Cache(""),
            stats=hippo_encode.Stats(),
            counts=hippo_judge.Counts(),
        )
        assert judged is None
    result = await router.route(new, judged)
    assert (result.action, result.item, result.target) == ("RELATED", 2, 1)
    first, second = await index.fetch_items([1, 2])
    assert (first.status, second.status) == ("active", "active")
    assert (first.related, second.related) == ((2,), (1,))
    assert [i.id for i in await index.active_items()] == [1, 2]


async def test_a_same_and_an_other_are_counted_and_only_the_same_gives_a_verdict(
    index: ConversationIndex,
) -> None:
    router = await seeded(index)
    counts = hippo_judge.Counts()
    same = await judge_once(
        router, StubEncoder(lambda _: verdict("same")), counts=counts
    )
    assert same is not None and (same.verdict, same.target) == ("same", 1)
    other = await judge_once(router, StubEncoder(lambda _: verdict()), counts=counts)
    assert other is None
    assert (counts.judged, counts.same, counts.other, counts.failed) == (2, 1, 1, 0)


@pytest.mark.parametrize(
    "reply",
    [
        "RAISE",
        "no idea",
        "```json\n" + verdict("same") + "\n```",
        '{"verdict": "same", "item": 4}',
        '{"verdict": "same", "item": 0}',
        '{"verdict": "same", "item": true}',
        '{"verdict": "same", "item": null}',
        '{"verdict": "same", "item": 1, "why": "alike"}',
        '{"verdict": "same"}',
        '{"verdict": "maybe", "item": 1}',
        '{"verdict": "other", "item": 9}',
        "[1]",
    ],
    ids=[
        "error",
        "not json",
        "code fence",
        "item not shown",
        "item zero",
        "item a boolean",
        "same without an item",
        "extra key",
        "missing key",
        "unknown verdict",
        "other with an item not shown",
        "not an object",
    ],
)
async def test_any_failure_counts_as_other_and_acts_on_nothing(
    index: ConversationIndex, reply: str
) -> None:
    router = await seeded(index)
    counts = hippo_judge.Counts()
    got = await judge_once(router, StubEncoder(reply), counts=counts)
    assert got is None
    assert (counts.judged, counts.failed) == (1, 1)
    assert counts.same == counts.changes == counts.other == counts.unsupported == 0


async def test_a_request_that_never_answers_times_out_as_other(
    index: ConversationIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(note_judge, "JUDGE_TIMEOUT_S", 0.05)
    router = await seeded(index)
    counts = hippo_judge.Counts()
    held = Held()  # a judge request waits, and nothing releases it
    assert await judge_once(router, held, counts=counts) is None
    assert held.asked.is_set() and (counts.judged, counts.failed) == (1, 1)
    assert held.prompts == []  # one attempt, and it never answered


async def test_a_reply_is_cached_by_the_judge_version_and_a_second_ask_makes_no_request(
    index: ConversationIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = await seeded(index)
    cache = Cache(str(tmp_path / "cache.jsonl"))
    provider = StubEncoder(lambda _: verdict("same"))
    stats = hippo_encode.Stats()
    for _ in range(2):
        got = await judge_once(router, provider, cache=cache, stats=stats)
        assert got is not None and got.verdict == "same"
    assert len(provider.prompts) == 1 and (stats.calls, stats.cache_hits) == (1, 1)
    # The file holds a key and a verdict, no text of an item or of the person.
    kept = (tmp_path / "cache.jsonl").read_text(encoding="utf-8")
    assert "Utrecht" not in kept and "Pittsburgh" not in kept and "same" in kept
    # A new judge version is a new key; the encoder's own key does not move.
    window_key = hippo_encode.window_key("stub-encoder", "a window")
    monkeypatch.setattr(hippo_judge, "JUDGE_VERSION", "judge-next")
    assert hippo_encode.window_key("stub-encoder", "a window") == window_key
    await judge_once(router, provider, cache=cache, stats=stats)
    assert len(provider.prompts) == 2
    # A reply kept under the previous judge version is not reused.
    stale = Cache("")
    stale.put(
        hippo_encode.window_key("stub-encoder", provider.prompts[0], "judge-1"),
        json.loads(verdict("same")),
    )
    fresh = StubEncoder(lambda _: verdict())
    monkeypatch.undo()
    assert hippo_judge.JUDGE_VERSION != "judge-1"  # the real constant, not the patch
    assert await judge_once(router, fresh, cache=stale) is None
    assert len(fresh.prompts) == 1


async def test_no_request_is_made_for_an_item_with_no_close_item(
    index: ConversationIndex,
) -> None:
    router = await seeded(index)
    provider, counts = StubEncoder(lambda _: verdict("same")), hippo_judge.Counts()
    got = await judge_once(
        router,
        provider,
        text="The user adopted a grey cat named Pixel on 2023-04-02.",
        counts=counts,
    )
    assert got is None and provider.prompts == [] and counts.judged == 0


async def test_the_request_holds_only_the_typed_words_cleaned_of_secrets_and_bounded(
    index: ConversationIndex,
) -> None:
    router = await seeded(index)
    provider = StubEncoder(lambda _: verdict())
    typed = (
        "I live in Utrecht now. My card is 4242 4242 4242 4242. "
        "My password is hunter2.",
        "x" * (note_judge.MAX_TYPED_CHARS * 2),
    )
    await judge_once(router, provider, typed)
    (prompt,) = provider.prompts
    assert "4242" not in prompt and "hunter2" not in prompt
    sent = json.loads(prompt)["typed"]
    assert sent[0].startswith("I live in Utrecht now")
    assert sum(len(text) for text in sent) <= note_judge.MAX_TYPED_CHARS


@pytest.mark.parametrize(
    ("new", "old", "said", "wanted"),
    [
        (
            "The user now lives in Utrecht.",
            PITTSBURGH,
            ["I live in Utrecht now."],
            True,
        ),
        (
            "The user pays 120 euros a month.",
            "The user pays 100 euros a month.",
            ["it is 120 now"],
            True,
        ),
        ("用户现在住在上海。", "用户住在北京。", ["我搬到上海了"], True),
        ("The user now lives in Utrecht.", PITTSBURGH, ["I moved house."], False),
        ("The user now lives in Utrecht.", PITTSBURGH, [], False),
        (  # a word the old item has is not new
            "The user said they live in Utrecht.",
            "The user said they live in Utrecht, near the canal.",
            ["I live in Utrecht"],
            False,
        ),
        (  # a plain lowercase word is a value
            "The user now drinks green tea.",
            "The user drinks black coffee.",
            ["I drink green tea now"],
            True,
        ),
        (  # a number word is a value
            "The user has five cousins.",
            "The user has two cousins.",
            ["actually five"],
            True,
        ),
        (  # the new words are melo's, not the person's
            "The user now drinks green tea.",
            "The user drinks black coffee.",
            ["what do I drink?"],
            False,
        ),
        (  # "now" says something changed, not what the value is
            "The user now lives in Utrecht.",
            "The user lives in Utrecht.",
            ["since now"],
            False,
        ),
        (  # a weekday is one, however it is written
            "The user's class is now on Fridays.",
            "The user's class is on Wednesdays.",
            ["it moved to friday"],
            True,
        ),
        (
            "The user now has class on Thursdays.",
            "The user has class on Mondays.",
            ["moved it to Thurs"],
            True,
        ),
        ("周五有课。", "周三有课。", ["星期五吧"], True),
        ("周五有课。", "周三有课。", ["礼拜五吧"], True),
        ("周日有课。", "周三有课。", ["星期天吧"], True),
        (
            "The user's class is now on Saturdays.",
            "The user's class is on Wednesdays.",
            ["it moved to friday"],
            False,
        ),
        (  # a short form that is also a word counts only when capitalised
            "The user's class is now on Saturdays.",
            "The user's class is on Wednesdays.",
            ["I sat down for a while"],
            False,
        ),
        ("周五有课。", "周三有课。", ["星期四吧"], False),
        (  # the old item has the day already, written another way
            "The user's class is now on Fridays at 18:00.",
            "The user's class is on Friday.",
            ["friday"],
            False,
        ),
        ("课改到周五了。", "课在周三。", ["你改到哪里了"], False),
        ("课改到周五了。", "课在周三。", ["改到星期四了"], False),
        ("名字改成了小宇。", "名字是阿凯。", ["改成什么了"], False),
        ("会议调到周五。", "会议在周三。", ["调到哪里去了"], False),
        ("会议换到周五。", "会议在周三。", ["换到别处了"], False),
        (  # a change word is no value, in English either
            "Updated: the user lives in Utrecht.",
            "The user lives in Utrecht.",
            ["updated"],
            False,
        ),
        ("课改到周五了。", "课在周三。", ["我改到星期五了"], True),
        ("会议改到15点了。", "会议在10点。", ["改到15点"], True),
        (  # a plain word typed only in a question names a topic, not a value
            "The user now takes the ferry to work.",
            "The user takes the bus to work.",
            ["do you know when the ferry leaves?"],
            False,
        ),
        (  # a fragment of "that's" is no value
            "The user's scarf is wool.",
            "The user has a cotton scarf.",
            ["that's my scarf."],
            False,
        ),
        (  # one CJK character, from a question
            "用户养了一只猫。",
            "用户养了一只狗。",
            ["我的猫是什么颜色？"],
            False,
        ),
        (  # a modal in a question
            "The user would host it in the garden.",
            "The user declined to host it.",
            ["would that work? could they come?"],
            False,
        ),
        (  # a word that carries no value, in a statement
            "The user will bring the cake.",
            "The user declined to bring the cake.",
            ["I will see."],
            False,
        ),
        (
            "The user is not coming.",
            "The user is coming.",
            ["I am not sure."],
            False,
        ),
        (  # the plain word is in a statement, after a question
            "The user now drinks oat milk.",
            "The user drinks cow milk.",
            ["which one? We switched to oat milk."],
            True,
        ),
        (  # a number word in a statement
            "The user keeps seven hens.",
            "The user keeps three hens.",
            ["we have seven now."],
            True,
        ),
        (  # a distinctive value counts in a question too
            "The user's dance class is now on Thursdays.",
            "The user's dance class is on Tuesdays.",
            ["can you move it to Thursday?"],
            True,
        ),
        (
            "The user's call is at 14:30.",
            "The user's call is at 9:00.",
            ["is 14:30 ok?"],
            True,
        ),
        ("用户喜欢喝豆浆。", "用户喜欢喝咖啡。", ["我最近改喝豆浆了。"], True),
        ("用户喜欢喝豆浆。", "用户喜欢喝咖啡。", ["豆浆还有吗？"], True),
    ],
    ids=[
        "name",
        "number",
        "cjk",
        "not typed",
        "nothing typed",
        "old has it",
        "plain word",
        "number word",
        "not typed, only asked",
        "change word now",
        "weekday plural",
        "weekday short form",
        "weekday 星期",
        "weekday 礼拜",
        "weekday 星期天",
        "another weekday",
        "a word that is not a day",
        "another Chinese weekday",
        "old has the weekday",
        "change word 改到",
        "change word 改到 then a weekday",
        "change word 改成",
        "change word 调到",
        "change word 换到",
        "capitalised change word",
        "change word and a Chinese weekday",
        "change word and a number",
        "plain word only in a question",
        "fragment of a contraction",
        "one CJK character in a question",
        "modal in a question",
        "no-value word will",
        "no-value word not",
        "plain word in a statement after a question",
        "number word in a statement",
        "distinctive word in a question",
        "number in a question",
        "Chinese value in a statement",
        "Chinese value in a question",
    ],
)
def test_a_change_needs_a_new_value_word_in_the_typed_words(
    new: str, old: str, said: list[str], wanted: bool
) -> None:
    assert hippo_judge.supported(new, old, said) is wanted
