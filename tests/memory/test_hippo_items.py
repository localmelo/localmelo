"""Memory items in the conversation index, and Hippo's routing of a new item.

The ``items`` table is created with the others and holds what a chat model encoded
from chats; deleting a chat, or one turn an item names, takes the item with it, and
``forget_all`` clears them. A turn is hidden from recall only while a stored item
cites it. A new item is routed by the pure policy against the active items of its
kind: NEW, UPDATE (the target is marked replaced, unless it is the newer fact),
RELATED (linked both ways), CONFLICT (both marked, both stay) or IGNORE (not
stored), and is stored in one transaction with its update or link. Synthetic items,
no model.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.melo.memory.hippo.policy import decide
from localmelo.support.memory import hippo_route
from localmelo.support.memory.hippo_route import POOL, REFUSE, Router
from localmelo.support.memory.index import (
    ConversationIndex,
    Document,
    MemoryItem,
    document_chunks,
)
from localmelo.tests.memory.hippo_helpers import relate

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


async def add_turn(index: ConversationIndex, session: str, turn: int) -> None:
    await index.add_many(
        [
            Document(
                session_id=session,
                turn=turn,
                surface="gateway",
                kind="chat",
                title=f"Chat {session}",
                text=text,
                created_at=1000.0 + turn,
                chunk=n,
            )
            for n, text in enumerate(document_chunks(f"Question {turn}?", "Answer."))
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
        await route(router, text, kind="profile", at=at, turns=(turn,))
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


# ── Routing ──

PITTSBURGH = "The user said they live in Pittsburgh."
UTRECHT = "The user said they now live in Utrecht."


async def route(router: Router, text: str, **kw: object):  # type: ignore[no-untyped-def]
    return await router.route(make(text, **kw))  # type: ignore[arg-type]


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
    new = await route(router, UTRECHT, kind="profile", at=2.0)
    assert (new.action, new.item, new.target) == ("UPDATE", 2, old.item)
    assert new.reason == "same subject; update of entry 1"
    first, second = await index.fetch_items([1, 2])
    assert (first.status, first.replaced_by) == ("replaced", 2)
    assert (second.status, second.replaced_by) == ("active", None)
    # The replaced item is no longer compared with, the new one is.
    assert [i.id for i in await index.active_items("profile")] == [2]
    third = await route(
        router, "The user said they now live in Lisbon.", kind="profile"
    )
    assert (third.action, third.target) == ("UPDATE", 2)


@pytest.mark.parametrize("newer_first", [False, True], ids=["in order", "reversed"])
async def test_the_newer_fact_stays_current_whatever_order_they_are_routed_in(
    index: ConversationIndex, newer_first: bool
) -> None:
    router = Router(index)
    facts = [(PITTSBURGH, 1.0), (UTRECHT, 2.0)]
    for text, at in reversed(facts) if newer_first else facts:
        await route(router, text, kind="profile", at=at)
    # An older statement that arrives late is kept, but already replaced.
    by_text = {i.text: i for i in await index.fetch_items([1, 2])}
    older, newer = by_text[PITTSBURGH], by_text[UTRECHT]
    assert (older.status, older.replaced_by) == ("replaced", newer.id)
    assert (newer.status, newer.replaced_by) == ("active", None)
    assert [i.id for i in await index.active_items("profile")] == [newer.id]
    # The router keeps comparing with the newer one only.
    third = await route(
        router, "The user said they now live in Lisbon.", kind="profile"
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
            await route(router, UTRECHT, kind="profile", at=2.0)
    # Neither the new item nor the replacement was kept.
    assert await index.count_items() == 1
    (old,) = await index.fetch_items([1])
    assert (old.status, old.replaced_by) == ("active", None)
    # A retry is not mistaken for a duplicate: it completes the update.
    again = await route(router, UTRECHT, kind="profile", at=2.0)
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


async def test_items_are_compared_by_kind_and_a_vector_can_relate_them(
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
    assert other_kind.action == "NEW"  # an episodic item is not a semantic one's peer


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
    everything = await index.active_items("semantic")
    expected = decide(
        new.text,
        kind="semantic",
        provenance={"candidate_id": "new", "item_index": 99},
        active=[_Entry(i.id, i.text, i.created_at) for i in everything],
    )
    result = await router.route(new)
    assert result.action == expected.action
    assert result.target == expected.target_id == 5


class _Entry:
    """An active item as the policy reads it, with no vector and an item-local subject."""

    def __init__(self, id: int, text: str, created_at: float) -> None:
        self.id, self.text, self.created_at = id, text, created_at
        self.vector = None
        self.provenance = {"candidate_id": f"m{id}", "item_index": 0}
