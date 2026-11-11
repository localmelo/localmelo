"""Card and ID numbers have a place, and a title is cleaned like the text.

What memory keeps across chats never holds a card or an ID number: not in the
recall index's text, not in a title. ``records.number_spans`` finds the places of
the numbers in a text as given, so the index cuts a number that no sentence holds
(alone on a line, or wrapped over two lines), and a card is read the ways it is
written (unusual spaces, dots, one line break) or sits beside a short number (an
expiry date, a security code). A stored title, and the title a note's source
names its chat by, go through the function a text does. Constructed numbers only
(the test card 4111 1111 1111 1111, made-up ID numbers).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.index import (
    _SCHEMA,
    ConversationIndex,
    Document,
    cleaned,
    document_text,
)
from localmelo.support.memory.recall import chat_label
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.support.web import search_tool
from localmelo.tests.linear import assert_linear

DATA = Path(__file__).parent / "data"
KEY = "sk-" + "a1B2c3D4" * 4
CARD = "4111 1111 1111 1111"
CARD_TITLE = f"Pay with card {CARD}"
PASSWORD_TITLE = "Sign in with password hunter2 please"
NBSP = "\N{NO-BREAK SPACE}"
THIN = "\N{THIN SPACE}"
NARROW = "\N{NARROW NO-BREAK SPACE}"
IDEOGRAPHIC = "\N{IDEOGRAPHIC SPACE}"
ZERO_WIDTH = "\N{ZERO WIDTH SPACE}"
# The Unicode spaces and invisible characters that may stand between two digits.
UNICODE_GAPS = [
    "\u00a0",
    "\u1680",
    "\u202f",
    "\u205f",
    "\u2060",
    "\u3000",
    "\ufeff",
] + [chr(code) for code in range(0x2000, 0x200E)]


# ── The three gaps ──


def test_a_digits_only_line_with_a_full_stop_is_cut_by_its_place() -> None:
    # The splitter gives no sentence for the line; the number's place cuts it.
    assert cleaned("Here it is.\n4111111111111111.\nThanks.") == (
        "Here it is.\n[left out].\nThanks."
    )


def test_a_card_wrapped_over_two_lines_is_a_secret_and_is_cut() -> None:
    wrapped = "My card is 4111 1111\n1111 1111."
    assert records.sensitive(wrapped)
    assert "4111" not in cleaned(wrapped) and "1111" not in cleaned(wrapped)


@pytest.mark.parametrize(
    "number",
    [
        CARD,
        "4111-1111-1111-1111",
        "4111.1111.1111.1111",
        NBSP.join(["4111", "1111", "1111", "1111"]),
        THIN.join(["4111", "1111", "1111", "1111"]),
        NARROW.join(["4111", "1111", "1111", "1111"]),
        IDEOGRAPHIC.join(["4111", "1111", "1111", "1111"]),
        ZERO_WIDTH.join("4111111111111111"),
        "4111 1111\n1111 1111",
        "4111 1111  \n  1111 1111",
        "4111 1111\r\n1111 1111",
        "4111 11\n11 1111 1111",
        "4111111111111111",
        "5500 0000 0000 0004",
        "340000000000009",
    ],
)
def test_the_ways_a_card_number_is_written_are_secrets(number: str) -> None:
    for text in (number, f"My card is {number}, thanks."):
        assert records.sensitive(text)
        assert records.holds_secret_value(text)
        assert not re.search(r"\d{4}", cleaned(text))
    assert records.number_spans(f"pay {number} now") == [(4, 4 + len(number))]


def test_each_unicode_space_or_invisible_character_may_stand_between_groups() -> None:
    for gap in UNICODE_GAPS:
        number = gap.join(["4111", "1111", "1111", "1111"])
        assert records.number_spans(f"pay {number} now") == [(4, 4 + len(number))], gap


@pytest.mark.parametrize(
    "text",
    [
        "4111\n1111\n1111\n1111",  # a column of groups: two or more line breaks
        "4111 1111\n\n1111 1111",  # a blank line between
        "4111\t1111\t1111\t1111",  # tabs set out a table
        "4111  1111  1111  1111",  # two spaces
        "4111   1111   1111   1111",  # three spaces
        "4111    1111    1111    1111",  # four spaces
        "4111.1111 1111 1111",  # a dot beside another separator
        "4111-1111.1111-1111",
        "4111 1111 1111 1112",  # the checksum fails
        "4111 1111 1111",  # 12 digits
        "41111111111111111111",  # 20 digits in one run
        "4111 / 1111 / 1111 / 1111",
        "4111, 1111, 1111, 1111",
    ],
)
def test_what_is_not_written_as_a_card_is_left_alone(text: str) -> None:
    assert records.number_spans(text) == []
    assert not records.sensitive(text)
    assert cleaned(text) == text


def test_an_id_number_has_its_place() -> None:
    assert records.number_spans("id 123-45-6789.") == [(3, 14)]
    assert records.number_spans("id 11010519491231002X ok") == [(3, 21)]
    assert records.number_spans("v 1234-56-7890") == []


@pytest.mark.parametrize(
    "text",
    [
        f"{CARD}  12/27",
        f"Total 12.50  {CARD}",
        f"{CARD}\n123 Main Street",
        f"Order 12345\n{CARD}",
        f"{CARD}.50",
        f"{CARD}\t12/27",
    ],
)
def test_a_card_the_plain_reading_finds_is_still_found_beside_other_digits(
    text: str,
) -> None:
    # Read as it is written, the card and the digits next to it join into one longer
    # number that fails the checksum; the plain reading (one space or hyphen) finds the card.
    assert any(
        records._luhn(re.sub(r"\D", "", m.group()))
        for m in records._CARD.finditer(text)
    )
    assert records.sensitive(text)
    assert "4111" not in cleaned(text)


# A wrapped number is a number only when it is the whole run: the rows of a table
# are not one wrapped number, whatever the checksum says.
TABLE = "Totals\n1500 2500 3500\n4500 5500 6500"
LONG_TABLE = "7151 7511 7877\n2196 4252 5420\n6519 2427 6098\n6449 1248 7718"


@pytest.mark.parametrize(
    "text",
    [
        TABLE,
        LONG_TABLE,
        "12 4111 1111\n1111 1111",  # a number joined before a wrapped card: not found
    ],
)
def test_a_line_break_joins_a_number_only_when_it_is_the_whole_run(text: str) -> None:
    assert records.number_spans(text) == []
    assert not records.sensitive(text)
    assert cleaned(text) == text


def test_a_wrapped_card_that_is_the_whole_run_is_found() -> None:
    text = "My card is 4111 1111\n1111 1111."
    assert records.number_spans(text) == [(11, 30)]


@pytest.mark.parametrize(
    "text",
    [
        f"My card is {CARD} 12/27.",
        f"{CARD} 12 27 123",
        f"Card {CARD} 123",
        f"Order 87 {CARD}",
        "4111111111111111 12/27",
        "3400 000000 00009 12/30",
        f"{CARD}.50",
    ],
)
def test_a_card_beside_a_short_number_is_found_without_it(text: str) -> None:
    # An expiry date, a security code or an order number joins the card into one
    # longer number that fails the checksum; the card alone is found again.
    card = next(
        found
        for found in ("3400 000000 00009", "4111111111111111", CARD)
        if found in text
    )
    start = text.index(card)
    assert records.number_spans(text) == [(start, start + len(card))]
    assert card[:4] not in cleaned(text)


@pytest.mark.parametrize(
    "text",
    [
        "Fibonacci: 1 1 2 3 5 8 13 21 34 55 89 144",
        "Pages 101 102 103 104 105 106 107",
    ],
)
def test_the_retry_beside_a_short_number_adds_no_place_to_a_row_of_numbers(
    text: str,
) -> None:
    # The second has the one place the plain reading gives it by the checksum alone;
    # the retry gives neither a place it did not have.
    plain = [
        match.span()
        for match in records._CARD.finditer(text)
        if records._luhn(re.sub(r"\D", "", match.group()))
    ]
    assert records.number_spans(text) == plain


def _repeated(unit: str) -> Callable[[int], str]:
    return lambda size: unit * (size // len(unit))


@pytest.mark.parametrize(
    "unit",
    [
        "1 ",
        "1234 ",
        "12 \n  34\t5.6-",
        "1" + " " * 7,
        "1" * 40 + " ",
        "1\u200b",
        "1.2",
        "4111 1111 1111 1112 12\n",
        "1" + " " * 8 + "\r\n" + " " * 8,
    ],
)
def test_finding_the_numbers_takes_time_in_proportion_to_the_text(unit: str) -> None:
    assert_linear(_repeated(unit), records.number_spans)


# ── The index cuts every place ──

SAFE = ("The garden needs water.", "Basil grows fast.")
NUMBERS = {
    "card": CARD,
    "wrapped card": "4111 1111\n1111 1111",
    "digits-only line": "4111111111111111.",
    "ID number": "123-45-6789",
}
ELSEWHERE = {
    "": "",
    "password": "\nMy wifi password is hunter2 for the shed.",
    "key": f"\nUse the key {KEY} for the build.",
}


def grid() -> Iterator[tuple[str, str, str, str]]:
    first, second = SAFE
    for kind, number in NUMBERS.items():
        spots = {
            "alone": number,
            "after a sentence": f"{first}\n{number}",
            "before a sentence": f"{number}\n{second}",
            "in a quoted answer": f'{first}\nThe answer was: "{number}"\n{second}',
            "twice": f"{first}\n{number}\n{second}\n{number}",
        }
        for spot, text in spots.items():
            for extra, more in ELSEWHERE.items():
                yield f"{kind} / {spot} / {extra or 'no secret'}", kind, text, more


@pytest.mark.parametrize(
    ("name", "kind", "text", "more"), list(grid()), ids=[g[0] for g in grid()]
)
def test_a_number_is_never_in_what_the_index_keeps(
    name: str, kind: str, text: str, more: str
) -> None:
    whole = text + more
    out = cleaned(whole)
    assert not re.search(r"\d{4}", out), out
    assert "6789" not in out and "hunter2" not in out and KEY not in out
    assert "[left out]" in out
    for sentence in SAFE:
        if sentence in whole:
            assert sentence in out, out
    assert cleaned(out) == out


def test_the_marking_word_may_stand_in_another_sentence_than_the_number() -> None:
    text = "My card number is below.\n4111111111111111."
    out = cleaned(text)
    assert "4111" not in out and "[left out]" in out
    assert cleaned(out) == out


def test_a_secret_next_to_a_number_does_not_hide_it() -> None:
    text = f"The password is hunter2.\n{CARD}\nSsn 123-45-6789 is on file."
    out = cleaned(text)
    assert not re.search(r"\d{4}|hunter2", out), out


# ── The ordinary texts with long digit runs ──


def _plain_reading(text: str) -> bool:
    """What the rule answered for a text without a secret word when a card was read
    only as digits joined by one space or hyphen."""
    return (
        any(
            records._luhn(re.sub(r"\D", "", m.group()))
            for m in records._CARD.finditer(text)
        )
        or records._ID_NUMBER.search(text) is not None
        or bool(redact_secrets(text)[1])
    )


def test_ordinary_digit_runs_are_refused_as_they_were_except_one() -> None:
    rows = json.loads((DATA / "number_precision.json").read_text("utf-8"))
    assert len(rows) == 69
    texts = [row["text"] for row in rows]
    # "before" is the plain reading's answer, here as the code gives it.
    assert [row["before"] for row in rows] == [_plain_reading(t) for t in texts]
    after = [records.sensitive(t) for t in texts]
    assert sum(row["before"] for row in rows) == 5
    assert sum(after) == 6
    changed = [
        t
        for t, now, row in zip(texts, after, rows, strict=True)
        if now != row["before"]
    ]
    # A dot joins the years of a row, and the checksum passes.
    assert changed == ["Seasons 2010.2011.2012.2013"]
    assert all(now for now, row in zip(after, rows, strict=True) if row["before"])


# ── A title is cleaned like the text ──


@pytest.mark.parametrize(
    "title", [CARD_TITLE, PASSWORD_TITLE, f"Pay {CARD[:9]}\n{CARD[10:]}"]
)
def test_a_tasks_title_in_a_documents_text_is_cleaned(title: str) -> None:
    assert document_text("Do it.", "Done.", title=title) == (
        "Task: [left out]\nUser: Do it.\nmelo: Done."
    )
    assert cleaned(cleaned(title)) == cleaned(title) == "[left out]"


def test_an_ordinary_title_is_kept_and_an_empty_one_has_no_task_line() -> None:
    assert document_text("Do it.", None, title="Water the beds") == (
        "Task: Water the beds\nUser: Do it."
    )
    assert document_text("Do it.", None, title="  ") == "User: Do it."


def _doc(kind: str, title: str, session: str = "s1", turn: int = 1) -> Document:
    return Document(
        session_id=session,
        turn=turn,
        surface="gateway",
        kind=kind,
        title=title,
        text=document_text("tomato garden plan", "water daily"),
        created_at=time.time(),
    )


async def test_a_stored_title_is_cleaned_whatever_the_kind(tmp_path: Path) -> None:
    index = ConversationIndex(str(tmp_path / "index.db"))
    try:
        await index.add_many(
            [
                _doc("task", CARD_TITLE, "t1"),
                _doc("chat", PASSWORD_TITLE, "c1"),
                _doc("task", "Water the beds", "t2"),
                _doc("front", "", "f1"),
            ]
        )
        stored = await index.fetch(await index.ids())
        assert {d.session_id: d.title for d in stored} == {
            "t1": "[left out]",
            "c1": "[left out]",
            "t2": "Water the beds",
            "f1": "",
        }
        labels = await index.session_labels({"t1", "c1"})
        assert labels["t1"][1] == labels["c1"][1] == "[left out]"
    finally:
        await index.aclose()


async def test_recall_reads_a_cleaned_title_back(tmp_path: Path) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        await index.add(_doc("task", CARD_TITLE, "t1"))
        await index.add(_doc("chat", PASSWORD_TITLE, "c1", turn=2))
        items = await memory.search("tomato garden plan", docs_k=4, records_k=0)
        where = sorted(item.where.split(",")[0] for item in items)
        assert where == ["chat “[left out]”", "task “[left out]”"]
        assert chat_label("task", "[left out]", "gateway") == "task “[left out]”"
    finally:
        await memory.aclose()


async def test_the_documents_a_turn_is_indexed_as_keep_a_cleaned_title(
    tmp_path: Path,
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        turns = [
            (
                {"session_id": "t1", "kind": "task", "title": CARD_TITLE},
                1,
                "Do it.",
                "Done.",
                1.0,
            ),
            (
                {"session_id": "c1", "kind": "chat", "title": PASSWORD_TITLE},
                1,
                "Hi.",
                "Hello.",
                2.0,
            ),
        ]
        await memory.index_turns(turns)
        _, index = memory.stores()
        stored = {d.session_id: d for d in await index.fetch(await index.ids())}
        assert stored["t1"].title == stored["c1"].title == "[left out]"
        assert stored["t1"].text.startswith("Task: [left out]\n")
        assert "4111" not in "".join(d.text for d in stored.values())
    finally:
        await memory.aclose()


@pytest.mark.parametrize("title", [CARD_TITLE, PASSWORD_TITLE])
async def test_a_notes_source_names_its_chat_by_a_cleaned_title(
    tmp_path: Path, title: str
) -> None:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        typed = "Remember that I prefer short answers."
        turn = await store.begin_turn(
            None,
            typed,
            holder="test-host:1:memory",
            title=title,
            new_session=NewSession(),
        )
        turns = await store.list_turns(turn.session_id)
        found = await memory.before_turn(store, turns[-1], typed)
        assert [w.text for w in found.remembered] == ["You prefer short answers."]
        (entry,) = await memory.records()
        assert entry.provenance["title"] == "[left out]"
        assert entry.provenance["chat"] == "chat “[left out]”"
        stored = json.dumps(entry.provenance, ensure_ascii=False)
        assert CARD not in stored and "hunter2" not in stored
        items = await memory.search("short answers", docs_k=0, records_k=4)
        assert [item.where.split(",")[0] for item in items] == ["chat “[left out]”"]
    finally:
        await memory.aclose()
        await store.aclose()


# ── Rows already in an index ──


def _old_insert(path: str, rows: list[tuple[str, str, str]]) -> None:
    """Rows as an index.db made before titles were cleaned holds them: the title
    stored as given (session id, kind and title of each)."""
    raw = sqlite3.connect(path)
    raw.executescript(_SCHEMA)
    for turn, (session, kind, title) in enumerate(rows, 1):
        raw.execute(
            "INSERT INTO documents (session_id, turn, surface, kind, title, text, "
            "created_at, vector, embed_model, chunk) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session,
                turn,
                "gateway",
                kind,
                title,
                "User: hi\nmelo: hello",
                1.0,
                None,
                "",
                0,
            ),
        )
    raw.commit()
    raw.close()


async def test_an_index_made_before_titles_were_cleaned_is_cleaned_once(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "index.db")
    _old_insert(
        path,
        [
            ("t1", "task", CARD_TITLE),
            ("c1", "chat", PASSWORD_TITLE),
            ("t2", "task", "Water the beds"),
            ("t3", "task", CARD_TITLE),
        ],
    )
    index = ConversationIndex(path)
    try:
        stored = await index.fetch(await index.ids())
        assert {d.session_id: d.title for d in stored} == {
            "t1": "[left out]",
            "c1": "[left out]",
            "t2": "Water the beds",
            "t3": "[left out]",
        }
        assert {d.text for d in stored} == {"User: hi\nmelo: hello"}
        assert await index.meta("titles_cleaned") == 1
    finally:
        await index.aclose()
    # The old titles are gone from the file and its log, not only from the rows.
    kept = b"".join(p.read_bytes() for p in tmp_path.glob("index.db*"))
    assert b"4111 1111" not in kept and b"hunter2" not in kept
    # Opened again it does not look at the titles again: the step runs once.
    _old_insert(path, [("t9", "task", CARD_TITLE)])
    again = ConversationIndex(path)
    try:
        titles = {d.session_id: d.title for d in await again.fetch(await again.ids())}
        assert titles["t1"] == "[left out]" and titles["t9"] == CARD_TITLE
    finally:
        await again.aclose()


# ── A search query ──


@pytest.mark.parametrize(
    "query",
    [
        f"is {NBSP.join(['4111', '1111', '1111', '1111'])} valid",
        "is 4111 1111\n1111 1111 valid",
        "is 4111.1111.1111.1111 valid",
        f"is {CARD} 12/27 valid",
    ],
)
def test_a_search_query_with_a_card_written_in_another_way_asks_as_a_secret(
    query: str,
) -> None:
    assert search_tool._doubt(query) == "secret"


@pytest.mark.parametrize(
    "query",
    [
        "tomato blight 2019 2020 2021 2022",
        "isbn 978 0 306 40615 7",
        "scores 0.8387 0.7794 0.3050 0.1542",
        "is 4111\t1111\t1111\t1111 valid",
        "is 4111  1111  1111  1111 valid",
    ],
)
def test_an_ordinary_search_query_with_digits_does_not_ask(query: str) -> None:
    assert search_tool._doubt(query) == ""
