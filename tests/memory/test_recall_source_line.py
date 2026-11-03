"""How a recalled note reads by its source tag (``Item.how``).

Source monitoring: the quote of a record is the user's own words, so only a record
tagged ``typed`` (or with no tag, as every record was before the note gate) may read
"the user said". Words from a page, a file or a finished task's result are never
shown as the user's, whether or not the record carries a quote. Synthetic items, no
model.
"""

from __future__ import annotations

import pytest

from localmelo.support.memory.recall import USER_SOURCES, Item

DAY = "Oct 30"
QUOTE = "I have an MSc in Robotics."
SECOND = "You have an MSc in Robotics."
FIRST = "I have an MSc in Robotics."


def _item(how: str, text: str, quote: str = "", where: str = "") -> Item:
    return Item(
        key="r:1",
        text=text,
        where=where or f"Main chat, {DAY}",
        session_id="s1",
        turn=1,
        created_at=0.0,
        quote=quote,
        how=how,
    )


def _line(item: Item) -> str:
    # ``where`` that does not end in the item's own day is shown as it is
    return item.line(now=0.0)


@pytest.mark.parametrize("how", ["", "typed"])
def test_a_typed_or_untagged_record_with_a_quote_reads_as_the_user_said_it(
    how: str,
) -> None:
    item = _item(how, SECOND, QUOTE, "Main chat")
    assert _line(item) == f"the user said “{QUOTE}” (Main chat)"


@pytest.mark.parametrize("how", ["pointed", "result", "inferred"])
def test_a_record_from_elsewhere_never_reads_as_the_user_said_it(how: str) -> None:
    with_quote = _line(_item(how, SECOND, QUOTE, "jane.example.com"))
    assert "the user said" not in with_quote
    # a quote makes no difference to the line
    assert with_quote == _line(_item(how, SECOND, "", "jane.example.com"))


@pytest.mark.parametrize("quote", ["", QUOTE])
def test_a_pointed_or_result_note_reads_as_a_note_to_the_user(quote: str) -> None:
    for how in ("pointed", "result"):
        item = _item(how, SECOND, quote, "from jane.example.com")
        assert _line(item) == f"a note to the user “{SECOND}” (from jane.example.com)"


@pytest.mark.parametrize("quote", ["", QUOTE])
def test_a_pointed_or_result_note_in_the_first_person_is_the_users_note(
    quote: str,
) -> None:
    for how in ("pointed", "result"):
        item = _item(how, FIRST, quote, "from jane.example.com")
        assert _line(item) == f"the user's note “{FIRST}” (from jane.example.com)"


@pytest.mark.parametrize("quote", ["", QUOTE])
def test_an_inferred_note_keeps_its_melo_inferred_line(quote: str) -> None:
    item = _item("inferred", SECOND, quote, "Main chat")
    assert _line(item) == f"melo inferred “{SECOND}” (Main chat)"


@pytest.mark.parametrize("how", ["", "typed"])
def test_a_typed_or_untagged_record_without_a_quote_reads_as_it_always_did(
    how: str,
) -> None:
    assert _line(_item(how, SECOND, "", "Main chat")) == (
        f"a note to the user “{SECOND}” (Main chat)"
    )
    assert _line(_item(how, FIRST, "", "Main chat")) == (
        f"the user's note “{FIRST}” (Main chat)"
    )


def test_only_the_user_sources_may_use_a_quote() -> None:
    assert USER_SOURCES == ("", "typed")
