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
    cleaned_title,
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
# A title cut by its places: the words stay, the secret's place is left out.
CARD_LABEL = "Pay with card [left out]"
PASSWORD_LABEL = "Sign in with password [left out] please"
# Titles whose words a note would refuse; a title keeps them.
WORD_TITLES = [
    "Reset password flow for the login page",
    "Add a PIN pad to the checkout form",
    "Debug OTP timeout",
    "Wi-Fi password rotation",
    "Password-protected PDF export",
    "Why is the OTP not arriving?",
]
NBSP = "\N{NO-BREAK SPACE}"
THIN = "\N{THIN SPACE}"
NARROW = "\N{NARROW NO-BREAK SPACE}"
IDEOGRAPHIC = "\N{IDEOGRAPHIC SPACE}"
ZERO_WIDTH = "\N{ZERO WIDTH SPACE}"
EN_DASH = "\N{EN DASH}"
# The Unicode hyphens and dashes, and the minus sign, that stand between a card's
# groups only when every separator is the same one.
DASHES = [
    "\N{HYPHEN}",
    "\N{NON-BREAKING HYPHEN}",
    "\N{FIGURE DASH}",
    EN_DASH,
    "\N{EM DASH}",
    "\N{MINUS SIGN}",
]
# Characters that neither separate a card's groups nor join a digit to a number.
NO_SEPARATORS = [
    "\N{HORIZONTAL BAR}",
    "\N{SMALL EM DASH}",
    "\N{SMALL HYPHEN-MINUS}",
    "\N{FULLWIDTH HYPHEN-MINUS}",
    "\N{SOFT HYPHEN}",
]
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
        EN_DASH.join(["4111", "1111", "1111", "1111"]),
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


@pytest.mark.parametrize("mark", NO_SEPARATORS)
def test_a_character_that_is_no_separator_joins_no_digit_to_a_wrapped_card(
    mark: str,
) -> None:
    # A second card, wrapped, right after the first: the character between them is
    # no part of a number, so the wrapped one is the whole run and is found whole.
    text = f"Card {CARD}{mark}4111 1111\n1111 1111"
    assert not re.search(r"\d", cleaned(text))
    number = mark.join(["4111", "1111", "1111", "1111"])
    assert records.number_spans(f"pay {number} now") == []


def test_each_unicode_space_or_invisible_character_may_stand_between_groups() -> None:
    for gap in UNICODE_GAPS:
        number = gap.join(["4111", "1111", "1111", "1111"])
        assert records.number_spans(f"pay {number} now") == [(4, 4 + len(number))], gap


@pytest.mark.parametrize("dash", [*DASHES, "."])
def test_a_dot_or_a_dash_separates_a_cards_digits_only_in_one_form(dash: str) -> None:
    # Every separator the same, at least three, every group but the last of four
    # digits. The digits of each text below pass the checksum.
    number = dash.join(["4111", "1111", "1111", "1111"])
    assert records.number_spans(f"pay {number} now") == [(4, 4 + len(number))]
    for text in (
        f"4111{dash}1111 1111 1111",  # beside a space
        f"4111{dash}1111-1111-1111",  # beside a hyphen
        dash.join(["41111", "1111", "1111", "111"]),  # a group of five
        dash.join(["4", "111111111111111"]),  # one separator
        dash.join(["3400", "000000", "00009"]),  # two separators
    ):
        assert records.number_spans(text) == [], text


def test_a_number_does_not_start_in_a_decimal_numbers_fraction() -> None:
    # The digits after "<digit>." are the fraction of a number, whatever the checksum
    # says; a card written beside such a number is still found.
    assert records.number_spans("Score 0.4111111111111111") == []
    # Eighteen digits there are no ID number either; after a word they are one.
    assert records.number_spans("Score 0.001728625909087178") == []
    assert records.number_spans("Id 110101199003070018.") == [(3, 21)]
    assert records.number_spans(f"Total 12.50 {CARD}") == [(12, 12 + len(CARD))]
    # Inside a number of dotted groups the card is read without its first group.
    dotted = "4111.1111.1111.1111"
    assert records.number_spans(f"Order 12.{dotted}") == [(9, 9 + len(dotted))]


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
        f"4111{EN_DASH}1111 1111 1111",  # a dash beside a space
        f"2010{EN_DASH}2011 2012{EN_DASH}2013",  # a row of year ranges
        "3.14159265358979",  # a float: the digits after the point are no number
        "0.8387461234567891",
        "40.71277612345, -74.00597412345",  # a latitude and a longitude
        "70.68089788244, 51.81896411662",  # the same, each number passing the checksum
        "968.8489275020787",  # a float that passes the checksum
        "2024.11.10.123456.7890",  # a build number: groups are not of four digits
        "Score 0.4111111111111111",
        "Score 0.4111 1111 1111 1111",
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
        "1\u2013",
        "1.4111.1111.1111.",
        "4111 1111 1111 1112 12\n",
        "1" + " " * 8 + "\r\n" + " " * 8,
    ],
)
def test_finding_the_numbers_takes_time_in_proportion_to_the_text(unit: str) -> None:
    assert_linear(_repeated(unit), records.number_spans)


@pytest.mark.parametrize(
    "unit",
    [
        "The password is hunter2\n4111 1111\n1111 1111\nThanks.\n",  # two passes
        "4111 1111\n1111 1111\n",
        "1234 ",
    ],
)
def test_cleaning_a_text_takes_time_in_proportion_to_the_text(unit: str) -> None:
    assert_linear(_repeated(unit), cleaned)


@pytest.mark.parametrize(
    "unit", ["PIN 4821 is set. ", "password is hunter\n", "password is ", "a" * 50]
)
def test_cleaning_a_title_takes_time_in_proportion_to_the_title(unit: str) -> None:
    assert_linear(_repeated(unit), cleaned_title)


def test_a_card_under_three_lines_that_are_each_cut_is_found() -> None:
    # Each wrapped card is readable only when the line above it is gone: four layers.
    wrapped = "4111 1111 1111\n1111"
    text = "OTP 482913\n" + " code 12\n".join([wrapped] * 3)
    once = cleaned(text)
    assert not re.search(r"\d{4}", once)
    assert cleaned(once) == once


# ── The index cuts every place ──

SAFE = ("The garden needs water.", "Basil grows fast.")
NUMBERS = {
    "card": CARD,
    "wrapped card": "4111 1111\n1111 1111",
    "digits-only line": "4111111111111111.",
    "ID number": "123-45-6789",
}
# A secret next to the number: after it, or on the line before it. A line before ends
# in a digit, so a wrapped card under it is not the whole run until that line is cut.
ELSEWHERE = {
    "": ("", ""),
    "password": ("", "\nMy wifi password is hunter2 for the shed."),
    "key": ("", f"\nUse the key {KEY} for the build."),
    "password line before": ("The password is hunter2\n", ""),
    "key line before": (f"Use key {KEY}\n", ""),
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
            for extra, (before, after) in ELSEWHERE.items():
                name = f"{kind} / {spot} / {extra or 'no secret'}"
                yield name, kind, before + text + after, extra


@pytest.mark.parametrize(
    ("name", "kind", "whole", "extra"), list(grid()), ids=[g[0] for g in grid()]
)
def test_a_number_is_never_in_what_the_index_keeps(
    name: str, kind: str, whole: str, extra: str
) -> None:
    out = cleaned(whole)
    assert not re.search(r"\d{4}", out), out
    assert "6789" not in out and "hunter2" not in out and KEY not in out
    assert "[left out]" in out
    for sentence in SAFE:
        if sentence in whole:
            assert sentence in out, out
    assert cleaned(out) == out


@pytest.mark.parametrize("lead", ["The password is hunter2", f"Use key {KEY}"])
def test_a_card_under_a_line_that_ends_in_a_digit_is_cut_in_one_call(lead: str) -> None:
    # The line before it is cut first; only then does the card read as the whole run
    # (a digit joined to it through a line break says it is a table's row). The cut
    # is repeated, so one call cleans what the second would have.
    out = cleaned(f"{lead}\n4111 1111\n1111 1111\nThanks.")
    assert out == "[left out]\n[left out]\nThanks."
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


# A card read only as digits joined by one space or hyphen, wherever it starts.
PLAIN_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


# An ID number read wherever its digits start.
PLAIN_ID = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4}|\d{17}[\dXx])(?![\dA-Za-z])")


def _plain_reading(text: str) -> bool:
    """What the rule answered for a text without a secret word when a card was read
    only as digits joined by one space or hyphen, and an ID number wherever its
    digits start."""
    return (
        any(
            records._luhn(re.sub(r"\D", "", m.group()))
            for m in PLAIN_CARD.finditer(text)
        )
        or PLAIN_ID.search(text) is not None
        or bool(redact_secrets(text)[1])
    )


def test_ordinary_digit_runs_are_refused_as_they_were_except_for_the_changes() -> None:
    rows = json.loads((DATA / "number_precision.json").read_text("utf-8"))
    assert len(rows) == 81
    texts = [row["text"] for row in rows]
    # "before" is the plain reading's answer, here as the code gives it.
    assert [row["before"] for row in rows] == [_plain_reading(t) for t in texts]
    after = [records.sensitive(t) for t in texts]
    assert sum(row["before"] for row in rows) == 9
    assert sum(after) == 6
    changed = [
        t
        for t, now, row in zip(texts, after, rows, strict=True)
        if now != row["before"]
    ]
    assert changed == [
        # A dot joins the years of a row, and the checksum passes.
        "Seasons 2010.2011.2012.2013",
        # The digits after a decimal point are no card.
        "Score 0.4425935186026221",
        "Score 0.46432068068045473",
        # Eighteen digits after a decimal point are no ID number.
        "Score 0.001728625909087178",
        "Score 0.026612664415084475",
    ]
    assert all(
        now
        for text, now, row in zip(texts, after, rows, strict=True)
        if row["before"] and row["kind"] != "fraction"
    )


# ── A title is cleaned by its places ──

TITLES = {
    CARD_TITLE: CARD_LABEL,
    PASSWORD_TITLE: PASSWORD_LABEL,
    f"Use key {KEY} for the build": "Use key [left out] for the build",
    "Ssn 123-45-6789 lookup": "Ssn [left out] lookup",
    f"Pay {CARD[:9]}\n{CARD[10:]} now": "Pay [left out] now",
    "Fix login, password: hunter2": "Fix login, password: [left out]",
    "OTP 482913 timeout": "OTP [left out] timeout",
    "The password is hunter2\n4111 1111\n1111 1111\nThanks.": (
        "The password is [left out]\n[left out]\nThanks."
    ),
    # A sentence that names a password or a code: its runs with a digit, and what
    # follows the word and a link.
    "PIN 4821 for the bank": "PIN [left out] for the bank",
    "4821 is my PIN": "[left out] is my PIN",
    "CVV 737 on the card": "CVV [left out] on the card",
    "security code 123 on the back": "security code [left out] on the back",
    "pin number 4821 is set": "pin number [left out] is set",
    "Reset my password to hunter2.": "Reset my password to [left out].",
    "my password was hunter2 before": "my password was [left out]",
    "password — hunter2": "password — [left out]",
    "password is correct horse battery staple": "password is [left out]",
    "My wifi password is sunflowerfields": "My wifi password is [left out]",
    "The passwords are alpha and beta. Next: deploy.": (
        "The passwords are [left out]. Next: deploy."
    ),
    f"Card {CARD.replace(' ', '-')} and PIN 4821": (
        "Card [left out] and PIN [left out]"
    ),
    "修复密码页面的3个问题": "修复密码页面的[left out]个问题",
    "密码是correct horse": "密码是[left out]",
    # A value of letters only, after each link.
    "password to sunflower": "password to [left out]",
    "The passwords were alpha": "The passwords were [left out]",
    "The PINs are alpha and beta": "The PINs are [left out]",
    "OTP = abcd": "OTP = [left out]",
    "security code: abcd": "security code: [left out]",
    f"password {EN_DASH} sunflower": f"password {EN_DASH} [left out]",
    "password - sunflower": "password - [left out]",
    "密码为sunflower": "密码为[left out]",
    # The places are cut first, to the end: a card under a number that is itself
    # cut is found whole, and a word that marks a password is not cut before it.
    "123-45-6789 4111 1111\n1111 1111 PIN": "[left out] [left out] PIN",
    "PIN 123-45-6789 4111 1111\n1111 1111": "PIN [left out] [left out]",
    "4111111111111111pw: sunflower and PIN": "[left out]pw: [left out] and PIN",
    # Then no digit of a title that names one is kept, on any line of it.
    "PIN\n4821": "PIN\n[left out]",
    "PIN 12 4111 1111\n1111 1111": (
        "PIN [left out] [left out] [left out]\n[left out] [left out]"
    ),
    **{title: title for title in WORD_TITLES},
}


@pytest.mark.parametrize(("title", "label"), list(TITLES.items()))
def test_a_title_loses_the_places_of_its_secrets_and_keeps_its_words(
    title: str, label: str
) -> None:
    assert cleaned_title(title) == label
    assert cleaned_title(label) == label
    assert not re.search(
        r"\d|hunter2|a1B2|horse|sunflower|alpha|abcd", cleaned_title(title)
    )


def test_a_password_of_letters_that_no_link_comes_before_stays_in_a_title() -> None:
    # The stated limit: nothing says which word is the value.
    for title in ("sunflower is my password", "password sunflower"):
        assert records.named_secret_spans(title) == []
        assert cleaned_title(title) == title


def test_a_text_that_names_no_password_or_code_gives_no_such_place() -> None:
    assert records.named_secret_spans("Room 4821 is booked. Call 555 0100.") == []
    # One that names one keeps no digit, in any sentence of it.
    assert records.named_secret_spans("PIN 4821, then room 12. Call 555.") == [
        (4, 8),
        (20, 22),
        (29, 32),
    ]


def test_a_title_with_a_private_key_block_loses_its_body() -> None:
    title = "Key -----BEGIN PRIVATE KEY-----\nMIIabcdefghijklmnop\n-----END PRIVATE KEY-----"
    assert "MIIabcdefghijklmnop" not in cleaned_title(title)
    assert cleaned_title(cleaned_title(title)) == cleaned_title(title)


@pytest.mark.parametrize(("title", "label"), list(TITLES.items()))
def test_a_tasks_title_in_a_documents_text_is_cleaned(title: str, label: str) -> None:
    text = document_text("Do it.", "Done.", title=title)
    assert text == f"Task: {label}\nUser: Do it.\nmelo: Done."


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
                _doc("chat", WORD_TITLES[0], "w1"),
                _doc("task", "  ", "w2"),
            ]
        )
        stored = await index.fetch(await index.ids())
        assert {d.session_id: d.title for d in stored} == {
            "t1": CARD_LABEL,
            "c1": PASSWORD_LABEL,
            "t2": "Water the beds",
            "f1": "",
            "w1": WORD_TITLES[0],
            "w2": "",
        }
        labels = await index.session_labels({"t1", "c1"})
        assert labels["t1"][1] == CARD_LABEL
        assert labels["c1"][1] == PASSWORD_LABEL
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
        assert where == [f"chat “{PASSWORD_LABEL}”", f"task “{CARD_LABEL}”"]
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
        assert stored["t1"].title == CARD_LABEL
        assert stored["c1"].title == PASSWORD_LABEL
        assert stored["t1"].text.startswith(f"Task: {CARD_LABEL}\n")
        assert "4111" not in "".join(d.text for d in stored.values())
    finally:
        await memory.aclose()


@pytest.mark.parametrize(
    ("title", "label"),
    [
        (CARD_TITLE, CARD_LABEL),
        (PASSWORD_TITLE, PASSWORD_LABEL),
        (WORD_TITLES[0], WORD_TITLES[0]),
    ],
)
async def test_a_notes_source_names_its_chat_by_a_cleaned_title(
    tmp_path: Path, title: str, label: str
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
        assert entry.provenance["title"] == label
        assert entry.provenance["chat"] == f"chat “{label}”"
        stored = json.dumps(entry.provenance, ensure_ascii=False)
        assert CARD not in stored and "hunter2" not in stored
        items = await memory.search("short answers", docs_k=0, records_k=4)
        assert [item.where.split(",")[0] for item in items] == [f"chat “{label}”"]
    finally:
        await memory.aclose()
        await store.aclose()


@pytest.mark.parametrize(
    ("title", "label"),
    [
        (CARD_TITLE, CARD_LABEL),
        (PASSWORD_TITLE, PASSWORD_LABEL),
        (WORD_TITLES[1], WORD_TITLES[1]),
    ],
)
async def test_a_note_stored_with_a_raw_title_reads_a_cleaned_label_at_recall(
    tmp_path: Path, title: str, label: str
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        # The source as it was written before a title was cleaned: the title as the
        # chat has it, and the label made from it.
        source = {
            "session_id": "s1",
            "turn": 1,
            "turn_id": 1,
            "kind": "chat",
            "title": title,
            "surface": "gateway",
            "chat": chat_label("chat", title, "gateway"),
            "workspace": "",
        }
        await records.write(
            memory.stores()[0],
            [
                records.Candidate(
                    "You prefer short answers.",
                    "I prefer short answers.",
                    "preferences",
                    "user_statement",
                )
            ],
            [None],
            source=source,
            embed_model="",
        )
        # What is stored stays as it was; the line the model reads is cleaned.
        (entry,) = await memory.records()
        assert entry.provenance["title"] == title
        items = await memory.search("short answers", docs_k=0, records_k=4)
        assert [item.where.split(",")[0] for item in items] == [f"chat “{label}”"]
        assert not re.search(r"\d{4}|hunter2", items[0].where)
    finally:
        await memory.aclose()


# ── Rows already in an index ──


def _old_text(kind: str, title: str) -> str:
    """A turn's text as it was written before a title was cleaned: a task's title line
    went through the redaction only."""
    lead = redact_secrets(f"Task: {title.strip()}\n")[0] if kind == "task" else ""
    return f"{lead}User: hi\nmelo: hello"


USER_TITLE = f"Setup\nUser: card {CARD}"


@pytest.mark.parametrize(
    ("title", "line", "label"),
    [
        # As given, where today's redaction would have changed the line.
        (
            f"Use key {KEY} for the build",
            f"Use key {KEY} for the build",
            "Use key [left out] for the build",
        ),
        # Written another way: found by where the user's words start.
        (
            f"Use key {KEY} for the build",
            "Use key (hidden) for the build",
            "Use key [left out] for the build",
        ),
        # A title that holds a line of the user's itself, as given and another way.
        (USER_TITLE, USER_TITLE, "Setup\nUser: card [left out]"),
        (USER_TITLE, "Setup\nUser: card (hidden)", "Setup\nUser: card [left out]"),
    ],
)
async def test_a_title_line_an_older_version_wrote_is_replaced_whole(
    tmp_path: Path, title: str, line: str, label: str
) -> None:
    path = str(tmp_path / "index.db")
    _old_insert(path, [("t1", "task", title)])
    raw = sqlite3.connect(path)
    raw.execute(
        "UPDATE documents SET text = ?", (f"Task: {line}\nUser: hi\nmelo: hello",)
    )
    raw.commit()
    raw.close()
    index = ConversationIndex(path)
    try:
        (doc,) = await index.fetch(await index.ids())
        assert doc.title == label
        assert doc.text == f"Task: {label}\nUser: hi\nmelo: hello"
    finally:
        await index.aclose()
    kept = b"".join(p.read_bytes() for p in tmp_path.glob("index.db*"))
    assert not re.search(rb"a1B2|4111|1111 1111", kept)


def _old_insert(path: str, rows: list[tuple[str, str, str]]) -> None:
    """Rows as an index.db made before titles were cleaned holds them: the title
    stored as given and a task's text starting with its title line (session id,
    kind and title of each)."""
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
                _old_text(kind, title),
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
    key_title = f"Use key {KEY} for the build"
    wrapped = f"Pay {CARD[:9]}\n{CARD[10:]} now"
    rows = [
        ("t1", "task", CARD_TITLE),
        ("c1", "chat", PASSWORD_TITLE),
        ("t2", "task", "Water the beds"),
        ("t3", "task", CARD_TITLE),
        ("t4", "task", key_title),
        ("t5", "task", wrapped),
        ("t6", "task", WORD_TITLES[0]),
    ]
    _old_insert(path, rows)
    # The text of the older code starts with the title it was given.
    assert _old_text("task", CARD_TITLE).startswith(f"Task: {CARD_TITLE}\n")
    index = ConversationIndex(path)
    try:
        stored = {d.session_id: d for d in await index.fetch(await index.ids())}
        assert {session: d.title for session, d in stored.items()} == {
            "t1": CARD_LABEL,
            "c1": PASSWORD_LABEL,
            "t2": "Water the beds",
            "t3": CARD_LABEL,
            "t4": "Use key [left out] for the build",
            "t5": "Pay [left out] now",
            "t6": WORD_TITLES[0],
        }
        # The task line is the cleaned title; nothing else of a text changes.
        body = "User: hi\nmelo: hello"
        assert {session: d.text for session, d in stored.items()} == {
            "t1": f"Task: {CARD_LABEL}\n{body}",
            "c1": body,
            "t2": f"Task: Water the beds\n{body}",
            "t3": f"Task: {CARD_LABEL}\n{body}",
            "t4": f"Task: Use key [left out] for the build\n{body}",
            "t5": f"Task: Pay [left out] now\n{body}",
            "t6": f"Task: {WORD_TITLES[0]}\n{body}",
        }
        assert await index.meta("titles_cleaned") == 1
    finally:
        await index.aclose()
    # The old titles are gone from the file and its log, not only from the rows.
    kept = b"".join(p.read_bytes() for p in tmp_path.glob("index.db*"))
    assert not re.search(rb"4111|hunter2|a1B2|1111 1111", kept)
    # Opened again it does not look at the rows again: the step runs once.
    _old_insert(path, [("t9", "task", CARD_TITLE)])
    again = ConversationIndex(path)
    try:
        docs = {d.session_id: d for d in await again.fetch(await again.ids())}
        assert docs["t1"].title == CARD_LABEL
        assert docs["t9"].title == CARD_TITLE
        assert docs["t9"].text == _old_text("task", CARD_TITLE)
        assert docs["t1"].text == f"Task: {CARD_LABEL}\n{body}"
    finally:
        await again.aclose()


async def test_a_text_that_does_not_start_with_its_title_line_is_left_as_it_is(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "index.db")
    _old_insert(path, [("t1", "task", CARD_TITLE)])
    raw = sqlite3.connect(path)
    raw.execute("UPDATE documents SET text = 'User: hi\nmelo: hello'")
    raw.commit()
    raw.close()
    index = ConversationIndex(path)
    try:
        (doc,) = await index.fetch(await index.ids())
        assert doc.title == CARD_LABEL
        assert doc.text == "User: hi\nmelo: hello"
    finally:
        await index.aclose()


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
