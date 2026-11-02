"""Which text boxes melo never types into, and what a form shows of its fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser.driver import BrowserError, FieldAttrs
from localmelo.support.browser.fields import (
    HIDDEN,
    form_pairs,
    normalized,
    refuse_typing,
    sensitive_kind,
)

# One table for the rules, read by the JS-engine tests too: the page script
# must decide as this module does.
CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).with_name("js") / "secret_cases.json").read_text(encoding="utf-8")
)


def attrs_of(spec: dict[str, Any]) -> FieldAttrs:
    return FieldAttrs(**spec)


@pytest.mark.parametrize("case", CASES, ids=[c["why"] for c in CASES])
def test_a_field_is_secret_by_what_it_is_not_by_what_the_page_calls_it(
    case: dict[str, Any],
) -> None:
    assert sensitive_kind(attrs_of(case["attrs"])) == case["kind"]


def test_when_unsure_a_field_counts_as_secret() -> None:
    # No word says so; the shape of a code does.
    shape = FieldAttrs(type="text", inputmode="numeric", maxlength=6, words="field")
    assert sensitive_kind(shape) == "code"
    assert sensitive_kind(None) == ""


def test_refusing_names_what_the_field_is_for() -> None:
    with pytest.raises(BrowserError, match="a payment field"):
        refuse_typing(FieldAttrs(words="Credit card"))
    with pytest.raises(BrowserError, match="never types into passwords"):
        refuse_typing(FieldAttrs(type="password"))
    refuse_typing(FieldAttrs(words="Search"))  # an ordinary one is no problem


def test_words_are_compared_with_punctuation_and_case_spaced_out() -> None:
    assert normalized("userPassword") == "user password"
    assert normalized("user_password") == "user password"
    assert normalized("Card-Number") == "card number"


# ── What the question about a form shows ──


def test_a_form_shows_its_fields_and_hides_what_looks_secret() -> None:
    rows: list[object] = [
        ["email", "me@example.org"],
        ["password", "hunter2"],  # secret by its name
        ["note", "plain", "h"],  # flagged hidden by the page script
        ["session", "AbC123dEf456GhI789jKl012MnO345"],  # a token's shape
        ["card number", "4111 1111 1111 1111"],
        ["x" * 60, "word " * 30],
    ]
    pairs = form_pairs(rows)
    assert pairs[0] == ("email", "me@example.org")
    assert [value for _, value in pairs[1:5]] == [HIDDEN] * 4
    assert len(pairs[5][0]) == 40 and len(pairs[5][1]) == 80


def test_a_form_shows_at_most_twelve_fields_and_skips_malformed_rows() -> None:
    rows: list[object] = [[f"f{i}", "v"] for i in range(30)]
    assert len(form_pairs(rows)) == 12
    assert form_pairs(["x", [], ["only-name"], None, 3]) == ()  # type: ignore[list-item]
