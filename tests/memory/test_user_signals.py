"""The user signals: only a declarative, stable statement is a fact."""

from __future__ import annotations

import pytest

from localmelo.melo.memory.hippo import classify_kind, classify_user_text
from localmelo.melo.memory.hippo.signals import (
    request_candidates,
    request_note,
    stands_alone,
    user_candidates,
)


def test_a_question_never_asks_to_remember_or_states_a_fact() -> None:
    assert classify_user_text("Do you remember the deployment?").remember is False
    assert classify_user_text("Do I prefer tea?").facts == ()
    assert classify_user_text("Do you remember the deployment?").facts == ()
    assert classify_user_text("Do you remember the deployment?").request_facts == ()


def test_a_meta_sentence_is_not_a_stable_fact() -> None:
    for text in (
        "My task is to fix the build.",
        "My question is how do I deploy?",
        "My current task is to fix the build.",
        "My plan is to read the file.",
        "我的任务是修复构建。",
        "我的计划是阅读文件。",
    ):
        signals = classify_user_text(text)
        assert signals.facts == (), text
        assert signals.request_facts == (), text
        assert signals.remember is False, text


def test_a_direct_user_statement_still_is_one() -> None:
    signals = classify_user_text("My birthday is 12 May 1990.")
    assert signals.facts == ("My birthday is 12 May 1990.",)
    assert classify_kind(signals.facts[0], source="user") == "profile"


def test_a_remember_request_states_the_fact_after_the_marker() -> None:
    signals = classify_user_text("Remember that the Bluebird launch code is NECTAR-42.")
    assert signals.remember is True
    assert signals.facts == ()
    assert signals.request_facts == ("the Bluebird launch code is NECTAR-42.",)
    assert classify_kind(signals.request_facts[0], source="user") == "semantic"


def test_a_remember_request_still_carries_its_fact() -> None:
    signals = classify_user_text("Remember that my timezone is CET.")
    assert signals.remember is True
    assert signals.facts == ()
    assert signals.request_facts == ("my timezone is CET.",)
    assert classify_kind(signals.request_facts[0], source="user") == "profile"


def test_a_remember_request_states_a_first_person_fact_as_profile() -> None:
    signals = classify_user_text("Remember that I have a dog named Rex.")
    assert signals.request_facts == ("I have a dog named Rex.",)
    assert classify_kind(signals.request_facts[0], source="user") == "profile"


def test_a_chinese_remember_request_carries_its_fact() -> None:
    signals = classify_user_text("记住，服务器地址是 10.0.0.5。")
    assert signals.remember is True
    assert signals.request_facts == ("服务器地址是 10.0.0.5。",)
    assert classify_kind(signals.request_facts[0], source="user") == "semantic"


def test_a_remember_request_refuses_a_meta_sentence_after_the_marker() -> None:
    for text in (
        "Remember that my task is to fix the build.",
        "记住，我的计划是阅读文件。",
    ):
        signals = classify_user_text(text)
        assert signals.remember is True, text
        assert signals.request_facts == (), text


def test_a_remember_request_to_do_carries_no_fact() -> None:
    signals = classify_user_text("Remember to buy milk.")
    assert signals.remember is True
    assert signals.facts == ()
    assert signals.request_facts == ()


@pytest.mark.parametrize(
    "text",
    [
        "Please remember where I work now and the years of my degree, then write it.",
        "Don't forget her email and where she works, I'll need both before the meeting.",
        "Remember that she prefers email.",
        "记住她的邮箱和她在哪里工作。",
    ],
)
def test_a_remember_request_that_points_elsewhere_carries_no_fact(text: str) -> None:
    signals = classify_user_text(text)
    assert signals.remember is True
    assert signals.facts == ()
    assert signals.request_facts == ()


@pytest.mark.parametrize(
    ("note", "alone"),
    [
        ("the launch code is NECTAR-42.", True),
        ("my sister Ana lives in Leeds and her birthday is May 3.", True),
        ("I'm allergic to peanuts.", True),
        ("Items are kept in the blue box.", True),
        ("我对花生过敏。", True),
        ("我姐姐住在利兹。", True),
        ("where I work now", False),
        ("What the room number is", False),
        ("whether she moved", False),
        ('"Who" is Ana', False),
        ("her email and where she works", False),
        ("he lives in Leeds.", False),
        ("They moved to Leeds.", False),
        ("its start time", False),
        ("to buy milk", False),
        ("哪里是我的车位", False),
        ("什么时候开会", False),
        ("她的邮箱", False),
        ("他们住在利兹。", False),
        ("它的地址", False),
        ("几点开始", False),
    ],
)
def test_a_note_stands_alone_unless_it_opens_by_pointing_elsewhere(
    note: str, alone: bool
) -> None:
    assert stands_alone(note) is alone


def test_a_request_note_cuts_the_marker_and_keeps_the_rest() -> None:
    assert request_note("Remember that I'm allergic to peanuts.") == (
        "I'm allergic to peanuts."
    )
    assert request_note("I live in Leeds, remember that.") == (
        "I live in Leeds, remember that."
    )
    assert request_note("She lives in Leeds, remember that.") == ""
    assert request_note("Remember to say that I live in Leeds.") == ""


def test_a_third_party_statement_is_not_a_user_fact() -> None:
    assert classify_user_text("Alice's address is 12 Main Street.").facts == ()
    assert classify_user_text("Alice's address is 12 Main Street.").request_facts == ()


def test_user_candidates_do_not_emit_a_meta_sentence() -> None:
    candidates = user_candidates("My task is to fix the build.", task_id="t1")
    assert candidates == []


def test_user_candidates_skip_a_plain_short_statement_and_a_question() -> None:
    text = "Mina lives in Paris. Does Mina live in Rome?"
    assert user_candidates(text, task_id="task") == []


def test_request_candidates_are_remember_candidates() -> None:
    candidates = request_candidates(
        "Remember that the Bluebird launch code is NECTAR-42.", task_id="t1"
    )
    assert [candidate.text for candidate in candidates] == [
        "the Bluebird launch code is NECTAR-42."
    ]
    assert candidates[0].source == "user"
    assert candidates[0].remember is True
    assert candidates[0].remember_by == "rule"
    assert candidates[0].provenance["task_id"] == "t1"
