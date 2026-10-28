"""The user signals: only a declarative, stable statement is a fact."""

from __future__ import annotations

from localmelo.melo.memory.hippo import classify_kind, classify_user_text
from localmelo.melo.memory.hippo.signals import user_candidates


def test_a_question_never_asks_to_remember_or_states_a_fact() -> None:
    assert classify_user_text("Do you remember the deployment?").remember is False
    assert classify_user_text("Do I prefer tea?").facts == ()
    assert classify_user_text("Do you remember the deployment?").facts == ()


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
        assert signals.remember is False, text


def test_a_direct_user_statement_still_is_one() -> None:
    signals = classify_user_text("My birthday is 12 May 1990.")
    assert signals.facts == ("My birthday is 12 May 1990.",)
    assert classify_kind(signals.facts[0], source="user") == "profile"


def test_a_remember_request_still_carries_its_fact() -> None:
    signals = classify_user_text("Remember that my timezone is CET.")
    assert signals.remember is True
    assert signals.facts == ("Remember that my timezone is CET.",)


def test_a_remember_request_to_do_carries_no_fact() -> None:
    assert classify_user_text("Remember to buy milk.").facts == ()


def test_a_third_party_statement_is_not_a_user_fact() -> None:
    assert classify_user_text("Alice's address is 12 Main Street.").facts == ()


def test_user_candidates_do_not_emit_a_meta_sentence() -> None:
    candidates = user_candidates("My task is to fix the build.", task_id="t1")
    assert candidates == []
