"""The typed note rule on labelled sets of restatements and changed notes (#442).

``tests/memory/data/note_gate_restate.json`` (the set the rule was shaped on) and
``note_gate_heldout.json`` (written afterwards, before the share was changed, and not
edited since) hold notes with the texts the user typed and a label: ``restate`` (the
note says the same in other words, admit); ``swap`` (the same with one detail
changed), ``added`` (a detail the user never said) and ``inferred`` (a conclusion the
user did not state) are refused; ``crosslang`` (a note in the other language) is
reported by ``scripts/eval_note_gate.py`` and not scored here. The cases the rule gets
wrong are listed below so that a change in either direction is a decision, the rates
are floors, and the targets the rule does not reach are strict ``xfail`` tests.

The rule: a Chinese or Japanese term is a value (it must be found) when it sits in
a quoted span, holds a number character, or holds a character that no typed text has
and that is no common wording; every other CJK term is wording. The share of content
words that must be found counts words and single CJK characters, not pairs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from localmelo.support.memory import records

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
CASES: list[dict[str, Any]] = json.loads(
    (DATA / "note_gate_restate.json").read_text("utf-8")
)
HELDOUT: list[dict[str, Any]] = json.loads(
    (DATA / "note_gate_heldout.json").read_text("utf-8")
)
BY_ID = {case["id"]: case for case in [*CASES, *HELDOUT]}
INCIDENT = (
    "按照我说的用微信登录：点「微信登录」，等一会儿会出现「微信快捷登录」，"
    "点它就登进去了，然后完成签到"
)

# Restatements the rule still refuses: a noun the note added (床, 左右, 名叫, 重新尝试,
# 一律) or so many added or swapped characters that fewer than 80% of the note's
# characters are found. The English and Japanese ones are refused as they were before.
REFUSED_RESTATEMENTS = {
    "restate-zh-04",
    "restate-zh-06",
    "restate-zh-09",
    "restate-zh-11",
    "restate-zh-13",
    "restate-zh-17",
    "restate-zh-24",
    "restate-zh-33",
    "restate-en-04",
    "restate-en-05",
    "restate-en-10",
    "restate-en-11",
    "restate-en-13",
    "restate-ja-01",
    "restate-ja-03",
}
# Changed notes the rule admits: an address (its English words keep the English
# slack) and two hours exchanged (every pair of characters is one the user typed).
ADMITTED_CHANGES = {"swap-zh-05", "swap-zh-30"}
# The held-out set: the wording the rule cannot tell from a detail ("起床" for "醒来
# 后", "左右" for "大概", "主题" for "题目", "已经", "特别", "大约", "发送"), and
# several added or swapped characters at once.
HELDOUT_REFUSED_RESTATEMENTS = {
    f"ho-restate-zh-{n:02d}"
    for n in (1, 2, 5, 6, 7, 11, 13, 18, 20, 22, 27, 34, 36, 38, 39, 40)
}
# Changed notes it admits: a name that shares a character and a second one made of
# characters typed (团团), two typed values exchanged (times, weekdays, cities), and
# a name after "Mr." (a title ends the sentence for the capital-letter rule).
HELDOUT_ADMITTED_CHANGES = {
    "ho-swap-zh-09",
    "ho-swap-zh-23",
    "ho-swap-zh-24",
    "ho-swap-zh-25",
    "ho-swap-en-04",
}


@pytest.fixture(scope="module")
def script() -> Iterator[ModuleType]:
    """``scripts/eval_note_gate.py``, which holds the arms the rule is compared with."""
    spec = importlib.util.spec_from_file_location(
        "eval_note_gate", ROOT / "scripts" / "eval_note_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_note_gate"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("eval_note_gate", None)


def admitted(case: dict[str, Any]) -> bool:
    return not records.unsupported(case["note"], case["said"])


def old_unsupported(note: str, said: list[str]) -> list[str]:
    """:func:`records.unsupported` as it was: every CJK term is a value."""
    original = records.value_terms
    records.value_terms = lambda note, evidence=None: original(note)  # type: ignore[assignment]
    try:
        return records.unsupported(note, said)
    finally:
        records.value_terms = original  # type: ignore[assignment]


def of(
    kind: str, *, lang: str | None = None, cases: list[dict[str, Any]] = CASES
) -> list[dict[str, Any]]:
    return [
        c for c in cases if c["kind"] == kind and (lang is None or c["lang"] == lang)
    ]


def rate(
    group: list[dict[str, Any]], rule: Any = None, *, refused: bool = False
) -> float:
    """The share of *group* that *rule* (default: the shipped one) admits, or with
    *refused* the share it refuses."""
    rule = rule or records.unsupported
    hits = sum(bool(rule(c["note"], c["said"])) == refused for c in group)
    return hits / len(group)


def test_the_set_is_labelled_and_big_enough() -> None:
    kinds = Counter((c["kind"], c["lang"]) for c in CASES)
    assert len(CASES) >= 110 and len({c["id"] for c in CASES}) == len(CASES)
    assert kinds["restate", "zh"] >= 30 and kinds["restate", "en"] >= 15
    assert kinds["swap", "zh"] >= 18 and kinds["swap", "en"] >= 7
    assert len(of("added")) >= 15 and len(of("inferred")) >= 10
    assert len(of("crosslang")) >= 10
    for case in CASES:
        want = "admit" if case["kind"] == "restate" else "refuse"
        if case["kind"] != "crosslang" and case["id"] != "incident-zh-3":
            assert case["label"] == want, case["id"]
        assert case["label"] in ("admit", "refuse") and case["said"] and case["note"]
    for case in of("swap"):  # a swap is a restatement with one detail changed
        base = BY_ID[case["base"]]
        assert base["said"] == case["said"] and base["note"] != case["note"]


def test_the_incident_notes_are_kept_and_the_one_naming_a_new_site_is_not() -> None:
    incident = [c for c in CASES if c["id"].startswith("incident-zh-")]
    assert len(incident) == 5 and all(c["said"] == [INCIDENT] for c in incident)
    for case in incident:
        if case["label"] == "admit":
            assert records.unsupported(case["note"], case["said"]) == [], case["id"]
            assert old_unsupported(case["note"], case["said"]), case[
                "id"
            ]  # was refused
    added = BY_ID["incident-zh-3"]
    assert added["label"] == "refuse" and added["kind"] == "added"
    missing = records.unsupported(added["note"], added["said"])
    # The details come first: the characters of the site the user never typed.
    assert missing[:3] == ["亩", "三", "分"]


def test_most_chinese_restatements_are_admitted() -> None:
    zh = of("restate", lang="zh")
    assert rate(zh) >= 0.8  # 20% when every character was a value, 76% by pairs
    assert rate(zh, lambda n, s: old_unsupported(n, s)) < 0.25


@pytest.mark.xfail(strict=True, reason="the rule admits 84% of them; the aim is 90%")
def test_nine_in_ten_chinese_restatements_are_admitted() -> None:
    assert rate(of("restate", lang="zh")) >= 0.9


def test_the_restatements_still_refused_are_the_known_ones() -> None:
    refused = {c["id"] for c in of("restate") if not admitted(c)}
    assert refused == REFUSED_RESTATEMENTS


def test_a_changed_detail_is_refused_but_for_the_known_gaps() -> None:
    swaps = of("swap")
    assert {c["id"] for c in swaps if admitted(c)} == ADMITTED_CHANGES
    zh = [c for c in of("swap", lang="zh") if c["id"] not in ADMITTED_CHANGES]
    assert len(zh) >= 25 and not any(admitted(c) for c in zh)
    assert not any(admitted(c) for c in of("swap", lang="en"))


@pytest.mark.xfail(strict=True, reason="two changed notes are still admitted")
def test_every_changed_detail_is_refused() -> None:
    assert not any(admitted(c) for c in of("swap"))


def test_the_heldout_set_is_new_labelled_and_big_enough() -> None:
    kinds = Counter((c["kind"], c["lang"]) for c in HELDOUT)
    assert len(HELDOUT) >= 75 and len({c["id"] for c in HELDOUT}) == len(HELDOUT)
    assert kinds["restate", "zh"] >= 35 and kinds["swap", "zh"] >= 20
    assert kinds["added", "zh"] >= 8 and kinds["inferred", "zh"] >= 5
    assert kinds["swap", "en"] >= 7
    first = {c["note"] for c in CASES} | {t for c in CASES for t in c["said"]}
    for case in HELDOUT:
        assert case["note"] not in first and not first.intersection(case["said"])
        want = "admit" if case["kind"] == "restate" else "refuse"
        if case["kind"] != "crosslang":
            assert case["label"] == want, case["id"]
    for case in of("swap", cases=HELDOUT):
        base = BY_ID[case["base"]]
        assert base["said"] == case["said"] and base["note"] != case["note"]
    # The rule's own examples of detail-looking wording, and of swapped typed values.
    assert any("起床" in c["note"] and "醒来后" in c["said"][0] for c in HELDOUT)
    assert any("左右" in c["note"] and "大概" in c["said"][0] for c in HELDOUT)


def test_the_heldout_restatements_still_refused_are_the_known_ones() -> None:
    refused = {c["id"] for c in of("restate", cases=HELDOUT) if not admitted(c)}
    assert refused == HELDOUT_REFUSED_RESTATEMENTS


def test_a_changed_detail_in_the_heldout_set_is_refused_but_for_the_known_gaps() -> (
    None
):
    wrong = {
        c["id"]
        for kind in ("swap", "added", "inferred")
        for c in of(kind, cases=HELDOUT)
        if admitted(c)
    }
    assert wrong == HELDOUT_ADMITTED_CHANGES
    assert rate(of("swap", lang="zh", cases=HELDOUT), refused=True) >= 0.85
    assert rate(of("added", lang="zh", cases=HELDOUT), refused=True) == 1.0
    assert rate(of("inferred", lang="zh", cases=HELDOUT), refused=True) == 1.0
    assert rate(of("swap", lang="en", cases=HELDOUT), refused=True) >= 0.9


def test_the_heldout_chinese_restatements_are_admitted_at_a_floor() -> None:
    zh = of("restate", lang="zh", cases=HELDOUT)
    assert rate(zh) >= 0.6
    assert rate(zh, lambda n, s: old_unsupported(n, s)) < 0.3


@pytest.mark.xfail(strict=True, reason="62% on the held-out set; the aim is 90%")
def test_nine_in_ten_heldout_chinese_restatements_are_admitted() -> None:
    assert rate(of("restate", lang="zh", cases=HELDOUT)) >= 0.9


@pytest.mark.xfail(strict=True, reason="five changed notes are still admitted")
def test_every_changed_detail_in_the_heldout_set_is_refused() -> None:
    assert not any(admitted(c) for c in of("swap", cases=HELDOUT))


def test_counting_characters_beats_counting_pairs_on_the_heldout_set(
    script: ModuleType,
) -> None:
    """Why the share counts single characters (the rule that decided it): on the set
    it was not shaped on, more Chinese restatements are admitted and no more changed
    or added notes are."""
    zh = of("restate", lang="zh", cases=HELDOUT)
    assert rate(zh, script.typed_t1b) > rate(zh, script.typed_t1)
    for kind in ("swap", "added"):
        group = of(kind, lang="zh", cases=HELDOUT)
        assert rate(group, script.typed_t1b, refused=True) >= rate(
            group, script.typed_t1, refused=True
        )


def test_the_shipped_rule_is_the_scripts_character_share(script: ModuleType) -> None:
    for case in [*CASES, *HELDOUT]:
        assert records.unsupported(case["note"], case["said"]) == script.typed_t1b(
            case["note"], case["said"]
        ), case["id"]
    # T0 is the rule before CJK wording; a pair added by a reordering costs nothing now.
    case = BY_ID["restate-zh-03"]
    assert script.typed_t1(case["note"], case["said"])  # refused by pairs
    assert records.unsupported(case["note"], case["said"]) == []


@pytest.mark.parametrize("kind", ["added", "inferred"])
def test_a_detail_or_a_conclusion_the_user_did_not_give_is_refused(kind: str) -> None:
    group = of(kind)
    assert group and not any(admitted(c) for c in group), [
        c["id"] for c in group if admitted(c)
    ]


def test_english_is_exactly_as_it_was() -> None:
    plain = [
        c
        for c in [*CASES, *HELDOUT]
        if all(t.isascii() for t in [c["note"], *c["said"]])
    ]
    assert len(plain) >= 60
    for case in plain:
        assert records.unsupported(case["note"], case["said"]) == old_unsupported(
            case["note"], case["said"]
        ), case["id"]


@pytest.mark.parametrize(
    ("case_id", "missing"),
    [
        ("restate-en-01", []),
        ("restate-en-04", ["do", "not", "include"]),
        ("restate-en-10", ["year", "old", "attend"]),
        ("swap-en-01", ["7"]),
        ("swap-en-03", ["rao"]),
        ("swap-en-05", ["rotterdam", "take"]),
        ("swap-en-08", ["thursday"]),
        ("added-en-03", ["hate", "treadmill"]),
        ("added-en-05", ["philip", "work"]),
        ("inferred-en-02", ["insomnia"]),
    ],
)
def test_english_words_listed_as_missing_are_the_same_as_before(
    case_id: str, missing: list[str]
) -> None:
    case = BY_ID[case_id]
    assert records.unsupported(case["note"], case["said"]) == missing


def test_wording_may_be_added_to_what_the_user_typed() -> None:
    # 让 was the whole of the incident's refusal; 要求, 到时 and 能 were the others.
    for word in ("让", "要求", "到时", "能"):
        note = f"你{word}melo按照说的用微信登录：点「微信登录」，等一会儿会出现「微信快捷登录」，点它就登进去了，然后完成签到"
        assert records.unsupported(note, [INCIDENT]) == [], word


def test_a_character_the_user_never_typed_is_a_detail() -> None:
    typed = ["我现在住在苏州，通勤坐地铁大概四十分钟。"]
    # 杭 is new and no wording; 州 is the user's. The detail is named first.
    gone = records.unsupported("你住在杭州，坐地铁通勤大概四十分钟。", typed)
    assert gone[:2] == ["杭", "杭州"]
    assert "州" not in gone
    assert records.unsupported("你住在苏州，坐地铁通勤大概四十分钟。", typed) == []


def test_a_number_beside_a_new_neighbour_is_a_detail_but_not_beside_wording() -> None:
    typed = ["周二下午三点开会，周四上午十点复诊。"]
    # 三 is typed, but 周三 is not: weekday and hour are held as typed.
    assert records.unsupported("你周三下午三点开会，周四上午十点复诊。", typed) == [
        "周三",
        "三下",
    ]
    # 会 is wording: a note may put it beside a number.
    assert records.unsupported("你周二下午三点会开会，周四上午十点复诊。", typed) == []


def test_a_quoted_span_must_be_found_whole_even_in_common_wording() -> None:
    for opening, closing in (("「", "」"), ("『", "』"), ("“", "”"), ('"', '"')):
        note = f"你点了{opening}再来{closing}。"
        assert records.unsupported(note, ["我点了再看看。"])  # 来 is not typed
        assert records.value_terms(note, ["我点了再来"]) >= {"再", "来", "再来"}
    # Outside quotes the same characters are wording, not values.
    assert records.value_terms("你点了再来。", ["我点了再看看。"]) == set()


def test_without_evidence_every_cjk_term_is_a_value() -> None:
    # The sleep pass asks for the values of a note with no typed text to look in.
    note = "你让我再来"
    assert records.value_terms(note) == {"让", "再", "来", "再来"}
    assert records.value_terms(note, [note]) == set()


def test_the_wording_set_is_small_and_leaves_out_details() -> None:
    wording = records._CJK_WORDING
    assert 100 <= len(wording) <= 200
    assert not wording & records._CJK_FILLER
    assert not wording & records._CJK_NUMBERS
    for char in "州山江京市省区路街村湖海":  # places
        assert char not in wording, char
    for char in "王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗":  # surnames
        assert char not in wording, char
    for char in "不没无未别非一二三四五六七八九十":  # negations and numbers
        assert char not in wording, char


def test_the_evaluation_script_runs_without_a_model(script: ModuleType) -> None:
    for cases in (CASES, HELDOUT):
        results = script.run(cases, None, 0.55)
        assert list(results) == ["T0", "T1", "T1b"]
        table = "\n".join(script.table(cases, results))
        assert "| restate | zh |" in table and "base admitted" in table
        assert script.mistakes(cases, results)
    incident = BY_ID["incident-zh-1"]["id"]
    results = script.run(CASES, None, 0.55)
    assert not results["T0"][incident].admitted and results["T1b"][incident].admitted


def test_the_candidate_meaning_checks_in_the_script_read_numbers_titles_and_values(
    script: ModuleType,
) -> None:
    # A number word is a key token, held by the word or its digits.
    assert script.unheld_keys("You have seven cats.", ["I have six cats."]) == ["seven"]
    assert script.unheld_keys("You have six cats.", ["I have 6 cats."]) == []
    assert script.unheld_keys("You have 6 cats.", ["I have six cats."]) == []
    # A name after a title is a key token.
    typed = ["My doctor is Dr. Lindqvist."]
    assert script.unheld_keys("Your doctor is Dr. Lindgren.", typed) == ["Lindgren"]
    assert script.unheld_keys("Your doctor is Dr. Lindqvist.", typed) == []
    # A Chinese value is found or the note is refused; another script has none to find.
    said = ["我住在苏州，坐地铁通勤。"]
    assert script.unheld_cjk_values("你住在杭州，坐地铁通勤。", said)[:2] == [
        "杭",
        "杭州",
    ]
    assert script.unheld_cjk_values("你住在苏州，会坐地铁通勤。", said) == []
    assert script.unheld_cjk_values("你住在苏州。", ["I live in Suzhou."]) == []
