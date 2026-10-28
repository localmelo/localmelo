"""Facts-2 extraction and attribution regressions."""

from __future__ import annotations

import pytest

from localmelo.melo.memory.notes import (
    _item_source_ids,
    note_from_reply,
    split_sentences,
)


@pytest.mark.parametrize(
    "text",
    [
        "The meeting was cancelled on 4/9.",
        "The plumber is not coming on 4/9.",
        "The train no longer stops at Union Station.",
        "会议已取消。",
        "会议已取消！",
        "The meeting was cancelled on 4/9!",
        "小王不再住在北京。",
        "小李住在上海。",
    ],
)
def test_explicit_assertions_keep_original_polarity(text: str) -> None:
    note, _ = note_from_reply(text, writer="heuristic")
    assert text in note.found


@pytest.mark.parametrize(
    "text",
    [
        "Not on 4/9.",
        "Is the meeting cancelled on 4/9?",
        "The meeting may not happen on 4/9.",
        "If the meeting is cancelled on 4/9, call me.",
        "The room is not only large but also quiet at 9:00.",
        "The meeting was not cancelled on 4/9.",
        "The meeting is not not at 20:00.",
        "He said 'the meeting is cancelled on 4/9'.",
        "When the meeting is cancelled, call me.",
        "会议是否取消？",
        "如果会议取消，我们就去上海。",
        "小李不但聪明而且努力。",
    ],
)
def test_ambiguous_or_nonassertions_are_not_facts(text: str) -> None:
    note, _ = note_from_reply(text, writer="heuristic")
    assert not note.found


def test_mixed_punctuation_boundaries() -> None:
    text = "会议已取消。The meeting was cancelled!下一场在北京。"
    assert [sentence for _, sentence in split_sentences(text)] == [
        "会议已取消。",
        "The meeting was cancelled!",
        "下一场在北京。",
    ]


def test_negation_never_cites_an_affirmative_record() -> None:
    item = "The meeting was cancelled on 4/9."
    assert not _item_source_ids(item, [("wrong", "The meeting was held on 4/9.")])
    assert _item_source_ids(item, [("right", item)]) == ["right"]


def test_cjk_citation_requires_evidence_not_a_shared_date() -> None:
    item = "小王不再住在北京。"
    assert not _item_source_ids(item, [("wrong", "小王住在北京。")])
    assert _item_source_ids(item, [("right", item)]) == ["right"]


@pytest.mark.parametrize(
    "text", ["The meeting is on May 5.", "The offsite is May 5 at 9:00."]
)
def test_may_date_is_not_a_modal(text: str) -> None:
    assert note_from_reply(text, writer="heuristic")[0].found == [text]


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        ("The meeting is at 20:00.", "The meeting is not at 20:00."),
        ("The meeting is at 20:00.", "The meeting isn't at 20:00."),
        ("会议在20:00举行。", "会议不在20:00举行。"),
    ],
)
def test_citation_rejects_opposite_polarity_both_ways(
    positive: str, negative: str
) -> None:
    assert not _item_source_ids(positive, [("negative", negative)])
    assert not _item_source_ids(negative, [("positive", positive)])


def test_t4_concert_update_chooses_current_source() -> None:
    item = "Static Bloom concert doors moved to 21:00 on 2026-04-11."
    records = [
        ("old", "Static Bloom concert 2026-04-11 doors at 20:00."),
        ("new", "Static Bloom concert 2026-04-11 doors moved to 21:00."),
    ]
    note, _ = note_from_reply(item, writer="heuristic")
    assert note.found == [item]
    assert _item_source_ids(item, records) == ["new"]
    assert not _item_source_ids(item, records[:1])
