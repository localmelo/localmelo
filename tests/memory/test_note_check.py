"""What a note must find in the user's words: values exactly, wording loosely.

``records.unsupported(note, evidence)`` lists the content words of a note the
typed texts lack, or nothing when the note is grounded. Every value (a digit, a
weekday or number word, a capitalised word inside a sentence, a quoted span, a
Chinese character) must be found, and at least 80% of the other content words.
The cases are notes a chat's model really wrote for what the user typed.
"""

from __future__ import annotations

import pytest

from localmelo.support.memory import records
from localmelo.support.memory.recall import stem

PARKING = "My parking spot at work is C14."
SUMMARIES = (
    "I prefer that your summaries of long documents start with a one-line "
    "verdict, followed by a short list of the three most important points, and "
    "that you keep any background material, definitions and caveats until the "
    "very end of the reply, where I can skip them easily if I do not need them."
)
EXPLAIN = (
    "I prefer that when you explain a technical topic to me you choose plain "
    "everyday words and one small example done step by step before any formal "
    "definition, and that you tell me clearly at the start of the reply whether "
    "you are sure of the answer or only fairly confident about it."
)
LISTS = (
    "I prefer that every list you write for me has no more than six items, that "
    "dates are written out in full with the weekday, and that you never start a "
    "reply with an apology or a compliment, because I would rather get straight "
    "to the content of the answer."
)


@pytest.mark.parametrize(
    ("word", "same"),
    [
        ("replies", "reply"),
        ("summaries", "summary"),
        ("hobbies", "hobby"),
        ("movies", "movie"),  # "-ie" ends in y too, so the plural still finds it
        ("meets", "meet"),
        ("meeting", "meet"),
        ("tuesdays", "tuesday"),
        ("pies", "pie"),  # too short for the "-ies" rule: the plain plural
    ],
)
def test_the_stemmer_ends_ies_and_ie_in_y_and_keeps_the_old_endings(
    word: str, same: str
) -> None:
    assert stem(stem(word)) == stem(stem(same))


@pytest.mark.parametrize(
    ("typed", "note"),
    [
        # What the cell refused: "then" and "key" for "followed by" and "most
        # important" (one word in fifteen).
        (
            SUMMARIES,
            "You prefer summaries of long documents to start with a one-line "
            "verdict, then three key points, with background, definitions and "
            "caveats at the very end.",
        ),
        # "explanation" for "explain", "use" for "choose", "each" for "the".
        (
            EXPLAIN,
            "You prefer that technical explanations use plain everyday words with "
            "one small step-by-step example before any formal definition, and "
            "that each reply starts by stating whether you are sure or only "
            "fairly confident about the answer.",
        ),
        # "replies" against the user's "reply".
        (
            LISTS,
            "You prefer lists with no more than six items, dates written out in "
            "full with the weekday, and replies that start straight with the "
            "content — never an apology or a compliment.",
        ),
        (
            LISTS,
            "You prefer that every list has no more than six items, that dates "
            "are written out in full with the weekday, and that replies never "
            "start with an apology or a compliment.",
        ),
        # The user's own sentence, quoted ("said" restates).
        (
            LISTS,
            'You said: "I prefer that every list you write for me has no more '
            "than six items, that dates are written out in full with the "
            "weekday, and that you never start a reply with an apology or a "
            'compliment."',
        ),
    ],
)
def test_a_note_in_other_words_is_grounded_when_its_values_are_the_users(
    typed: str, note: str
) -> None:
    assert records.unsupported(note, [typed]) == []


@pytest.mark.parametrize(
    ("typed", "note", "missing"),
    [
        # A value changed: a digit, a capitalised word, a weekday, a number word.
        (PARKING, "Your parking spot at work is C15.", ["c15"]),
        ("My dentist is Lee Park.", "Your dentist is Sam Park.", ["sam"]),
        (
            "My weekly sync is on Wednesday.",
            "Your weekly sync is on thursday.",
            ["thursday"],
        ),
        (LISTS, "You prefer lists with no more than seven items.", ["seven"]),
        # A quoted span is exact.
        (LISTS, 'You said: "I prefer lists of ten items."', ["ten"]),
    ],
)
def test_a_note_with_a_changed_value_is_not_grounded(
    typed: str, note: str, missing: list[str]
) -> None:
    assert records.unsupported(note, [typed]) == missing


@pytest.mark.parametrize(
    "note",
    [
        "你的公司停车位是 C14。",  # another language than the user's
        "你的机场停车位是 L7。",
        "你的每周例会是在周五。",
    ],
)
def test_a_note_in_another_language_is_not_grounded(note: str) -> None:
    typed = [PARKING, "My weekly sync is on Friday."]
    missing = records.unsupported(note, typed)
    assert missing and all(not word.isascii() for word in missing[:3])
    assert records.other_script(missing, typed)


def test_most_of_the_other_words_must_be_found() -> None:
    typed = "I prefer plain words, short sentences and one example for every answer."
    note = "You prefer plain words, short sentences and one example."
    brief = note.replace("short", "brief")
    assert records.unsupported(note, [typed]) == []
    # Four of the five other words found (80%): the wording may vary.
    assert records.unsupported(brief, [typed]) == []
    # Three of five: refused, naming what is missing.
    easy = brief.replace("plain", "easy")
    assert records.unsupported(easy, [typed]) == ["easy", "brief"]
    # A short note has no word to spare.
    assert records.unsupported("You prefer brief answers.", [typed]) == ["brief"]


def test_the_words_of_a_restating_note_are_not_looked_for() -> None:
    for word in ("said", "told", "then", "each", "use", "also", "always", "every"):
        assert word not in records.note_terms(f"You {word} short answers.")
    # A note of nothing but such words keeps nothing.
    assert records.note_terms("You said that.") == []


def test_the_script_hint_needs_a_script_the_user_never_typed() -> None:
    assert records.other_script(["公"], ["My bike is red."])
    assert records.other_script(["bike"], ["我的自行车是红色的。"])
    assert not records.other_script(["bike"], ["My car is red."])
    assert not records.other_script(["公"], ["我的公司在北京。"])
    assert not records.other_script(["14"], ["我的车位是14。"])


@pytest.mark.parametrize(
    ("typed", "note"),
    [
        ("My piano teacher is Mr. Bellamy.", "Your piano teacher is Mr. Bellamy."),
        (
            "My thesis reader is Prof. Ines Duarte.",
            "Your thesis reader is Prof. Ines Duarte.",
        ),
        ("My vet is Dr. Yuki Tanaka.", "Your vet is Dr. Yuki Tanaka."),
        ("My advisor is J. Smith.", "Your advisor is J. Smith."),
        ("My team is Reds vs. Blues.", "Your team is Reds vs. Blues."),
    ],
)
def test_an_abbreviation_or_an_initial_does_not_end_the_quoted_sentence(
    typed: str, note: str
) -> None:
    assert records.quote_for(note, [typed]) == typed
    assert records.quote_for(note, [f"I like tea. {typed} Thanks."]) == typed


@pytest.mark.parametrize(
    "typed",
    [
        "I like tea. My name is Ray.",
        "So do I. My name is Ray.",
        "I live in the U.S. My name is Ray.",
        "I like apples, pears, etc. My name is Ray.",
        "I like tea; my name is Ray.",
    ],
)
def test_a_text_of_two_sentences_still_splits_for_the_quote(typed: str) -> None:
    assert records.quote_for("Your name is Ray.", [typed]) == (
        "my name is Ray." if ";" in typed else "My name is Ray."
    )


def test_a_request_with_a_title_is_kept_whole() -> None:
    (cand,) = records.candidates("Remember that my piano teacher is Mr. Bellamy.")[0]
    assert cand.text == "Your piano teacher is Mr. Bellamy."
    assert cand.quote == "my piano teacher is Mr. Bellamy."
