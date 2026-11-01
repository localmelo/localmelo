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
    ("word", "same"),
    [
        ("running", "run"),  # a doubled consonant the ending brought drops one
        ("stopped", "stop"),
        ("sitting", "sit"),
        ("planned", "plan"),
        ("falling", "fall"),  # l, s and z stay: they are in the bare word
        ("missing", "miss"),
        ("buzzing", "buzz"),
        ("stuffed", "stuff"),  # f and d stay too ("stuff", "add")
        ("adding", "add"),
        ("added", "add"),
        ("called", "call"),
        ("meeting", "meet"),
    ],
)
def test_the_stemmer_drops_the_doubled_consonant_an_ending_brought(
    word: str, same: str
) -> None:
    assert stem(stem(word)) == stem(stem(same))
    assert stem(word) == same


def test_a_running_coach_is_grounded_in_what_the_user_typed() -> None:
    note = "Ms. Petrova is your running coach."
    # "running" in the note, "run" typed: one term, not a word the user lacks.
    assert records.unsupported(note, ["Ms. Petrova is my coach for the run."]) == []
    assert records.unsupported(note, ["Ms. Petrova is my running coach."]) == []


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


@pytest.mark.parametrize(
    ("note", "typed", "name", "wording"),
    [
        # The case from the follow-up cell: Marco's studio became the user's.
        ("你的工作室在中山路88号。", "马可的工作室在中山路88号。", "马可", "马可的…"),
        (
            "Your studio is on Main Street.",
            "Marco's studio is on Main Street.",
            "Marco",
            "Marco's …",
        ),
        (
            "Your office is in the Atrium Wing.",
            "Prof. Duarte's office is in the Atrium Wing.",
            "Prof. Duarte",
            "Prof. Duarte's …",
        ),
        (
            "Your phone number is 555-0147.",
            "Tom Briggs's phone number is 555-0147.",
            "Tom Briggs",
            "Tom Briggs's …",
        ),
        # A reduplicated nickname and a relative are people without a surname.
        ("你的猫叫豆豆。", "思思的猫叫豆豆。", "思思", "思思的…"),
        ("你的猫叫豆豆。", "小明的猫叫豆豆。", "小明", "小明的…"),
        ("你的生日是5月3日。", "妈妈的生日是5月3日。", "妈妈", "妈妈的…"),
    ],
)
def test_a_note_about_someone_else_may_not_become_one_about_the_user(
    note: str, typed: str, name: str, wording: str
) -> None:
    assert records.wrong_owner(note, typed) == (
        f"Not remembered: that is about {name}, not the user. "
        f"Write the note about them: “{wording}”."
    )


@pytest.mark.parametrize(
    ("note", "typed"),
    [
        ("马可的工作室在中山路88号。", "马可的工作室在中山路88号。"),
        ("Marco's studio is on Main Street.", "Marco's studio is on Main Street."),
        ("Your studio is on Main Street.", "My studio is on Main Street."),
        ("Your sister's car is red.", "My sister's car is red."),
        ("你的室友阿杰的老家在宜昌。", "我的室友阿杰的老家在宜昌。"),
        ("阿杰的老家在宜昌。", "我的室友阿杰的老家在宜昌。"),
        ("Anna's car is red.", "My sister Anna's car is red."),
        # Not a possessor: a contraction, a time word, a clause before 的.
        ("Your parking spot is C14.", "That's why my parking spot is C14."),
        ("Your advisor is John.", "John's my advisor."),
        ("Your meeting is at 3pm.", "Tomorrow's meeting is at 3pm."),
        ("你的周末计划是爬山。", "周末的计划是爬山。"),
        ("你的家乡是宜昌。", "我爱吃的东西是面条。"),
        # No possessor at all.
        ("You live in Utrecht.", "I live in Utrecht."),
    ],
)
def test_the_same_owner_or_none_is_no_mix_up(note: str, typed: str) -> None:
    assert records.wrong_owner(note, typed) == ""


@pytest.mark.parametrize(
    ("note", "typed"),
    [
        # What comes before 的 or 's is a modifier, a language or a tool, not a person.
        ("你的代码一律用英文。", "开发的代码一律用英文。"),
        ("你的环境变量写进 .envrc。", "项目需要的环境变量写进 .envrc。"),
        ("你的服务一律用本地模型。", "本地的服务一律用本地模型。"),
        ("你的模型默认用小一点的。", "默认的模型默认用小一点的。"),
        ("Your GIL is going away.", "Python's GIL is going away."),
        ("Your index is in .git.", "Git's index is in .git."),
        ("Your plan is ready.", "Everything's ready."),
        ("你的代码放在 src 下。", "小程序的代码放在 src 下。"),
        ("你的版本放在 src 下。", "老版本的版本放在 src 下。"),
    ],
)
def test_a_rule_about_the_work_is_not_refused_as_someone_elses_fact(
    note: str, typed: str
) -> None:
    assert records.wrong_owner(note, typed) == ""


def test_the_owner_of_a_sentence_is_the_user_a_person_or_none() -> None:
    assert records._owner("你的代码一律用英文") == "user"
    assert records._owner("Your GIL is going away.") == "user"
    assert records._owner("马可的工作室在中山路88号") == "马可"
    assert records._owner("Marco's studio is on Main St") == "Marco"
    assert records._owner("Dr. Tanaka's office is in room 4") == "Dr. Tanaka"
    assert records._owner("思思的猫叫豆豆") == "思思"
    assert records._owner("妈妈的生日是5月3日") == "妈妈"
    assert records._owner("小明的猫叫豆豆") == "小明"
    assert records._owner("小程序的代码放在 src 下") is None
    assert records._owner("开发的代码一律用英文") is None
    assert records._owner("项目需要的环境变量写进 .envrc") is None
    assert records._owner("Python's GIL is going away") is None
    assert records._owner("GitHub's API is rate limited") is None


def test_a_given_name_without_a_surname_is_not_an_owner_so_its_mix_up_passes() -> None:
    # The trade-off of the owner check (see ``records._owner``): this is
    # let through to keep the user's own rules from being refused.
    assert records._owner("嘉怡的工作室在中山路88号") is None
    assert (
        records.wrong_owner("你的工作室在中山路88号。", "嘉怡的工作室在中山路88号。")
        == ""
    )


def test_the_user_typed_about_themselves_and_the_note_names_someone_else() -> None:
    refusal = records.wrong_owner(
        "Marco's studio is on Main Street.", "My studio is on Main Street."
    )
    assert refusal.startswith("Not remembered: the user said that about themselves")
    assert "not about Marco" in refusal and "“Your …”" in refusal
