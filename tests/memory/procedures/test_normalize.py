"""Recognising a recurring task: normalized queries, keys, template ids, similar."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from localmelo.melo.memory.procedures.normalize import (
    DATE,
    LENGTH_OPTIONS,
    MAX_QUERY_TEXT,
    SIMILAR_MIN_ROUTINES,
    SimilarMatch,
    canonical_json,
    find_similar,
    key_for_query,
    key_for_routine,
    known_model,
    length_options,
    normalize_query,
    routine_key,
    run_place,
    stored_query_text,
    template_id,
)

# ── normalize_query ─────────────────────────────────────────────────────────


def test_nfkc_lowercase_and_whitespace() -> None:
    assert normalize_query("  Ｆｉｘ\tThe\n\nTESTS  ") == "fix the tests"


@pytest.mark.parametrize(
    "query",
    [
        "report for 2026-10-26",
        "report for 2026/10/26",
        "report for 10/26",
        "report for 10/26/2026",
        "report for 26.10.2026",
        "report for October 26",
        "report for Oct. 26th, 2026",
        "report for 26 October 2026",
        "report for october 2026",
        "report for Monday",
        "report for 2026年",
        "report for 10月26日",
        "report for 2026年10月26日",
        "report for 星期一",
        "report for 周一",
    ],
)
def test_each_date_form_is_one_mask(query: str) -> None:
    assert normalize_query(query) == f"report for {DATE}"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("sync at 09:30", "sync at <time>"),
        ("sync at 9:30:15", "sync at <time>"),
        ("sync at 9:30 pm", "sync at <time>"),
        ("sync at 23:59", "sync at <time>"),
    ],
)
def test_times_are_masked(query: str, expected: str) -> None:
    assert normalize_query(query) == expected


def test_two_days_of_a_daily_routine_normalize_alike() -> None:
    monday = "Summarize today's unread mail (Monday, October 26) at 08:00"
    tuesday = "Summarize today's unread mail (Tuesday, October 27) at 08:15"
    assert normalize_query(monday) == normalize_query(tuesday)


def test_other_numbers_stay() -> None:
    assert normalize_query("fix issue 123") != normalize_query("fix issue 456")
    assert normalize_query("run 3 times") == "run 3 times"
    # A version number is not a date.
    assert normalize_query("python 3.11") == "python 3.11"


def test_a_number_glued_to_a_date_is_not_a_date() -> None:
    assert normalize_query("id 2026-10-261") == "id 2026-10-261"


# ── keys ────────────────────────────────────────────────────────────────────


def test_routine_key_is_the_sha256_of_text_nul_place() -> None:
    expected = hashlib.sha256(b"fix the tests\x00/work").hexdigest()
    assert routine_key("fix the tests", "/work") == expected


def test_the_same_words_in_another_place_are_another_routine() -> None:
    here = key_for_query("Fix the tests", "/a")
    there = key_for_query("Fix the tests", "/b")
    assert here.key != there.key
    assert here.query_text == there.query_text == "fix the tests"
    assert key_for_query("fix  THE tests", "/a").key == here.key


def test_a_named_routine_is_one_routine_in_any_folder() -> None:
    first = key_for_routine("daily-mail", "Summarize the mail")
    second = key_for_routine(" daily-mail ", "Mail summary please")
    assert first.key == second.key == routine_key("routine:daily-mail", "")
    assert first.place == ""
    assert first.name == "daily-mail"
    assert first.query_text == "summarize the mail"


def test_a_blank_routine_name_is_refused() -> None:
    with pytest.raises(ValueError, match="blank"):
        key_for_routine("  ")


def test_the_stored_query_text_is_redacted_and_cut() -> None:
    secret = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    text = stored_query_text(f"use {secret} " + "x" * 3000)
    assert secret not in text
    assert "[redacted]" in text
    assert len(text) == MAX_QUERY_TEXT


def test_the_key_hashes_the_query_not_its_redacted_copy() -> None:
    a = key_for_query("token sk-" + "a" * 24, "/p")
    b = key_for_query("token sk-" + "b" * 24, "/p")
    assert a.query_text == b.query_text
    assert a.key != b.key


def test_run_place_is_the_root_or_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert run_place(str(link)) == os.path.realpath(real)
    monkeypatch.chdir(link)
    assert run_place(None) == os.path.realpath(real)
    assert run_place("") == os.path.realpath(real)


@pytest.mark.parametrize(
    ("model", "known"),
    [("", False), ("default", False), (" default ", False), ("qwen-27b", True)],
)
def test_known_model(model: str, known: bool) -> None:
    assert known_model(model) is known


def test_length_options_are_the_five_config_keys() -> None:
    config = {
        "max_steps": 30,
        "attempt_policy": "reflect",
        "steps_per_attempt": 10,
        "reflection_gate": "flags",
        "timeout_s": 600,
        "tools": ["shell_exec"],
    }
    options = length_options(config)
    assert tuple(options) == LENGTH_OPTIONS
    assert options["max_attempts"] is None
    assert "timeout_s" not in options


def test_template_id_hashes_model_and_options() -> None:
    options = {"max_steps": 30, "attempt_policy": "reflect"}
    tid = template_id("r1", "m", options)
    digest = hashlib.sha256(f"m\0{canonical_json(options)}".encode()).hexdigest()
    assert tid == f"r1:{digest[:12]}"
    # Key order does not matter; the model and every option do.
    assert template_id("r1", "m", dict(reversed(options.items()))) == tid
    assert template_id("r1", "m2", options) != tid
    assert template_id("r1", "m", {**options, "max_steps": 20}) != tid
    assert template_id("r2", "m", options).startswith("r2:")


# ── find_similar ────────────────────────────────────────────────────────────

ROUTINES = [
    ("r-mail", "summarize today's unread mail into the daily notes file"),
    ("r-tests", "run pytest in the project and fix the failing tests"),
    ("r-csv", "convert the sales csv export into a json report"),
    ("r-backup", "copy the photos folder to the backup drive"),
    ("r-logs", "delete log files older than a week from the logs directory"),
]


def test_a_reworded_query_matches_its_routine() -> None:
    match = find_similar(
        "summarize the unread mail of today into the daily notes", ROUTINES
    )
    assert isinstance(match, SimilarMatch)
    assert match.routine_id == "r-mail"
    assert match.ratio >= 0.6
    assert match.second_ratio <= 0.5


def test_an_unrelated_query_matches_nothing() -> None:
    assert find_similar("write a haiku about autumn leaves", ROUTINES) is None


def test_fewer_than_five_routines_never_match() -> None:
    assert SIMILAR_MIN_ROUTINES == 5
    query = "summarize today's unread mail into the daily notes file"
    assert find_similar(query, ROUTINES[:4]) is None
    assert find_similar(query, ROUTINES) is not None


def test_two_close_routines_are_ambiguous() -> None:
    routines = [*ROUTINES, ("r-mail-2", "summarize today's unread mail into notes")]
    match = find_similar(
        "summarize today's unread mail into the daily notes file", routines
    )
    # The second is more than half as good as the best: no match.
    assert match is None


def test_a_blank_query_matches_nothing() -> None:
    assert find_similar("   ", ROUTINES) is None
    assert find_similar("!!!", ROUTINES) is None


def test_the_self_score_counts_the_query_in_the_index() -> None:
    # A query identical to a stored text scores as well as itself: ratio 1.
    match = find_similar(ROUTINES[1][1], ROUTINES)
    assert match is not None
    assert match.routine_id == "r-tests"
    assert match.ratio == pytest.approx(1.0)


def test_a_routine_id_like_the_probe_does_not_collide() -> None:
    routines = [*ROUTINES, ("\0query", "something else entirely")]
    match = find_similar(ROUTINES[0][1], routines)
    assert match is not None and match.routine_id == "r-mail"
