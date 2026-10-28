"""Hippo's five actions, its kinds and its v1 thresholds."""

from __future__ import annotations

from typing import Any

from localmelo.melo.memory.hippo import (
    CHANGE_MARKERS,
    REMEMBER_MARKERS,
    SEMANTIC_ATTRIBUTES,
    USER_STATEMENT_MARKERS,
    classify_kind,
    decide,
    normalise,
    overlap,
)
from localmelo.melo.schema import MemoryCandidate


def _entry(
    entry_id: int,
    text: str,
    *,
    admitted_by: str = "",
    created_at: float = 1.0,
    task_text: str = "",
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "text": text,
        "kind": "episodic",
        "admitted_by": admitted_by,
        "created_at": created_at,
        "provenance": {"task_text": task_text},
    }


def _decision(
    text: str,
    *,
    kind: str = "episodic",
    source: str = "reasoning_note",
    active: list[dict[str, Any]] | None = None,
    task_text: str = "",
    first_person: bool | None = None,
) -> Any:
    return decide(
        text,
        kind=kind,
        source=source,
        provenance={"task_text": task_text},
        active=active or [],
    )


def test_a_duplicate_is_ignored_and_its_target_is_the_compared_entry() -> None:
    active = [_entry(7, "the project uses PostgreSQL for the analytics store")]
    decision = _decision(active[0]["text"], active=active)
    assert decision.action == "IGNORE"
    assert decision.target_id == 7


def test_an_overlap_of_at_least_0_90_is_ignored() -> None:
    shared = [f"token{index}" for index in range(9)]
    active = [_entry(3, " ".join([*shared, "alpha"]))]
    decision = _decision(" ".join([*shared, "beta"]), active=active)
    assert decision.overlap >= 0.9
    assert decision.action == "IGNORE"


def test_the_threshold_boundary_is_inclusive_at_0_90_and_0_35() -> None:
    nine = {f"t{index}" for index in range(9)}
    assert overlap(nine | {"x"}, nine | {"y"}) == 0.9

    def pair(shared: int) -> tuple[set[str], set[str]]:
        first = {f"a{index}" for index in range(shared)}
        second = {f"a{index}" for index in range(shared)}
        return (
            first | {f"x{index}" for index in range(100 - shared)},
            second | {f"y{index}" for index in range(100 - shared)},
        )

    assert overlap(*pair(89)) == 0.89
    assert overlap(*pair(90)) == 0.90
    assert overlap(*pair(34)) == 0.34
    assert overlap(*pair(35)) == 0.35
    assert normalise("  Hello, World.  ") == "hello, world"


def test_a_same_subject_compatible_fact_updates_and_the_old_one_is_superseded() -> None:
    active = [_entry(1, "I live in Berlin near the river and work at Acme")]
    decision = _decision(
        "I live in Berlin near the park and work at Acme",
        kind="profile",
        source="user",
        active=active,
    )
    assert decision.action == "UPDATE"
    assert decision.target_id == 1


def test_a_same_subject_changed_value_with_a_change_marker_updates() -> None:
    active = [_entry(1, "I live in Berlin and work at Acme")]
    decision = _decision(
        "I moved to Paris and now live there and work at Acme",
        kind="profile",
        source="user",
        active=active,
    )
    assert decision.action == "UPDATE"


def test_a_same_subject_contradiction_without_a_marker_conflicts() -> None:
    active = [_entry(1, "I live in Berlin in a small flat")]
    decision = _decision(
        "I live in Paris in a large house", kind="profile", source="user", active=active
    )
    assert decision.action == "CONFLICT"
    assert decision.target_id == 1


def test_an_overlapping_distinct_fact_is_related() -> None:
    active = [_entry(1, "Alice's address is 12 Main Street in Springfield")]
    decision = _decision(
        "Bob's address is 12 Main Street in Springfield",
        kind="semantic",
        source="user",
        active=active,
    )
    # The two subjects differ, so a shared body of words is RELATED.
    assert decision.action == "RELATED"


def test_a_new_fact_is_new() -> None:
    decision = _decision("the deployment window is every second Thursday")
    assert decision.action == "NEW"
    assert decision.target_id is None


def test_a_same_subject_entry_never_produces_new() -> None:
    active = [_entry(1, "I live in Berlin and work at Acme")]
    for text in (
        "I live in Berlin and work at Acme on the analytics team",
        "I live in Paris and work at Acme",
    ):
        assert (
            _decision(text, kind="profile", source="user", active=active).action
            != "NEW"
        )


def test_a_short_fact_and_a_task_file_fact_are_ignored_with_a_reason() -> None:
    short = _decision("Paris")
    assert (short.action, short.reason) == ("IGNORE", "short fact")
    file_only = _decision("read list_ops.py and tests/test_list_ops.py passed")
    assert (file_only.action, file_only.reason) == (
        "IGNORE",
        "task-file-only fact",
    )


def test_a_third_party_address_is_semantic_and_a_user_fact_is_profile() -> None:
    assert (
        classify_kind("Alice's address is 12 Main Street", source="reasoning_note")
        == "semantic"
    )
    assert classify_kind("my birthday is 12 May", source="user") == "profile"
    # A note's first-person item is never a profile: a note is tool output.
    assert classify_kind("my birthday is 12 May", source="reasoning_note") == "episodic"
    assert (
        classify_kind("the build takes four minutes", source="reasoning_note")
        == "episodic"
    )
    assert classify_kind("Attempt 1: tried it", source="reflection") == "reflection"
    assert classify_kind("task: q\nfacts: x", source="digest") == "episodic"


def test_the_marker_constants_are_public_and_hold_english_and_chinese() -> None:
    assert "remember" in REMEMBER_MARKERS
    assert "记住" in REMEMBER_MARKERS
    assert "I prefer" in USER_STATEMENT_MARKERS
    assert "我喜欢" in USER_STATEMENT_MARKERS
    assert "address" in SEMANTIC_ATTRIBUTES
    assert "地址" in SEMANTIC_ATTRIBUTES
    assert "moved" in CHANGE_MARKERS
    assert "搬到" in CHANGE_MARKERS


def test_a_candidate_carries_its_items_and_remember_bool() -> None:
    candidate = MemoryCandidate(
        candidate_id="n1",
        source="reasoning_note",
        text="I live in Berlin",
        remember_by="model",
        task_id="t1",
        remember=True,
    )
    assert candidate.items == ()
    assert candidate.remember
