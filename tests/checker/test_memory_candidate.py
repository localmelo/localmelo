"""R4's candidate boundary: redact, sanitize, cap, and the refusals."""

from __future__ import annotations

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import MemoryCandidatePayload
from localmelo.melo.checker.validators import (
    CANDIDATE_LIMITS,
    CANDIDATE_SOURCES,
    CANDIDATE_TARGETS,
    CANDIDATE_TRUNCATION_SUFFIX,
    MAX_MEMORY_TEXT_LEN,
    validate_memory_candidate,
)


def _candidate(
    text: str,
    source: str = "reasoning_note",
    metadata: dict[str, object] | None = None,
) -> MemoryCandidatePayload:
    return MemoryCandidatePayload(
        text=text, source=source, target="candidate", metadata=dict(metadata or {})
    )


def test_cap_and_metadata() -> None:
    result = validate_memory_candidate(_candidate("a" * 2_000, metadata={"step": 3}))
    assert result.allowed
    payload = result.sanitized_payload
    assert len(payload.text) == CANDIDATE_LIMITS["reasoning_note"]
    assert payload.text.endswith(CANDIDATE_TRUNCATION_SUFFIX)
    assert payload.metadata == {"step": 3, "redactions": 0}


def test_every_cap_is_inside_the_memory_text_cap() -> None:
    for source in CANDIDATE_SOURCES:
        assert CANDIDATE_LIMITS[source] <= MAX_MEMORY_TEXT_LEN
    assert set(CANDIDATE_LIMITS) == set(CANDIDATE_SOURCES)
    assert set(CANDIDATE_TARGETS) == {"candidate", "prompt", "long"}


def test_text_within_the_cap_is_unchanged() -> None:
    result = validate_memory_candidate(_candidate("plan: read list_ops.py"))
    assert result.allowed
    assert result.sanitized_payload.text == "plan: read list_ops.py"


def test_a_verification_code_is_redacted_and_counted() -> None:
    result = validate_memory_candidate(
        _candidate("The verification code is 482913, so I can log in.")
    )
    assert result.allowed
    assert "[redacted]" in result.sanitized_payload.text
    assert "482913" not in result.sanitized_payload.text
    assert result.sanitized_payload.metadata["redactions"] == 1


def test_an_sk_key_is_redacted() -> None:
    result = validate_memory_candidate(
        _candidate("found: the key sk-" + "a" * 30 + " in the config")
    )
    assert result.allowed
    assert "sk-" + "a" * 30 not in result.sanitized_payload.text
    assert result.sanitized_payload.metadata["redactions"] == 1


def test_tool_and_role_tokens_are_escaped() -> None:
    result = validate_memory_candidate(
        _candidate('the tool returned <|user|> and <tool_call>{"x": 1}</tool_call>')
    )
    assert result.allowed
    text = result.sanitized_payload.text
    assert "&lt;|user|&gt;" in text
    assert "&lt;tool_call&gt;" in text
    assert "&lt;/tool_call&gt;" in text


def test_a_surrogate_pair_is_never_cut() -> None:
    # The cap's cut (keep = 1,185) falls between the pair's two halves:
    # index 1,184 holds the high surrogate and 1,185 the low one.
    text = "a" * 1_184 + "\ud83d\ude00" + "b" * 100
    result = validate_memory_candidate(_candidate(text))
    assert result.allowed
    kept = result.sanitized_payload.text[: -len(CANDIDATE_TRUNCATION_SUFFIX)]
    kept.encode("utf-8")  # no lone surrogate
    assert "\ud83d" not in kept and "\ude00" not in kept
    assert kept == "a" * 1_184


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_refused(text: str) -> None:
    result = validate_memory_candidate(_candidate(text))
    assert not result.allowed
    assert "must not be empty" in result.reason


def test_unknown_source_and_target_are_refused() -> None:
    source = validate_memory_candidate(_candidate("text", source="mystery"))
    assert not source.allowed
    assert "Unknown memory candidate source" in source.reason
    target = validate_memory_candidate(
        MemoryCandidatePayload(text="text", source="draft", target="nowhere")
    )
    assert not target.allowed
    assert "Unknown memory candidate target" in target.reason


def test_a_profile_long_entry_must_come_from_the_user() -> None:
    refused = validate_memory_candidate(
        MemoryCandidatePayload(
            text="likes tea", source="reasoning_note", target="long", kind="profile"
        )
    )
    assert not refused.allowed
    assert "must come from the user" in refused.reason
    allowed = validate_memory_candidate(
        MemoryCandidatePayload(
            text="likes tea", source="user", target="long", kind="profile"
        )
    )
    assert allowed.allowed


def test_checker_exposes_the_candidate_boundary() -> None:
    result = Checker().check_memory_candidate(_candidate("a note"))
    assert result.allowed
