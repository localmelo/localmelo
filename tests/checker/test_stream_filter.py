"""Fragment-boundary and conservative display filtering tests."""

import pytest

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.stream_filter import StreamFilter
from localmelo.melo.schema import split_thinking


def run_fragments(fragments: list[str]) -> tuple[str, str]:
    stream = StreamFilter()
    for fragment in fragments:
        assert stream.feed(fragment) == ("", "")
    return stream.finish()


@pytest.mark.parametrize(
    "text",
    [
        "answer sk-" + "a" * 120 + " end",
        "ghp_" + "A" * 80 + " and github_pat_" + "x" * 80,
        "AKIA" + "A" * 16 + " end",
        "bearer " + " " * 100 + "a" * 80 + " end",
        "123456" + " " * 20 + "OTP",
        "verification code: 123456",
        "<think>secret sk-" + "b" * 60 + "</think> answer",
        "implicit thought</think> answer",
        "answer <think>unfinished thought",
    ],
)
def test_every_split_and_character_fragments(text: str) -> None:
    expected = run_fragments([text])
    for index in range(len(text) + 1):
        assert run_fragments([text[:index], text[index:]]) == expected
    assert run_fragments(list(text)) == expected


def test_unbounded_runs_are_not_released_early() -> None:
    key = "sk-" + "x" * 10000
    stream = StreamFilter()
    for fragment in ("before ", key[:200], key[200:], " after"):
        assert stream.feed(fragment) == ("", "")
    assert stream.finish() == ("", "before [redacted] after")


def test_redaction_after_joining_both_channels() -> None:
    key = "sk-" + "x" * 24
    text = "sk-" + "x" * 10 + "<think>secret</think>" + "x" * 14
    assert run_fragments([text]) == ("secret", redact_secrets(key)[0])
    text = "<think>sk-" + "x" * 10 + "</think><think>" + "x" * 14 + "</think>"
    assert run_fragments([text]) == ("[redacted]", "")


@pytest.mark.parametrize(
    ("text", "thinking", "content"),
    [
        ("before <THINK>thought</think> after", "thought", "before  after"),
        ("before <think>unfinished", "unfinished", "before "),
        ("unfinished</think> answer", "unfinished", " answer"),
        ("<think>a<think>b</think>still thinking", "abstill thinking", ""),
        ("<think>one</think> <think>two</think> answer", "onetwo", "  answer"),
        ("answer <thi", "<thi", "answer "),
        ("answer </think", "</think", "answer "),
        ("plain answer", "", "plain answer"),
    ],
)
def test_thinking_separation(text: str, thinking: str, content: str) -> None:
    assert run_fragments(list(text)) == (thinking, content)


@pytest.mark.parametrize(
    "text",
    [
        "plain",
        " <think>reason</think> answer ",
        "reason</think>answer",
        "<think>unfinished",
    ],
)
def test_common_cases_match_schema(text: str) -> None:
    assert run_fragments([text]) == split_thinking(text, strip_answer=False)


def test_redacts_code_across_thinking_boundary() -> None:
    text = "<think>OTP</think> 123456"
    assert "123456" not in run_fragments(list(text))[1]
    text = "<think>123456</think> OTP"
    assert "123456" not in "".join(run_fragments(list(text)))


def test_size_limit_fails_closed() -> None:
    stream = StreamFilter()
    assert stream.feed("x" * 1_048_576) == ("", "")
    with pytest.raises(ValueError, match="size limit"):
        stream.feed("x")
    with pytest.raises(RuntimeError):
        stream.finish()


def test_lifecycle_and_empty_input() -> None:
    stream = StreamFilter()
    assert stream.feed("") == ("", "")
    with pytest.raises(TypeError):
        stream.feed(None)  # type: ignore[arg-type]
    assert stream.finish() == ("", "")
    with pytest.raises(RuntimeError):
        stream.finish()
    with pytest.raises(RuntimeError):
        stream.feed("late")
