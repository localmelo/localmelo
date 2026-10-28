from __future__ import annotations

import codecs

import pytest

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.stream_filter import (
    ReasoningTags,
    StreamFilter,
    split_framed,
)


@pytest.mark.parametrize(
    ("hint", "text"),
    [
        ("none", "hello 中文\nBearer\n  " + "x" * 30 + "\nend"),
        ("none", "ordinary </think> remains literal\n"),
        ("explicit", "  <think>private OTP 123456</think>hello\nworld\n"),
        ("explicit", "hello\nno tag\n"),
        ("explicit", "<think>not yet finished"),
        ("closing_only", "private 验证码 123456</think>answer\nnext\n"),
        ("closing_only", "İstanbul private</think>answer\n"),
        ("closing_only", "unfinished secret"),
        ("explicit", "<think>OTP</think> 123456\nmore\n"),
        ("explicit", "<think>secret</think>sk-" + "x" * 30 + "\n"),
        ("closing_only", "sk-" + "x" * 10 + "</think>" + "x" * 15 + "\n"),
    ],
)
def test_byte_splits_equal_whole_policy(hint: ReasoningTags, text: str) -> None:
    raw = text.encode("utf-8")
    thinking, content = split_framed(redact_secrets(text)[0], hint)
    expected = redact_secrets(thinking)[0], redact_secrets(content)[0]
    for i in range(len(raw) + 1):
        decoder = codecs.getincrementaldecoder("utf-8")()
        stream = StreamFilter(reasoning_tags=hint)
        fragments = [decoder.decode(raw[:i]), decoder.decode(raw[i:], final=True)]
        observed = [stream.feed(fragment) for fragment in fragments] + [stream.finish()]
        assert (
            "".join(t for t, _ in observed),
            "".join(c for _, c in observed),
        ) == expected
    decoder = codecs.getincrementaldecoder("utf-8")()
    stream = StreamFilter(reasoning_tags=hint)
    observed = [stream.feed(decoder.decode(bytes([byte]))) for byte in raw]
    observed.append(stream.feed(decoder.decode(b"", final=True)))
    observed.append(stream.finish())
    assert (
        "".join(t for t, _ in observed),
        "".join(c for _, c in observed),
    ) == expected


def test_redaction_rules_are_snapshotted_per_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import re

    from localmelo.melo.checker import redaction

    stream = StreamFilter(reasoning_tags="none")
    assert stream.feed("first\n") == ("", "first\n")
    monkeypatch.setattr(redaction, "_KEY", re.compile(r"new-secret"))
    assert stream.feed("new-secret\n") == ("", "new-secret\n")
    assert stream.finish() == ("", "")
    assert stream.redact_for_storage("first\nnew-secret\n") == "first\n[redacted]\n"

    later = StreamFilter(reasoning_tags="none")
    assert later.feed("new-secret\n") == ("", "")
    assert later.finish() == ("", "[redacted]\n")


def test_declared_none_releases_and_unknown_buffers() -> None:
    known = StreamFilter(reasoning_tags="none")
    assert known.feed("safe line\n") == ("", "safe line\n")
    assert known.finish() == ("", "")
    unknown = StreamFilter()
    assert unknown.feed("safe line\n") == ("", "")
    assert unknown.finish() == ("", "safe line\n")


def test_declared_separate_reasoning_never_enters_content() -> None:
    stream = StreamFilter(separate_reasoning=True)
    assert stream.feed_reasoning("private sk-" + "x" * 30) == ("", "")
    assert stream.feed("answer\n") == ("", "answer\n")
    assert stream.finish() == ("private [redacted]", "")
    with pytest.raises(ValueError):
        StreamFilter(separate_reasoning=True, reasoning_tags="explicit")


def test_closing_only_never_releases_reasoning_as_content() -> None:
    stream = StreamFilter(reasoning_tags="closing_only")
    assert stream.feed("private line\n") == ("", "")
    assert stream.feed("</think>answer line\n") == ("", "answer line\n")
    assert stream.finish() == ("private line", "")
