from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from localmelo.melo.checker import redaction
from localmelo.melo.checker._stream_redaction import LineRedactor


@pytest.mark.parametrize(
    "text",
    [
        "plain line\nsecond line\nlast",
        "start sk-" + "a" * 30 + "\nend",
        "ghp_" + "b" * 40 + "\nAKIA" + "A" * 16 + "\n",
        "Bearer\n  " + "x" * 40 + "\nnext\n",
        "123456 -- OTP\nverification code: 765432\n",
        "中文 验证码 123456\nhello\n",
    ],
)
def test_every_utf8_byte_split_matches_whole_redaction(text: str) -> None:
    import codecs

    raw = text.encode("utf-8")
    expected = redaction.redact_secrets(text)[0]
    for offset in range(len(raw) + 1):
        decoder = codecs.getincrementaldecoder("utf-8")()
        release = LineRedactor()
        out = release.feed(decoder.decode(raw[:offset]))
        out += release.feed(decoder.decode(raw[offset:], final=True))
        assert out + release.finish() == expected


def test_completed_line_releases_and_open_bearer_waits() -> None:
    release = LineRedactor()
    assert release.feed("safe\nBearer\n ") == "safe\n"
    assert release.feed("x" * 20 + "\nnext\n") == "Bearer\n [redacted]\nnext\n"
    assert release.finish() == ""


def test_unknown_rules_force_whole_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from localmelo.melo.checker import _stream_redaction

    monkeypatch.setattr(_stream_redaction, "_known_rules", lambda: False)
    release = LineRedactor()
    assert release.feed("first\nsecond\n") == ""
    assert release.finish() == "first\nsecond\n"


def test_rule_inventory_requires_deliberate_update() -> None:
    # A changed regex must be assessed for cross-line matches before release.
    # Any change, including a new pass, forces a review of line locality.
    assert redaction.__file__ is not None
    assert hashlib.sha256(Path(redaction.__file__).read_bytes()).hexdigest() == (
        "7f17ad11ae7be99bb8095a0cb398b8e5233d2937ee0171a724a1768070a1110f"
    )
    assert redaction._KEY.pattern == (
        r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"
        r"|(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
        r"|(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"
    )
    assert redaction._BEARER.pattern == r"(\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}"
    assert redaction._CODE_KEYWORD.pattern == (
        r"verification code|passcode|(?<![A-Za-z0-9_])otp(?![A-Za-z0-9_])|验证码"
    )
    assert redaction._CODE.pattern == r"(?<![A-Za-z0-9])[0-9]{4,8}(?![A-Za-z0-9])"
    assert redaction._CODE_REACH == 20
