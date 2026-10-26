"""redact_secrets: one-time codes next to a keyword, and key shapes."""

from __future__ import annotations

import string
import time

import pytest

from localmelo.melo.checker.redaction import REDACTED, redact_secrets

LETTERS = string.ascii_letters[:24]
GITHUB = "ghp_" + "aB3" * 12  # 36 letters and digits
AWS = "AKIA" + "ABCDEFGH12345678"  # 16 capitals and digits

# (text, result, count): every example of R1's redaction table.
EXAMPLES = [
    ("Your verification code is 482913.", "Your verification code is [redacted].", 1),
    ("OTP: 1234", "OTP: [redacted]", 1),
    ("验证码：123456", "验证码：[redacted]", 1),
    (
        "G-482913 is your Google verification code",
        "G-[redacted] is your Google verification code",
        1,
    ),
    ("exit code 127", "exit code 127", 0),
    ("verification code: 12", "verification code: 12", 0),
    ("OTP 1234567890", "OTP 1234567890", 0),
    ("verification code:\n482913", "verification code:\n482913", 0),
    (f"key sk-{LETTERS}", "key [redacted]", 1),
    ("sk-short", "sk-short", 0),
    ("task-1234567890123456789012", "task-1234567890123456789012", 0),
    (GITHUB, REDACTED, 1),
    (AWS, REDACTED, 1),
    (
        "Authorization: Bearer abcdefghijklmnop1234",
        "Authorization: Bearer [redacted]",
        1,
    ),
    (f"Bearer sk-{LETTERS}", "Bearer [redacted]", 1),
    ("the bearer of bad news", "the bearer of bad news", 0),
    (f"OTP 1234sk-{LETTERS}", f"OTP 1234sk-{LETTERS}", 0),
    (f"{AWS}1234 OTP", f"{AWS}1234 OTP", 0),
    (
        "verification code 1234OTP" + " " * 15 + "5678",
        "verification code 1234OTP" + " " * 15 + "5678",
        0,
    ),
]


@pytest.mark.parametrize(("text", "result", "count"), EXAMPLES)
def test_the_examples(text: str, result: str, count: int) -> None:
    assert redact_secrets(text) == (result, count)


@pytest.mark.parametrize(("text", "result", "count"), EXAMPLES)
def test_a_second_call_replaces_nothing(text: str, result: str, count: int) -> None:
    assert redact_secrets(result) == (result, 0)


def test_the_examples_strung_together() -> None:
    text = "\n".join(text for text, _, _ in EXAMPLES)
    redacted, count = redact_secrets(text)
    assert redacted == "\n".join(result for _, result, _ in EXAMPLES)
    assert count == sum(count for _, _, count in EXAMPLES)
    assert redact_secrets(redacted) == (redacted, 0)


def test_the_count_covers_every_span() -> None:
    text = (
        f"sk-{LETTERS} then OTP 1234 and 5678, passcode 99999999, "
        f"{GITHUB} and bearer {'t' * 16}"
    )
    assert redact_secrets(text) == (
        "[redacted] then OTP [redacted] and [redacted], passcode [redacted], "
        "[redacted] and bearer [redacted]",
        6,
    )


@pytest.mark.parametrize(
    "text",
    [
        f"sk-proj-{LETTERS}",
        f"sk-ant-{LETTERS}",
        "gho_" + "x" * 36,
        "ghu_" + "x" * 40,
        "ghs_" + "x" * 36,
        "ghr_" + "x" * 36,
        "github_pat_" + "a_1" * 8,
        "ASIA" + "Z" * 16,
    ],
)
def test_the_other_key_shapes(text: str) -> None:
    assert redact_secrets(f"key: {text}.") == ("key: [redacted].", 1)


@pytest.mark.parametrize(
    "text",
    [
        "AKIA" + "A" * 15,  # one short
        "AKIA" + "A" * 17,  # one too many
        "xAKIA" + "A" * 16,  # a letter before it
        "ghp_" + "x" * 35,
        "_ghp_" + "x" * 36,
        f"risk-{LETTERS}",
    ],
)
def test_near_misses_stay(text: str) -> None:
    assert redact_secrets(text) == (text, 0)


def test_keywords_in_any_case_and_codes_on_either_side() -> None:
    assert redact_secrets("Verification Code:   654321") == (
        "Verification Code:   [redacted]",
        1,
    )
    assert redact_secrets("PASSCODE=0000") == ("PASSCODE=[redacted]", 1)
    assert redact_secrets("482913 is your otp") == ("[redacted] is your otp", 1)
    # 21 characters away, or past a newline: too far.
    assert redact_secrets("otp" + " " * 21 + "1234") == ("otp" + " " * 21 + "1234", 0)
    assert redact_secrets("1234\nOTP") == ("1234\nOTP", 0)
    # OTP inside a word is no keyword.
    assert redact_secrets("TOTPS 1234") == ("TOTPS 1234", 0)


def test_empty_text() -> None:
    assert redact_secrets("") == ("", 0)


def test_a_million_characters_take_well_under_a_second() -> None:
    chunk = f"verification code 482913, sk-{LETTERS} OTP 12345678 bearer {'t' * 20} "
    text = (chunk * (1_000_000 // len(chunk) + 1))[:1_000_000]
    started = time.perf_counter()
    redacted, count = redact_secrets(text)
    assert time.perf_counter() - started < 1.0
    assert count > 0 and "482913" not in redacted
