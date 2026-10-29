from __future__ import annotations

import base64
import random

import pytest

from localmelo.melo.checker import _stream_redaction, redaction
from localmelo.melo.checker._stream_redaction import LineRedactor

LABELS = [
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
    "PRIVATE KEY",
    "PGP PRIVATE KEY BLOCK",
]


def fake_body(lines: int = 4, seed: int = 1) -> str:
    rng = random.Random(seed)
    return "\n".join(base64.b64encode(rng.randbytes(45)).decode() for _ in range(lines))


def block(label: str, body: str | None = None, end: bool = True) -> str:
    body = fake_body() if body is None else body
    text = f"-----BEGIN {label}-----\n{body}\n"
    return text + (f"-----END {label}-----" if end else "")


def stream(text: str, offsets: tuple[int, ...] = ()) -> str:
    release = LineRedactor()
    out, last = "", 0
    for offset in (*offsets, len(text)):
        out += release.feed(text[last:offset])
        last = offset
    return out + release.finish()


@pytest.mark.parametrize("label", LABELS)
def test_whole_block_keeps_marker_lines_and_drops_body(label: str) -> None:
    body = fake_body()
    text = f"before\n{block(label, body)}\nafter\n"
    out, count = redaction.redact_secrets(text)
    assert out == (
        f"before\n-----BEGIN {label}-----\n[redacted]\n-----END {label}-----\nafter\n"
    )
    assert count == 1
    assert body.splitlines()[0] not in out


@pytest.mark.parametrize("label", LABELS)
def test_block_without_end_is_redacted_to_the_end(label: str) -> None:
    out, count = redaction.redact_secrets("x\n" + block(label, end=False) + "\n")
    assert out == f"x\n-----BEGIN {label}-----\n[redacted]"
    assert count == 1


def test_count_is_one_per_block_and_second_call_changes_nothing() -> None:
    text = block("RSA PRIVATE KEY") + "\nmid\n" + block("OPENSSH PRIVATE KEY")
    out, count = redaction.redact_secrets(text)
    assert count == 2
    assert redaction.redact_secrets(out) == (out, 0)
    cut, count = redaction.redact_secrets(block("EC PRIVATE KEY", end=False))
    assert (count, redaction.redact_secrets(cut)) == (1, (cut, 0))


def test_other_passes_still_run_around_a_block() -> None:
    text = f"sk-{'a' * 30}\n{block('PRIVATE KEY')}\nOTP 123456"
    assert redaction.redact_secrets(text) == (
        "[redacted]\n-----BEGIN PRIVATE KEY-----\n[redacted]\n"
        "-----END PRIVATE KEY-----\nOTP [redacted]",
        3,
    )


@pytest.mark.parametrize(
    "text",
    [
        block("PUBLIC KEY"),
        block("CERTIFICATE"),
        block("RSA PUBLIC KEY"),
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI" + "A" * 30 + " user@host\n",
    ],
)
def test_public_material_is_unchanged(text: str) -> None:
    assert redaction.redact_secrets(text) == (text, 0)
    assert stream(text) == text


@pytest.mark.parametrize("label", ["RSA PRIVATE KEY", "PRIVATE KEY"])
def test_stream_split_at_every_offset_matches_whole(label: str) -> None:
    text = f"hi\n{block(label, fake_body(2))}\ntail\n"
    expected = redaction.redact_secrets(text)[0]
    for offset in range(len(text) + 1):
        assert stream(text, (offset,)) == expected
    for offset in range(len(text) + 1):
        cut = text[: text.index("-----END")]
        assert stream(cut, (offset,)) == redaction.redact_secrets(cut)[0]


@pytest.mark.parametrize("limit", [0, 120, 16 * 1024])
def test_stream_random_splits_match_whole(
    limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_stream_redaction, "PEM_HOLD_LIMIT", limit)
    rng = random.Random(7)
    parts = [
        "log line\n",
        block("OPENSSH PRIVATE KEY", fake_body(6)),
        "\nBearer\n " + "x" * 30 + "\n",
        block("CERTIFICATE"),
        "\nzzz " + block("EC PRIVATE KEY", fake_body(3)) + "tail sk-" + "b" * 25,
        "\n" + block("PRIVATE KEY", fake_body(2), end=False),
    ]
    for _ in range(200):
        text = "".join(rng.sample(parts, len(parts)))
        offsets = tuple(
            sorted(rng.randrange(len(text)) for _ in range(rng.randint(0, 12)))
        )
        assert stream(text, offsets) == redaction.redact_secrets(text)[0]


def test_open_block_is_held_until_its_end_line() -> None:
    release = LineRedactor()
    assert release.feed("ok\n-----BEGIN RSA PRIVATE KEY-----\n") == "ok\n"
    assert release.feed(fake_body(2) + "\n") == ""
    assert release.feed("-----END RSA PRIVATE KEY-----\nnext\n") == (
        "-----BEGIN RSA PRIVATE KEY-----\n[redacted]\n"
        "-----END RSA PRIVATE KEY-----\nnext\n"
    )
    assert release.finish() == ""


def test_block_over_the_cap_is_dropped_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_stream_redaction, "PEM_HOLD_LIMIT", 200)
    body = fake_body(40)
    text = f"head\n{block('OPENSSH PRIVATE KEY', body)}\ntail sk-{'c' * 25}\n"
    expected = redaction.redact_secrets(text)[0]
    release = LineRedactor()
    seen = ""
    for i in range(0, len(text), 13):
        seen += release.feed(text[i : i + 13])
        assert not any(line in seen for line in body.splitlines())
        assert len(release._pending) <= 200 + 13 + len(text.splitlines()[-1])
    assert seen + release.finish() == expected
    assert stream(text, (50, 51, 400)) == expected


def test_cap_path_without_end_matches_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_stream_redaction, "PEM_HOLD_LIMIT", 100)
    text = "a\n" + block("RSA PRIVATE KEY", fake_body(30), end=False)
    release = LineRedactor()
    out = "".join(release.feed(text[i : i + 7]) for i in range(0, len(text), 7))
    assert out + release.finish() == redaction.redact_secrets(text)[0]
    assert len(release._pending) < 96


def test_unreviewed_edit_still_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    monkeypatch.setattr(redaction, "_PEM", re.compile(r"-----BEGIN PRIVATE KEY-----"))
    release = LineRedactor()
    assert release.feed("a\nb\n") == ""
