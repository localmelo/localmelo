"""The credentials mask of procedural memory's query texts and labels.

After ``redact_secrets``, what procedures.db keeps also loses a URL's
password and the value of a ``NAME=value`` or ``NAME: value`` whose name
says key, token, secret, password, passwd or pwd. History and events
keep ``redact_secrets`` alone: their rows are the bench's.
"""

from __future__ import annotations

import time

import pytest

from localmelo.melo.memory.procedures.classify import clean_label
from localmelo.melo.memory.procedures.mask import MASK, mask_credentials
from localmelo.melo.memory.procedures.normalize import (
    query_text_of,
    stored_query_text,
)


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        (
            "connect to postgres://app:hunter2@db.local/prod",
            "connect to postgres://app:<redacted>@db.local/prod",
        ),
        (
            "https://x.org/a?api_key=abc123&page=2",
            "https://x.org/a?api_key=<redacted>&page=2",
        ),
        ("API_KEY=abc123 make deploy", "API_KEY=<redacted> make deploy"),
        ("DB_PASSWORD: s3cr3t, then go", "DB_PASSWORD: <redacted>, then go"),
        ("export GITHUB_TOKEN='a b c' now", "export GITHUB_TOKEN=<redacted> now"),
        ('mypwd = "x y" end', "mypwd = <redacted> end"),
        ("client_secret=Zz9", "client_secret=<redacted>"),
        ("passwd:qwerty", "passwd:<redacted>"),
    ],
)
def test_what_is_masked(text: str, masked: str) -> None:
    assert mask_credentials(text) == masked


@pytest.mark.parametrize(
    "text",
    [
        "run pytest and fix the failures",
        "open https://example.org/docs and summarize",
        "user@host: list files",
        "set mode=fast and retries: 3",
        "ssh://git@github.com/org/repo",
    ],
)
def test_what_is_kept(text: str) -> None:
    assert mask_credentials(text) == text


def test_masking_twice_changes_nothing() -> None:
    once = mask_credentials("postgres://a:b@h API_KEY=x token: y")
    assert mask_credentials(once) == once
    assert once.count(MASK) == 3


def test_the_mask_takes_linear_time() -> None:
    for text in ("a." * 100_000, "a" * 200_000 + "=", "x://" * 50_000):
        started = time.perf_counter()
        mask_credentials(text)
        assert time.perf_counter() - started < 0.5


def test_query_texts_are_masked() -> None:
    query = "Back up postgres://app:hunter2@db.local/prod with API_KEY=abc123"
    kept = query_text_of(query)
    assert "hunter2" not in kept and "abc123" not in kept
    assert (
        kept
        == "back up postgres://app:<redacted>@db.local/prod with api_key=<redacted>"
    )
    assert stored_query_text("token: zzz9") == "token: <redacted>"


def test_labels_are_masked() -> None:
    assert clean_label("Read api_token=abc123.txt") == "Read api_token=<redacted>"
