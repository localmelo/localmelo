"""Read-only attachment vectors and resume settings do not change direct output."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from localmelo.support import config
from localmelo.support.sessions.attachments import parse_user_text
from localmelo.support.sessions.display import resume_note


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        "a\n\n<attachment malformed>",
        'a\n\n<attachment name="x" chars="0">\nx\n</attachment>',
        'a\n\n<attachment name="x" chars="1">\nx\n</attachment>suffix',
    ],
)
def test_nonblocks_and_malformed_suffixes_remain_typed_text(text: str) -> None:
    assert parse_user_text(text) == (text, [])


def test_multiple_attachments_cut_and_escaped_closer() -> None:
    text = (
        'typed\n\n<attachment name="a.txt" chars="5">\nfirst\n</attachment>'
        '\n\n<attachment name="b.txt" chars="100" cut="20">\n<\\/attachment>\n</attachment>'
    )
    typed, files = parse_user_text(text)
    assert typed == "typed"
    assert [(f.name, f.chars, f.kept) for f in files] == [
        ("a.txt", 5, 5),
        ("b.txt", 100, 20),
    ]
    assert files[1].text == "</attachment>"


@pytest.mark.parametrize(
    "mode,turns,checkpoint,expected",
    [
        (
            "pairs",
            1,
            0,
            "1 turn is back in the model's memory; its tool calls are not.",
        ),
        (
            "pairs",
            6,
            0,
            "6 turns are back in the model's memory; their tool calls are not.",
        ),
        (
            "window",
            6,
            6,
            "6 turns are back in the model's memory, with their tool steps.",
        ),
        (
            "window",
            6,
            4,
            "4 turns are back in the model's memory, with their tool steps; 2 later turns without them.",
        ),
    ],
)
def test_resume_notes_use_post_eviction_counts(
    mode: str, turns: int, checkpoint: int, expected: str
) -> None:
    assert resume_note(mode, turns, checkpoint) == expected


@pytest.mark.parametrize("mode", ["pairs", "window", "invalid"])
def test_sessions_setting_round_trips_without_load_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.toml"))
    cfg = config.Config()
    cfg.sessions.resume_mode = mode
    config.save(cfg)
    with caplog.at_level(logging.WARNING):
        loaded = config.load()
    assert loaded.sessions.resume_mode == mode
    assert not caplog.records
    text = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert ("[sessions]" in text) == (mode != "pairs")
