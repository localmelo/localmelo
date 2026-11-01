"""``[memory]`` recall settings: developer settings that default to today's recall.

``recall_records``, ``recall_docs``, ``recall_chars``, ``recall_notes`` and
``search_when_empty`` are saved only when they differ from the default, checked
against their bounds and values, set for one run with ``-X``, and listed by the
developer help only, each with the issue that decides it.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.config import RECALL_KEYS, Config, MemoryConfig, save

CHANGED = {
    "recall_records": 8,
    "recall_docs": 0,
    "recall_chars": 4000,
    "recall_notes": "off",
    "search_when_empty": "hint",
}


def errors(**memory: object) -> list[str]:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return [error for error in cfg.validate() if "[memory]" in error]


def test_they_are_saved_only_when_changed_and_read_back(tmp_path: Path) -> None:
    assert set(RECALL_KEYS) == set(CHANGED)
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        assert (
            "recall_" not in path.read_text() and "search_when" not in path.read_text()
        )
        cfg.memory = MemoryConfig(recall_records=3, recall_notes="on")  # the defaults
        save(cfg)
        assert "[memory]" not in path.read_text()
        cfg.memory = MemoryConfig(**CHANGED)  # type: ignore[arg-type]
        save(cfg)
        text = path.read_text()
        assert (
            "[memory]\nrecall_records = 8\nrecall_docs = 0\nrecall_chars = 4000\n"
            'recall_notes = "off"\nsearch_when_empty = "hint"\n'
        ) in text
        assert config.load().memory == MemoryConfig(**CHANGED)  # type: ignore[arg-type]
        for name, value in CHANGED.items():  # each alone
            cfg.memory = MemoryConfig(**{name: value})  # type: ignore[arg-type]
            save(cfg)
            keys = [line.split(" ")[0] for line in path.read_text().splitlines()]
            assert [key for key in keys if key in CHANGED] == [name]
            assert config.load().memory == cfg.memory


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("recall_records", 0),
        ("recall_records", 20),
        ("recall_docs", 0),
        ("recall_docs", 20),
        ("recall_chars", 200),
        ("recall_chars", 8000),
        ("recall_notes", "on"),
        ("recall_notes", "off"),
        ("search_when_empty", "off"),
        ("search_when_empty", "hint"),
    ],
)
def test_the_bounds_and_values_are_accepted(name: str, value: object) -> None:
    assert errors(**{name: value}) == []


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("recall_records", -1, "[memory] recall_records must be from 0 to 20, got -1"),
        ("recall_records", 21, "[memory] recall_records must be from 0 to 20, got 21"),
        ("recall_docs", 21, "[memory] recall_docs must be from 0 to 20, got 21"),
        (
            "recall_chars",
            199,
            "[memory] recall_chars must be from 200 to 8000, got 199",
        ),
        (
            "recall_chars",
            8001,
            "[memory] recall_chars must be from 200 to 8000, got 8001",
        ),
        ("recall_docs", "3", "[memory] recall_docs must be a whole number, got '3'"),
        ("recall_docs", 2.5, "[memory] recall_docs must be a whole number, got 2.5"),
        (
            "recall_records",
            True,
            "[memory] recall_records must be a whole number, got True",
        ),
        (
            "recall_notes",
            "maybe",
            "[memory] recall_notes 'maybe' is not recognised. "
            "Must be one of ('on', 'off').",
        ),
        (
            "recall_notes",
            False,
            "[memory] recall_notes False is not recognised. "
            "Must be one of ('on', 'off').",
        ),
        (
            "search_when_empty",
            "always",
            "[memory] search_when_empty 'always' is not recognised. "
            "Must be one of ('off', 'hint').",
        ),
    ],
)
def test_a_value_out_of_range_is_a_config_error_that_says_why(
    name: str, value: object, message: str
) -> None:
    assert errors(**{name: value}) == [message]


def test_x_sets_each_for_one_run_and_only_the_developer_help_lists_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv: list[str] = []
    for name, value in CHANGED.items():
        argv += ["-X", f"memory.{name}={value}"]
    cli._parse([*argv, "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert config.overrides() == {f"memory.{k}": v for k, v in CHANGED.items()}
    assert config.load().memory == MemoryConfig(**CHANGED)  # type: ignore[arg-type]
    assert config_cli.main(["keys", "recall_records", "-X", argv[1]]) == 0
    assert "memory.recall_records" in capsys.readouterr().out
    developer = cli._build_parser(developer=True).format_help().splitlines()
    plain = cli._build_parser(developer=False).format_help()
    for name in CHANGED:
        key = f"memory.{name}"
        assert any(
            line.startswith(f"  {key} ") and line.endswith("(#349)")
            for line in developer
        ), key
        assert key not in plain


@pytest.mark.parametrize(
    "setting",
    [
        "memory.recall_records=21",
        "memory.recall_chars=100",
        "memory.recall_docs=two",
        "memory.recall_notes=maybe",
        "memory.search_when_empty=always",
    ],
)
def test_x_refuses_a_bad_value_with_a_clear_message(
    setting: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli._parse(["-X", setting, "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert exit_.value.code == 2
    assert setting.partition("=")[0] in capsys.readouterr().err
