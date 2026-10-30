"""R6-lite's two direct-mode flags, their config table and their refusals."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config
from localmelo.support.config import Config, MemoryConfig, save

from .test_reasoning_flags import _error_line, _main, _parse, _write_config


def _options(argv: list[str], **kwargs: Any) -> AgentOptions:
    parser, args = _parse(argv)
    return cli._direct_options(parser, args)


def test_the_two_flags_parse_into_agent_options() -> None:
    options = _options(
        [
            "--reasoning",
            "compress",
            "--admit-on-outcome",
            "on",
            "--keep-decision-text",
            "q",
        ]
    )
    assert options.admit_on_outcome is True
    assert options.keep_decision_text is True


def test_their_defaults_are_off() -> None:
    options = _options(["--reasoning", "compress", "q"])
    assert options.admit_on_outcome is False
    assert options.keep_decision_text is False


def test_admit_on_outcome_is_accepted_and_inert_in_legacy() -> None:
    options = _options(["--reasoning", "legacy", "--admit-on-outcome", "on", "q"])
    assert options.reasoning == "legacy"
    assert options.admit_on_outcome is True


def test_keep_decision_text_is_refused_in_the_interactive_ui(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(["--keep-decision-text"], monkeypatch) == 2
    assert "only for direct mode" in _error_line(capsys)


def test_admit_on_outcome_is_accepted_in_the_interactive_ui() -> None:
    parser, args = _parse(["--admit-on-outcome", "on"])
    assert args.admit_on_outcome == "on"


def test_a_bad_admit_on_outcome_value_is_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--admit-on-outcome", "maybe", "q"])
    assert exc.value.code == 2


def test_the_config_table_applies_only_when_the_flag_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        agent=(
            '\n[agent]\nreasoning = "compress"\n'
            "\n[memory]\nadmit_on_outcome = true\n"
        ),
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert built.admit_on_outcome is True

    parser, args = _parse(["--admit-on-outcome", "off", "q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    assert agent_cls.call_args.kwargs["options"].admit_on_outcome is False


def test_a_bad_memory_value_is_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        agent='\n[agent]\nreasoning = "compress"\n\n[memory]\nadmit_on_outcome = "yes"\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["q"], monkeypatch) == 1
    assert "[memory] admit_on_outcome must be a bool" in capsys.readouterr().err


def test_the_table_round_trips_only_when_set(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(admit_on_outcome=True)
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
    assert "[memory]" in path.read_text()
    with mock.patch.object(config, "CONFIG_PATH", str(path)):
        loaded = config.load()
    assert loaded.memory == MemoryConfig(admit_on_outcome=True)

    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(Config(chat_backend="vllm", embedding_backend="none"))
    assert "[memory]" not in path.read_text()


def test_memory_across_chats_is_on_unless_the_file_turns_it_off(tmp_path: Path) -> None:
    assert MemoryConfig().across_chats is True
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(across_chats=False)
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
    assert "[memory]\nacross_chats = false\n" in path.read_text()
    with mock.patch.object(config, "CONFIG_PATH", str(path)):
        assert config.load().memory == MemoryConfig(across_chats=False)
    cfg.memory = MemoryConfig(across_chats="no")  # type: ignore[arg-type]
    assert "[memory] across_chats must be a bool, got 'no'" in cfg.validate()
