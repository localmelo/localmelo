"""``[memory] project_instructions``: whether a task reads its folder's instruction file.

``"on"`` is the default and is never written; ``"off"`` is saved and read back,
checked against its two values, set for one run with ``-X``, and listed by the
full help like every other key.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, settings
from localmelo.support.config import Config, MemoryConfig, save


def errors(**memory: object) -> list[str]:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return [error for error in cfg.validate() if "[memory]" in error]


def test_it_is_on_by_default_and_saved_only_when_off(tmp_path: Path) -> None:
    assert MemoryConfig().project_instructions == "on"
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        assert "project_instructions" not in path.read_text()
        cfg.memory = MemoryConfig(project_instructions="off")
        save(cfg)
        assert '[memory]\nproject_instructions = "off"\n' in path.read_text()
        assert config.load().memory.project_instructions == "off"
        cfg.memory = MemoryConfig(project_instructions="on", recall_records=5)
        save(cfg)
        assert "project_instructions" not in path.read_text()


@pytest.mark.parametrize("value", ["on", "off"])
def test_its_two_values_are_accepted(value: str) -> None:
    assert errors(project_instructions=value) == []


@pytest.mark.parametrize("value", ["maybe", False, ""])
def test_another_value_is_a_config_error_that_says_why(value: object) -> None:
    assert errors(project_instructions=value) == [
        f"[memory] project_instructions {value!r} is not recognised. "
        "Must be one of ('on', 'off')."
    ]


def test_x_sets_it_for_one_run_and_the_full_help_lists_it() -> None:
    cli._parse(
        ["-X", "memory.project_instructions=off", "--base-url", "http://x/v1", "q"]
    )
    assert config.overrides() == {"memory.project_instructions": "off"}
    assert config.load().memory.project_instructions == "off"
    assert "memory.project_instructions" in settings.catalog()
    full = cli._build_parser(developer=True).format_help().splitlines()
    assert any(line.startswith("  memory.project_instructions ") for line in full)


def test_x_refuses_a_bad_value(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli._parse(
            [
                "-X",
                "memory.project_instructions=maybe",
                "--base-url",
                "http://x/v1",
                "q",
            ]
        )
    assert exit_.value.code == 2
    assert "memory.project_instructions" in capsys.readouterr().err
