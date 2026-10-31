"""``[memory] context_query``: a developer setting, off by default.

With it on, a chat message is searched together with the message before it, so
a follow-up finds what that message was about; by default only the message
itself is searched. It is saved only when on, checked as a bool, set for one run
with ``-X``, and listed by the developer help only.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.config import Config, MemoryConfig, save

KEY = "memory.context_query"


def test_it_is_off_by_default_saved_only_when_on_and_checked(tmp_path: Path) -> None:
    assert MemoryConfig().context_query is False
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        assert "context_query" not in path.read_text()
        cfg.memory = MemoryConfig(context_query=True)
        save(cfg)
    assert "[memory]\ncontext_query = true\n" in path.read_text()
    with mock.patch.object(config, "CONFIG_PATH", str(path)):
        assert config.load().memory == MemoryConfig(context_query=True)
    cfg.memory = MemoryConfig(context_query="yes")  # type: ignore[arg-type]
    assert "[memory] context_query must be a bool, got 'yes'" in cfg.validate()


def test_x_turns_it_on_for_one_run_and_only_the_developer_help_lists_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{KEY}=on", "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert config.overrides() == {KEY: True}
    assert config.load().memory.context_query is True
    assert config_cli.main(["keys", "context_query", "-X", f"{KEY}=on"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert [KEY, "true", "false", "-X"] in rows
    developer = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith(f"  {KEY} ") and line.endswith("(#299)")
        for line in developer.splitlines()
    )
    assert KEY not in cli._build_parser(developer=False).format_help()
