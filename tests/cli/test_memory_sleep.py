"""``[memory] sleep`` and ``melo memory sleep``: a developer setting, off by default.

The setting is ``off``, ``rules``, ``model`` or ``model+tags``.
It is saved only when not ``off``, checked, set for one run with
``-X``, and listed by the developer help only. The command is for developer
mode: it runs the setting's sleep over the notes and chats and prints the report.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli, memory_cli
from localmelo.support.config import Config, MemoryConfig, save
from localmelo.support.memory import ChatMemory
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach

KEY = "memory.sleep"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml"))
    yield tmp_path


def test_it_is_off_by_default_saved_only_when_set_and_checked(tmp_path: Path) -> None:
    assert MemoryConfig().sleep == "off"
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        assert "sleep" not in path.read_text()
        for variant in ("rules", "model+tags"):
            cfg.memory = MemoryConfig(sleep=variant)
            assert not [e for e in cfg.validate() if "[memory]" in e]
            save(cfg)
            assert f'[memory]\nsleep = "{variant}"\n' in path.read_text()
            assert config.load().memory == MemoryConfig(sleep=variant)
    cfg.memory = MemoryConfig(sleep="deep")
    assert any("[memory] sleep 'deep' is not recognised" in e for e in cfg.validate())


def test_sleep_lock_round_trips_and_is_validated(home: Path) -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory.sleep_lock = str(home / "idle.lock")
    save(cfg)
    assert config.load().memory.sleep_lock == cfg.memory.sleep_lock
    cfg.memory.sleep_lock = 3  # type: ignore[assignment]
    assert any("sleep_lock must be a path string" in e for e in cfg.validate())


def test_x_sets_it_for_one_run_and_only_the_developer_help_lists_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{KEY}=rules", "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert config.overrides() == {KEY: "rules"}
    assert config.load().memory.sleep == "rules"
    assert config_cli.main(["keys", "sleep", "-X", f"{KEY}=model+tags"]) == 0
    assert any(line.startswith(KEY) for line in capsys.readouterr().out.splitlines())
    developer = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith(f"  {KEY} ") and line.endswith("(#363)")
        for line in developer.splitlines()
    )
    assert KEY not in cli._build_parser(developer=False).format_help()


async def _seed(home: Path) -> None:
    chats = SessionStore(str(home / "sessions.db"), writer="gateway")
    await chats.open()
    memory = ChatMemory(lambda: None)
    try:
        turn = await chats.begin_turn(
            None, "I prefer short answers.", holder="h:1:c", new_session=NewSession()
        )
        await teach(memory, chats, turn, "I prefer short answers.")
    finally:
        await memory.aclose()
        await chats.aclose()


def test_the_command_is_for_developer_mode_and_prints_the_report(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asyncio.run(_seed(home))
    assert memory_cli.main(["sleep"]) == 2
    assert 'needs [ui] mode = "developer"' in capsys.readouterr().err
    path = home / "cfg" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[ui]\nmode = "developer"\n')
    assert memory_cli.main(["sleep"]) == 0  # the setting is off
    assert "Sleep off: 0 changes, 0 model calls, run " in capsys.readouterr().out
    assert memory_cli.main(["sleep", "rules"]) == 0
    assert "Sleep rules: 0 changes, 0 model calls, run " in capsys.readouterr().out
    assert memory_cli.main(["sleep", "model"]) == 0
    assert "Sleep model: 0 changes, 0 model calls, run " in capsys.readouterr().out
