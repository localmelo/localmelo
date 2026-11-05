"""The settings that were removed: an old config file still loads, the options exit 2."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.option_tiers import REMOVED_SETTINGS

OLD_FILE = """\
[agent]
fixed_prefix = false
reflect_tool_guard = false
cache_layout = "v2"
reflection_gate = "utility"
window_eviction = "slide"
window_messages = 32

[memory]
admit_on_outcome = true
"""


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    (tmp_path / "config.toml").write_text(OLD_FILE, encoding="utf-8")
    return tmp_path


def test_a_file_that_holds_removed_keys_loads(config_home: Path) -> None:
    agent = config.load().agent
    assert agent.window_messages == 32  # the keys that stay are read as before
    assert agent.window_eviction == "chunk"  # "slide" reads as the one way left
    for name in REMOVED_SETTINGS:
        section, _, key = name.partition(".")
        assert not hasattr(getattr(config.load(), section), key)


def test_melo_config_names_each_removed_key_once(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert config_cli.main([]) == 0
    err = capsys.readouterr().err.splitlines()
    assert len(err) == len(REMOVED_SETTINGS) + 1  # and the "slide" line
    for name in REMOVED_SETTINGS:
        assert sum(f"note: {name} was removed" in line for line in err) == 1
    assert sum("window_eviction" in line for line in err) == 1
    assert all(line.endswith("Delete the line.") for line in err)


def test_saving_the_config_drops_the_removed_keys(config_home: Path) -> None:
    config.save(config.load())
    text = (config_home / "config.toml").read_text(encoding="utf-8")
    assert "window_messages = 32" in text
    for name in REMOVED_SETTINGS:
        assert name.partition(".")[2] not in text


@pytest.mark.parametrize("name", REMOVED_SETTINGS)
def test_the_option_of_a_removed_setting_exits_2_and_says_so(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    flag = "--" + name.rpartition(".")[2].replace("_", "-")
    with pytest.raises(SystemExit) as exc:
        cli._parse([flag, "off", "task"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"{flag} was removed: {REMOVED_SETTINGS[name]}." in err


@pytest.mark.parametrize("name", REMOVED_SETTINGS)
def test_the_setter_of_a_removed_setting_exits_2_and_says_so(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse(["-X", f"{name}=off", "task"])
    assert exc.value.code == 2
    assert f"{name} was removed: {REMOVED_SETTINGS[name]}." in capsys.readouterr().err
