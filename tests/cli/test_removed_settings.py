"""The settings that were removed: an old config file still loads, the options exit 2."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.option_tiers import REMOVED_SETTINGS, REMOVED_VALUES

OLD_FILE = """\
[agent]
fixed_prefix = false
reflect_tool_guard = false
cache_layout = "v2"
reflection_gate = "utility"
window_eviction = "slide"
window_messages = 32
thinking_steps = "first_and_failed"
lean_output = true
brief_items = true
captured_history = true
replay_turns = 3
tool_images_keep = 4
tool_images_retire = 1

[memory]
admit_on_outcome = true
followup_query = "message"
recall_records = 5
recall_docs = 4
recall_chars = 3000
recall_item_chars = 600
compress_at_chars = 8000
gather_min_cosine = 0.3
excerpt_open = "off"
premise_check = "off"
gather = "wide"
search_when_empty = "hint"
recall_notes = "off"
open_sources = "on"
keep_statements = true
note_gate = "meaning"
note_min_cosine = 0.5
recall_min_cosine = 0.5
recall_min_coverage = 0.8
recall_excerpt = "match"
recall_form = "list"
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
    assert config.load().memory.recall_form == "list"  # a [memory] key that stays
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


@pytest.mark.parametrize(
    "line",
    [
        'note_gate = "meaning"',
        'note_gate = "typed"',
        "note_min_cosine = 0.5",
        "recall_min_cosine = 0.5",
        "recall_min_coverage = 1",
        'recall_excerpt = "match"',
    ],
)
def test_a_file_with_one_removed_memory_key_loads_with_a_note_and_the_default(
    line: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    (tmp_path / "config.toml").write_text(f"[memory]\n{line}\n", encoding="utf-8")
    assert config.load().memory == config.MemoryConfig()
    assert config_cli.main([]) == 0
    key = line.partition(" ")[0]
    assert f"note: memory.{key} was removed" in capsys.readouterr().err


@pytest.fixture
def removed_value_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    return tmp_path


@pytest.mark.parametrize(("name", "value"), REMOVED_VALUES)
def test_the_option_and_the_setter_of_a_removed_value_exit_2_and_say_so(
    name: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    flag = "--" + name.rpartition(".")[2].replace("_", "-")
    for argv in (
        [flag, value, "list files"],
        [f"{flag}={value}", "list files"],
        ["-X", f"{name}={value}", "list files"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli._parse(argv)
        assert exc.value.code == 2, argv
        err = capsys.readouterr().err
        assert f"{value} was removed: {REMOVED_VALUES[name, value]}." in err, argv


@pytest.mark.parametrize(("name", "value"), REMOVED_VALUES)
def test_melo_config_set_refuses_a_removed_value(
    name: str,
    value: str,
    removed_value_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        config_cli.main(["set", name, value])
    assert exc.value.code == 2
    assert f"{name}={value} was removed" in capsys.readouterr().err
    assert not (removed_value_home / "config.toml").exists()


@pytest.mark.parametrize("value", ["legacy", "compress"])
def test_the_values_that_stay_are_not_refused(value: str) -> None:
    for argv in (
        ["--reasoning", value, "list files"],
        ["-X", f"agent.reasoning={value}", "list files"],
    ):
        _, args = cli._parse(argv)
        assert args.reasoning == value


@pytest.mark.parametrize(("name", "value"), REMOVED_VALUES)
def test_a_file_that_holds_a_removed_value_loads_reads_the_default_and_is_named(
    name: str,
    value: str,
    removed_value_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    section, _, key = name.rpartition(".")
    (removed_value_home / "config.toml").write_text(
        f'[{section}]\n{key} = "{value}"\n', encoding="utf-8"
    )
    default = getattr(getattr(config.Config(), section), key)
    assert getattr(getattr(config.load(), section), key) == default
    parser, args = cli._parse(["list files"])
    assert cli._direct_options(parser, args).reasoning == default
    assert config_cli.main([]) == 0
    err = capsys.readouterr().err.splitlines()
    assert [line for line in err if name in line] == [
        f'melo: note: {name} = "{value}" was removed '
        f"({REMOVED_VALUES[name, value]}); "
        f'it reads as "{default}". Delete the line.'
    ]
