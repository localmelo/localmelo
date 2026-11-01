"""``[workspace] extra_folders`` in ``config.toml`` (#405).

The folders "Always allow this folder" keeps: validated, written only when
the list holds a folder, a bad value kept for the error and never used.
HOME is a temporary folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.support import config, settings


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    return Path(os.path.realpath(folder))


def test_an_empty_list_writes_no_table() -> None:
    config.save(config.Config())
    assert "[workspace]" not in Path(config.config_path()).read_text()
    assert config.load().workspace.extra_folders == []


def test_a_kept_folder_round_trips(home: Path) -> None:
    (home / "developer" / "career").mkdir(parents=True)
    assert config.add_extra_folder(str(home / "developer" / "career")) is True
    assert config.add_extra_folder(str(home / "developer" / "career")) is False
    text = Path(config.config_path()).read_text()
    assert '[workspace]\nextra_folders = ["~/developer/career"]\n' in text
    cfg = config.load()
    assert cfg.workspace.extra_folders == ["~/developer/career"]
    assert config.extra_folders(cfg) == [str(home / "developer" / "career")]
    assert config.SavedFolders().folders() == [str(home / "developer" / "career")]


@pytest.mark.parametrize(
    "bad",
    ["~/.ssh", "~", "/", "relative/path", "/etc", "~/.cache/localmelo", "a\nb"],
)
def test_a_folder_that_may_not_be_kept_is_refused(bad: str, home: Path) -> None:
    assert config.extra_folder_problem(bad) is not None
    if bad.startswith(("~", "/")):
        # What the policy keeps is a real folder; one it may not keep raises.
        with pytest.raises(config.ConfigError):
            config.add_extra_folder(os.path.expanduser(bad))


def test_a_bad_value_is_kept_reported_and_never_used(home: Path) -> None:
    Path(config.config_dir()).mkdir(parents=True)
    Path(config.config_path()).write_text(
        'chat_backend = ""\nembedding_backend = ""\n'
        '[workspace]\nextra_folders = ["~", "~/ok", 3]\n'
    )
    cfg = config.load()
    assert cfg.workspace.extra_folders == ["~", "~/ok", 3]
    errors = cfg.setting_errors()
    assert any("'~'" in error for error in errors)
    assert any("3" in error and "string" in error for error in errors)
    assert config.extra_folders(cfg) == [str(home / "ok")]
    # It is not a gateway error: the gateway still starts.
    assert not any("extra_folders" in error for error in cfg.validate())


def test_not_a_list_is_reported(home: Path) -> None:
    Path(config.config_dir()).mkdir(parents=True)
    Path(config.config_path()).write_text('[workspace]\nextra_folders = "~/a"\n')
    cfg = config.load()
    assert cfg.setting_errors() == [
        "[workspace] extra_folders must be a list, got '~/a'"
    ]
    assert config.extra_folders(cfg) == []


def test_a_folder_that_became_a_symlink_covers_nothing_new(home: Path) -> None:
    career = home / "career"
    career.mkdir()
    config.add_extra_folder(str(career))
    career.rmdir()
    (home / "private").mkdir()
    career.symlink_to(home / "private")
    # The kept path is compared as kept: the link's target is another folder.
    assert config.extra_folders(config.load()) == [str(career)]
    assert os.path.realpath(career) == str(home / "private")


def test_the_setting_is_a_known_key() -> None:
    key = settings.catalog()["workspace.extra_folders"]
    assert key.section == "workspace" and key.file
