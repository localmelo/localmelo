"""Folders outside the workspace (#405): the places never offered, and what a grant covers.

The list of places melo never opens lives in ``melo.contracts.folders``
alone; these tests pin it: each place, a symlink into one, ``..`` tricks,
other spellings of the same place, and the home folder as such.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.melo.checker.review import paths as review_paths
from localmelo.melo.contracts.folders import (
    CREDENTIAL_PLACES,
    MELO_DATA,
    SYSTEM_PLACES,
    above_home,
    asked_text,
    covers,
    melo_data_folders,
    never_offered,
    never_text,
    outside_access,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary home folder, real path, set as HOME."""
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    return Path(os.path.realpath(folder))


# ── Places never offered ─────────────────────────────────────────────────────


@pytest.mark.parametrize("place", CREDENTIAL_PLACES)
def test_every_credential_place_is_never_offered(home: Path, place: str) -> None:
    assert never_offered(str(home / place)) == "credentials"
    assert never_offered(str(home / place / "inner")) == "credentials"


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/id_ed25519",
        ".gnupg",
        ".aws/credentials",
        ".config/gcloud/x.json",
        ".kube/config",
        ".docker/config.json",
        ".netrc",
        ".password-store/a.gpg",
        "Library/Keychains/login.keychain-db",
    ],
)
def test_the_named_places_of_the_issue_are_refused(home: Path, relative: str) -> None:
    assert never_offered(str(home / relative)) == "credentials"


def test_a_credential_place_counts_anywhere_as_whole_components(tmp_path: Path) -> None:
    # A backup of another home still holds keys.
    assert never_offered(str(tmp_path / "Backup/Users/x/.ssh/id_rsa")) == "credentials"
    # A name that only starts like one does not count.
    assert never_offered(str(tmp_path / "notes/.sshfs-mounts")) is None
    assert never_offered(str(tmp_path / "notes/.netrc.example")) is None


def test_melos_own_data_is_never_offered(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert never_offered(str(home / MELO_DATA)) == "melo"
    assert never_offered(str(home / MELO_DATA / "config.toml")) == "melo"
    other = tmp_path / "memory-elsewhere"
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(other))
    assert os.path.realpath(other) in melo_data_folders()
    assert never_offered(str(other / "history.db")) == "melo"


@pytest.mark.parametrize("place", [*SYSTEM_PLACES, "/dev", "/private/var/db"])
def test_system_folders_are_never_offered(home: Path, place: str) -> None:
    assert never_offered(place + "/x") == "system"


def test_usr_local_is_open_and_root_is_not(home: Path) -> None:
    assert never_offered("/usr/local/share/x") is None
    assert never_offered("/usr/bin/env") == "system"
    assert never_offered("/") == "system"


def test_the_whole_home_folder_and_above_are_never_offered(home: Path) -> None:
    assert never_offered(str(home)) == "home"
    assert never_offered(str(home.parent)) == "home"
    # A folder inside it is fine.
    assert never_offered(str(home / "developer" / "career")) is None
    assert above_home(str(home)) and above_home("/")
    assert not above_home(str(home / "developer"))


def test_a_symlink_into_a_place_counts_as_its_target(home: Path) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    (home / "notes").mkdir()
    link = home / "notes" / "innocent.txt"
    link.symlink_to(home / ".ssh" / "id_rsa")
    assert never_offered(str(link)) == "credentials"
    folder_link = home / "notes" / "keys"
    folder_link.symlink_to(home / ".ssh")
    assert never_offered(str(folder_link / "id_rsa")) == "credentials"


def test_dot_dot_tricks_are_resolved_first(home: Path) -> None:
    (home / "developer").mkdir()
    sneaky = str(home / "developer" / ".." / ".ssh" / "id_rsa")
    assert never_offered(sneaky) == "credentials"
    above = str(home / "developer" / ".." / "..")
    assert never_offered(above) == "home"
    assert never_offered("/usr/local/../bin/sh") == "system"


def test_another_case_or_unicode_form_is_still_refused(home: Path) -> None:
    assert never_offered(str(home / ".SSH" / "id_rsa")) == "credentials"
    assert never_offered(str(home / "library" / "KEYCHAINS")) == "credentials"
    assert never_offered("/SYSTEM/Library") == "system"


def test_the_review_reads_the_same_credential_list() -> None:
    """One list: the action review's key places are the never-offered ones."""
    listed = [p for p in review_paths._CREDENTIALS if not p.startswith("**/")]
    assert set(CREDENTIAL_PLACES) <= set(listed)
    assert MELO_DATA in listed
    assert list(SYSTEM_PLACES) == review_paths._SYSTEM


def test_the_refusal_text_says_nobody_is_asked() -> None:
    text = never_text("~/.ssh/id_rsa", "credentials")
    assert text.startswith("Path '~/.ssh/id_rsa' is outside the workspace")
    assert "the user is not asked" in text
    assert "whole home folder" in never_text("~", "home")


# ── What a grant covers ──────────────────────────────────────────────────────


def test_a_grant_covers_its_folder_and_below_on_a_path_boundary() -> None:
    assert covers("/a/b", "/a/b")
    assert covers("/a/b", "/a/b/c/d.txt")
    assert covers("/a/b/", "/a/b/c")
    assert not covers("/a/b", "/a/bc")
    assert not covers("/a/b", "/a/bc/d")
    assert not covers("/a/b", "/a")
    assert not covers("/", "/a/b")
    assert not covers("", "/a/b")
    assert not covers("/a/b", "relative/a/b")


def test_another_spelling_counts_only_when_it_is_the_same_folder(
    tmp_path: Path,
) -> None:
    real = Path(os.path.realpath(tmp_path)) / "Career"
    real.mkdir()
    (real / "cv.md").write_text("x")
    other = str(real.parent / "career" / "cv.md")
    same = os.path.exists(other)  # a case-insensitive file system
    assert covers(str(real), other) is same
    # A folder that does not exist under the other spelling never matches.
    assert not covers(str(real), str(real.parent / "CAREERS" / "cv.md"))


# ── The outside access of a call ─────────────────────────────────────────────


def test_a_read_offers_the_folder_that_holds_the_file(home: Path) -> None:
    career = home / "developer" / "career"
    career.mkdir(parents=True)
    (career / "cv.md").write_text("x")
    access = outside_access("file_read", str(career / "cv.md"))
    assert (access.path, access.folder, access.access, access.grantable) == (
        str(career / "cv.md"),
        str(career),
        "read",
        True,
    )
    listed = outside_access("list_files", str(career))
    assert listed.folder == str(career)


def test_a_write_is_never_grantable(home: Path) -> None:
    access = outside_access("file_write", str(home / "developer" / "new.md"))
    assert access.access == "write" and not access.grantable
    assert outside_access("file_edit", str(home / "x" / "y")).access == "write"


def test_a_file_directly_in_home_is_asked_once_only(home: Path) -> None:
    access = outside_access("file_read", str(home / "todo.txt"))
    assert access.folder == str(home)
    assert not access.grantable


def test_the_line_the_model_reads_names_the_folder(home: Path) -> None:
    read = outside_access("file_read", str(home / "career" / "cv.md"))
    assert asked_text(read) == (
        f"Reading {home / 'career'} outside the workspace needs the user's OK; "
        "they were asked."
    )
    write = outside_access("file_write", str(home / "career" / "cv.md"))
    assert asked_text(write).startswith(f"Writing to {home / 'career'} outside")
