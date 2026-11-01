"""Folders outside the workspace (#405): the places never offered, and what a grant covers.

The list of places melo never opens lives in ``melo.contracts.folders``
alone; these tests pin it: each place, a symlink into one, ``..`` tricks,
other spellings of the same place, and the home folder as such.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from localmelo.melo.checker.review import paths as review_paths
from localmelo.melo.contracts.folders import (
    CREDENTIAL_PLACES,
    MELO_DATA,
    SYSTEM_PLACES,
    PathChangedError,
    above_home,
    asked_text,
    canonical_path,
    covers,
    fd_path,
    home_label,
    is_alias,
    melo_data_folders,
    never_offered,
    never_text,
    open_checked,
    outside_access,
)

# macOS names a file by number under /.vol, and a path another way under
# /.resolve and /.nofollow; realpath leaves each as it is.
ALIASES = os.path.isdir("/.vol") and sys.platform == "darwin"
needs_aliases = pytest.mark.skipif(not ALIASES, reason="no file-system aliases here")


def vol(path: Path) -> str:
    """``/.vol/<device>/<inode>``: *path* named by number."""
    info = os.stat(path)
    return f"/.vol/{info.st_dev}/{info.st_ino}"


def alias_forms(folder: Path, name: str) -> list[str]:
    """The four alias forms of *folder*/*name*, and one in another case."""
    path = folder / name
    return [
        vol(path),
        f"{vol(folder)}/{name}",
        f"/.resolve/1{path}",
        f"/.nofollow{path}",
        "/.VOL" + vol(path)[len("/.vol") :],
    ]


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
    assert never_offered(str(home / "Documents" / "letters")) is None
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
    real = Path(os.path.realpath(tmp_path)) / "Letters"
    real.mkdir()
    (real / "cover-letter.md").write_text("x")
    other = str(real.parent / "letters" / "cover-letter.md")
    same = os.path.exists(other)  # a case-insensitive file system
    assert covers(str(real), other) is same
    # A folder that does not exist under the other spelling never matches.
    assert not covers(str(real), str(real.parent / "LETTER-BOX" / "cover-letter.md"))


# ── The outside access of a call ─────────────────────────────────────────────


def test_a_read_offers_the_folder_that_holds_the_file(home: Path) -> None:
    letters = home / "Documents" / "letters"
    letters.mkdir(parents=True)
    (letters / "cover-letter.md").write_text("x")
    access = outside_access("file_read", str(letters / "cover-letter.md"))
    assert (access.path, access.folder, access.access, access.grantable) == (
        str(letters / "cover-letter.md"),
        str(letters),
        "read",
        True,
    )
    listed = outside_access("list_files", str(letters))
    assert listed.folder == str(letters)


def test_a_write_is_never_grantable(home: Path) -> None:
    access = outside_access("file_write", str(home / "developer" / "new.md"))
    assert access.access == "write" and not access.grantable
    assert outside_access("file_edit", str(home / "x" / "y")).access == "write"


def test_a_file_directly_in_home_is_asked_once_only(home: Path) -> None:
    access = outside_access("file_read", str(home / "todo.txt"))
    assert access.folder == str(home)
    assert not access.grantable


def test_the_line_the_model_reads_names_the_folder(home: Path) -> None:
    read = outside_access("file_read", str(home / "letters" / "cover-letter.md"))
    assert asked_text(read) == (
        f"Reading {home / 'letters'} outside the workspace needs the user's OK; "
        "they were asked."
    )
    write = outside_access("file_write", str(home / "letters" / "cover-letter.md"))
    assert asked_text(write).startswith(f"Writing to {home / 'letters'} outside")


def test_a_path_below_home_reads_with_a_tilde(home: Path) -> None:
    assert (
        home_label(str(home / "developer" / "cover-letter.md"))
        == "~/developer/cover-letter.md"
    )
    assert home_label(str(home)) == "~"
    assert home_label(str(home) + "-other/x") == str(home) + "-other/x"
    assert home_label("/opt/data/x") == "/opt/data/x"


# ── Canonical paths and aliases ──────────────────────────────────────────────


def test_a_case_variant_is_named_as_the_file_system_keeps_it(tmp_path: Path) -> None:
    real = Path(os.path.realpath(tmp_path)) / "Letters"
    (real / "Notes").mkdir(parents=True)
    (real / "Notes" / "cover-letter.md").write_text("x")
    other = str(real.parent / "LETTERS" / "notes" / "COVER-LETTER.md")
    if os.path.exists(other):  # a case-insensitive file system
        assert canonical_path(other) == str(real / "Notes" / "cover-letter.md")
    # A part that does not exist yet keeps its spelling below the named folder.
    assert canonical_path(str(real / "Notes" / "New" / "a.md")) == str(
        real / "Notes" / "New" / "a.md"
    )


def test_a_fifo_is_never_opened_to_name_it(tmp_path: Path) -> None:
    folder = Path(os.path.realpath(tmp_path))
    os.mkfifo(folder / "pipe")
    # Opening a FIFO to read would wait for a writer; it is named by its folder.
    assert canonical_path(str(folder / "pipe")) == str(folder / "pipe")


@needs_aliases
@pytest.mark.parametrize("form", range(5))
def test_an_alias_into_a_key_folder_is_named_and_refused(home: Path, form: int) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    alias = alias_forms(home / ".ssh", "id_rsa")[form]
    assert is_alias(alias)
    assert never_offered(alias) is not None
    assert never_offered(canonical_path(alias)) is not None


@needs_aliases
@pytest.mark.parametrize("form", range(5))
def test_an_alias_is_never_offered_even_to_a_harmless_folder(
    home: Path, form: int
) -> None:
    (home / "notes").mkdir()
    (home / "notes" / "a.txt").write_text("a")
    alias = alias_forms(home / "notes", "a.txt")[form]
    assert never_offered(alias) == "system"
    if form < 4:  # the file system names the four real forms
        assert canonical_path(alias) == str(home / "notes" / "a.txt")
    access = outside_access("file_read", alias)
    if is_alias(access.folder):
        assert not access.grantable
    assert not covers(os.path.dirname(alias), alias)


@needs_aliases
def test_a_volume_alias_folder_covers_nothing(home: Path) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    device = os.stat(home).st_dev
    assert not covers(f"/.vol/{device}", vol(home / ".ssh" / "id_rsa"))
    assert not covers(f"/.vol/{device}", str(home / ".ssh" / "id_rsa"))
    assert is_alias(f"/.Vol/{device}") and not is_alias("/volumes/x")


@pytest.mark.parametrize(
    "relative",
    [
        "Library/Application Support/Arc/User Data",
        "Library/Application Support/Chromium/Default",
        "Library/Application Support/Vivaldi",
        "Library/Application Support/com.operasoftware.Opera",
        "Library/Application Support/Slack/Cookies",
        ".config/github-copilot/hosts.json",
        ".claude/.credentials.json",
        ".codex/auth.json",
        ".gemini/oauth_creds.json",
        ".terraform.d/credentials.tfrc.json",
        ".cargo/credentials.toml",
        ".vault-token",
        ".oci/config",
        ".config/cloudflared/cert.pem",
    ],
)
def test_more_sign_in_places_are_never_offered(home: Path, relative: str) -> None:
    assert never_offered(str(home / relative)) == "credentials"


def test_fd_path_names_the_open_file(tmp_path: Path) -> None:
    path = Path(os.path.realpath(tmp_path)) / "a.txt"
    path.write_text("x")
    fd = os.open(path, os.O_RDONLY)
    try:
        named = fd_path(fd)
    finally:
        os.close(fd)
    assert named in (str(path), None)  # None where the system cannot say


def test_open_checked_refuses_a_folder_swapped_after_the_check(
    tmp_path: Path,
) -> None:
    root = Path(os.path.realpath(tmp_path))
    (root / "letters").mkdir()
    (root / "letters" / "cover-letter.md").write_text("letter")
    (root / "keys").mkdir()
    (root / "keys" / "cover-letter.md").write_text("key")
    checked = canonical_path(str(root / "letters" / "cover-letter.md"))
    fd = os.open(root / "letters" / "cover-letter.md", os.O_RDONLY)
    nameable = fd_path(fd) is not None
    os.close(fd)
    # A part of the path is swapped for a symlink after the check.
    (root / "letters").rename(root / "letters-old")
    (root / "letters").symlink_to(root / "keys")
    if not nameable:
        pytest.skip("this system cannot name a descriptor")
    with pytest.raises(PathChangedError, match="changed before it was opened"):
        open_checked(str(root / "letters" / "cover-letter.md"), os.O_RDONLY, checked)
    # Without a check to keep, the open stands as it always did.
    os.close(open_checked(str(root / "letters" / "cover-letter.md"), os.O_RDONLY, None))
