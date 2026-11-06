"""A chat's work folder: private, named after the chat, and nothing a link or a page's name can steer (#551)."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from localmelo.melo.contracts import work_folder


def _mode(path: str) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_the_folder_is_made_private_under_the_temporary_folder_and_removed() -> None:
    path = work_folder.ensure("abc123")
    assert path == work_folder.path_for("abc123")
    assert path.startswith(os.path.realpath(tempfile.gettempdir()))
    assert path.endswith("/localmelo/abc123") and _mode(path) == 0o700
    (Path(path) / "a.txt").write_text("x")
    assert work_folder.remove("abc123") and not os.path.exists(path)
    assert not work_folder.remove("abc123"), "nothing left to remove"


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "-x", "a" * 65, "a b"])
def test_an_id_that_is_no_plain_name_cannot_name_another_place(bad: str) -> None:
    with pytest.raises(ValueError):
        work_folder.ensure(bad)
    assert not work_folder.usable(bad) and not work_folder.remove(bad)


def test_a_looser_folder_is_made_private_and_a_link_is_refused(tmp_path: Path) -> None:
    path = work_folder.ensure("abc123")
    os.chmod(path, 0o755)
    work_folder.ensure("abc123")
    assert _mode(path) == 0o700
    work_folder.remove("abc123")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, Path(path))  # a link where the chat's folder should be
    with pytest.raises(work_folder.WorkFolderError):
        work_folder.ensure("abc123")
    assert not work_folder.remove("abc123") and elsewhere.exists(), "a link is left"


def test_holds_resolves_links_and_dot_dot_first(tmp_path: Path) -> None:
    path = work_folder.ensure("abc123")
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    os.symlink(outside, Path(path) / "link.txt")
    assert work_folder.holds("abc123", os.path.join(path, "a.pdf"))
    assert not work_folder.holds("abc123", os.path.join(path, "link.txt"))
    assert not work_folder.holds("abc123", os.path.join(path, "..", "other", "a.pdf"))
    assert not work_folder.holds("../x", path)


def _saved(tmp_path: Path, name: str, text: str = "data") -> str:
    folder = tmp_path / "downloads"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(text)
    return str(folder)


@pytest.mark.parametrize(
    ("page_name", "ending"),
    [("Report.PDF", ".pdf"), ("x.command", ""), ("x.app", ""), ("x", ""), ("a.sh", "")],
)
def test_a_download_keeps_the_browsers_id_and_only_a_harmless_ending(
    tmp_path: Path, page_name: str, ending: str
) -> None:
    folder = _saved(tmp_path, "3f2a-9c")
    kept = work_folder.take("abc123", folder, "3f2a-9c", page_name)
    assert kept == os.path.join(work_folder.path_for("abc123"), "3f2a-9c" + ending)
    assert Path(kept).read_text() == "data" and _mode(kept) == 0o600
    assert not os.path.exists(os.path.join(folder, "3f2a-9c")), "moved, not copied"


def test_a_link_or_a_name_with_a_path_is_never_moved(tmp_path: Path) -> None:
    folder = _saved(tmp_path, "real", "secret")
    os.symlink(os.path.join(folder, "real"), os.path.join(folder, "link"))
    assert work_folder.take("abc123", folder, "link", "a.pdf") is None
    assert work_folder.take("abc123", folder, "../downloads/real", "a.pdf") is None
    assert not os.path.exists(work_folder.path_for("abc123")), "nothing was made for it"
    assert (Path(folder) / "real").read_text() == "secret"
