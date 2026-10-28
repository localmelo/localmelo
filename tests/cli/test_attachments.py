from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from localmelo.support.cli import attachments as att


def test_snapshot_is_immutable_and_redacted(tmp_path: Path) -> None:
    file = tmp_path / "note.txt"
    file.write_text("hello\x1b[31m sk-" + "x" * 24, encoding="utf-8")
    snapshot = att.load(str(tmp_path), "note.txt")
    file.write_text("different")
    assert snapshot.content == "hello[31m [redacted]"
    assert snapshot.changed
    assert snapshot.characters == len(snapshot.content)
    assert snapshot.size_bytes > snapshot.characters
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.content = "changed"  # type: ignore[misc]


def test_plain_unicode_and_absolute_path(tmp_path: Path) -> None:
    file = tmp_path / "note.txt"
    file.write_text("你好\n", encoding="utf-8")
    snapshot = att.load(str(tmp_path), str(file))
    assert (snapshot.path, snapshot.content, snapshot.size_bytes, snapshot.changed) == (
        "note.txt",
        "你好\n",
        7,
        False,
    )
    name = "x" * 201
    (tmp_path / name).write_text("text")
    with pytest.raises(att.AttachmentError, match="200 characters"):
        att.load(str(tmp_path), name)


@pytest.mark.parametrize("name", [".env", "credentials.json", "foo.key", ".ssh/id_rsa"])
def test_secret_files_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(att.AttachmentError):
        att.load(str(tmp_path), name)


def test_traversal_links_and_special_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    (root / "link").symlink_to(outside)
    (root / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(root / "fifo")
    for path in (
        "../secret.txt",
        str(outside),
        "link",
        "dirlink/secret.txt",
        "fifo",
        ".",
        "https://example.com/x",
    ):
        with pytest.raises(att.AttachmentError):
            att.load(str(root), path)


def test_bad_text_and_limits(tmp_path: Path) -> None:
    file = tmp_path / "note.txt"
    for data in (b"\xff", b"abc\x00def", b"x" * (att.MAX_FILE_BYTES + 1)):
        file.write_bytes(data)
        with pytest.raises(att.AttachmentError):
            att.load(str(tmp_path), "note.txt")
    file.write_bytes(b"x" * att.MAX_FILE_BYTES)
    snapshot = att.load(str(tmp_path), "note.txt")
    staged = att.add(att.add((), snapshot), snapshot)
    with pytest.raises(att.AttachmentError):
        att.add(staged, snapshot)
    file.write_text("")
    empty = att.load(str(tmp_path), "note.txt")
    for _ in range(6):
        staged = att.add(staged, empty)
    with pytest.raises(att.AttachmentError):
        att.add(staged, empty)


def test_file_mutated_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "note.txt"
    file.write_text("initial")
    real_read = os.read

    def read_and_mutate(fd: int, count: int) -> bytes:
        result = real_read(fd, count)
        if result:
            file.write_text("changed contents")
        return result

    monkeypatch.setattr(att.os, "read", read_and_mutate)
    with pytest.raises(att.AttachmentError):
        att.load(str(tmp_path), "note.txt")
