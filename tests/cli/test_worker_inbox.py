from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.support.cli.worker_inbox import MAX_INBOX_BYTES, Inbox, InboxError


def test_missing_then_complete_messages_and_partial_utf8(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    inbox = Inbox(path)
    assert inbox.poll() == []
    path.write_bytes(b"first\n\xe2\x82")
    assert inbox.poll() == [(6, "first")]
    inbox.accept(6)
    with path.open("ab") as file:
        file.write(b"\xac\nsecond\npartial")
    assert inbox.poll() == [(10, "€"), (17, "second")]
    inbox.accept(10)
    inbox.accept(17)
    assert inbox.poll() == []
    with path.open("ab") as file:
        file.write(b"!\n")
    assert inbox.poll() == [(26, "partial!")]
    inbox.accept(26)
    assert inbox.poll() == []


def test_unaccepted_message_is_not_consumed(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    path.write_text("one\ntwo\n", encoding="utf-8")
    inbox = Inbox(path)
    assert inbox.poll() == [(4, "one"), (8, "two")]
    inbox.accept(4)
    assert inbox.poll() == [(8, "two")]
    with pytest.raises(InboxError):
        inbox.accept(4)


@pytest.mark.parametrize("content", [b"\n", b" \n", b"\xff\n", b"x" * 16385 + b"\n"])
def test_invalid_complete_message_fails(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "inbox"
    path.write_bytes(content)
    with pytest.raises(InboxError):
        Inbox(path).poll()


def test_overlong_partial_and_file_fail(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    path.write_bytes(b"x" * 16385)
    with pytest.raises(InboxError):
        Inbox(path).poll()
    path.write_bytes(b"a" * (MAX_INBOX_BYTES + 1))
    with pytest.raises(InboxError):
        Inbox(path).poll()


@pytest.mark.parametrize("change", ["truncate", "rewrite", "replace", "delete"])
def test_changed_inbox_fails(tmp_path: Path, change: str) -> None:
    path = tmp_path / "inbox"
    path.write_bytes(b"first\npartial")
    inbox = Inbox(path)
    inbox.poll()
    inbox.accept(6)
    if change == "truncate":
        path.write_bytes(b"first\n")
    elif change == "rewrite":
        path.write_bytes(b"first\nchanged")
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"first\npartial\n")
        replacement.replace(path)
    else:
        path.unlink()
    with pytest.raises(InboxError):
        inbox.poll()


def test_refuses_symlink_and_nonregular_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("task\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path in (link, fifo, tmp_path):
        with pytest.raises(InboxError):
            Inbox(path).poll()
