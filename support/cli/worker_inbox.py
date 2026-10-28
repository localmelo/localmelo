"""Bounded append-only inbox for a single live worker run."""

from __future__ import annotations

import os
import stat
from pathlib import Path

MAX_INBOX_BYTES = 1024 * 1024
MAX_MESSAGE_BYTES = 16 * 1024


class InboxError(ValueError):
    """The inbox cannot be read safely as an append-only file."""


class Inbox:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.identity: tuple[int, int] | None = None
        self.snapshot = b""
        self.offset = 0
        self.pending: list[int] = []

    def poll(self) -> list[tuple[int, str]]:
        """Snapshot complete messages; caller explicitly commits each insertion."""
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except FileNotFoundError as exc:
            if self.identity is None:
                return []
            raise InboxError("inbox disappeared") from exc
        except OSError as exc:
            raise InboxError("cannot open inbox as a regular file") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise InboxError("inbox is not a regular file")
            identity = (info.st_dev, info.st_ino)
            if self.identity is not None and identity != self.identity:
                raise InboxError("inbox was replaced")
            if info.st_size > MAX_INBOX_BYTES:
                raise InboxError("inbox exceeds 1 MiB")
            chunks = []
            remaining = info.st_size
            while remaining:
                chunk = os.read(fd, min(remaining, 65536))
                if not chunk:
                    raise InboxError("inbox was truncated while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            current = os.stat(self.path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != identity:
                raise InboxError("inbox was replaced while reading")
        except OSError as exc:
            raise InboxError("cannot read inbox") from exc
        finally:
            os.close(fd)
        if not data.startswith(self.snapshot):
            raise InboxError("inbox was truncated or modified")
        self.identity = identity
        self.snapshot = data
        messages = []
        offset = self.offset
        parts = data[offset:].split(b"\n")
        if len(parts[-1]) > MAX_MESSAGE_BYTES:
            raise InboxError("inbox message exceeds 16 KiB")
        for raw in parts[:-1]:
            if len(raw) > MAX_MESSAGE_BYTES:
                raise InboxError("inbox message exceeds 16 KiB")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InboxError("inbox message is not UTF-8") from exc
            if not text.strip():
                raise InboxError("inbox message is empty")
            offset += len(raw) + 1
            messages.append((offset, text))
        self.pending = [end for end, _ in messages]
        return messages

    def accept(self, offset: int) -> None:
        """Commit the next message only after insertion into live context."""
        if not self.pending or offset != self.pending[0]:
            raise InboxError("inbox messages must be accepted in order")
        self.pending.pop(0)
        self.offset = offset
