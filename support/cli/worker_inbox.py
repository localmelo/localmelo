"""Bounded append-only inboxes for a single live worker run.

A file inbox is append-only on disk; an fd inbox reads an inherited pipe whose
write end the parent keeps, so the worker's own tool subprocesses never get it.
Both share the line, size and acknowledgement rules.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from localmelo.melo.contracts.inbox import InboxError

MAX_INBOX_BYTES = 1024 * 1024
MAX_MESSAGE_BYTES = 16 * 1024


def _complete_messages(
    data: bytes, offset: int
) -> tuple[list[tuple[int, str]], list[int]]:
    """Parse complete LF-terminated messages after *offset*.

    The same rules for a file snapshot and a pipe: a partial last line waits,
    each line is nonempty UTF-8 of at most 16 KiB, and the total is capped
    before this is called. Returned offsets count bytes from the run's start.
    """
    messages: list[tuple[int, str]] = []
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
    return messages, [end for end, _ in messages]


class Inbox:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.identity: tuple[int, int] | None = None
        self.snapshot = b""
        self.offset = 0
        self.pending: list[int] = []

    def _check_location(self, workspace: str | None) -> None:
        """Refuse a path the worker's own file tools could write."""
        if workspace is None:
            return
        root = Path(os.path.realpath(os.path.expanduser(workspace)))
        target = Path(os.path.realpath(self.path))
        if target == root or root in target.parents:
            raise InboxError(
                "inbox is inside the agent workspace; move it outside the workspace"
            )

    def validate(self, workspace: str | None = None) -> None:
        """Check the path and target once before the first model call."""
        self._check_location(workspace)
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise InboxError("cannot open inbox as a regular file") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise InboxError("inbox is not a regular file")
            if info.st_size > MAX_INBOX_BYTES:
                raise InboxError("inbox exceeds 1 MiB")
        except OSError as exc:
            raise InboxError("cannot read inbox") from exc
        finally:
            os.close(fd)

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
        messages, self.pending = _complete_messages(data, self.offset)
        return messages

    def accept(self, offset: int) -> None:
        """Commit the next message only after insertion into live context."""
        if not self.pending or offset != self.pending[0]:
            raise InboxError("inbox messages must be accepted in order")
        self.pending.pop(0)
        self.offset = offset


class FdInbox:
    """An inherited pipe fd, shared with the parent and no tool subprocess.

    The parent keeps the write end. ``validate`` makes the read end
    close-on-exec before any tool provider starts, so python, shell and MCP
    children get no copy; the parent can append lines to the same channel.
    """

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.identity: tuple[int, int] | None = None
        self.snapshot = b""
        self.offset = 0
        self.pending: list[int] = []
        self.eof = False

    def _prepare(self) -> None:
        """Check it is still our pipe, then make it nonblocking and CLOEXEC."""
        try:
            info = os.fstat(self.fd)
        except OSError as exc:
            raise InboxError("cannot open inbox fd") from exc
        if not stat.S_ISFIFO(info.st_mode):
            raise InboxError("inbox fd is not a pipe")
        identity = (info.st_dev, info.st_ino)
        if self.identity is not None and identity != self.identity:
            raise InboxError("inbox fd was replaced")
        self.identity = identity
        try:
            os.set_inheritable(self.fd, False)
            os.set_blocking(self.fd, False)
        except OSError as exc:
            raise InboxError("cannot read inbox fd") from exc

    def validate(self, workspace: str | None = None) -> None:
        """Check the fd before the first model call; *workspace* is irrelevant."""
        self._prepare()

    def poll(self) -> list[tuple[int, str]]:
        """Read what the pipe holds, then return its complete messages."""
        self._prepare()
        while not self.eof:
            try:
                chunk = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            except OSError as exc:
                raise InboxError("cannot read inbox fd") from exc
            if not chunk:
                self.eof = True
                break
            self.snapshot += chunk
            if len(self.snapshot) > MAX_INBOX_BYTES:
                raise InboxError("inbox exceeds 1 MiB")
        messages, self.pending = _complete_messages(self.snapshot, self.offset)
        return messages

    def accept(self, offset: int) -> None:
        """Commit the next message only after insertion into live context."""
        if not self.pending or offset != self.pending[0]:
            raise InboxError("inbox messages must be accepted in order")
        self.pending.pop(0)
        self.offset = offset
