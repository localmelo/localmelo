"""Bounded, private snapshots of workspace text attachments.

No symlinks are followed, even when their targets are inside the workspace.
Secret detection is best effort, using the same redactor as session storage;
callers must show the snapshot and obtain confirmation when ``changed`` is true.
"""

from __future__ import annotations

import os
import stat
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.support.sessions.worker import sensitive_read

MAX_ATTACHMENTS = 8
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 512 * 1024


class AttachmentError(ValueError):
    """A recoverable attachment refusal; messages do not contain file contents."""


@dataclass(frozen=True)
class Attachment:
    path: str  # Workspace-relative, sanitized display name, not a live file handle.
    content: str
    size_bytes: int  # Original UTF-8 bytes, before redaction/sanitization.
    characters: int  # Characters in the snapshot shown to the user.
    changed: bool


def _clean(text: str) -> str:
    return "".join(
        c for c in text if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cf"}
    )


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def load(workspace: str, path: str) -> Attachment:
    """Read one file once; refuse paths, files, or changes unsafe to snapshot.

    Absolute paths inside the workspace are accepted. Relative traversal and all
    symlinks are rejected. Errors are intentionally content-free. This function
    cannot prevent a concurrent writer that can manipulate filesystem timestamps.
    """
    if not path or "\x00" in path or "\x00" in workspace:
        raise AttachmentError("Invalid attachment path")
    if urlsplit(path).scheme or any(p == ".." for p in path.split(os.sep)):
        raise AttachmentError("URLs and parent traversal are not allowed")
    root = os.path.realpath(workspace)
    target = os.path.abspath(path if os.path.isabs(path) else os.path.join(root, path))
    try:
        relative = os.path.relpath(target, root)
        if relative == "." or os.path.commonpath((root, target)) != root:
            raise AttachmentError("Attachment must be inside the workspace")
    except ValueError as exc:
        raise AttachmentError("Attachment must be inside the workspace") from exc
    if sensitive_read("file_read", {"path": relative}, root):
        raise AttachmentError("Secret files cannot be attached")
    if not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise AttachmentError("Secure attachment loading is unavailable")

    parts = relative.split(os.sep)
    fds: list[int] = []
    entries: list[tuple[int, str, os.stat_result]] = []
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(root_fd)
        root_info = os.fstat(root_fd)
        parent = root_fd
        for part in parts[:-1]:
            fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            fds.append(fd)
            entries.append((parent, part, os.fstat(fd)))
            parent = fd
        fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        fds.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AttachmentError("Attachment must be a regular file")
        if before.st_size > MAX_FILE_BYTES:
            raise AttachmentError("Attachment exceeds 256 KiB")
        data = bytearray()
        while len(data) <= MAX_FILE_BYTES:
            chunk = os.read(fd, min(65536, MAX_FILE_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAX_FILE_BYTES:
            raise AttachmentError("Attachment exceeds 256 KiB")
        if (
            _identity(before) != _identity(os.fstat(fd))
            or _identity(before)
            != _identity(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False))
            or (root_info.st_dev, root_info.st_ino)
            != (
                os.stat(root, follow_symlinks=False).st_dev,
                os.stat(root, follow_symlinks=False).st_ino,
            )
            or any(
                (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)
                for directory, name, info in entries
                for current in [os.stat(name, dir_fd=directory, follow_symlinks=False)]
            )
        ):
            raise AttachmentError("Attachment changed during reading")
    except OSError as exc:
        raise AttachmentError("Cannot safely read attachment") from exc
    finally:
        for opened in reversed(fds):
            os.close(opened)
    try:
        text = bytes(data).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AttachmentError("Attachment is not valid UTF-8 text") from exc
    if "\x00" in text:
        raise AttachmentError("Binary attachments are not supported")
    clean = _clean(text)
    content, _ = redact_secrets(clean)
    safe_path, _ = redact_secrets(
        _clean(relative).replace("\n", " ").replace("\t", " ")
    )
    if not safe_path or len(safe_path) > 200:
        raise AttachmentError("Attachment name must be at most 200 characters")
    return Attachment(
        safe_path,
        content,
        len(data),
        len(content),
        content != text or safe_path != relative,
    )


def add(
    attachments: tuple[Attachment, ...], attachment: Attachment
) -> tuple[Attachment, ...]:
    """Return a new staged tuple, enforcing count and original-byte budgets."""
    if len(attachments) >= MAX_ATTACHMENTS:
        raise AttachmentError("At most 8 attachments are allowed")
    if (
        sum(item.size_bytes for item in attachments) + attachment.size_bytes
        > MAX_TOTAL_BYTES
    ):
        raise AttachmentError("Attachments exceed 512 KiB in total")
    return (*attachments, attachment)
