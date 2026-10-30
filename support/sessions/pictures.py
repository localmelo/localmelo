"""Pictures a console turn makes: finding them, and checking one before it is served.

A tool that writes a file (``file_write``, ``python_exec``, ``shell_exec``) leaves
no record of which file it wrote, so a turn looks at its folder twice:
:func:`scan` at the start and at the end, and :func:`new_pictures` lists the
picture files that appeared or changed in between. :func:`mentioned` adds the
picture paths the answer names, when they are inside the folder. The runner
stores the list as the turn's ``turn.pictures`` event, so a reopened chat has it.

:func:`read` is the one door the console serves a picture through. It refuses a
path whose real location is outside the folder (``..``, an absolute path, a link
that leads out), a file that is not a regular file, one whose name or content is
not a PNG, JPEG, GIF, WebP or SVG, and one over :data:`MAX_PICTURE_BYTES`.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

# What a picture file is called, by its suffix; the content has to agree (see sniff).
PICTURE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})
MAX_PICTURE_BYTES: Final = 20 * 1024 * 1024
# One scan visits at most this many directory entries, in folders up to this deep.
SCAN_MAX_ENTRIES: Final = 4_000
SCAN_MAX_DEPTH: Final = 3
# Folders a scan never enters, besides the hidden ones.
SKIP_DIRS: Final = frozenset({"node_modules", "__pycache__", "site-packages"})
# Pictures one turn lists, and picture paths read out of one answer.
MAX_LISTED: Final = 24
MAX_MENTIONS: Final = 8

# Relative path (with "/") -> (size in bytes, mtime in nanoseconds).
Snapshot = dict[str, tuple[int, int]]

_MENTION = re.compile(
    r"[^\s\"'`()<>\[\]{}|*?]+\.(?:png|jpe?g|gif|webp|svg)(?![\w/-]|\.\w)", re.IGNORECASE
)


class PictureError(Exception):
    """A picture the console will not serve: the status, API code and short reason."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Picture:
    """A picture's bytes, with the type its content showed."""

    mime: str
    data: bytes


# ── Finding the pictures of a turn ──


def scan(root: str) -> Snapshot:
    """The picture files under *root*: shallow, bounded, never through a link.

    Hidden folders, ``node_modules`` and similar folders are skipped; at most
    :data:`SCAN_MAX_ENTRIES` entries are visited, widest levels first. A folder
    that cannot be read is left out.
    """
    found: Snapshot = {}
    queue: deque[tuple[str, str, int]] = deque([(root, "", 0)])
    visited = 0
    while queue and visited < SCAN_MAX_ENTRIES:
        folder, prefix, depth = queue.popleft()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    visited += 1
                    if visited > SCAN_MAX_ENTRIES:
                        break
                    name = entry.name
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if (
                                depth < SCAN_MAX_DEPTH
                                and not name.startswith(".")
                                and name not in SKIP_DIRS
                            ):
                                queue.append(
                                    (entry.path, f"{prefix}{name}/", depth + 1)
                                )
                        elif (
                            os.path.splitext(name)[1].lower() in PICTURE_SUFFIXES
                            and entry.is_file(follow_symlinks=False)
                            and not name.startswith(".")
                        ):
                            info = entry.stat(follow_symlinks=False)
                            found[f"{prefix}{name}"] = (info.st_size, info.st_mtime_ns)
                    except OSError:
                        continue
        except OSError:
            continue
    return found


def new_pictures(before: Snapshot, after: Snapshot) -> list[dict[str, Any]]:
    """The pictures of *after* that *before* lacks (``new``) or that differ (``modified``)."""
    listed: list[tuple[int, str, dict[str, Any]]] = []
    for path, (size, mtime_ns) in after.items():
        old = before.get(path)
        if old == (size, mtime_ns):
            continue
        state = "new" if old is None else "modified"
        listed.append(
            (
                mtime_ns,
                path,
                {"path": path, "size": size, "mtime": mtime_ns / 1e9, "state": state},
            )
        )
    listed.sort(key=lambda item: item[:2])
    return [item[2] for item in listed[:MAX_LISTED]]


def mentioned(root: str, text: str, known: set[str]) -> list[dict[str, Any]]:
    """Picture files that *text* names and that are inside *root*, not in *known*.

    A name is a relative path or an absolute one; it counts only when it names
    an existing regular file whose real location is inside *root*.
    """
    base = os.path.realpath(root)
    found: list[dict[str, Any]] = []
    seen = set(known)
    for match in _MENTION.finditer(text):
        raw = match.group(0)
        if raw.startswith("~/"):
            raw = os.path.expanduser(raw)
        try:
            real = os.path.realpath(
                raw if os.path.isabs(raw) else os.path.join(base, raw)
            )
            if not _inside(real, base):
                continue
            info = os.stat(real)
        except (OSError, ValueError):
            continue
        relative = os.path.relpath(real, base).replace(os.sep, "/")
        if not stat.S_ISREG(info.st_mode) or relative in seen:
            continue
        seen.add(relative)
        found.append(
            {
                "path": relative,
                "size": info.st_size,
                "mtime": info.st_mtime_ns / 1e9,
                "state": "mentioned",
            }
        )
        if len(found) >= MAX_MENTIONS:
            break
    return found


# ── Serving one ──


def _inside(real: str, base: str) -> bool:
    """Whether *real* is *base* or below it; both are real paths."""
    try:
        return real != base and os.path.commonpath([real, base]) == base
    except ValueError:
        return False


def sniff(head: bytes) -> str | None:
    """The picture type the first bytes of a file show, or ``None``."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if _is_svg(head):
        return "image/svg+xml"
    return None


def _is_svg(head: bytes) -> bool:
    """Whether the text starts, after an XML declaration, comments and a doctype, with ``<svg``."""
    text = head.removeprefix(b"\xef\xbb\xbf").decode("utf-8", "ignore").lstrip()
    for _ in range(8):  # a few prolog items; no pattern to backtrack over
        lowered = text[:9].lower()
        if lowered.startswith("<?xml"):
            end = text.find("?>")
            skip = 2
        elif lowered.startswith("<!--"):
            end = text.find("-->")
            skip = 3
        elif lowered.startswith("<!doctype"):
            end = text.find(">")
            if end >= 0 and "[" in text[:end]:  # an internal subset ends with "]>"
                end = text.find("]", end)
                end = text.find(">", end) if end >= 0 else end
            skip = 1
        else:
            break
        if end < 0:
            return False
        text = text[end + skip :].lstrip()
    return text[:4].lower() == "<svg" and text[4:5] in (
        "",
        " ",
        "\t",
        "\r",
        "\n",
        "/",
        ">",
    )


def _opened_path(fd: int) -> str | None:
    """Where the open file *fd* really is, when the system can say; else ``None``."""
    try:
        if sys.platform == "darwin":
            import fcntl

            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 1024)
            return os.fsdecode(raw.split(b"\0", 1)[0])
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, ImportError, AttributeError):
        return None


def read(workspace: str, requested: str) -> Picture:
    """The picture *requested* names, relative to *workspace* or absolute, or :class:`PictureError`.

    The real path (links followed) must be inside the real workspace, and is
    checked again on the open file; the file must be a regular one, of a picture
    name, whose content shows the same kind of picture, up to
    :data:`MAX_PICTURE_BYTES`.
    """
    if not requested or "\0" in requested:
        raise PictureError(404, "not_found", "No such picture.")
    if ".." in PurePosixPath(requested).parts:
        raise PictureError(403, "outside_workspace", "That path leaves the folder.")
    base = os.path.realpath(workspace)
    real = os.path.realpath(
        requested if os.path.isabs(requested) else os.path.join(base, requested)
    )
    if not _inside(real, base):
        raise PictureError(
            403, "outside_workspace", "That file is not in this chat's folder."
        )
    if os.path.splitext(real)[1].lower() not in PICTURE_SUFFIXES:
        raise PictureError(
            415, "not_a_picture", "Only PNG, JPEG, GIF, WebP and SVG are shown."
        )
    try:
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise PictureError(404, "not_found", "No such picture.") from None
    except OSError:
        raise PictureError(
            403, "outside_workspace", "That file cannot be opened."
        ) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PictureError(415, "not_a_picture", "That is not a file.")
        opened = _opened_path(fd)
        if opened is not None and not _inside(os.path.realpath(opened), base):
            raise PictureError(
                403, "outside_workspace", "That file is not in this chat's folder."
            )
        if info.st_size > MAX_PICTURE_BYTES:
            raise PictureError(413, "too_large", "That picture is over 20 MB.")
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_PICTURE_BYTES:
            chunk = os.read(fd, min(1 << 20, MAX_PICTURE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    finally:
        os.close(fd)
    if total > MAX_PICTURE_BYTES:
        raise PictureError(413, "too_large", "That picture is over 20 MB.")
    data = b"".join(chunks)
    mime = sniff(data[:4096])
    if mime is None:
        raise PictureError(415, "not_a_picture", "That file is not a picture.")
    return Picture(mime, data)
