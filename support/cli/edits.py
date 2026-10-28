"""Reading a ``file_write`` target, and the diff the approval panel shows.

The panel's body is a unified diff of the file as it is now and the text
about to be written (section 6 of the U2 spec), and a file write's feed
block shows the counts. :func:`read_target` reads the old text the way
``display.approval_target`` reads the facts: with
``O_RDONLY | O_NOFOLLOW | O_NONBLOCK``, so a final symlink is not followed
and a FIFO never blocks.
"""

from __future__ import annotations

import difflib
import os
import stat
from dataclasses import dataclass
from typing import Final, Literal

from localmelo.support.sessions.display import file_facts

# A larger file gets no diff: the panel would not show it whole either.
DIFF_MAX_BYTES: Final = 200_000


@dataclass(frozen=True)
class Target:
    """What the path holds now, for the panel and the block.

    ``text`` is the file's text when it is a regular, text file of at most
    :data:`DIFF_MAX_BYTES`; else ``problem`` says why there is no diff
    (``"is not a regular file"``, ``"is not text"``, or
    ``"overwrites a <size> file"``).
    """

    path: str
    exists: bool
    size_bytes: int = 0
    text: str | None = None
    problem: str = ""

    @property
    def diffable(self) -> bool:
        return self.text is not None


@dataclass(frozen=True)
class DiffLine:
    """One row of a diff: its kind, the two line numbers, and its text."""

    kind: Literal["+", "-", " ", "…"]
    old: int | None
    new: int | None
    text: str


def read_target(path: str, *, max_bytes: int = DIFF_MAX_BYTES) -> Target:
    """Read the file at *path* for a diff; never follows a final symlink."""
    exists, size = file_facts(path)
    if not exists:
        return Target(path=path, exists=False)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except PermissionError:
        return Target(path=path, exists=True, size_bytes=size, problem="cannot be read")
    except OSError:
        # A symlink (ELOOP), a path below a file: the file is there, but its
        # text cannot be shown.
        return Target(
            path=path, exists=True, size_bytes=size, problem="is not a regular file"
        )
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return Target(
                path=path,
                exists=True,
                size_bytes=info.st_size,
                problem="is not a regular file",
            )
        if info.st_size > max_bytes:
            from localmelo.support.sessions.display import format_bytes

            return Target(
                path=path,
                exists=True,
                size_bytes=info.st_size,
                problem=f"overwrites a {format_bytes(info.st_size)} file",
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # the file object owns it now
            data = handle.read(max_bytes + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return Target(
            path=path,
            exists=True,
            size_bytes=len(data),
            problem="is not text",
        )
    return Target(path=path, exists=True, size_bytes=len(data), text=text)


def diff(old: str, new: str) -> tuple[list[DiffLine], int, int]:
    """The diff of *old* and *new*, and its added and removed line counts.

    A unified diff with 2 context lines and a ``…`` row between hunks, as
    the panel shows it. A removed line carries the old line's number, any
    other row the new one.
    """
    # Keep the line endings: a CRLF rewrite or a missing final newline is
    # a change the panel must show (F2).
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    rows: list[DiffLine] = []
    added = removed = 0
    old_number = new_number = 0
    in_hunk = False
    for line in difflib.unified_diff(
        old_lines, new_lines, n=2, fromfile="", tofile="", lineterm=""
    ):
        if line.startswith("@@"):
            in_hunk = True
            # @@ -5,3 +5,4 @@
            header = line.split(" ")
            old_start = int(header[1][1:].split(",")[0])
            new_start = int(header[2][1:].split(",")[0])
            old_number, new_number = old_start, new_start
            if rows:
                rows.append(DiffLine("…", None, None, ""))
            continue
        if not in_hunk:
            # Only difflib's own ---/+++ header lines come before the first
            # hunk; a changed line may start with them after it (F1).
            continue
        sign, text = line[:1], line[1:].rstrip("\r\n")
        if sign == "-":
            rows.append(DiffLine("-", old_number, None, text))
            old_number += 1
            removed += 1
        elif sign == "+":
            rows.append(DiffLine("+", None, new_number, text))
            new_number += 1
            added += 1
        else:
            rows.append(DiffLine(" ", old_number, new_number, text))
            old_number += 1
            new_number += 1
    return rows, added, removed
