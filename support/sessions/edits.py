"""Reading a ``file_write`` target, and the diff the approval panels show.

The terminal panel's body is a unified diff of the file as it is now and the
text about to be written (section 6 of the U2 spec), and a file write's feed
block shows the counts. :func:`read_target` reads the old text the way
``display.approval_target`` reads the facts: with
``O_RDONLY | O_NOFOLLOW | O_NONBLOCK``, so a final symlink is not followed
and a FIFO never blocks. The web console's approval card shows the same
diff, bounded and redacted, from :func:`approval_diff`.
"""

from __future__ import annotations

import difflib
import os
import stat
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, Literal

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.executor.builtins import edited_text
from localmelo.support.sessions.display import file_facts

# A larger file gets no diff: the panel would not show it whole either.
DIFF_MAX_BYTES: Final = 200_000
# The console's card gets at most this many diff rows; a bigger change is told
# in words.
DIFF_MAX_ROWS: Final = 1_000


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


# The console's words for a row kind, in place of the diff's one-character signs.
_ROW_KINDS: Final = {"+": "added", "-": "removed", " ": "context", "…": "gap"}
# Why a target's text could not be read, as `Target.problem` says it.
_WHY: Final = {
    "is not text": "the file is not text",
    "cannot be read": "the file cannot be read",
    "is not a regular file": "it is not a regular file",
}
_TOO_MANY: Final = "too many lines change"


def approval_diff(path: str, content: str) -> dict[str, Any]:
    """What the console's approval card shows of ``file_write`` replacing *path*.

    ``{"added", "removed", "rows", "note"}``: *rows* are those of :func:`diff`
    as ``{"kind", "line", "text"}``, where ``kind`` is ``removed``, ``added``,
    ``context`` or ``gap`` and ``line`` is the old line's number for a removed
    row and the new line's for any other (``None`` for a gap). Secrets are
    hidden as ``redact_secrets`` hides them, in every row; the rows come from
    the texts as they are, so a line that only changes in a hidden value is
    still a removed row and an added one.

    The diff is whole or absent: no text over :data:`DIFF_MAX_BYTES` and at
    most :data:`DIFF_MAX_ROWS` rows, and no line is cut. When melo cannot
    compare (the file is not text, cannot be read or is not a regular file; a
    text is too large; too many lines change; or a text holds a private key
    block, which redaction shrinks to one line) *rows* is empty and *note*
    says why in words. Otherwise *note* is ``""``.
    """
    target = read_target(path)
    old = target.text
    if old is None:
        return _not_shown(_why(target.problem))
    if len(content) > DIFF_MAX_BYTES:
        return _not_shown("the new text is too large")
    old_shown, new_shown = _shown(old), _shown(content)
    if old_shown is None or new_shown is None:
        return _not_shown("the file or the new text holds a private key")
    # The lines that only one text has are all rows. Counting them first
    # spares a diff, whose cost grows with its changes, that would not show.
    if _fewest_rows(old, content) > DIFF_MAX_ROWS:
        return _not_shown(_TOO_MANY)
    rows, added, removed = diff(old, content)
    if len(rows) > DIFF_MAX_ROWS:
        return _not_shown(_TOO_MANY)
    shown: list[dict[str, Any]] = []
    for row in rows:
        if row.kind == "…":
            shown.append({"kind": "gap", "line": None, "text": ""})
            continue
        number = row.old if row.kind == "-" else row.new
        assert number is not None  # every row but a gap has its line's number
        lines = old_shown if row.kind == "-" else new_shown
        shown.append(
            {"kind": _ROW_KINDS[row.kind], "line": number, "text": lines[number - 1]}
        )
    return {"added": added, "removed": removed, "rows": shown, "note": ""}


def edit_diff(path: str, old: object, new: object) -> dict[str, Any]:
    """What a ``file_edit`` of *path* would change, as :func:`approval_diff` shows it.

    The edit is applied to the file's text as it is now; when it cannot be
    (the text to replace is missing or not unique), *note* says so.
    """
    target = read_target(path)
    if target.text is None:
        return _not_shown(_why(target.problem))
    if not isinstance(old, str) or not isinstance(new, str):
        return _not_shown("the edit has no text to replace")
    try:
        content = edited_text(target.text, old, new, "the file")
    except ValueError as exc:
        return _not_shown(str(exc).split(";")[0].lower())
    return approval_diff(path, content)


def _not_shown(why: str) -> dict[str, Any]:
    note = f"melo cannot show what this replaces: {why}."
    return {"added": 0, "removed": 0, "rows": [], "note": note}


def _why(problem: str) -> str:
    if problem.startswith("overwrites"):
        return "the file is too large"
    return _WHY.get(problem, "the file cannot be compared")


def _shown(text: str) -> list[str] | None:
    """The lines of *text* with secrets hidden, or None when that dropped a line.

    A private key block becomes one line, so the lines no longer match the
    text's and a row could not say which line of the text it came from.
    """
    lines = redact_secrets(text)[0].splitlines()
    return lines if len(lines) == len(text.splitlines()) else None


def _fewest_rows(old: str, new: str) -> int:
    """The fewest rows a diff of *old* and *new* can have: the lines that only one has."""
    before, after = Counter(old.splitlines()), Counter(new.splitlines())
    return before.total() + after.total() - 2 * (before & after).total()
