"""A project's own instruction file, read when a task starts in its folder.

Coding projects keep a file that tells assistants how the project is built,
tested and laid out: ``AGENTS.md``, ``CLAUDE.md`` or ``GEMINI.md``.
:func:`instructions` finds the ones that apply to a folder and returns their
text, bounded, for a task's first request.

The walk starts at the folder's real path and goes up to, and including, the
person's home folder, never above it; a folder outside home reads only itself.
At each level the first of the three names that exists counts. Nearest first:
the nearest file's text comes first, then its parents', at most
:data:`MAX_FILES` files and *limit* characters in all; a file that does not fit
whole is cut at a heading or a paragraph end and ends ``… (cut)``.

Only files in the project count, and they are only read:

* A name that is not a regular file is not a file. A symlink counts only when
  it points at a ``.md`` file inside the walked tree: below the folder the link
  sits in, or in a folder of the walk itself (``pkg/AGENTS.md`` ->
  ``../AGENTS.md``).
* A file over :data:`MAX_FILE_BYTES` is not read, and is named in
  :attr:`ProjectInstructions.skipped`.
* The home folder's own file (``~/CLAUDE.md``) and what lies under the folders
  where other assistants keep their personal files (``~/.claude``,
  ``~/.codex``, ``~/.gemini``, ...) belong to the person, not to a project:
  the importer brings what those say in as notes, so they are never read here.
* Secrets are removed (:func:`~localmelo.melo.checker.redaction.redact_secrets`)
  and every line :func:`~localmelo.support.memory.records.sensitive` flags is
  dropped.

Nothing here is written and nothing leaves the walked tree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Final, NamedTuple

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.support.memory import records
from localmelo.support.memory.importer import sources

# The names looked for at each level, in order: the first that exists counts.
NAMES: Final = ("AGENTS.md", "CLAUDE.md", "GEMINI.md")
MAX_FILES: Final = 3
# The text one task gets, labels included.
DEFAULT_LIMIT: Final = 6_000
MAX_FILE_BYTES: Final = sources.MAX_FILE_BYTES
CUT_MARK: Final = "… (cut)"
# A parent file gets no room smaller than this: a label and a crumb help nobody.
_MIN_ROOM: Final = 200
# Folders under home where other assistants keep their own, personal files.
_PERSONAL: Final = frozenset(
    {path.split("/")[0] for path in sources.CLAUDE_ROOTS}
    | {path.split("/")[0] for _, path in sources.FILES}
    | {".hermes"}
)
_HEADING = re.compile(r"^#{1,6}(?=\s)", re.MULTILINE)


class Used(NamedTuple):
    """One file's part in the text: *path* as seen from the task's folder."""

    path: str
    chars: int
    cut: bool


@dataclass(frozen=True)
class ProjectInstructions:
    """What the project's files say for a task, and what was read to say it.

    ``text`` is the files' text, nearest first, each under a ``--- path ---``
    line; ``files`` lists them for display; ``skipped`` names the files too big
    to read. ``text`` is empty when only a skipped file was found.
    """

    text: str
    files: tuple[Used, ...]
    skipped: tuple[str, ...] = ()


def instructions(
    folder: str, *, home: str | None = None, limit: int = DEFAULT_LIMIT
) -> ProjectInstructions | None:
    """The instruction files that apply to *folder*, or None when it has none.

    *home* is the person's home folder (``~`` when ``None``): the walk never
    goes above it. *limit* bounds the text in characters.
    """
    start = os.path.realpath(folder)
    top = os.path.realpath(home if home is not None else os.path.expanduser("~"))
    if not os.path.isdir(start):
        return None
    chain = _chain(start, top)
    parts: list[str] = []
    used: list[Used] = []
    skipped: list[str] = []
    seen: set[str] = set()
    remaining = limit
    for level in chain:
        if len(used) == MAX_FILES:
            break
        found = _file_at(level, chain, top)
        if found is None or found[1] in seen:
            continue
        name, real = found
        shown = os.path.relpath(os.path.join(level, name), start)
        seen.add(real)
        text = _read(real)
        if text is None:
            skipped.append(shown)
            continue
        if not text:
            continue
        label = f"--- {shown} ---\n"
        gap = 2 if parts else 0
        room = remaining - gap - len(label)
        cut = len(text) > room
        if cut:
            room -= len(CUT_MARK) + 1
            if parts and room < _MIN_ROOM:
                break
            text = _cut(text, room)
            if not text:
                break
        parts.append(label + text + (f"\n{CUT_MARK}" if cut else ""))
        used.append(Used(shown, len(text), cut))
        remaining -= gap + len(parts[-1])
        if cut:
            break
    if not parts and not skipped:
        return None
    return ProjectInstructions("\n\n".join(parts), tuple(used), tuple(skipped))


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _chain(start: str, top: str) -> list[str]:
    """The folders walked: *start* up to *top*, or *start* alone when outside it."""
    if not _inside(start, top):
        return [start]
    walked = [start]
    while walked[-1] != top:
        walked.append(os.path.dirname(walked[-1]))
    return walked


def _personal(real: str, home: str) -> bool:
    """Whether *real* is the person's own file: in home itself, or an assistant's folder."""
    if os.path.dirname(real) == home:
        return True
    if not _inside(real, home):
        return False
    return os.path.relpath(real, home).split(os.sep)[0] in _PERSONAL


def _file_at(level: str, chain: list[str], home: str) -> tuple[str, str] | None:
    """The file that counts at *level*: its name and real path, or None."""
    for name in NAMES:
        path = os.path.join(level, name)
        real = os.path.realpath(path)
        if not os.path.isfile(real):
            continue  # missing, a folder, or something that is no regular file
        if real != path and not (
            real.endswith(".md")
            and (_inside(real, level) or os.path.dirname(real) in chain)
        ):
            continue  # a link that leaves the walked tree, or points at no .md file
        if _personal(real, home):
            continue
        return name, real
    return None


def _read(real: str) -> str | None:
    """A file's text with secrets out, "" when nothing is left, None when too big."""
    try:
        if os.stat(real).st_size > MAX_FILE_BYTES:
            return None
        with open(real, "rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError:
        return ""
    if len(raw) > MAX_FILE_BYTES:
        return None
    text, _ = redact_secrets(
        raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    )
    return "\n".join(
        line for line in text.split("\n") if not records.sensitive(line)
    ).strip()


def _cut(text: str, room: int) -> str:
    """The start of *text* in at most *room* characters, ended at a block.

    A block ends before a blank line or before the next heading; a text with
    neither in reach ends at a line, then a word, then wherever it must.
    """
    if room <= 0:
        return ""
    stops = [
        m.start() for m in _HEADING.finditer(text[: room + 3]) if 0 < m.start() <= room
    ]
    blank = text.rfind("\n\n", 0, room + 2)
    if blank > 0:
        stops.append(blank)
    if stops:
        return text[: max(stops)].rstrip()
    for mark in ("\n", " "):
        at = text.rfind(mark, 1, room + 1)
        if at > 0:
            return text[:at].rstrip()
    return text[:room].rstrip()
