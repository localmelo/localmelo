"""The files a task wrote, and the files its request asked it to write (#440).

- :func:`written_files`: the files a task's file tools wrote or edited, in
  every run of it, that are there now, named as the person reads them (a
  path in the task's folder, else the full path). The main chat tells them
  with the task's answer; nothing here guesses what a command wrote.
- :func:`asked_files`: the file names the person's words ask a task to write
  or save ("write the advice to career-advice.md", "把结果保存到 notes.csv"): a
  name with a known file extension right after a writing verb and its
  preposition. Only names written as files count; nothing is guessed.
- :func:`missing_files`: of those, the ones not in the task's folder: the
  path as written, or a bare name anywhere under the folder.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Sequence
from typing import Final

from localmelo.melo.contracts.folders import WRITE_TOOLS
from localmelo.melo.contracts.task_files import (
    MAX_FILES,
    asked_files,
    resolve,
    shown_path,
    written_paths,
)
from localmelo.support.sessions.store import SessionStore

# The tool events read to find them, newest first, at most.
MAX_EVENTS: Final = 2000
# Folders a walk for a bare name never enters, and how many entries it reads.
_SKIP: Final = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".cache", ".tox"}
)
MAX_WALK: Final = 5000

# What the chat reads of the files a task wrote: one line, each name at most
# NAME_CHARS, the whole at most LINE_CHARS.
NAME_CHARS: Final = 120
LINE_CHARS: Final = 400


def files_line(names: Sequence[str]) -> str:
    """*names* as one line the chat reads: control characters and line breaks
    gone, each name at most :data:`NAME_CHARS`, the whole at most
    :data:`LINE_CHARS`. A file's name is the task's text, never melo's words."""
    shown = []
    for name in names:
        flat = "".join(
            " " if unicodedata.category(ch)[0] in "CZ" and ch != " " else ch
            for ch in str(name)
        )
        flat = " ".join(flat.split())
        if len(flat) > NAME_CHARS:
            flat = flat[: NAME_CHARS - 1] + "…"
        if flat:
            shown.append(flat)
    line = ", ".join(shown)
    return line if len(line) <= LINE_CHARS else line[: LINE_CHARS - 1] + "…"


async def written_files(
    store: SessionStore, task_id: str, folder: str | None
) -> list[str]:
    """The files *task_id*'s file tools wrote or edited, that are there now (newest last).

    Each ``tool.started`` of a writing tool is paired with its
    ``tool.finished`` by run and call id; only a call that succeeded counts.
    A file named twice is named once, where it was last written (see
    :func:`~localmelo.melo.contracts.task_files.written_paths`, which a run's
    work checkpoint reads its own calls with).
    """
    started = await store.events_of_type(task_id, "tool.started", limit=MAX_EVENTS)
    finished = await store.events_of_type(task_id, "tool.finished", limit=MAX_EVENTS)
    status: dict[tuple[object, object], object] = {}
    for stored in finished:
        data = stored.event.get("data") or {}
        status[(stored.event.get("run_id"), data.get("call_id"))] = data.get("status")
    calls: list[tuple[object, object, object]] = []
    for stored in started:
        data = stored.event.get("data") or {}
        key = (stored.event.get("run_id"), data.get("call_id"))
        calls.append((data.get("name"), data.get("arguments"), status.get(key)))
    return written_paths(calls, folder)


def _found_under(folder: str, name: str) -> bool:
    """Whether a file called *name* is anywhere under *folder* (a bounded walk)."""
    seen = 0
    for _root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith(".")]
        if name in files:
            return True
        seen += len(dirs) + len(files)
        if seen > MAX_WALK:
            return False
    return False


def missing_files(folder: str | None, names: Sequence[str]) -> list[str]:
    """Of *names* (:func:`asked_files`), the ones not in *folder*.

    A path is looked for as written (relative to the folder, ``~`` expanded);
    a bare name is found anywhere under the folder too. Without a folder, a
    relative name cannot be found: it is missing.
    """
    missing: list[str] = []
    for name in names:
        resolved = resolve(folder, name)
        if resolved is not None and os.path.isfile(resolved):
            continue
        bare = "/" not in name and not name.startswith("~")
        if bare and folder and os.path.isdir(folder) and _found_under(folder, name):
            continue
        missing.append(name)
    return missing


__all__ = [
    "MAX_FILES",
    "WRITE_TOOLS",
    "asked_files",
    "files_line",
    "missing_files",
    "shown_path",
    "written_files",
]
