"""The files a task's file tools wrote, as the core and the gateway both name them.

One derivation for both readers: the work checkpoint lists the files a run wrote
from the calls it made, and the gateway (``support.gateway.task_files``) lists the
files a task wrote from its stored ``tool.*`` events. Each pairs a call with its
outcome its own way and hands the pairs to :func:`written_paths`. Nothing here
guesses what a command wrote. This module imports only the standard library and
:mod:`localmelo.melo.contracts.folders`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Final

from localmelo.melo.contracts.folders import WRITE_TOOLS

# The files a list names at most.
MAX_FILES: Final = 20


def _inside(folder: str, path: str) -> bool:
    try:
        return os.path.commonpath([folder, path]) == folder
    except ValueError:
        return False


def resolve(folder: str | None, path: str) -> str | None:
    """*path* as the file tools read it: ``~`` expanded, relative to *folder*."""
    expanded = os.path.expanduser(path.strip())
    if not expanded:
        return None
    if not os.path.isabs(expanded):
        if not folder:
            return None
        expanded = os.path.join(folder, expanded)
    return os.path.realpath(expanded)


def shown_path(folder: str | None, path: str) -> str:
    """*path* as the person reads it: relative to the task's folder when it is in it."""
    if folder:
        root = os.path.realpath(folder)
        if _inside(root, path):
            return os.path.relpath(path, root)
    return path


def written_paths(
    calls: Iterable[tuple[object, object, object]], folder: str | None
) -> list[str]:
    """The files that *calls* wrote or edited and that are there now (newest last).

    *calls* are ``(tool name, arguments, status)`` in the order they ran; only a
    file tool that writes (:data:`~localmelo.melo.contracts.folders.WRITE_TOOLS`)
    with the status ``success`` counts, at its ``path`` argument. A file named
    twice is named once, where it was last written; at most :data:`MAX_FILES`,
    each as :func:`shown_path` names it.
    """
    paths: list[str] = []
    for name, arguments, status in calls:
        if name not in WRITE_TOOLS or status != "success":
            continue
        given = arguments.get("path") if isinstance(arguments, dict) else None
        resolved = resolve(folder, given) if isinstance(given, str) else None
        if resolved is None or not os.path.isfile(resolved):
            continue
        if resolved in paths:
            paths.remove(resolved)
        paths.append(resolved)
    return [shown_path(folder, path) for path in paths[-MAX_FILES:]]


__all__ = ["MAX_FILES", "resolve", "shown_path", "written_paths"]
