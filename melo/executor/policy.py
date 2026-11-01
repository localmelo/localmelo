from __future__ import annotations

import os

from localmelo.melo.contracts.folders import canonical_path


class WorkspacePolicy:
    """Restricts file operations to an allowed workspace root directory."""

    def __init__(self, allowed_root: str | None = None) -> None:
        self._root = canonical_path(allowed_root) if allowed_root else None

    @property
    def root(self) -> str | None:
        return self._root

    def resolve(self, path: object) -> str | None:
        """The canonical path of *path* under the root, or ``None`` outside it.

        A relative *path* joins the root, and an absolute one is taken as
        it is. ``..`` and symlinks are resolved: a link inside the root that
        points out is outside, and a link outside that points in is inside.
        ``~`` is not expanded, since a file name may hold it. The file system
        names the path (see
        :func:`~localmelo.melo.contracts.folders.canonical_path`), so another
        letter case of a folder on a case-insensitive file system, or a
        file-system alias such as macOS ``/.vol``, counts as the folder it
        names. The canonical path is compared with the root by its
        components, so a root of ``/`` holds every path. Without a root,
        *path* is resolved against the process directory and never refused.

        Raises ``ValueError`` unless *path* is a non-empty ``str`` without
        NUL characters, which ``os.path.realpath`` cannot take.
        """
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("path must be a non-empty str without NUL characters")
        if self._root is None:
            return os.path.realpath(path)
        real = canonical_path(os.path.join(self._root, path))
        return real if self._inside(real) else None

    def check_path(self, path: str) -> bool:
        """Return True if *path* is inside the allowed root (or no root set).

        A relative *path* is resolved against the process directory; see
        :meth:`resolve` for one resolved against the root.
        """
        if self._root is None:
            return True
        return self._inside(canonical_path(path))

    def _inside(self, real: str) -> bool:
        assert self._root is not None
        return os.path.commonpath([self._root, real]) == self._root
