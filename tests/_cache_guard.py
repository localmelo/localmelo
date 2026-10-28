"""Guards that keep the test suite out of the real ``~/.cache/localmelo``.

The cache holds the user's config, memory, sessions, CLI history and the
gateway's call logs. The suite once wrote ~140 call-log run directories per
full run there (T52): two defaults were expanded at import time, before a
test's temporary HOME applied.

The guards act on the cause, inside the test process, so another process
writing the same cache (another lane's test run, a live gateway) cannot
trip them:

- a write of any kind (open, ``os.open``, ``os.mkdir``, rename, delete,
  ``sqlite3.connect`` …) whose target is under the real cache raises
  :class:`RealCacheError` before it happens;
- resolving a ``~`` path to the real cache raises too, so an import-time
  default is caught at the import that makes it.

``@pytest.mark.allow_real_cache`` opts a test out of the real-home rule;
writing the real cache is never allowed.
"""

from __future__ import annotations

import builtins
import io
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


class RealCacheError(RuntimeError):
    """The suite was about to use the user's real cache or home."""


def real_home() -> Path:
    """The home directory the process started with."""
    value = os.environ.get("HOME", "")
    return Path(value) if value else Path.home()


def real_cache_dir(home: Path | None = None) -> Path:
    """``~/.cache/localmelo`` for *home*, or for the real home by default."""
    return (home if home is not None else real_home()) / ".cache" / "localmelo"


def _text(path: Any) -> str | None:
    """*path* as a string, or ``None`` for a file descriptor or non-path."""
    if path is None or isinstance(path, int):
        return None
    try:
        raw = os.fspath(path)
    except TypeError:
        return None
    return os.fsdecode(raw) if isinstance(raw, bytes) else raw


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _writes(mode: Any) -> bool:
    return isinstance(mode, str) and any(char in mode for char in "wax+")


def _is_tilde(path: Any) -> bool:
    text = _text(path)
    return text is not None and text.startswith("~")


class RealCacheGuard:
    """Refuse reads of the real cache, writes to it, and the real home.

    ``watch_home`` adds the stricter rule used while a test runs: a ``~``
    path that resolves to the real home raises, because a test should run
    with HOME in a temporary directory. ``allow_real_home`` is the opt-out
    of that rule only; the cache itself is never allowed.
    """

    def __init__(self, root: Path, home: Path | None = None) -> None:
        self.root = Path(os.path.abspath(root))
        self.home = Path(os.path.abspath(home if home is not None else real_home()))
        self.allow_real_home = False
        self.watch_home = False
        self._original_expanduser: Callable[[Any], str] = os.path.expanduser
        self._original_open: Callable[..., Any] = io.open
        self._original_os_open: Callable[..., int] = os.open
        self._original_mkdir: Callable[..., None] = os.mkdir
        self._original_rename: Callable[..., None] = os.rename
        self._original_replace: Callable[..., None] = os.replace
        self._original_remove: Callable[..., None] = os.remove
        self._original_unlink: Callable[..., None] = os.unlink
        self._original_rmdir: Callable[..., None] = os.rmdir
        self._original_symlink: Callable[..., None] = os.symlink
        self._original_utime: Callable[..., None] = os.utime
        self._original_chmod: Callable[..., None] = os.chmod
        self._original_connect: Callable[..., sqlite3.Connection] = sqlite3.connect
        self._patched: list[tuple[Any, str, Any]] = []

    # -- the rules

    def expanduser(self, path: Any) -> str:
        """``os.path.expanduser`` that refuses the real cache and home."""
        result = self._original_expanduser(path)
        target = os.path.abspath(result)
        if _under(target, str(self.root)):
            raise RealCacheError(
                f"{path!r} resolves to {result}, under the real cache "
                f"{self.root}; tests must not use the user's cache "
                f"(HOME is {os.environ.get('HOME')!r}). Resolve the path "
                f"when it is used, not at import."
            )
        if (
            self.watch_home
            and not self.allow_real_home
            and _is_tilde(path)
            and _under(target, str(self.home))
        ):
            raise RealCacheError(
                f"{path!r} resolves to {result}, the real home {self.home}; "
                f"tests run with HOME in a temporary directory. Set HOME, "
                f"or mark the test allow_real_cache."
            )
        return result

    def check(self, path: Any, action: str) -> None:
        """Raise when *path* is under the real cache and would be written."""
        text = _text(path)
        if text is None:
            return
        target = os.path.abspath(text)
        if _under(target, str(self.root)):
            raise RealCacheError(
                f"refusing to {action} {target}: the real cache "
                f"{self.root} belongs to the user; tests write under their "
                f"own HOME ({os.environ.get('HOME')!r})."
            )

    # -- the guarded operations (also called by the tests of this module)

    def open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if _writes(mode):
            self.check(file, f"open with mode {mode!r}")
        return self._original_open(file, mode, *args, **kwargs)

    def os_open(self, path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & _WRITE_FLAGS:
            self.check(path, "open for writing")
        return self._original_os_open(path, flags, *args, **kwargs)

    def mkdir(self, path: Any, *args: Any, **kwargs: Any) -> None:
        self.check(path, "create a directory")
        self._original_mkdir(path, *args, **kwargs)

    def rename(self, src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        self.check(src, "move")
        self.check(dst, "move to")
        self._original_rename(src, dst, *args, **kwargs)

    def replace(self, src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        self.check(src, "move")
        self.check(dst, "move to")
        self._original_replace(src, dst, *args, **kwargs)

    def remove(self, path: Any, *args: Any, **kwargs: Any) -> None:
        self.check(path, "delete")
        self._original_remove(path, *args, **kwargs)

    def unlink(self, path: Any, *args: Any, **kwargs: Any) -> None:
        self.check(path, "delete")
        self._original_unlink(path, *args, **kwargs)

    def rmdir(self, path: Any, *args: Any, **kwargs: Any) -> None:
        self.check(path, "delete")
        self._original_rmdir(path, *args, **kwargs)

    def symlink(self, src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        self.check(dst, "create a symlink")
        self._original_symlink(src, dst, *args, **kwargs)

    def utime(self, path: Any, *args: Any, **kwargs: Any) -> None:
        self.check(path, "change the timestamps of")
        self._original_utime(path, *args, **kwargs)

    def chmod(self, path: Any, *args: Any, **kwargs: Any) -> None:
        self.check(path, "change the permissions of")
        self._original_chmod(path, *args, **kwargs)

    def connect(self, database: Any, *args: Any, **kwargs: Any) -> Any:
        # A database under the cache is a write, journal and lock files
        # included; ":memory:" and a mode=memory URI touch no file.
        text = _text(database)
        in_memory = text == ":memory:" or (
            text is not None and text.startswith("file:") and "mode=memory" in text
        )
        if not in_memory:
            self.check(database, "open a database")
        return self._original_connect(database, *args, **kwargs)

    # -- installation, for the pytest session hooks

    def install(self) -> None:
        """Patch the write and resolution functions of this process."""
        if self._patched:
            return
        patched = [
            (builtins, "open", self.open),
            (io, "open", self.open),
            (os, "open", self.os_open),
            (os, "mkdir", self.mkdir),
            (os, "rename", self.rename),
            (os, "replace", self.replace),
            (os, "remove", self.remove),
            (os, "unlink", self.unlink),
            (os, "rmdir", self.rmdir),
            (os, "symlink", self.symlink),
            (os, "utime", self.utime),
            (os, "chmod", self.chmod),
            (os.path, "expanduser", self.expanduser),
            (sqlite3, "connect", self.connect),
        ]
        for module, name, wrapper in patched:
            self._patched.append((module, name, getattr(module, name)))
            setattr(module, name, wrapper)

    def uninstall(self) -> None:
        """Restore every function :meth:`install` patched."""
        while self._patched:
            module, name, original = self._patched.pop()
            setattr(module, name, original)
