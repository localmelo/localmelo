"""Folders outside the workspace: what a person may be asked about (#405).

A file tool's path whose real path is outside the workspace is refused
unless a person allows it. This module holds the places melo never offers
(:func:`never_offered`), what a grant of a folder covers (:func:`covers`),
and the :class:`OutsideAccess` an approval request carries. The list of
places melo never opens lives here and nowhere else; the action review
reads its key and credential places from it too.

Paths are canonical (:func:`canonical_path`): a symlink counts as its
target, ``..`` is resolved, and the file system itself names what a path
reaches, so neither another letter case nor an alias that names a file by
number (macOS ``/.vol``, ``/.resolve``, ``/.nofollow``) gets past a check.
A file tool opens what was checked through :func:`open_checked`, which
makes sure the descriptor it reads or writes with is that file. The
never-offered check also compares without case and in one Unicode form; a
grant compares exactly, or by the file system's own answer that two
spellings name the same folder. This module imports only the standard
library.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
import unicodedata
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, Literal

try:
    import fcntl
except ImportError:  # pragma: no cover - not a POSIX system
    fcntl = None  # type: ignore[assignment]

Access = Literal["read", "write"]
# Why a place is never offered: keys or passwords, melo's own data, a
# system folder, or the person's whole home folder (or a folder above it).
NeverReason = Literal["credentials", "melo", "system", "home"]

# The file tools that write. Every other file tool reads.
WRITE_TOOLS: Final = frozenset({"file_write", "file_edit"})

# Key and credential places: a folder or a file, as a path under the home
# folder. Each also counts wherever it appears as whole path components (a
# backup's ``Users/x/.ssh`` too).
CREDENTIAL_PLACES: Final = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".config/gcloud",
    ".config/gh",
    ".config/op",
    ".kube",
    ".docker/config.json",
    ".netrc",
    ".pgpass",
    ".pypirc",
    ".npmrc",
    ".git-credentials",
    ".password-store",
    ".local/share/keyrings",
    "Library/Keychains",
    "Library/Cookies",
    "Library/Safari",
    "Library/Application Support/com.apple.TCC",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Application Support/BraveSoftware",
    "Library/Application Support/Microsoft Edge",
    "Library/Application Support/Arc",
    "Library/Application Support/Chromium",
    "Library/Application Support/Vivaldi",
    "Library/Application Support/com.operasoftware.Opera",
    "Library/Application Support/Slack",
    "Library/Containers/com.tinyspeck.slackmacgap",
    ".config/github-copilot",
    ".config/cloudflared",
    # Sign-ins of command-line coding agents.
    ".claude",
    ".claude.json",
    ".codex",
    ".gemini",
    ".cursor",
    ".terraform.d",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".vault-token",
    ".oci",
)
# melo's own data folder, under the home folder. ``LOCALMELO_MEMORY_DIR``
# names another one when it is set.
MELO_DATA: Final = ".cache/localmelo"
MELO_DATA_ENV: Final = "LOCALMELO_MEMORY_DIR"
# System folders. A place under one is never offered, except under
# :data:`SYSTEM_OPEN`.
SYSTEM_PLACES: Final = (
    "/System",
    "/Library",
    "/private/etc",
    "/etc",
    "/bin",
    "/sbin",
    "/usr",
)
SYSTEM_OPEN: Final = ("/usr/local",)
# Devices, the system's databases and a process's own files (its
# environment holds keys): never offered either. Reading a device can block
# for ever.
DEVICE_PLACES: Final = ("/dev", "/private/var/db", "/proc", "/sys")
# File-system aliases (macOS): ``/.vol/<device>/<inode>`` names a file by
# number, ``/.resolve/1/<path>`` and ``/.nofollow/<path>`` another way to a
# path. ``os.path.realpath`` leaves them as they are, so a path under one
# that the file system cannot name (:func:`canonical_path`) is never
# offered, granted or kept.
ALIAS_PLACES: Final = ("/.vol", "/.resolve", "/.nofollow")

# What the model reads for a place never offered; {path} is the path it gave.
NEVER_TEXT: Final[dict[str, str]] = {
    "credentials": (
        "Path '{path}' is outside the workspace, in a place that holds keys or "
        "passwords. melo never opens it, and the user is not asked."
    ),
    "melo": (
        "Path '{path}' is outside the workspace, in melo's own data folder. "
        "melo never opens it, and the user is not asked."
    ),
    "system": (
        "Path '{path}' is outside the workspace, in a system folder. melo never "
        "opens it, and the user is not asked."
    ),
    "home": (
        "Path '{path}' is the user's whole home folder or a folder above it. "
        "melo never opens it, and the user is not asked; name a folder inside it."
    ),
}
# The one line the model reads before a denial of an outside path.
ASKED_TEXT: Final = (
    "{access} {folder} outside the workspace needs the user's OK; they were asked."
)


@dataclass(frozen=True)
class OutsideAccess:
    """One file tool call whose real path is outside the workspace.

    ``path`` is the real path the call opens; ``folder`` the real folder a
    grant would cover (the path itself for ``list_files`` of a folder, else
    the folder that holds it); ``access`` whether the call reads or writes.
    ``grantable`` says whether "this chat" and "always" may be offered: never
    for a write, which asks every time, and never for a folder that is the
    home folder or above it.
    """

    path: str
    folder: str
    access: Access
    grantable: bool

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "folder": self.folder,
            "access": self.access,
            "grantable": self.grantable,
        }


def _key(path: str) -> str:
    """*path* compared without case and in one Unicode form."""
    return unicodedata.normalize("NFC", path).casefold()


def _under(path: str, folder: str) -> bool:
    """Whether *path* is *folder* or below it, on a path boundary (exact)."""
    folder = folder.rstrip("/")
    return path == folder or path.startswith(folder + "/") or folder == ""


def _home(home: str | None) -> str:
    return canonical_path(home or os.path.expanduser("~"))


def is_alias(path: str) -> bool:
    """Whether *path* lies under a file-system alias (:data:`ALIAS_PLACES`), in any case."""
    key = _key(path)
    return any(_under(key, place) for place in ALIAS_PLACES)


# macOS: fcntl F_GETPATH and the longest path it writes (MAXPATHLEN).
_F_GETPATH: Final = 50
_PATH_MAX: Final = 1024
# Opened for a lookup only: no read access on macOS (O_EVTONLY) or Linux
# (O_PATH), never a terminal, never waiting on a FIFO, never through a
# symlink that took the place of the last part.
_LOOKUP: Final = (
    (0x8000 if sys.platform == "darwin" else getattr(os, "O_PATH", 0))
    | os.O_NONBLOCK
    | getattr(os, "O_NOCTTY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def fd_path(fd: int) -> str | None:
    """The path the file system gives for the open descriptor *fd*, or ``None``.

    ``fcntl(F_GETPATH)`` on macOS, ``/proc/self/fd`` on Linux: the real
    path in the case the file system keeps, whatever path opened it (an
    alias, another letter case). ``None`` where the system cannot say.
    """
    if sys.platform == "darwin" and fcntl is not None:
        try:
            raw = fcntl.fcntl(fd, _F_GETPATH, bytes(_PATH_MAX))
        except (OSError, ValueError):
            return None
        path = os.fsdecode(raw.split(b"\0", 1)[0])
    elif os.path.isdir("/proc/self/fd"):
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            return None
        if path.endswith(" (deleted)"):
            return None
    else:
        return None
    return path if path.startswith("/") else None


def canonical_path(path: str) -> str:
    """*path* as the file system names it: real, alias-free, in its own case.

    ``os.path.realpath`` first (symlinks and ``..``). Then the deepest part
    of that path that exists as a folder or a regular file is opened for a
    lookup only (see :data:`_LOOKUP`: nothing is read, and a FIFO or a
    device is never opened) and named by the file system
    (:func:`fd_path`); the parts below it are joined back as given. Where
    nothing can be named, the real path stands; one still under an alias
    is never offered (:func:`never_offered`).
    """
    real = os.path.realpath(path)
    head = real
    rest: list[str] = []
    while True:
        named = _named(head)
        if named is not None:
            return os.path.join(named, *reversed(rest)) if rest else named
        parent = os.path.dirname(head)
        if parent == head:
            return real
        rest.append(os.path.basename(head))
        head = parent


def _named(path: str) -> str | None:
    """The file system's name for the folder or regular file *path*, else ``None``."""
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return None
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        return None
    try:
        fd = os.open(path, _LOOKUP)
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            return None  # swapped between the two looks
        return fd_path(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class CheckedPath:
    """The path the executor checked for one file tool call.

    ``path`` is canonical; ``outside`` says it is outside the workspace (a
    person allowed it), so a listing does not enter a place never offered.
    """

    path: str
    outside: bool = False


# The path a file tool's call was checked with, while the call runs (set
# by the executor, read by the file tools; see :func:`open_checked`).
CHECKED_PATH: ContextVar[CheckedPath | None] = ContextVar(
    "localmelo_checked_path", default=None
)


class PathChangedError(OSError):
    """The file a tool opened is not the one that was checked: it does not run."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Path '{path}' changed before it was opened; the call did not run."
        )
        self.path = path


class HardLinkError(OSError):
    """The file has another name (a hard link) than the one melo checked: it does not run.

    A write through it would change the file under its other name too,
    which may be anywhere on the same volume, and a read outside the
    workspace may be reading a file that is never offered.
    """

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Path '{path}': this file has another name outside what melo checked "
            "(a hard link); the call did not run."
        )
        self.path = path


def checked_for(target: str) -> CheckedPath | None:
    """The executor's check of *target*, when the running call checked that path."""
    checked = CHECKED_PATH.get()
    return checked if checked is not None and checked.path == target else None


def open_checked(path: str, flags: int, expect: str | None, mode: int = 0o666) -> int:
    """``os.open(path, flags)`` without following a symlink in its last part.

    With *expect* (the canonical path a check let through), the descriptor
    must name that file (:func:`fd_path`), or it is closed and
    :class:`PathChangedError` raised: whatever took a part of the path's place
    after the check, the tool reads or writes only the file that was
    checked, through this descriptor. Where the system cannot name a
    descriptor, the open stands as it is.
    """
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags, mode)
    except OSError as exc:
        # A symlink took the checked last part's place (ELOOP, or ENOTDIR
        # for a folder on macOS).
        swapped = exc.errno in (errno.ELOOP, errno.ENOTDIR) and os.path.islink(path)
        if expect is not None and swapped:
            raise PathChangedError(expect) from exc
        raise
    if expect is None:
        return fd
    named = fd_path(fd)
    if named is not None and named != expect:
        os.close(fd)
        raise PathChangedError(expect)
    return fd


def _has_place(path_key: str, place: str) -> bool:
    """Whether *place* (relative) is whole components anywhere in *path_key*."""
    return f"/{_key(place).strip('/')}/" in path_key + "/"


def melo_data_folders(home: str | None = None) -> tuple[str, ...]:
    """melo's own data folders: ``~/.cache/localmelo``, and the memory folder when set."""
    folders = [os.path.join(_home(home), MELO_DATA)]
    other = os.environ.get(MELO_DATA_ENV, "")
    if other:
        folders.append(canonical_path(os.path.expanduser(other)))
    return tuple(folders)


def never_offered(
    path: str, *, home: str | None = None, canonical: bool = False
) -> NeverReason | None:
    """Why *path* is never offered, or ``None`` when it may be.

    Key and credential places (anywhere, see :data:`CREDENTIAL_PLACES`),
    melo's own data, system folders and devices (``/usr/local`` excepted),
    ``/``, and the person's home folder itself or a folder above it. A
    folder below the home folder is fine. A path given under a file-system
    alias (:data:`ALIAS_PLACES`) counts as a system folder, whatever it
    names. Compared without case, on the canonical path
    (:func:`canonical_path`); with *canonical*, *path* is one already.
    """
    if is_alias(os.path.realpath(path)):
        return "system"
    real = path if canonical else canonical_path(path)
    if is_alias(real):
        return "system"
    key = _key(real)
    home_key = _key(_home(home))
    if any(_has_place(key, place) for place in CREDENTIAL_PLACES):
        return "credentials"
    if any(_under(key, _key(f)) for f in melo_data_folders(home)) or _has_place(
        key, MELO_DATA
    ):
        return "melo"
    if key == "/":
        return "system"
    system = any(_under(key, _key(p)) for p in (*SYSTEM_PLACES, *DEVICE_PLACES))
    if system and not any(_under(key, _key(p)) for p in SYSTEM_OPEN):
        return "system"
    if _under(home_key, key):
        return "home"
    return None


def above_home(folder: str, *, home: str | None = None) -> bool:
    """Whether *folder* is the home folder or a folder above it (``/`` too)."""
    return _under(_key(_home(home)), _key(canonical_path(folder)))


def home_label(path: str, *, home: str | None = None) -> str:
    """*path* as the person reads it: ``~/...`` below the home folder, else as it is."""
    root = _home(home).rstrip("/")
    if path == root:
        return "~"
    if path.startswith(root + "/"):
        return "~" + path[len(root) :]
    return path


def never_text(path: str, reason: NeverReason) -> str:
    """What the model reads for *path* (as it gave it), never offered for *reason*."""
    return NEVER_TEXT[reason].format(path=path)


def asked_text(outside: OutsideAccess) -> str:
    """The one line the model reads before a denial: the folder needed the user's OK."""
    access = "Writing to" if outside.access == "write" else "Reading"
    return ASKED_TEXT.format(access=access, folder=outside.folder)


def outside_access(
    tool_name: str, path: str, *, home: str | None = None
) -> OutsideAccess:
    """The :class:`OutsideAccess` of a call of *tool_name* on *path*.

    The path is made canonical (:func:`canonical_path`). ``list_files`` of
    a folder is offered that folder; any other call the folder that holds
    its path. A write is never grantable, nor is a folder that is the home
    folder or above it, nor one under a file-system alias.
    """
    real = canonical_path(path)
    if tool_name == "list_files" and os.path.isdir(real):
        folder = real
    else:
        folder = os.path.dirname(real) or "/"
    access: Access = "write" if tool_name in WRITE_TOOLS else "read"
    grantable = (
        access == "read" and not is_alias(folder) and not above_home(folder, home=home)
    )
    return OutsideAccess(real, folder, access, grantable)


def covers(folder: str, path: str) -> bool:
    """Whether a grant of the real folder *folder* covers the real path *path*.

    That is *folder* itself or a path below it, on a path boundary:
    ``/a/b`` covers ``/a/b/c`` but not ``/a/bc``. A different spelling of
    the same folder counts only when the file system says it is the same
    folder (a case-insensitive file system). ``/``, ``""`` and a folder
    under a file-system alias cover nothing.
    """
    folder = folder.rstrip("/")
    if not folder or not path.startswith("/") or is_alias(folder) or is_alias(path):
        return False
    if path == folder or path.startswith(folder + "/"):
        return True
    path_key, folder_key = _key(path), _key(folder)
    if not (path_key == folder_key or path_key.startswith(folder_key + "/")):
        return False
    depth = folder.count("/")
    prefix = "/".join(path.split("/")[: depth + 1])
    try:
        return os.path.samefile(prefix, folder)
    except OSError:
        return False


__all__ = [
    "ALIAS_PLACES",
    "ASKED_TEXT",
    "CHECKED_PATH",
    "CREDENTIAL_PLACES",
    "DEVICE_PLACES",
    "MELO_DATA",
    "NEVER_TEXT",
    "SYSTEM_OPEN",
    "SYSTEM_PLACES",
    "WRITE_TOOLS",
    "Access",
    "CheckedPath",
    "HardLinkError",
    "NeverReason",
    "OutsideAccess",
    "PathChangedError",
    "above_home",
    "asked_text",
    "canonical_path",
    "checked_for",
    "covers",
    "fd_path",
    "home_label",
    "is_alias",
    "melo_data_folders",
    "never_offered",
    "never_text",
    "open_checked",
    "outside_access",
]
