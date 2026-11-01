"""Folders outside the workspace: what a person may be asked about (#405).

A file tool's path whose real path is outside the workspace is refused
unless a person allows it. This module holds the places melo never offers
(:func:`never_offered`), what a grant of a folder covers (:func:`covers`),
and the :class:`OutsideAccess` an approval request carries. The list of
places melo never opens lives here and nowhere else; the action review
reads its key and credential places from it too.

Paths are real paths (``os.path.realpath``): a symlink counts as its
target and ``..`` is resolved before anything here looks at it. The
never-offered check compares without case and in one Unicode form, so a
case-insensitive file system cannot spell its way past it; a grant
compares exactly, or by the file system's own answer that two spellings
name the same folder. This module imports only the standard library.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

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
    return os.path.realpath(home or os.path.expanduser("~"))


def _has_place(path_key: str, place: str) -> bool:
    """Whether *place* (relative) is whole components anywhere in *path_key*."""
    return f"/{_key(place).strip('/')}/" in path_key + "/"


def melo_data_folders(home: str | None = None) -> tuple[str, ...]:
    """melo's own data folders: ``~/.cache/localmelo``, and the memory folder when set."""
    folders = [os.path.join(_home(home), MELO_DATA)]
    other = os.environ.get(MELO_DATA_ENV, "")
    if other:
        folders.append(os.path.realpath(os.path.expanduser(other)))
    return tuple(folders)


def never_offered(path: str, *, home: str | None = None) -> NeverReason | None:
    """Why the real path *path* is never offered, or ``None`` when it may be.

    Key and credential places (anywhere, see :data:`CREDENTIAL_PLACES`),
    melo's own data, system folders and devices (``/usr/local`` excepted),
    ``/``, and the person's home folder itself or a folder above it. A
    folder below the home folder is fine. Compared without case.
    """
    real = os.path.realpath(path)
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
    return _under(_key(_home(home)), _key(os.path.realpath(folder)))


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
    """The :class:`OutsideAccess` of a call of *tool_name* on the real path *path*.

    ``list_files`` of a folder is offered that folder; any other call the
    folder that holds its path. A write is never grantable, nor is a folder
    that is the home folder or above it.
    """
    real = os.path.realpath(path)
    if tool_name == "list_files" and os.path.isdir(real):
        folder = real
    else:
        folder = os.path.dirname(real) or "/"
    access: Access = "write" if tool_name in WRITE_TOOLS else "read"
    grantable = access == "read" and not above_home(folder, home=home)
    return OutsideAccess(real, folder, access, grantable)


def covers(folder: str, path: str) -> bool:
    """Whether a grant of the real folder *folder* covers the real path *path*.

    That is *folder* itself or a path below it, on a path boundary:
    ``/a/b`` covers ``/a/b/c`` but not ``/a/bc``. A different spelling of
    the same folder counts only when the file system says it is the same
    folder (a case-insensitive file system). ``/`` and ``""`` cover nothing.
    """
    folder = folder.rstrip("/")
    if not folder or not path.startswith("/"):
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
    "ASKED_TEXT",
    "CREDENTIAL_PLACES",
    "DEVICE_PLACES",
    "MELO_DATA",
    "NEVER_TEXT",
    "SYSTEM_OPEN",
    "SYSTEM_PLACES",
    "WRITE_TOOLS",
    "Access",
    "NeverReason",
    "OutsideAccess",
    "above_home",
    "asked_text",
    "covers",
    "home_label",
    "melo_data_folders",
    "never_offered",
    "never_text",
    "outside_access",
]
