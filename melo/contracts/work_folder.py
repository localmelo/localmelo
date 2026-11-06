"""The work folder of a chat: a private temporary folder for the files its pages hand over (#551).

``$TMPDIR/localmelo/<chat id>/`` (``/tmp`` when ``TMPDIR`` is not set), readable
by this user only (mode 0700). It is made when a chat first needs it, removed
when the chat is deleted, and may be cleared by the system like any temporary
folder, so nothing in it is the only copy of anything. A file a page makes the
browser download lands here; melo reads files here without asking, and may attach
them to a page (what such a file says is a page's text, never the person's words).

Every folder on the way is checked before it is used: it must be a real folder
(never a link) of this user, and is made private if it was not. A chat id is a
short name of letters, digits, ``_`` and ``-``, so it cannot name another place.
This module imports only the standard library and
:mod:`localmelo.melo.contracts.folders`.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import shutil
import stat
import tempfile
from typing import Final

from localmelo.melo.contracts.folders import canonical_path, covers

# The folder under the temporary folder that holds every chat's work folder.
ROOT: Final = "localmelo"
# The kinds a downloaded file keeps its ending for: pictures and documents that
# do nothing by themselves. Any other ending is dropped; the page chose it.
KEPT_ENDINGS: Final = frozenset(
    {".pdf", ".txt", ".csv", ".json", ".md", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
    | {".docx", ".xlsx", ".pptx"}
)
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class WorkFolderError(OSError):
    """A folder on the way to the work folder is not this user's own private folder."""


def usable(chat_id: object) -> bool:
    """Whether *chat_id* is a plain name a work folder can be named after."""
    return isinstance(chat_id, str) and _ID.fullmatch(chat_id) is not None


def ephemeral_id() -> str:
    """A name for a work folder of a chat that has no id of its own (the terminal's), unique per run."""
    return "run-" + secrets.token_hex(6)


def _base() -> str:
    return os.path.join(tempfile.gettempdir(), ROOT)


def path_for(chat_id: str) -> str:
    """The real path of *chat_id*'s work folder, made or not; ``ValueError`` for an id that is no plain name."""
    if not _ID.fullmatch(chat_id):
        raise ValueError(f"{chat_id!r} is not a chat id")
    return canonical_path(os.path.join(_base(), chat_id))


def _private(folder: str) -> None:
    """Make *folder* if it is missing, and require it to be a real folder of this user, mode 0700."""
    old = os.umask(0o077)
    try:
        os.mkdir(folder, 0o700)
    except FileExistsError:
        pass
    finally:
        os.umask(old)
    info = os.lstat(folder)  # never through a link
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise WorkFolderError(f"{folder} is not a folder of this user's own")
    if info.st_mode & 0o077:
        os.chmod(folder, 0o700)


def ensure(chat_id: str) -> str:
    """The real path of *chat_id*'s work folder, made private if it was missing.

    ``ValueError`` for an id that is no plain name, :class:`WorkFolderError` when
    a folder on the way is a link, another user's or not a folder.
    """
    path_for(chat_id)  # the id is checked first
    base = _base()
    _private(base)
    _private(os.path.join(base, chat_id))
    return path_for(chat_id)


def take(chat_id: str, folder: str, saved_as: str, page_name: str) -> str | None:
    """Move the file *saved_as* out of *folder* (where the browser saved it) into *chat_id*'s work folder; its new path, or None.

    *saved_as* is the browser's own id for the file, never a name a page chose,
    and must be a regular file of this user (not a link). The new name is that
    id, plus the ending of *page_name* only when it is one of
    :data:`KEPT_ENDINGS`, so a page cannot pick a name that opens something.
    The file is readable by this user only. None when anything is not as it
    should be: nothing is moved, and a file left behind is removed.
    """
    source = os.path.join(folder, saved_as)
    try:
        if not usable(saved_as):
            return None
        info = os.lstat(source)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return None
        ending = os.path.splitext(page_name)[1].lower()
        target = os.path.join(ensure(chat_id), saved_as)
        if ending in KEPT_ENDINGS:
            target += ending
        shutil.move(source, target)
        os.chmod(target, 0o600)
        return target
    except (OSError, ValueError):
        with contextlib.suppress(OSError):
            os.unlink(source)
        return None


def remove(chat_id: str) -> bool:
    """Remove *chat_id*'s work folder and what is in it; whether there was one. A link, another user's folder or a bad id is left alone."""
    try:
        folder = os.path.join(_base(), chat_id)
        path_for(chat_id)
        info = os.lstat(folder)
    except (ValueError, OSError):
        return False
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def holds(chat_id: str, path: str) -> bool:
    """Whether the real path *path* is *chat_id*'s work folder or below it (links and ``..`` resolved first)."""
    try:
        return covers(path_for(chat_id), canonical_path(path))
    except ValueError:
        return False
