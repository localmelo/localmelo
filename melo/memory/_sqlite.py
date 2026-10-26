"""Shared aiosqlite helpers for the persistent memory backends."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from typing import Final

import aiosqlite

logger = logging.getLogger(__name__)

# Days that history keeps the model's reasoning (``llm_calls.thinking``);
# older reasoning is cleared (see SqliteHistory). The variable below
# replaces the default.
THINKING_RETENTION_DAYS: Final = 30.0
THINKING_RETENTION_ENV: Final = "LOCALMELO_THINKING_RETENTION_DAYS"

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA foreign_keys=ON",
    "PRAGMA synchronous=NORMAL",
)


async def apply_pragmas(conn: aiosqlite.Connection) -> None:
    """Apply the standard PRAGMA set on a freshly opened connection."""
    for stmt in _PRAGMAS:
        await conn.execute(stmt)


def make_private_dir(path: str) -> None:
    """Create the directory *path* with mode 0700 when it is missing.

    Missing parents get the default mode; only *path* itself is set to
    0700, with ``chmod``, so the umask cannot change it. An existing
    directory keeps its mode. Raises what ``os.makedirs(path,
    exist_ok=True)`` raises: ``FileExistsError`` for a file,
    ``NotADirectoryError`` below a file, ``PermissionError``.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        if os.path.isdir(path):
            return
        raise
    os.chmod(path, 0o700)


def create_private_file(path: str) -> None:
    """Create the file *path*, empty and with mode 0600, when it is missing.

    An existing file keeps its mode, also one that another opener creates
    first. SQLite gives a database's ``-wal`` and ``-shm`` files the
    database file's mode.
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    try:
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def tighten_private_dir(path: str) -> None:
    """Set the directory *path* to mode 0700 when it is the user's and open.

    Only a directory that the current user owns and whose mode grants any
    group or other bit is changed, with one INFO line. Anything else is
    left as it is, silently. An ``OSError`` is logged at WARNING, not
    raised.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:  # no POSIX owners on this platform
        return
    try:
        status = os.stat(path)
        if status.st_uid != getuid() or not status.st_mode & 0o077:
            return
        os.chmod(path, 0o700)
    except OSError as exc:
        logger.warning("Could not restrict %s to mode 0700: %s", path, exc)
        return
    logger.info("Restricted %s to mode 0700", path)


def thinking_retention_days(environ: Mapping[str, str] | None = None) -> float:
    """The days of reasoning to keep, from ``LOCALMELO_THINKING_RETENTION_DAYS``.

    *environ* defaults to ``os.environ``. Unset or empty gives
    :data:`THINKING_RETENTION_DAYS`. Any other value must be a finite
    number above 0, else ``ValueError`` names the variable and the value.
    """
    env = os.environ if environ is None else environ
    value = env.get(THINKING_RETENTION_ENV, "")
    if not value:
        return THINKING_RETENTION_DAYS
    try:
        days = float(value)
    except ValueError:
        days = math.nan
    if not (math.isfinite(days) and days > 0):
        raise ValueError(
            f"{THINKING_RETENTION_ENV} must be a number of days greater than 0, "
            f"not '{value}'"
        )
    return days
