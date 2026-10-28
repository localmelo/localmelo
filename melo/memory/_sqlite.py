"""Shared aiosqlite helpers for the persistent memory backends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import sqlite3
from collections.abc import AsyncIterator, Mapping
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


# How long prepare_connection runs its steps again while SQLite refuses
# them with "database is locked".
LOCKED_RETRY_S: Final = 5.0


async def apply_pragmas(
    conn: aiosqlite.Connection, *, busy_timeout_ms: int | None = None
) -> None:
    """Apply the standard PRAGMA set on a freshly opened connection.

    *busy_timeout_ms* replaces the set's busy timeout of 5,000 ms.
    """
    for stmt in _PRAGMAS:
        if busy_timeout_ms is not None and stmt.startswith("PRAGMA busy_timeout"):
            stmt = f"PRAGMA busy_timeout={int(busy_timeout_ms)}"
        await conn.execute(stmt)


async def prepare_connection(
    conn: aiosqlite.Connection,
    schema: str,
    *,
    busy_timeout_ms: int | None = None,
    locked_retry_s: float | None = None,
) -> None:
    """Apply the PRAGMAs and create *schema* on a freshly opened connection.

    Connections that open one file at the same time can get ``database is
    locked`` at once, without waiting for the busy timeout: when one
    switches the file to WAL, or when a schema read cannot become a write
    because another connection wrote first. Both steps can run again, so
    they do, after a short wait, until *locked_retry_s* (by default
    :data:`LOCKED_RETRY_S`) has passed; then the last error is raised. Any
    other error is raised at once. *busy_timeout_ms* replaces the PRAGMAs'
    busy timeout (see :func:`apply_pragmas`).
    """
    loop = asyncio.get_running_loop()
    retry_s = LOCKED_RETRY_S if locked_retry_s is None else locked_retry_s
    deadline = loop.time() + retry_s
    delay = 0.01
    while True:
        try:
            await apply_pragmas(conn, busy_timeout_ms=busy_timeout_ms)
            await conn.executescript(schema)
            await conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc) or loop.time() >= deadline:
                raise
        with contextlib.suppress(sqlite3.Error):
            await conn.rollback()
        await asyncio.sleep(min(delay, max(0.0, deadline - loop.time())))
        delay = min(delay * 2, 0.2)


def make_private_dir(path: str) -> None:
    """Create the directory *path* with mode 0700 when it is missing.

    Missing parents get the default mode; only *path* itself is set to
    0700, with ``chmod``, so the umask cannot change it. An existing
    directory keeps its mode. Raises what ``os.makedirs(path,
    exist_ok=True)`` raises: ``FileExistsError`` for a file,
    ``NotADirectoryError`` below a file, ``PermissionError``. A ``chmod``
    that the file system refuses (FAT, some network mounts) is logged at
    WARNING, not raised: the directory is made, with the mode it got.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        if os.path.isdir(path):
            return
        raise
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        logger.warning("Could not restrict %s to mode 0700: %s", path, exc)


def create_private_file(path: str) -> None:
    """Create the file *path*, empty and with mode 0600, when it is missing.

    An existing file keeps its mode, also one that another opener creates
    first. SQLite gives a database's ``-wal`` and ``-shm`` files the
    database file's mode. A ``fchmod`` that the file system refuses is
    ignored: the file was created with 0600 less the umask, so with no
    group or other bit.
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    try:
        os.fchmod(fd, 0o600)
    except OSError as exc:
        logger.debug("Could not set %s to mode 0600: %s", path, exc)
    finally:
        os.close(fd)


@contextlib.asynccontextmanager
async def secure_deletes(conn: aiosqlite.Connection) -> AsyncIterator[None]:
    """Make SQLite overwrite what a ``DELETE`` frees, then restore the setting.

    ``PRAGMA secure_delete=ON`` makes SQLite zero the cells a delete frees,
    so their text does not stay in the database file's free pages. The
    connection's previous setting (``OFF``, ``ON`` or ``FAST``) is put back
    when the block ends, also when it raises. Set this before the
    transaction whose deletes it is to cover: the pragma is a connection
    setting, not a transaction one.
    """
    async with conn.execute("PRAGMA secure_delete") as cur:
        row = await cur.fetchone()
    previous = int(row[0]) if row else 0
    await conn.execute("PRAGMA secure_delete=ON")
    try:
        yield
    finally:
        mode = {0: "OFF", 1: "ON", 2: "FAST"}.get(previous, "OFF")
        with contextlib.suppress(Exception):
            await conn.execute(f"PRAGMA secure_delete={mode}")


async def checkpoint_wal(conn: aiosqlite.Connection) -> bool:
    """Fold a WAL database's ``-wal`` file into the database and truncate it.

    Returns ``True`` when the checkpoint ran and the ``-wal`` file is
    empty, ``False`` when SQLite reports it busy (another connection holds
    the file) or raises. A caller for which the checkpoint is a privacy
    step, such as a forget, is to log the ``False``: the committed rows are
    still gone, but their old copies can stay in the WAL until the next
    checkpoint truncates it.
    """
    try:
        async with conn.execute("PRAGMA wal_checkpoint(TRUNCATE)") as cur:
            row = await cur.fetchone()
    except sqlite3.Error:
        return False
    return not (row and int(row[0]))


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
