"""The shared GPU lock file, read only: the one parser of it.

The lock is a few ``key: value`` lines that another process writes. The
gateway's machine monitor, the idle gates and memory sleep all read it through
:func:`read_lock_file` and :func:`classify_lock`, so they never disagree about
whether it is held. localmelo never creates, writes, renames or removes a lock.

The module sits beside ``support/gateway`` and not in it, because importing
anything there needs the gateway extra, which ``support/memory`` does not, and not
under ``support/scheduler``, which reads no clock and does no I/O.
"""

from __future__ import annotations

import math
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# The lock is a few hundred bytes; a larger file is not a lock claim.
MAX_LOCK_BYTES = 64 * 1024

# How long past its ETA a lock is displayed as stale, and never enforced
# differently (section 3.5).
STALE_GRACE_S = 600.0

LockState = Literal["none", "self", "foreign"]


@dataclass(frozen=True)
class LockSnapshot:
    """One reading of the GPU lock: absent, present-and-readable, or unreadable.

    ``present`` is false only for a missing file. An empty or unreadable file
    is present and ``readable=False``, because a lock that exists is a claim.
    """

    present: bool = False
    readable: bool = True
    holder: str = ""
    purpose: str = ""
    start: str = ""
    eta: str = ""
    run: str = ""
    mem_gib: float = 0.0
    stale: bool = False
    error: str = ""
    alarms: tuple[str, ...] = ()
    # The string this snapshot was parsed for, so classify_lock() can be
    # called with the snapshot alone as well as with the string.
    self_holder: str = ""

    @classmethod
    def absent(cls) -> LockSnapshot:
        """The file does not exist."""
        return cls()

    @classmethod
    def unreadable(cls, error: str) -> LockSnapshot:
        """The file exists but cannot be read as a lock."""
        return cls(present=True, readable=False, error=error)


def _parse_instant(raw: str) -> datetime | None:
    """Parse the writers' two start forms, or ``None``.

    ISO 8601 with an offset first; then ``%Y-%m-%d %H:%M:%S %Z``; when the
    zone name is not understood (``EDT`` is not on every platform), the last
    token is stripped and the remaining naive local time is used, which is
    what the agent-benchmark writer printed.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        pass
    parts = text.rsplit(" ", 1)
    if len(parts) == 2:
        try:
            return datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def _timestamp(value: datetime) -> float | None:
    """The instant as a Unix timestamp; a naive value is local time."""
    try:
        return value.timestamp()
    except (OverflowError, ValueError, OSError):
        return None


_ETA_SUFFIX = re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*min\s*\)\s*$", re.IGNORECASE)
_ETA_MINUTES = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*min\s*$", re.IGNORECASE)


def _eta_instant(eta: str, start: datetime | None) -> datetime | None:
    """The ETA as an instant: absolute, absolute with ``(N min)``, or ``N min``."""
    text = eta.strip()
    if not text:
        return None
    minutes = _ETA_MINUTES.match(text)
    if minutes is not None:
        if start is None:
            return None
        from datetime import timedelta

        try:
            return start + timedelta(minutes=float(minutes.group(1)))
        except (OverflowError, ValueError, OSError):
            # A writer's malformed or absurd duration is display-only, never
            # a reason to fail the monitor (section 3.3).
            return None
    absolute = _ETA_SUFFIX.sub("", text)
    return _parse_instant(absolute)


def _strip_value(raw: str) -> str:
    """A lock line's value: trailing CR and spaces removed, not inner spaces."""
    return raw.rstrip("\r ")


def parse_lock(
    text: str,
    *,
    now_wall: float,
    self_holder: str = "",
    stale_grace_s: float = STALE_GRACE_S,
) -> LockSnapshot:
    """Parse the lock's ``key: value`` lines (section 3.3).

    ``self_holder`` is carried on the snapshot; comparing a holder with it
    is :func:`classify_lock`'s job.
    """
    if not text.strip():
        return LockSnapshot.unreadable("empty lock file")
    values: dict[str, str] = {}
    holder_count = 0
    for raw in text.splitlines():
        line = _strip_value(raw)
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key == "holder":
            holder_count += 1
        if key not in values:
            values[key] = value
    holder = values.get("holder", "")
    if not holder:
        return LockSnapshot.unreadable("lock file has no holder line")
    alarms: list[str] = []
    if holder_count > 1:
        alarms.append("lock_duplicate_holder")
    start = values.get("start", "")
    eta = values.get("eta", "")
    start_at = _parse_instant(start)
    eta_at = _eta_instant(eta, start_at)
    eta_ts = None if eta_at is None else _timestamp(eta_at)
    stale = eta_ts is not None and now_wall >= eta_ts + float(stale_grace_s)
    mem_gib = 0.0
    raw_mem = values.get("mem_gib", "").strip()
    if raw_mem:
        try:
            parsed = float(raw_mem)
        except ValueError:
            parsed = -1.0
        if math.isfinite(parsed) and parsed >= 0.0:
            mem_gib = parsed
    return LockSnapshot(
        present=True,
        readable=True,
        holder=holder,
        purpose=values.get("purpose", ""),
        start=start,
        eta=eta,
        run=values.get("run", ""),
        mem_gib=mem_gib,
        stale=stale,
        alarms=tuple(alarms),
        self_holder=self_holder,
    )


def classify_lock(snapshot: LockSnapshot, self_holder: str | None = None) -> LockState:
    """``none`` / ``self`` / ``foreign`` for *snapshot* (section 3.4).

    Exact and case-sensitive. ``lock_self`` is the whole holder, or
    ``holder:run`` matched against the file's ``holder`` and ``run``. An
    unreadable present file is foreign: fail closed.
    """
    if not snapshot.present:
        return "none"
    if not snapshot.readable:
        return "foreign"
    wanted = (snapshot.self_holder if self_holder is None else self_holder).strip()
    if not wanted:
        return "foreign"
    if snapshot.holder == wanted:
        return "self"
    holder, sep, run = wanted.rpartition(":")
    if sep and holder and snapshot.holder == holder and snapshot.run == run:
        return "self"
    return "foreign"


# The spec also names the one-argument classifier ``classify``.
classify = classify_lock


def read_lock_file(
    path: Path | str,
    *,
    now_wall: float,
    self_holder: str = "",
    stale_grace_s: float = STALE_GRACE_S,
) -> LockSnapshot:
    """Read and parse one lock file; never writes and never blocks long.

    A missing file is absent; a directory, a permission error, invalid UTF-8,
    an empty file, a missing ``holder:`` line and a file above 64 KiB are
    unreadable, which the gate treats as foreign.
    """
    target = Path(path).expanduser()
    try:
        info = target.stat(follow_symlinks=False)
    except FileNotFoundError:
        return LockSnapshot.absent()
    except OSError as exc:
        return LockSnapshot.unreadable(f"{type(exc).__name__}: {exc.strerror or exc}")
    if not stat.S_ISREG(info.st_mode):
        return LockSnapshot.unreadable("not a regular file")
    try:
        with target.open("rb") as handle:
            raw = handle.read(MAX_LOCK_BYTES + 1)
    except FileNotFoundError:
        return LockSnapshot.absent()
    except OSError as exc:
        return LockSnapshot.unreadable(f"{type(exc).__name__}: {exc.strerror or exc}")
    if len(raw) > MAX_LOCK_BYTES:
        return LockSnapshot.unreadable("lock file is larger than 64 KiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return LockSnapshot.unreadable(f"UnicodeDecodeError: {exc}")
    snapshot = parse_lock(
        text,
        now_wall=now_wall,
        self_holder=self_holder,
        stale_grace_s=stale_grace_s,
    )
    return snapshot
