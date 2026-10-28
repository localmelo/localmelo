"""The gateway's read-only machine readers: the shared GPU lock and the battery.

``LockWatcher`` follows ``[scheduler] gpu_lock_file``; ``BatteryReader`` polls
``pmset -g batt``. Both are observation only: localmelo never creates, writes,
renames or removes a lock, and never changes a power setting. ``MachineMonitor``
combines the two snapshots, carries S1's battery band with its hysteresis, and
hands one :class:`MachineSnapshot` to the gate whenever the machine changes.

The module lives in ``support/gateway`` and not under ``support/scheduler``
because S1's determinism tests forbid a clock and a non-stdlib import there
(decision D4). It imports no ``fastapi`` and no ``playground``; the clock,
sleep, ``pmset`` runner, reader and ``kqueue`` factory are injectable so the
tests never wait on a real interval or touch the real lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import select
import stat
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from localmelo.support.scheduler import Machine
from localmelo.support.scheduler.machine_gates import battery_band

logger = logging.getLogger("localmelo.gateway")

# The lock is a few hundred bytes; a larger file is not a lock claim.
MAX_LOCK_BYTES = 64 * 1024

# How long past its ETA a lock is displayed as stale, and never enforced
# differently (section 3.5).
STALE_GRACE_S = 600.0

# The default intervals (section 4.2 and 5.4).
LOCK_POLL_S = 1.0
LOCK_GUARD_POLL_S = 2.0
LOCK_SETTLE_S = 0.2
BATTERY_POLL_S = 30.0

# The machine log's own schema, never the call trace's (decision D8).
MACHINE_SCHEMA = "localmelo.sched.machine"
MACHINE_VERSION = 1

LockState = Literal["none", "self", "foreign"]


# ── Lock parsing ──


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


# ── The battery ──


@dataclass(frozen=True)
class BatterySnapshot:
    """One reading of ``pmset -g batt``.

    ``on_ac`` is ``True`` for AC, ``False`` for battery, ``None`` when the
    output names neither. ``known`` is false until a reading succeeded.
    """

    on_ac: bool | None = True
    pct: float | None = None
    state: str = ""
    raw: str = ""
    known: bool = False


_BATTERY_PCT = re.compile(r"(\d+(?:\.\d+)?)%")
_BATTERY_STATE = re.compile(r";\s*([^;]+?)\s*;")


def parse_battery(text: str) -> BatterySnapshot:
    """Parse the output of ``pmset -g batt`` (section 5.2)."""
    lines = text.splitlines()
    raw = lines[0].strip() if lines else ""
    if "drawing from 'AC Power'" in text:
        on_ac: bool | None = True
    elif "drawing from 'Battery Power'" in text:
        on_ac = False
    else:
        on_ac = None
    pct: float | None = None
    state = ""
    for line in lines:
        if "InternalBattery" not in line and "Battery" not in line:
            continue
        match = _BATTERY_PCT.search(line)
        if match is None:
            continue
        value = float(match.group(1))
        if 0.0 <= value <= 100.0:
            pct = value
        after = line[match.end() :]
        found = _BATTERY_STATE.search(after)
        if found is not None:
            state = found.group(1).strip()
        break
    return BatterySnapshot(
        on_ac=on_ac,
        pct=pct,
        state=state,
        raw=raw,
        known=bool(text.strip()),
    )


def _battery_usable(snapshot: BatterySnapshot) -> bool:
    """A reading that names a power source or a percentage; else unparseable."""
    return snapshot.known and (snapshot.on_ac is not None or snapshot.pct is not None)


def read_pmset() -> str:
    """Run the read-only ``pmset -g batt``; raise when it cannot be read."""
    result = subprocess.run(
        ["pmset", "-g", "batt"],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(f"pmset exited with {result.returncode}")
    if not result.stdout.strip():
        raise OSError("pmset printed nothing")
    return result.stdout


class BatteryReader:
    """Poll ``pmset -g batt`` every *poll_s*, keeping the last good reading.

    A failure logs ``battery_unknown`` once and never raises; before the
    first good reading the state is AC with no percentage (band 4), so a
    battery reading can never block a chat.
    """

    def __init__(
        self,
        runner: Callable[[], str],
        on_change: Callable[[BatterySnapshot], None],
        *,
        poll_s: float = BATTERY_POLL_S,
        initial: BatterySnapshot | None = None,
        initial_failed: bool = False,
    ) -> None:
        self._runner = runner
        self._on_change = on_change
        self._poll_s = max(0.001, poll_s)
        self._good = initial
        self._last = initial if initial is not None else BatterySnapshot()
        # The monitor already logged the initial failure (F5): one warning
        # for one outage.
        self._warned = initial_failed
        self._task: asyncio.Task[None] | None = None

    @property
    def snapshot(self) -> BatterySnapshot:
        """The last good reading, or the AC/unknown default."""
        return self._good if self._good is not None else BatterySnapshot()

    async def poll_once(self) -> BatterySnapshot:
        """Read once, keep the last good state on a failure, return the state."""
        try:
            text = await asyncio.to_thread(self._runner)
            parsed = parse_battery(text)
            if not _battery_usable(parsed):
                raise ValueError("pmset output was not parseable")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._warned:
                logger.warning("battery unknown: %s", exc)
                self._warned = True
            return self.snapshot
        self._good = parsed
        # ``battery_unknown`` is one warning per process, not one per outage.
        if parsed != self._last:
            self._last = parsed
            self._on_change(parsed)
        return parsed

    async def start(self) -> None:
        """Start the poll task; a second start is a no-op."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # poll_once already contains its errors
                logger.warning("battery monitor task failed", exc_info=True)
            await asyncio.sleep(self._poll_s)

    async def stop(self) -> None:
        """Cancel the poll task; a second stop is a no-op."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ── The lock watcher ──


LockRead = Callable[[Path], LockSnapshot]


class LockWatcher:
    """Follow one lock path with kqueue and a safety poll, or by polling only.

    The callback runs only when the parsed snapshot changes, so a re-read of
    the same bytes is a no-op and an atomic replacement is one transition, not
    absent-then-present. The watcher never writes to the lock.
    """

    def __init__(
        self,
        path: Path | str,
        on_change: Callable[[LockSnapshot], None],
        *,
        read: LockRead | None = None,
        poll_s: float = LOCK_POLL_S,
        settle_s: float = LOCK_SETTLE_S,
        guard_poll_s: float = LOCK_GUARD_POLL_S,
        use_kqueue: bool = True,
        kqueue_factory: Callable[[], Any] | None = None,
        initial: LockSnapshot | None = None,
    ) -> None:
        self._path = Path(path).expanduser()
        self._on_change = on_change
        self._read = read
        self._poll_s = max(0.001, poll_s)
        self._settle_s = max(0.0, settle_s)
        self._guard_poll_s = max(0.001, guard_poll_s)
        self._kqueue_factory = kqueue_factory
        self._use_kqueue = use_kqueue and self._kqueue_supported()
        self._last = initial
        self._task: asyncio.Task[None] | None = None
        self._kq: Any | None = None
        self._dir_fd: int | None = None
        self._file_fd: int | None = None
        # A wake that completed without error resets the failure count (F4).
        self._healthy = False

    @property
    def snapshot(self) -> LockSnapshot | None:
        """The last snapshot the watcher emitted."""
        return self._last

    @staticmethod
    def _kqueue_supported() -> bool:
        return hasattr(select, "kqueue")

    def _read_once(self) -> LockSnapshot:
        if self._read is not None:
            return self._read(self._path)
        return read_lock_file(self._path, now_wall=time.time())

    def _check(self) -> None:
        try:
            snapshot = self._read_once()
        except Exception as exc:  # a broken reader is shown, never fatal
            snapshot = LockSnapshot.unreadable(f"{type(exc).__name__}: {exc}")
        if snapshot != self._last:
            self._last = snapshot
            self._on_change(snapshot)

    async def start(self) -> None:
        """Start the watch task; a second start is a no-op.

        The first read and, on macOS, the kqueue fds are ready before this
        returns, so a file created right after ``start()`` cannot fall into
        the unarmed window before the first wake.
        """
        if self._task is not None:
            return
        if self._last is None:
            self._check()
        if self._use_kqueue:
            try:
                self._prepare_kqueue()
            except Exception as exc:
                logger.warning(
                    "lock kqueue setup failed (%s); falling back to polling",
                    type(exc).__name__,
                )
                self._close_kqueue()
                self._use_kqueue = False
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        failures = 0
        while True:
            try:
                if self._use_kqueue:
                    try:
                        await self._kqueue_loop()
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        failures = 1 if self._healthy else failures + 1
                        self._healthy = False
                        logger.warning(
                            "lock monitor task failed (%s); falling back to polling",
                            type(exc).__name__,
                        )
                        self._close_kqueue()
                        if failures > 5:
                            self._use_kqueue = False
                        else:
                            await asyncio.sleep(1.0)
                            continue
                await self._poll_loop()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures = 1 if self._healthy else failures + 1
                self._healthy = False
                logger.warning("lock monitor task failed (%s)", type(exc).__name__)
                if failures > 5:
                    self._use_kqueue = False
                await asyncio.sleep(1.0)

    async def _poll_loop(self) -> None:
        while True:
            self._check()
            await asyncio.sleep(self._poll_s)

    def _open_watch_fds(self) -> None:
        """Open the directory and, when it exists, the lock file for vnode notes."""
        self._dir_fd = os.open(self._path.parent, os.O_RDONLY)
        self._arm(self._dir_fd, is_dir=True)
        self._open_file_fd()

    def _open_file_fd(self) -> None:
        self._close_file_fd()
        try:
            fd = os.open(self._path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return
        self._file_fd = fd
        self._arm(fd, is_dir=False)

    def _arm(self, fd: int, *, is_dir: bool) -> None:
        assert self._kq is not None
        if is_dir:
            flags = (
                select.KQ_NOTE_WRITE
                | select.KQ_NOTE_DELETE
                | select.KQ_NOTE_RENAME
                | select.KQ_NOTE_EXTEND
            )
        else:
            flags = (
                select.KQ_NOTE_WRITE
                | select.KQ_NOTE_EXTEND
                | select.KQ_NOTE_DELETE
                | select.KQ_NOTE_RENAME
                | select.KQ_NOTE_ATTRIB
                | getattr(select, "KQ_NOTE_REVOKE", 0)
            )
        event = select.kevent(
            fd,
            filter=select.KQ_FILTER_VNODE,
            flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_CLEAR,
            fflags=flags,
        )
        self._kq.control([event], 0, 0)

    def _prepare_kqueue(self) -> None:
        factory = self._kqueue_factory
        self._kq = factory() if factory is not None else select.kqueue()
        self._open_watch_fds()

    async def _kqueue_loop(self) -> None:
        if self._kq is None:
            self._prepare_kqueue()
        kq = self._kq
        assert kq is not None
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()

        def ready() -> None:
            wake.set()

        # The fd is read on the loop, never in a worker: stop() may close it
        # as soon as this task is cancelled (F1). The guard sleep is the
        # safety poll when the fs gives no vnode notes.
        loop.add_reader(kq.fileno(), ready)
        try:
            while True:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wake.wait(), timeout=self._guard_poll_s)
                wake.clear()
                # Drain every pending event without blocking.
                while kq.control([], 1, 0):
                    pass
                if self._settle_s:
                    await asyncio.sleep(self._settle_s)
                # Re-read and re-arm after every wake, a timeout included,
                # so a replaced inode cannot leave the watcher on a dead fd.
                self._check()
                if self._dir_fd is None:
                    raise RuntimeError("lock directory fd closed")
                self._open_file_fd()
                self._healthy = True
        finally:
            with contextlib.suppress(Exception):
                loop.remove_reader(kq.fileno())

    def _close_file_fd(self) -> None:
        if self._file_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._file_fd)
            self._file_fd = None

    def _close_kqueue(self) -> None:
        self._close_file_fd()
        if self._dir_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._dir_fd)
            self._dir_fd = None
        if self._kq is not None:
            with contextlib.suppress(OSError):
                self._kq.close()
            self._kq = None

    async def stop(self) -> None:
        """Cancel the watch task and close every fd; a second stop is a no-op."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._close_kqueue()


# ── The combined monitor ──


@dataclass(frozen=True)
class MachineSnapshot:
    """One machine reading as the page, the gate and ``machine.jsonl`` see it."""

    lock: LockState = "none"
    holder: str = ""
    purpose: str = ""
    run: str = ""
    eta: str = ""
    stale: bool = False
    readable: bool = True
    lock_error: str = ""
    mem_gib: float = 0.0
    on_ac: bool = True
    battery_pct: float | None = None
    battery_state: str = ""
    band: int = 4
    gate: str = "record"

    @property
    def machine(self) -> Machine:
        """The S1 snapshot the policy may see in ``wait`` mode."""
        return Machine(
            lock=self.lock,
            lock_mem_gib=self.mem_gib,
            on_ac=self.on_ac,
            battery_pct=self.battery_pct,
        )

    def payload(self) -> dict[str, Any]:
        """The public page/API payload (R15)."""
        return {
            "lock": self.lock,
            "holder": self.holder,
            "purpose": self.purpose,
            "run": self.run,
            "eta": self.eta,
            "stale": self.stale,
            "readable": self.readable,
            "on_ac": self.on_ac,
            "battery_pct": self.battery_pct,
            "battery_state": self.battery_state,
            "band": self.band,
            "gate": self.gate,
        }

    def log_values(self) -> dict[str, Any]:
        """The fixed keys of one ``machine.jsonl`` line (section 6.4)."""
        return {
            "lock": self.lock,
            "holder": self.holder,
            "purpose": self.purpose,
            "run": self.run,
            "eta": self.eta,
            "stale": self.stale,
            "readable": self.readable,
            "on_ac": self.on_ac,
            "battery_pct": self.battery_pct,
            "battery_state": self.battery_state,
            "band": self.band,
        }


MachineWrite = Callable[[dict[str, Any]], None]
MachineCallback = Callable[[MachineSnapshot], None]


class MachineMonitor:
    """One lock watcher, one battery reader and one canonical snapshot.

    The callback runs on the monitor's loop when the snapshot changes;
    ``write`` appends a ``machine.jsonl`` line. Neither may raise into the
    gateway: a write failure is the writer's warning, and the monitor keeps
    running.
    """

    def __init__(
        self,
        *,
        lock_path: Path | str | None = None,
        lock_self: str = "",
        lock_gate: str = "record",
        write: MachineWrite | None = None,
        on_change: MachineCallback | None = None,
        lock_read: LockRead | None = None,
        battery_runner: Callable[[], str] | None = None,
        lock_poll_s: float = LOCK_POLL_S,
        battery_poll_s: float = BATTERY_POLL_S,
        settle_s: float = LOCK_SETTLE_S,
        guard_poll_s: float = LOCK_GUARD_POLL_S,
        stale_grace_s: float = STALE_GRACE_S,
        use_kqueue: bool = True,
        kqueue_factory: Callable[[], Any] | None = None,
        now_wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock_path = Path(lock_path).expanduser() if lock_path else None
        self._lock_self = lock_self
        self._lock_gate = lock_gate
        self._write = write
        self._on_change = on_change
        self._lock_read = lock_read
        self._battery_runner = battery_runner
        self._lock_poll_s = lock_poll_s
        self._battery_poll_s = battery_poll_s
        self._settle_s = settle_s
        self._guard_poll_s = guard_poll_s
        self._stale_grace_s = stale_grace_s
        self._use_kqueue = use_kqueue
        self._kqueue_factory = kqueue_factory
        self._now_wall = now_wall
        self._mono = mono
        self._started_mono: float | None = None
        self._lock = LockSnapshot.absent()
        self._battery = BatterySnapshot()
        self._band = 4
        self._snapshot: MachineSnapshot | None = None
        self._warned: set[str] = set()
        self._watcher: LockWatcher | None = None
        self._reader: BatteryReader | None = None
        self._started = False
        self._battery_failed = False

    @property
    def snapshot(self) -> MachineSnapshot | None:
        """The last emitted snapshot, or ``None`` before a start."""
        return self._snapshot

    def payload(self) -> dict[str, Any] | None:
        """The page/API payload, or ``None`` before a start."""
        return None if self._snapshot is None else self._snapshot.payload()

    async def start(self) -> None:
        """Read once, log the start and start both readers; idempotent."""
        if self._started:
            return
        self._started = True
        self._started_mono = self._mono()
        if self._lock_path is not None:
            self._lock = self._read_lock()
        if self._battery_runner is not None:
            self._battery = await self._read_battery()
        self._emit("start", force=True)
        if self._lock_path is not None:
            self._watcher = LockWatcher(
                self._lock_path,
                self._on_lock,
                read=lambda path: self._read_lock(path),
                poll_s=self._lock_poll_s,
                settle_s=self._settle_s,
                guard_poll_s=self._guard_poll_s,
                use_kqueue=self._use_kqueue,
                kqueue_factory=self._kqueue_factory,
                initial=self._lock,
            )
            await self._watcher.start()
        if self._battery_runner is not None:
            self._reader = BatteryReader(
                self._battery_runner,
                self._on_battery,
                poll_s=self._battery_poll_s,
                initial=self._battery,
                initial_failed=self._battery_failed,
            )
            await self._reader.start()

    async def stop(self) -> None:
        """Stop both readers and log the clean stop; a second stop is a no-op."""
        if not self._started:
            return
        if self._watcher is not None:
            await self._watcher.stop()
            self._watcher = None
        if self._reader is not None:
            await self._reader.stop()
            self._reader = None
        self._emit("stop", force=True)
        self._started = False

    def _read_lock(self, path: Path | None = None) -> LockSnapshot:
        target = path if path is not None else self._lock_path
        if target is None:
            return LockSnapshot.absent()
        if self._lock_read is not None:
            try:
                return self._lock_read(target)
            except Exception as exc:  # the seam is a caller's function
                return LockSnapshot.unreadable(f"{type(exc).__name__}: {exc}")
        return read_lock_file(
            target,
            now_wall=self._now_wall(),
            self_holder=self._lock_self,
            stale_grace_s=self._stale_grace_s,
        )

    async def _read_battery(self) -> BatterySnapshot:
        runner = self._battery_runner
        if runner is None:
            return BatterySnapshot()
        try:
            text = await asyncio.to_thread(runner)
            parsed = parse_battery(text)
            if not _battery_usable(parsed):
                raise ValueError("pmset output was not parseable")
            return parsed
        except asyncio.CancelledError:
            raise
        except Exception:
            self._alarm("battery_unknown")
            self._battery_failed = True
            return BatterySnapshot()

    def _on_lock(self, snapshot: LockSnapshot) -> None:
        if snapshot == self._lock:
            return
        self._lock = snapshot
        self._emit("lock")

    def _on_battery(self, snapshot: BatterySnapshot) -> None:
        if snapshot == self._battery:
            return
        self._battery = snapshot
        self._emit("battery")

    def _band_for(self, battery: BatterySnapshot) -> int:
        on_ac = battery.on_ac is not False
        machine = Machine(on_ac=on_ac, battery_pct=battery.pct)
        return battery_band(machine, self._band)

    def _build(self) -> MachineSnapshot:
        self._band = self._band_for(self._battery)
        lock_state = classify_lock(self._lock, self._lock_self)
        on_ac = self._battery.on_ac is not False
        return MachineSnapshot(
            lock=lock_state,
            holder=self._lock.holder,
            purpose=self._lock.purpose,
            run=self._lock.run,
            eta=self._lock.eta,
            stale=self._lock.stale,
            readable=self._lock.readable,
            lock_error=self._lock.error,
            mem_gib=self._lock.mem_gib,
            on_ac=on_ac,
            battery_pct=self._battery.pct,
            battery_state=self._battery.state,
            band=self._band,
            gate=self._lock_gate,
        )

    def _emit(self, event: str, *, force: bool = False) -> None:
        snapshot = self._build()
        if not force and snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        self._raise_alarms()
        if self._write is not None:
            values = {
                **snapshot.log_values(),
                "event": event,
                "harness": "localmelo",
                "at_s": self._elapsed(),
                "wall_s": self._now_wall(),
            }
            try:
                self._write(values)
            except Exception as exc:  # the log may never fail the gateway
                logger.warning("machine log write failed: %s", exc)
        self._log_line(snapshot)
        if self._on_change is not None:
            try:
                self._on_change(snapshot)
            except Exception:  # a callback must never stop the monitor
                logger.warning("machine change callback failed", exc_info=True)

    def _raise_alarms(self) -> None:
        if not self._lock.readable:
            key = f"lock_unreadable:{self._lock.error}"
            if key not in self._warned:
                self._warned.add(key)
                logger.warning("scheduler alarm: %s", key)
        for alarm in self._lock.alarms:
            if alarm not in self._warned:
                self._warned.add(alarm)
                logger.warning("scheduler alarm: %s", alarm)
        if self._lock.stale and self._lock.holder:
            key = f"lock_stale:{self._lock.holder}"
            if key not in self._warned:
                self._warned.add(key)
                logger.warning("scheduler alarm: %s", key)

    def _alarm(self, alarm: str) -> None:
        if alarm in self._warned:
            return
        self._warned.add(alarm)
        logger.warning("scheduler alarm: %s", alarm)

    def _elapsed(self) -> float:
        if self._started_mono is None:
            return 0.0
        return max(0.0, self._mono() - self._started_mono)

    def _log_line(self, snapshot: MachineSnapshot) -> None:
        holder = f' holder="{snapshot.holder}"' if snapshot.holder else ""
        purpose = f' purpose="{snapshot.purpose}"' if snapshot.purpose else ""
        if snapshot.battery_pct is None:
            battery = "ac"
        else:
            battery = f"{snapshot.battery_pct:g}% {snapshot.battery_state}".rstrip()
        logger.info(
            "scheduler machine: lock=%s%s%s battery=%s band=%d gate=%s",
            snapshot.lock,
            holder,
            purpose,
            battery,
            snapshot.band,
            snapshot.gate,
        )


# ── The machine log's reader ──


def read_machine(path: Path | str) -> tuple[dict[str, Any], ...]:
    """Read ``machine.jsonl``, rejecting a wrong schema or version.

    A truncated final line (a stopped write) is ignored, like ``read_trace``.
    """
    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines):
                break
            raise ValueError(f"{target}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{target}:{number}: not a JSON object")
        version = value.get("version")
        if (
            value.get("schema") != MACHINE_SCHEMA
            or isinstance(version, bool)
            or version != 1
        ):
            raise ValueError(
                f"{target}:{number}: not {MACHINE_SCHEMA} version {MACHINE_VERSION}"
            )
        out.append(value)
    return tuple(out)


def canonical_machine_line(values: dict[str, Any]) -> str:
    """One canonical machine line, with its own schema and version."""
    payload = {**values, "schema": MACHINE_SCHEMA, "version": MACHINE_VERSION}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
