"""Conservative process-group identity and liveness checks."""

from __future__ import annotations

import ctypes
import os
import re
import sys
import uuid
from pathlib import Path

_BOOT_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_BOOT_TIME = re.compile(r"^[0-9]+:[0-9]{1,6}$")


def _boot_kind(value: str) -> str | None:
    """The boot-id shape *value* has, or ``None`` when it does not parse.

    A stored id that does not parse, or that has the other platform's shape, is
    malformed evidence, not a boot this machine could have rebooted away from.
    """
    if _BOOT_UUID.fullmatch(value):
        return "uuid"
    match = _BOOT_TIME.fullmatch(value)
    if match and int(value.rsplit(":", 1)[1]) < 1_000_000:
        return "time"
    return None


class _BSDInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("xstatus", ctypes.c_uint32),
        ("pid", ctypes.c_uint32),
        ("ppid", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("ruid", ctypes.c_uint32),
        ("rgid", ctypes.c_uint32),
        ("svuid", ctypes.c_uint32),
        ("svgid", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("comm", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        ("nfiles", ctypes.c_uint32),
        ("pgid", ctypes.c_uint32),
        ("pjobc", ctypes.c_uint32),
        ("tdev", ctypes.c_uint32),
        ("tpgid", ctypes.c_uint32),
        ("nice", ctypes.c_int32),
        ("start_sec", ctypes.c_uint64),
        ("start_usec", ctypes.c_uint64),
    ]


class _Timeval(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("microseconds", ctypes.c_int)]


def _boot() -> str:
    if sys.platform.startswith("linux"):
        try:
            return str(
                uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
            )
        except ValueError as exc:
            raise OSError("invalid Linux boot identity") from exc
    if sys.platform != "darwin":
        raise OSError("process identity unavailable on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    libc.sysctlbyname.restype = ctypes.c_int
    value = _Timeval()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if (
        libc.sysctlbyname(
            b"kern.boottime", ctypes.byref(value), ctypes.byref(size), None, 0
        )
        != 0
        or size.value != ctypes.sizeof(value)
        or value.seconds <= 0
        or not 0 <= value.microseconds < 1_000_000
    ):
        raise OSError(ctypes.get_errno(), "cannot read boot identity")
    return f"{value.seconds}:{value.microseconds}"


def _birth(pid: int) -> tuple[int, str]:
    if sys.platform.startswith("linux"):
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            end = text.rfind(")")
            if end < 0 or int(text[: text.index(" ")]) != pid:
                raise ValueError("malformed process identity")
            fields = text[end + 1 :].split()
            pgid, start = int(fields[2]), int(fields[19])
            if pgid <= 0 or start <= 0:
                raise ValueError("invalid process identity")
            return pgid, str(start)
        except FileNotFoundError as exc:
            os.kill(pid, 0)
            raise OSError("cannot read process start identity") from exc
        except (ValueError, IndexError) as exc:
            raise OSError("malformed Linux process identity") from exc
    if sys.platform != "darwin":
        raise OSError("process identity unavailable on this platform")
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    libproc.proc_pidinfo.restype = ctypes.c_int
    info = _BSDInfo()
    size = libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    if (
        size != ctypes.sizeof(info)
        or info.pid != pid
        or not info.start_sec
        or info.start_usec >= 1_000_000
    ):
        # Only ESRCH establishes absence; denied/unsupported reads stay unknown.
        os.kill(pid, 0)
        raise OSError("cannot read process start identity")
    return int(info.pgid), f"{info.start_sec}:{info.start_usec}"


def identity(pid: int) -> tuple[int, int, str, str]:
    """Identify a live group leader before releasing its start gate."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("invalid process ID")
    group, token = _birth(pid)
    if group != pid:
        raise OSError("child is not its process-group leader")
    return group, pid, _boot(), token


def gone(record: tuple[int, int, str, str]) -> bool:
    """Return true only when a recorded group is verifiably absent now.

    A verified boot change (both ids read, both well formed and different)
    proves the group is gone: no process survives a reboot. An unreadable or
    malformed boot id stays unknown.
    """
    pgid, leader, boot, token = record
    if pgid <= 0 or leader <= 0 or pgid != leader or not boot or not token:
        return False
    try:
        current_boot = _boot()
    except (OSError, OverflowError, ValueError):
        return False
    if current_boot != boot:
        kind = _boot_kind(current_boot)
        return kind is not None and kind == _boot_kind(boot)
    try:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            try:
                _, current = _birth(leader)
            except ProcessLookupError:
                return True
            return current != token
        return False
    except (OSError, OverflowError, ValueError):
        return False
