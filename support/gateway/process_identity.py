"""Conservative process-group identity and liveness checks."""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

_BOOT_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_BOOT_TIME = re.compile(r"^[0-9]+:[0-9]{1,6}$")
_PLATFORM_UUID = re.compile(
    r'"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-'
    r'[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"'
)
_MACHINE_ID = re.compile(r"^[0-9a-f]{8,64}$")


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


def _host() -> str:
    """A durable identity for this machine, or raise OSError.

    A boot id changes on reboot and says nothing about which machine read it:
    a database copied to a second host would otherwise read its own boot id
    against the first host's and call a live group gone. ``/etc/machine-id``
    and macOS's IOPlatformUUID (the hardware UUID, kept as a SHA-256 digest)
    stay the same across reboots of one machine. ``kern.uuid`` would not do:
    it names a kernel build, shared by every Mac running that build.
    """
    if sys.platform.startswith("linux"):
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                text = Path(path).read_text().strip().lower()
            except OSError:
                continue
            if _MACHINE_ID.fullmatch(text):
                return text
        raise OSError("cannot read host identity")
    if sys.platform != "darwin":
        raise OSError("host identity unavailable on this platform")
    try:
        done = subprocess.run(
            ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OSError("cannot read host identity") from exc
    match = _PLATFORM_UUID.search(done.stdout) if done.returncode == 0 else None
    if match is None:
        raise OSError("cannot read host identity")
    # Only a digest is kept: the hardware UUID itself never reaches the
    # database, a log or an error message.
    return hashlib.sha256(match.group(1).lower().encode("ascii")).hexdigest()


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


def identity(pid: int) -> tuple[int, int, str, str, str]:
    """Identify a live group leader before releasing its start gate."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("invalid process ID")
    group, token = _birth(pid)
    if group != pid:
        raise OSError("child is not its process-group leader")
    return group, pid, _host(), _boot(), token


def gone(record: tuple[int, int, str, str], host_id: str = "") -> bool:
    """Return true only when a recorded group is verifiably absent now.

    A verified boot change (both ids read, both well formed and different)
    proves the group is gone only on the host that recorded it: no process
    survives a reboot *of that machine*. A boot id read on another host, or a
    record whose host is unknown, proves nothing. An unreadable boot id stays
    unknown as well.
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
        if kind is None or kind != _boot_kind(boot):
            return False
        try:
            current_host = _host()
        except (OSError, OverflowError, ValueError):
            return False
        return bool(host_id) and host_id == current_host
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
