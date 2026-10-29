"""Orphan checks require known identity and never send a destructive signal."""

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway import process_identity as identity


def test_process_group_absence_checks_boot_pid_and_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boot = "11111111-1111-1111-1111-111111111111"
    other_boot = "22222222-2222-2222-2222-222222222222"
    host = "host-a"
    monkeypatch.setattr(identity, "_boot", lambda: boot)
    monkeypatch.setattr(identity, "_host", lambda: host)

    def absent(pgid: int, sig: int) -> None:
        assert pgid == 123 and sig == 0
        raise ProcessLookupError()

    monkeypatch.setattr(identity.os, "killpg", absent)

    def missing(pid: int) -> tuple[int, str]:
        raise ProcessLookupError()

    monkeypatch.setattr(identity, "_birth", missing)
    assert identity.gone((123, 123, boot, "old"))
    monkeypatch.setattr(identity, "_birth", lambda pid: (pid, "new"))
    assert identity.gone((123, 123, boot, "old"))
    monkeypatch.setattr(identity, "_birth", lambda pid: (pid, "old"))
    assert not identity.gone((123, 123, boot, "old"))
    # A verified boot change counts as gone on the host that recorded it; a
    # malformed stored id, another host or an unknown host is unknown.
    assert identity.gone((123, 123, other_boot, "old"), host)
    monkeypatch.setattr(identity, "_host", lambda: "host-b")
    assert not identity.gone((123, 123, other_boot, "old"), host)
    assert not identity.gone((123, 123, other_boot, "old"))
    monkeypatch.setattr(identity, "_host", lambda: host)
    assert not identity.gone((123, 123, "other-boot", "old"), host)
    # A stored id with the other platform's shape is malformed too.
    assert not identity.gone((123, 123, "1:2", "old"), host)


def test_unreadable_boot_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable() -> str:
        raise OSError("cannot read boot identity")

    monkeypatch.setattr(identity, "_boot", unreadable)
    monkeypatch.setattr(identity.os, "killpg", lambda pgid, sig: None)
    assert not identity.gone((123, 123, "11111111-1111-1111-1111-111111111111", "old"))


def test_group_alive_or_permission_denied_is_not_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity, "_boot", lambda: "boot")
    monkeypatch.setattr(identity.os, "killpg", lambda pgid, sig: None)
    assert not identity.gone((123, 123, "boot", "start"))

    def denied(pgid: int, sig: int) -> None:
        raise PermissionError()

    monkeypatch.setattr(identity.os, "killpg", denied)
    assert not identity.gone((123, 123, "boot", "start"))


def test_linux_boot_identity_is_normalized_and_invalid_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "linux")
    boot_file = Path("/proc/sys/kernel/random/boot_id")

    def read_boot(path: Path) -> str:
        assert path == boot_file
        return " ABCDEF01-2345-6789-ABCD-EF0123456789\n"

    monkeypatch.setattr(Path, "read_text", read_boot)
    assert identity._boot() == "abcdef01-2345-6789-abcd-ef0123456789"
    monkeypatch.setattr(Path, "read_text", lambda path: "not-a-uuid")
    with pytest.raises(OSError, match="invalid Linux boot"):
        identity._boot()


def test_unreadable_host_is_not_evidence_of_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_boot = "22222222-2222-2222-2222-222222222222"
    monkeypatch.setattr(
        identity, "_boot", lambda: "11111111-1111-1111-1111-111111111111"
    )

    def unreadable() -> str:
        raise OSError("cannot read host identity")

    monkeypatch.setattr(identity, "_host", unreadable)
    assert not identity.gone((123, 123, other_boot, "old"), "host-a")


def test_linux_host_identity_is_normalized_and_invalid_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "linux")
    machine_id = Path("/etc/machine-id")

    def read_id(path: Path) -> str:
        if path == machine_id:
            return " ABCDEF0123456789ABCDEF0123456789\n"
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", read_id)
    assert identity._host() == "abcdef0123456789abcdef0123456789"
    monkeypatch.setattr(Path, "read_text", lambda path: "not a machine id")
    with pytest.raises(OSError, match="cannot read host identity"):
        identity._host()


def test_linux_birth_parses_command_with_spaces_and_parentheses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "linux")
    fields = ["S", "7", "123"] + ["0"] * 16 + ["98765"]

    def read_stat(path: Path) -> str:
        assert path == Path("/proc/123/stat")
        return "123 (a command (with) spaces) " + " ".join(fields)

    monkeypatch.setattr(Path, "read_text", read_stat)
    assert identity._birth(123) == (123, "98765")


@pytest.mark.parametrize(
    "stat",
    [
        "",
        "123 (cmd) S",
        "124 (cmd) S",
        "123 (cmd) " + " ".join(["S", "1", "0"] + ["0"] * 16 + ["1"]),
    ],
)
def test_linux_malformed_birth_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    stat: str,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", lambda path: stat)
    with pytest.raises(OSError):
        identity._birth(123)


def test_linux_missing_stat_distinguishes_missing_from_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "linux")

    def missing_file(path: Path) -> str:
        raise FileNotFoundError()

    monkeypatch.setattr(Path, "read_text", missing_file)
    monkeypatch.setattr(identity.os, "kill", lambda pid, sig: None)
    with pytest.raises(OSError, match="cannot read"):
        identity._birth(123)

    def missing_process(pid: int, sig: int) -> None:
        assert (pid, sig) == (123, 0)
        raise ProcessLookupError()

    monkeypatch.setattr(identity.os, "kill", missing_process)
    with pytest.raises(ProcessLookupError):
        identity._birth(123)


class _NativeFunction:
    def __init__(self, callback: Callable[..., int]) -> None:
        self.callback = callback

    def __call__(self, *args: Any) -> int:
        return self.callback(*args)


def test_darwin_boot_identity_and_invalid_native_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "darwin")
    result = {
        "status": 0,
        "size": ctypes.sizeof(identity._Timeval),
        "seconds": 100,
        "usec": 42,
    }

    def sysctl(name: Any, value: Any, size: Any, new: Any, new_size: int) -> int:
        assert name == b"kern.boottime" and new is None and new_size == 0
        boot = ctypes.cast(value, ctypes.POINTER(identity._Timeval)).contents
        boot.seconds, boot.microseconds = result["seconds"], result["usec"]
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t)).contents.value = result[
            "size"
        ]
        return result["status"]

    class Library:
        sysctlbyname = _NativeFunction(sysctl)

    monkeypatch.setattr(identity.ctypes, "CDLL", lambda *args, **kwargs: Library())
    assert identity._boot() == "100:42"
    for key, invalid in (
        ("status", -1),
        ("size", 0),
        ("seconds", 0),
        ("usec", 1_000_000),
    ):
        original = result[key]
        result[key] = invalid
        with pytest.raises(OSError):
            identity._boot()
        result[key] = original


def test_darwin_birth_validates_native_result_and_process_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity.sys, "platform", "darwin")
    result = {
        "size": ctypes.sizeof(identity._BSDInfo),
        "pid": 123,
        "seconds": 100,
        "usec": 42,
    }

    def read_proc_info(pid: int, flavor: int, arg: int, buffer: Any, size: int) -> int:
        assert (pid, flavor, arg, size) == (123, 3, 0, ctypes.sizeof(identity._BSDInfo))
        info = ctypes.cast(buffer, ctypes.POINTER(identity._BSDInfo)).contents
        info.pid, info.pgid = result["pid"], 123
        info.start_sec, info.start_usec = result["seconds"], result["usec"]
        return result["size"]

    class Library:
        proc_pidinfo = _NativeFunction(read_proc_info)

    monkeypatch.setattr(identity.ctypes, "CDLL", lambda *args, **kwargs: Library())
    monkeypatch.setattr(identity.os, "kill", lambda pid, sig: None)
    assert identity._birth(123) == (123, "100:42")
    for key, invalid in (
        ("size", 0),
        ("pid", 124),
        ("seconds", 0),
        ("usec", 1_000_000),
    ):
        original = result[key]
        result[key] = invalid
        with pytest.raises(OSError, match="cannot read"):
            identity._birth(123)
        result[key] = original

    def missing(pid: int, sig: int) -> None:
        assert (pid, sig) == (123, 0)
        raise ProcessLookupError()

    result["size"] = 0
    monkeypatch.setattr(identity.os, "kill", missing)
    with pytest.raises(ProcessLookupError):
        identity._birth(123)


def test_unsupported_platform_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity.sys, "platform", "unsupported")
    with pytest.raises(OSError, match="unavailable"):
        identity._boot()
    with pytest.raises(OSError, match="unavailable"):
        identity._birth(123)
    assert not identity.gone((123, 123, "boot", "start"))


@pytest.mark.parametrize("pid", [True, 0, -1, "123"])
def test_identity_rejects_invalid_pid(pid: Any) -> None:
    with pytest.raises(ValueError):
        identity.identity(pid)


def test_identity_requires_group_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity, "_birth", lambda pid: (999, "start"))
    with pytest.raises(OSError, match="leader"):
        identity.identity(123)
    monkeypatch.setattr(identity, "_birth", lambda pid: (pid, "start"))
    monkeypatch.setattr(identity, "_boot", lambda: "boot")
    monkeypatch.setattr(identity, "_host", lambda: "host")
    assert identity.identity(123) == (123, 123, "host", "boot", "start")


@pytest.mark.parametrize(
    "record",
    [
        (0, 0, "boot", "start"),
        (123, 124, "boot", "start"),
        (123, 123, "", "start"),
        (123, 123, "boot", ""),
    ],
)
def test_invalid_record_never_proves_absence(record: tuple[int, int, str, str]) -> None:
    assert not identity.gone(record)


def test_huge_pid_never_proves_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity, "_boot", lambda: "boot")
    assert not identity.gone((2**63, 2**63, "boot", "start"))


def test_absent_group_with_unknown_birth_is_not_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity, "_boot", lambda: "boot")

    def absent(pgid: int, sig: int) -> None:
        raise ProcessLookupError()

    def denied(pid: int) -> tuple[int, str]:
        raise PermissionError()

    monkeypatch.setattr(identity.os, "killpg", absent)
    monkeypatch.setattr(identity, "_birth", denied)
    assert not identity.gone((123, 123, "boot", "start"))
