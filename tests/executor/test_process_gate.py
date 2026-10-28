"""A gated child must not execute task code before durable capture."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.executor.builtins import _run_python, _run_shell
from localmelo.melo.executor.process_gate import PROCESS_RECORD, gated_exec

# The driver crashes with os._exit so no Python cleanup runs in the parent.
_CRASH_DRIVER = """
import asyncio, os, sys
from pathlib import Path
from localmelo.melo.executor.process_gate import gated_exec

mode, marker, record_file, pid_file = sys.argv[1:5]

async def record(pid):
    Path(pid_file).write_text(str(pid))
    if mode == "before-record":
        os._exit(9)
    Path(record_file).write_text(str(pid))
    if mode == "after-record":
        os._exit(9)

async def main():
    proc = await gated_exec(
        [sys.executable, "-c",
         "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", marker],
        record=record,
    )
    if mode == "after-release":
        os._exit(9)
    await proc.wait()

asyncio.run(main())
"""


async def test_bootstrap_does_not_inherit_target_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = asyncio.create_subprocess_exec
    boot_env: dict[str, str] = {}

    async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        boot_env.update(kwargs["env"])
        return await original(*args, **kwargs)

    async def record(pid: int) -> None:
        assert "TEST_LOADER" not in boot_env
        assert "TEST_PARENT_SECRET" not in boot_env

    monkeypatch.setenv("TEST_PARENT_SECRET", "parent-only")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    proc = await gated_exec(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('TEST_LOADER'),"
            " os.environ.get('TEST_PARENT_SECRET'))",
        ],
        record=record,
        env={"TEST_LOADER": "target-only"},
        stdout=asyncio.subprocess.PIPE,
    )
    output, _ = await proc.communicate()
    assert output == b"target-only None\n"


async def test_child_waits_for_capture_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    entered, release = asyncio.Event(), asyncio.Event()

    async def record(pid: int) -> None:
        assert pid > 0
        entered.set()
        await release.wait()

    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    launch = asyncio.create_task(gated_exec(argv, record=record))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert not marker.exists()
        release.set()
        proc = await asyncio.wait_for(launch, 3)
        assert await asyncio.wait_for(proc.wait(), 3) == 0
        assert marker.exists()
    finally:
        release.set()
        if not launch.done():
            launch.cancel()
            await asyncio.gather(launch, return_exceptions=True)


async def test_failed_capture_never_releases_child(tmp_path: Path) -> None:
    marker = tmp_path / "executed"

    async def record(pid: int) -> None:
        raise RuntimeError("capture failed")

    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    with pytest.raises(RuntimeError, match="capture failed"):
        await gated_exec(argv, record=record)
    assert not marker.exists()


async def test_suppressed_cancellation_still_closes_gate(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    entered = asyncio.Event()

    async def record(pid: int) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    launch = asyncio.create_task(gated_exec(argv, record=record))
    await asyncio.wait_for(entered.wait(), 3)
    launch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(launch, 3)
    assert not marker.exists()


async def test_cancel_during_capture_does_not_release_child(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    entered = asyncio.Event()

    async def record(pid: int) -> None:
        entered.set()
        await asyncio.Event().wait()

    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    launch = asyncio.create_task(gated_exec(argv, record=record))
    await asyncio.wait_for(entered.wait(), 3)
    launch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(launch, 3)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mode", "recorded", "released"),
    [
        ("before-record", False, False),
        ("after-record", True, False),
        ("after-release", True, True),
    ],
)
async def test_parent_crash_never_releases_an_unrecorded_child(
    tmp_path: Path, mode: str, recorded: bool, released: bool
) -> None:
    marker, record_file, pid_file = (
        tmp_path / "executed",
        tmp_path / "record",
        tmp_path / "pid",
    )
    driver = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CRASH_DRIVER,
        mode,
        str(marker),
        str(record_file),
        str(pid_file),
        stderr=asyncio.subprocess.PIPE,
    )
    _, errors = await asyncio.wait_for(driver.communicate(), 10)
    assert driver.returncode == 9, errors.decode()
    assert record_file.exists() is recorded
    if released:
        for _ in range(50):
            if marker.exists():
                break
            await asyncio.sleep(0.1)
        assert marker.exists()
        return
    assert not marker.exists()
    pid = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("an unreleased child survived its crashed parent")


@pytest.mark.parametrize("kind", ["python", "shell"])
async def test_builtin_exec_waits_for_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / "marker"
    entered, release = asyncio.Event(), asyncio.Event()

    async def record(pid: int) -> None:
        assert pid > 0
        entered.set()
        await release.wait()

    token = PROCESS_RECORD.set(record)
    try:
        work = asyncio.create_task(
            _run_python(f"open({str(marker)!r}, 'w').close()", cwd=str(tmp_path))
            if kind == "python"
            else _run_shell(f"touch {marker}", cwd=str(tmp_path))
        )
        await asyncio.wait_for(entered.wait(), 3)
        assert not marker.exists()
        release.set()
        await asyncio.wait_for(work, 3)
        assert marker.exists()
    finally:
        release.set()
        PROCESS_RECORD.reset(token)
