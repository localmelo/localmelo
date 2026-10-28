"""A gated child must not execute task code before durable capture."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.executor.process_gate import gated_exec


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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    proc = await gated_exec(
        [sys.executable, "-c", "import os; print(os.environ['TEST_LOADER'])"],
        record=record,
        env={"TEST_LOADER": "target-only"},
        stdout=asyncio.subprocess.PIPE,
    )
    output, _ = await proc.communicate()
    assert output == b"target-only\n"


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
