"""Start a subprocess behind a parent-owned release pipe."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

PROCESS_RECORD: contextvars.ContextVar[Callable[[int], Awaitable[None]] | None] = (
    contextvars.ContextVar("PROCESS_RECORD", default=None)
)


class UnsupportedProcessGateError(RuntimeError):
    """The platform cannot spawn the gated bootstrap."""


def gate_supported() -> bool:
    """Whether this platform can run a gated launch and record its identity."""
    return sys.platform.startswith("linux") or sys.platform == "darwin"


# This trusted bootstrap performs no task action before the parent releases it.
_BOOTSTRAP = """import json,os,sys
f = int(sys.argv[1])
def read(n):
    data = b''
    while len(data) < n:
        part = os.read(f, n - len(data))
        if not part:
            os._exit(125)
        data += part
    return data
n = int.from_bytes(read(4), 'big')
env = json.loads(read(n))
if read(1) != b'1':
    os._exit(125)
os.close(f)
os.execvpe(sys.argv[2], sys.argv[2:], env)
"""


async def _reap_unreleased(proc: asyncio.subprocess.Process) -> None:
    try:
        await asyncio.wait_for(proc.wait(), 3)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


async def _writable(fd: int) -> None:
    loop = asyncio.get_running_loop()
    ready = loop.create_future()

    def wake() -> None:
        if not ready.done():
            ready.set_result(None)

    loop.add_writer(fd, wake)
    try:
        await ready
    finally:
        loop.remove_writer(fd)


async def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        try:
            remaining = remaining[os.write(fd, remaining) :]
        except BlockingIOError:
            await _writable(fd)


async def gated_exec(
    argv: Sequence[str],
    *,
    record: Callable[[int], Awaitable[None]],
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    """Spawn a new group, record its identity, then release it to exec.

    The callback must commit its identity and verify Stop under its own
    coordinator before returning. An EOF, cancellation, or failed callback
    never releases the child; only the new child is reaped here.
    """
    if not argv or not sys.executable:
        raise ValueError("a subprocess command and Python interpreter are required")
    target_env = kwargs.pop("env", None)
    encoded = json.dumps(dict(os.environ if target_env is None else target_env)).encode(
        "utf-8"
    )
    read_fd, write_fd = os.pipe()
    payload = len(encoded).to_bytes(4, "big") + encoded
    os.set_blocking(write_fd, False)
    proc: asyncio.subprocess.Process | None = None
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _BOOTSTRAP,
            str(read_fd),
            *argv,
            pass_fds=(read_fd,),
            start_new_session=True,
            env={"PATH": os.defpath, "LANG": "C"},
            **kwargs,
        )
    )
    try:
        try:
            proc = await asyncio.shield(spawn)
        except NotImplementedError as exc:
            raise UnsupportedProcessGateError(
                "gated subprocess spawn is unsupported"
            ) from exc
        except asyncio.CancelledError:
            os.close(write_fd)
            write_fd = -1
            proc = await spawn
            await _reap_unreleased(proc)
            raise
        finally:
            os.close(read_fd)
        try:
            await _write_all(write_fd, payload)
            await _writable(write_fd)
            try:
                await record(proc.pid)
            except UnsupportedProcessGateError as exc:
                raise RuntimeError("process recording failed") from exc
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise asyncio.CancelledError()
            if os.write(write_fd, b"1") != 1:
                raise RuntimeError("short subprocess gate write")
        except BaseException:
            os.close(write_fd)
            write_fd = -1
            await _reap_unreleased(proc)
            raise
        return proc
    finally:
        if write_fd >= 0:
            os.close(write_fd)
