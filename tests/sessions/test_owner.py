"""The gateway file's owner lock: one owner, released at once when it dies.

The subprocess tests start Python children that open the store; every
process a test starts is killed should the test fail.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    SessionStoreError,
    StoreOwnedError,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")

HOLDER = "testhost:1:aaaaaaaa"

OWNER = textwrap.dedent("""
    import asyncio
    import os
    import sys

    from localmelo.support.sessions.store import SessionStore


    async def main() -> None:
        store = SessionStore(sys.argv[1], writer="gateway")
        await store.open()
        if sys.argv[2] == "hold":
            print("ready", flush=True)
            await asyncio.Event().wait()
        # "orphan": a tool child in its own session, as shell_exec starts one,
        # with every inheritable descriptor left open, then a crash.
        child = await asyncio.create_subprocess_exec(
            "sleep", "60", start_new_session=True, close_fds=False
        )
        print(child.pid, flush=True)
        os._exit(0)


    asyncio.run(main())
    """)


@contextlib.asynccontextmanager
async def owner(
    path: Path, mode: str
) -> AsyncIterator[tuple[asyncio.subprocess.Process, str]]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        OWNER,
        str(path),
        mode,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        yield process, line.decode().strip()
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()


async def test_a_second_gateway_store_in_this_process_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    first = SessionStore(path, writer="gateway")
    await first.open()
    second = SessionStore(path, writer="gateway")
    with pytest.raises(StoreOwnedError) as raised:
        await second.open()
    assert str(raised.value) == (
        f"Another melo gateway (pid {os.getpid()}) is using {path}. "
        "Stop it to use the console here."
    )
    assert isinstance(raised.value, SessionStoreError)
    await first.aclose()
    await second.open()  # free at once after aclose()
    await second.aclose()


async def test_another_process_owns_the_file_until_it_is_killed(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    async with owner(path, "hold") as (process, line):
        assert line == "ready"
        store = SessionStore(path, writer="gateway")
        with pytest.raises(StoreOwnedError, match=f"pid {process.pid}"):
            await store.open()
        process.send_signal(signal.SIGKILL)
        await process.wait()
        await store.open()  # the kernel released the lock with the process
        await store.aclose()


async def test_a_failed_open_releases_the_lock_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sessions.db"
    failures = [OSError("No space left on device")]
    original = SessionStore._retain

    async def failing_retain(self: SessionStore, conn: object) -> int:
        if failures:
            raise failures.pop()
        return await original(self, conn)  # type: ignore[arg-type]

    monkeypatch.setattr(SessionStore, "_retain", failing_retain)
    store = SessionStore(path, writer="gateway")
    with pytest.raises(SessionStoreError, match="No space left on device"):
        await store.open()
    other = SessionStore(path, writer="gateway")
    await other.open()  # the failed open released the lock
    await other.aclose()
    await store.open()
    await store.aclose()


async def test_a_cli_file_is_refused_before_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    terminal = SessionStore(path, writer="cli")
    await terminal.create_session()
    await terminal.aclose()
    store = SessionStore(path, writer="gateway")
    with pytest.raises(SessionStoreError, match="belongs to the terminal melo"):
        await store.open()
    assert not Path(f"{path}.owner").exists()


async def test_a_tool_child_that_outlives_its_gateway_does_not_keep_the_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    async with owner(path, "orphan") as (process, line):
        sleep_pid = int(line)
        try:
            await process.wait()
            os.kill(sleep_pid, 0)  # the child still runs
            store = SessionStore(path, writer="gateway")
            await store.open()
            await store.aclose()
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(sleep_pid, signal.SIGKILL)


async def test_event_ids_continue_above_the_old_maximum(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer="gateway")
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    old = max(e.id or 0 for e in await store.list_events(turn.session_id))
    await store.finish_turn(
        turn.session_id,
        1,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="x",
    )
    await store.delete_session(turn.session_id)  # even with the rows gone
    await store.aclose()
    again = SessionStore(path, writer="gateway")
    turn = await again.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    new = min(e.id or 0 for e in await again.list_events(turn.session_id))
    assert new > old + 1  # turn.finished took old + 1
    await again.aclose()
