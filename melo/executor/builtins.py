"""Built-in tool definitions and their implementations."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
from collections.abc import Callable, Coroutine
from typing import Any

from localmelo.melo.schema import ToolDef

BUILTINS: list[tuple[ToolDef, Callable[..., Coroutine[Any, Any, str]]]] = []

# Seconds a shell_exec command may run before it is killed.
SHELL_TIMEOUT_S = 30.0

# Seconds to wait for a killed shell_exec command to be reaped.
_REAP_TIMEOUT_S = 5.0


def _builtin(
    tool_def: ToolDef,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, str]]], Callable[..., Coroutine[Any, Any, str]]
]:
    def decorator(
        fn: Callable[..., Coroutine[Any, Any, str]],
    ) -> Callable[..., Coroutine[Any, Any, str]]:
        BUILTINS.append((tool_def, fn))
        return fn

    return decorator


def builtin_names() -> frozenset[str]:
    """Names of all built-in tools.

    The one source of known names: every selection of built-ins is
    checked against it.
    """
    return frozenset(tool_def.name for tool_def, _ in BUILTINS)


@_builtin(
    ToolDef(
        name="shell_exec",
        description="Execute a shell command and return its output",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command"}
            },
            "required": ["command"],
        },
        semantic_tags=["shell", "bash", "command", "terminal", "exec", "run"],
    )
)
async def _shell_exec(command: str) -> str:
    # A new session puts the shell and everything it starts in one process
    # group, which is killed as a whole when the command times out or the
    # call is cancelled (the executor's timeout, the run's timeout_s).
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=SHELL_TIMEOUT_S
        )
    except BaseException:
        await _kill_process_group(proc)
        raise
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    return f"{out}\n{err}".strip() if err else out.strip()


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the process group *proc* leads, then reap *proc*."""
    # ProcessLookupError: the group is gone already. PermissionError: a
    # member switched users. Neither may replace the error being handled.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    # Bounded: before Python 3.13, wait() also waits for the output pipes
    # to close, and a process that left the group could hold them open.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_S)


@_builtin(
    ToolDef(
        name="file_read",
        description="Read the contents of a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
        semantic_tags=["file", "read", "open", "content", "text"],
    )
)
async def _file_read(path: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: open(path).read())


@_builtin(
    ToolDef(
        name="file_write",
        description="Write content to a file (creates or overwrites)",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        semantic_tags=["file", "write", "save", "create", "output"],
    )
)
async def _file_write(path: str, content: str) -> str:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: open(path, "w").write(content))
    return f"Written {len(content)} bytes to {path}"


@_builtin(
    ToolDef(
        name="python_exec",
        description="Execute Python code and return the result",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code"}},
            "required": ["code"],
        },
        semantic_tags=["python", "code", "execute", "eval", "script", "compute"],
    )
)
async def _python_exec(code: str) -> str:
    import contextlib
    import io

    buf = io.StringIO()
    ns: dict[str, Any] = {}
    # exec() runs on the event loop's thread and holds it until the code
    # returns, so neither the executor's timeout nor the run's timeout_s
    # can interrupt it (a documented limit).
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
    return buf.getvalue().strip()
