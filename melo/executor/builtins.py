"""Built-in tool definitions and their implementations."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import contextvars
import os
import signal
import stat
import subprocess
from collections.abc import Callable, Coroutine
from typing import Any

from localmelo.melo.checker.validators import MAX_OUTPUT_LEN
from localmelo.melo.schema import ToolDef

# A built-in tool's callable.
ToolFunction = Callable[..., Coroutine[Any, Any, str]]

BUILTINS: list[tuple[ToolDef, ToolFunction]] = []

# The built-ins that run code or change files. register_builtins marks them,
# so a call of one asks the run's approval policy, when the run has one,
# before it runs (see Executor.execute_structured).
APPROVAL_BUILTINS: frozenset[str] = frozenset(
    {"shell_exec", "python_exec", "file_write"}
)

# Exit statuses of the child processes a tool call ran, for the exit_code of
# its tool.finished event. A run with an event sink sets a new list around
# each call. A tool that runs a child process may append the child's
# returncode (negative when a signal ended it) once the child has exited by
# itself; a tool that kills its child appends nothing. Without a list
# (None, the default), there is nothing to record.
TOOL_EXIT_CODE: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "TOOL_EXIT_CODE", default=None
)

# Seconds a shell_exec command may run before it is killed.
SHELL_TIMEOUT_S = 30.0

# Seconds to wait for a killed shell_exec command to be reaped.
_REAP_TIMEOUT_S = 5.0

# Bytes the file_read of a workspace reads at most. As UTF-8 they always
# hold the MAX_OUTPUT_LEN characters that the executor's result check keeps.
MAX_FILE_READ_BYTES = 200_000


def _builtin(tool_def: ToolDef) -> Callable[[ToolFunction], ToolFunction]:
    def decorator(fn: ToolFunction) -> ToolFunction:
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
    except TimeoutError:
        await _kill_process_group(proc)
        # The executor reports a tool's own TimeoutError by its message, so
        # the error names this limit rather than the executor's timeout.
        raise TimeoutError(f"Command timed out after {SHELL_TIMEOUT_S:g} s") from None
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
        try:
            exec(code, ns)
        except SystemExit as exc:
            # sys.exit() and exit() raise SystemExit, which would end melo
            # itself, before the run is reported. The call fails instead,
            # as for any other exception, and the run goes on.
            raise RuntimeError(
                f"Code called sys.exit({exc.code!r}), which python_exec "
                "does not allow"
            ) from None
    return buf.getvalue().strip()


# ── Built-ins with a workspace root ──


def rooted_builtins(root: str) -> dict[str, tuple[ToolFunction, float | None]]:
    """The built-ins that work in the workspace *root*, by tool name.

    Each value is the tool's callable and its timeout in milliseconds, or
    ``None`` for the executor's default. A callable takes exactly the
    parameters of the tool's :class:`ToolDef`, so a call cannot pass more.

    - ``file_read`` and ``file_write`` open the path they get, which the
      executor has resolved against its own root and fenced (see
      :meth:`~localmelo.melo.executor.policy.WorkspacePolicy.resolve`), with
      ``O_NOFOLLOW`` and ``O_NONBLOCK``: a symlink that took the checked
      file's place is refused, and so is anything but a regular file, such
      as a FIFO, which would block a worker thread forever. A relative path
      joins *root*. ``file_read`` reads at most :data:`MAX_FILE_READ_BYTES`
      bytes as strict UTF-8; a longer text keeps as many characters as fit
      in ``MAX_OUTPUT_LEN`` together with a note of the file's size, so the
      executor's result check never cuts the note off. ``file_write``
      writes UTF-8 and reports the real path.

    The other built-ins keep their Phase 1a functions.
    """
    root = os.path.realpath(root)
    return {
        "file_read": (_rooted_file_read(root), None),
        "file_write": (_rooted_file_write(root), None),
    }


def _rooted_file_read(root: str) -> ToolFunction:
    async def file_read(path: str) -> str:
        target = os.path.join(root, path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read_regular_file, target)

    return file_read


def _rooted_file_write(root: str) -> ToolFunction:
    async def file_write(path: str, content: str) -> str:
        target = os.path.join(root, path)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_regular_file, target, content)
        # The count is in characters, as Phase 1a's file_write reports it.
        return f"Written {len(content)} bytes to {target}"

    return file_write


def _read_regular_file(path: str) -> str:
    """The text of the regular file *path*, cut as :func:`rooted_builtins` says."""
    limit = MAX_FILE_READ_BYTES
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _require_regular_file(fd, path)
        chunks: list[bytes] = []
        left = limit + 1  # one byte more tells whether the file is longer
        while left > 0:
            chunk = os.read(fd, left)
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        data = b"".join(chunks)
        size = max(os.fstat(fd).st_size, len(data))
    finally:
        os.close(fd)
    cut = len(data) > limit
    # Strict UTF-8; a character that the limit cuts in two is dropped.
    text = codecs.getincrementaldecoder("utf-8")().decode(data[:limit], final=not cut)
    if not cut and len(text) <= MAX_OUTPUT_LEN:
        return text
    note = f"\n... [truncated: the file has {size} bytes]"
    return text[: MAX_OUTPUT_LEN - len(note)] + note


def _write_regular_file(path: str, content: str) -> None:
    """Write *content* to the regular file *path* as UTF-8."""
    # Encoded first, so a text that UTF-8 cannot encode (a lone surrogate)
    # fails before the file is emptied.
    data = content.encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags, 0o666)
    try:
        _require_regular_file(fd, path)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
    finally:
        os.close(fd)


def _require_regular_file(fd: int, path: str) -> None:
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise ValueError(f"Not a regular file: {path}")
