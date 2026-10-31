"""Built-in tool definitions and their implementations."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import contextvars
import itertools
import os
import signal
import stat
import subprocess
import sys
from collections.abc import Callable, Coroutine, Generator, Mapping
from typing import Any

from localmelo.melo.checker.tiers import BUILTIN_TIERS, tier_requires_approval
from localmelo.melo.checker.validators import MAX_OUTPUT_LEN
from localmelo.melo.contracts.tool_capture import _ENTRY_MARKED
from localmelo.melo.executor.env import child_env
from localmelo.melo.executor.git_status import GIT_STATUS_DEF, run_git_status
from localmelo.melo.executor.process_gate import (
    PROCESS_RECORD,
    UnsupportedProcessGateError,
    gate_supported,
    gated_exec,
)
from localmelo.melo.executor.sandbox import Confinement
from localmelo.melo.schema import ToolDef

# A built-in tool's callable.
ToolFunction = Callable[..., Coroutine[Any, Any, str]]

BUILTINS: list[tuple[ToolDef, ToolFunction]] = []

# The built-ins whose tier is not read. register_builtins marks them, so a
# call of one asks the run's approval policy, when the run has one, before
# it runs (see Executor.execute_structured).
APPROVAL_BUILTINS: frozenset[str] = frozenset(
    name for name, tier in BUILTIN_TIERS.items() if tier_requires_approval(tier)
)

# Exit statuses of the child processes a tool call ran, for the exit_code of
# its tool.finished event. A run with an event sink sets a new list around
# each call. A tool that runs a child process may append the child's
# returncode (negative when a signal ended it) once the child has exited by
# itself; a tool that kills its child appends nothing. Without a list
# (None, the default), there is nothing to record. shell_exec and
# python_exec record their child's status, with or without a workspace root.
TOOL_EXIT_CODE: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "TOOL_EXIT_CODE", default=None
)

# Seconds a shell_exec command may run before it is killed, unless the
# shell_exec of a workspace root has a tool_timeout_s of its own.
SHELL_TIMEOUT_S = 30.0

# Seconds the executor's limit of the rooted shell_exec lies above the
# shell's own, more than killing and reaping its process group can take
# (_REAP_TIMEOUT_S), so the shell's own limit fires first and names itself.
_SHELL_TIMEOUT_MARGIN_S = 30.0

# Characters of output that the error of a failed python_exec keeps.
_PYTHON_ERROR_TAIL = 4_000

# Resource limits of a python_exec child: seconds of CPU time and open file
# descriptors, as RLIMIT_CPU and RLIMIT_NOFILE. Both are a backstop, not the
# call's limit: the executor's timeout (60 s by default) is a wall clock and
# fires first, killing the child's process group. The CPU limit only ends a
# child that outlives melo, as one orphaned by SIGKILL, and lies well above
# the longest wall limit (600 s in the interactive UI); the descriptor limit
# keeps a broken loop from exhausting the machine's table. RLIMIT_AS is not
# set: macOS refuses a finite address-space limit with ValueError ("current
# limit exceeds maximum limit"), so setting it would end every call before
# the code runs.
PYTHON_CPU_LIMIT_S = 3_600
PYTHON_NOFILE_LIMIT = 1_024

# Seconds to wait for a killed shell_exec command to be reaped.
_REAP_TIMEOUT_S = 5.0

# Bytes the file_read of a workspace reads at most. As UTF-8 they always
# hold the MAX_OUTPUT_LEN characters that the executor's result check keeps.
MAX_FILE_READ_BYTES = 200_000

# What one list_files call shows at most (entries at every level), how deep
# it goes, and how many more entries it counts to say how many it left out
# (a larger rest reads "N+"; the limit bounds the walk of a huge tree).
LIST_FILES_MAX_ENTRIES = 200
LIST_FILES_MAX_DEPTH = 3
_LIST_FILES_COUNT_LIMIT = 10_000


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
    # /dev/null, never this process's stdin: a shell whose stdin is a
    # terminal that is not its controlling one (the pty slave of
    # pty.openpty(), melo started with start_new_session) opens
    # ttyname(stdin) without O_NOCTTY in its start-up check and takes it as
    # its controlling terminal; when the shell exits, the pty is hung up
    # and melo's own streams return EIO, so the flush at exit turns 0 into
    # 120.  The rooted shell_exec passes /dev/null too. The shell gets the
    # minimal environment of _run_shell, not this process's.
    text, _ = await _run_shell(command, stdin=subprocess.DEVNULL)
    return text


def _require_incomplete_entry() -> None:
    """Refuse a launch whose entry marker may not have committed."""
    if not _ENTRY_MARKED.get():
        raise RuntimeError(
            "execution evidence could not be recorded; the call was not run"
        )


async def _gated_or_fallback(
    argv: list[str], *, record: Any, **kwargs: Any
) -> asyncio.subprocess.Process:
    try:
        return await gated_exec(argv, record=record, **kwargs)
    except UnsupportedProcessGateError:
        # Only Executor can establish that the incomplete marker committed.
        _require_incomplete_entry()
        return await asyncio.create_subprocess_exec(
            *argv, start_new_session=True, **kwargs
        )


async def _run_shell(
    command: str,
    *,
    cwd: str | None = None,
    stdin: int | None = None,
    timeout_s: float | None = None,
    tool_env: Mapping[str, str] | None = None,
    confinement: Confinement | None = None,
) -> tuple[str, int]:
    """Run *command* in a shell; return its output and its exit status.

    *cwd* and *stdin* go to the shell as :mod:`subprocess` takes them
    (``None``: this process's directory and standard input). *timeout_s*
    limits the command, ``None`` standing for :data:`SHELL_TIMEOUT_S` as it
    is when the command starts. A command that runs past it is killed with
    its process group, and ``TimeoutError("Command timed out after <limit>
    s")`` is raised. The shell gets the environment of
    :func:`~localmelo.melo.executor.env.child_env` with *tool_env* on top,
    never this process's own. With *confinement*, the shell starts under its
    profile with its temporary folder, and the output starts with its line
    for refused writes (see :meth:`~localmelo.melo.executor.sandbox.WriteConfinement.note`).
    """
    limit = SHELL_TIMEOUT_S if timeout_s is None else timeout_s
    argv = ["/bin/sh", "-c", command]
    env = child_env(tool_env)
    if confinement is not None:
        argv = confinement.argv(argv)
        env.update(confinement.env())
    # A new session puts the shell and everything it starts in one process
    # group, which is killed as a whole when the command times out or the
    # call is cancelled (the executor's timeout, the run's timeout_s).
    record = PROCESS_RECORD.get()
    if record is None or not gate_supported():
        if record is not None:
            _require_incomplete_entry()
        # argv is what create_subprocess_shell runs: /bin/sh -c command.
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    else:
        proc = await _gated_or_fallback(
            argv,
            record=record,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=limit)
    except TimeoutError:
        await _kill_process_group(proc)
        # The executor reports a tool's own TimeoutError by its message, so
        # the error names this limit rather than the executor's timeout.
        raise TimeoutError(f"Command timed out after {limit:g} s") from None
    except BaseException:
        await _kill_process_group(proc)
        raise
    assert proc.returncode is not None  # communicate() waits for the exit
    text = _output_text(stdout, stderr)
    if confinement is not None:
        text = confinement.note(text) + text
    return text, proc.returncode


def _output_text(stdout: bytes, stderr: bytes) -> str:
    """A child's output as a tool returns it: stdout, then stderr."""
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
        name="list_files",
        description=(
            "List the files and folders in a folder: names and sizes, never "
            "contents. Folders end in /, links in @."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Folder path (default: the workspace)",
                },
                "depth": {
                    "type": "integer",
                    "description": "Levels of folders to list (default 1)",
                    "minimum": 1,
                    "maximum": LIST_FILES_MAX_DEPTH,
                },
            },
        },
        semantic_tags=["list", "folder", "directory", "files", "ls", "tree"],
    )
)
async def _list_files(path: str = ".", depth: int = 1) -> str:
    loop = asyncio.get_running_loop()
    target = os.path.realpath(path or ".")  # like file_read: a given path is followed
    return await loop.run_in_executor(None, _list_directory, target, depth)


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


# Bytes of the file that a file_edit changes, at most: an edit reads it whole.
MAX_EDIT_BYTES = 1_000_000


@_builtin(
    ToolDef(
        name="file_edit",
        description=(
            "Change part of an existing file: replace the exact text old, which "
            "must occur once in the file (take a few lines around the change), "
            "with new. Use it for every change to an existing file, so the "
            "change shows as a diff; file_write is for new files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old": {
                    "type": "string",
                    "description": "The exact text to replace; it must occur once",
                },
                "new": {"type": "string", "description": "The text to put there"},
            },
            "required": ["path", "old", "new"],
        },
        semantic_tags=["file", "edit", "change", "replace", "modify", "fix", "patch"],
    )
)
async def _file_edit(path: str, old: str, new: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _edit_regular_file, path, old, new)


@_builtin(GIT_STATUS_DEF)
async def _git_status(
    command: str,
    paths: list[str] | None = None,
    rev: str = "",
    count: int = 10,
    stat: bool = False,
    staged: bool = False,
) -> str:
    return await _rooted_git_status(os.getcwd())(
        command, paths=paths, rev=rev, count=count, stat=stat, staged=staged
    )


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
    # A child process, as the rooted python_exec is: code that loops, sleeps
    # or calls os._exit() must not hold or end the agent's own process, and
    # the executor's timeout and the run's timeout_s must be able to stop it.
    # Its directory is this process's, and the code arrives on its standard
    # input, a pipe, which the wrapper turns into /dev/null before the code
    # runs, so it never reads the terminal melo runs in (see _shell_exec).
    # Its environment is the minimal one of _run_python, not this process's.
    return await _run_python(code)


# ── The child environment ──


def builtins_with_env(
    tool_env: Mapping[str, str],
    confinement: Confinement | None = None,
) -> dict[str, ToolFunction]:
    """The unrooted ``shell_exec`` and ``python_exec``, with *tool_env* added.

    The functions of :data:`BUILTINS` give their child the environment of
    :func:`~localmelo.melo.executor.env.child_env` and nothing from this
    process beyond the allowlist; these add the run's own variables, such
    as ``AgentOptions.tool_env``, on top, and start under *confinement*
    when it is given (see :func:`rooted_builtins`). The dict goes where
    :func:`rooted_builtins` does (see
    :func:`~localmelo.melo.executor.register_builtins`).
    """

    async def shell_exec(command: str) -> str:
        text, _ = await _run_shell(
            command,
            stdin=subprocess.DEVNULL,
            tool_env=tool_env,
            confinement=confinement,
        )
        return text

    async def python_exec(code: str) -> str:
        return await _run_python(code, tool_env=tool_env, confinement=confinement)

    return {"shell_exec": shell_exec, "python_exec": python_exec}


# ── The child Python ──


def _posix_rlimits() -> str:
    """The program that gives a python_exec child its POSIX limits.

    The limits above are lowered to the parent's hard limit when that is
    smaller. It is a Python program because asyncio's subprocess calls
    cannot set a resource limit in the child; ``preexec_fn`` would run
    Python in a forked child of this threaded process, which can deadlock.
    The names the limits use are deleted, so the code sees no new names.
    """
    return (
        "import resource as _localmelo_resource\n"
        "def _localmelo_cap(_which, _want):\n"
        "    _soft, _hard = _localmelo_resource.getrlimit(_which)\n"
        "    if _hard != _localmelo_resource.RLIM_INFINITY and _hard < _want:\n"
        "        _want = _hard\n"
        "    _localmelo_resource.setrlimit(_which, (_want, _want))\n"
        f"_localmelo_cap(_localmelo_resource.RLIMIT_CPU, {PYTHON_CPU_LIMIT_S})\n"
        f"_localmelo_cap(_localmelo_resource.RLIMIT_NOFILE, {PYTHON_NOFILE_LIMIT})\n"
        "del _localmelo_cap, _localmelo_resource\n"
    )


def _python_wrapper() -> str:
    """The fixed program a python_exec child runs: its limits, then the code.

    The code is never in the program or in argv: it arrives on the child's
    standard input, a pipe, and this program reads it to end of file. argv
    would show the code to every user of the machine (``ps``) and would
    fail with ``E2BIG`` for code near the system's ``ARG_MAX``; a pipe has
    no such limit. ``-c`` keeps ``sys.argv == ["-c"]``, as it was before,
    and the pipe carries only the user's code (``python -`` would read the
    wrapper from the same stream, so it would need a delimiter).

    Having read the code, the program points fd 0 at ``/dev/null``
    (``os.dup2``), so ``input()`` and ``sys.stdin.read()`` in the code get
    end of file even though standard input was a pipe while the wrapper
    read it; the code never reads the terminal melo runs in.

    The code is compiled under the name ``<python_exec>`` and its lines go
    into :mod:`linecache` under that name, so a traceback names the code's
    own lines and shows them. (Compiling it as ``<string>``, the name the
    ``-c`` program itself gets, would show the limits' lines instead.) It
    runs in a fresh globals dict, as Phase 1a's ``exec(code, {})`` did,
    with ``__name__`` set to ``"__main__"``, so the names the limits use
    stay out of its reach. A failure's traceback is printed from the code's
    own frame: the frame of this program that runs it is dropped, and the
    exit status is 1, as an uncaught exception's is, or 130 for
    ``KeyboardInterrupt``, as Python's is. ``SystemExit``
    propagates, so ``sys.exit()`` keeps its status.
    """
    limits = _posix_rlimits() if os.name == "posix" else ""
    read = (
        "import os as _localmelo_os\n"
        "import sys as _localmelo_sys\n"
        "_localmelo_code = _localmelo_sys.stdin.buffer.read().decode(\n"
        "    'utf-8', 'surrogatepass')\n"
        "_localmelo_devnull = _localmelo_os.open(\n"
        "    _localmelo_os.devnull, _localmelo_os.O_RDONLY)\n"
        "_localmelo_os.dup2(_localmelo_devnull, 0)\n"
        "_localmelo_os.close(_localmelo_devnull)\n"
        "del _localmelo_devnull\n"
    )
    run = (
        "import linecache as _localmelo_linecache\n"
        "_localmelo_linecache.cache['<python_exec>'] = (\n"
        "    len(_localmelo_code), None,\n"
        "    _localmelo_code.splitlines(True), '<python_exec>')\n"
        "import traceback as _localmelo_traceback\n"
        "try:\n"
        "    exec(compile(_localmelo_code, '<python_exec>', 'exec'),\n"
        "         {'__name__': '__main__', '__builtins__': __builtins__})\n"
        "except SystemExit:\n"
        "    raise\n"
        "except BaseException:\n"
        "    _localmelo_type, _localmelo_value, _localmelo_tb = "
        "_localmelo_sys.exc_info()\n"
        "    _localmelo_traceback.print_exception(\n"
        "        _localmelo_type, _localmelo_value,\n"
        "        None if _localmelo_tb is None else _localmelo_tb.tb_next)\n"
        "    raise SystemExit(\n"
        "        130 if isinstance(_localmelo_value, KeyboardInterrupt) else 1)\n"
    )
    return limits + read + run


async def _run_python(
    code: str,
    *,
    cwd: str | None = None,
    tool_env: Mapping[str, str] | None = None,
    confinement: Confinement | None = None,
) -> str:
    """Run *code* in a child Python, in *cwd* (``None``: this directory).

    The child runs ``sys.executable -c`` with the fixed wrapper of
    :func:`_python_wrapper`; *code* goes to its standard input, a pipe,
    never into argv, so it is not in the process list and has no
    ``ARG_MAX`` limit, and the wrapper points fd 0 at ``/dev/null`` before
    running it, so ``input()`` gets EOF and it never reads the terminal
    melo runs in. The child starts a new session, so it and everything it
    starts are one process group, and the POSIX limits of
    :func:`_posix_rlimits` apply to it. Its environment is the one of
    :func:`~localmelo.melo.executor.env.child_env` with *tool_env* on top,
    never this process's own.

    A timeout or a cancellation kills that group, as for shell_exec. Once
    the child has exited by itself, its status goes to
    :data:`TOOL_EXIT_CODE`. A non-zero exit raises ``RuntimeError("Python
    exited with code <n>")``, followed by a colon, a line break and the
    last :data:`_PYTHON_ERROR_TAIL` characters of its output when there is
    any; zero returns the output, stdout then stderr. *confinement* applies
    as it does to :func:`_run_shell`; its line comes before the output.
    """
    if not sys.executable:
        raise RuntimeError("No Python interpreter is available for python_exec")
    argv = [sys.executable, "-c", _python_wrapper()]
    env = child_env(tool_env)
    if confinement is not None:
        argv = confinement.argv(argv)
        env.update(confinement.env())
    record = PROCESS_RECORD.get()
    if record is None or not gate_supported():
        if record is not None:
            _require_incomplete_entry()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    else:
        proc = await _gated_or_fallback(
            argv,
            record=record,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
    try:
        stdout, stderr = await proc.communicate(code.encode("utf-8", "surrogatepass"))
    except BaseException:
        await _kill_process_group(proc)
        raise
    assert proc.returncode is not None  # communicate() waits for the exit
    _record_exit_code(proc.returncode)
    text = _output_text(stdout, stderr)
    note = "" if confinement is None else confinement.note(text)
    if proc.returncode:
        message = f"Python exited with code {proc.returncode}"
        if text:
            message = f"{message}:\n{note}{text[-_PYTHON_ERROR_TAIL:]}"
        raise RuntimeError(message)
    return note + text


# ── Built-ins with a workspace root ──


def rooted_builtins(
    root: str,
    *,
    tool_timeout_s: float | None = None,
    tool_env: Mapping[str, str] | None = None,
    confinement: Confinement | None = None,
) -> dict[str, tuple[ToolFunction, float | None]]:
    """The built-ins that work in the workspace *root*, by tool name.

    Each value is the tool's callable and its timeout in milliseconds, or
    ``None`` for the executor's default. A callable takes exactly the
    parameters of the tool's :class:`ToolDef`, so a call cannot pass more,
    such as a ``cwd``.

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
    - ``list_files`` lists the folder the executor has resolved and fenced
      like a file's path (the root when it gets none), as
      :func:`_list_directory` says; it never reads a file or enters a link.
    - ``shell_exec`` runs the command in *root*, with its standard input
      on ``/dev/null``, and within *tool_timeout_s* seconds, else
      :data:`SHELL_TIMEOUT_S`. With *tool_timeout_s*, the executor's limit
      for it is 30 s above that; without, the executor's default (60 s)
      stays above the shell's 30 s. So the shell's own limit fires first,
      and the error names it. A non-zero exit status starts the text with
      the line ``[exit code <n>]`` (``-<signal>`` for a child a signal
      ended), and the call still succeeds: every later cut keeps a text's
      start.
    - ``python_exec`` runs the code in a child Python, ``sys.executable
      -c`` with a fixed wrapper, in *root*. The code arrives on the child's
      standard input, a pipe, and the wrapper points fd 0 at ``/dev/null``
      before running it, so ``input()`` gets EOF, and ``sys.exit()`` or
      ``os.chdir()`` stays in the child. The executor's limit,
      *tool_timeout_s* when it is set, else its default, stops it. A
      non-zero exit raises ``RuntimeError("Python exited with code
      <n>")``, followed by a colon, a line break and the last 4,000
      characters of the output when there is any. Without a root,
      ``python_exec`` runs the same child, in this process's directory.

    Both run in a process group of their own, which a timeout or a
    cancellation kills as a whole. Once their child has exited by itself,
    they record its status in :data:`TOOL_EXIT_CODE`. Both children get
    the environment of :func:`~localmelo.melo.executor.env.child_env` with
    *tool_env* on top, never this process's own: a secret in the agent's
    environment reaches neither, unless *tool_env* names it. With
    *confinement* (see :mod:`~localmelo.melo.executor.sandbox`), both start
    under its profile, which refuses a file write outside the workspace,
    with ``TMPDIR``, ``TMP`` and ``TEMP`` on its temporary folder, and the
    text of a call that reports a refused write starts with a line that
    says so (after ``[exit code <n>]``).
    """
    root = os.path.realpath(root)
    shell_timeout_ms = (
        None
        if tool_timeout_s is None
        else (tool_timeout_s + _SHELL_TIMEOUT_MARGIN_S) * 1000
    )
    python_timeout_ms = None if tool_timeout_s is None else tool_timeout_s * 1000
    return {
        "file_read": (_rooted_file_read(root), None),
        "list_files": (_rooted_list_files(root), None),
        "file_write": (_rooted_file_write(root), None),
        "file_edit": (_rooted_file_edit(root), None),
        "git_status": (_rooted_git_status(root), None),
        "shell_exec": (
            _rooted_shell_exec(root, tool_timeout_s, tool_env, confinement),
            shell_timeout_ms,
        ),
        "python_exec": (
            _rooted_python_exec(root, tool_env, confinement),
            python_timeout_ms,
        ),
    }


def _rooted_shell_exec(
    root: str,
    tool_timeout_s: float | None,
    tool_env: Mapping[str, str] | None,
    confinement: Confinement | None = None,
) -> ToolFunction:
    async def shell_exec(command: str) -> str:
        text, returncode = await _run_shell(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            timeout_s=tool_timeout_s,
            tool_env=tool_env,
            confinement=confinement,
        )
        _record_exit_code(returncode)
        if returncode:
            line = f"[exit code {returncode}]"
            return f"{line}\n{text}" if text else line
        return text

    return shell_exec


def _rooted_python_exec(
    root: str,
    tool_env: Mapping[str, str] | None,
    confinement: Confinement | None = None,
) -> ToolFunction:
    async def python_exec(code: str) -> str:
        return await _run_python(
            code, cwd=root, tool_env=tool_env, confinement=confinement
        )

    return python_exec


def _record_exit_code(returncode: int) -> None:
    """Append *returncode* to the call's :data:`TOOL_EXIT_CODE` list, if any."""
    codes = TOOL_EXIT_CODE.get()
    if codes is not None:
        codes.append(returncode)


def _rooted_file_read(root: str) -> ToolFunction:
    async def file_read(path: str) -> str:
        target = os.path.join(root, path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read_regular_file, target)

    return file_read


def _rooted_list_files(root: str) -> ToolFunction:
    async def list_files(path: str = ".", depth: int = 1) -> str:
        target = os.path.join(root, path or ".")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _list_directory, target, depth)

    return list_files


def _rooted_file_write(root: str) -> ToolFunction:
    async def file_write(path: str, content: str) -> str:
        target = os.path.join(root, path)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_regular_file, target, content)
        # The count is in characters, as Phase 1a's file_write reports it.
        return f"Written {len(content)} bytes to {target}"

    return file_write


def _rooted_file_edit(root: str) -> ToolFunction:
    async def file_edit(path: str, old: str, new: str) -> str:
        target = os.path.join(root, path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _edit_regular_file, target, old, new)

    return file_edit


def _rooted_git_status(root: str) -> ToolFunction:
    async def git_status(
        command: str,
        paths: list[str] | None = None,
        rev: str = "",
        count: int = 10,
        stat: bool = False,
        staged: bool = False,
    ) -> str:
        return await run_git_status(
            root,
            command,
            paths=paths,
            rev=rev,
            count=count,
            stat=stat,
            staged=staged,
        )

    return git_status


def edited_text(text: str, old: str, new: str, path: str = "the file") -> str:
    """*text* with its one occurrence of *old* replaced by *new*.

    ``ValueError`` says why not: *old* is empty, missing or not unique.
    """
    if not old:
        raise ValueError("old must not be empty; use file_write for a new file")
    count = text.count(old)
    if count == 0:
        raise ValueError(f"The text to replace is not in {path}; read it again")
    if count > 1:
        raise ValueError(
            f"The text to replace occurs {count} times in {path}; "
            "take more lines around it so it occurs once"
        )
    return text.replace(old, new, 1)


def _edit_regular_file(path: str, old: str, new: str) -> str:
    """Replace the one *old* in the regular UTF-8 file *path* with *new*."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _require_regular_file(fd, path)
        if os.fstat(fd).st_size > MAX_EDIT_BYTES:
            raise ValueError(f"{path} is over {MAX_EDIT_BYTES} bytes; not edited")
        chunks = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
    finally:
        os.close(fd)
    text = b"".join(chunks).decode("utf-8")
    _write_regular_file(path, edited_text(text, old, new, path))
    lines = (old.count("\n") + 1, new.count("\n") + 1)
    return f"Edited {path}: {lines[0]} line(s) replaced with {lines[1]}"


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


def _list_directory(path: str, depth: int) -> str:
    """The listing of the folder *path*, *depth* levels down (1 to 3).

    One line per entry: folders first, each level sorted by name. A folder
    ends in ``/`` and is followed by its entries, indented; a regular file by
    its size; a symlink ends in ``@`` and is never entered. No file is read,
    and ``.git`` is left out. After :data:`LIST_FILES_MAX_ENTRIES` lines,
    ``... and N more`` counts the rest; nothing to show is ``(empty)``; a
    folder inside that cannot be read is marked ``(unreadable)``, while
    *path* itself raises the ``OSError`` it gets.
    """
    depth = max(1, min(depth, LIST_FILES_MAX_DEPTH))
    fd = _open_folder(path)
    try:
        # Closing the walk closes the folders it still holds open.
        with contextlib.closing(_walk(fd, 1, depth)) as lines:
            shown = list(itertools.islice(lines, LIST_FILES_MAX_ENTRIES))
            left = sum(1 for _ in itertools.islice(lines, _LIST_FILES_COUNT_LIMIT + 1))
    finally:
        os.close(fd)
    if left:
        plus = "+" if left > _LIST_FILES_COUNT_LIMIT else ""
        shown.append(f"... and {min(left, _LIST_FILES_COUNT_LIMIT)}{plus} more")
    return "\n".join(shown) or "(empty)"


def _open_folder(name: str, parent: int | None = None) -> int:
    """A descriptor of the folder *name* (of *parent*), never through a symlink."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent)


def _walk(fd: int, level: int, depth: int) -> Generator[str, None, None]:
    """The lines of the folder open at *fd*, and below it down to *depth*."""
    with os.scandir(fd) as scan:
        entries = sorted(
            scan,
            key=lambda e: (
                not e.is_dir(follow_symlinks=False),
                e.name.casefold(),
                e.name,
            ),
        )
    indent = "  " * (level - 1)
    for entry in entries:
        if entry.name == ".git":
            continue
        # A name is data: a control character in it must not start a line.
        name = entry.name
        if not name.isprintable():
            name = name.encode("unicode_escape").decode("ascii")
        if entry.is_symlink():
            yield f"{indent}{name}@"
        elif not entry.is_dir(follow_symlinks=False):
            yield f"{indent}{name}{_size_suffix(entry)}"
        else:
            yield f"{indent}{name}/"
            if level == depth:
                continue
            try:
                # Through the parent's descriptor, without following a link:
                # a folder swapped for a symlink since the scan is not entered.
                child = _open_folder(entry.name, fd)
                try:
                    yield from _walk(child, level + 1, depth)
                finally:
                    os.close(child)
            except OSError:
                yield f"{indent}  (unreadable)"


def _size_suffix(entry: os.DirEntry[str]) -> str:
    """``  1.5 KB`` for a regular file, else nothing (a pipe, a socket)."""
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return ""
    if not stat.S_ISREG(info.st_mode):
        return ""
    size, unit = float(info.st_size), "B"
    for larger in ("KB", "MB", "GB"):
        if size < 1024:
            break
        size, unit = size / 1024, larger
    return f"  {info.st_size} B" if unit == "B" else f"  {size:.1f} {unit}"


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
