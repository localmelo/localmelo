"""Executor: the workspace root.

With a root, the executor resolves the path of ``file_read`` and
``file_write`` against it, refuses a path whose real path is outside it,
and hands the tool the real path. The file tools that ``register_builtins``
gives for a root open that path without following a final symlink, refuse
anything but a regular file, and cap what they read. Its ``shell_exec`` and
``python_exec`` run in the root, in a process group of their own, within
``tool_timeout_s``; the shell starts the text of a non-zero exit with
``[exit code <n>]``, and both record their child's exit status in
``TOOL_EXIT_CODE``. Without a root, the built-ins are Phase 1a's functions.
An agent with a root names it in the system prompt of its planning requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.executor.builtins as builtin_tools
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import REFLECTION_PROMPT, SYSTEM_PROMPT
from localmelo.melo.agent.agent import WORKSPACE_NOTE
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ExecutorResultPayload
from localmelo.melo.checker.validators import (
    MAX_OUTPUT_LEN,
    TRUNCATION_SUFFIX,
    validate_executor_result,
)
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor import (
    BUILTINS,
    TOOL_EXIT_CODE,
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    WorkspacePolicy,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import Message, ToolCall, ToolDef

# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return Path(os.path.realpath(path))


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    path = tmp_path / "outside"
    path.mkdir()
    return Path(os.path.realpath(path))


def _executor(
    root: Path | str | None,
    *,
    rooted_tools: bool = True,
    tool_timeout_s: float | None = None,
) -> Executor:
    """An executor with every built-in, fenced by *root*.

    With *rooted_tools*, the built-ins are the ones ``register_builtins``
    gives for *root*; otherwise Phase 1a's.
    """
    hippo = Hippo()
    workspace_root = None if root is None else str(root)
    executor = Executor(hippo, Checker(), workspace_root=workspace_root)
    register_builtins(
        executor,
        hippo,
        workspace_root=workspace_root if rooted_tools else None,
        tool_timeout_s=tool_timeout_s,
    )
    return executor


class Spy:
    """Stands in for the file tools and records the arguments they get."""

    def __init__(self, executor: Executor) -> None:
        self.calls: list[dict[str, Any]] = []
        executor.register("file_read", self.file_read)
        executor.register("file_write", self.file_write, requires_approval=True)

    async def file_read(self, path: str) -> str:
        self.calls.append({"path": path})
        return "read"

    async def file_write(self, path: str, content: str) -> str:
        self.calls.append({"path": path, "content": content})
        return "written"


async def _run(executor: Executor, tool: str, **arguments: Any) -> ExecutionOutcome:
    return await executor.execute_structured(
        ExecutionRequest(tool_name=tool, arguments=arguments)
    )


async def _run_recording(
    executor: Executor, tool: str, **arguments: Any
) -> tuple[ExecutionOutcome, list[int]]:
    """Run one call with a ``TOOL_EXIT_CODE`` slot, as a run with a sink does."""
    codes: list[int] = []
    slot = TOOL_EXIT_CODE.set(codes)
    try:
        outcome = await _run(executor, tool, **arguments)
    finally:
        TOOL_EXIT_CODE.reset(slot)
    return outcome, codes


@contextlib.contextmanager
def _stdin_holding(data: bytes) -> Iterator[None]:
    """Put a pipe that holds *data* on this process's standard input.

    pytest's capture already puts ``/dev/null`` on fd 0, where a command
    that inherited it would read EOF too.
    """
    saved = os.dup(0)
    read_end, write_end = os.pipe()
    os.write(write_end, data)
    os.close(write_end)
    os.dup2(read_end, 0)
    os.close(read_end)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)


async def _pid(pid_file: Path) -> int:
    """The PID a command wrote to *pid_file*, once it is there."""
    for _ in range(500):
        text = pid_file.read_text() if pid_file.exists() else ""
        if text.endswith("\n"):
            return int(text)
        await asyncio.sleep(0.01)
    raise AssertionError("the command did not write its PID")


async def _group_gone(pgid: int, timeout_s: float = 5.0) -> bool:
    """Whether no process of group *pgid* is left within *timeout_s*.

    ``PermissionError`` means "not gone yet": macOS raises it for zombies
    that are not reaped yet.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        await asyncio.sleep(0.02)
    return False


@pytest.fixture
def pid_file(tmp_path: Path) -> Iterator[Path]:
    """Where a command writes its PID; a leftover group is killed afterwards."""
    path = tmp_path / "pid"
    yield path
    if path.exists() and path.read_text().strip():
        with contextlib.suppress(OSError):
            os.killpg(int(path.read_text()), signal.SIGKILL)


# ── Paths ───────────────────────────────────────────────────────────────────


async def test_a_relative_path_resolves_against_the_root(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    executor = _executor(root)
    spy = Spy(executor)

    read = await _run(executor, "file_read", path="sub/notes.txt")
    written = await _run(executor, "file_write", path="new.txt", content="hi")

    assert read.status == written.status == ExecutionStatus.SUCCESS
    assert spy.calls == [
        {"path": str(root / "sub" / "notes.txt")},
        {"path": str(root / "new.txt"), "content": "hi"},
    ]
    assert list(elsewhere.iterdir()) == []


async def test_the_rooted_file_tools_work_in_the_root(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    executor = _executor(root)

    written = await _run(executor, "file_write", path="notes.txt", content="héllo")
    read = await _run(executor, "file_read", path="notes.txt")

    target = str(root / "notes.txt")
    assert (written.status, written.output) == (
        ExecutionStatus.SUCCESS,
        f"Written 5 bytes to {target}",
    )
    assert [a.path for a in written.artifacts] == [target]
    assert (root / "notes.txt").read_text(encoding="utf-8") == "héllo"
    assert (read.status, read.output) == (ExecutionStatus.SUCCESS, "héllo")
    assert [a.path for a in read.artifacts] == [target]


async def test_the_callers_request_is_not_changed(root: Path) -> None:
    executor = _executor(root)
    Spy(executor)
    request = ExecutionRequest(tool_name="file_read", arguments={"path": "a.txt"})

    await executor.execute_structured(request)

    assert request.arguments == {"path": "a.txt"}


@pytest.mark.parametrize(
    "arguments",
    [{"path": ""}, {"path": "a\0b.txt"}, {}],
    ids=["empty", "nul", "missing"],
)
@pytest.mark.parametrize("tool", ["file_read", "file_write"])
async def test_an_invalid_path_is_refused_with_step_7s_message(
    root: Path, tool: str, arguments: dict[str, Any]
) -> None:
    executor = _executor(root)
    spy = Spy(executor)
    if tool == "file_write":
        arguments = {**arguments, "content": "x"}

    outcome = await _run(executor, tool, **arguments)

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == (
        f"Argument 'path' for {tool} must be a non-empty string without NUL "
        "characters."
    )
    assert spy.calls == []


async def test_a_path_that_is_not_a_string_gets_step_4s_message(root: Path) -> None:
    executor = _executor(root)
    spy = Spy(executor)

    outcome = await _run(executor, "file_write", path=7, content="x")

    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == (
        "Arguments for file_write: 'path' must be of type string, not integer"
    )
    assert spy.calls == []


async def test_a_root_of_slash_holds_every_path(tmp_path: Path) -> None:
    notes = Path(os.path.realpath(tmp_path)) / "notes.txt"
    notes.write_text("hello")
    executor = _executor("/")

    absolute = await _run(executor, "file_read", path=str(notes))
    relative = await _run(executor, "file_read", path=str(notes).lstrip("/"))

    assert (absolute.status, absolute.output) == (ExecutionStatus.SUCCESS, "hello")
    assert (relative.status, relative.output) == (ExecutionStatus.SUCCESS, "hello")
    policy = WorkspacePolicy("/")
    assert policy.check_path(str(notes))
    assert policy.resolve(str(notes).lstrip("/")) == str(notes)


async def test_a_link_inside_that_points_out_is_refused(
    root: Path, outside: Path
) -> None:
    (outside / "secret.txt").write_text("secret")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="link.txt")

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert outcome.error == "Path 'link.txt' is outside allowed workspace root"
    assert "secret" not in outcome.output


async def test_a_dangling_link_that_points_out_is_refused_on_write(
    root: Path, outside: Path
) -> None:
    (root / "link.txt").symlink_to(outside / "new.txt")
    executor = _executor(root)

    outcome = await _run(executor, "file_write", path="link.txt", content="x")

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert not (outside / "new.txt").exists()


async def test_a_link_outside_that_points_in_is_allowed(
    root: Path, outside: Path
) -> None:
    (root / "notes.txt").write_text("inside")
    (outside / "link.txt").symlink_to(root / "notes.txt")
    executor = _executor(root)
    spy = Spy(executor)

    outcome = await _run(executor, "file_read", path=str(outside / "link.txt"))

    assert outcome.status == ExecutionStatus.SUCCESS
    assert spy.calls == [{"path": str(root / "notes.txt")}]


async def test_dot_dot_is_resolved(root: Path, outside: Path) -> None:
    (root / "sub").mkdir()
    executor = _executor(root)
    spy = Spy(executor)

    inside = await _run(executor, "file_read", path="sub/../notes.txt")
    out = await _run(executor, "file_read", path="../outside/secret.txt")

    assert inside.status == ExecutionStatus.SUCCESS
    assert spy.calls == [{"path": str(root / "notes.txt")}]
    assert out.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert out.error == "Path '../outside/secret.txt' is outside allowed workspace root"


async def test_a_tilde_is_not_expanded(root: Path) -> None:
    (root / "~").mkdir()
    (root / "~" / "notes.txt").write_text("in the root")
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="~/notes.txt")

    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, "in the root")
    assert [a.path for a in outcome.artifacts] == [str(root / "~" / "notes.txt")]


async def test_a_path_in_another_case_than_the_root_is_refused(
    tmp_path: Path,
) -> None:
    # On a case-insensitive file system this names the same file, but the
    # fence compares the spelling, so it fails closed.
    root = Path(os.path.realpath(tmp_path)) / "Root"
    root.mkdir()
    (root / "notes.txt").write_text("hello")
    executor = _executor(root)
    other_case = str(root.parent / "root" / "notes.txt")

    outcome = await _run(executor, "file_read", path=other_case)

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION


def test_the_policy_compares_whole_components() -> None:
    policy = WorkspacePolicy("/srv/work")
    assert policy.resolve("notes.txt") == os.path.realpath("/srv/work/notes.txt")
    assert policy.resolve("/srv/work") == os.path.realpath("/srv/work")
    assert policy.resolve("/srv/workshop/notes.txt") is None
    assert not policy.check_path("/srv/workshop/notes.txt")
    assert policy.check_path("/srv/work/sub/notes.txt")


@pytest.mark.parametrize("path", ["", "a\0b", None, 3, ["a"]])
def test_the_policy_refuses_what_is_not_a_path(path: object) -> None:
    with pytest.raises(ValueError, match="non-empty str without NUL"):
        WorkspacePolicy("/srv/work").resolve(path)


# ── File hardening ──────────────────────────────────────────────────────────


async def test_a_fifo_is_refused_for_read(root: Path) -> None:
    os.mkfifo(root / "pipe")
    executor = _executor(root)

    outcome = await asyncio.wait_for(_run(executor, "file_read", path="pipe"), 5.0)

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert outcome.error == f"Not a regular file: {root / 'pipe'}"


async def test_a_fifo_is_refused_for_write(root: Path) -> None:
    fifo = root / "pipe"
    os.mkfifo(fifo)
    executor = _executor(root)

    # Without a reader, opening it for writing fails at once (ENXIO).
    alone = await asyncio.wait_for(
        _run(executor, "file_write", path="pipe", content="x"), 5.0
    )
    # With a reader, it opens, and the regular-file check refuses it.
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        read_end_open = await asyncio.wait_for(
            _run(executor, "file_write", path="pipe", content="x"), 5.0
        )
        assert os.read(reader, 10) == b""
    finally:
        os.close(reader)

    assert alone.status == ExecutionStatus.ERROR
    assert alone.error_category == ErrorCategory.RUNTIME_ERROR
    assert read_end_open.error == f"Not a regular file: {fifo}"


async def test_a_directory_is_refused_for_read(root: Path) -> None:
    (root / "sub").mkdir()
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="sub")

    assert outcome.error == f"Not a regular file: {root / 'sub'}"


async def test_a_long_file_is_cut_with_a_note_the_result_check_keeps(
    root: Path,
) -> None:
    (root / "big.txt").write_text("x" * 300_000)
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="big.txt")

    note = "\n... [truncated: the file has 300000 bytes]"
    assert outcome.status == ExecutionStatus.SUCCESS
    assert outcome.output == "x" * (MAX_OUTPUT_LEN - len(note)) + note
    check = validate_executor_result(
        ExecutorResultPayload(tool_name="file_read", output=outcome.output)
    )
    assert check.allowed and check.sanitized_payload is None


async def test_a_text_longer_than_the_output_limit_is_cut(root: Path) -> None:
    # Within the byte cap, but more characters than the result check keeps.
    (root / "exact.txt").write_text("a" * MAX_OUTPUT_LEN)
    (root / "over.txt").write_text("b" * (MAX_OUTPUT_LEN + 1))
    executor = _executor(root)

    exact = await _run(executor, "file_read", path="exact.txt")
    over = await _run(executor, "file_read", path="over.txt")

    assert exact.output == "a" * MAX_OUTPUT_LEN
    note = f"\n... [truncated: the file has {MAX_OUTPUT_LEN + 1} bytes]"
    assert over.output == "b" * (MAX_OUTPUT_LEN - len(note)) + note


async def test_the_read_cap_holds_the_output_limit_in_any_utf_8(root: Path) -> None:
    # Four bytes a character: the cap still holds MAX_OUTPUT_LEN of them.
    (root / "wide.txt").write_text("😀" * 60_000, encoding="utf-8")
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="wide.txt")

    note = "\n... [truncated: the file has 240000 bytes]"
    assert outcome.output == "😀" * (MAX_OUTPUT_LEN - len(note)) + note


async def test_a_character_cut_at_the_cap_is_dropped(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin_tools, "MAX_FILE_READ_BYTES", 5)
    (root / "euro.txt").write_text("abc€", encoding="utf-8")  # 6 bytes
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="euro.txt")

    assert (outcome.status, outcome.output) == (
        ExecutionStatus.SUCCESS,
        "abc\n... [truncated: the file has 6 bytes]",
    )


async def test_a_file_that_is_not_utf_8_is_an_error(root: Path) -> None:
    (root / "latin1.txt").write_bytes("café".encode("latin-1"))
    executor = _executor(root)

    outcome = await _run(executor, "file_read", path="latin1.txt")

    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert "utf-8" in outcome.error


@pytest.mark.parametrize("tool", ["file_read", "file_write"])
async def test_a_final_symlink_created_after_resolution_is_refused(
    root: Path, outside: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    secret = outside / "secret.txt"
    secret.write_text("secret")
    notes = root / "notes.txt"
    notes.write_text("notes")
    resolve = WorkspacePolicy.resolve

    def resolve_then_swap(self: WorkspacePolicy, path: object) -> str | None:
        real = resolve(self, path)
        notes.unlink()
        notes.symlink_to(secret)
        return real

    monkeypatch.setattr(WorkspacePolicy, "resolve", resolve_then_swap)
    executor = _executor(root)
    arguments: dict[str, Any] = {"path": "notes.txt"}
    if tool == "file_write":
        arguments["content"] = "overwritten"

    outcome = await _run(executor, tool, **arguments)

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert "secret" not in outcome.output
    assert secret.read_text() == "secret"


async def test_a_final_symlink_inside_the_root_reaches_its_target(
    root: Path,
) -> None:
    # The path is resolved before the tool opens it, so a symlink to a file
    # in the root reaches that file: only one that takes the file's place
    # after the check is refused (above).
    target = root / "target.txt"
    target.write_text("hello")
    (root / "alias").symlink_to(target)
    executor = _executor(root)

    read = await _run(executor, "file_read", path="alias")
    write = await _run(executor, "file_write", path="alias", content="new")

    assert (read.status, read.output) == (ExecutionStatus.SUCCESS, "hello")
    assert (write.status, write.output) == (
        ExecutionStatus.SUCCESS,
        f"Written 3 bytes to {target}",
    )
    assert target.read_text() == "new"
    assert (root / "alias").is_symlink()


async def test_text_utf_8_cannot_encode_leaves_the_file(root: Path) -> None:
    (root / "notes.txt").write_text("keep me")
    executor = _executor(root)

    outcome = await _run(executor, "file_write", path="notes.txt", content="\ud800")

    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert (root / "notes.txt").read_text() == "keep me"


# ── With and without a root ─────────────────────────────────────────────────


def test_without_a_root_the_callables_are_phase_1as() -> None:
    executor = _executor(None)
    for tool_def, fn in BUILTINS:
        assert executor._callables[tool_def.name] is fn
    assert executor._timeouts_ms == {}


def test_with_a_root_the_builtins_take_exactly_their_parameters(
    root: Path,
) -> None:
    executor = _executor(root)
    for tool_def, phase1a_fn in BUILTINS:
        fn = executor._callables[tool_def.name]
        assert fn is not phase1a_fn
        assert list(inspect.signature(fn).parameters) == list(
            tool_def.parameters["properties"]
        )
    # Without tool_timeout_s, the limits are Phase 1a's: the executor's
    # default for every tool, and SHELL_TIMEOUT_S inside the shell.
    assert executor._timeouts_ms == {}


def test_tool_timeout_s_needs_a_workspace_root() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    with pytest.raises(ValueError, match="^tool_timeout_s needs a workspace_root$"):
        register_builtins(executor, hippo, tool_timeout_s=10)
    assert executor._callables == {}


async def test_without_a_root_python_exec_runs_in_process() -> None:
    executor = _executor(None)

    outcome = await _run(executor, "python_exec", code="import os; print(os.getpid())")

    assert outcome.output == str(os.getpid())


# ── shell_exec with a root ──────────────────────────────────────────────────


async def test_the_shell_starts_in_the_root_with_no_input(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    executor = _executor(root)
    read = 'read line; echo "read: $?"'

    where = await _run(executor, "shell_exec", command="pwd")
    # This process's stdin holds a line, so only a shell whose stdin is
    # /dev/null reads EOF: one that inherited it reads the line.
    with _stdin_holding(b"hello\n"):
        stdin = await asyncio.wait_for(_run(executor, "shell_exec", command=read), 10.0)
    with _stdin_holding(b"hello\n"):
        inherited, _ = await asyncio.wait_for(builtin_tools._run_shell(read), 10.0)

    assert (where.status, where.output) == (ExecutionStatus.SUCCESS, str(root))
    assert stdin.output == "read: 1"
    assert inherited == "read: 0"


async def test_a_cwd_argument_never_runs(root: Path) -> None:
    executor = _executor(root)
    asked: list[ExecutionRequest] = []

    async def gate(request: ExecutionRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision(True)

    arguments = {"command": "touch ran", "cwd": "/"}
    without_a_policy = await _run(executor, "shell_exec", **arguments)
    with_a_policy = await executor.execute_structured(
        ExecutionRequest(tool_name="shell_exec", arguments=arguments), approve=gate
    )

    assert without_a_policy.error_category == ErrorCategory.RUNTIME_ERROR
    assert "unexpected keyword argument 'cwd'" in without_a_policy.error
    assert with_a_policy.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert asked == []
    assert not (root / "ran").exists()


@pytest.mark.parametrize(
    ("command", "output", "code"),
    [
        ("echo hi; exit 3", "[exit code 3]\nhi", 3),
        ("exit 3", "[exit code 3]", 3),
        ("echo out; echo err >&2; exit 2", "[exit code 2]\nout\n\nerr", 2),
        ("echo hi; exit 0", "hi", 0),
        ("exit 0", "", 0),
        ("kill -9 $$", "[exit code -9]", -9),
    ],
    ids=["output", "no_output", "stderr", "zero", "zero_no_output", "signal"],
)
async def test_the_exit_status_starts_the_text(
    root: Path, command: str, output: str, code: int
) -> None:
    executor = _executor(root)

    outcome, codes = await _run_recording(executor, "shell_exec", command=command)

    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, output)
    assert codes == [code]


async def test_the_exit_status_line_survives_the_result_check(root: Path) -> None:
    executor = _executor(root)

    outcome, codes = await _run_recording(
        executor,
        "shell_exec",
        command="head -c 60000 /dev/zero | tr '\\0' x; exit 1",
    )

    # execute_structured runs no result check: the whole output follows the
    # line.
    assert outcome.output == "[exit code 1]\n" + "x" * 60_000
    assert codes == [1]
    check = validate_executor_result(
        ExecutorResultPayload(tool_name="shell_exec", output=outcome.output)
    )
    checked = check.sanitized_payload.output
    assert len(checked) == MAX_OUTPUT_LEN + len(TRUNCATION_SUFFIX)
    assert checked.startswith("[exit code 1]\n")
    assert checked.endswith(TRUNCATION_SUFFIX)


async def test_the_unrooted_shell_records_no_exit_status() -> None:
    executor = _executor(None)

    outcome, codes = await _run_recording(executor, "shell_exec", command="exit 3")

    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, "")
    assert codes == []


async def test_tool_timeout_s_limits_the_shell(root: Path, pid_file: Path) -> None:
    executor = _executor(root, tool_timeout_s=1)

    outcome, codes = await _run_recording(
        executor, "shell_exec", command=f'echo $$ > "{pid_file}"; sleep 5'
    )

    assert outcome.status == ExecutionStatus.TIMEOUT
    assert outcome.error == "Command timed out after 1 s"
    assert codes == []
    assert await _group_gone(await _pid(pid_file))
    assert executor._timeouts_ms["shell_exec"] == 31_000


async def test_a_cancelled_shell_records_no_exit_status(
    root: Path, pid_file: Path
) -> None:
    executor = _executor(root)
    codes: list[int] = []

    async def call() -> ExecutionOutcome:
        TOOL_EXIT_CODE.set(codes)
        return await _run(
            executor, "shell_exec", command=f'echo $$ > "{pid_file}"; sleep 60'
        )

    running = asyncio.create_task(call())
    pgid = await _pid(pid_file)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert codes == []
    assert await _group_gone(pgid)


async def test_without_tool_timeout_s_the_shell_keeps_shell_timeout_s(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin_tools, "SHELL_TIMEOUT_S", 0.5)
    rooted = _executor(root)
    unrooted = _executor(None)

    in_the_root = await _run(rooted, "shell_exec", command="sleep 5")
    phase1a = await _run(unrooted, "shell_exec", command="sleep 5")

    assert in_the_root.error == phase1a.error == "Command timed out after 0.5 s"
    assert "shell_exec" not in rooted._timeouts_ms


# ── python_exec with a root ─────────────────────────────────────────────────


async def test_python_runs_in_a_child_in_the_root(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    executor = _executor(root)

    outcome, codes = await _run_recording(
        executor,
        "python_exec",
        code="import os\nprint(os.getpid())\nprint(os.getcwd())\nos.chdir('/')",
    )

    pid, cwd = outcome.output.splitlines()
    assert outcome.status == ExecutionStatus.SUCCESS
    assert int(pid) != os.getpid()
    assert cwd == str(root)
    assert codes == [0]
    assert os.getcwd() == str(Path(os.path.realpath(tmp_path)))


@pytest.mark.parametrize(
    ("code", "status", "error", "exit_code"),
    [
        (
            "import sys\nsys.exit(3)",
            ExecutionStatus.ERROR,
            "Python exited with code 3",
            3,
        ),
        ("import sys\nprint('bye')\nsys.exit(0)", ExecutionStatus.SUCCESS, "", 0),
        (
            "print('hi')\nraise SystemExit(2)",
            ExecutionStatus.ERROR,
            "Python exited with code 2:\nhi",
            2,
        ),
    ],
    ids=["exit_3", "exit_0", "exit_2_with_output"],
)
async def test_python_survives_sys_exit(
    root: Path,
    code: str,
    status: ExecutionStatus,
    error: str,
    exit_code: int,
) -> None:
    executor = _executor(root)

    outcome, codes = await _run_recording(executor, "python_exec", code=code)

    assert (outcome.status, outcome.error) == (status, error)
    if status == ExecutionStatus.ERROR:
        assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    else:
        assert outcome.output == "bye"
    assert codes == [exit_code]


async def test_a_python_error_keeps_the_end_of_the_output(root: Path) -> None:
    executor = _executor(root)

    outcome = await _run(
        executor, "python_exec", code="print('y' * 10_000)\nraise ValueError('boom')"
    )

    assert outcome.error.startswith("Python exited with code 1:\n")
    tail = outcome.error.removeprefix("Python exited with code 1:\n")
    assert len(tail) == 4_000
    assert tail.endswith("ValueError: boom")


async def test_input_gets_eof_instead_of_hanging(root: Path) -> None:
    executor = _executor(root)

    outcome = await asyncio.wait_for(
        _run(executor, "python_exec", code="input()"), 10.0
    )

    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert "EOFError" in outcome.error


def _hang(pid_file: Path) -> str:
    """Python code that writes its PID to *pid_file* and then never ends."""
    return (
        "import os\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()) + '\\n')\n"
        "while True: pass\n"
    )


async def test_a_hang_ends_at_the_executors_limit(root: Path, pid_file: Path) -> None:
    executor = _executor(root)

    outcome = await executor.execute_structured(
        ExecutionRequest(
            tool_name="python_exec",
            arguments={"code": _hang(pid_file)},
            timeout_ms=500,
        )
    )

    assert outcome.status == ExecutionStatus.TIMEOUT
    assert outcome.error_category == ErrorCategory.TIMEOUT
    assert await _group_gone(await _pid(pid_file))


async def test_tool_timeout_s_limits_python(root: Path, pid_file: Path) -> None:
    executor = _executor(root, tool_timeout_s=1)

    outcome, codes = await _run_recording(executor, "python_exec", code=_hang(pid_file))

    assert outcome.status == ExecutionStatus.TIMEOUT
    assert outcome.error == "Execution timed out after 1.0s"
    assert codes == []
    assert await _group_gone(await _pid(pid_file))
    assert executor._timeouts_ms["python_exec"] == 1_000


async def test_python_without_an_interpreter(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(root)
    monkeypatch.setattr(sys, "executable", "")

    outcome = await _run(executor, "python_exec", code="print(1)")

    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert outcome.error == "No Python interpreter is available for python_exec"


# ── What the model is told ──────────────────────────────────────────────────


class _PromptRecorder(BaseLLMProvider):
    """Reads one file, then reflects; keeps the system prompt of each request."""

    def __init__(self) -> None:
        self.replies = [
            Message(
                role="assistant",
                tool_calls=[ToolCall(tool_name="file_read", arguments={"path": "a"})],
            ),
            Message(role="assistant", content='{"recommended_action": "stop"}'),
        ]
        self.system_prompts: list[str] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.system_prompts.append(messages[0].content)
        return self.replies.pop(0)

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("rooted", [True, False], ids=["root", "no_root"])
async def test_the_workspace_note_is_given_only_with_a_root(
    root: Path, monkeypatch: pytest.MonkeyPatch, rooted: bool
) -> None:
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)
    llm = _PromptRecorder()
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset({"file_read"}),
            max_attempts=1,
            steps_per_attempt=1,
            workspace_root=str(root) if rooted else None,
        ),
    )

    result = await agent.run_task("read a")

    assert result.status == "budget_exhausted"
    planning, reflecting = llm.system_prompts
    note = WORKSPACE_NOTE.format(root=root)
    assert note == (
        f"Working directory: {root}. Relative paths and shell and Python commands "
        "run there; file_read and file_write refuse paths outside it."
    )
    if rooted:
        assert planning == f"{SYSTEM_PROMPT}\n\n{note}"
    else:
        assert planning == SYSTEM_PROMPT
    # A reflection builds its own system prompt, without the note.
    assert reflecting.startswith(REFLECTION_PROMPT)
    assert "Working directory:" not in reflecting
