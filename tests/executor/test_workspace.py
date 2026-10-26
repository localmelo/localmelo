"""Executor: the workspace root.

With a root, the executor resolves the path of ``file_read`` and
``file_write`` against it, refuses a path whose real path is outside it,
and hands the tool the real path. The file tools that ``register_builtins``
gives for a root open that path without following a final symlink, refuse
anything but a regular file, and cap what they read. Without a root, the
built-ins are Phase 1a's functions.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.executor.builtins as builtin_tools
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ExecutorResultPayload
from localmelo.melo.checker.validators import MAX_OUTPUT_LEN, validate_executor_result
from localmelo.melo.executor import (
    BUILTINS,
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    WorkspacePolicy,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo

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


def _executor(root: Path | str | None, *, rooted_tools: bool = True) -> Executor:
    """An executor with every built-in, fenced by *root*.

    With *rooted_tools*, the built-ins are the ones ``register_builtins``
    gives for *root*; otherwise Phase 1a's.
    """
    hippo = Hippo()
    workspace_root = None if root is None else str(root)
    executor = Executor(hippo, Checker(), workspace_root=workspace_root)
    register_builtins(
        executor, hippo, workspace_root=workspace_root if rooted_tools else None
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


def test_with_a_root_the_file_tools_take_exactly_their_parameters(
    root: Path,
) -> None:
    executor = _executor(root)
    phase1a = {tool_def.name: fn for tool_def, fn in BUILTINS}
    for tool_def, _ in BUILTINS:
        if tool_def.name not in ("file_read", "file_write"):
            continue
        fn = executor._callables[tool_def.name]
        assert fn is not phase1a[tool_def.name]
        assert list(inspect.signature(fn).parameters) == list(
            tool_def.parameters["properties"]
        )
