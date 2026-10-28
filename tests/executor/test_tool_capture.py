"""Capture is marked incomplete only when an untracked tool actually runs."""

import asyncio
import contextlib
import functools
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.executor import Executor, register_builtins
from localmelo.melo.executor.builtins import _run_python, _run_shell
from localmelo.melo.executor.models import ExecutionRequest, ExecutionStatus
from localmelo.melo.executor.process_gate import (
    PROCESS_RECORD,
    UnsupportedProcessGateError,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import ToolDef


async def test_capture_hook_follows_approval_and_trusted_file_registration(
    tmp_path: Path,
) -> None:
    hippo = Hippo()
    executor = Executor(
        hippo, Checker(mode="interactive"), workspace_root=str(tmp_path)
    )
    register_builtins(
        executor,
        hippo,
        names={"file_read", "python_exec"},
        workspace_root=str(tmp_path),
    )
    (tmp_path / "a").write_text("safe")
    calls: list[str] = []

    async def capture() -> None:
        calls.append("capture")

    async def denied(request: ExecutionRequest) -> ApprovalDecision:
        return ApprovalDecision(False)

    async def custom(path: str) -> str:
        calls.append("tool")
        return "custom"

    token = TOOL_ENTRY.set(capture)
    try:
        result = await executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": "a"})
        )
        assert result.status == ExecutionStatus.SUCCESS and calls == []
        result = await executor.execute_structured(
            ExecutionRequest(tool_name="python_exec", arguments={"code": "pass"}),
            approve=denied,
        )
        assert result.status == ExecutionStatus.BLOCKED and calls == []
        executor.register("file_read", custom)
        result = await executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": "a"})
        )
        assert result.status == ExecutionStatus.SUCCESS and calls == ["capture", "tool"]

        async def fail() -> None:
            raise RuntimeError("capture unavailable")

        TOOL_ENTRY.set(fail)
        calls.clear()
        result = await executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": "a"})
        )
        assert result.status == ExecutionStatus.ERROR and calls == []
    finally:
        TOOL_ENTRY.reset(token)


async def test_custom_tool_entry_after_validation_and_approval() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="interactive"))
    hippo.register_tool(ToolDef(name="mcp_search", description="Search", parameters={}))
    events: list[str] = []

    async def tool(query: str) -> str:
        events.append("tool")
        return query

    async def capture() -> None:
        events.append("entry")

    async def approve(request: ExecutionRequest) -> ApprovalDecision:
        events.append("approval")
        return ApprovalDecision(True)

    executor.register("mcp_search", tool, requires_approval=True)
    token = TOOL_ENTRY.set(capture)
    try:
        bad = await executor.execute_structured(
            ExecutionRequest(tool_name="mcp_search", arguments={}), approve=approve
        )
        assert bad.status == ExecutionStatus.ERROR and events == []
        bad = await executor.execute_structured(
            ExecutionRequest(tool_name="mcp_search", arguments={})
        )
        assert bad.status == ExecutionStatus.ERROR and events == []
        good = await executor.execute_structured(
            ExecutionRequest(tool_name="mcp_search", arguments={"query": "ok"}),
            approve=approve,
        )
        assert good.status == ExecutionStatus.SUCCESS
        assert events == ["approval", "entry", "tool"]
    finally:
        TOOL_ENTRY.reset(token)


async def test_wrapped_callable_mismatch_cannot_run_without_entry() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="interactive"))
    hippo.register_tool(ToolDef(name="wrapped", description="Wrapped", parameters={}))
    calls: list[str] = []

    async def original(required: str) -> str:
        return required

    @functools.wraps(original)
    async def wrapper(**kwargs: Any) -> str:
        calls.append("wrapper")
        return "executed"

    async def capture() -> None:
        calls.append("entry")

    executor.register("wrapped", wrapper)
    token = TOOL_ENTRY.set(capture)
    try:
        outcome = await executor.execute_structured(
            ExecutionRequest(tool_name="wrapped", arguments={})
        )
        assert outcome.status == ExecutionStatus.ERROR
        assert "required" in outcome.error
        assert calls == []
        outcome = await executor.execute_structured(
            ExecutionRequest(tool_name="wrapped", arguments={"required": "ok"})
        )
        assert outcome.status == ExecutionStatus.SUCCESS
        assert calls == ["entry", "wrapper"]
    finally:
        TOOL_ENTRY.reset(token)


@pytest.mark.parametrize("entry_fails", [False, True])
async def test_unsupported_gate_falls_back_only_after_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_fails: bool
) -> None:
    hippo = Hippo()
    executor = Executor(
        hippo, Checker(mode="interactive"), workspace_root=str(tmp_path)
    )
    register_builtins(
        executor, hippo, names={"python_exec"}, workspace_root=str(tmp_path)
    )
    events: list[str] = []
    marker = tmp_path / "marker"

    async def capture() -> None:
        events.append("entry")
        if entry_fails:
            raise RuntimeError("store unavailable")

    async def record(pid: int) -> None:
        events.append("record")

    async def unsupported(*args: Any, **kwargs: Any) -> Any:
        events.append("spawn")
        raise UnsupportedProcessGateError("subprocess gate unsupported")

    monkeypatch.setattr("localmelo.melo.executor.builtins.gated_exec", unsupported)
    entry_token = TOOL_ENTRY.set(capture)
    process_token = PROCESS_RECORD.set(record)
    try:
        outcome = await executor.execute_structured(
            ExecutionRequest(
                tool_name="python_exec",
                arguments={"code": "open('marker', 'w').close()"},
            )
        )
        assert outcome.status == (
            ExecutionStatus.ERROR if entry_fails else ExecutionStatus.SUCCESS
        )
        assert events == (["entry"] if entry_fails else ["entry", "spawn"])
        assert marker.exists() is not entry_fails
    finally:
        PROCESS_RECORD.reset(process_token)
        TOOL_ENTRY.reset(entry_token)


@pytest.mark.parametrize("name", ["python_exec", "shell_exec"])
async def test_executor_entry_and_process_record_order(
    tmp_path: Path, name: str
) -> None:
    hippo = Hippo()
    executor = Executor(
        hippo, Checker(mode="interactive"), workspace_root=str(tmp_path)
    )
    register_builtins(executor, hippo, names={name}, workspace_root=str(tmp_path))
    events: list[str] = []
    marker = tmp_path / "marker"

    async def capture() -> None:
        events.append("entry")

    async def record(pid: int) -> None:
        assert pid > 0 and not marker.exists()
        events.append("record")

    entry_token = TOOL_ENTRY.set(capture)
    process_token = PROCESS_RECORD.set(record)
    try:
        args = (
            {"code": "open('marker', 'w').close()"}
            if name == "python_exec"
            else {"command": "touch marker"}
        )
        result = await executor.execute_structured(
            ExecutionRequest(tool_name=name, arguments=args)
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert events == ["entry", "record"] and marker.exists()
    finally:
        PROCESS_RECORD.reset(process_token)
        TOOL_ENTRY.reset(entry_token)


async def test_entry_callback_suppressed_cancellation_never_invokes_tool() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="interactive"))
    hippo.register_tool(ToolDef(name="custom", description="Custom", parameters={}))
    entered = asyncio.Event()
    calls: list[str] = []

    async def tool() -> str:
        calls.append("tool")
        return "ok"

    async def capture() -> None:
        entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()

    executor.register("custom", tool)
    token = TOOL_ENTRY.set(capture)
    try:
        task = asyncio.create_task(
            executor.execute_structured(
                ExecutionRequest(tool_name="custom", arguments={})
            )
        )
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == []
    finally:
        TOOL_ENTRY.reset(token)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("python_exec", {"code": "open('marker', 'w').close()"}),
        ("shell_exec", {"command": "touch marker"}),
    ],
)
@pytest.mark.parametrize("marked", [True, False])
async def test_unsupported_platform_fallback_needs_incomplete_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, str],
    marked: bool,
) -> None:
    marker = tmp_path / "marker"
    events: list[str] = []

    async def capture() -> None:
        events.append("entry")

    async def record(pid: int) -> None:
        raise AssertionError("must not record")

    monkeypatch.setattr(
        "localmelo.melo.executor.builtins.gate_supported", lambda: False
    )
    process_token = PROCESS_RECORD.set(record)
    entry_token = TOOL_ENTRY.set(capture) if marked else None
    try:
        if marked:
            hippo = Hippo()
            executor = Executor(
                hippo, Checker(mode="interactive"), workspace_root=str(tmp_path)
            )
            register_builtins(
                executor, hippo, names={name}, workspace_root=str(tmp_path)
            )
            result = await executor.execute_structured(
                ExecutionRequest(tool_name=name, arguments=arguments)
            )
            assert result.status == ExecutionStatus.SUCCESS
            assert events == ["entry"] and marker.exists()
        else:
            with pytest.raises(RuntimeError, match="the call was not run"):
                if name == "python_exec":
                    await _run_python("open('marker', 'w').close()", cwd=str(tmp_path))
                else:
                    await _run_shell("touch marker", cwd=str(tmp_path))
            assert not marker.exists()
    finally:
        if entry_token is not None:
            TOOL_ENTRY.reset(entry_token)
        PROCESS_RECORD.reset(process_token)
