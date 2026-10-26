"""Executor: the approval gate of execute_structured.

A tool registered as requiring approval asks the gate the caller passes,
after every check that refuses a call by itself and before the tool's
timeout starts. The marked built-ins are replaced by stand-ins with their
parameters wherever a call is refused, so nothing real runs if a check
ever lets one through.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.approvals import ApprovalDecision, denial_text
from localmelo.melo.executor import (
    APPROVAL_BUILTINS,
    ApprovalGate,
    Executor,
    builtin_names,
    register_builtins,
)
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import RAW_ARGUMENTS_KEY, ToolDef

# ── Fakes ───────────────────────────────────────────────────────────────────


class Gate:
    """An approval gate that records each request and gives *answer*.

    *answer* is a decision, an exception to raise, or any other value to
    return. With *wait*, the gate first waits until it is set; ``asked`` is
    set once a request arrives.
    """

    def __init__(
        self,
        answer: object = None,
        *,
        delay_s: float = 0.0,
        wait: asyncio.Event | None = None,
    ) -> None:
        self.answer = ApprovalDecision(True) if answer is None else answer
        self.delay_s = delay_s
        self.wait = wait
        self.asked = asyncio.Event()
        self.requests: list[ExecutionRequest] = []

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        self.requests.append(request)
        self.asked.set()
        if self.wait is not None:
            await self.wait.wait()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer  # type: ignore[return-value]


class StandIns:
    """The marked built-ins' parameters, without their effects."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def shell_exec(self, command: str) -> str:
        self.calls.append(("shell_exec", {"command": command}))
        return f"ran {command}"

    async def python_exec(self, code: str) -> str:
        self.calls.append(("python_exec", {"code": code}))
        return "ran"

    async def file_write(self, path: str, content: str) -> str:
        self.calls.append(("file_write", {"path": path, "content": content}))
        return "written"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return path


def _executor(
    root: Path | None = None, stand_ins: StandIns | None = None
) -> tuple[Executor, Hippo]:
    """An executor with every built-in; *stand_ins* replace the marked ones."""
    hippo = Hippo()
    executor = Executor(
        hippo, Checker(), workspace_root=None if root is None else str(root)
    )
    register_builtins(executor, hippo)
    if stand_ins is not None:
        for name in APPROVAL_BUILTINS:
            executor.register(name, getattr(stand_ins, name), requires_approval=True)
    return executor, hippo


async def _run(
    executor: Executor,
    tool: str,
    arguments: dict[str, Any],
    gate: ApprovalGate | None,
    *,
    allowed: Collection[str] | None = None,
) -> ExecutionOutcome:
    return await executor.execute_structured(
        ExecutionRequest(tool_name=tool, arguments=arguments),
        allowed=allowed,
        approve=gate,
    )


# ── Which calls ask ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tool", "arguments", "allowed", "category", "error"),
    [
        pytest.param(
            "nope",
            {},
            None,
            ErrorCategory.TOOL_NOT_FOUND,
            "Tool not found in registry: nope",
            id="not_registered",
        ),
        pytest.param(
            "shell_exec",
            {"command": "ls"},
            frozenset({"file_read"}),
            ErrorCategory.TOOL_NOT_OFFERED,
            "Tool 'shell_exec' was not offered in this step.",
            id="not_offered",
        ),
        pytest.param(
            "shell_exec",
            {RAW_ARGUMENTS_KEY: "{not json"},
            None,
            ErrorCategory.INVALID_ARGUMENTS,
            "Arguments for shell_exec are not a JSON object: {not json",
            id="raw_arguments",
        ),
        pytest.param(
            "shell_exec",
            {"command": ["ls"]},
            None,
            ErrorCategory.INVALID_ARGUMENTS,
            "Arguments for shell_exec: 'command' must be of type string, not array",
            id="wrong_json_type",
        ),
        pytest.param(
            "shell_exec",
            {"comand": "ls"},
            None,
            ErrorCategory.INVALID_ARGUMENTS,
            "Arguments for shell_exec do not match its parameters: "
            "missing a required argument: 'command'",
            id="misspelled_parameter",
        ),
        pytest.param(
            "python_exec",
            {},
            None,
            ErrorCategory.INVALID_ARGUMENTS,
            "Arguments for python_exec do not match its parameters: "
            "missing a required argument: 'code'",
            id="missing_parameter",
        ),
        pytest.param(
            "shell_exec",
            {"command": "pwd", "cwd": "/"},
            None,
            ErrorCategory.INVALID_ARGUMENTS,
            "Arguments for shell_exec do not match its parameters: "
            "got an unexpected keyword argument 'cwd'",
            id="parameter_it_does_not_take",
        ),
        pytest.param(
            "shell_exec",
            {"command": "rm -rf /"},
            None,
            ErrorCategory.BLOCKED_BY_CHECKER,
            "Blocked: Blocked command: rm -rf /",
            id="checker",
        ),
        pytest.param(
            "file_write",
            {"path": "../outside.txt", "content": "x"},
            None,
            ErrorCategory.PATH_POLICY_VIOLATION,
            "Path '../outside.txt' is outside allowed workspace root",
            id="outside_the_root",
        ),
    ],
)
async def test_a_refused_call_never_asks(
    root: Path,
    tool: str,
    arguments: dict[str, Any],
    allowed: Collection[str] | None,
    category: ErrorCategory,
    error: str,
) -> None:
    stand_ins = StandIns()
    executor, _ = _executor(root, stand_ins)
    gate = Gate()

    outcome = await _run(executor, tool, arguments, gate, allowed=allowed)

    assert (outcome.error_category, outcome.error) == (category, error)
    assert outcome.status != ExecutionStatus.SUCCESS
    assert gate.requests == []
    assert stand_ins.calls == []


async def test_a_tool_without_a_callable_never_asks() -> None:
    executor, hippo = _executor()
    hippo.register_tool(
        ToolDef(name="orphan", description="No callable", parameters={})
    )
    gate = Gate()

    outcome = await _run(executor, "orphan", {}, gate)

    assert outcome.error_category == ErrorCategory.TOOL_NOT_FOUND
    assert outcome.error == "No callable registered for: orphan"
    assert gate.requests == []


async def test_an_unmarked_tool_never_asks(tmp_path: Path) -> None:
    executor, _ = _executor()
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    gate = Gate(ApprovalDecision(False))

    outcome = await _run(executor, "file_read", {"path": str(notes)}, gate)

    assert not executor.requires_approval("file_read")
    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, "hello")
    assert gate.requests == []


async def test_without_a_gate_a_marked_tool_runs(tmp_path: Path) -> None:
    executor, _ = _executor()
    target = tmp_path / "out.txt"

    outcome = await _run(
        executor, "file_write", {"path": str(target), "content": "hi"}, None
    )

    assert executor.requires_approval("file_write")
    assert outcome.status == ExecutionStatus.SUCCESS
    assert target.read_text() == "hi"


async def test_without_a_gate_arguments_are_not_matched_to_parameters() -> None:
    # The parameter check belongs to the approval step, so without a gate a
    # parameter the tool does not take fails as the call itself, as before.
    stand_ins = StandIns()
    executor, _ = _executor(stand_ins=stand_ins)

    outcome = await _run(executor, "shell_exec", {"command": "pwd", "cwd": "/"}, None)

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert "unexpected keyword argument 'cwd'" in outcome.error
    assert stand_ins.calls == []


async def test_an_unmarked_tool_keeps_its_runtime_error_under_a_gate() -> None:
    executor, hippo = _executor()
    received: list[str] = []

    async def note(text: str) -> str:
        received.append(text)
        return text

    hippo.register_tool(ToolDef(name="note", description="Keep a note", parameters={}))
    executor.register("note", note)
    gate = Gate()

    outcome = await _run(executor, "note", {"txt": "a"}, gate)

    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert (gate.requests, received) == ([], [])


# ── What the gate decides ───────────────────────────────────────────────────


async def test_an_allowed_call_runs_with_the_request_the_gate_saw() -> None:
    stand_ins = StandIns()
    executor, _ = _executor(stand_ins=stand_ins)
    gate = Gate(ApprovalDecision(True, origin="policy"))

    outcome = await _run(executor, "shell_exec", {"command": "ls -la"}, gate)

    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, "ran ls -la")
    [request] = gate.requests
    assert (request.tool_name, request.arguments) == (
        "shell_exec",
        {"command": "ls -la"},
    )
    assert stand_ins.calls == [("shell_exec", {"command": "ls -la"})]


@pytest.mark.parametrize(
    "decision",
    [
        ApprovalDecision(False),
        ApprovalDecision(False, "not now.", origin="user"),
        ApprovalDecision(False, "approvals are set to deny", origin="policy"),
        ApprovalDecision(False, "No answer within 300 s", origin="timeout"),
    ],
)
async def test_a_denied_call_does_not_run(decision: ApprovalDecision) -> None:
    stand_ins = StandIns()
    executor, _ = _executor(stand_ins=stand_ins)

    outcome = await _run(executor, "python_exec", {"code": "1"}, Gate(decision))

    assert outcome.status == ExecutionStatus.BLOCKED
    assert outcome.error_category == ErrorCategory.APPROVAL_DENIED
    assert outcome.error == denial_text(decision)
    assert (outcome.output, outcome.duration_ms) == ("", 0.0)
    assert stand_ins.calls == []


async def test_the_denial_text_reaches_the_outcome_literally() -> None:
    executor, _ = _executor(stand_ins=StandIns())

    outcome = await _run(
        executor,
        "shell_exec",
        {"command": "ls"},
        Gate(ApprovalDecision(False, "not now.")),
    )

    assert outcome.error == (
        "The user denied this call: not now. Do not run it again or reach the "
        "same result another way unless the user asks."
    )


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (
            RuntimeError("the policy is down"),
            "Approval failed, so the call did not run: RuntimeError: the policy is "
            "down",
        ),
        (
            ValueError(),
            "Approval failed, so the call did not run: ValueError",
        ),
        (
            "yes",
            "Approval failed, so the call did not run: TypeError: the approval "
            "gate returned str, not an ApprovalDecision",
        ),
    ],
    ids=["raises", "raises_without_a_message", "returns_no_decision"],
)
async def test_a_failing_gate_fails_closed(answer: object, error: str) -> None:
    stand_ins = StandIns()
    executor, _ = _executor(stand_ins=stand_ins)

    outcome = await _run(executor, "shell_exec", {"command": "ls"}, Gate(answer))

    assert outcome.status == ExecutionStatus.BLOCKED
    assert outcome.error_category == ErrorCategory.APPROVAL_DENIED
    assert outcome.error == error
    assert stand_ins.calls == []


async def test_a_cancellation_while_the_gate_decides_propagates() -> None:
    stand_ins = StandIns()
    executor, _ = _executor(stand_ins=stand_ins)
    gate = Gate(wait=asyncio.Event())

    running = asyncio.create_task(_run(executor, "shell_exec", {"command": "ls"}, gate))
    await asyncio.wait_for(gate.asked.wait(), 5.0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert stand_ins.calls == []


async def test_a_cancelled_error_from_the_gate_propagates() -> None:
    stand_ins = StandIns()
    executor, _ = _executor(stand_ins=stand_ins)

    with pytest.raises(asyncio.CancelledError):
        await _run(
            executor, "shell_exec", {"command": "ls"}, Gate(asyncio.CancelledError())
        )

    assert stand_ins.calls == []


async def test_approval_wait_is_outside_the_tool_timeout() -> None:
    executor, hippo = _executor()
    ran: list[str] = []

    async def quick(text: str = "") -> str:
        ran.append(text)
        return "done"

    hippo.register_tool(
        ToolDef(name="quick", description="Return at once", parameters={})
    )
    executor.register("quick", quick, timeout_ms=100, requires_approval=True)
    gate = Gate(delay_s=0.3)

    outcome = await _run(executor, "quick", {"text": "a"}, gate)

    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, "done")
    assert ran == ["a"]
    # The tool's clock started once the gate had answered.
    assert outcome.duration_ms < 100


async def test_a_tool_that_takes_any_keyword_asks_about_any_arguments() -> None:
    # Like an MCP tool's function, whose server checks the arguments.
    executor, hippo = _executor()

    async def anything(**arguments: Any) -> str:
        return "ok"

    hippo.register_tool(
        ToolDef(name="anything", description="Take any arguments", parameters={})
    )
    executor.register("anything", anything, requires_approval=True)
    gate = Gate()

    outcome = await _run(executor, "anything", {"whatever": 1}, gate)

    assert outcome.status == ExecutionStatus.SUCCESS
    assert len(gate.requests) == 1


async def test_a_signature_that_cannot_be_read_is_not_checked() -> None:
    executor, hippo = _executor()

    async def opaque(**arguments: Any) -> str:
        return "ok"

    # inspect.signature() raises TypeError for such a __signature__.
    opaque.__signature__ = "unreadable"  # type: ignore[attr-defined]
    hippo.register_tool(
        ToolDef(name="opaque", description="Hide its parameters", parameters={})
    )
    executor.register("opaque", opaque, requires_approval=True)
    gate = Gate()

    outcome = await _run(executor, "opaque", {"x": 1}, gate)

    assert outcome.status == ExecutionStatus.SUCCESS
    assert len(gate.requests) == 1


# ── Marks ───────────────────────────────────────────────────────────────────


def test_register_builtins_marks_the_approval_builtins() -> None:
    executor, _ = _executor()
    assert {"shell_exec", "python_exec", "file_write"} == APPROVAL_BUILTINS
    assert {
        name for name in builtin_names() if executor.requires_approval(name)
    } == APPROVAL_BUILTINS


def test_a_selection_marks_the_approval_builtins_it_registers() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    register_builtins(executor, hippo, names={"file_read", "shell_exec"})
    assert executor.requires_approval("shell_exec")
    assert not executor.requires_approval("file_read")
    assert not executor.requires_approval("file_write")  # not registered


def test_registering_again_without_the_mark_clears_it() -> None:
    executor, _ = _executor()

    async def shell(command: str) -> str:
        return command

    executor.register("shell_exec", shell)
    assert not executor.requires_approval("shell_exec")
    executor.register("shell_exec", shell, requires_approval=True)
    assert executor.requires_approval("shell_exec")
    executor.register("shell_exec", shell, timeout_ms=5_000)
    assert not executor.requires_approval("shell_exec")
    assert not executor.requires_approval("never_registered")
