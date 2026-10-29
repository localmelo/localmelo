from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Awaitable, Callable, Collection, Coroutine, Mapping
from typing import TYPE_CHECKING, Any

from localmelo.melo.checker.checker import HARNESS_CALL, Checker
from localmelo.melo.contracts.approvals import (
    APPROVAL_FAILED,
    ApprovalDecision,
    denial_text,
)
from localmelo.melo.contracts.tool_capture import _ENTRY_MARKED, TOOL_ENTRY
from localmelo.melo.executor.builtins import builtin_names
from localmelo.melo.executor.models import (
    ArtifactMeta,
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    ToolOutput,
)
from localmelo.melo.executor.policy import WorkspacePolicy
from localmelo.melo.schema import RAW_ARGUMENTS_KEY, ToolCall, ToolDef

if TYPE_CHECKING:
    from localmelo.melo.memory.coordinator import Hippo

_DEFAULT_TIMEOUT_MS = 60_000

# The tools whose "path" argument the workspace policy resolves and fences.
_FILE_TOOLS = ("file_read", "file_write")

# A tool callable returns plain text or a ToolOutput (text + images).
ToolCallable = Callable[..., Coroutine[Any, Any, "str | ToolOutput"]]

# Asked before a call of a tool that requires approval runs (see
# Executor.execute_structured); its decision lets the call run or denies it.
ApprovalGate = Callable[[ExecutionRequest], Awaitable[ApprovalDecision]]


class Executor:
    def __init__(
        self,
        hippo: Hippo,
        checker: Checker,
        *,
        timeout_ms: float = _DEFAULT_TIMEOUT_MS,
        workspace_root: str | None = None,
    ) -> None:
        self.hippo = hippo
        self.checker = checker
        self._callables: dict[str, ToolCallable] = {}
        self._process_free: set[str] = set()
        self._timeouts_ms: dict[str, float] = {}
        # The tools registered as requiring approval.
        self._marked_for_approval: set[str] = set()
        # Tools only the harness may call (see register_harness_tool).
        self._harness_defs: dict[str, ToolDef] = {}
        self._default_timeout_ms = timeout_ms
        self._policy = WorkspacePolicy(workspace_root)

    @property
    def workspace_root(self) -> str | None:
        """The real path of the root the file tools are kept in, or ``None``."""
        return self._policy.root

    def register(
        self,
        name: str,
        fn: ToolCallable,
        *,
        timeout_ms: float | None = None,
        requires_approval: bool = False,
        process_free: bool = False,
    ) -> None:
        """Register *fn* as the callable for tool *name*.

        *timeout_ms* overrides the executor default for this tool (an
        ``ExecutionRequest.timeout_ms`` still takes precedence).
        *requires_approval* marks the tool: a call of it asks the approval
        gate of :meth:`execute_structured`, when there is one, before it
        runs. Registering a tool again resets both to what that call
        passes, so a call without *requires_approval* clears the mark.
        """
        self._callables[name] = fn
        if process_free:
            self._process_free.add(name)
        else:
            self._process_free.discard(name)
        if timeout_ms is not None:
            self._timeouts_ms[name] = timeout_ms
        else:
            self._timeouts_ms.pop(name, None)
        if requires_approval:
            self._marked_for_approval.add(name)
        else:
            self._marked_for_approval.discard(name)

    def register_harness_tool(
        self, tool_def: ToolDef, fn: ToolCallable, *, timeout_ms: float | None = None
    ) -> None:
        """Register *fn* for the harness's own calls, not for the model.

        The tool is not in the hippo, so the model is not offered it and its
        own call is ``TOOL_NOT_FOUND``; only :meth:`execute_structured` with
        ``harness=True`` runs it, by the same path as any other call.
        """
        self._harness_defs[tool_def.name] = tool_def
        self.register(tool_def.name, fn, timeout_ms=timeout_ms)

    def requires_approval(self, name: str) -> bool:
        """Whether tool *name* is registered as requiring approval."""
        return name in self._marked_for_approval

    # ── Public API ──

    async def execute_structured(
        self,
        request: ExecutionRequest,
        *,
        allowed: Collection[str] | None = None,
        approve: ApprovalGate | None = None,
        harness: bool = False,
    ) -> ExecutionOutcome:
        """Structured execution path with rich outcome metadata.

        A name that is no tool of the run is ``TOOL_NOT_FOUND``: a
        built-in that the run does not offer (``shell_exec`` under the
        default direct set) gets an error naming the run's available
        tools, while any other unknown name keeps
        ``Tool not found in registry: <name>``.

        *allowed* names the tools the caller offered the model in this
        step. A registered tool outside it is refused with
        ``TOOL_NOT_OFFERED`` before its arguments are looked at. ``None``
        skips that check.

        With a workspace root (the executor's, or the request's
        ``workspace_root``, which replaces it), the ``path`` of
        ``file_read`` and ``file_write`` is resolved against the root (see
        :meth:`WorkspacePolicy.resolve`). A path that is empty, missing or
        holds a NUL character is refused with ``INVALID_ARGUMENTS``, and
        one whose real path is outside the root with
        ``PATH_POLICY_VIOLATION``. Otherwise the call goes on with the real
        path in place of the given one, in a new request: the approval
        gate, the tool and the outcome's artifacts get the path that was
        checked, and the caller's request is not changed.

        *approve* is asked about each call of a tool registered as
        requiring approval (see :meth:`register`), after every check that
        refuses a call by itself and before the tool's timeout starts, so
        the time it takes to decide never counts against that timeout. It
        gets the request that would run, and is asked only when the
        arguments match the tool's parameters: a missing parameter, or one
        the tool does not take, is refused with ``INVALID_ARGUMENTS``
        instead. A denial is ``BLOCKED`` with ``APPROVAL_DENIED`` and the
        text of :func:`~localmelo.melo.contracts.approvals.denial_text`. A
        gate that raises an exception, or returns something other than an
        :class:`~localmelo.melo.contracts.approvals.ApprovalDecision`,
        fails closed: the call is refused the same way, with the text of
        :data:`~localmelo.melo.contracts.approvals.APPROVAL_FAILED`. A
        ``CancelledError`` propagates, and nothing runs. ``None`` asks no
        one: a tool that requires approval then runs as any other does.
        Once the gate allows a file tool's call under a root, its path is
        resolved again; when the result differs, as when a symlink took the
        file's place during the wait, the call is refused with
        ``PATH_POLICY_VIOLATION`` and does not run.

        *harness* marks a call of the harness itself, such as a brief's
        check: it also finds the tools of :meth:`register_harness_tool`, and
        the checker's refusal of an ``exec`` tool that the mode does not
        offer the model does not apply. Every other check still does.
        """
        tool_call = ToolCall(tool_name=request.tool_name, arguments=request.arguments)

        # 1. Authoritative registry resolution
        tool_def = self.hippo.get_tool(request.tool_name)
        if tool_def is None and harness:
            tool_def = self._harness_defs.get(request.tool_name)
        if tool_def is None:
            # A built-in that the run's mode leaves out (P13b): say what the
            # run does offer, so the model can pick a tool that exists. No
            # other hint (e.g. no "use python_exec for a shell"). An unknown
            # name that is no built-in keeps the plain registry message.
            error = f"Tool not found in registry: {request.tool_name}"
            if request.tool_name in builtin_names():
                available = sorted(tool.name for tool in self.hippo.list_tools())
                error = (
                    f"{request.tool_name} is not available in this run. "
                    f"Available tools: {', '.join(available) or 'none'}."
                )
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=error,
                error_category=ErrorCategory.TOOL_NOT_FOUND,
            )

        # 2. Allowlist: only the tools the caller offered may run.
        if allowed is not None and request.tool_name not in allowed:
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=f"Tool '{request.tool_name}' was not offered in this step.",
                error_category=ErrorCategory.TOOL_NOT_OFFERED,
            )

        # 3. Arguments the provider could not parse into a JSON object:
        #    report them to the model and run nothing.
        if RAW_ARGUMENTS_KEY in request.arguments:
            raw = str(request.arguments[RAW_ARGUMENTS_KEY])
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=(
                    f"Arguments for {request.tool_name} are not a JSON object: "
                    f"{raw[:200]}"
                ),
                error_category=ErrorCategory.INVALID_ARGUMENTS,
            )

        # 4. A built-in tool's arguments must have the JSON types its schema
        #    declares. The checks below read them outside the guarded call,
        #    and file_write would empty the file before failing on content
        #    that is not text. Provided tools' servers check their own.
        if request.tool_name in builtin_names():
            problem = _wrong_type(request.arguments, tool_def.parameters)
            if problem:
                return ExecutionOutcome(
                    tool_name=request.tool_name,
                    status=ExecutionStatus.ERROR,
                    error=f"Arguments for {request.tool_name}: {problem}",
                    error_category=ErrorCategory.INVALID_ARGUMENTS,
                )

        # 5. Pre-execution safety check
        slot = HARNESS_CALL.set(harness)
        try:
            check = await self.checker.pre_execute(tool_call, tool_def)
        finally:
            HARNESS_CALL.reset(slot)
        if not check.allowed:
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.BLOCKED,
                error=f"Blocked: {check.reason}",
                error_category=ErrorCategory.BLOCKED_BY_CHECKER,
            )

        # 6. Resolve callable
        fn = self._callables.get(request.tool_name)
        if fn is None:
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=f"No callable registered for: {request.tool_name}",
                error_category=ErrorCategory.TOOL_NOT_FOUND,
            )

        # 7. Workspace path policy for the file tools: the path is resolved
        #    against the root, and the call goes on with the real path, so
        #    the tool opens the file that was checked (and the approval
        #    policy sees it).
        policy: WorkspacePolicy | None = None
        resolved: str | None = None
        if request.tool_name in _FILE_TOOLS:
            policy = (
                WorkspacePolicy(request.workspace_root)
                if request.workspace_root
                else self._policy
            )
            if policy.root is None:
                policy = None
            else:
                path = request.arguments.get("path")
                try:
                    resolved = policy.resolve(path)
                except ValueError:
                    return ExecutionOutcome(
                        tool_name=request.tool_name,
                        status=ExecutionStatus.ERROR,
                        error=(
                            f"Argument 'path' for {request.tool_name} must be a "
                            "non-empty string without NUL characters."
                        ),
                        error_category=ErrorCategory.INVALID_ARGUMENTS,
                    )
                if resolved is None:
                    return ExecutionOutcome(
                        tool_name=request.tool_name,
                        status=ExecutionStatus.ERROR,
                        error=f"Path '{path}' is outside allowed workspace root",
                        error_category=ErrorCategory.PATH_POLICY_VIOLATION,
                    )
                # A new request: the caller's arguments stay as they were.
                request = dataclasses.replace(
                    request, arguments={**request.arguments, "path": resolved}
                )

        # 8. Approval, for a tool that requires it: asked only about a call
        #    that every check above let through, and before the tool's
        #    timeout starts.
        asked = False
        if approve is not None and self.requires_approval(request.tool_name):
            mismatch = _parameter_mismatch(fn, request.arguments)
            if mismatch:
                return ExecutionOutcome(
                    tool_name=request.tool_name,
                    status=ExecutionStatus.ERROR,
                    error=(
                        f"Arguments for {request.tool_name} do not match its "
                        f"parameters: {mismatch}"
                    ),
                    error_category=ErrorCategory.INVALID_ARGUMENTS,
                )
            # A CancelledError propagates: the call does not run.
            try:
                decision: object = await approve(request)
            except Exception as exc:
                return _approval_failed(request.tool_name, exc)
            if not isinstance(decision, ApprovalDecision):
                return _approval_failed(
                    request.tool_name,
                    TypeError(
                        f"the approval gate returned {type(decision).__name__}, "
                        "not an ApprovalDecision"
                    ),
                )
            if not decision.allow:
                return ExecutionOutcome(
                    tool_name=request.tool_name,
                    status=ExecutionStatus.BLOCKED,
                    error=denial_text(decision),
                    error_category=ErrorCategory.APPROVAL_DENIED,
                )
            asked = True

        # 9. Re-check: an approved file path is resolved again, since a
        #    person may take a while to answer. A path that resolves
        #    elsewhere now, as when a symlink took the file's place, does
        #    not run. This narrows the window between check and open.
        if (
            asked
            and policy is not None
            and resolved is not None
            and policy.resolve(resolved) != resolved
        ):
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=(
                    f"Path '{resolved}' changed while waiting for approval; "
                    "the call did not run."
                ),
                error_category=ErrorCategory.PATH_POLICY_VIOLATION,
            )

        # 10. Execute with timeout
        timeout_ms = (
            request.timeout_ms
            or self._timeouts_ms.get(request.tool_name)
            or self._default_timeout_ms
        )
        timeout_s = timeout_ms / 1000
        start = time.perf_counter()
        deadline = asyncio.timeout(timeout_s)
        try:
            async with deadline:
                capture = TOOL_ENTRY.get()
                # Only a captured task execution is checked: direct mode and
                # chat turns call the tool exactly as they did before.
                if capture is not None:
                    mismatch = _parameter_mismatch(fn, request.arguments)
                    if mismatch:
                        return ExecutionOutcome(
                            tool_name=request.tool_name,
                            status=ExecutionStatus.ERROR,
                            error=(
                                f"Arguments for {request.tool_name} do not match "
                                f"its parameters: {mismatch}"
                            ),
                            error_category=ErrorCategory.INVALID_ARGUMENTS,
                        )
                marked = False
                if capture is not None and request.tool_name not in self._process_free:
                    await capture()
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise asyncio.CancelledError()
                    marked = True
                token = _ENTRY_MARKED.set(marked)
                try:
                    output = await fn(**request.arguments)
                finally:
                    _ENTRY_MARKED.reset(token)
        except TimeoutError as exc:
            duration = (time.perf_counter() - start) * 1000
            if deadline.expired():
                error = f"Execution timed out after {timeout_s:.1f}s"
            else:
                # A limit of the tool's own, such as shell_exec's command
                # timeout: the tool's message names it.
                error = str(exc) or f"{request.tool_name} timed out"
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.TIMEOUT,
                error=error,
                error_category=ErrorCategory.TIMEOUT,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=str(e),
                error_category=ErrorCategory.RUNTIME_ERROR,
                duration_ms=duration,
            )

        duration = (time.perf_counter() - start) * 1000

        # 11. Normalize rich output (text + images)
        if isinstance(output, ToolOutput):
            if output.is_error:
                return ExecutionOutcome(
                    tool_name=request.tool_name,
                    status=ExecutionStatus.ERROR,
                    error=output.text or "Tool reported an error",
                    error_category=ErrorCategory.TOOL_ERROR,
                    duration_ms=duration,
                    images=list(output.images),
                )
            text, images = output.text, list(output.images)
        else:
            text, images = (output if isinstance(output, str) else str(output)), []

        # 12. Build outcome with artifact metadata
        artifacts = self._collect_artifacts(request, text)

        return ExecutionOutcome(
            tool_name=request.tool_name,
            status=ExecutionStatus.SUCCESS,
            output=text,
            duration_ms=duration,
            artifacts=artifacts,
            images=images,
        )

    # ── Internals ──

    def _collect_artifacts(
        self, request: ExecutionRequest, output: str
    ) -> list[ArtifactMeta]:
        artifacts: list[ArtifactMeta] = []
        if request.tool_name == "file_write":
            path = request.arguments.get("path", "")
            content = request.arguments.get("content", "")
            artifacts.append(
                ArtifactMeta(
                    kind="file",
                    path=path,
                    size_bytes=len(content.encode()),
                    description="Written file",
                )
            )
        elif request.tool_name == "file_read":
            path = request.arguments.get("path", "")
            artifacts.append(
                ArtifactMeta(
                    kind="file",
                    path=path,
                    size_bytes=len(output.encode()),
                    description="Read file",
                )
            )
        return artifacts


def _parameter_mismatch(fn: ToolCallable, arguments: Mapping[str, Any]) -> str:
    """Why *fn* cannot be called with *arguments* as keywords, or ``""``.

    That is a missing parameter, or one *fn* does not take. A callable whose
    signature :mod:`inspect` cannot read is not checked, and one that takes
    ``**kwargs`` accepts any name.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    try:
        signature.bind(**arguments)
    except TypeError as exc:
        return str(exc)
    return ""


def _approval_failed(tool_name: str, exc: BaseException) -> ExecutionOutcome:
    """The outcome of a call whose approval gate failed: the call did not run."""
    message = str(exc)
    error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return ExecutionOutcome(
        tool_name=tool_name,
        status=ExecutionStatus.BLOCKED,
        error=APPROVAL_FAILED.format(error=error),
        error_category=ErrorCategory.APPROVAL_DENIED,
    )


def _json_type(value: Any) -> str:
    """Name the JSON type of a value parsed from JSON."""
    if value is None:
        return "null"
    # bool before int: True is an int in Python but not a JSON integer.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _wrong_type(arguments: dict[str, Any], parameters: dict[str, Any]) -> str:
    """Name the first argument whose JSON type its schema does not declare.

    Only top-level properties with a single ``type`` are checked; arguments
    the schema does not declare are left to the tool. Returns ``""`` when
    nothing is wrong.
    """
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return ""
    for name, value in arguments.items():
        schema = properties.get(name)
        expected = schema.get("type") if isinstance(schema, dict) else None
        if not isinstance(expected, str):
            continue
        actual = _json_type(value)
        if actual == expected or (expected == "number" and actual == "integer"):
            continue
        return f"'{name}' must be of type {expected}, not {actual}"
    return ""
