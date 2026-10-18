from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from localmelo.melo.checker.checker import Checker
from localmelo.melo.executor.models import (
    ArtifactMeta,
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    ToolOutput,
)
from localmelo.melo.executor.policy import WorkspacePolicy
from localmelo.melo.schema import ToolCall

if TYPE_CHECKING:
    from localmelo.melo.memory.coordinator import Hippo

_DEFAULT_TIMEOUT_MS = 60_000

# A tool callable returns plain text or a ToolOutput (text + images).
ToolCallable = Callable[..., Coroutine[Any, Any, "str | ToolOutput"]]


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
        self._timeouts_ms: dict[str, float] = {}
        self._default_timeout_ms = timeout_ms
        self._policy = WorkspacePolicy(workspace_root)

    def register(
        self, name: str, fn: ToolCallable, *, timeout_ms: float | None = None
    ) -> None:
        """Register *fn* as the callable for tool *name*.

        *timeout_ms* overrides the executor default for this tool (an
        ``ExecutionRequest.timeout_ms`` still takes precedence).
        """
        self._callables[name] = fn
        if timeout_ms is not None:
            self._timeouts_ms[name] = timeout_ms
        else:
            self._timeouts_ms.pop(name, None)

    # ── Public API ──

    async def execute_structured(self, request: ExecutionRequest) -> ExecutionOutcome:
        """Structured execution path with rich outcome metadata."""
        tool_call = ToolCall(tool_name=request.tool_name, arguments=request.arguments)

        # 1. Authoritative registry resolution
        tool_def = self.hippo.get_tool(request.tool_name)
        if tool_def is None:
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=f"Tool not found in registry: {request.tool_name}",
                error_category=ErrorCategory.TOOL_NOT_FOUND,
            )

        # 2. Pre-execution safety check
        check = await self.checker.pre_execute(tool_call, tool_def)
        if not check.allowed:
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.BLOCKED,
                error=f"Blocked: {check.reason}",
                error_category=ErrorCategory.BLOCKED_BY_CHECKER,
            )

        # 3. Resolve callable
        fn = self._callables.get(request.tool_name)
        if fn is None:
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.ERROR,
                error=f"No callable registered for: {request.tool_name}",
                error_category=ErrorCategory.TOOL_NOT_FOUND,
            )

        # 4. Workspace path policy for file tools
        if request.tool_name in ("file_read", "file_write"):
            effective_root = request.workspace_root or self._policy.root
            if effective_root:
                policy = WorkspacePolicy(effective_root)
                path = request.arguments.get("path", "")
                if not policy.check_path(path):
                    return ExecutionOutcome(
                        tool_name=request.tool_name,
                        status=ExecutionStatus.ERROR,
                        error=f"Path '{path}' is outside allowed workspace root",
                        error_category=ErrorCategory.PATH_POLICY_VIOLATION,
                    )

        # 5. Execute with timeout
        timeout_ms = (
            request.timeout_ms
            or self._timeouts_ms.get(request.tool_name)
            or self._default_timeout_ms
        )
        timeout_s = timeout_ms / 1000
        start = time.perf_counter()
        try:
            output = await asyncio.wait_for(fn(**request.arguments), timeout=timeout_s)
        except TimeoutError:
            duration = (time.perf_counter() - start) * 1000
            return ExecutionOutcome(
                tool_name=request.tool_name,
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {timeout_s:.1f}s",
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

        # 6. Normalize rich output (text + images)
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

        # 7. Build outcome with artifact metadata
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
