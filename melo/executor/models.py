from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from localmelo.melo.schema import ImagePart, ToolResult


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class ErrorCategory(StrEnum):
    NONE = "none"
    TOOL_NOT_FOUND = "tool_not_found"
    BLOCKED_BY_CHECKER = "blocked_by_checker"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    PATH_POLICY_VIOLATION = "path_policy_violation"
    TOOL_ERROR = "tool_error"  # the tool itself reported failure (MCP isError)


@dataclass
class ArtifactMeta:
    kind: str  # "file", "stdout", "stderr"
    path: str = ""
    size_bytes: int = 0
    description: str = ""


@dataclass
class LogMeta:
    level: str = "info"  # "info", "warning", "error"
    message: str = ""
    detail: str = ""


@dataclass
class ToolOutput:
    """Rich return value of a tool callable: text plus optional images.

    Callables may still return a plain ``str``. ``is_error=True`` marks a
    failure the tool itself reported (an MCP result with ``isError``); the
    text then becomes the outcome's error.
    """

    text: str = ""
    images: list[ImagePart] = field(default_factory=list)
    is_error: bool = False


@dataclass
class ExecutionRequest:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    timeout_ms: float | None = None
    workspace_root: str | None = None


@dataclass
class ExecutionOutcome:
    tool_name: str
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    output: str = ""
    error: str = ""
    error_category: ErrorCategory = ErrorCategory.NONE
    duration_ms: float = 0.0
    artifacts: list[ArtifactMeta] = field(default_factory=list)
    logs: list[LogMeta] = field(default_factory=list)
    images: list[ImagePart] = field(default_factory=list)

    def to_tool_result(self) -> ToolResult:
        return ToolResult(
            tool_name=self.tool_name,
            output=self.output,
            error=self.error,
            duration_ms=self.duration_ms,
            images=list(self.images),
        )
