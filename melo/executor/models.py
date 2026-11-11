from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from localmelo.melo.schema import ImagePart, ToolResult

if TYPE_CHECKING:
    from localmelo.melo.contracts.folders import OutsideAccess


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
    # Arguments that were not a JSON object, a built-in tool's argument of a
    # JSON type its schema does not declare, or, before the approval policy
    # is asked, arguments that do not match the tool's parameters or that
    # the tool's own check refuses (a file_edit whose text is not in the file).
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_NOT_OFFERED = "tool_not_offered"  # registered, but not offered this step
    # The approval policy denied the call, or failed while it decided.
    APPROVAL_DENIED = "approval_denied"


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
    text then becomes the outcome's error. ``no_more_calls=True`` says that the
    tool takes no more calls in this run (a note refused for the last time): the
    agent stops offering it, and a later call to it does not run. Only the
    tool's own code sets it, never a reader of its text.
    """

    text: str = ""
    images: list[ImagePart] = field(default_factory=list)
    is_error: bool = False
    no_more_calls: bool = False


@dataclass
class ExecutionRequest:
    """One call to run.

    ``outside`` is set by the executor alone, on the request it hands the
    approval gate, for a file tool's path outside the workspace (#405).
    ``ask``: the gate is asked about this call whatever its tool, as about one
    that requires approval (a call whose effect is unknown, #571).
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    timeout_ms: float | None = None
    workspace_root: str | None = None
    outside: OutsideAccess | None = None
    ask: bool = False


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
    no_more_calls: bool = False

    def to_tool_result(self) -> ToolResult:
        return ToolResult(
            tool_name=self.tool_name,
            output=self.output,
            error=self.error,
            duration_ms=self.duration_ms,
            images=list(self.images),
            no_more_calls=self.no_more_calls,
        )
