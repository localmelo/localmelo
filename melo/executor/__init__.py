"""Executor: structured tool execution with safety boundaries."""

from __future__ import annotations

from collections.abc import Collection

from localmelo.melo.executor.builtins import BUILTINS, TOOL_EXIT_CODE, builtin_names
from localmelo.melo.executor.executor import Executor
from localmelo.melo.executor.models import (
    ArtifactMeta,
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    ToolOutput,
)
from localmelo.melo.executor.policy import WorkspacePolicy
from localmelo.melo.executor.preview import call_preview


def register_builtins(
    executor: Executor, hippo: object, names: Collection[str] | None = None
) -> None:
    """Register built-in tools with the executor and hippo.

    *names* selects the built-ins to register: ``None`` registers all of
    them, an empty collection none. An unknown name raises ``ValueError``
    before anything is registered.
    """
    selected = None if names is None else set(names)
    if selected is not None:
        known = builtin_names()
        unknown = selected - known
        if unknown:
            raise ValueError(
                f"Unknown built-in tools: {', '.join(sorted(unknown))} "
                f"(known: {', '.join(sorted(known))})"
            )
    for tool_def, fn in BUILTINS:
        if selected is None or tool_def.name in selected:
            hippo.register_tool(tool_def)  # type: ignore[attr-defined]
            executor.register(tool_def.name, fn)


__all__ = [
    "ArtifactMeta",
    "BUILTINS",
    "ErrorCategory",
    "ExecutionOutcome",
    "ExecutionRequest",
    "ExecutionStatus",
    "Executor",
    "TOOL_EXIT_CODE",
    "ToolOutput",
    "WorkspacePolicy",
    "builtin_names",
    "call_preview",
    "register_builtins",
]
