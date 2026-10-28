"""Executor: structured tool execution with safety boundaries."""

from __future__ import annotations

from collections.abc import Collection

from localmelo.melo.checker.tiers import tier_of, tier_requires_approval
from localmelo.melo.executor.builtins import (
    APPROVAL_BUILTINS,
    BUILTINS,
    MAX_FILE_READ_BYTES,
    TOOL_EXIT_CODE,
    builtin_names,
    rooted_builtins,
)
from localmelo.melo.executor.executor import ApprovalGate, Executor
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
    executor: Executor,
    hippo: object,
    names: Collection[str] | None = None,
    *,
    workspace_root: str | None = None,
    tool_timeout_s: float | None = None,
) -> None:
    """Register built-in tools with the executor and hippo.

    *names* selects the built-ins to register: ``None`` registers all of
    them, an empty collection none. An unknown name raises ``ValueError``
    before anything is registered.

    The tools whose tier is not read are registered as requiring approval
    (the names in :data:`APPROVAL_BUILTINS`): a call of one asks the
    approval gate that :meth:`Executor.execute_structured` is given, if
    any, before it runs.

    Without *workspace_root*, the functions of :data:`BUILTINS` themselves
    are registered, and a *tool_timeout_s* raises ``ValueError``. Its
    ``python_exec`` is a child process too, in this process's directory,
    within the executor's default timeout. With a
    root, the tools that :func:`rooted_builtins` gives for it take their
    place, with the timeouts it gives them: ``file_read`` and ``file_write``
    refuse a symlink that takes the checked file's place and anything but a
    regular file, and ``file_read`` caps what it reads; ``shell_exec`` and
    ``python_exec`` run in the root, in a child process, within
    *tool_timeout_s* seconds when it is set, and the rooted ``shell_exec``
    starts the text of a non-zero exit with ``[exit code <n>]``. The
    executor fences the file tools' paths with its own root, so an executor
    built with the same ``workspace_root`` goes with them. ``shell_exec``
    and ``python_exec`` can still reach any path: approval is their control.
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
    if tool_timeout_s is not None and not workspace_root:
        raise ValueError("tool_timeout_s needs a workspace_root")
    rooted = (
        rooted_builtins(workspace_root, tool_timeout_s=tool_timeout_s)
        if workspace_root
        else {}
    )
    for tool_def, phase1a_fn in BUILTINS:
        if selected is None or tool_def.name in selected:
            fn, timeout_ms = rooted.get(tool_def.name, (phase1a_fn, None))
            tier = tier_of(tool_def)
            hippo.register_tool(tool_def)  # type: ignore[attr-defined]
            executor.register(
                tool_def.name,
                fn,
                timeout_ms=timeout_ms,
                requires_approval=tier_requires_approval(tier),
            )


__all__ = [
    "APPROVAL_BUILTINS",
    "ApprovalGate",
    "ArtifactMeta",
    "BUILTINS",
    "ErrorCategory",
    "ExecutionOutcome",
    "ExecutionRequest",
    "ExecutionStatus",
    "Executor",
    "MAX_FILE_READ_BYTES",
    "TOOL_EXIT_CODE",
    "ToolOutput",
    "WorkspacePolicy",
    "builtin_names",
    "call_preview",
    "register_builtins",
    "rooted_builtins",
]
