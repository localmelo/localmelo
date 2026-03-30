"""Boundary validators for the Checker."""

from __future__ import annotations

import re

from localmelo.melo.checker.payloads import (
    ExecutorRequest,
    ExecutorResultPayload,
    GatewayIngressPayload,
    MemoryWritePayload,
    SessionTransition,
    ToolResolutionResult,
    ValidationResult,
)

# ── Constants ──

MAX_QUERY_LEN = 100_000
MAX_OUTPUT_LEN = 50_000
MAX_MEMORY_TEXT_LEN = 50_000
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

VALID_SESSION_STATES = {"idle", "running", "closed"}
LEGAL_SESSION_TRANSITIONS: set[tuple[str, str]] = {
    ("idle", "running"),
    ("running", "idle"),
    ("running", "closed"),
    ("idle", "closed"),
    ("running", "failed"),
}

VALID_MEMORY_ROLES = {"user", "assistant", "system", "tool"}

BLOCKED_COMMANDS = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{.*\}\s*;",
    r"\bshutdown\b",
    r"\breboot\b",
]
_BLOCKED_RE = [re.compile(p) for p in BLOCKED_COMMANDS]


# ── Gateway ingress ──


def validate_gateway_ingress(payload: GatewayIngressPayload) -> ValidationResult:
    if not isinstance(payload.query, str) or not payload.query.strip():
        return ValidationResult(
            allowed=False, reason="Query must be a non-empty string"
        )

    if len(payload.query) > MAX_QUERY_LEN:
        return ValidationResult(
            allowed=False,
            reason=f"Query too large: {len(payload.query)} chars (max {MAX_QUERY_LEN})",
        )

    if payload.session_id is not None and not SESSION_ID_PATTERN.match(
        payload.session_id
    ):
        return ValidationResult(
            allowed=False,
            reason=f"Invalid session_id format: {payload.session_id!r}",
        )

    # Sanitize: strip leading/trailing whitespace from query
    sanitized = GatewayIngressPayload(
        query=payload.query.strip(),
        session_id=payload.session_id,
    )
    return ValidationResult(allowed=True, sanitized_payload=sanitized)


# ── Session state transitions ──


def validate_session_transition(transition: SessionTransition) -> ValidationResult:
    if transition.from_status not in VALID_SESSION_STATES:
        return ValidationResult(
            allowed=False,
            reason=f"Unknown source state: {transition.from_status!r}",
        )

    if transition.to_status not in VALID_SESSION_STATES:
        return ValidationResult(
            allowed=False,
            reason=f"Unknown target state: {transition.to_status!r}",
        )

    pair = (transition.from_status, transition.to_status)
    if pair not in LEGAL_SESSION_TRANSITIONS:
        return ValidationResult(
            allowed=False,
            reason=f"Illegal transition: {transition.from_status} -> {transition.to_status}",
        )

    return ValidationResult(allowed=True)


# ── Tool resolution ──


def validate_tool_resolution(result: ToolResolutionResult) -> ValidationResult:
    if not result.query.strip():
        return ValidationResult(
            allowed=False, reason="Tool resolution query must not be empty"
        )

    for name in result.resolved_tool_names:
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                allowed=False,
                reason=f"Invalid resolved tool name: {name!r}",
            )

    seen: set[str] = set()
    for name in result.resolved_tool_names:
        if name in seen:
            return ValidationResult(
                allowed=False,
                reason=f"Duplicate resolved tool: {name!r}",
            )
        seen.add(name)

    return ValidationResult(allowed=True)


# ── Executor request ──


def validate_executor_request(request: ExecutorRequest) -> ValidationResult:
    if not request.tool_name or not request.tool_name.strip():
        return ValidationResult(
            allowed=False, reason="Executor request must specify a tool_name"
        )

    if request.tool_def_name is None:
        return ValidationResult(
            allowed=False,
            reason=f"Unknown tool: {request.tool_name} (no matching tool_def)",
        )

    if request.tool_name != request.tool_def_name:
        return ValidationResult(
            allowed=False,
            reason=f"Tool name mismatch: call={request.tool_name!r} vs def={request.tool_def_name!r}",
        )

    # Block dangerous shell commands
    if request.tool_name == "shell_exec":
        cmd = request.arguments.get("command", "")
        for pattern in _BLOCKED_RE:
            if pattern.search(cmd):
                return ValidationResult(
                    allowed=False,
                    reason=f"Blocked command: {cmd[:100]}",
                )

    return ValidationResult(allowed=True)


# ── Executor result ──


def validate_executor_result(result: ExecutorResultPayload) -> ValidationResult:
    if result.duration_ms < 0:
        return ValidationResult(
            allowed=False,
            reason=f"Negative duration: {result.duration_ms}ms",
        )

    if not result.tool_name or not result.tool_name.strip():
        return ValidationResult(
            allowed=False, reason="Executor result must have a tool_name"
        )

    # Truncate oversized output
    if len(result.output) > MAX_OUTPUT_LEN:
        truncated = result.output[:MAX_OUTPUT_LEN] + "\n... [truncated]"
        sanitized = ExecutorResultPayload(
            tool_name=result.tool_name,
            output=truncated,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        return ValidationResult(
            allowed=True,
            reason="Output truncated",
            sanitized_payload=sanitized,
        )

    return ValidationResult(allowed=True)


# ── Memory write ──


def validate_memory_write(payload: MemoryWritePayload) -> ValidationResult:
    if not payload.text or not payload.text.strip():
        return ValidationResult(
            allowed=False, reason="Memory write text must not be empty"
        )

    if len(payload.text) > MAX_MEMORY_TEXT_LEN:
        return ValidationResult(
            allowed=False,
            reason=f"Memory write too large: {len(payload.text)} chars (max {MAX_MEMORY_TEXT_LEN})",
        )

    if payload.role and payload.role not in VALID_MEMORY_ROLES:
        return ValidationResult(
            allowed=False,
            reason=f"Invalid memory role: {payload.role!r}",
        )

    return ValidationResult(allowed=True)
