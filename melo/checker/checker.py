from __future__ import annotations

from localmelo.melo.checker.payloads import (
    ExecutorRequest,
    ExecutorResultPayload,
    GatewayIngressPayload,
    MemoryWritePayload,
    SessionTransition,
    ToolResolutionResult,
    ValidationResult,
)
from localmelo.melo.checker.validators import (
    _BLOCKED_RE,
    BLOCKED_COMMANDS,  # noqa: F401 – re-exported for tests
    MAX_OUTPUT_LEN,  # noqa: F401 – re-exported for tests
    validate_executor_request,
    validate_executor_result,
    validate_gateway_ingress,
    validate_memory_write,
    validate_session_transition,
    validate_tool_resolution,
)
from localmelo.melo.schema import CheckResult, Message, ToolCall, ToolDef


class Checker:
    def __init__(self) -> None:
        self._blocked = _BLOCKED_RE

    # ── Core → Memory boundary ──

    async def pre_plan(self, messages: list[Message]) -> CheckResult:
        total = sum(len(m.content) for m in messages)
        if total > 100_000:
            return CheckResult(allowed=False, reason=f"Prompt too large: {total} chars")
        return CheckResult(allowed=True)

    async def post_plan(self, response: Message) -> CheckResult:
        if response.tool_call and not response.tool_call.tool_name:
            return CheckResult(allowed=False, reason="Empty tool name in response")
        return CheckResult(allowed=True)

    # ── Core → Executor boundary ──

    async def pre_execute(
        self, tool_call: ToolCall, tool_def: ToolDef | None
    ) -> CheckResult:
        if tool_def is None:
            return CheckResult(
                allowed=False, reason=f"Unknown tool: {tool_call.tool_name}"
            )

        # check dangerous shell commands
        if tool_call.tool_name == "shell_exec":
            cmd = tool_call.arguments.get("command", "")
            for pattern in self._blocked:
                if pattern.search(cmd):
                    return CheckResult(
                        allowed=False, reason=f"Blocked command: {cmd[:100]}"
                    )

        return CheckResult(allowed=True)

    # ── Structured boundary validators ──

    def check_gateway_ingress(self, payload: GatewayIngressPayload) -> ValidationResult:
        return validate_gateway_ingress(payload)

    def check_session_transition(
        self, transition: SessionTransition
    ) -> ValidationResult:
        return validate_session_transition(transition)

    def check_tool_resolution(self, result: ToolResolutionResult) -> ValidationResult:
        return validate_tool_resolution(result)

    def check_executor_request(self, request: ExecutorRequest) -> ValidationResult:
        return validate_executor_request(request)

    def check_executor_result(self, result: ExecutorResultPayload) -> ValidationResult:
        return validate_executor_result(result)

    def check_memory_write(self, payload: MemoryWritePayload) -> ValidationResult:
        return validate_memory_write(payload)
