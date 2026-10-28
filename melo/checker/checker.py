from __future__ import annotations

import json
from collections.abc import Collection

from localmelo.melo.checker.payloads import (
    ExecutorResultPayload,
    GatewayIngressPayload,
    MemoryCandidatePayload,
    MemoryWritePayload,
    SessionTransition,
    ToolResolutionResult,
    ValidationResult,
)
from localmelo.melo.checker.policy import ToolMode, exec_refusal, validated_mode
from localmelo.melo.checker.tiers import tier_of
from localmelo.melo.checker.validators import (
    MAX_OUTPUT_LEN,  # noqa: F401 – re-exported for tests
    validate_executor_result,
    validate_gateway_ingress,
    validate_memory_candidate,
    validate_memory_write,
    validate_session_transition,
    validate_tool_resolution,
)
from localmelo.melo.schema import CheckResult, Message, ToolCall, ToolDef, ToolTier


def _prompt_chars(msg: Message, *, count_thinking: bool = False) -> int:
    """Characters *msg* contributes to a planning prompt.

    With *count_thinking*, an assistant message's ``thinking`` counts too:
    a replay request sends it, so it must fit the cap. The default counts
    the content and tool calls only, as every request before replay did.
    """
    total = len(msg.content)
    if count_thinking and msg.role == "assistant":
        total += len(msg.thinking)
    for tc in msg.tool_calls:
        total += len(tc.tool_name)
        total += len(json.dumps(tc.arguments, ensure_ascii=False, default=str))
    return total


class Checker:
    def __init__(
        self,
        mode: ToolMode = "direct",
        *,
        builtins: Collection[str] | None = None,
    ) -> None:
        """A checker for a run of *mode*.

        The default is direct mode, which is what a checker built on its
        own has; :class:`Agent` always passes its ``AgentOptions.mode``.
        *builtins* is the explicit set the run's caller named, if any: an
        ``exec`` tool in it is opt-in and runs, while one that only the
        mode's default set would offer is refused in a mode with no
        approver (see :func:`~melo.checker.policy.exec_refusal`).
        """
        self._mode = validated_mode(mode)
        self._named_builtins = None if builtins is None else frozenset(builtins)

    # ── Permission tiers ──

    def tier(self, tool_def: ToolDef) -> ToolTier:
        """The permission tier of *tool_def* (see :mod:`melo.checker.tiers`)."""
        return tier_of(tool_def)

    # ── Core → Memory boundary ──

    async def pre_plan(
        self, messages: list[Message], *, count_thinking: bool = False
    ) -> CheckResult:
        # Tool-call arguments are replayed to the model with the assistant
        # message, so they count toward the prompt size as well. With
        # *count_thinking*, so does the reasoning a replay request sends.
        total = sum(_prompt_chars(m, count_thinking=count_thinking) for m in messages)
        if total > 100_000:
            return CheckResult(allowed=False, reason=f"Prompt too large: {total} chars")
        return CheckResult(allowed=True)

    async def post_plan(self, response: Message) -> CheckResult:
        seen_ids: set[str] = set()
        for tc in response.tool_calls:
            if not tc.tool_name:
                return CheckResult(allowed=False, reason="Empty tool name in response")
            if tc.id in seen_ids:
                return CheckResult(
                    allowed=False, reason=f"Duplicate tool call id in response: {tc.id}"
                )
            seen_ids.add(tc.id)
        return CheckResult(allowed=True)

    # ── Core → Executor boundary ──

    async def pre_execute(
        self, tool_call: ToolCall, tool_def: ToolDef | None
    ) -> CheckResult:
        if tool_def is None:
            return CheckResult(
                allowed=False, reason=f"Unknown tool: {tool_call.tool_name}"
            )

        tier = self.tier(tool_def)
        refusal = exec_refusal(
            tool_call.tool_name, tier, self._mode, self._named_builtins
        )
        if refusal is not None:
            return CheckResult(allowed=False, reason=refusal)
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

    def check_executor_result(self, result: ExecutorResultPayload) -> ValidationResult:
        return validate_executor_result(result)

    def check_memory_write(self, payload: MemoryWritePayload) -> ValidationResult:
        return validate_memory_write(payload)

    def check_memory_candidate(
        self, payload: MemoryCandidatePayload
    ) -> ValidationResult:
        return validate_memory_candidate(payload)
