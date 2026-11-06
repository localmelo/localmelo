from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING

from localmelo.melo.checker.commits import commit_credit_refusal
from localmelo.melo.checker.payloads import (
    ActionReviewPayload,
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

if TYPE_CHECKING:
    from localmelo.melo.contracts.review import ActionReview


# The prompt cap without a context window: the characters a planning or
# reflection request is fitted to before it is sent (``Chat._request``), and
# past which ``Checker.pre_plan`` refuses it.
PROMPT_CAP_CHARS = 100_000

# The characters one image with pixels counts for in the prompt cap: a
# 1280x1024 screenshot is about 1,300 tokens on the local vision models, at
# about 3.45 characters per token. A reference (no pixels) counts nothing.
IMAGE_PROMPT_CHARS = 4_500


def _prompt_chars(msg: Message, *, count_thinking: bool = False) -> int:
    """Characters *msg* contributes to a request, for the prompt cap.

    With *count_thinking*, an assistant message's ``thinking`` counts too:
    a request that sends it (the provider has a reasoning field) must fit
    the cap with it. Tool schemas never count. Each image the message sends
    (one with data) counts :data:`IMAGE_PROMPT_CHARS`.
    """
    total = len(msg.content)
    total += IMAGE_PROMPT_CHARS * sum(1 for image in msg.images if image.data)
    if count_thinking and msg.role == "assistant":
        total += len(msg.thinking)
    for tc in msg.tool_calls:
        total += len(tc.tool_name)
        total += len(json.dumps(tc.arguments, ensure_ascii=False, default=str))
    return total


def prompt_chars(messages: Sequence[Message], *, count_thinking: bool = False) -> int:
    """Characters *messages* count toward :data:`PROMPT_CAP_CHARS`.

    The system prompt does not count: the cap bounds what a run adds (task,
    history, recall, reflections), and the fixed prompt never grows.
    """
    return sum(
        _prompt_chars(m, count_thinking=count_thinking)
        for m in messages
        if m.role != "system"
    )


# True while the executor checks a call of the harness itself (see
# Executor.execute_structured's ``harness``).
HARNESS_CALL: ContextVar[bool] = ContextVar("localmelo_harness_call", default=False)


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
        ``exec`` tool in it is opt-in and runs in direct and ``python``
        mode (gateway still refuses it), while one that only the mode's
        default set would offer is refused in a mode with no approver (see
        :func:`~melo.checker.policy.exec_refusal`).
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
        """The check after the fit: refuse a request over :data:`PROMPT_CAP_CHARS`.

        With no context window, ``Chat`` fits each planning and reflection
        request to the cap (``_context_fit.fit_request``) and then asks this;
        it refuses only a request whose pinned part alone is over the cap.
        Tool-call arguments count, as the model reads them with the message;
        with *count_thinking*, so does the reasoning the request sends.
        """
        total = prompt_chars(messages, count_thinking=count_thinking)
        if total > PROMPT_CAP_CHARS:
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
        """Allow or refuse a call before it runs.

        A call that :data:`HARNESS_CALL` marks, one the harness makes itself
        (a brief's check), skips the mode's refusal of an ``exec`` tool: that
        refusal keeps the tool from the model, and the person who wrote the
        check chose it.
        """
        if tool_def is None:
            return CheckResult(
                allowed=False, reason=f"Unknown tool: {tool_call.tool_name}"
            )

        tier = self.tier(tool_def)
        refusal = None
        if not HARNESS_CALL.get():
            refusal = exec_refusal(
                tool_call.tool_name, tier, self._mode, self._named_builtins
            )
        if refusal is None and not HARNESS_CALL.get():
            refusal = commit_credit_refusal(tool_call.tool_name, tool_call.arguments)
        if refusal is not None:
            return CheckResult(allowed=False, reason=refusal)
        return CheckResult(allowed=True)

    def review_action(self, payload: ActionReviewPayload) -> ActionReview:
        """Classify a call before an approval policy decides (#279).

        A session grant covers only a routine call; see
        :mod:`localmelo.melo.contracts.review`. Imported here: the rules use
        the contracts package, which imports the executor, which imports this.
        """
        from localmelo.melo.checker.review import review_action

        return review_action(payload)

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
