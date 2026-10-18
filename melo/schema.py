from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ── Config ──

SHORT_TERM_MAX = 20
LONG_TERM_TOP_K = 5
TOOL_SEARCH_TOP_K = 3

MAX_AGENT_STEPS = 30
MIN_AGENT_STEPS = 3
STEPS_PER_ATTEMPT = 10
MAX_ATTEMPTS = 5

# Reflection budget. Only one reflection is carried into the next attempt's
# prompt. If its serialized form exceeds the char cap the agent forces a
# decompose action instead of continuing.
MAX_REFLECTIONS_IN_PROMPT = 1
MAX_REFLECTION_CHARS = 2000


# ── Data types ──


@dataclass(init=False)
class Message:
    """One chat message.

    An assistant message may carry several tool calls (``tool_calls``).
    A ``role="tool"`` message answers exactly one of them and names it in
    ``tool_call_id``. ``tool_call`` is kept as a compatibility accessor
    for single-call callers: it reads and writes the first entry of
    ``tool_calls``.
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None
    tool_call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    usage: dict[str, int] | None = None
    thinking: str = ""

    def __init__(
        self,
        role: str,
        content: str = "",
        tool_call: ToolCall | None = None,
        tool_result: ToolResult | None = None,
        timestamp: float | None = None,
        usage: dict[str, int] | None = None,
        thinking: str = "",
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str = "",
    ) -> None:
        # The positional order matches the previous dataclass signature so
        # existing ``Message(...)`` calls keep working unchanged.
        if tool_call is not None and tool_calls:
            raise TypeError("Message() takes tool_call or tool_calls, not both")
        self.role = role
        self.content = content
        if tool_calls:
            self.tool_calls = list(tool_calls)
        else:
            self.tool_calls = [tool_call] if tool_call is not None else []
        self.tool_result = tool_result
        self.tool_call_id = tool_call_id
        self.timestamp = time.time() if timestamp is None else timestamp
        self.usage = usage
        self.thinking = thinking

    @property
    def tool_call(self) -> ToolCall | None:
        """The first tool call, or ``None`` (compatibility accessor)."""
        return self.tool_calls[0] if self.tool_calls else None

    @tool_call.setter
    def tool_call(self, value: ToolCall | None) -> None:
        self.tool_calls = [value] if value is not None else []


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    semantic_tags: list[str] = field(default_factory=list)


def new_tool_call_id() -> str:
    """Return a fresh OpenAI-style tool call id (``call_`` + 24 hex chars)."""
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Pairs the call with its ``role="tool"`` result message. Providers keep
    # the id the backend assigned; when there is none (or it is empty) one
    # is generated. Equality ignores it so calls compare by name and args.
    id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_tool_call_id()


@dataclass
class ToolResult:
    tool_name: str
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class StepRecord:
    thought: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskRecord:
    query: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    steps: list[StepRecord] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    result: str = ""
    attempts_completed: int = 0


@dataclass
class CheckResult:
    allowed: bool = True
    reason: str = ""
    modified_payload: Any = None


# ── Reflection ──


@dataclass
class ReflectionEntry:
    """Structured reflection from one attempt."""

    attempt_id: int = 0
    summary: str = ""
    failed_hypotheses: list[str] = field(default_factory=list)
    disproven_actions: list[str] = field(default_factory=list)
    useful_evidence: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    recommended_avoids: list[str] = field(default_factory=list)
    next_promising_directions: list[str] = field(default_factory=list)
    failure_type: str = ""  # "stuck", "budget", "error", ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflectionDecision:
    """Structured decision from the reflection step."""

    progress_made: bool = False
    task_still_feasible: bool = True
    new_information_gained: bool = False
    next_step_is_concrete: bool = False
    next_step_is_novel: bool = False
    recommended_action: str = "stop"  # "continue" | "stop" | "decompose"
    rationale: str = ""
    handoff_summary: str = ""
    tried_memory: ReflectionEntry | None = None
    best_effort_result: str = ""
    # Active-learning acquisition signals
    estimated_info_gain: float = 0.0
    estimated_cost: float = 0.0
    repeat_risk: float = 0.0
    novelty: float = 0.0
    feasibility: float = 0.0
