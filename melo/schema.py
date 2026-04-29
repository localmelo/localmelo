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


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    timestamp: float = field(default_factory=time.time)
    usage: dict[str, int] | None = None
    thinking: str = ""


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    semantic_tags: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
