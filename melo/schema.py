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


# ── Data types ──


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    timestamp: float = field(default_factory=time.time)


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


@dataclass
class CheckResult:
    allowed: bool = True
    reason: str = ""
    modified_payload: Any = None
