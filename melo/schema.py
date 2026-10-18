from __future__ import annotations

import base64
import binascii
import hashlib
import time
import uuid
from collections.abc import Sequence
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

# Image budget. Only the most recent images keep their pixels in a prompt;
# older ones are replaced by their text reference (see ImagePart.ref), the
# usual practice for screenshot-driven computer use.
MAX_IMAGES_IN_PROMPT = 3


# ── Data types ──


def decode_base64(data: str) -> bytes | None:
    """Strictly decode base64 *data* (whitespace ignored); ``None`` if invalid."""
    try:
        return base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError):
        return None


@dataclass(frozen=True)
class ImagePart:
    """An image attached to a message or a tool result.

    ``data`` is the base64 payload without a ``data:`` prefix. ``sha256``
    and ``size_bytes`` describe the decoded bytes and are computed once,
    on construction. A reference copy (:meth:`as_reference`) drops
    ``data`` and is what history and long-term memory keep. Invalid
    base64 yields ``size_bytes == 0``, which the checker rejects.
    """

    mime_type: str
    data: str = field(default="", repr=False)
    sha256: str = ""
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.data and not self.sha256:
            raw = decode_base64(self.data)
            digest_input = raw if raw is not None else self.data.encode()
            object.__setattr__(self, "sha256", hashlib.sha256(digest_input).hexdigest())
            object.__setattr__(self, "size_bytes", len(raw) if raw is not None else 0)

    @classmethod
    def from_bytes(cls, raw: bytes, mime_type: str) -> ImagePart:
        return cls(
            mime_type=mime_type,
            data=base64.b64encode(raw).decode("ascii"),
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.data}"

    @property
    def ref(self) -> str:
        """Stable text reference, e.g. ``sha256:3f2a9c0d1b7e image/png``."""
        return f"sha256:{self.sha256[:12]} {self.mime_type}"

    def as_reference(self) -> ImagePart:
        """Copy without the base64 payload."""
        return ImagePart(
            mime_type=self.mime_type, sha256=self.sha256, size_bytes=self.size_bytes
        )


def describe_images(images: Sequence[ImagePart]) -> str:
    """In-text marker for images, e.g. ``[1 image: sha256:3f2a9c0d1b7e image/png]``."""
    if not images:
        return ""
    noun = "image" if len(images) == 1 else "images"
    return f"[{len(images)} {noun}: {', '.join(img.ref for img in images)}]"


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
    # Images that belong to this message (e.g. a screenshot a tool returned).
    # ``content`` stays plain text; providers render the multimodal form.
    images: list[ImagePart] = field(default_factory=list)
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
        images: Sequence[ImagePart] | None = None,
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
        self.images = list(images) if images else []
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
    images: list[ImagePart] = field(default_factory=list)


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
