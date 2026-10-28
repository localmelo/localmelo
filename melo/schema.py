from __future__ import annotations

import base64
import binascii
import hashlib
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ── Config ──

SHORT_TERM_MAX = 20
LONG_TERM_TOP_K = 5
TOOL_SEARCH_TOP_K = 3

MAX_AGENT_STEPS = 30
MIN_AGENT_STEPS = 3
STEPS_PER_ATTEMPT = 10
MAX_ATTEMPTS = 5

# Reflection budget. Only one reflection is carried into the next attempt's
# prompt. If its serialized form exceeds the char cap, the agent trims it to
# fit (AgentOptions.reflection_gate="flags", the default) or forces a
# decompose action instead of continuing ("utility").
MAX_REFLECTIONS_IN_PROMPT = 1
MAX_REFLECTION_CHARS = 2000

# ── Empty replies ──

# A planning reply with neither a tool call nor text is no answer. Working
# memory records it as EMPTY_REPLY_LABEL, an assistant message, followed by
# EMPTY_REPLY_NUDGE, a user message, and the agent plans again. The two
# form one unit of the window (see melo.memory.short.group_turns), so the
# nudge never follows another user message.
EMPTY_REPLY_LABEL = "[empty reply]"
EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Continue the task: call a tool or give your "
    "final answer."
)

# ── Stopped tool calls ──

# The error of a tool call that the run's cancellation, or its timeout_s,
# cut short after the call was handed to the executor.
CALL_INTERRUPTED = (
    "The run was stopped before this call finished; it may have partly run."
)
# The error that a cancelled interactive run gives each call of its last
# step that had not started when the run stopped (see Agent.run_task).
CALL_NOT_STARTED = "The run was stopped before this call started; it did not run."

# ── Answers kept in working memory ──

# With AgentOptions.record_answers, each run ends by appending its answer to
# working memory as an assistant message (see answer_text, clip_session_text
# and split_thinking). These labels stand in for a run that gave no answer.
# They are structural, in brackets, because models tend to echo prose
# placeholders that they find in their own earlier replies.
INTERRUPT_LABEL = "[interrupt: no assistant output for this turn]"
NO_ANSWER_LABEL = "[no answer: the run ended with status {status}]"
# Replaces a kept answer that the memory-write check refuses, with the
# check's reason; a status label would claim that no answer was given.
ANSWER_NOT_KEPT_LABEL = "[answer not kept in working memory: {reason}]"

# Characters of an answer or a user text that a session keeps in working
# memory (see clip_session_text).
SESSION_TEXT_CHARS = 8_000


def answer_text(status: str, result: str) -> str:
    """The answer of a run that ended with *status* and *result*, as kept.

    *result* itself when it has non-whitespace text; otherwise
    :data:`INTERRUPT_LABEL` for a ``cancelled`` run, and
    :data:`NO_ANSWER_LABEL` with the status for any other, so the kept
    answer is never blank.
    """
    if result.strip():
        return result
    if status == "cancelled":
        return INTERRUPT_LABEL
    return NO_ANSWER_LABEL.format(status=status)


def clip_session_text(text: str) -> str:
    """*text* cut to its first :data:`SESSION_TEXT_CHARS` characters.

    A longer text ends with a line break and
    ``... [<n> more characters not shown]``, where ``n`` counts the
    characters left out; a text that fits is returned as it is.
    """
    if len(text) <= SESSION_TEXT_CHARS:
        return text
    left_out = len(text) - SESSION_TEXT_CHARS
    return f"{text[:SESSION_TEXT_CHARS]}\n... [{left_out} more characters not shown]"


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPENING = re.compile(r"\s*<think>", re.IGNORECASE)
_THINK_CLOSING = re.compile(r"</think>", re.IGNORECASE)


ReasoningTags = Literal["explicit", "closing_only", "none"]


def split_thinking(
    text: str, *, strip_answer: bool = True, reasoning_tags: ReasoningTags | None = None
) -> tuple[str, str]:
    """Split the reasoning in a ``<think>`` block off *text*: ``(thinking, text)``.

    - The first ``<think>...</think>`` block, in any letter case and
      anywhere in *text*, is the thinking.
    - Otherwise a ``<think>`` at the start, after leading whitespace, that
      is never closed makes the rest of *text* the thinking, as when the
      output was cut off at the token limit while the model thought.
    - Otherwise a ``</think>`` with no ``<think>`` before it makes the text
      before it the thinking. Chat templates that open the block in the
      prompt (DeepSeek-R1 and Qwen3 thinking models) leave only the closing
      tag in the reply, and vLLM's R1 reasoning parser reads it the same way.
    - The rest, stripped, is the text. Without a block the result is
      ``("", text.strip())``. With ``strip_answer=False``, preserve the
      answer's surrounding whitespace, including Markdown code indentation.

    The thinking is returned stripped too. Ollama's provider, and an
    OpenAI-compatible server without a reasoning parser, put the reasoning
    at the start of a reply's content as such a block.
    A declared ``reasoning_tags`` selects the same trusted framing used by
    streaming: ``explicit`` recognizes a leading opening tag, ``closing_only``
    treats text before the first closing tag as thinking, and ``none`` keeps
    tags literal. With no declaration, the original rules above are unchanged.
    """
    if reasoning_tags is not None:
        if reasoning_tags == "none":
            return "", text.strip() if strip_answer else text
        if reasoning_tags == "explicit":
            leading = len(text) - len(text.lstrip())
            if not text[leading:].lower().startswith("<think>"):
                return "", text.strip() if strip_answer else text
            text = text[leading + len("<think>") :]
        elif reasoning_tags != "closing_only":
            raise ValueError("unknown reasoning_tags")
        closing = _THINK_CLOSING.search(text)
        if closing is None:
            return text.strip(), ""
        answer = text[closing.end() :]
        return text[: closing.start()].strip(), (
            answer.strip() if strip_answer else answer
        )
    block = _THINK_BLOCK.search(text)
    if block is not None:
        rest = text[: block.start()] + text[block.end() :]
        return block.group(1).strip(), rest.strip() if strip_answer else rest
    opening = _THINK_OPENING.match(text)
    if opening is not None:
        return text[opening.end() :].strip(), ""
    closing = _THINK_CLOSING.search(text)
    if closing is not None:
        rest = text[closing.end() :]
        return text[: closing.start()].strip(), rest.strip() if strip_answer else rest
    return "", text.strip() if strip_answer else text


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

    ``data`` is the base64 payload without a ``data:`` prefix. Whitespace
    in it (line-wrapped base64) is removed on construction, so what is
    sent is what the checker saw. Whenever there is ``data``, ``sha256``
    and ``size_bytes`` are derived from it on construction (values passed
    in are ignored) and describe the decoded bytes. A reference copy
    (:meth:`as_reference`) drops ``data`` and keeps the digest and size it
    is given. The agent keeps only reference copies of tool-result images
    in history and working memory, so no payload outlives the step that
    produced it; long-term memory gets text only. Invalid base64 yields
    ``size_bytes == 0``, which the checker rejects.
    """

    mime_type: str
    data: str = field(default="", repr=False)
    sha256: str = ""
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.data:
            return  # a reference copy keeps the digest and size it is given
        data = "".join(self.data.split())
        raw = decode_base64(data)
        digest_input = raw if raw is not None else data.encode()
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sha256", hashlib.sha256(digest_input).hexdigest())
        object.__setattr__(self, "size_bytes", len(raw) if raw is not None else 0)

    @classmethod
    def from_bytes(cls, raw: bytes, mime_type: str) -> ImagePart:
        return cls(mime_type=mime_type, data=base64.b64encode(raw).decode("ascii"))

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


# The permission tier of a tool: what a call can reach, not whether it is
# allowed. The table and the decision live in melo.checker.tiers; the type
# lives here so ToolDef, ToolInfo and ApprovalRequest can name it without
# importing the checker.
ToolTier = Literal["read", "write", "exec", "network"]


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    semantic_tags: list[str] = field(default_factory=list)
    # MCP annotations exactly as the server gave them, raw and unused by a
    # provider: melo.checker.tiers derives the tier. A shallow copy is kept
    # here, made in __post_init__, and nothing in melo or support mutates it.
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.annotations = dict(self.annotations)


@dataclass(frozen=True)
class ToolInfo:
    """A tool an agent can run, as :meth:`Agent.list_tools` reports it.

    ``source`` says who registered it: ``"builtin"`` for a built-in tool,
    ``"provided"`` for a tool of a tool provider (an MCP server, say), and
    ``"custom"`` for one registered with the agent's executor and tool
    registry directly. ``requires_approval`` says whether an approval
    policy is asked before a call of it runs; ``tier`` is the permission
    tier the checker derives for it (see ``melo.checker.tiers``).
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    source: Literal["builtin", "provided", "custom"]
    requires_approval: bool
    # The agent always fills this; "write" is the cautious default for a
    # caller that builds a ToolInfo itself.
    tier: ToolTier = "write"


def new_tool_call_id() -> str:
    """Return a fresh OpenAI-style tool call id (``call_`` + 24 hex chars)."""
    return f"call_{uuid.uuid4().hex[:24]}"


# Providers keep tool-call arguments that are not a JSON object as
# ``{RAW_ARGUMENTS_KEY: <text>}`` instead of failing the run. The executor
# rejects such a call before running anything and reports it to the model.
# When the call is replayed to the model, the providers send an empty
# object in place of this form.
RAW_ARGUMENTS_KEY = "_raw_arguments"


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


def new_record_id() -> str:
    """Return a fresh task or step id: a full UUID4 as 32 hex characters.

    History keeps every task and step under its id for good, so ids must
    not collide in a long-lived ``history.db``. A repeated task id would
    make ``save_task`` (an ``INSERT OR REPLACE``) overwrite the older
    task, and a repeated step id would fail the newer step's insert.
    """
    return uuid.uuid4().hex


@dataclass
class StepRecord:
    thought: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    step_id: str = field(default_factory=new_record_id)
    timestamp: float = field(default_factory=time.time)
    capture_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMCallRecord:
    """One model response, as a row of history's ``llm_calls`` table.

    ``seq`` numbers the responses of task ``task_id`` from 0, in the order
    they arrived. ``kind`` is the request's call kind (``plan``,
    ``reflect``, ...), ``outcome`` the shape of a ``plan`` reply
    (``tools``, ``answer`` or ``empty``) and ``aux`` for any other kind.
    ``content`` and ``thinking`` are the reply's, redacted, and
    ``redactions`` counts the spans replaced. ``reasoning_key`` names the
    response key the provider read the thinking from. ``usage`` is the
    whole usage the server reported, ``duration_s`` the request's time
    and ``timestamp`` when the response arrived.
    """

    task_id: str
    seq: int
    kind: str
    step: int | None
    attempt: int | None
    outcome: str
    content: str
    thinking: str
    reasoning_key: str
    redactions: int
    tool_calls: tuple[ToolCall, ...]
    usage: dict[str, int] | None
    duration_s: float
    timestamp: float
    # The recall records the request carried for the first time (R5), or
    # None when it carried no new block. Serialized as ``recall_json``.
    recall: Any | None = None


@dataclass
class TaskRecord:
    query: str
    task_id: str = field(default_factory=new_record_id)
    steps: list[StepRecord] = field(default_factory=list)
    # "running" until the run ends, then the run's status: one of the
    # RunResult statuses documented in docs/run-results.md ("Statuses"),
    # "cancelled" (an interactive run that was cancelled) included.
    status: str = "running"
    # The run's result, or its error when the result is empty.
    result: str = ""
    attempts_completed: int = 0


@dataclass
class RunResult:
    """How one run of :meth:`Agent.run_task` ended.

    ``status`` is one of the statuses documented in ``docs/run-results.md``
    ("Statuses"). ``cancelled`` is only for an interactive run, one with an
    event sink or an approval policy, whose task was cancelled: its result
    is empty and its error the reason, which is the exit's reason after
    :meth:`Agent.begin_exit`; any other cancelled run ends ``crashed``
    (see :meth:`Agent.run_task`). ``result`` is the final answer or a best
    effort, and may be empty; ``error`` is empty unless something failed.
    The counters cover this run only:

    - ``steps``: planning calls made, counted when each is sent;
    - ``attempts``: attempts started (1 for an answer on the first try);
    - ``n_llm``: LLM requests that returned a response, split into
      ``n_main`` (requests that offered tools) and ``n_aux`` (requests
      without tools, such as a reflection, or a planning call in a step
      that offered no tool);
    - ``n_tool``: tool calls handed to the executor, including calls it
      refused (an unknown tool, or one that step did not offer);
    - ``llm_errors``: LLM requests that raised instead of responding,
      counting each failed attempt of a retried request and a request
      cut short by a cancellation, so ``n_llm + llm_errors`` is the
      number of requests sent;
    - ``usage``: the integer usage fields of every response, summed by
      name (for example ``prompt_tokens``, ``completion_tokens``,
      ``total_tokens``).

    ``duration_s`` is the run's wall-clock time in seconds, and
    ``config`` is the effective run configuration (see
    :meth:`Agent.run_task`), so every record says which mode it ran in.

    ``memory_stats`` holds the run's replay counters in the explicit
    reasoning modes (see :meth:`Agent.run_task`), and is ``None`` in
    ``legacy``; :meth:`to_dict` leaves the key out then, so a legacy JSON
    line keeps the shape it had before R2.
    """

    task_id: str
    status: str
    result: str = ""
    error: str = ""
    steps: int = 0
    attempts: int = 0
    n_llm: int = 0
    n_main: int = 0
    n_aux: int = 0
    n_tool: int = 0
    llm_errors: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    memory_stats: dict[str, Any] | None = None
    context_budget: list[dict[str, Any]] | None = None
    prefix_cache: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """A deep copy as a plain dict, with keys in field order.

        ``memory_stats`` is omitted when it is ``None``, which keeps a
        ``legacy`` line byte-identical to the one before this field.
        """
        data = asdict(self)
        if data.get("memory_stats") is None:
            del data["memory_stats"]
        if data.get("context_budget") is None:
            del data["context_budget"]
        if data.get("prefix_cache") is None:
            del data["prefix_cache"]
        return data


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
    recommended_action: str = "stop"  # "continue" | "stop" | "decompose" | "done"
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


# ── Long-term memory candidates (R6-lite) ──


@dataclass(frozen=True)
class CandidateItem:
    """One item's text and records; None means no per-item attribution supplied."""

    text: str
    source_ids: tuple[str, ...] | None = None
    evidence: str | None = None


@dataclass
class MemoryCandidate:
    """One document offered to long-term memory, before the write boundary.

    ``candidate_id`` is the note id for a note and a fresh record id for a
    user fact; ``items`` carries a note's found/decided items so the policy
    routes each one, ``()`` routes the whole text. ``entry_id`` and
    ``processed`` are the staged copy's bookkeeping: an unchanged candidate
    that was already decided is not written twice.
    """

    candidate_id: str
    source: str
    text: str
    remember_by: str
    task_id: str
    session_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    remember: bool = False
    items: tuple[CandidateItem, ...] = ()
    entry_id: int | None = None
    processed: bool = False
