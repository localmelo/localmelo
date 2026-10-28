from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import sqlite3
import time
import uuid
from collections.abc import (
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from localmelo.melo.agent._call_log import CallHistory, CallLog
from localmelo.melo.agent._chat import (
    CALL_RECORDER,
    RETRY_HOOK,
    Chat,
    RequestStats,
    lacks_task_query,
)
from localmelo.melo.agent._events import _Emitter, check_sink
from localmelo.melo.agent.options import AgentOptions, ReflectionGate
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import (
    ExecutorResultPayload,
    MemoryWritePayload,
    ToolResolutionResult,
)
from localmelo.melo.checker.tiers import tier_of, tier_requires_approval
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from localmelo.melo.contracts.events import EventSink
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import (
    TOOL_EXIT_CODE,
    Executor,
    builtin_names,
    call_preview,
    register_builtins,
)
from localmelo.melo.executor.models import ExecutionRequest
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.procedures.normalize import (
    RoutineKey,
    key_for_query,
    key_for_routine,
    length_options,
    run_place,
)
from localmelo.melo.memory.procedures.store import RunFacts, harness_version
from localmelo.melo.memory.procedures.template import TemplateView
from localmelo.melo.memory.procedures.trace import Trace
from localmelo.melo.memory.procedures.tracker import (
    Limits,
    progress_state,
    shown_view,
)
from localmelo.melo.memory.short import group_turns
from localmelo.melo.schema import (
    ANSWER_NOT_KEPT_LABEL,
    CALL_INTERRUPTED,
    CALL_NOT_STARTED,
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    MAX_AGENT_STEPS,
    MAX_REFLECTION_CHARS,
    MIN_AGENT_STEPS,
    ImagePart,
    Message,
    ReflectionDecision,
    ReflectionEntry,
    RunResult,
    StepRecord,
    TaskRecord,
    ToolCall,
    ToolDef,
    ToolInfo,
    ToolResult,
    ToolTier,
    answer_text,
    clip_session_text,
    describe_images,
    new_tool_call_id,
    split_thinking,
)

if TYPE_CHECKING:
    from localmelo.support.config import Config

logger = logging.getLogger(__name__)

# ── Reflection coercion helpers ──


def _coerce_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _coerce_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _coerce_unit_float(value: object, default: float = 0.0) -> float:
    """Coerce to a float clamped to [0.0, 1.0]. Non-finite values use *default*."""
    return max(0.0, min(1.0, _coerce_float(value, default)))


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value] if value else []
    if value is None:
        return []
    return [str(value)]


# The decisions a reflection may recommend. Under reflection_gate="utility"
# any other value reads "stop"; under "flags" an object without one of them
# is no reflection (see _find_reflection).
REFLECTION_ACTIONS = ("continue", "stop", "decompose", "done")


def _strip_code_fences(text: str) -> str:
    """*text* without a Markdown code fence around it, such as ```json ... ```."""
    body = text.strip()
    if body.startswith("```"):
        # The opening fence may name a language; drop its whole line.
        newline = body.find("\n")
        body = body[newline + 1 :] if newline != -1 else body[3:]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def _json_values(text: str) -> Iterator[object]:
    """Yield the JSON values of a reflection reply, in the order they are read.

    A Markdown code fence around the reply is stripped first. When the
    rest is JSON as a whole, that value is the only one. Otherwise each
    balanced ``{...}`` in it that parses as a JSON object is one, so text
    around the objects does not matter. After a ``{`` that does not
    start one, the search goes on from where the decoder gave up, and
    after an object, from its end, so it reads the reply about once and
    never starts inside a string of the text it already read, nor inside
    an object it yielded. Nesting deeper than the decoder's recursion
    limit ends the search.
    """
    body = _strip_code_fences(text)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        pass
    else:
        yield value
        return
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
            continue
        except RecursionError:
            return
        yield value
        start = body.find("{", end)


def _decode_reflection(text: str) -> object:
    """Return the first JSON value of a reflection reply; ``ValueError`` if none.

    See :func:`_json_values` for how the reply is read: the reply as a
    whole when it is JSON, else the first ``{...}`` object in it.
    """
    for value in _json_values(text):
        return value
    raise ValueError("No JSON object in the reflection response")


def _holds_reflection_object(text: str) -> bool:
    """Whether a reflection reply holds a JSON object (see :func:`_decode_reflection`)."""
    try:
        return isinstance(_decode_reflection(_coerce_str(text)), dict)
    except ValueError:
        return False


def _find_reflection(text: str) -> dict[str, Any] | None:
    """The reflection a reply holds under ``reflection_gate="flags"``, or ``None``.

    That is the first JSON value of the reply (see :func:`_json_values`)
    that is an object whose ``recommended_action``, as a string, is one
    of :data:`REFLECTION_ACTIONS`. An object without one, such as the
    arguments of a tool call written out as text, is passed over for the
    objects after it, but an object nested inside it is not looked at.
    JSON that is not an object holds no reflection.
    """
    try:
        for value in _json_values(_coerce_str(text)):
            if not isinstance(value, dict):
                continue
            if _coerce_str(value.get("recommended_action")) in REFLECTION_ACTIONS:
                return value
    except ValueError:  # a number the decoder cannot convert ends the search
        pass
    return None


def _as_planned(window: list[Message], query: str) -> list[Message]:
    """*window* as a planning request carries it, for size checks.

    That is with the task *query* in front once the window no longer holds
    it, as :meth:`Chat.plan_step` puts it back (see
    :func:`~localmelo.melo.agent._chat.lacks_task_query`).
    """
    if lacks_task_query(window, query):
        return [Message(role="user", content=query), *window]
    return window


def _serialize_reflections(entries: list[ReflectionEntry]) -> list[Message]:
    """Convert reflection entries into compact system Messages for retrieval context."""
    if not entries:
        return []
    parts: list[str] = []
    for r in entries:
        lines = [f"[reflection] Attempt {r.attempt_id}: {r.summary}"]
        if r.recommended_avoids:
            lines.append(f"  Avoid: {'; '.join(r.recommended_avoids)}")
        if r.next_promising_directions:
            lines.append(f"  Try: {'; '.join(r.next_promising_directions)}")
        if r.useful_evidence:
            lines.append(f"  Evidence: {'; '.join(r.useful_evidence)}")
        if r.failed_hypotheses:
            lines.append(f"  Failed: {'; '.join(r.failed_hypotheses)}")
        parts.append("\n".join(lines))
    return [Message(role="system", content="\n".join(parts))]


# The lists of a reflection entry that _fit_reflection empties, in this
# order and each from its end, until the entry fits its budget.
REFLECTION_TRIM_ORDER = (
    "failed_hypotheses",
    "useful_evidence",
    "recommended_avoids",
    "next_promising_directions",
)


def _reflection_size(entry: ReflectionEntry) -> int:
    """Characters of *entry* as :func:`_serialize_reflections` writes it."""
    return len(_serialize_reflections([entry])[0].content)


def _shorten(text: str, width: int) -> str:
    """*text* in at most *width* characters, cut at a word boundary, ending in "…".

    Text that fits is returned as it is. When the cut falls inside the
    first word, there is no boundary to go back to, and the text is cut
    where *width* falls.
    """
    if len(text) <= width:
        return text
    if width < 1:
        return ""
    head = text[: width - 1]
    if not text[len(head)].isspace() and not head[-1:].isspace():
        # The cut falls inside a word: leave that word out.
        words = head.rsplit(None, 1)
        if len(words) == 2:
            head = words[0]
    return head.rstrip() + "…"


def _fit_reflection(entry: ReflectionEntry, limit: int) -> ReflectionEntry:
    """*entry*, trimmed until its serialized form fits in *limit* characters.

    The serialized form is the one :func:`_serialize_reflections` writes.
    Items are dropped from the end of each list of
    :data:`REFLECTION_TRIM_ORDER` in turn (``failed_hypotheses`` first,
    ``next_promising_directions`` last), one at a time and only until the
    entry fits. When the lists are empty and it still does not fit, the
    summary is shortened at a word boundary and ends with "…" (see
    :func:`_shorten`). *entry* itself is not changed: one that fits is
    returned as it is, any other as a trimmed copy.
    """
    if _reflection_size(entry) <= limit:
        return entry
    fitted = dataclasses.replace(entry)
    for name in REFLECTION_TRIM_ORDER:
        items = list(getattr(entry, name))  # a copy: entry keeps its own
        setattr(fitted, name, items)
        while items:
            items.pop()
            if _reflection_size(fitted) <= limit:
                return fitted
    room = limit - _reflection_size(dataclasses.replace(fitted, summary=""))
    fitted.summary = _shorten(fitted.summary, room)
    return fitted


# Characters of a withheld tool result that stay visible to the model. A
# result left less room than this in its turn is withheld, not cut.
WITHHELD_PREVIEW_CHARS = 200

# Characters of a turn's working-memory budget that the assistant's text
# keeps ahead of the tool results. A text this short stays beside results
# cut to fit; a longer one is dropped when it does not fit beside them
# (see Agent._checked_tool_turn). History keeps the text either way.
ASSISTANT_TEXT_SHARE = 2_000

# Key of the placeholder that replaces tool-call arguments too long for
# working memory. Only the working-memory copy of a turn carries it:
# history and the executor always get the real arguments.
ELIDED_ARGUMENTS_KEY = "_elided"


def _render_arguments(arguments: dict[str, Any]) -> str:
    """Text form of one call's arguments, as memory-write checks see it."""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)


def _render_tool_calls(calls: Sequence[ToolCall]) -> str:
    """Text form of tool calls, used for memory-write checks."""
    return "\n".join(
        f"{tc.tool_name}({_render_arguments(tc.arguments)})" for tc in calls
    )


def _elide_arguments(call: ToolCall) -> ToolCall:
    """Copy of *call* with its arguments replaced by a short placeholder.

    The copy keeps the call's id, so the call's result still pairs with it.
    """
    size = len(_render_arguments(call.arguments))
    return dataclasses.replace(
        call, arguments={ELIDED_ARGUMENTS_KEY: f"{size} chars withheld"}
    )


def _image_references(result: ToolResult) -> ToolResult:
    """Copy of *result* whose images keep no base64 payload.

    History and working memory get this copy, so no payload outlives the
    step that produced it (see :meth:`ImagePart.as_reference`).
    """
    return dataclasses.replace(
        result, images=[image.as_reference() for image in result.images]
    )


def _result_digest(result: ToolResult) -> str:
    """Digest of what one call returned, for stuck detection.

    It covers the output, the error and each image's digest, but not the
    duration, so the same result read twice gives the same digest.
    """
    returned = [result.output, result.error, [image.sha256 for image in result.images]]
    return hashlib.sha256(json.dumps(returned).encode()).hexdigest()[:16]


def _tool_result_text(result: ToolResult) -> str:
    """Text of a tool result, without the line that names its images.

    Empty or whitespace-only output reads ``(no output)``, unless the
    result has images: an image-only result has no text, and its tool
    message is just that line (see :func:`_tool_message_content`).
    """
    if result.error:
        return f"Error: {result.error}"
    if result.output.strip():
        return result.output
    return "" if result.images else "(no output)"


def _tool_message_content(text: str, images: Sequence[ImagePart]) -> str:
    """Content of the ``role="tool"`` message that answers a call.

    *text* is the result's text (see :func:`_tool_result_text`), cut to
    fit or not, or the notice that replaced it. The images of a result
    are not sent to the model, so a result with images ends with a line
    that names them (see :func:`describe_images`), whether its text is
    kept, cut or withheld.
    """
    return "\n".join(part for part in (text, describe_images(images)) if part)


def _withheld_notice(text: str, reason: str) -> str:
    """Stand-in for a tool result kept out of working memory."""
    notice = f"[Tool result withheld from working memory: {reason}."
    preview = text[:WITHHELD_PREVIEW_CHARS]
    if not preview:
        return f"{notice}]"  # an image-only result: nothing to preview
    return f"{notice} First {len(preview)} characters follow.]\n{preview}"


def _truncation_marker(left_out: int) -> str:
    """Ends a tool result that was cut to fit working memory."""
    return f"\n... [truncated: {left_out} more characters]"


def _cut_to(text: str, room: int, *, cut_before: int = 0) -> str:
    """*text*, or its start and a truncation marker, in at most *room* chars.

    The marker says how many characters were left out (see
    :func:`_truncation_marker`): those of *text*, plus *cut_before*, the
    characters by which the executor-result check had already shortened
    the tool's output when it made *text* (it cuts the output to
    ``MAX_OUTPUT_LEN`` and ends it with a short suffix of its own, which
    this cut drops). So the count covers the tool's whole output, and a
    model that sees a 120,000-character output cut to fit learns that
    about 70,000 characters are missing, not the few its last cut took.
    Every caller leaves at least ``WITHHELD_PREVIEW_CHARS`` of room, which
    is ample for the marker.
    """
    if len(text) <= room:
        return text
    # The marker is never longer than this one, whose count is the
    # largest possible, so the cut text never passes *room*.
    keep = max(room - len(_truncation_marker(len(text) + cut_before)), 0)
    return text[:keep] + _truncation_marker(len(text) - keep + cut_before)


def _within_turn_budget(
    texts: Sequence[str],
    withheld: set[int],
    calls_chars: int,
    *,
    limit: int,
    cut_before: Sequence[int],
) -> list[str]:
    """Return *texts* with the largest cut, or replaced, until they fit.

    The rendered tool calls of one turn (*calls_chars* characters) and its
    tool results share *limit* characters: ``MAX_MEMORY_TEXT_LEN`` less
    the share kept for the assistant's text (see
    :meth:`Agent._checked_tool_turn`). The calls keep their share and the
    results get the rest. While the results do not fit, the largest is
    cut to the room the others leave it, with a marker that says how much
    was left out, counting the characters by which the executor-result
    check had already shortened it (*cut_before*, one count per text; see
    :func:`_cut_to`). A result left less than ``WITHHELD_PREVIEW_CHARS``
    of room is replaced by the withheld notice instead, and its index
    added to *withheld*. Indices already in *withheld* hold notices and
    are left alone.
    """
    fitted = list(texts)
    budget = limit - calls_chars
    total = sum(len(text) for text in fitted)
    if total <= budget:
        return fitted
    reason = (
        f"Tool calls and results too large: {calls_chars + total} chars "
        f"(max {limit})"
    )
    largest_first = sorted(
        (i for i in range(len(fitted)) if i not in withheld),
        key=lambda i: len(fitted[i]),
        reverse=True,
    )
    for index in largest_first:
        if total <= budget:
            break
        text = fitted[index]
        room = budget - (total - len(text))
        if room >= WITHHELD_PREVIEW_CHARS:
            kept = _cut_to(text, room, cut_before=cut_before[index])
        else:
            kept = _withheld_notice(text, reason)
            if len(kept) >= len(text):
                break  # only results a notice would not shorten are left
            withheld.add(index)
        total += len(kept) - len(text)
        fitted[index] = kept
    return fitted


# Empty replies in a row that end the run as failed ("empty replies").
# Until then, working memory records each one as EMPTY_REPLY_LABEL
# followed by EMPTY_REPLY_NUDGE (see melo.schema), and the model plans
# again (see Agent._run_attempt).
MAX_EMPTY_REPLIES = 2

# The utility (see _utility) that reflection_gate="utility" requires for
# the next attempt to start.
CONTINUATION_UTILITY_THRESHOLD = 0.1


def _utility(decision: ReflectionDecision) -> float:
    """The utility of another attempt, from the reflection's own estimates.

    ``estimated_info_gain * feasibility * novelty - estimated_cost -
    repeat_risk``, each a number in [0, 1] that the model chose. Only
    ``reflection_gate="utility"`` acts on it (see :func:`_should_continue`).
    """
    return (
        decision.estimated_info_gain * decision.feasibility * decision.novelty
        - decision.estimated_cost
        - decision.repeat_risk
    )


def _should_continue(
    decision: ReflectionDecision, gate: ReflectionGate = "flags"
) -> bool:
    """Whether the next attempt starts after a reflection's *decision*.

    With either *gate* (see ``AgentOptions.reflection_gate``), only when
    the decision is ``continue`` and its three checks hold:
    ``task_still_feasible``, ``next_step_is_concrete`` and
    ``next_step_is_novel``. ``"flags"`` asks nothing more. ``"utility"``,
    the gate from before that option, also requires :func:`_utility` to
    reach :data:`CONTINUATION_UTILITY_THRESHOLD`.
    """
    if decision.recommended_action != "continue":
        return False
    if not decision.task_still_feasible:
        return False
    if not decision.next_step_is_concrete:
        return False
    if not decision.next_step_is_novel:
        return False
    if gate == "utility":
        return _utility(decision) >= CONTINUATION_UTILITY_THRESHOLD
    return True


def _providers_from_config(
    cfg: Config,
) -> tuple[BaseLLMProvider, BaseEmbeddingProvider | None]:
    """Build LLM and embedding providers from a Config object.

    Uses the split backend model: ``cfg.chat_backend`` selects the chat
    provider and ``cfg.embedding_backend`` selects the embedding provider.
    The two may refer to different backend adapters.
    """
    from localmelo.support.backends import get_backend

    chat_backend = get_backend(cfg.chat_backend)
    llm = chat_backend.build_chat_provider(cfg)

    embedding = None
    if cfg.has_embedding:
        emb_key = cfg.embedding_backend  # direct key, no mapping needed
        emb_backend = get_backend(emb_key)
        embedding = emb_backend.build_embedding_provider(cfg)

    return llm, embedding


# SQLite directory when LOCALMELO_PERSIST_MEMORY is set without
# LOCALMELO_MEMORY_DIR. Expanded when used, not at import.
DEFAULT_MEMORY_DIR = "~/.cache/localmelo/memory"

# Seconds that promoting a run's reflections may take once the run is
# over. Without a bound, a run that ended because the embedding server
# stopped answering would wait for it again, as long as the provider's
# own HTTP timeout.
PROMOTION_TIMEOUT_S = 10.0

# Seconds of an exit's time limit (see Agent.begin_exit) that closing the
# tool providers leaves for saving the run's task record.
EXIT_SAVE_RESERVE_S = 1.0

# Seconds that writing the history rows and the task record of a cancelled
# interactive run may take together (see Agent._finalize). Agent.close()
# waits one second more for such a run to be finalized.
FINALIZE_TIMEOUT_S = 10.0

# The error of a cancelled interactive run whose cancellation gave no
# reason (see Agent.run_task).
CANCELLED_ERROR = "Cancelled by the caller"

# Seconds that a run's start may spend finding the template its progress
# rests on, inside the run's own timeout (see Agent._look_up). A lookup
# that takes longer, or fails, leaves the run at level 0.
LOOKUP_TIMEOUT_S = 1.0

# Seconds that recording a run in the procedural store may take once the
# run has ended (see Agent._learn): after run.finished, before run_task
# returns, outside RunResult.duration_s.
LEARN_TIMEOUT_S = 2.0

# The installed localmelo's version, which each episode records; read once.
_harness_version = functools.cache(harness_version)

# The note that every planning request of an agent with a workspace root
# carries after the system prompt, with {root} filled in (see Chat).
WORKSPACE_NOTE = (
    "Working directory: {root}. Relative paths and shell and Python commands run "
    "there; file_read and file_write refuse paths outside it."
)


@dataclasses.dataclass(frozen=True)
class _Exit:
    """An exit under way (see :meth:`Agent.begin_exit`)."""

    reason: str
    # Event-loop time by which closing and saving must be done.
    deadline: float

    def time_left(self, *, keep: float = 0.0) -> float:
        """Seconds until *keep* seconds before the deadline, at least 0."""
        now = asyncio.get_running_loop().time()
        return max(self.deadline - keep - now, 0.0)


def _resolve_memory_dir(options: AgentOptions) -> str | None:
    """Directory for SQLite-backed memory, or ``None`` for in-memory backends.

    ``options.memory_dir`` comes first, with ``~`` expanded. Otherwise any
    non-empty ``LOCALMELO_PERSIST_MEMORY`` turns persistence on, in
    ``LOCALMELO_MEMORY_DIR`` (used as given) or :data:`DEFAULT_MEMORY_DIR`.
    """
    if options.memory_dir is not None:
        return os.path.expanduser(options.memory_dir)
    if os.environ.get("LOCALMELO_PERSIST_MEMORY"):
        return os.environ.get(
            "LOCALMELO_MEMORY_DIR", os.path.expanduser(DEFAULT_MEMORY_DIR)
        )
    return None


def _resolve_workspace_root(value: str | None) -> str | None:
    """The real path of the root *value*, with ``~`` expanded; ``None`` for none.

    Raises ``ValueError`` when that path is not a directory.
    """
    if value is None:
        return None
    try:
        root = os.path.realpath(os.path.expanduser(value))
    except ValueError:  # a NUL character: no file has that path
        root = ""
    if not root or not os.path.isdir(root):
        raise ValueError(f"workspace_root is not a directory: {value!r}")
    return root


def _answers_every_call(call_msg: Message, results: Sequence[Message]) -> bool:
    """Whether *results* answer each tool call of *call_msg* exactly once."""
    ids = [call.id for call in call_msg.tool_calls]
    answered = [msg.tool_call_id for msg in results]
    return len(set(ids)) == len(ids) and sorted(ids) == sorted(answered)


@dataclasses.dataclass
class _InflightStep:
    """The step whose tool calls are under way (see ``_RunState.inflight``).

    It lets a cancellation that lands while the calls run, or while
    :meth:`Agent._do_memorize` records them, close the step in working
    memory and write the history rows it is missing (see
    :meth:`Agent._on_cancelled` and :meth:`Agent._finalize`).
    """

    # The planning reply whose tool calls run.
    response: Message
    # The checked result of each call that returned, in call order ...
    results: list[ToolResult] = dataclasses.field(default_factory=list)
    # ... and, for each, the characters the executor-result check cut.
    cut_before: list[int] = dataclasses.field(default_factory=list)
    # The history records _do_memorize created, in call order, and how
    # many of them store_step finished writing: all of them, or all but
    # the last.
    records: list[StepRecord] = dataclasses.field(default_factory=list)
    stored: int = 0
    # The result the closed step gives each call, once it is closed.
    closed: list[ToolResult] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _RunState:
    """Progress of one :meth:`Agent.run_task` call.

    ``status`` stays ``"running"`` until :meth:`end` sets one of the
    RunResult statuses. The counters cover this run only (see
    :class:`~localmelo.melo.schema.RunResult`).
    """

    status: str = "running"
    result: str = ""
    error: str = ""
    steps: int = 0
    attempts: int = 0
    n_tool: int = 0
    # The latest non-blank text of a planning reply in this run: the
    # answer of a reflection's "done" that gives none, and the result of a
    # final boundary whose reflection is unreadable (reflection_gate="flags").
    last_text: str = ""
    # Planning replies in a row with neither a tool call nor text.
    empty_replies: int = 0
    # The run's task id ("" for a call without a task record).
    task_id: str = ""
    # Reports the run's events to the caller's sink; silent without one.
    events: _Emitter = dataclasses.field(default_factory=_Emitter)
    # The approval policy the caller passed to run_task, or None.
    approval: ApprovalPolicy | None = None
    # Whether working memory holds the run's query.
    query_in_memory: bool = False
    # The id of the call whose approval the policy is deciding, or "".
    waiting_call_id: str = ""
    # The step whose tool calls are under way, from right before its first
    # call to right after _do_memorize has recorded them, else None.
    inflight: _InflightStep | None = None
    # The RunResult, once it is built.
    outcome: RunResult | None = None
    # The task that finalizes the run after an interactive cancellation
    # (see Agent._finalize_in_task), once it is started.
    finalizer: asyncio.Task[None] | None = None
    # The run's procedural trace: its steps, tool calls, reflections and
    # stuck notice, appended where their events are emitted, with or
    # without a sink (see note). It builds no event and keeps no output.
    trace: Trace = dataclasses.field(default_factory=Trace)
    # Whether the trace still follows the run: false once updating it
    # failed, which ends the run's progress events and its learning.
    tracing: bool = True
    # Whether the run still reports progress: false once the tracker
    # failed, which ends the run's progress events only.
    reporting: bool = True
    # The routine name the caller gave (run_task's routine), the folder the
    # run's tools work in, read at its start (None when that failed), and
    # the routine's key once it is needed (see Agent._routine_key).
    routine: str | None = None
    place: str | None = None
    key: RoutineKey | None = None
    # What the run-start lookup found: the template the run's progress
    # rests on, at the level its gate gives, before the level cap of
    # shown_view. None for level 0.
    view: TemplateView | None = None
    # The run's budgets, which run.progress reports.
    limits: Limits | None = None

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def interactive(self) -> bool:
        """Whether a caller follows the run, with ``on_event`` or a policy.

        An observer (``observe``) does not make a run interactive.
        """
        return self.events.interactive or self.approval is not None

    def end(self, status: str, *, result: str = "", error: str = "") -> None:
        self.status = status
        self.result = result
        self.error = error

    def note(self, update: Callable[[Trace], object]) -> None:
        """Apply *update* to the trace, then report the run's progress.

        Never raises: a failure is logged at DEBUG, and the trace stops
        following the run, so no further ``run.progress`` event comes.
        """
        if not self.tracing:
            return
        try:
            update(self.trace)
        except Exception as exc:
            self.tracing = False
            logger.debug(
                "The trace of task %s stopped: %s", self.task_id, _exception_text(exc)
            )
            return
        self.report_progress()

    def report_progress(self, *, ended: str | None = None) -> None:
        """A ``run.progress`` event, when the caller asked for progress.

        *ended* is the run's status on the event that comes right before
        ``run.finished``. The level shown is capped by
        :func:`~localmelo.melo.memory.procedures.tracker.shown_view`.
        Never raises: a failure of the tracker is logged at DEBUG and ends
        the run's progress events, not the run or its learning.
        """
        if not (self.tracing and self.reporting and self.events.reports_progress):
            return
        try:
            view = shown_view(self.view)
            state = progress_state(view, self.trace, self.limits, ended=ended)
        except Exception as exc:
            self.reporting = False
            logger.debug(
                "The progress of task %s stopped: %s",
                self.task_id,
                _exception_text(exc),
            )
            return
        self.events.progress(state)


def _exception_text(exc: BaseException) -> str:
    """``"<type>: <message>"``, or the type alone when the message is empty."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _is_cancellation(exc: BaseException) -> TypeGuard[asyncio.CancelledError]:
    """Whether *exc* is a cancellation of the current task.

    That is a ``CancelledError`` while something cancels the task: its
    ``cancelling()`` count is above 0, as after ``task.cancel()``, a
    scope such as ``asyncio.timeout`` that cancels it, or the run's own
    ``timeout_s`` until the deadline turns the cancellation into a
    ``TimeoutError``. A ``CancelledError`` that a component raises while
    nothing cancels the task, for example a tool function's, is that
    component's failure.
    """
    if not isinstance(exc, asyncio.CancelledError):
        return False
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _check_approval(approval: object) -> None:
    """Raise ``TypeError`` unless *approval* is ``None`` or an approval policy.

    That is an :class:`ApprovalPolicy` whose ``decide`` is a coroutine
    function.
    """
    if approval is None:
        return
    if not (
        isinstance(approval, ApprovalPolicy)
        and inspect.iscoroutinefunction(approval.decide)
    ):
        raise TypeError("approval must have an async decide(request) method")


class _ApprovalGate:
    """The executor's approval gate for one call of a run with a policy.

    The executor calls it just before a tool that requires approval would
    run (see :meth:`Executor.execute_structured`). It builds the
    :class:`ApprovalRequest` from the request that would run, asks the
    run's policy, and keeps what came of it in :attr:`asked` for the
    call's ``tool.finished`` event. While the policy decides,
    ``run.waiting_call_id`` names the call, so that a cancellation there
    closes the call as not started (see :meth:`Agent._on_cancelled`).

    The run's cancellation propagates, and the prompt counts as withdrawn.
    The policy fails closed otherwise: an exception it raises, a
    ``CancelledError`` while nothing cancels the run, and a return value
    other than an :class:`ApprovalDecision` each reach the executor as an
    exception, and the executor refuses the call.
    """

    def __init__(
        self,
        policy: ApprovalPolicy,
        run: _RunState,
        call: ToolCall,
        workspace_root: str | None,
        tier: ToolTier,
    ) -> None:
        self._policy = policy
        self._run = run
        self._call = call
        self._workspace_root = workspace_root
        self._tier = tier
        # The approval of tool.finished: request_id, allowed, reason,
        # remember and origin, once the policy is asked; None until then.
        self.asked: dict[str, Any] | None = None

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        run, call = self._run, self._call
        asking = ApprovalRequest(
            request_id=uuid.uuid4().hex,
            run_id=run.task_id,
            call_id=call.id,
            tool_name=request.tool_name,
            arguments=copy.deepcopy(request.arguments),
            preview=call_preview(request.tool_name, request.arguments),
            workspace_root=self._workspace_root,
            tier=self._tier,
        )
        run.waiting_call_id = call.id
        try:
            decided: object = await self._policy.decide(asking)
        except asyncio.CancelledError as exc:
            if _is_cancellation(exc):
                # waiting_call_id stays set: the call did not start.
                self._record(asking, origin="withdrawn")
                raise
            run.waiting_call_id = ""
            spurious = RuntimeError(
                "the approval policy raised CancelledError while the run was "
                "not cancelled"
            )
            self._record(asking, origin="error", reason=_exception_text(spurious))
            raise spurious from None
        except BaseException as exc:
            run.waiting_call_id = ""
            self._record(asking, origin="error", reason=_exception_text(exc))
            raise
        run.waiting_call_id = ""
        if not isinstance(decided, ApprovalDecision):
            no_decision = TypeError(
                f"the approval policy returned {type(decided).__name__}, "
                "not an ApprovalDecision"
            )
            self._record(asking, origin="error", reason=_exception_text(no_decision))
            raise no_decision
        self._record(
            asking,
            origin=decided.origin,
            allowed=decided.allow,
            reason=decided.reason,
            remember=decided.remember,
        )
        return decided

    def _record(
        self,
        asking: ApprovalRequest,
        *,
        origin: str,
        allowed: bool = False,
        reason: str = "",
        remember: bool = False,
    ) -> None:
        """Keep the outcome of *asking*.

        *origin* is the decision's, ``"error"`` when the policy failed (the
        reason is then the failure), or ``"withdrawn"`` when the run was
        cancelled while the policy decided.
        """
        self.asked = {
            "request_id": asking.request_id,
            "allowed": allowed,
            "reason": reason,
            "remember": remember,
            "origin": origin,
        }


class _ToolSetupError(Exception):
    """The tool providers could not be set up; the message is the run's error."""


class Agent:
    """Task-solving agent with tool use and memory.

    Two construction modes:

    1. **Config-based** (gateway / full app)::

        from localmelo.support.config import load
        agent = Agent(config=load())

    2. **Direct provider injection** (testing / custom setups)::

        agent = Agent(llm=my_llm_provider, embedding=my_emb_provider)

    Either mode takes ``options=AgentOptions(...)`` to set step budgets,
    the attempt policy and its reflection gate, tool exposure, the
    built-in tools, memory and a run timeout; the defaults keep the
    behavior from before options existed, except that every step now
    offers all registered built-ins and runs only the tools it offered,
    and that with ``reflection_gate="flags"`` a reflection decides on its
    own checks whether the next attempt starts, a reply without a
    reflection does not stop the run, and an over-long reflection is
    trimmed (``"utility"`` keeps the behavior from before that option,
    see :meth:`_run_reflective_attempts`).
    ``tool_providers`` supply further tools, for example those of MCP
    servers: the first run starts them and registers their tools next to
    the built-ins (see :meth:`_ensure_tools`), and :meth:`close` closes
    them.

    With ``workspace_root``, the built-in tools work in that directory
    (see :class:`AgentOptions`); with ``record_answers``, each run's answer
    stays in working memory for the next run.

    :meth:`run_task` runs one task and returns a
    :class:`~localmelo.melo.schema.RunResult`; :meth:`run` returns its
    outcome as text. Runs on one agent share its working memory and
    request counters, so they must not overlap. :meth:`start` sets up the
    tool providers ahead of the first run, :meth:`list_tools` lists the
    tools, :meth:`run_tool` runs one tool call outside a run, and
    :meth:`conversation` and :meth:`load_conversation` read and replace the
    working-memory window between runs.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        llm: BaseLLMProvider | None = None,
        embedding: BaseEmbeddingProvider | None = None,
        options: AgentOptions | None = None,
        tool_providers: Sequence[BaseToolProvider] = (),
    ) -> None:
        if llm is not None:
            self._llm = llm
            self._embedding = embedding
        elif config is not None:
            self._llm, self._embedding = _providers_from_config(config)
        else:
            raise TypeError("Agent requires either config= or llm= to be provided")

        self.options = options if options is not None else AgentOptions()
        # Resolved first, so that a bad root raises before anything, such as
        # the memory directory, is created.
        workspace_root = _resolve_workspace_root(self.options.workspace_root)
        self._tool_providers: tuple[BaseToolProvider, ...] = tuple(tool_providers)
        # Tool setup (_ensure_tools) runs once, under this lock. It ends
        # either ready or failed; a failure is kept as its error text.
        self._tools_lock = asyncio.Lock()
        self._tools_ready = False
        self._tools_failure: str | None = None
        # The names of the tools that the setup registered (see list_tools).
        self._provided_names: frozenset[str] = frozenset()
        # close() or begin_exit() has begun: no tool provider is started
        # from then on.
        self._closing = False
        # begin_exit() has been called: the process exits soon.
        self._exit: _Exit | None = None
        # Provider start() and close() calls an exit stopped waiting for;
        # kept so they can finish in the background.
        self._abandoned: set[asyncio.Future[Any]] = set()
        # The finalizers of cancelled interactive runs still under way, each
        # with its run's task id (see _finalize_in_task); close() waits for
        # them.
        self._finalizers: dict[asyncio.Task[None], str] = {}
        # Runs under way: run_task calls, each until it returns or raises,
        # or after an interactive cancellation until its finalizer ends.
        self._active_runs = 0

        # Optional persistent memory backends: options.memory_dir, or the
        # LOCALMELO_PERSIST_MEMORY / LOCALMELO_MEMORY_DIR opt-in. History and
        # the procedural store are SQLite whenever persistence is on;
        # long-term only when memory is on and an embedding provider exists.
        self._memory_dir = _resolve_memory_dir(self.options)
        history_backend = None
        long_backend = None
        procedures_backend = None
        if self._memory_dir is not None:
            from localmelo.melo.memory._sqlite import (
                make_private_dir,
                thinking_retention_days,
                tighten_private_dir,
            )
            from localmelo.melo.memory.history.sqlite import SqliteHistory
            from localmelo.melo.memory.long.sqlite import SqliteLongTerm
            from localmelo.melo.memory.procedures.store import SqliteProcedures

            # Read first, so that a bad value raises before the directory is
            # created.
            retention_days = thinking_retention_days()
            make_private_dir(self._memory_dir)
            if (
                self.options.memory_dir is None
                and "LOCALMELO_MEMORY_DIR" not in os.environ
            ):
                # The default directory, localmelo's own: one that an older
                # version made with the umask's mode is closed to others.
                tighten_private_dir(self._memory_dir)
            history_backend = SqliteHistory(
                os.path.join(self._memory_dir, "history.db"),
                thinking_retention_days=retention_days,
            )
            if self._embedding and self.options.memory:
                long_backend = SqliteLongTerm(
                    os.path.join(self._memory_dir, "long_term.db")
                )
            # Opened, and created, at its first use: a run's end.
            procedures_backend = SqliteProcedures(
                os.path.join(self._memory_dir, "procedures.db")
            )

        # memory=False runs Hippo in its no-embedding mode, which is exactly
        # the documented behavior of that option: no long-term recall, no
        # long-term writes, no reflection promotion, and reflections are
        # still cleared. self._embedding stays set so close() closes it.
        self.hippo = Hippo(
            embedding=self._embedding if self.options.memory else None,
            history=history_backend,
            long=long_backend,
            procedures=procedures_backend,
        )
        self.checker = Checker()
        self.executor = Executor(
            self.hippo, self.checker, workspace_root=workspace_root
        )
        notes = (
            []
            if workspace_root is None
            else [WORKSPACE_NOTE.format(root=workspace_root)]
        )
        self.chat = Chat(self._llm, system_notes=notes)

        register_builtins(
            self.executor,
            self.hippo,
            names=self.options.builtins,
            workspace_root=workspace_root,
            tool_timeout_s=self.options.tool_timeout_s,
        )
        # The names of the built-ins registered above (see list_tools).
        self._builtin_names = (
            builtin_names() if self.options.builtins is None else self.options.builtins
        )

    # ── Workspace, tools and conversation ──

    @property
    def workspace_root(self) -> str | None:
        """The real path of the agent's workspace root, or ``None``."""
        return self.executor.workspace_root

    async def start(self) -> None:
        """Start the tool providers and register their tools now.

        That is the setup the first run would do otherwise (see
        :meth:`_ensure_tools`), under the same lock, so it happens once:
        a later call, and the first run, find it done. Its exception
        propagates, also the ``RuntimeError`` of a setup that failed
        earlier or of an agent that is closing. Start an agent whose runs
        a caller may cancel before its first run: a cancellation during
        the setup fails it for the agent's life.
        """
        await self._ensure_tools()

    def list_tools(self) -> list[ToolInfo]:
        """The tools registered now, sorted by name.

        ``source`` is ``"builtin"`` for a built-in tool, ``"provided"`` for
        a tool of the tool providers, which appear once :meth:`start` or
        the first run has set them up, and ``"custom"`` for a tool
        registered with the agent's executor and tool registry directly.
        ``tier`` is the permission tier the checker derives for it.
        """
        tools: list[ToolInfo] = []
        for tool in sorted(self.hippo.list_tools(), key=lambda t: t.name):
            source: Literal["builtin", "provided", "custom"] = "custom"
            if tool.name in self._provided_names:
                source = "provided"
            elif tool.name in self._builtin_names:
                source = "builtin"
            tools.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    parameters=copy.deepcopy(tool.parameters),
                    source=source,
                    requires_approval=self.executor.requires_approval(tool.name),
                    tier=self.checker.tier(tool),
                )
            )
        return tools

    async def run_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        approval: ApprovalPolicy | None = None,
    ) -> ToolResult:
        """Run one call of *tool_name* with *arguments*, outside a run.

        The call goes through the checks a model's call goes through, apart
        from the offered tools, so any registered tool can run: the
        registry, the arguments, the pre-execute check, the workspace
        policy, *approval* for a tool that requires it (see
        :meth:`run_task`), the tool's timeout and the result check. The
        result is the checked one, with a denial or any other refusal as
        its error. The call gets a fresh id, the policy's request has an
        empty ``run_id``, and the call emits no event and writes no memory
        or history. :meth:`start` runs first.

        Raises ``RuntimeError`` during a run, and ``TypeError`` when
        *approval* has no ``async def decide``.
        """
        if self._active_runs:
            raise RuntimeError("Cannot run a tool while a run is active")
        _check_approval(approval)
        await self.start()
        call = ToolCall(
            tool_name=tool_name, arguments=dict(arguments), id=new_tool_call_id()
        )
        run = _RunState(task_id="", approval=approval)
        result, _ = await self._do_execute(call, allowed=None, run=run)
        return result

    def conversation(self) -> list[Message]:
        """A copy of the working-memory window, oldest message first."""
        return copy.deepcopy(self.hippo.working.get_window())

    def load_conversation(self, messages: Sequence[Message]) -> int:
        """Replace the working-memory window with copies of *messages*.

        Working memory and its reflection are cleared, then *messages* go
        in by unit (see :func:`~localmelo.melo.memory.short.group_turns`):
        an assistant message with tool calls together with tool messages
        that answer each of its calls exactly once, an empty reply's label
        together with the nudge after it, and each other user, assistant
        or system message on its own. Any other unit, such as a tool call
        with a result missing or extra, or a tool message without its call,
        is dropped with a WARNING. Working memory evicts its oldest units
        as usual. Returns the number of messages kept; ``[]`` clears the
        conversation.

        Raises ``RuntimeError`` during a run, and ``TypeError``, before
        anything is cleared, for an item that is not a
        :class:`~localmelo.melo.schema.Message`.
        """
        if self._active_runs:
            raise RuntimeError("Cannot load a conversation while a run is active")
        loaded = list(messages)
        for msg in loaded:
            if not isinstance(msg, Message):
                raise TypeError(
                    f"load_conversation takes Message objects, got {type(msg).__name__}"
                )
        working = self.hippo.working
        working.clear_all()
        for unit in group_turns(copy.deepcopy(loaded)):
            head = unit[0]
            if head.role == "assistant" and head.tool_calls:
                if _answers_every_call(head, unit[1:]):
                    working.append_turn(head, unit[1:])
                    continue
                reason = "a tool call whose results do not answer each call once"
            elif len(unit) > 1:  # an empty reply's label and its nudge
                working.append_turn(head, unit[1:])
                continue
            elif head.role in ("user", "assistant", "system"):
                working.append(head)
                continue
            elif head.role == "tool":
                reason = "a tool result without its call"
            else:
                reason = f"a message with the role {head.role!r}"
            logger.warning(
                "Dropped %d message(s) from the loaded conversation: %s",
                len(unit),
                reason,
            )
        return len(working.get_window())

    # ── Tool providers ──

    async def _ensure_tools(self) -> None:
        """Start the tool providers and register their tools, once.

        The first call starts every provider in order and collects the
        tools they return. Only when all of them have started, and no
        provided tool's name is taken by a registered tool or by another
        provided tool, are the tools registered with the tool registry and
        the executor (with each tool's ``timeout_ms`` and
        ``requires_approval``), so a failed setup registers nothing. A
        taken name raises ``ValueError``; a failure of a provider's
        ``start()`` propagates unchanged. On any failure,
        cancellation included, every provider whose ``start()`` was called
        is closed before the exception propagates, so none is left
        running. A cancellation that arrives during this clean-up waits
        for it to finish and then propagates in place of the failure (see
        :meth:`_close_providers`). A cancellation during a ``start()`` is
        passed on to it, and the clean-up that ``start()`` does itself is
        waited for the same way (see :meth:`_start_provider`); after
        :meth:`begin_exit`, both waits keep to the exit's time limit.

        Setup is not retried, because the providers it closed may not
        start again: after a failure every call raises ``RuntimeError``
        naming that failure. Calls are serialized, so no provider starts
        twice, and none starts once :meth:`close` has begun.
        """
        if self._tools_ready or not self._tool_providers:
            return
        async with self._tools_lock:
            if self._tools_ready:
                return
            if self._tools_failure is not None:
                raise RuntimeError(
                    f"Tool setup failed in an earlier run: {self._tools_failure}"
                )
            started: list[BaseToolProvider] = []
            try:
                provided: list[ProvidedTool] = []
                for provider in self._tool_providers:
                    self._check_not_closing()
                    started.append(provider)
                    provided.extend(await self._start_provider(provider))
                # close() may have begun while the last provider started.
                self._check_not_closing()
                self._check_tool_names(provided)
            except BaseException as exc:
                self._tools_failure = str(exc) or _exception_text(exc)
                await self._close_providers(started)
                raise
            for tool in provided:
                # The checker's tier only adds caution: require_approval
                # can force a read tool to ask, but false cannot un-mark a
                # write, exec or network one.
                self.hippo.register_tool(tool.definition)
                self.executor.register(
                    tool.definition.name,
                    tool.function,
                    timeout_ms=tool.timeout_ms,
                    requires_approval=(
                        tier_requires_approval(tier_of(tool.definition))
                        or tool.requires_approval
                    ),
                )
            self._provided_names = frozenset(tool.definition.name for tool in provided)
            self._tools_ready = True

    async def _start_provider(self, provider: BaseToolProvider) -> list[ProvidedTool]:
        """Return ``await provider.start()``, with an exit's limit on its clean-up.

        A cancellation is passed on to ``start()``, and the clean-up that
        ``start()`` then does itself (an MCP provider closes the servers it
        started, with their grace periods) is waited for before the
        cancellation propagates, as :meth:`_close_providers` waits for a
        ``close()``. After :meth:`begin_exit`, that waiting ends when only
        :data:`EXIT_SAVE_RESERVE_S` of the exit's time is left: the provider
        is killed and its ``start()`` left to end in the background, so the
        task record can still be saved in time.
        """
        # shield(): a cancellation reaches start() only through the cancel()
        # below, once this coroutine has caught it.
        starting = asyncio.ensure_future(provider.start())
        try:
            return await asyncio.shield(starting)
        except asyncio.CancelledError:
            starting.cancel()
            await self._wait_for_provider(starting, provider)
            raise

    def _check_not_closing(self) -> None:
        if self._closing:
            raise RuntimeError("Cannot start the tool providers: the agent is closed")

    def _check_tool_names(self, provided: Sequence[ProvidedTool]) -> None:
        """Raise ``ValueError`` if a provided tool's name is already taken."""
        names: set[str] = set()
        for tool in provided:
            name = tool.definition.name
            if self.hippo.get_tool(name) is not None:
                raise ValueError(
                    f"Provided tool {name!r} collides with a registered tool"
                )
            if name in names:
                raise ValueError(f"More than one provided tool is named {name!r}")
            names.add(name)

    async def _close_providers(self, providers: Iterable[BaseToolProvider]) -> None:
        """Close each of *providers* in turn; a failure is logged, not raised.

        A cancellation does not cut this short: the close() it interrupts
        runs to its end, the remaining providers are still closed, and the
        cancellation is raised once they all are. Otherwise a run's
        ``timeout_s`` expiring here could leave a provider half shut down
        and the rest not closed at all, with their processes running.

        Once :meth:`begin_exit` has been called, the waiting ends when only
        :data:`EXIT_SAVE_RESERVE_S` of the exit's time is left: from then
        on a provider that has not closed is killed instead
        (:meth:`BaseToolProvider.kill`), and its close() is left to finish
        in the background (see :meth:`_wait_for_provider`).
        """
        cancelled: asyncio.CancelledError | None = None
        for provider in providers:
            closing = asyncio.create_task(self._close_provider(provider))
            cancelled = await self._wait_for_provider(closing, provider) or cancelled
        if cancelled is not None:
            raise cancelled

    async def _wait_for_provider(
        self, call: asyncio.Future[Any], provider: BaseToolProvider
    ) -> asyncio.CancelledError | None:
        """Wait for *call*, a ``start()`` or ``close()`` of *provider*, to end.

        A cancellation does not cut the waiting short: it is returned, for
        the caller to raise once it is done. Without an exit the waiting has
        no limit. After :meth:`begin_exit` it ends when only
        :data:`EXIT_SAVE_RESERVE_S` of the exit's time is left: *provider*
        is then killed (:meth:`BaseToolProvider.kill`) and *call* is left
        to end in the background.
        """
        cancelled: asyncio.CancelledError | None = None
        while not call.done():
            try:
                # wait() neither cancels the call nor raises its outcome.
                await asyncio.wait([call], timeout=self._close_time_left())
            except asyncio.CancelledError as exc:
                cancelled = exc
                continue  # begin_exit() may have set a deadline meanwhile
            if not call.done():  # only with an exit: out of time
                self._kill_provider(provider)
                self._abandoned.add(call)
                call.add_done_callback(self._abandoned.discard)
                break
        return cancelled

    def _close_time_left(self) -> float | None:
        """Seconds closing may still wait for a provider; ``None``: no limit."""
        if self._exit is None:
            return None
        return self._exit.time_left(keep=EXIT_SAVE_RESERVE_S)

    def _kill_provider(self, provider: BaseToolProvider) -> None:
        name = type(provider).__name__
        logger.warning("Tool provider %s did not close in time; killing it", name)
        try:
            provider.kill()
        except Exception as exc:
            logger.warning(
                "Could not kill tool provider %s: %s", name, _exception_text(exc)
            )

    async def _close_provider(self, provider: BaseToolProvider) -> None:
        try:
            await provider.close()
        except Exception as exc:
            # Not a crash: one line, no traceback (the CLI prints tracebacks
            # only for crashed runs).
            logger.warning(
                "Could not close tool provider %s: %s",
                type(provider).__name__,
                _exception_text(exc),
            )

    # ── Stage helpers ──
    # Each helper owns one phase of the agent loop.  They return a
    # short status string when the loop should break, or None to
    # continue to the next stage.

    async def _fit_prompt_cap(self, others: list[Message], query: str) -> None:
        """Drop the oldest working memory that the planning prompt has no room for.

        ``Checker.pre_plan`` refuses a step whose prompt, the working-memory
        window plus *others* (long-term recall and the reflection), is
        over its cap. Units of the window (see :func:`group_turns`) are
        dropped oldest first and whole, so a tool call keeps its results,
        until the prompt fits; the newest unit always stays. The window is
        measured as the planning request carries it: once the unit that
        holds the task *query* is dropped, the request puts the query back
        in front (see :func:`_as_planned`), so dropping that unit saves
        nothing. When even the newest unit alone, with the query, does not
        fit, nothing is dropped and pre_plan refuses the step as before.

        Each turn fits half the cap (see :meth:`_checked_tool_turn`), but
        several large turns, of one run or of earlier runs on this agent,
        can pass it together. Without this, the task would fail with
        "Prompt too large", and so would each later run on this agent,
        until eviction by message count caught up.
        """
        units = group_turns(self.hippo.working.get_window())
        for first in range(len(units)):
            kept = [msg for unit in units[first:] for msg in unit]
            if (await self.checker.pre_plan(others + _as_planned(kept, query))).allowed:
                break
        else:
            return  # not even the newest unit fits: keep the window
        if first:
            dropped = self.hippo.working.drop_oldest_units(first)
            logger.info(
                "Dropped the %d oldest working-memory messages so that the "
                "planning prompt fits",
                dropped,
            )

    async def _do_retrieval(
        self, query: str
    ) -> tuple[list[Message], list[Message], list[ToolDef], str | None]:
        """Stage 1+2: retrieve context, resolve tools, run boundary checks.

        With ``tool_exposure="retrieve"`` the tools are every registered
        built-in, sorted by name, then the other tools that the model's
        tool calls and BM25 resolve among themselves, also sorted by name;
        with ``"all"`` they are every registered tool, sorted by name.

        Before anything reads the working-memory window, its oldest units
        are dropped when the planning prompt would not fit
        ``Checker.pre_plan``'s cap otherwise (see :meth:`_fit_prompt_cap`).

        Returns (long_context, short_window, tools, fail_reason).
        *fail_reason* is non-None when a check fails and the loop must break.
        """
        long_context = await self.hippo.retrieve_context(query)
        reflection_msgs = _serialize_reflections(self.hippo.working.get_reflections())
        await self._fit_prompt_cap(long_context + reflection_msgs, query)
        short_window = self.hippo.working.get_window()

        tool_hints: list[str] = []
        if self.options.tool_exposure == "all":
            # The same list on every step, so the tools part of the request
            # stays byte-stable for prefix caching.
            tools = sorted(self.hippo.list_tools(), key=lambda t: t.name)
        else:
            # Every registered built-in, sorted by name, on every step. Hints
            # and BM25 choose among the other tools only, which follow,
            # sorted by name too: the list stays byte-identical for as long
            # as the chosen set does not change, whatever the hint order.
            builtins = builtin_names()
            # Hints are the model's tool calls, which live in the window; the
            # long-term and reflection messages carry none.
            hint_sources = long_context + short_window + reflection_msgs
            tool_hints = [
                name
                for name in self.hippo.extract_tool_hints(hint_sources)
                if name not in builtins
            ]
            tools = sorted(
                (t for t in self.hippo.list_tools() if t.name in builtins),
                key=lambda t: t.name,
            )
            tools += sorted(
                self.hippo.resolve_tools(query, hints=tool_hints, exclude=builtins),
                key=lambda t: t.name,
            )

        resolution_check = self.checker.check_tool_resolution(
            ToolResolutionResult(
                query=query,
                hints=tool_hints,
                resolved_tool_names=[t.name for t in tools],
            )
        )
        if not resolution_check.allowed:
            return (
                long_context,
                short_window,
                tools,
                (f"Tool resolution failed: {resolution_check.reason}"),
            )

        # As the planning request will carry it: with the query put back.
        all_msgs = long_context + _as_planned(short_window, query) + reflection_msgs
        check = await self.checker.pre_plan(all_msgs)
        if not check.allowed:
            return (
                long_context,
                short_window,
                tools,
                (f"Plan check failed: {check.reason}"),
            )

        return long_context, short_window, tools, None

    async def _do_plan(
        self,
        long_context: list[Message],
        short_window: list[Message],
        tools: list[ToolDef],
        query: str,
        reflections: list[ReflectionEntry] | None = None,
    ) -> tuple[Message, str | None]:
        """Stage 3: LLM planning step with post-plan check.

        Returns (response, fail_reason).
        """
        response = await self.chat.plan_step(
            context=long_context,
            short=short_window,
            tools=tools,
            query=query,
            reflections=reflections,
        )

        check = await self.checker.post_plan(response)
        if not check.allowed:
            return response, f"Response check failed: {check.reason}"

        return response, None

    async def _do_execute(
        self, call: ToolCall, *, allowed: Collection[str] | None, run: _RunState
    ) -> tuple[ToolResult, int]:
        """Stage 4: execute one tool call and validate the result.

        *allowed* holds the names of the tools offered in the planning
        call that produced *call*; the executor refuses any other tool.
        ``None``, from :meth:`run_tool`, lets any registered tool run.

        Uses the sanitized payload from the executor-result checker
        when truncation or modification was applied; falls back to
        the raw outcome otherwise.

        Returns the result and the number of characters by which the
        check shortened its output (0 when it did not), so that working
        memory can count them when it cuts the result further (see
        :func:`_cut_to`).

        With an approval policy (``run.approval``), the executor asks it
        about a call of a tool that requires approval, through an
        :class:`_ApprovalGate`; without one, no gate is built, and such a
        tool runs as any other does. A denial, or a policy that fails,
        refuses the call, and the model reads why as the call's error.

        With an event sink, *run* reports ``tool.started`` before the call
        is handed to the executor and one ``tool.finished`` after: with the
        checked result, or, when an exception ends the call, before the
        exception propagates, with the status ``cancelled`` for a
        cancellation of the run (see :func:`_is_cancellation`), its
        ``timeout_s`` included, and ``crashed`` for any other exception. A
        cancelled call's error says that it did not start when the policy
        was still deciding about it, and that it may have partly run
        otherwise. ``tool.finished`` carries the policy's answer whenever
        the policy was asked. The call runs with a new ``TOOL_EXIT_CODE``
        list, so a tool can record its child's exit status for the event.
        """
        events = run.events
        events.tool_started(call)
        exit_codes: list[int] | None = [] if events.active else None
        slot = TOOL_EXIT_CODE.set(exit_codes) if exit_codes is not None else None
        gate = None
        if run.approval is not None:
            called_tool = self.hippo.get_tool(call.tool_name)
            tier: ToolTier = (
                "write" if called_tool is None else self.checker.tier(called_tool)
            )
            gate = _ApprovalGate(
                run.approval, run, call, self.executor.workspace_root, tier
            )
        try:
            exec_request = ExecutionRequest(
                tool_name=call.tool_name,
                arguments=dict(call.arguments),
            )
            outcome = await self.executor.execute_structured(
                exec_request, allowed=allowed, approve=gate
            )

            exec_result_check = self.checker.check_executor_result(
                ExecutorResultPayload(
                    tool_name=outcome.tool_name,
                    output=outcome.output,
                    error=outcome.error,
                    duration_ms=outcome.duration_ms,
                    images=list(outcome.images),
                )
            )
            if exec_result_check.sanitized_payload is not None:
                sp = exec_result_check.sanitized_payload
                result = ToolResult(
                    tool_name=sp.tool_name,
                    output=sp.output,
                    error=sp.error,
                    duration_ms=sp.duration_ms,
                    images=list(sp.images),
                )
                # The model reads the error of a failed call, which the check
                # never cuts, and the output otherwise.
                cut = 0 if sp.error else max(len(outcome.output) - len(sp.output), 0)
            else:
                result, cut = outcome.to_tool_result(), 0
        except BaseException as exc:
            events.tool_stopped(
                call,
                exc,
                cancelled=_is_cancellation(exc),
                exit_codes=exit_codes,
                not_started=run.waiting_call_id == call.id,
                approval=None if gate is None else gate.asked,
            )
            raise
        finally:
            if slot is not None:
                TOOL_EXIT_CODE.reset(slot)
        events.tool_finished(
            call,
            outcome,
            result,
            exit_codes=exit_codes,
            approval=None if gate is None else gate.asked,
        )
        run.note(
            lambda trace: trace.tool(
                call.tool_name, call.arguments, outcome.status.value, result.output
            )
        )
        return result, cut

    async def _do_memorize(self, task_id: str, step: _InflightStep) -> None:
        """Stage 5: record the step and write checked results to memory.

        *step* holds the planning reply and, for each of its calls, the
        checked result and the characters the executor-result check cut
        from it.

        - History gets one StepRecord per executed call (the assistant
          text is attached to the first), unconditionally.
        - Long-term memory gets each record's text summary, if the checker
          allows the write.
        - Working memory always gets the real protocol turn: the assistant
          message with its tool calls, then one ``role="tool"`` message per
          call paired by ``tool_call_id``, appended as one unit. Checked
          writes may shorten it but never leave a call or result out (see
          :meth:`_checked_tool_turn`, which gets ``step.cut_before``).

        History and working memory get the same copies of the results,
        whose images are references without the base64 payload (see
        :func:`_image_references`), so no payload outlives the step.

        Each record joins ``step.records`` before it is written, and
        ``step.stored`` counts it once ``store_step`` has returned, so a
        cancellation in between leaves what is missing from history known
        (see :meth:`_save_interrupted_step`). Once every record is written,
        the run's call recorder, when set, lists their ids in order on the
        planning reply's ``llm_calls`` row (``CallLog.link_steps``).
        Appending the turn is the last statement and is synchronous:
        nothing comes between it and the caller's clearing of
        ``run.inflight``.
        """
        response = step.response
        calls = response.tool_calls
        kept = [_image_references(result) for result in step.results]
        for index, (call, result) in enumerate(zip(calls, kept, strict=True)):
            record = StepRecord(
                thought=response.content if index == 0 else "",
                tool_call=call,
                tool_result=result,
            )
            step.records.append(record)
            summary = await self.hippo.store_step(task_id, record)
            step.stored += 1

            # Checked write: step summary -> long-term memory only
            mem_check = self.checker.check_memory_write(
                MemoryWritePayload(
                    text=summary,
                    role="assistant",
                    metadata={"step_id": record.step_id},
                )
            )
            if mem_check.allowed:
                await self.hippo.memorize(
                    summary, metadata={"step_id": record.step_id}, to_working=False
                )

        recorder = CALL_RECORDER.get()
        if recorder is not None:
            # Every record is written: the planning reply's row lists them.
            await recorder.link_steps(
                [record.step_id for record in step.records[: step.stored]]
            )

        assistant_msg, tool_msgs = self._checked_tool_turn(
            response, kept, step.cut_before
        )
        self.hippo.working.append_turn(assistant_msg, tool_msgs)

    def _memory_write_allowed(self, text: str, role: str) -> bool:
        check = self.checker.check_memory_write(
            MemoryWritePayload(text=text, role=role)
        )
        return check.allowed

    def _checked_calls(self, calls: Sequence[ToolCall]) -> list[ToolCall]:
        """Return the working-memory copy of one turn's tool calls.

        ``Checker.check_memory_write`` sees the calls together, as one
        rendered text. While it rejects them (oversized arguments), the
        largest arguments are replaced first by an
        ``{"_elided": "<n> chars withheld"}`` placeholder. Ids are kept, so
        every result still pairs with its call, and the original calls,
        which history holds, are not modified. Calls still rejected with
        every argument elided (thousands of calls, or a huge tool name) are
        kept anyway: a call that already ran must stay visible.
        """
        kept = list(calls)
        largest_first = sorted(
            range(len(kept)),
            key=lambda i: len(_render_arguments(kept[i].arguments)),
            reverse=True,
        )
        for index in largest_first:
            if self._memory_write_allowed(_render_tool_calls(kept), "assistant"):
                break
            kept[index] = _elide_arguments(kept[index])
        return kept

    def _checked_tool_turn(
        self,
        response: Message,
        results: Sequence[ToolResult],
        cut_before: Sequence[int],
    ) -> tuple[Message, list[Message]]:
        """Build the working-memory copy of one tool-call turn.

        Every part goes through ``Checker.check_memory_write``. Oversized
        arguments are elided (see :meth:`_checked_calls`), rejected
        assistant text is dropped while its tool calls are kept, and a
        tool result too long for working memory is cut (see below). A
        result the check rejects for any other reason is replaced by a
        short notice bound to the same ``tool_call_id``. No call or result
        is ever left out, so the turn stays protocol-valid and the model
        still sees every call it made and what each returned. A result's
        images (references, see :func:`_image_references`) are checked
        with its text, cut as it will be kept. A result replaced by a
        notice keeps no images on its message, but its message still
        ends with the line that names them (see
        :func:`_tool_message_content`); a cut result keeps its images.

        The whole turn also shares one budget of ``MAX_MEMORY_TEXT_LEN``
        characters, half the 100,000 at which ``Checker.pre_plan`` fails
        the task, so that one step leaves room in the next planning
        prompt. The calls keep their share, and so does the assistant
        text when the check allows it and it has at most
        ``ASSISTANT_TEXT_SHARE`` characters. The results share the rest,
        the largest cut first to the room the others leave it, ending
        with a marker that says how many characters were left out (see
        :func:`_within_turn_budget`); for a result whose output the
        executor-result check had cut, the count includes the characters
        that check left out (*cut_before*, one count per result, see
        :func:`_cut_to`). This is also how a result longer
        than one memory write allows is shortened. A result left less
        than ``WITHHELD_PREVIEW_CHARS`` of room is replaced by the notice
        instead. A longer text gets no share: it could not fit beside
        results that had to be cut, so room kept for it would only cut
        them further. A text that does not fit beside the results is
        dropped, and history keeps it. Notices, and results shorter than
        one, cannot shrink further, so a turn whose calls nearly fill the
        budget, or a turn of hundreds of calls, can still pass it.
        History keeps each result as the executor-result check left it.

        The lines that name images are added after the check and the
        budget and count toward neither, so they never shorten a result
        that fits without them. Each is at most a few hundred characters,
        so a turn can pass the budget by that much per result with images.
        """
        calls = self._checked_calls(response.tool_calls)
        calls_text = _render_tool_calls(calls)
        content, thinking = response.content, response.thinking
        full_text = "\n".join(part for part in (content, calls_text) if part)
        text_allowed = self._memory_write_allowed(full_text, "assistant")
        # The text's share includes the line break that joins it to the calls.
        text_share = 0
        if text_allowed and len(content) <= ASSISTANT_TEXT_SHARE:
            text_share = len(full_text) - len(calls_text)

        texts: list[str] = []
        withheld: set[int] = set()
        for index, result in enumerate(results):
            text = _tool_result_text(result)
            # A text too long for one write is checked as the budget below
            # keeps it: cut. A memory write needs text: an image-only
            # result is checked with the line that names its images.
            checked = _cut_to(text, MAX_MEMORY_TEXT_LEN, cut_before=cut_before[index])
            check = self.checker.check_memory_write(
                MemoryWritePayload(
                    text=checked or describe_images(result.images),
                    role="tool",
                    images=list(result.images),
                )
            )
            if not check.allowed:
                text = _withheld_notice(text, check.reason)
                withheld.add(index)
            texts.append(text)
        texts = _within_turn_budget(
            texts,
            withheld,
            len(calls_text),
            limit=MAX_MEMORY_TEXT_LEN - text_share,
            cut_before=cut_before,
        )

        turn_chars = len(full_text) + sum(len(text) for text in texts)
        if turn_chars > MAX_MEMORY_TEXT_LEN or not text_allowed:
            content, thinking = "", ""
        assistant_msg = Message(
            role="assistant", content=content, tool_calls=calls, thinking=thinking
        )

        tool_msgs = [
            Message(
                role="tool",
                content=_tool_message_content(text, result.images),
                tool_call_id=call.id,
                tool_result=result,
                images=[] if index in withheld else result.images,
            )
            for index, (call, result, text) in enumerate(
                zip(calls, results, texts, strict=True)
            )
        ]
        return assistant_msg, tool_msgs

    # ── Step estimation (kept for backward compat, no longer called by run) ──

    async def _estimate_max_steps(self, query: str) -> int:
        """Ask the LLM for a conservative step-count estimate.

        The result is clamped to [MIN_AGENT_STEPS, MAX_AGENT_STEPS].
        On parse failure the hard ceiling is used as a safe fallback.
        """
        estimate = await self.chat.estimate_steps(query)
        if estimate < 0:
            return MAX_AGENT_STEPS
        return max(MIN_AGENT_STEPS, min(estimate, MAX_AGENT_STEPS))

    # ── Stuck detection ──

    @staticmethod
    def _make_fingerprint(
        calls: Sequence[ToolCall], results: Sequence[ToolResult]
    ) -> tuple[str, str, str]:
        """Return one ``(tool_names, step_hash, errors)`` fingerprint per step.

        ``step_hash`` covers each call's name and arguments together with
        what the call returned: its output, its error and the digests of
        its images (see :func:`_result_digest`). A step with several calls
        is fingerprinted as a whole: calls are sorted (the same batch in
        another order matches) and their errors joined. ``_detect_stuck``
        therefore fires when the same batch returns the same results, or
        the same errors repeat, across steps — not merely because one step
        contains several calls, and not when repeated polling returns
        new results.
        """
        entries = sorted(
            (
                tc.tool_name,
                json.dumps(tc.arguments, sort_keys=True, default=str),
                _result_digest(result),
            )
            for tc, result in zip(calls, results, strict=True)
        )
        return (
            ",".join(name for name, _, _ in entries),
            hashlib.sha256(json.dumps(entries).encode()).hexdigest()[:16],
            "\n".join(sorted(r.error for r in results if r.error)),
        )

    @staticmethod
    def _detect_stuck(
        fingerprints: list[tuple[str, str, str]],
    ) -> str:
        """Return a failure_type string if stuck heuristics fire, else ''."""
        if len(fingerprints) < 3:
            return ""
        last3 = fingerprints[-3:]
        # The same calls with the same results 3 times
        keys = {(name, step_h) for name, step_h, _ in last3}
        if len(keys) == 1 and last3[0][0]:
            return "stuck"
        # Same error repeated 3 times
        errors = [err for _, _, err in last3 if err]
        if len(errors) == 3 and len(set(errors)) == 1:
            return "stuck"
        return ""

    # ── Reflection ──

    async def _do_reflect(
        self,
        task: TaskRecord,
        attempt_id: int,
        failure_type: str,
        *,
        final: bool = False,
        replies: list[Message] | None = None,
    ) -> tuple[ReflectionEntry, ReflectionDecision] | None:
        """Structured reflection at attempt boundary.

        *final* says that no further attempt can start; the request then
        tells the model so (see :meth:`Chat.reflect`). *replies*, when
        given, receives each reply as it comes, one or two, so the caller
        can report how many requests the reflection took and their usage,
        also when no reflection comes of them.

        A reply that holds no reflection, often a tool call, is asked for
        once more, with the window's turns written out as text and
        :data:`REFLECTION_RETRY_NOTE` at the end (see :meth:`Chat.reflect`).
        What counts as a reflection, and how one whose serialized form
        exceeds :data:`MAX_REFLECTION_CHARS` is kept to that budget,
        depends on ``reflection_gate``:

        - ``"flags"``: a JSON object with a valid ``recommended_action``
          (see :func:`_find_reflection`), so a reply whose only JSON is
          something else, such as a tool call written out as text, is
          asked for again too. When the second reply holds no reflection
          either, this returns ``None``: the boundary has no reflection
          (see :meth:`_run_reflective_attempts`). A reflection over the
          budget is trimmed to fit (see :func:`_fit_reflection`),
          whatever its decision: the decision and its rationale stay the
          model's, and its ``tried_memory`` is the trimmed entry.
        - ``"utility"``, the behavior from before that option: any JSON
          object (see :func:`_holds_reflection_object`). The second reply
          is parsed whatever it holds, so one that still has no object
          falls back to ``stop`` (see :meth:`_parse_reflection`). A
          reflection over the budget forces
          ``decision.recommended_action = "decompose"``, with a note in
          the rationale, so the next attempt does not start. The model is
          meant to keep the reflection compact — when it cannot, the
          agent stops chasing instead of carrying an ever-growing
          scratchpad. A ``done`` decision is left as it is: it ends the
          run, so no attempt would carry the reflection.
        """
        flags = self.options.reflection_gate == "flags"
        short_window = self.hippo.working.get_window()
        prior_reflections = self.hippo.working.get_reflections()
        response = await self.chat.reflect(
            short=short_window,
            query=task.query,
            attempt_id=attempt_id,
            failure_type=failure_type,
            prior_reflections=prior_reflections or None,
            final=final,
        )
        if replies is not None:
            replies.append(response)
        if flags:
            readable = _find_reflection(response.content) is not None
        else:
            readable = _holds_reflection_object(response.content)
        if not readable:
            # Often a tool call: the window ends in tool turns, and some
            # models continue them although the request offers no tools.
            # Ask once more, with those turns written out as text.
            response = await self.chat.reflect(
                short=short_window,
                query=task.query,
                attempt_id=attempt_id,
                failure_type=failure_type,
                prior_reflections=prior_reflections or None,
                final=final,
                retry=True,
            )
            if replies is not None:
                replies.append(response)
        if flags:
            data = _find_reflection(response.content)
            if data is None:
                return None
            entry, decision = self._reflection_from_object(
                data, attempt_id, failure_type
            )
            entry = _fit_reflection(entry, MAX_REFLECTION_CHARS)
            decision.tried_memory = entry
            return entry, decision
        entry, decision = self._parse_reflection(
            response.content, attempt_id, failure_type
        )
        serialized = _serialize_reflections([entry])
        size = serialized[0].content.__len__() if serialized else 0
        if size > MAX_REFLECTION_CHARS and decision.recommended_action != "done":
            decision.recommended_action = "decompose"
            note = f" (reflection budget exceeded: {size} > {MAX_REFLECTION_CHARS})"
            decision.rationale = (decision.rationale or "") + note
        return entry, decision

    @staticmethod
    def _parse_reflection(
        text: str,
        attempt_id: int,
        failure_type: str,
    ) -> tuple[ReflectionEntry, ReflectionDecision]:
        """Parse LLM reflection JSON with strict coercion and conservative fallback.

        The parser of ``reflection_gate="utility"``; ``"flags"`` reads the
        object that :func:`_find_reflection` finds instead (see
        :meth:`_do_reflect`).

        Parsing is tolerant: a Markdown code fence around the reply is
        stripped, and a reply that is not JSON as a whole is read from the
        first ``{...}`` object in it (see :func:`_decode_reflection`). Only
        a reply without such an object falls back to ``stop``, with the
        rationale "Could not parse reflection response"; a reply that is
        JSON but not an object stops too. The object itself is read by
        :meth:`_reflection_from_object`.
        """
        try:
            data = _decode_reflection(_coerce_str(text))
        except ValueError:
            entry = ReflectionEntry(
                attempt_id=attempt_id,
                summary=_coerce_str(text)[:200],
                failure_type=failure_type,
            )
            return entry, ReflectionDecision(
                recommended_action="stop",
                rationale="Could not parse reflection response",
                best_effort_result=_coerce_str(text)[:500],
                tried_memory=entry,
            )

        if not isinstance(data, dict):
            entry = ReflectionEntry(
                attempt_id=attempt_id,
                summary=str(data)[:200],
                failure_type=failure_type,
            )
            return entry, ReflectionDecision(
                recommended_action="stop",
                rationale="Reflection response was not a JSON object",
                best_effort_result="",
                tried_memory=entry,
            )
        return Agent._reflection_from_object(data, attempt_id, failure_type)

    @staticmethod
    def _reflection_from_object(
        data: dict[str, Any],
        attempt_id: int,
        failure_type: str,
    ) -> tuple[ReflectionEntry, ReflectionDecision]:
        """The reflection entry and decision that the JSON object *data* gives.

        Each value is coerced to its field's type, the five estimates
        clamped to [0, 1], and a missing one takes a default. A
        ``recommended_action`` that is not one of
        :data:`REFLECTION_ACTIONS` reads ``stop``.
        """
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            summary=_coerce_str(data.get("summary")),
            failed_hypotheses=_coerce_str_list(data.get("failed_hypotheses")),
            disproven_actions=_coerce_str_list(data.get("disproven_actions")),
            useful_evidence=_coerce_str_list(data.get("useful_evidence")),
            unresolved_questions=_coerce_str_list(data.get("unresolved_questions")),
            recommended_avoids=_coerce_str_list(data.get("recommended_avoids")),
            next_promising_directions=_coerce_str_list(
                data.get("next_promising_directions")
            ),
            failure_type=failure_type or _coerce_str(data.get("failure_type")),
            confidence=_coerce_float(data.get("confidence")),
        )

        action = _coerce_str(data.get("recommended_action")) or "stop"
        if action not in REFLECTION_ACTIONS:
            action = "stop"

        decision = ReflectionDecision(
            progress_made=_coerce_bool(data.get("progress_made")),
            task_still_feasible=_coerce_bool(data.get("task_still_feasible", True)),
            new_information_gained=_coerce_bool(data.get("new_information_gained")),
            next_step_is_concrete=_coerce_bool(data.get("next_step_is_concrete")),
            next_step_is_novel=_coerce_bool(data.get("next_step_is_novel")),
            recommended_action=action,
            rationale=_coerce_str(data.get("rationale")),
            best_effort_result=_coerce_str(data.get("best_effort_result")),
            tried_memory=entry,
            # Active-learning fields (clamped to [0, 1])
            estimated_info_gain=_coerce_unit_float(data.get("estimated_info_gain")),
            estimated_cost=_coerce_unit_float(data.get("estimated_cost")),
            repeat_risk=_coerce_unit_float(data.get("repeat_risk")),
            novelty=_coerce_unit_float(data.get("novelty")),
            feasibility=_coerce_unit_float(data.get("feasibility")),
        )
        return entry, decision

    # ── Attempt ──

    async def _run_attempt(
        self, task: TaskRecord, query: str, budget: int, run: _RunState
    ) -> tuple[int, str]:
        """Run one attempt of the agent loop.

        Returns ``(steps_used, failure_type)``.
        *failure_type* is ``""`` when the attempt ended the run,
        ``"budget"`` when the step budget was exhausted, or ``"stuck"``
        when stuck detection fired. The attempt ends *run* as
        ``completed`` with a direct answer as the result, or as
        ``failed`` with a checker's refusal as the error.

        A reply with neither a tool call nor text is no answer: its step
        counts, working memory records it as :data:`EMPTY_REPLY_LABEL`,
        an assistant message, followed by :data:`EMPTY_REPLY_NUDGE` as a
        user message, and the model plans again. The label keeps the
        roles alternating, since some chat templates (Mistral, Gemma)
        reject two user messages in a row, such as the query and the
        nudge. Neither message goes to history, and neither counts as a
        reply of the model: the latest text, the empty-reply count and
        stuck detection only see what the model returned. The
        :data:`MAX_EMPTY_REPLIES`-th empty reply in a row in this run
        (reflections in between do not break the row) ends it as
        ``failed`` with the error ``empty replies``.
        """
        fingerprints: list[tuple[str, str, str]] = []
        reflections = self.hippo.working.get_reflections() or None

        for step in range(budget):
            # Retrieval + tool resolution + boundary checks
            long_context, short_window, tools, fail = await self._do_retrieval(query)
            if fail is not None:
                run.end("failed", error=fail)
                return step + 1, ""

            # LLM planning step (with reflection context). The step counts
            # once the planning call is made, even if it then raises.
            run.steps += 1
            run.events.step_started(run.steps, run.attempts)
            run.note(lambda trace: trace.step(run.steps, run.attempts))
            response, fail = await self._do_plan(
                long_context,
                short_window,
                tools,
                query,
                reflections=reflections,
            )
            # Also a reply the post-plan check refuses.
            run.events.assistant_message(response)
            if fail is not None:
                run.end("failed", error=fail)
                return step + 1, ""
            if response.content.strip():
                run.last_text = response.content

            if not response.tool_calls:
                if response.content.strip():
                    # No tool call → direct answer
                    run.end("completed", result=response.content)
                    return step + 1, ""
                # An empty reply is no answer: ask for one and plan again.
                run.empty_replies += 1
                if run.empty_replies >= MAX_EMPTY_REPLIES:
                    run.end("failed", error="empty replies")
                    return step + 1, ""
                # The label stands in for the reply, so the nudge does not
                # follow another user message; as one unit, the two are
                # only ever dropped together.
                self.hippo.working.append_turn(
                    Message(role="assistant", content=EMPTY_REPLY_LABEL),
                    [Message(role="user", content=EMPTY_REPLY_NUDGE)],
                )
                run.events.empty_reply(run.empty_replies)
                continue
            run.empty_replies = 0

            # Execute every call in order, each through the executor boundary,
            # which runs only the tools this planning call offered. Until the
            # step is in working memory, run.inflight holds it, so that a
            # cancellation can close it (see _on_cancelled).
            allowed = frozenset(t.name for t in tools)
            calls = list(response.tool_calls)
            inflight = run.inflight = _InflightStep(response=response)
            for call in calls:
                # Counted when handed over, so a call that timeout_s cuts
                # short counts too.
                run.n_tool += 1
                result, cut = await self._do_execute(call, allowed=allowed, run=run)
                inflight.results.append(result)
                inflight.cut_before.append(cut)
            await self._do_memorize(task.task_id, inflight)
            run.inflight = None

            # Stuck detection (one fingerprint per step)
            fingerprints.append(self._make_fingerprint(calls, inflight.results))
            stuck = self._detect_stuck(fingerprints)
            if stuck:
                run.events.stuck()
                run.note(Trace.stuck)
                return step + 1, stuck

        run.events.budget_used()
        return budget, "budget"

    # ── Main loop ──

    async def run(self, query: str) -> str:
        """Run one task and return its outcome as text.

        The text is the run's result when that is non-empty, otherwise
        ``"[<status>] <error>"``, so a run that failed never reads as an
        empty answer. :meth:`run_task` returns the whole record.
        """
        outcome = await self.run_task(query)
        return outcome.result or f"[{outcome.status}] {outcome.error}"

    async def run_task(
        self,
        query: str,
        *,
        on_event: EventSink | None = None,
        approval: ApprovalPolicy | None = None,
        observe: EventSink | None = None,
        progress: bool = False,
        routine: str | None = None,
    ) -> RunResult:
        """Run one task and report how it ended.

        The run ends with one of these statuses:

        - ``completed``: the model gave a final answer (the result), or a
          reflection decided that the task was done (its answer, see
          :meth:`_run_reflective_attempts`);
        - ``failed``: a checker refused the plan or the run (the error
          says why), or the model gave two empty replies in a row (the
          error is ``empty replies``, see :meth:`_run_attempt`);
        - ``gate_stopped``: the reflection gate decided not to continue
          while attempts and steps were left. The reflection's best
          effort, if any, is the result and its rationale the error;
        - ``budget_exhausted``: ``max_steps`` or ``max_attempts`` ran out.
          With ``attempt_policy="reflect"`` the last reflection's best
          effort, if any, is the result, or the latest planning text when
          that reflection was unreadable (``reflection_gate="flags"``, see
          :meth:`_run_reflective_attempts`);
        - ``stuck``: stuck detection fired with ``attempt_policy="none"``;
        - ``provider_error``: a model server request failed with a
          :class:`~localmelo.melo.contracts.providers.ProviderError`: a
          chat request after its retries, or an embedding request. The
          error is the provider's message; no traceback is logged;
        - ``setup_error``: the tool providers could not be set up at the
          start of the first run (see :meth:`_ensure_tools`): one failed
          to start, or a provided tool has the name of a registered tool.
          The error says which, and no provider is left running. Setup
          is not retried: every later run on this agent ends as
          ``setup_error`` too, before its first step;
        - ``timeout``: the run was still going after ``timeout_s``
          seconds. It is cancelled wherever it was: a request in flight
          is abandoned (and counts in ``llm_errors``), a shell command is
          killed with every process it started, and tool providers still
          being set up are closed (so later runs end as ``setup_error``).
          ``python_exec`` code without a ``workspace_root`` is the
          exception: it runs on the event loop, so neither ``timeout_s``
          nor the executor's timeout can interrupt it, and the run ends
          only once that code returns. With a root it runs in a child
          process, which is killed with every process it started.
          Closing the providers is not cut short, also when the deadline
          expires while a failed setup is closing them, so the run
          returns once they are closed, which can be after ``timeout_s``;
        - ``crashed``: an unexpected exception ended the run. It is
          logged and reported as the error, not raised. A cancellation
          from outside the run, or any other exception that is not an
          ``Exception``, still propagates once the task is saved. A
          cancellation after :meth:`begin_exit` has the exit's reason as
          its error, and ends the run as that method describes. All this
          holds for a run without *on_event* and *approval*; an
          interactive run that is cancelled ends ``cancelled`` instead;
        - ``cancelled``: only for an interactive run, one given *on_event*
          or *approval*: the task awaiting this method was cancelled, by
          its caller (a Stop) or by a scope such as ``asyncio.timeout``,
          a TaskGroup or ``asyncio.run`` on Ctrl+C. The result is empty.
          The error is the exit's reason after :meth:`begin_exit`, else
          the message of ``task.cancel(msg)`` when it is a non-empty
          string, else :data:`CANCELLED_ERROR`. Working memory keeps the
          query and, closed, the step whose tool calls were under way:
          each of its calls that had not returned reads
          :data:`~localmelo.melo.schema.CALL_INTERRUPTED` for the call
          under way, or :data:`~localmelo.melo.schema.CALL_NOT_STARTED`
          when *approval* was still deciding about it, and
          :data:`~localmelo.melo.schema.CALL_NOT_STARTED` for the later
          ones. The cancellation propagates once the run is
          finalized (below), so the RunResult is the one ``run.finished``
          carries. ``timeout_s`` still ends a run as ``timeout``, and a
          ``CancelledError`` that something raises while nothing cancels
          the run is a crash.

        With ``record_answers``, the run's answer is appended to working
        memory once the run has ended, before it is saved (see
        :meth:`_record_answer`), whatever the status.

        Whatever happens, the task record is saved with the status and
        ``result or error``, and the reflections are cleared (promoted
        to long-term memory first when ``memory`` is on); a failure while
        doing so is logged without a traceback and does not replace the
        run's own outcome. ``timeout_s`` bounds the run itself, not this
        final step. The save has no bound of its own; promotion has
        :data:`PROMOTION_TIMEOUT_S`, so an embedding server that stopped
        answering delays the return by that much at most. A cancellation
        for an exit (see :meth:`begin_exit`) that lands during this final
        step does not leave the record unsaved: the record, with the
        run's own status, is saved again as that method describes, and
        the cancellation then propagates.

        With persistence on, or any history that keeps model calls, every
        response of the run's model requests is also recorded, as it
        arrives, in history's ``llm_calls`` table, and each tool step's
        rows are listed on the row of the response that made them (see
        :class:`~localmelo.melo.agent._call_log.CallLog`). A record that
        fails is logged, the run's first at WARNING, and changes nothing
        of the run: not its status, result, error or counters.

        A ``cancelled`` run is finalized in a task of its own instead,
        which a second cancellation does not stop (see :meth:`_finalize`):
        history gets the rows its interrupted step is missing, one per
        call, and the task record is saved, both within
        :data:`FINALIZE_TIMEOUT_S`, and then, within what is left, the
        rows written are listed on their response's ``llm_calls`` row; the
        reflections are cleared without being promoted, so no network call
        follows the cancellation. After :meth:`begin_exit`, the finalizer
        closes the tool providers first and keeps to the exit's deadline,
        as :meth:`begin_exit` describes for other runs.

        ``RunResult.config`` reports the effective configuration: the
        options, the registered built-ins (``builtins``), the resolved
        ``memory_dir`` and every tool registered once the tool providers
        are set up (``tools``), with tool names sorted. It leaves out
        ``workspace_root``, ``record_answers`` and ``tool_timeout_s``:
        ``run.started`` and :attr:`workspace_root` give the root, and
        ``options`` the others.

        *on_event*, when given, follows the run: it is called once per
        :class:`~localmelo.melo.contracts.events.AgentEvent`,
        synchronously and in order, from the task that runs the run, or,
        once a cancellation has ended the run, from its finalizer's task.
        The first event is ``run.started`` and the last ``run.finished``,
        which carries the RunResult's ``to_dict()`` and also comes when
        this method raises; :mod:`localmelo.melo.contracts.events` lists
        the others. An event on which the sink raises an ``Exception`` is
        lost, and the run goes on. The sink must not block: slow work
        belongs on a queue. The ``thinking`` and ``content`` of
        ``assistant.message`` events, and the ``rationale``, ``summary``
        and ``best_effort_result`` of ``reflection`` events, pass
        :func:`~localmelo.melo.checker.redaction.redact_secrets`, while
        working memory, history's tasks and steps and the RunResult keep
        the text as it is; history's model calls store it redacted too.
        Without a sink, no event is built.

        *approval*, an
        :class:`~localmelo.melo.contracts.approvals.ApprovalPolicy`, is
        asked about each call of a tool that requires approval: the
        built-ins ``shell_exec``, ``python_exec`` and ``file_write``, and
        each provided tool whose ``requires_approval`` is set. It is asked
        just before the call would run, once every automatic check has let
        it through, so its time counts toward ``timeout_s`` but never
        toward the tool's own timeout. A denial refuses the call, and the
        model reads why as the call's error (see
        :func:`~localmelo.melo.contracts.approvals.denial_text`); so does a
        policy that raises, returns something other than an
        ``ApprovalDecision``, or raises ``CancelledError`` while the run is
        not cancelled. Either way the run goes on. Each call is asked about
        on its own: a decision's ``remember`` is reported, not acted on, so
        a policy that grants a tool for a session keeps the grant itself.
        ``tool.finished`` reports each answer, and ``run.started`` whether
        a policy was given. Without one, a tool that requires approval runs
        as any other does.

        *observe*, when given in place of *on_event*, receives the same
        events, but does not make the run interactive: a cancelled run
        ends ``crashed``, as one without a sink does. A display that only
        watches a run, such as a terminal's progress line, uses it.

        *progress*, with *on_event* or *observe*, adds ``run.progress``
        events: one right after ``run.started``, one after each
        ``step.started``, ``tool.finished``, ``reflection.started`` and
        ``stuck`` notice, and one whose ``state`` is ``ended`` right
        before ``run.finished``, on every path. Each says where the run
        stands: facts (step, attempt, phase, test counts), and, from the
        run's routine as the procedural store remembers it on this model
        with these options, past runs' facts and named steps (see
        :mod:`localmelo.melo.memory.procedures`). To find them, the run's
        start reads the store once, within :data:`LOOKUP_TIMEOUT_S` and
        the run's ``timeout_s``; a lookup that fails or takes longer
        leaves the run with facts only. Without *progress*, or without a
        sink, the run reads nothing at its start and reports no progress.

        *routine* names the run's routine: runs given the same name are
        one routine in any folder and whatever their wording. Without it,
        the routine is the query, with dates and times masked, in the
        folder the run's tools work in.

        Before the run starts, ``TypeError`` is raised when *on_event* or
        *observe* is not callable or is a coroutine function, when both
        are given, when *approval* has no ``async def decide``, or when
        *routine* is not a ``str``, and ``ValueError`` when *routine* is
        blank: no event is emitted, no task record is written, and working
        memory is left as it is.
        """
        check_sink(on_event)
        check_sink(observe, "observe")
        if on_event is not None and observe is not None:
            raise TypeError("pass on_event or observe, not both")
        _check_approval(approval)
        if routine is not None:
            if not isinstance(routine, str):
                raise TypeError("routine must be a str")
            if not routine.strip():
                raise ValueError("routine must not be blank")
        place = self._run_place()
        started = time.perf_counter()
        task = TaskRecord(query=query)
        run = _RunState(
            task_id=task.task_id,
            events=_Emitter(
                on_event if observe is None else observe,
                task.task_id,
                interactive=observe is None,
                progress=bool(progress),
            ),
            approval=approval,
            routine=routine,
            place=place,
            limits=self._limits() if progress else None,
        )
        stats = self.chat.reset_stats()
        timeout_s = self.options.timeout_s
        deadline = asyncio.timeout(timeout_s)  # never expires when None
        # The retry notices of this run's model requests; only with a sink.
        retries = RETRY_HOOK.set(run.events.retry) if run.events.active else None
        # Every response of this run's model requests becomes a row of
        # history's llm_calls table, when the history keeps model calls,
        # with or without a sink. Read at each run, so a history set after
        # the Agent was built records too.
        history = self.hippo.history
        recorder = CALL_RECORDER.set(
            CallLog(history, task.task_id, run)
            if isinstance(history, CallHistory)
            else None
        )
        self._active_runs += 1
        try:
            run.events.run_started(
                query,
                max_steps=self.options.max_steps,
                workspace_root=self.executor.workspace_root,
                approval=approval is not None,
            )
            try:
                async with deadline:
                    try:
                        await self._look_up(run, query)
                    finally:
                        # Also when a timeout or a stop cuts the lookup
                        # short: the first run.progress follows run.started.
                        run.report_progress()
                    await self.hippo.save_task(task)
                    try:
                        await self._ensure_tools()
                    except Exception as exc:
                        raise _ToolSetupError(str(exc) or _exception_text(exc)) from exc
                    self.hippo.working.append(Message(role="user", content=query))
                    run.query_in_memory = True
                    if self.options.attempt_policy == "none":
                        await self._run_single_attempt(task, query, run)
                    else:
                        await self._run_reflective_attempts(task, query, run)
            except _ToolSetupError as exc:
                # Reported in the result, not a crash: no traceback.
                logger.info("Task %s ended with setup_error: %s", task.task_id, exc)
                run.end("setup_error", error=str(exc))
            except ProviderError as exc:
                # Reported in the result, not a crash: no traceback.
                logger.info("Task %s ended with provider_error: %s", task.task_id, exc)
                run.end("provider_error", error=str(exc) or _exception_text(exc))
            except Exception as exc:
                if (
                    isinstance(exc, TimeoutError)
                    and timeout_s is not None
                    and deadline.expired()
                ):
                    # This run's own deadline; any other TimeoutError is a crash.
                    logger.info("Task %s timed out after %g s", task.task_id, timeout_s)
                    run.end("timeout", error=f"Timed out after {timeout_s:g} s")
                else:
                    logger.exception("Task %s crashed", task.task_id)
                    run.end("crashed", error=_exception_text(exc))
            except BaseException as exc:
                if run.interactive and _is_cancellation(exc):
                    # A caller stopped a run that it follows.
                    self._on_cancelled(query, run, exc)
                else:
                    if (
                        isinstance(exc, asyncio.CancelledError)
                        and self._exit is not None
                    ):
                        error = self._exit.reason  # cancelled for an exit
                    else:
                        error = _exception_text(exc)
                    run.end("crashed", error=error)
                raise
            finally:
                if run.running:
                    # Every path above ends the run; this keeps the record
                    # terminal should a later change miss one.
                    run.end("crashed", error="The run ended without a status")
                self._record_answer(run)
                if run.status == "cancelled":
                    # Only _on_cancelled ends a run so. Its finalizer saves the
                    # run and reports run.finished, in place of the lines below.
                    await self._finalize_in_task(task, run, stats, started)
                else:
                    try:
                        if self._exit is not None:
                            await self._finish_task_for_exit(task, run, self._exit)
                        else:
                            try:
                                await self._finish_task(task, run)
                            except asyncio.CancelledError:
                                # An exit that began during the final save
                                # cancelled it, perhaps before it was committed:
                                # save again, within the exit's time.
                                if self._exit is None:
                                    raise
                                await self._finish_task_for_exit(task, run, self._exit)
                                raise
                    finally:
                        # Built here, so that run.finished carries it on every
                        # path, also when run_task raises.
                        outcome = self._run_result(
                            task, run, stats, time.perf_counter() - started
                        )
                        run.outcome = outcome
                        run.report_progress(ended=run.status)
                        run.events.run_finished(outcome)
            await self._learn(run, query, outcome)
            return outcome
        finally:
            if run.finalizer is None:
                # Else the finalizer counts the run out when it ends.
                self._active_runs -= 1
            if retries is not None:
                RETRY_HOOK.reset(retries)
            CALL_RECORDER.reset(recorder)

    # ── Procedural memory ──

    def _run_place(self) -> str | None:
        """The folder a run's tools work in, as its routine's place.

        The workspace root, or else the working directory, as it is now
        (see :func:`~localmelo.melo.memory.procedures.normalize.run_place`);
        ``None`` when it cannot be read, such as a removed working
        directory: the run then has no routine of its own.
        """
        try:
            return run_place(self.executor.workspace_root)
        except (OSError, ValueError):
            return None

    def _routine_key(self, run: _RunState, query: str) -> RoutineKey | None:
        """The key of *run*'s routine, made at its first use; ``None`` without one."""
        if run.key is None:
            if run.routine is not None:
                run.key = key_for_routine(run.routine, query)
            elif run.place is not None:
                run.key = key_for_query(query, run.place)
        return run.key

    def _model_id(self) -> str:
        """The model id the procedural store keys on (``BaseLLMProvider.model``)."""
        model = getattr(self._llm, "model", "")
        return model if isinstance(model, str) else ""

    def _limits(self) -> Limits:
        """The budgets that ``run.progress`` reports; null where a run has none."""
        opts = self.options
        reflective = opts.attempt_policy != "none"
        return Limits(
            max_steps=opts.max_steps,
            steps_per_attempt=opts.steps_per_attempt if reflective else None,
            max_attempts=opts.max_attempts if reflective else None,
            timeout_s=opts.timeout_s,
        )

    async def _look_up(self, run: _RunState, query: str) -> None:
        """Find the template *run*'s progress rests on, as ``run.view``.

        Only for a run that reports progress (see :meth:`run_task`): one
        read of the procedural store, within :data:`LOOKUP_TIMEOUT_S`. No
        template, or a lookup that fails or takes longer, leaves the run
        at level 0; a failure is logged at INFO, without a traceback. The
        run's own cancellation propagates.
        """
        if not run.events.reports_progress:
            return
        bound = asyncio.timeout(LOOKUP_TIMEOUT_S)
        try:
            key = self._routine_key(run, query)
            if key is None:
                return
            async with bound:
                run.view = await self.hippo.procedures.lookup(
                    key,
                    model=self._model_id(),
                    options=length_options(self._run_config()),
                )
        except Exception as exc:
            if isinstance(exc, TimeoutError) and bound.expired():
                reason = f"no answer within {LOOKUP_TIMEOUT_S:g} s"
            else:
                reason = _exception_text(exc)
            logger.info("No progress template for task %s: %s", run.task_id, reason)

    async def _learn(self, run: _RunState, query: str, outcome: RunResult) -> None:
        """Record *run*, which has ended, in the procedural store.

        Its routine, its episode and, for a completed run, its template's
        next version (see :meth:`Procedures.learn
        <localmelo.melo.memory.procedures.store.Procedures.learn>`). After
        ``run.finished`` and before run_task returns, so the run's events
        and ``duration_s`` are what they would be without it. Skipped
        during an exit (see :meth:`begin_exit`), which never waits for it,
        and when the run's trace stopped following it. Bounded by
        :data:`LEARN_TIMEOUT_S`; a failure, such as a locked or damaged
        ``procedures.db``, is logged at INFO without a traceback and
        changes nothing of the run. The run's cancellation propagates.
        """
        if self._exit is not None or not run.tracing:
            return
        bound = asyncio.timeout(LEARN_TIMEOUT_S)
        try:
            async with bound:
                key = self._routine_key(run, query)
                if key is None:
                    return
                facts = RunFacts.from_trace(
                    run.trace,
                    task_id=outcome.task_id,
                    key=key,
                    model=self._model_id(),
                    options=length_options(outcome.config),
                    status=outcome.status,
                    steps=outcome.steps,
                    attempts=outcome.attempts,
                    duration_s=outcome.duration_s,
                    harness_version=_harness_version(),
                )
                await self.hippo.procedures.learn(facts, view=run.view)
        except Exception as exc:
            if isinstance(exc, TimeoutError) and bound.expired():
                reason = f"no answer within {LEARN_TIMEOUT_S:g} s"
            else:
                reason = _exception_text(exc)
            logger.info("Could not learn from task %s: %s", outcome.task_id, reason)

    def _record_answer(self, run: _RunState) -> None:
        """Append the answer of *run*, which has ended, to working memory.

        Only with ``record_answers``, and only when working memory holds
        the run's query, so the roles still alternate. Synchronous, in
        run_task's ``finally`` before every final save. The answer is one
        assistant message without tool calls: the result without its
        ``<think>`` block (see :func:`~localmelo.melo.schema.split_thinking`),
        or a label when that leaves no text (see
        :func:`~localmelo.melo.schema.answer_text`), cut to
        ``SESSION_TEXT_CHARS`` characters (see
        :func:`~localmelo.melo.schema.clip_session_text`). A text that the
        memory-write check refuses is replaced by
        :data:`~localmelo.melo.schema.ANSWER_NOT_KEPT_LABEL` with the
        check's reason. The result itself is not changed. A failure is
        logged, not raised, so it cannot replace the run's outcome or skip
        the save.
        """
        if not self.options.record_answers or not run.query_in_memory:
            return
        try:
            _, answer = split_thinking(run.result)
            text = clip_session_text(answer_text(run.status, answer))
            check = self.checker.check_memory_write(
                MemoryWritePayload(text=text, role="assistant")
            )
            if not check.allowed:
                text = ANSWER_NOT_KEPT_LABEL.format(reason=check.reason)
            self.hippo.working.append(Message(role="assistant", content=text))
        except Exception:
            logger.exception(
                "Could not keep the answer of task %s in working memory", run.task_id
            )

    async def _finish_task(self, task: TaskRecord, run: _RunState) -> None:
        """Save *task* with the run's terminal status and clear reflections.

        A failure is logged, not raised, so it cannot replace the run's
        outcome or the exception that ended it, and is logged without a
        traceback: the run did not crash because of it. Promoting the
        reflections may take at most :data:`PROMOTION_TIMEOUT_S` seconds.
        A failed embedding request (``ProviderError``) or one that the
        bound cuts short is logged at INFO, as ``provider_error`` is; any
        other failure at WARNING.
        """
        task.status = run.status
        task.result = run.result or run.error
        try:
            try:
                await self.hippo.save_task(task)
            except Exception as exc:
                logger.warning(
                    "Could not save the final state of task %s: %s",
                    task.task_id,
                    _exception_text(exc),
                )
            bound = asyncio.timeout(PROMOTION_TIMEOUT_S)
            try:
                # With memory=False Hippo has no embedding, so this only
                # clears the reflections (see __init__).
                async with bound:
                    await self.hippo.promote_reflections(task.task_id)
            except ProviderError as exc:
                logger.info(
                    "Could not promote the reflections of task %s: %s",
                    task.task_id,
                    exc,
                )
            except Exception as exc:
                if isinstance(exc, TimeoutError) and bound.expired():
                    logger.info(
                        "Could not promote the reflections of task %s: "
                        "no answer within %g s",
                        task.task_id,
                        PROMOTION_TIMEOUT_S,
                    )
                else:
                    logger.warning(
                        "Could not promote the reflections of task %s: %s",
                        task.task_id,
                        _exception_text(exc),
                    )
        finally:
            self.hippo.working.clear_reflections()

    async def _finish_task_for_exit(
        self, task: TaskRecord, run: _RunState, exit_: _Exit
    ) -> None:
        """:meth:`_finish_task` for a run that ends because the process exits.

        The tool providers are closed first, since their processes would
        outlive this one, while a record that is not saved only costs a
        row: closing them stops waiting with :data:`EXIT_SAVE_RESERVE_S`
        left and kills a provider still closing then (see
        :meth:`_close_providers`), so that a close() that hangs cannot
        prevent the save. The save is bounded by the exit's deadline. The
        reflections are cleared without being promoted: promotion calls
        the embedding server and may take longer than the exit allows. A
        failure is logged, not raised, and a cancellation during the
        closing is raised once the task is saved.
        """
        task.status = run.status
        task.result = run.result or run.error
        cancelled: asyncio.CancelledError | None = None
        try:
            try:
                await self._close_providers(self._tool_providers)
            except asyncio.CancelledError as exc:
                cancelled = exc
            bound = asyncio.timeout(exit_.time_left())
            try:
                async with bound:
                    await self.hippo.save_task(task)
            except Exception as exc:
                if isinstance(exc, TimeoutError) and bound.expired():
                    reason = "no time left before the exit"
                else:
                    reason = _exception_text(exc)
                logger.warning(
                    "Could not save the final state of task %s: %s",
                    task.task_id,
                    reason,
                )
        finally:
            self.hippo.working.clear_reflections()
        if cancelled is not None:
            raise cancelled

    # ── Cancelled interactive runs ──

    def _on_cancelled(
        self, query: str, run: _RunState, exc: asyncio.CancelledError
    ) -> None:
        """End *run*, an interactive run that a cancellation stopped.

        Synchronous, in run_task's handler of the cancellation. The run
        ends ``cancelled``, with :meth:`_cancel_reason` as its error.
        Working memory gets the query, when the cancellation came before
        it did (during the first save or the tool setup), and the step
        whose tool calls were under way, closed (see
        :meth:`_close_interrupted_step`); a cancellation during planning or
        reflection adds no partial turn. :meth:`_finalize` then saves the
        run.
        """
        run.end("cancelled", error=self._cancel_reason(exc))
        if not run.query_in_memory:
            self.hippo.working.append(Message(role="user", content=query))
            run.query_in_memory = True
        if run.inflight is None:
            return
        try:
            self._close_interrupted_step(run.inflight, run.waiting_call_id)
        except Exception:
            # Logged, not raised: nothing may take the cancellation's place.
            logger.exception(
                "Could not close the interrupted step of task %s", run.task_id
            )

    def _cancel_reason(self, exc: asyncio.CancelledError) -> str:
        """The error of an interactive run that the cancellation *exc* ended.

        The exit's reason when :meth:`begin_exit` was called first, as for
        any run; else the message of ``task.cancel(msg)``, when it is a
        non-empty ``str`` and arrives as the first argument of *exc*; else
        :data:`CANCELLED_ERROR`.
        """
        if self._exit is not None:
            return self._exit.reason
        message = exc.args[0] if exc.args else None
        if isinstance(message, str) and message:
            return message
        return CANCELLED_ERROR

    def _close_interrupted_step(
        self, step: _InflightStep, waiting_call_id: str
    ) -> None:
        """Answer every call of *step* and append the step to working memory.

        The calls that returned keep their results, as the image
        references that :meth:`_do_memorize` would have kept (see
        :func:`_image_references`). The call that was under way gets
        :data:`~localmelo.melo.schema.CALL_INTERRUPTED`, or
        :data:`~localmelo.melo.schema.CALL_NOT_STARTED` when its approval
        was still being decided (*waiting_call_id*), and each later call
        :data:`~localmelo.melo.schema.CALL_NOT_STARTED`; their cut counts
        are 0. When every call had returned, the cancellation came while
        :meth:`_do_memorize` recorded them, before it appended the turn.
        The turn goes through :meth:`_checked_tool_turn`, so the window
        stays protocol-valid, and ``step.closed`` keeps the results for the
        history rows (see :meth:`_save_interrupted_step`).
        """
        calls = step.response.tool_calls
        returned = len(step.results)
        closed = [_image_references(result) for result in step.results]
        for index in range(returned, len(calls)):
            call = calls[index]
            interrupted = index == returned and call.id != waiting_call_id
            error = CALL_INTERRUPTED if interrupted else CALL_NOT_STARTED
            closed.append(ToolResult(tool_name=call.tool_name, error=error))
        step.closed = closed
        cut_before = [*step.cut_before, *[0] * (len(calls) - returned)]
        assistant_msg, tool_msgs = self._checked_tool_turn(
            step.response, closed, cut_before
        )
        self.hippo.working.append_turn(assistant_msg, tool_msgs)

    async def _finalize_in_task(
        self, task: TaskRecord, run: _RunState, stats: RequestStats, started: float
    ) -> None:
        """Run :meth:`_finalize` in a task of its own and wait for it.

        In its own task, ``asyncio.timeout`` can bound the finalizer's
        steps although the run's task is being cancelled, and a second
        cancellation stops only this wait: run_task then raises at once,
        while the finalizer still saves the run and reports
        ``run.finished``. The run counts as under way until the finalizer
        ends, and :meth:`close` waits for it.
        """
        finalizer = asyncio.create_task(self._finalize(task, run, stats, started))
        run.finalizer = finalizer
        self._finalizers[finalizer] = task.task_id
        finalizer.add_done_callback(self._finalizer_done)
        await asyncio.shield(finalizer)

    def _finalizer_done(self, finalizer: asyncio.Task[None]) -> None:
        self._finalizers.pop(finalizer, None)
        self._active_runs -= 1

    async def _finalize(
        self, task: TaskRecord, run: _RunState, stats: RequestStats, started: float
    ) -> None:
        """Save a cancelled interactive run, then report its ``run.finished``.

        In order:

        1. During an exit only, the tool providers are closed, as
           :meth:`_finish_task_for_exit` closes them, since their processes
           would outlive this one: a provider still closing when
           :data:`EXIT_SAVE_RESERVE_S` of the exit's time is left is
           killed. Without an exit they stay open for the next run.
        2. History gets the rows the interrupted step is missing (see
           :meth:`_save_interrupted_step`), within
           :data:`FINALIZE_TIMEOUT_S`; during an exit, also only until
           :data:`EXIT_SAVE_RESERVE_S` of its time is left.
        3. The task record is saved, with the status ``cancelled`` and the
           error, within what is left of step 2's
           :data:`FINALIZE_TIMEOUT_S`, or during an exit within the exit's
           time. Then, when the run has a call recorder, the step's rows
           that are written are listed on its planning reply's
           ``llm_calls`` row, within what is still left; with no time left,
           the link is skipped. It comes after the save, so a held write
           lock cannot cost the task record its status: an unlinked row is
           the lesser loss. A failed link is logged by the recorder.
        4. The reflections are cleared without being promoted: no
           long-term write, and no embedding or other network call.
        5. The RunResult is built, kept as ``run.outcome`` and reported
           with ``run.finished``.

        Each step reads ``self._exit`` when it starts, so an exit that
        begins after the cancellation bounds the steps not yet started. A
        write that fails is logged, and the next one still runs; a bound
        that expires is logged at ERROR, and the next step still runs.
        Steps 4 and 5 always run. A cancellation that comes during step 1
        is raised once step 5 is done, as :meth:`_finish_task_for_exit`
        raises it.
        """
        loop = asyncio.get_running_loop()
        cancelled: asyncio.CancelledError | None = None
        try:
            if self._exit is not None:
                try:
                    await self._close_providers(self._tool_providers)
                except asyncio.CancelledError as exc:
                    cancelled = exc
            finalize_by = loop.time() + FINALIZE_TIMEOUT_S
            # The ids of the interrupted step's steps rows that are written.
            written: list[str] = []
            if run.inflight is not None:
                bound = FINALIZE_TIMEOUT_S
                if self._exit is not None:
                    bound = min(bound, self._exit.time_left(keep=EXIT_SAVE_RESERVE_S))
                await self._finalize_within(
                    bound,
                    task.task_id,
                    self._save_interrupted_step(task.task_id, run.inflight, written),
                )
            task.status = run.status
            task.result = run.result or run.error
            if self._exit is not None:
                bound = self._exit.time_left()
            else:
                bound = max(finalize_by - loop.time(), 0.0)
            await self._finalize_within(
                bound, task.task_id, self._save_final_state(task)
            )
            recorder = CALL_RECORDER.get()
            if recorder is not None and written:
                # After the task record's save, within what is left of its
                # time: an unlinked row is the lesser loss.
                if self._exit is not None:
                    bound = self._exit.time_left()
                else:
                    bound = max(finalize_by - loop.time(), 0.0)
                if bound > 0:
                    await self._finalize_within(
                        bound, task.task_id, recorder.link_steps(written)
                    )
        finally:
            self.hippo.working.clear_reflections()
            run.outcome = self._run_result(
                task, run, stats, time.perf_counter() - started
            )
            run.report_progress(ended=run.status)
            run.events.run_finished(run.outcome)
        if cancelled is not None:
            raise cancelled

    async def _finalize_within(
        self, bound: float, task_id: str, writes: Coroutine[Any, Any, None]
    ) -> None:
        """Await *writes*, one step of :meth:`_finalize`, for at most *bound* s.

        Neither an expired bound, logged at ERROR, nor a failure, logged
        with its traceback, is raised, so the next step still runs.
        """
        bounded = asyncio.timeout(bound)
        try:
            async with bounded:
                await writes
        except Exception as exc:
            if isinstance(exc, TimeoutError) and bounded.expired():
                logger.error(
                    "Finalizing cancelled task %s timed out; its records may be "
                    "incomplete",
                    task_id,
                )
            else:
                logger.exception("Could not finalize cancelled task %s", task_id)

    async def _save_interrupted_step(
        self, task_id: str, step: _InflightStep, written: list[str]
    ) -> None:
        """Write the history rows of *step* that the cancellation left out.

        The record whose ``store_step`` the cancellation interrupted, if
        any, is written again, with the same ``step_id``: an
        ``sqlite3.IntegrityError`` then means that the interrupted write
        had been committed after all, and is logged at DEBUG. Each call
        that has no record yet gets a new one, with the result the closed
        step gave it (see :meth:`_close_interrupted_step`), and the
        assistant's text goes on the step's first record, as in
        :meth:`_do_memorize`. Nothing goes to long-term memory. Each write
        is attempted on its own: a failure is logged, and the next write
        still runs.

        *written* gets the ids of the step's rows that are in history, in
        order: those ``store_step`` had stored, then the one written again
        (also when it turns out committed before), then each new one once
        its write returns. A write that fails adds none, so
        :meth:`_finalize` never links a row that does not exist.
        """
        written.extend(record.step_id for record in step.records[: step.stored])
        if step.stored < len(step.records):
            record = step.records[step.stored]
            try:
                await self.hippo.store_step(task_id, record)
                written.append(record.step_id)
            except sqlite3.IntegrityError:
                written.append(record.step_id)
                logger.debug(
                    "Step %s of task %s was saved before the cancellation",
                    record.step_id,
                    task_id,
                )
            except Exception as exc:
                logger.warning(
                    "Could not save a step of task %s: %s",
                    task_id,
                    _exception_text(exc),
                )
        calls = step.response.tool_calls
        for index in range(len(step.records), len(calls)):
            try:
                record = StepRecord(
                    thought=step.response.content if index == 0 else "",
                    tool_call=calls[index],
                    tool_result=step.closed[index],
                )
                await self.hippo.store_step(task_id, record)
                written.append(record.step_id)
            except Exception as exc:
                logger.warning(
                    "Could not save a step of task %s: %s",
                    task_id,
                    _exception_text(exc),
                )

    async def _save_final_state(self, task: TaskRecord) -> None:
        """Save *task*; a failure is logged without a traceback, not raised."""
        try:
            await self.hippo.save_task(task)
        except Exception as exc:
            logger.warning(
                "Could not save the final state of task %s: %s",
                task.task_id,
                _exception_text(exc),
            )

    def _run_result(
        self, task: TaskRecord, run: _RunState, stats: RequestStats, duration_s: float
    ) -> RunResult:
        return RunResult(
            task_id=task.task_id,
            status=run.status,
            result=run.result,
            error=run.error,
            steps=run.steps,
            attempts=run.attempts,
            n_llm=stats.n_llm,
            n_main=stats.n_main,
            n_aux=stats.n_aux,
            n_tool=run.n_tool,
            llm_errors=stats.llm_errors,
            usage=dict(stats.usage),
            duration_s=duration_s,
            config=self._run_config(),
        )

    def _run_config(self) -> dict[str, Any]:
        """The effective configuration that ``RunResult.config`` reports."""
        opts = self.options
        builtins = builtin_names() if opts.builtins is None else opts.builtins
        return {
            "max_steps": opts.max_steps,
            "attempt_policy": opts.attempt_policy,
            "steps_per_attempt": opts.steps_per_attempt,
            "max_attempts": opts.max_attempts,
            "reflection_gate": opts.reflection_gate,
            "tool_exposure": opts.tool_exposure,
            "builtins": sorted(builtins),
            "memory": opts.memory,
            "memory_dir": self._memory_dir,
            "timeout_s": opts.timeout_s,
            "tools": sorted(tool.name for tool in self.hippo.list_tools()),
        }

    async def _run_reflective_attempts(
        self, task: TaskRecord, query: str, run: _RunState
    ) -> None:
        """``attempt_policy="reflect"``: attempts with reflection between them.

        Each attempt gets at most ``steps_per_attempt`` steps, and all
        attempts together at most ``max_steps``. An attempt that runs out
        of steps or gets stuck is followed by a reflection, whose decision
        ends the run or starts the next attempt:

        - ``done`` ends the run as ``completed``. The answer is the
          decision's ``best_effort_result`` or, when that is blank, the
          latest non-blank text of a planning reply in this run (``""`` if
          there is none).
        - After the last attempt, or once ``max_steps`` is used up, no
          further attempt can start, and the reflection is told so: it can
          only finalize the answer. Any decision but ``done`` ends the run
          as ``budget_exhausted``, with its ``best_effort_result`` as the
          result.
        - Otherwise the continuation gate (:func:`_should_continue`)
          decides, as ``reflection_gate`` says. With ``"flags"`` the next
          attempt starts when the decision is ``continue`` and its three
          checks hold (``task_still_feasible``, ``next_step_is_concrete``
          and ``next_step_is_novel``). ``"utility"``, the gate from before
          that option, also requires the utility computed from the
          decision's own estimates (:func:`_utility`) to reach
          :data:`CONTINUATION_UTILITY_THRESHOLD`. A gate that stops the
          run ends it as ``gate_stopped``, with the ``best_effort_result``
          as the result and the decision's rationale as the error, or
          ``Stopped after attempt <n>`` (``n`` counting from 1) when the
          rationale is blank.

        With ``"flags"``, a reply that holds no reflection even when asked
        again (see :meth:`_do_reflect`) gives no decision and adds no
        reflection: the previous one, if any, stays in working memory, and
        nothing of the reply reaches long-term memory. The next attempt
        starts; when none can start, the run ends as ``budget_exhausted``
        with the boundary's usual error, ``Max attempts reached`` or
        ``Max steps reached``, and the latest non-blank text of a planning
        reply in this run as the result. One INFO line names the attempt.
        With ``"utility"`` such a reply reads ``stop`` (see
        :meth:`_parse_reflection`) and is kept as the reflection.

        A reflection over :data:`MAX_REFLECTION_CHARS` keeps its decision
        with ``"flags"`` and is trimmed to fit before it is kept; with
        ``"utility"`` any decision but ``done`` becomes ``decompose``,
        which the gate stops, and ``done`` ends the run as it is (see
        :meth:`_do_reflect`).
        """
        opts = self.options
        total_steps = 0

        for attempt_id in range(opts.max_attempts):
            # Never below 1: the previous attempt ended the run otherwise.
            budget = min(opts.steps_per_attempt, opts.max_steps - total_steps)
            run.attempts += 1
            steps_used, failure_type = await self._run_attempt(task, query, budget, run)
            total_steps += steps_used

            if not run.running:
                return

            # Reflect at attempt boundary
            task.attempts_completed = attempt_id + 1
            if run.attempts >= opts.max_attempts:
                out_of = "Max attempts reached"
            elif total_steps >= opts.max_steps:
                out_of = "Max steps reached"
            else:
                out_of = ""  # another attempt can start
            run.events.reflection_started(failure_type)
            run.note(Trace.reflection)
            reflection = await self._do_reflect(
                task,
                attempt_id,
                failure_type,
                final=bool(out_of),
                replies=run.events.replies,
            )
            if reflection is None:  # only with reflection_gate="flags"
                run.events.reflection(None, continue_run=not out_of)
                if out_of:
                    logger.info(
                        "Reflection after attempt %d unreadable after a retry, "
                        "ending the run: %s",
                        run.attempts,
                        out_of,
                    )
                    run.end("budget_exhausted", result=run.last_text, error=out_of)
                    return
                logger.info(
                    "Reflection after attempt %d unreadable after a retry, "
                    "starting attempt %d",
                    run.attempts,
                    run.attempts + 1,
                )
                continue
            entry, decision = reflection
            self.hippo.working.add_reflection(entry)

            best_effort = decision.best_effort_result
            if decision.recommended_action == "done":
                run.events.reflection(reflection, continue_run=False)
                answer = best_effort if best_effort.strip() else run.last_text
                run.end("completed", result=answer)
                return
            if out_of:
                run.events.reflection(reflection, continue_run=False)
                run.end("budget_exhausted", result=best_effort, error=out_of)
                return
            if not _should_continue(decision, opts.reflection_gate):
                run.events.reflection(reflection, continue_run=False)
                stopped = f"Stopped after attempt {run.attempts}"
                run.end(
                    "gate_stopped",
                    result=best_effort,
                    error=decision.rationale.strip() or stopped,
                )
                return
            run.events.reflection(reflection, continue_run=True)

    async def _run_single_attempt(
        self, task: TaskRecord, query: str, run: _RunState
    ) -> None:
        """``attempt_policy="none"``: one attempt with budget ``max_steps``.

        No reflection call is made. An attempt that runs out of steps ends
        the run as ``budget_exhausted``, one that gets stuck as ``stuck``.
        """
        run.attempts += 1
        _, failure_type = await self._run_attempt(
            task, query, self.options.max_steps, run
        )
        if not run.running:
            return
        task.attempts_completed = 1
        if failure_type == "stuck":
            run.end("stuck", error="Stuck: the same tool calls or errors repeated")
        else:
            run.end("budget_exhausted", error="Max steps reached")

    def begin_exit(self, reason: str, *, within_s: float) -> None:
        """Prepare for the process to exit within *within_s* seconds.

        For a signal such as SIGTERM: call this from the event loop, then
        cancel the task that runs :meth:`run_task`, if one does, and await
        :meth:`close`. A run that the cancellation ends is recorded as
        ``crashed`` with *reason* as its error; an interactive run, one
        with ``on_event`` or an approval policy, ends ``cancelled``
        instead, with *reason* as its error too, and its finalizer keeps
        to the same order and limit (see :meth:`run_task`). Before it
        saves its task record, it closes the tool providers, killing any
        that has not closed when only :data:`EXIT_SAVE_RESERVE_S` of the
        time is left (:meth:`BaseToolProvider.kill`); its reflections are
        cleared without being promoted. A provider that the cancellation
        finds still starting is waited for to the same limit, then killed
        (see :meth:`_start_provider`). A run that had already ended, and
        whose final save the cancellation cuts short, saves its record
        again the same way, with the status it ended with. close() keeps to
        the same limit when it waits for the finalizers of cancelled runs
        and when it closes the providers again, and no provider starts any
        more.

        The first call sets the deadline; later calls change nothing.
        """
        if self._exit is not None:
            return
        deadline = asyncio.get_running_loop().time() + within_s
        self._exit = _Exit(reason=reason, deadline=deadline)
        self._closing = True

    async def close(self) -> None:
        """Close the tool providers, the memory backends and the model providers.

        Every tool provider is closed, whether or not it was started; a
        provider that fails to close is logged and the others are still
        closed. No provider starts once close() has begun. If a run is
        still starting the providers, close() waits for that setup to end
        (closing an MCP tool provider makes its ``start()`` give up); the
        setup then closes whatever it started, so no provider is left
        running when close() returns. A cancellation of close() takes
        effect only once every tool provider is closed. Calling close()
        again closes the providers again, which their contract makes
        harmless.

        After :meth:`begin_exit`, close() waits for the providers and for
        a setup under way only until the exit's deadline (providers are
        killed :data:`EXIT_SAVE_RESERVE_S` before it, see
        :meth:`_close_providers`).

        Before all this, close() waits for the finalizers of cancelled
        interactive runs (see :meth:`run_task`) to save them, for at most
        ``FINALIZE_TIMEOUT_S + 1`` seconds, or after :meth:`begin_exit`
        only until the exit's deadline. A finalizer still running then is
        logged at ERROR and left to end on its own: close() never cancels a
        run.
        """
        try:
            self._closing = True
            if self._finalizers:
                try:
                    await self._wait_for_finalizers()
                except asyncio.CancelledError:
                    # As below: the cancellation takes effect only once every
                    # tool provider is closed.
                    await self._close_providers(self._tool_providers)
                    raise
            await self._close_providers(self._tool_providers)
            # Wait for a setup still running in a run (see above).
            await self._wait_for_setup()
        finally:
            await self.hippo.aclose()
            await self._llm.close()
            if self._embedding:
                await self._embedding.close()

    async def _wait_for_finalizers(self) -> None:
        """Wait for the finalizers of cancelled runs, for a bounded time.

        At most ``FINALIZE_TIMEOUT_S + 1`` seconds, or after
        :meth:`begin_exit` until the exit's deadline. Each finalizer still
        running then is logged at ERROR; none is cancelled.
        """
        pending = dict(self._finalizers)
        exit_ = self._exit
        bound = FINALIZE_TIMEOUT_S + 1 if exit_ is None else exit_.time_left()
        _, running = await asyncio.wait(set(pending), timeout=bound)
        for finalizer in running:
            logger.error(
                "Cancelled task %s is still being finalized; closing without "
                "waiting for it",
                pending[finalizer],
            )

    async def _wait_for_setup(self) -> None:
        """Wait until no run is setting up the tool providers.

        After :meth:`begin_exit`, only until the exit's deadline.
        """
        if self._exit is None:
            async with self._tools_lock:
                return
        bound = asyncio.timeout(self._exit.time_left())
        try:
            async with bound, self._tools_lock:
                return
        except TimeoutError:
            if not bound.expired():
                raise
            logger.warning("Tool setup still running at the exit; not waiting for it")
