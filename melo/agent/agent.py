from __future__ import annotations

import asyncio
import contextvars
import copy
import dataclasses
import functools
import hashlib
import inspect
import itertools
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

from localmelo.melo.agent._brief_aids import (
    DROPPED_KEPT,
    REFLECTION_ADDENDUM,
    STEP_PUSH,
    STEP_PUSH_STEPS,
    FileStamp,
    dropped_notes,
    edited_files,
    mark_items,
    parse_items,
    render_dropped,
    render_items,
    start_with,
    workspace_stamps,
)
from localmelo.melo.agent._brief_gate import (
    MAX_CHECK_RETRIES,
    check_commands,
    failure_message,
)
from localmelo.melo.agent._call_budget import CallBudgetError, is_budget_refusal
from localmelo.melo.agent._call_log import CallHistory, CallLog
from localmelo.melo.agent._captured_history import (
    capture_result,
    preview_result,
    split_image_notes,
)
from localmelo.melo.agent._chat import (
    BUDGET_ACCOUNTING,
    BUDGET_HOOK,
    CALL_RECORDER,
    LEAN_REFLECTION_PROMPT,
    LEAN_SYSTEM_PROMPT,
    PREFIX_RUN,
    PREFIX_VERSION,
    REFLECTION_PROMPT,
    RETRY_HOOK,
    SYSTEM_PROMPT,
    CallTelemetry,
    Chat,
    RequestStats,
    coalesce_user_messages,
    lacks_task_query,
)
from localmelo.melo.agent._context_fit import ContextFitError
from localmelo.melo.agent._events import GateVerdict, _Emitter, check_sink
from localmelo.melo.agent._json_repair import error_line, rejected_objects, repairs
from localmelo.melo.agent._tool_images import ToolImageStore
from localmelo.melo.agent.options import (
    REPLAY_TURN_CAP,
    AgentOptions,
    ReflectionGate,
    reasoning_refusal,
    retrieval_refusal,
)
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import (
    ActionReviewPayload,
    ExecutorResultPayload,
    MemoryCandidatePayload,
    MemoryWritePayload,
    ToolResolutionResult,
)
from localmelo.melo.checker.policy import builtin_set_for
from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.tiers import tier_of, tier_requires_approval
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN, _sanitize_images
from localmelo.melo.contracts.approvals import (
    TOOL_ASK,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from localmelo.melo.contracts.events import EventSink
from localmelo.melo.contracts.inbox import InboxError, InboxSource
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.contracts.review import SELF_DENIAL, ActionReview
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import (
    TOOL_EXIT_CODE,
    Executor,
    WriteConfinement,
    call_preview,
    confine,
    platform_note,
    register_builtins,
)
from localmelo.melo.executor.env import child_env
from localmelo.melo.executor.models import ExecutionRequest
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.hippo import (
    KINDS,
    SIGNALS,
    MemoryDecision,
    classify_user_text,
    request_candidates,
    user_candidates,
)
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.notes import (
    NOTE_MAX_DECIDED,
    NOTE_MAX_FOUND,
    NOTE_SHOWN_CHARS,
    Draft,
    ReasoningNote,
    SourceRecordLog,
    candidate_items,
    candidate_text,
    digest_note,
    draft_title,
    extract_draft,
    mark_drafts_written,
    note_from_reply,
    note_title,
    path_spans,
    reflection_candidate_text,
    render_digest,
    render_draft_block,
    render_note,
    same_path,
    source_records,
    split_note,
    thought_chunks,
)
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
from localmelo.melo.memory.recall import (
    DEFAULT_TRIGGERS,
    LONG_POOL,
    RECALL_BOUNDARY_CHARS,
    RECALL_DATA_NOTE,
    RECALL_END,
    RECALL_K_MAX,
    RECALL_LIVE_CHARS,
    RECALL_LONG_K,
    RECALL_THOUGHT_PER_TASK,
    RECALL_TOOL_DEF,
    RECALL_TOOL_TIMEOUT_S,
    RECALL_TURN_CHARS,
    RecallHit,
    RecallRecord,
    build_attempt_query,
    build_start_query,
    build_step_query,
    compact_record,
    default_kinds,
    docs_from_entries,
    docs_from_notes,
    fit_recall_block,
    hit_from_doc,
    recall_span_chars,
    recall_spans,
    record_to_dict,
    relation_counts,
    resolve_plan,
)
from localmelo.melo.memory.recall_gate import (
    digest_has_evidence,
    digest_task_key,
    gate_long_docs,
    is_task_digest,
    same_task,
)
from localmelo.melo.memory.retrieval import build_retriever
from localmelo.melo.memory.retrieval.base import Doc, RetrievalResult, Retriever
from localmelo.melo.memory.short import group_turns
from localmelo.melo.memory.short.note_store import NoteStoreEntry
from localmelo.melo.schema import (
    ANSWER_NOT_KEPT_LABEL,
    CALL_INTERRUPTED,
    CALL_NOT_STARTED,
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    MAX_AGENT_STEPS,
    MAX_REFLECTION_CHARS,
    MIN_AGENT_STEPS,
    SHORT_TERM_MAX,
    UNFINISHED_ANSWER_ERROR,
    WINDOW_EVICT_CHUNK,
    WINDOW_FIT_TARGET,
    CandidateItem,
    ImagePart,
    MemoryCandidate,
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
    new_record_id,
    new_tool_call_id,
    split_thinking,
    unfinished_as_thinking,
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
    JSON that is not an object holds no reflection. This is the strict
    reading; the loop reads replies with :func:`_read_reflection`.
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


# The most ``{...}`` objects of one reply that _read_reflection tries to repair.
REPAIR_TRIES = 8


def _read_reflection(text: str) -> tuple[dict[str, Any], str | None] | None:
    """The reflection a reply holds under ``"flags"``, and the repair it took.

    That is the reflection of :func:`_find_reflection` with ``None`` as the
    repair. A reply that holds none gets one more reading with a single
    conservative repair of a ``{...}`` the decoder rejects:
    ``"unmatched_closer"``, ``"trailing_comma"``, ``"control_char"``,
    ``"invalid_escape"`` or ``"unclosed_array"`` (see
    :func:`~localmelo.melo.agent._json_repair.repairs`). The repaired text is
    read as any reply is, so a repair counts only when it gives an object
    with a valid ``recommended_action``, under the order and the nesting rule
    of :func:`_find_reflection`. A reply that needs two repairs holds no
    reflection, and nothing but a backslash or a bracket is inserted, so
    truncated text stays unreadable. A reply that reads as it stands is never
    repaired.
    """
    found = _find_reflection(text)
    if found is not None:
        return found, None
    text = _coerce_str(text)
    if "recommended_action" not in text:
        return None
    body = _strip_code_fences(text)
    for start, error in itertools.islice(rejected_objects(body), REPAIR_TRIES):
        for repair, fixed in repairs(body, start, error):
            found = _find_reflection(fixed)
            if found is not None:
                return found, repair
    return None


def _json_error_line(text: str) -> str:
    """The line that a retry adds when a reply has JSON the decoder rejects, else ``""``.

    It quotes the error of the first such ``{...}`` and the text around it
    (see :func:`~localmelo.melo.agent._json_repair.error_line`); the position
    counts the characters of the reply, a code fence around it included.
    """
    text = _coerce_str(text)
    body = _strip_code_fences(text)
    first = next(rejected_objects(body), None)
    if first is None:
        return ""
    return error_line(body, first[1], max(text.find(body), 0))


def _unreadable_reason(reply: Message, *, flags: bool) -> str:
    """Why *reply* holds no reflection: ``tool_call``, ``invalid`` or ``no_json``.

    ``tool_call``: the reply has tool calls (and no reflection in its text);
    ``invalid``: its text has a JSON object that ``"flags"`` does not accept;
    ``no_json``: its text has no JSON object.
    """
    if reply.tool_calls:
        return "tool_call"
    if flags and _holds_reflection_object(reply.content):
        return "invalid"
    return "no_json"


def _as_planned(
    window: list[Message], query: str, query_block: str = ""
) -> list[Message]:
    """*window* as a planning request carries it, for size checks.

    That is with the task *query* in front once the window no longer holds
    it, as :meth:`Chat.plan_step` puts it back (see
    :func:`~localmelo.melo.agent._chat.lacks_task_query`). *query_block*,
    the attempt's boundary recall block, rides on that query message, in
    the request copy only.
    """
    text = f"{query}\n\n{query_block}" if query_block else query
    if lacks_task_query(window, query):
        return [Message(role="user", content=text), *window]
    if not query_block:
        return window
    out = list(window)
    matches = [
        index
        for index, message in enumerate(out)
        if message.role == "user" and message.content == query
    ]
    if matches:
        index = matches[-1]
        out[index] = dataclasses.replace(out[index], content=text)
    return out


def _split_inline_reasoning(response: Message) -> Message:
    """Move a ``<think>`` block out of *response*'s content into thinking.

    The explicit modes call this on every planning and reflection reply.
    It uses :func:`~localmelo.melo.schema.split_thinking`: a block becomes
    the thinking unless the provider already set one, in which case the
    provider's value wins and the block is only removed from content.
    A reply without a block is returned unchanged.
    """
    block, content = split_thinking(response.content)
    if not block:
        return response
    if response.thinking:
        return dataclasses.replace(response, content=content)
    return dataclasses.replace(response, content=content, thinking=block)


def _replay_plan(
    window: list[Message],
    options: AgentOptions,
    *,
    task_query: Message | None = None,
    requires_replay: bool = False,
) -> tuple[list[Message], list[int]]:
    """The window as an explicit-mode request carries it, and its cap skips.

    The first list is the one :func:`_as_sent` returns. The second holds the
    indexes in *window* of the assistant turns whose thinking was left out
    because it exceeds :data:`REPLAY_TURN_CAP` (empty on a replay-required
    provider, and in ``legacy``/``drop``). On a replay-required provider the
    whole window is replayed: the task scope and ``replay_turns`` do not
    apply, since a turn the server needs is not one the caller can leave out.
    """
    if options.reasoning == "legacy":
        return window, []
    start = 0
    if (
        not requires_replay
        and options.replay_scope != "window"
        and task_query is not None
    ):
        for index, msg in enumerate(window):
            if msg is task_query:
                start = index
                break
    in_scope = [
        index
        for index, msg in enumerate(window)
        if index >= start and msg.role == "assistant" and msg.thinking
    ]
    keep = (
        set(in_scope)
        if requires_replay or options.replay_turns is None
        else set(in_scope[-options.replay_turns :])
    )
    sent: list[Message] = []
    skipped: list[int] = []
    for index, msg in enumerate(window):
        if msg.role != "assistant" or not msg.thinking:
            sent.append(msg)
            continue
        if options.reasoning not in ("replay", "compress+replay") or index not in keep:
            sent.append(dataclasses.replace(msg, thinking=""))
            continue
        if not requires_replay and len(msg.thinking) > REPLAY_TURN_CAP:
            sent.append(dataclasses.replace(msg, thinking=""))
            skipped.append(index)
            continue
        sent.append(msg)
    return sent, skipped


def _as_sent(
    window: list[Message],
    options: AgentOptions,
    *,
    task_query: Message | None = None,
    requires_replay: bool = False,
) -> list[Message]:
    """The window as a request carries it, for send and for size checks.

    In ``legacy`` it is *window* itself, unchanged. Otherwise it is a new
    list in which an assistant message keeps its thinking only in
    ``replay`` and ``compress+replay``, in scope (the current task from *task_query* on, or the
    whole window), among the newest ``replay_turns`` such turns, and within
    :data:`REPLAY_TURN_CAP` (the cap does not apply to a provider that
    requires replay). A provider that requires replay gets every assistant
    turn in the window: the task scope and ``replay_turns`` are ignored
    there. A message that fails one of these is a copy with an empty
    thinking; one that passes is the original object.
    """
    return _replay_plan(
        window,
        options,
        task_query=task_query,
        requires_replay=requires_replay,
    )[0]


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


# The lists of a reflection that reflection_gate="patient" and answer_on_stop
# cap, the most items each keeps and the most characters of an item.
REFLECTION_CAPPED_LISTS = ("useful_evidence", "failed_hypotheses", "recommended_avoids")
REFLECTION_LIST_ITEMS = 8
REFLECTION_ITEM_CHARS = 300


def _cap_reflection_lists(entry: ReflectionEntry) -> ReflectionEntry:
    """*entry* with each of :data:`REFLECTION_CAPPED_LISTS` kept to its cap.

    A list keeps its first :data:`REFLECTION_LIST_ITEMS` items, each cut to
    :data:`REFLECTION_ITEM_CHARS` characters at a word boundary (see
    :func:`_shorten`). *entry* itself is not changed.
    """
    capped = dataclasses.replace(entry)
    for name in REFLECTION_CAPPED_LISTS:
        items = getattr(entry, name)[:REFLECTION_LIST_ITEMS]
        setattr(capped, name, [_shorten(item, REFLECTION_ITEM_CHARS) for item in items])
    return capped


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
    fit or not, or the notice that replaced it. The message keeps only
    references to a result's images (their pixels reach the model only
    with ``tool_images`` on, see :class:`ToolImageStore`), so a result
    with images ends with a line that names them (see
    :func:`describe_images`), whether its text is kept, cut or withheld.
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


# The line a brief run adds to the window when an attempt boundary finds the
# workspace unchanged since the run started.
NO_CHANGE_PUSH = (
    "Nothing has changed in the workspace yet. Start this attempt by writing the "
    "part of the task you can already write, then continue."
)


def _gate_stop_reason(
    decision: ReflectionDecision,
    gate: ReflectionGate = "flags",
    *,
    patient: bool = False,
) -> str | None:
    """The first check of the continuation gate that *decision* fails.

    The codes are the ones in ``docs/reference/events.md``:
    ``action_not_continue``, ``not_feasible``, ``not_concrete``,
    ``not_novel`` and, with ``"utility"`` only, ``utility_below_threshold``.
    ``None`` when every check holds, so that the next attempt may start.
    With ``"patient"`` and *patient* true (the run's one allowance is
    unspent), a decision that is not novel passes that check.
    """
    if decision.recommended_action != "continue":
        return "action_not_continue"
    if not decision.task_still_feasible:
        return "not_feasible"
    if not decision.next_step_is_concrete:
        return "not_concrete"
    if not decision.next_step_is_novel and not (gate == "patient" and patient):
        return "not_novel"
    if gate == "utility" and _utility(decision) < CONTINUATION_UTILITY_THRESHOLD:
        return "utility_below_threshold"
    return None


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
    return _gate_stop_reason(decision, gate) is None


# The stop reasons of the budget boundaries, by the error that the run
# ends with there.
_OUT_OF_REASONS = {
    "Max attempts reached": "no_attempt_left",
    "Max steps reached": "step_budget_used",
}


def _stop_reason(
    decision: ReflectionDecision | None,
    gate: ReflectionGate,
    out_of: str,
    *,
    brief: bool = False,
    patient: bool = False,
) -> str | None:
    """Why the run does not go on after a reflection, ``None`` if it does.

    *decision* is ``None`` for a reply that holds no reflection, and
    *out_of* is the boundary's error (``""`` when another attempt can
    start). The one place that decides: ``continue_run`` is whether this
    is ``None``. In a *brief* run (``AgentOptions.brief_probe``) a
    ``continue`` goes on whatever the gate's flags say. *patient* is
    :func:`_gate_stop_reason`'s.
    """
    if decision is None:
        return "unreadable_reply" if out_of else None
    if decision.recommended_action == "done":
        return "action_not_continue"
    if out_of:
        return _OUT_OF_REASONS[out_of]
    if brief and decision.recommended_action == "continue":
        return None
    return _gate_stop_reason(decision, gate, patient=patient)


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

# The compress modes' static system instruction, passed between
# SYSTEM_PROMPT and the workspace note (see Chat). Static per run, so a
# server's prefix cache still holds.
NOTE_INSTRUCTION = (
    "Before each tool call, write a short note in your reply:\n"
    "<note>\n"
    "plan: your next step and why, in one line\n"
    "found: what you have established (file contents, test results)\n"
    "open: what is still unknown\n"
    "remember: yes if a fact or decision here should be kept after this task, "
    "else no\n"
    "</note>\n"
    "At most 600 characters. Your earlier thinking is not shown to you again; "
    "your notes and code drafts are.\n"
    "Put code you drafted into the tool call, not into the note."
)

# The one sentence of NOTE_INSTRUCTION that compress+replay changes.
NOTE_REPLAY_SENTENCE = (
    "Your earlier thinking is not shown to you again; your notes and code "
    "drafts are."
)

# A note text after this many characters is cut in the reflection's block.
REFLECTION_NOTE_CHARS = NOTE_SHOWN_CHARS

# Notes of this attempt the reflection request carries at most.
REFLECTION_MAX_NOTES = 12


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
    captured: list[tuple[ToolResult, dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )
    # The history records _do_memorize created, in call order, and how
    # many of them store_step finished writing: all of them, or all but
    # the last.
    records: list[StepRecord] = dataclasses.field(default_factory=list)
    stored: int = 0
    # The result the closed step gives each call, once it is closed.
    closed: list[ToolResult] = dataclasses.field(default_factory=list)
    # In the compress modes, the reply's content without its <note> block,
    # and whether it carried a model note.
    rest: str = ""
    has_model_note: bool = False
    # The aliases of the thought chunks stored from this reply. They count
    # as shown while the reply's turn is in the window, so they leave with
    # it (see _do_memorize).
    thought_aliases: list[str] = dataclasses.field(default_factory=list)
    # The task's most recent successful read or write before this reply, a
    # draft target when the reply names none. The exporter takes the same
    # value: a path used in this reply is not the fallback for it.
    recent_path: str = ""


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
    # The confinement's refused writes when the run started.
    refused_writes_base: int = 0
    fixed_tools: list[ToolDef] | None = None
    # reflection_gate="patient": whether the run's one non-novel continue is spent.
    patience_spent: bool = False
    # answer_on_stop: whether result is the answer of the extra request.
    answered_on_stop: bool = False
    # How the run ended on its own limit or on a refusal: "calls" (max_calls) or
    # "refused" (a request refused for a budget or rate limit, with answer_on_stop).
    early_stop: str = ""
    # The latest non-blank text of a planning reply in this run: the
    # answer of a reflection's "done" that gives none, and the result of a
    # final boundary whose reflection is unreadable (reflection_gate="flags").
    last_text: str = ""
    # The thinking of the planning reply that set last_text, and of the last
    # reflection reply read; _record_answer replays the one behind the result.
    last_text_thinking: str = ""
    reflection_thinking: str = ""
    # The thinking of the reply that gave the result, once the run ends.
    answer_thinking: str = ""
    # Planning replies in a row with neither a tool call nor text, and
    # whether one of them was only unfinished thinking (#278).
    empty_replies: int = 0
    unfinished_row: bool = False
    # The run's task id ("" for a call without a task record).
    task_id: str = ""
    # The query Message this run appended to the window, for replay scope.
    task_query_msg: Message | None = None
    # User messages the run inserted mid-run (inbox): kept at a boundary drop.
    pinned: list[Message] = dataclasses.field(default_factory=list)
    # cache_layout v3: the message that carries the latest reflection in the
    # window, put back once if an eviction takes it.
    reflection_msg: Message | None = None
    # An opt-in source of parent instructions at planning boundaries.
    inbox: InboxSource | None = None
    # Replay counters (see memory_stats on RunResult).
    replayed_chars: int = 0
    replay_skipped: int = 0
    replay_stripped: int = 0
    # R4's counters (see memory_stats on RunResult), kept in every explicit
    # mode so the shape is stable; only the compress modes move them.
    notes: int = 0
    notes_model: int = 0
    notes_heuristic: int = 0
    note_chars: int = 0
    drafts: int = 0
    drafts_written: int = 0
    candidates: int = 0
    remember_model: int = 0
    remember_heuristic: int = 0
    candidates_refused: int = 0
    redactions: int = 0
    digests: int = 0
    # The task's tool-result records, for a later note's items: the
    # results still in the window and the rest, both newest first, bounded.
    source_log: SourceRecordLog = dataclasses.field(default_factory=SourceRecordLog)
    # The current attempt's executed calls and their checked results, for
    # its digest: files written and the last check. Only the compress
    # modes fill it, and it is cleared at the attempt's start and after its
    # digest.
    attempt_calls: list[tuple[ToolCall, ToolResult]] = dataclasses.field(
        default_factory=list
    )
    # Rehearsal signals: restated, acted and cited.
    rehearsed: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"restated": 0, "acted": 0, "cited": 0}
    )
    # The path of the latest successful file read or write, a draft's target
    # when the reply names none.
    recent_path: str = ""
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
    # R5: the attempt's frozen boundary block, whether the start trigger has
    # run, the recall records not yet sent, and the aliases shown but not yet
    # in the window.
    boundary_block: str = ""
    # A brief run (AgentOptions.brief_probe): the workspace digest at the
    # run's start, None outside a git tree.
    brief: bool = False
    workspace_digest: str | None = None
    # Its step-level push: the digest at the last change seen, and the steps
    # since the last probe. And the notes of the tool output the window lost
    # (newest DROPPED_KEPT).
    step_digest: str | None = None
    read_streak: int = 0
    dropped: list[str] = dataclasses.field(default_factory=list)
    # Its check gate (AgentOptions.brief_check): the brief's check commands,
    # the gate's runs so far and the retries it has used.
    check_commands: list[str] = dataclasses.field(default_factory=list)
    check_rounds: int = 0
    check_retries: int = 0
    # The workspace digest when the checks last ran, the attempt the gate
    # last ran in, and whether the running attempt is the run's last.
    check_digest: str | None = None
    check_attempt: int = 0
    last_attempt: bool = False
    # AgentOptions.brief_items: the brief's numbered items and the files git
    # listed as changed at the run's start (None: not tracked).
    items: list[str] = dataclasses.field(default_factory=list)
    items_baseline: dict[str, FileStamp] | None = None
    boundary_done: bool = False
    pending_recalls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    pending_recall_records: list[RecallRecord] = dataclasses.field(default_factory=list)
    plan_call: CallTelemetry | None = None
    pending_shown: set[str] = dataclasses.field(default_factory=set)
    # R6-lite: the run's user request, its last digest and final reflection,
    # and the admission counters (see memory_stats on RunResult).
    task_text: str = ""
    remember_request: bool = False
    final_digest: MemoryCandidate | None = None
    final_reflection: MemoryCandidate | None = None
    admitted: dict[str, int] = dataclasses.field(
        default_factory=lambda: {signal: 0 for signal in SIGNALS}
    )
    written: dict[str, int] = dataclasses.field(
        default_factory=lambda: {kind: 0 for kind in KINDS}
    )
    ignored: int = 0
    # R5's recall counters (see memory_stats on RunResult).
    recalls: int = 0
    recalls_by_trigger: dict[str, int] = dataclasses.field(
        default_factory=lambda: {trigger: 0 for trigger in (*DEFAULT_TRIGGERS, "tool")}
    )
    recall_hits: int = 0
    recall_same_task: int = 0
    recall_cross_task: int = 0
    recall_task_notes: int = 0
    recall_chars: int = 0
    recall_ms: int = 0
    recall_fallbacks: int = 0
    recall_skipped: int = 0
    recall_refused: int = 0
    recall_empty: int = 0
    aux_retrieve_calls: int = 0
    retrieve_errors: int = 0
    retrieve_prompt_tokens: int = 0
    retrieve_completion_tokens: int = 0
    thought_chunks: int = 0
    recall_log: list[dict[str, Any]] = dataclasses.field(default_factory=list)

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


# The run a ``recall`` tool call belongs to, set by run_task beside
# CALL_RECORDER. The tool's handler reads it; outside a run it is None and
# the handler answers in one line.
RECALL_RUN: contextvars.ContextVar[_RunState | None] = contextvars.ContextVar(
    "RECALL_RUN", default=None
)


def _checked_title(text: str) -> str:
    """The first sentence of checked *text*, at most 120 characters."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    for separator in (". ", "。", "! ", "? "):
        head, found, _tail = first.partition(separator)
        if found and head.strip():
            return _shorten(head.strip() + separator.strip(), 120)
    return _shorten(first, 120)


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

    With a *checker*, the request carries the call's review (#279), kept in
    :attr:`review` for the denial text; a call that would change localmelo
    itself (the ``self`` category) is denied without asking the policy.

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
        checker: Checker | None = None,
    ) -> None:
        self._policy = policy
        self._run = run
        self._call = call
        self._workspace_root = workspace_root
        self._tier = tier
        self._checker = checker
        self.review: ActionReview | None = None
        # The approval of tool.finished: request_id, allowed, reason,
        # remember and origin, once the policy is asked; None until then.
        self.asked: dict[str, Any] | None = None

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        run, call = self._run, self._call
        if self._checker is not None:
            self.review = self._checker.review_action(
                ActionReviewPayload(
                    request.tool_name, dict(request.arguments), self._workspace_root
                )
            )
        asking = ApprovalRequest(
            request_id=uuid.uuid4().hex,
            run_id=run.task_id,
            call_id=call.id,
            tool_name=request.tool_name,
            arguments=copy.deepcopy(request.arguments),
            preview=call_preview(request.tool_name, request.arguments),
            workspace_root=self._workspace_root,
            tier=self._tier,
            review=self.review,
        )
        if self.review is not None and "self" in self.review.categories:
            self._record(asking, origin="policy", reason=SELF_DENIAL)
            return ApprovalDecision(False, SELF_DENIAL, origin="policy")
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


class _ToolAsk:
    """What a running tool uses to ask the run's approval policy (``TOOL_ASK``).

    The request is for a name and arguments the tool chooses, not for the
    tool's own call, so a policy that keeps grants by name keeps one per name.
    It has the call's id, so a surface shows the question with the call. The
    run's cancellation propagates, and the prompt counts as withdrawn.
    """

    def __init__(
        self,
        policy: ApprovalPolicy,
        run: _RunState,
        call: ToolCall,
        workspace_root: str | None,
    ) -> None:
        self._policy = policy
        self._run = run
        self._call = call
        self._workspace_root = workspace_root

    async def __call__(
        self, tool_name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        decided: object = await self._policy.decide(
            ApprovalRequest(
                request_id=uuid.uuid4().hex,
                run_id=self._run.task_id,
                call_id=self._call.id,
                tool_name=tool_name,
                arguments=copy.deepcopy(dict(arguments)),
                preview=preview,
                workspace_root=self._workspace_root,
                tier="network",
            )
        )
        if not isinstance(decided, ApprovalDecision):
            raise TypeError(
                f"the approval policy returned {type(decided).__name__}, "
                "not an ApprovalDecision"
            )
        return decided


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
        if self.options.reasoning == "compress" and bool(
            getattr(self._llm, "requires_reasoning_replay", False)
        ):
            # A provider that requires replay cannot omit thinking, so
            # compress runs as compress+replay and the run reports the arm
            # it actually ran. replay_turns=None and the task scope are
            # what replay itself uses on such a provider.
            self.options = dataclasses.replace(
                self.options,
                reasoning="compress+replay",
                replay_turns=None,
                replay_scope="task",
            )
        if (
            config is not None
            and llm is None
            and self.options.reasoning not in ("replay", "compress+replay")
            and config.chat_backend != "ollama"
        ):
            # The section's key is only for replay: legacy and drop send no
            # reasoning, and the gateway, a configured UI session and a
            # Python caller all build their provider here. Ollama keeps its
            # fixed "thinking" field, whose native wire legacy already uses.
            self._llm.reasoning_field = None
        # The API user's error; the CLI refuses the same pairs before it
        # builds an Agent, so direct mode never reaches this traceback.
        refusal = reasoning_refusal(
            self.options.reasoning,
            getattr(self._llm, "reasoning_field", None),
            bool(getattr(self._llm, "requires_reasoning_replay", False)),
        )
        if refusal is not None:
            raise ValueError(refusal)
        # Resolved first, so that a bad root raises before anything, such as
        # the memory directory, is created.
        workspace_root = _resolve_workspace_root(self.options.workspace_root)
        # Then confinement, which raises ConfinementError when it is on and
        # cannot be applied (see AgentOptions.confine_writes).
        self._confinement: WriteConfinement | None = None
        if self.options.confine_writes:
            env = child_env(self.options.tool_env)
            folder = workspace_root or os.path.realpath(os.getcwd())
            self._confinement = confine(folder, env)
        self._tool_providers: tuple[BaseToolProvider, ...] = tuple(tool_providers)
        # Tool setup (_ensure_tools) runs once, under this lock. It ends
        # either ready or failed; a failure is kept as its error text.
        self._tools_lock = asyncio.Lock()
        self._tools_ready = False
        self._tools_failure: str | None = None
        # The names of the tools that the setup registered (see list_tools).
        self._provided_names: frozenset[str] = frozenset()
        # Of those, the ones in a group, which "retrieve" offers on every step.
        self._grouped_names: frozenset[str] = frozenset()
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
        long_backend: LongTerm | None = None
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
            # In legacy the file exists only with an embedding provider,
            # exactly as before; the explicit modes add a text-only store.
            if self.options.memory and (
                self._embedding is not None or self.options.reasoning != "legacy"
            ):
                long_backend = SqliteLongTerm(
                    os.path.join(self._memory_dir, "long_term.db")
                )
            # Opened, and created, at its first use: a run's end.
            procedures_backend = SqliteProcedures(
                os.path.join(self._memory_dir, "procedures.db")
            )

        if (
            long_backend is None
            and self._memory_dir is None
            and self.options.memory
            and self.options.reasoning != "legacy"
            and self.options.typed_assertions
        ):
            long_backend = LongTerm()

        # memory=False runs Hippo in its no-embedding mode, which is exactly
        # the documented behavior of that option: no long-term recall, no
        # long-term writes, no reflection promotion, and reflections are
        # still cleared. self._embedding stays set so close() closes it.
        # window_eviction "chunk" and "boundary": drop in chunks (see schema).
        self._evicts_in_chunks = self.options.window_eviction != "slide"
        # cache_layout "v3" (with the fixed prefix): a reflection is a message of
        # the window, appended at the attempt boundary, not rebuilt per request.
        self._reflection_in_window = (
            self.options.fixed_prefix and self.options.cache_layout == "v3"
        )
        # Pictures tools return (AgentOptions.tool_images): the store whose
        # payloads planning requests carry; "auto" asks the provider once.
        self._tool_images = ToolImageStore(
            self.options.tool_images_keep, self.options.tool_images_retire
        )
        self._images_on: bool | None = {"on": True, "off": False}.get(
            self.options.tool_images
        )
        self.hippo = Hippo(
            short_term_max=self.options.window_messages,
            embedding=self._embedding if self.options.memory else None,
            history=history_backend,
            long=long_backend,
            procedures=procedures_backend,
            short_term_empty_prefix=self.options.reasoning != "legacy",
            short_term_evict_chunk=(
                WINDOW_EVICT_CHUNK if self._evicts_in_chunks else 0
            ),
            memory=self.options.memory,
            admit_on_outcome=self.options.admit_on_outcome,
            keep_decision_text=self.options.keep_decision_text,
            typed_assertions=self.options.typed_assertions,
        )
        self.checker = Checker(mode=self.options.mode, builtins=self.options.builtins)
        self.executor = Executor(
            self.hippo, self.checker, workspace_root=workspace_root
        )
        # R5's retriever plan, resolved once: the effective trigger map, the
        # data note and (after Chat) the lazily built retrievers.
        self._plan = resolve_plan(
            self.options.retriever,
            reasoning=self.options.reasoning,
            has_embedding=self.hippo.embedding is not None,
            recall_at=self.options.recall_at,
        )
        refusal = self._retrieval_refusal()
        if refusal is not None:
            raise ValueError(refusal)
        self._retrievers: dict[str, Retriever] = {}
        self._recall_tool_registered = False
        self._effective_kinds = frozenset(
            default_kinds()
            if self.options.recall_kinds is None
            else self.options.recall_kinds
        )
        notes = (
            []
            if workspace_root is None
            else [WORKSPACE_NOTE.format(root=workspace_root)]
        )
        if self._plan.enabled() and self.options.reasoning != "legacy":
            notes.append(RECALL_DATA_NOTE)
        instruction = ""
        if self.options.reasoning in ("compress", "compress+replay"):
            instruction = NOTE_INSTRUCTION
            if self.options.reasoning == "compress+replay":
                kept = self.options.replay_turns
                if kept is not None:
                    step = "step" if kept == 1 else "steps"
                    sentence = (
                        f"Your thinking is kept for your last {kept} {step}; "
                        "older steps keep only their notes."
                    )
                elif self._requires_replay:
                    sentence = "Your thinking is kept for every step in the window."
                else:
                    sentence = "Your thinking is kept for every step of this task."
                instruction = instruction.replace(NOTE_REPLAY_SENTENCE, sentence)
        from localmelo.melo.agent._request_accounting import ContextBudget

        budget = None
        if self.options.context_window is not None:
            assert self.options.output_reserve is not None
            assert self.options.safety_margin is not None
            budget = ContextBudget(
                self.options.context_window,
                self.options.output_reserve,
                self.options.safety_margin,
            )
        self.chat = Chat(
            self._llm,
            system_notes=notes,
            note_instruction=instruction,
            empty_prefix=self.options.reasoning != "legacy",
            context_budget=budget,
            fixed_prefix=self.options.fixed_prefix,
            fit_target=WINDOW_FIT_TARGET if self._evicts_in_chunks else 1.0,
            tool_guard=self.options.reflect_tool_guard,
            window_reflection=self._reflection_in_window,
            # lean_output keeps the full system prompt in the compress modes,
            # whose note before each tool call is text between tool calls, and
            # the full reflection prompt under the utility gate, which reads
            # the estimates the lean one leaves out.
            system_prompt=(
                LEAN_SYSTEM_PROMPT
                if self.options.lean_output and not instruction
                else SYSTEM_PROMPT
            ),
            reflection_prompt=(
                LEAN_REFLECTION_PROMPT
                if self.options.lean_output
                and self.options.reflection_gate != "utility"
                else REFLECTION_PROMPT
            ),
            identity=self.options.identity or "",
            max_calls=self.options.max_calls,
            reserve_answer=self.options.answer_on_stop,
        )

        names = (
            None
            if self.options.builtins is None
            else frozenset(self.options.builtins - {"recall"})
        )
        register_builtins(
            self.executor,
            self.hippo,
            names=names,
            mode=self.options.mode,
            workspace_root=workspace_root,
            tool_timeout_s=self.options.tool_timeout_s,
            tool_env=self.options.tool_env,
            # A brief's check runs whatever the run offers the model.
            harness_only=(
                {"shell_exec"} if self.options.brief_check is not None else ()
            ),
            confinement=self._confinement,
        )
        # The names of the built-ins registered above (see list_tools), and
        # R5's recall tool when its trigger is on. It is not a builtin_names()
        # member, so register_builtins never registers it.
        self._builtin_names = builtin_set_for(self.options.mode, names)
        if "tool" in self._plan.enabled():
            self.hippo.register_tool(RECALL_TOOL_DEF)
            self.executor.register(
                "recall",
                self._recall_tool,
                timeout_ms=RECALL_TOOL_TIMEOUT_S * 1000,
                process_free=True,
            )
            self._builtin_names = self._builtin_names | {"recall"}
            self._recall_tool_registered = True

    # ── Workspace, tools and conversation ──

    @property
    def workspace_root(self) -> str | None:
        """The real path of the agent's workspace root, or ``None``."""
        return self.executor.workspace_root

    @property
    def _counts_thinking(self) -> bool:
        """Whether prompt checks count the reasoning a request carries."""
        return self.options.reasoning in ("replay", "compress+replay")

    @property
    def _requires_replay(self) -> bool:
        """Whether the provider refuses a request that omits its reasoning."""
        return bool(getattr(self._llm, "requires_reasoning_replay", False))

    async def _sends_images(self) -> bool:
        """Whether tool images reach the model; ``"auto"`` asks the provider once."""
        if self._images_on is None:
            ask = getattr(self._llm, "takes_images", None)
            try:
                self._images_on = callable(ask) and await ask() is True
            except Exception:  # a failed question is an unknown answer
                logger.info("Could not ask whether the model reads images")
                self._images_on = False
        return self._images_on

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
        conversation. Explicit reasoning modes split a legacy inline
        thinking block in the loaded copies; the caller's messages are
        unchanged, and prompt-cap fitting can strip thinking before facts.

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
        loaded = copy.deepcopy(loaded)
        if self.options.reasoning != "legacy":
            # Normalize legacy checkpoints once, before prompt-cap fitting
            # tracks canonical message identities and strips thinking.
            loaded = [
                _split_inline_reasoning(msg) if msg.role == "assistant" else msg
                for msg in loaded
            ]
        working = self.hippo.working
        working.clear_all()
        for unit in group_turns(loaded, empty_prefix=working.empty_prefix):
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

    def notes(self) -> list[NoteStoreEntry]:
        """Copies of the working-memory note store's entries, oldest first.

        The note store holds the compress modes' notes, drafts and attempt
        digests (R4). It is RAM only: no session or history row keeps it.
        The entries are internal/experimental, like the note store itself.
        """
        return self.hippo.working.notes.entries()

    def load_notes(self, entries: Sequence[NoteStoreEntry]) -> int:
        """Replace the note store with copies of *entries*.

        The store is cleared first, an entry keeps its alias, and the
        store's caps still apply in their eviction order. Returns the
        number of entries kept; ``[]`` clears the store. Both work in
        RAM: R4 proposes nothing to Hippo (R6-lite adds that).

        Raises ``RuntimeError`` during a run, and ``TypeError``, before
        anything is cleared, for an item that is not a
        :class:`~localmelo.melo.memory.short.note_store.NoteStoreEntry`.
        """
        if self._active_runs:
            raise RuntimeError("Cannot load notes while a run is active")
        loaded = list(entries)
        for entry in loaded:
            if not isinstance(entry, NoteStoreEntry):
                raise TypeError(
                    "load_notes takes NoteStoreEntry objects, got "
                    f"{type(entry).__name__}"
                )
        return self.hippo.working.notes.load(loaded)

    # ── Reasoning notes (R4) ──

    @property
    def _notes_mode(self) -> bool:
        """Whether the run compresses its replies into notes."""
        return self.options.reasoning in ("compress", "compress+replay")

    # ── Recall (R5) ──

    def _retrieval_refusal(self) -> str | None:
        return retrieval_refusal(
            self.options.retriever,
            self._plan,
            self.hippo.embedding is not None,
            self.options.reasoning,
            memory=self.options.memory,
            placement=self.options.recall_placement,
            recall_at=self.options.recall_at or (),
            builtins=self.options.builtins,
        )

    def _get_retriever(self, name: str) -> Retriever:
        """The Agent's one retriever called *name*, built on first use."""
        retriever = self._retrievers.get(name)
        if retriever is not None:
            return retriever
        options: dict[str, Any] = {}
        if name == "embedding":
            options["min_score"] = self.options.recall_min_score
        retriever = build_retriever(
            name,
            send=self.chat.retrieve,
            embedding=self.hippo.embedding,
            **options,
        )
        self._retrievers[name] = retriever
        return retriever

    async def _long_docs(self, query: str, run: _RunState) -> list[Doc]:
        """The other tasks' active long entries that may answer *query*, aliased."""
        if "long" not in self._effective_kinds:
            return []
        try:
            rows = await self.hippo.recall_long(
                query,
                exclude_task_ids={run.task_id},
                pool=LONG_POOL,
                task_key=digest_task_key(run.task_text, run.place or ""),
                min_score=self.options.recall_min_score,
            )
        except ProviderError as exc:
            run.retrieve_errors += 1
            logger.info("Long-term recall of task %s failed: %s", run.task_id, exc)
            return []
        except (IndexError, ValueError) as exc:
            run.retrieve_errors += 1
            logger.info("Long-term recall of task %s failed: %s", run.task_id, exc)
            return []
        # The cosine floor applies only to an embedding score; BM25 has no
        # absolute floor, and docs_from_entries reads each row's retriever.
        docs = docs_from_entries(
            rows,
            exclude_task_ids={run.task_id},
            min_score=self.options.recall_min_score,
            include_history=self.options.typed_assertions,
        )
        key = digest_task_key(run.task_text, run.place or "")
        docs = gate_long_docs(
            docs,
            key,
            ratio=self.options.cross_task_ratio,
            cap=self.options.cross_task_cap,
            cold=self.options.cross_task_cold,
        )[:RECALL_LONG_K]
        docs = [
            dataclasses.replace(
                doc,
                meta={
                    **doc.meta,
                    "task_relation": (
                        "same"
                        if same_task(doc, key)
                        else "cross" if is_task_digest(doc) else "unscoped"
                    ),
                },
            )
            for doc in docs
        ]
        # Same-task digests first; within each group keep #162's assertion
        # order (typed mode) or the creation-time order (default).
        if self.options.typed_assertions:
            docs.sort(
                key=lambda doc: (
                    doc.meta["task_relation"] != "same",
                    -float(doc.meta.get("assertion_order_time") or 0.0),
                    -int(doc.meta.get("entry_id") or 0),
                )
            )
        else:
            docs.sort(
                key=lambda doc: (
                    doc.meta["task_relation"] != "same",
                    float(doc.meta.get("created_at") or 0.0),
                    doc.doc_id,
                )
            )
        return [
            dataclasses.replace(doc, meta={**doc.meta, "alias": f"m{index}"})
            for index, doc in enumerate(docs, start=1)
        ]

    def _hits_from_result(
        self,
        result: RetrievalResult,
        docs: Sequence[Doc],
        *,
        k: int,
    ) -> list[RecallHit]:
        """The task-local hits of *result*, at most *k*, in rank order."""
        by_id = {doc.doc_id: doc for doc in docs}
        hits: list[RecallHit] = []
        for item in result.hits:
            doc = by_id.get(item.doc_id)
            if doc is None:
                continue
            hits.append(hit_from_doc(doc, score=item.score, rank=len(hits) + 1))
            if len(hits) == k:
                break
        return hits

    def _merge_llm_hits(
        self,
        result: RetrievalResult,
        docs: Sequence[Doc],
        *,
        k: int,
        long_ids: set[str],
    ) -> list[RecallHit]:
        """The llm hits split into at most *k* task and 2 long, in its order."""
        by_id = {doc.doc_id: doc for doc in docs}
        hits: list[RecallHit] = []
        task_kept = 0
        long_kept = 0
        for item in result.hits:
            doc = by_id.get(item.doc_id)
            if doc is None:
                continue
            if doc.doc_id in long_ids:
                if long_kept >= RECALL_LONG_K:
                    continue
                long_kept += 1
                score = float(doc.meta.get("score") or item.score)
            else:
                if task_kept >= k:
                    continue
                task_kept += 1
                score = item.score
            hits.append(hit_from_doc(doc, score=score, rank=len(hits) + 1))
        return hits

    async def _run_recall(
        self,
        trigger: str,
        query: str,
        run: _RunState,
        *,
        boundary: bool = False,
        include_superseded: bool = False,
        k: int | None = None,
        placement: str = "",
        exclude: Iterable[str] = (),
    ) -> tuple[RecallRecord | None, str, list[RecallHit]]:
        """Run one recall trigger and return its record, block and hits.

        ``([], "")`` when the trigger is off, when nothing is a candidate,
        or when the block cannot be made to fit; a refused block counts
        ``recall_refused`` and returns nothing.
        """
        name = getattr(self._plan, trigger, None)
        if name is None:
            return None, "", []
        started = time.perf_counter()
        # The query can carry a tool call's command or path, so it is a
        # request path P4 must redact too: before an llm or embedding
        # retriever sees it (bm25 ranks on the raw text; decision 1).
        redacted_query, query_spans = redact_secrets(query)
        search_query = redacted_query if name in ("llm", "embedding") else query
        limit_k = (
            self.options.recall_k if k is None else max(1, min(int(k), RECALL_K_MAX))
        )
        excluded = set(exclude) | set(run.pending_shown)
        task_docs: list[Doc] = []
        if trigger != "start":
            task_docs = docs_from_notes(
                self.hippo.working.notes.entries(),
                task_id=run.task_id,
                kinds=self._effective_kinds,
                exclude_aliases=excluded,
                include_superseded=include_superseded,
            )
        long_docs = await self._long_docs(redacted_query, run)
        if long_docs:
            shown = "\n".join(
                [*recall_spans(self.hippo.working.get_window()), run.boundary_block]
            )
            long_docs = [
                doc
                for doc in long_docs
                if hit_from_doc(doc, score=0.0, rank=1).text not in shown
            ]
        if not task_docs and not long_docs:
            run.recall_empty += 1
            return None, "", []
        hits: list[RecallHit] = []
        retriever = name
        fallback = ""
        if name == "llm":
            docs = [*task_docs, *long_docs]
            long_ids = {doc.doc_id for doc in long_docs}
            result = await self._get_retriever("llm").search(
                search_query,
                docs,
                k=limit_k + min(len(long_docs), RECALL_LONG_K),
            )
            if search_query.strip():
                run.aux_retrieve_calls += 1
            self._count_retrieve_usage(result, run)
            retriever, fallback = result.retriever, result.fallback
            hits = self._merge_llm_hits(result, docs, k=limit_k, long_ids=long_ids)
        else:
            if task_docs:
                result = await self._get_retriever(name).search(
                    search_query, task_docs, k=limit_k
                )
                retriever, fallback = result.retriever, result.fallback
                self._count_retrieve_usage(result, run)
                hits = self._hits_from_result(result, task_docs, k=limit_k)
            for doc in long_docs:
                score = float(doc.meta.get("score") or 0.0)
                # The floor is a cosine floor; a BM25-scored row has none.
                if (
                    doc.meta.get("retrieval") != "bm25"
                    and score < self.options.recall_min_score - 1e-6
                ):
                    continue
                hits.append(hit_from_doc(doc, score=score, rank=len(hits) + 1))
        hits.sort(key=lambda hit: hit.task_relation != "same")
        if self.options.typed_assertions:
            long_order = {doc.doc_id: index for index, doc in enumerate(long_docs)}
            ordered_long = iter(
                sorted(
                    (hit for hit in hits if hit.kind == "long"),
                    key=lambda hit: long_order.get(hit.doc_id, len(long_order)),
                )
            )
            hits = [next(ordered_long) if hit.kind == "long" else hit for hit in hits]
        hits = [
            dataclasses.replace(hit, rank=index) for index, hit in enumerate(hits, 1)
        ]
        if fallback:
            run.recall_fallbacks += 1
            run.retrieve_errors += 1
        if trigger == "start":
            hits = [hit for hit in hits if hit.kind == "long"][:RECALL_LONG_K]
        if not hits:
            run.recall_empty += 1
            return None, "", []
        candidates = len(task_docs) + len(long_docs)
        checked_hits: list[RecallHit] = []
        for hit in hits:
            text = hit.text
            checked = text
            if hit.kind == "long":
                checked_text = self._check_candidate(
                    text, "recall", run, target="prompt"
                )
                if checked_text is None:
                    continue
                checked = checked_text
            checked_hits.append(
                dataclasses.replace(
                    hit, text=checked, chars=len(checked), rank=len(checked_hits) + 1
                )
            )
        if not checked_hits:
            return None, "", []
        if boundary:
            limit = RECALL_BOUNDARY_CHARS
        else:
            live = RECALL_LIVE_CHARS - recall_span_chars(
                self.hippo.working.get_window()
            )
            limit = min(RECALL_TURN_CHARS, max(live, 0))
        if limit <= 0:
            run.recall_skipped += 1
            return None, "", []
        # The block's bytes do not carry the latency, so fitting is stable;
        # the record below keeps it for recall_json, the event and stats.
        ms = round((time.perf_counter() - started) * 1000)
        kept, block = fit_recall_block(
            checked_hits,
            trigger=trigger,
            retriever=retriever,
            placement=placement,
            limit=limit,
        )
        if not kept or not block:
            run.recall_skipped += 1
            return None, "", []
        checked_block = self._check_recall_block(block, run)
        if checked_block is None:
            return None, "", []
        run.redactions += query_spans
        record_hits = [
            dataclasses.replace(
                hit,
                doc_id=redact_secrets(hit.doc_id)[0],
                alias=redact_secrets(hit.alias)[0],
            )
            for hit in kept
        ]
        record = RecallRecord(
            trigger=trigger,
            placement=placement,
            retriever=retriever,
            query=redacted_query,
            k=limit_k,
            candidates=candidates,
            hits=record_hits,
            chars=len(checked_block),
            ms=ms,
            fallback=fallback,
            boundary=boundary,
            step=run.steps,
            attempt=run.attempts,
        )
        run.pending_recalls.append(record_to_dict(record))
        run.pending_recall_records.append(record)
        return record, checked_block, record_hits

    @staticmethod
    def _count_retrieve_usage(result: RetrievalResult, run: _RunState) -> None:
        if not result.usage:
            return
        run.retrieve_prompt_tokens += int(result.usage.get("prompt_tokens", 0) or 0)
        run.retrieve_completion_tokens += int(
            result.usage.get("completion_tokens", 0) or 0
        )

    @staticmethod
    def _commit_recalls(records: Sequence[RecallRecord], run: _RunState) -> None:
        """Count, log and emit the recall records a request has just carried."""
        for record in records:
            run.recalls += 1
            run.recalls_by_trigger[record.trigger] = (
                run.recalls_by_trigger.get(record.trigger, 0) + 1
            )
            run.recall_hits += len(record.hits)
            counts = relation_counts(record.hits)
            run.recall_same_task += counts["same_task"]
            run.recall_cross_task += counts["cross_task"]
            run.recall_task_notes += counts["task_notes"]
            run.recall_chars += record.chars
            run.recall_ms += record.ms
            run.recall_log.append(compact_record(record))
            run.events.recall(record)

    def _check_recall_block(self, text: str, run: _RunState) -> str | None:
        """Run one whole recall block through the candidate boundary."""
        run.candidates += 1
        check = self.checker.check_memory_candidate(
            MemoryCandidatePayload(text=text, source="recall", target="prompt")
        )
        if not check.allowed:
            run.candidates_refused += 1
            run.recall_refused += 1
            logger.debug(
                "Refused a recall block of task %s: %s", run.task_id, check.reason
            )
            return None
        payload = check.sanitized_payload
        run.redactions += int(payload.metadata.get("redactions", 0) or 0)
        checked = str(payload.text)
        if not checked.endswith(RECALL_END):
            run.recall_refused += 1
            logger.debug(
                "Refused a recall block of task %s: the boundary cut its "
                "closing delimiter",
                run.task_id,
            )
            return None
        return checked

    async def _step_recall(
        self,
        run: _RunState,
        note_entry: NoteStoreEntry | None,
        calls: Sequence[ToolCall],
        *,
        fallback: str,
        exclude: Iterable[str] = (),
    ) -> tuple[str, list[RecallHit]]:
        """The step trigger's block for the next planning call, and its hits.

        A note from an earlier attempt is ``superseded`` at that attempt's
        boundary, but its turn can still be evicted from the window, and
        R5's step trigger is what makes an evicted turn reachable again;
        step recall therefore ranks superseded notes too. The window's
        aliases still exclude what the request already shows.
        """
        if self._plan.step is None:
            return "", []
        note_text = "" if note_entry is None else note_entry.text
        outcome = ""
        if note_entry is not None and note_entry.note is not None:
            outcome = note_entry.note.outcome
        query = build_step_query(note_text, calls, fallback=fallback, outcome=outcome)
        _record, block, hits = await self._run_recall(
            "step",
            query,
            run,
            placement=self.options.recall_placement,
            exclude=exclude,
            include_superseded=True,
        )
        if block:
            run.pending_shown.update(hit.alias for hit in hits if hit.kind != "long")
        return block, hits

    async def _store_thought_chunks(
        self, response: Message, run: _RunState
    ) -> list[str]:
        """Store up to the task's remaining thought chunks from *response*.

        Returns the aliases of the chunks stored, so the caller can mark
        them shown with the reply's turn.
        """
        if "thought" not in self._effective_kinds or not self._notes_mode:
            return []
        stored = sum(
            1
            for entry in self.hippo.working.notes.of_kind("thought")
            if entry.task_id == run.task_id
        )
        remaining = RECALL_THOUGHT_PER_TASK - max(run.thought_chunks, stored)
        if remaining <= 0 or not response.thinking:
            return []
        aliases: list[str] = []
        for chunk in thought_chunks(response.thinking)[:remaining]:
            checked = self._check_candidate(chunk.text, "thought", run)
            if checked is None:
                continue
            entry = self.hippo.working.notes.add(
                "thought",
                checked,
                step=run.steps,
                attempt=run.attempts,
                task_id=run.task_id,
                title=_checked_title(checked),
            )
            if entry is not None:
                run.thought_chunks += 1
                aliases.append(entry.alias)
        return aliases

    async def _recall_tool(self, query: str, k: int | None = None) -> str:
        """The ``recall`` tool: rank the task's documents for *query*."""
        run = RECALL_RUN.get()
        if run is None:
            return "recall is only available inside a run"
        if not isinstance(query, str) or not query.strip():
            return "recall needs a non-empty query"
        if isinstance(k, bool) or not isinstance(k, int):
            k = self.options.recall_k
        refused_before = run.recall_refused
        _record, block, hits = await self._run_recall(
            "tool",
            query,
            run,
            include_superseded=True,
            k=k,
            placement="tool-result",
            exclude=self.hippo.working.shown_aliases(),
        )
        if not block:
            if run.recall_refused > refused_before:
                return "recall refused: the block did not pass the boundary"
            return "recall found nothing for that query"
        run.pending_shown.update(hit.alias for hit in hits if hit.kind != "long")
        return block

    @staticmethod
    def _last_call_seq() -> int:
        """The ``llm_calls.seq`` of the reply that just arrived, or 0."""
        recorder = CALL_RECORDER.get()
        return int(getattr(recorder, "last_seq", 0) or 0)

    def _reply_note(
        self,
        response: Message,
        results: Sequence[ToolResult],
        *,
        kind: str,
        task_id: str,
        run: _RunState,
    ) -> tuple[ReasoningNote, str]:
        """Build the note of *response* and the content without its note block."""
        window_ids = {
            message.tool_call_id
            for message in self.hippo.working.get_window()
            if message.role == "tool" and message.tool_call_id
        }
        window_records, task_records = run.source_log.earlier(window_ids)
        note, rest = note_from_reply(
            response.content,
            response.thinking,
            response.tool_calls,
            results,
            kind=kind,
            writer=self.options.note_writer,
            window_records=window_records,
            task_records=task_records,
        )
        note = dataclasses.replace(
            note,
            note_id=new_record_id(),
            task_id=task_id,
            step=run.steps,
            attempt=run.attempts,
            call_seq=self._last_call_seq(),
            created_at=time.time(),
        )
        return note, rest

    def _check_candidate(
        self, text: str, source: str, run: _RunState, *, target: str = "candidate"
    ) -> str | None:
        """Run one document through the candidate boundary; its checked text."""
        run.candidates += 1
        check = self.checker.check_memory_candidate(
            MemoryCandidatePayload(text=text, source=source, target=target)
        )
        if not check.allowed:
            run.candidates_refused += 1
            logger.debug(
                "Dropped a %s candidate of task %s: %s",
                source,
                run.task_id,
                check.reason,
            )
            return None
        payload = check.sanitized_payload
        run.redactions += int(payload.metadata.get("redactions", 0) or 0)
        return str(payload.text)

    def _checked_candidate_text(
        self, text: str, source: str, run: _RunState
    ) -> str | None:
        """A candidate text through the boundary, counted as one boundary check.

        The note or draft it comes from was counted when it entered its own
        store; this pass is the one R6-lite requires before a candidate is
        staged, so the policy never compares raw text. ``candidates`` counts
        boundary checks, not documents (review finding 11).
        """
        run.candidates += 1
        check = self.checker.check_memory_candidate(
            MemoryCandidatePayload(text=text, source=source, target="candidate")
        )
        if not check.allowed:
            run.candidates_refused += 1
            return None
        payload = check.sanitized_payload
        run.redactions += int((payload.metadata or {}).get("redactions", 0) or 0)
        return str(payload.text)

    def _memory_provenance(
        self,
        run: _RunState,
        *,
        note: ReasoningNote | None = None,
        calls: Sequence[ToolCall] = (),
    ) -> dict[str, Any]:
        """The provenance a candidate records about the turn that made it."""
        paths: list[str] = []
        commands: list[str] = []
        for call in calls:
            for name in self._touched_names(call):
                bucket = paths if path_spans(name) else commands
                if name not in bucket:
                    bucket.append(name)
        source_ids = list(note.source_ids) if note is not None else []
        return {
            "task_id": run.task_id,
            "session_id": "",
            "step": run.steps,
            "attempt": run.attempts,
            "call_seq": self._last_call_seq(),
            "task_text": run.task_text[:300],
            "paths": paths,
            "commands": commands,
            "source_ids": source_ids,
            "note_id": note.note_id if note is not None else "",
        }

    def _propose_note_candidate(
        self,
        note: ReasoningNote,
        run: _RunState,
        calls: Sequence[ToolCall] = (),
    ) -> None:
        """Stage a rememberable note's items for long-term memory."""
        if not note.remember or not (note.found or note.decided):
            return
        text = self._checked_candidate_text(candidate_text(note), "reasoning_note", run)
        if text is None:
            return
        items: list[CandidateItem] = []
        for item in candidate_items(note):
            checked = self._checked_candidate_text(item.text, "reasoning_note", run)
            if checked is not None:
                evidence = None
                if self.options.typed_assertions and item.source_ids:
                    evidence = next(
                        (
                            body
                            for record_id, body in note.source_records
                            if record_id in item.source_ids
                        ),
                        None,
                    )
                items.append(CandidateItem(checked, item.source_ids, evidence))
        candidate = MemoryCandidate(
            candidate_id=note.note_id,
            source="reasoning_note",
            text=text,
            remember_by=note.remember_by or "rule",
            task_id=run.task_id,
            session_id="",
            provenance=self._memory_provenance(run, note=note, calls=calls),
            created_at=note.created_at,
            remember=True,
            items=tuple(items),
        )
        self.hippo.propose(candidate)

    def _drain_decisions(
        self, decisions: Sequence[MemoryDecision], run: _RunState
    ) -> None:
        """Count the decisions an admission made and emit one event each."""
        for decision in decisions:
            if decision.refused:
                run.candidates_refused += 1
                continue
            run.redactions += decision.redactions
            run.admitted[decision.admitted_by] = (
                run.admitted.get(decision.admitted_by, 0) + 1
            )
            if decision.action == "IGNORE":
                run.ignored += 1
            else:
                run.written[decision.kind] = run.written.get(decision.kind, 0) + 1
            run.events.memory(
                candidate_id=decision.candidate_id,
                action=decision.action,
                kind=decision.kind,
                admitted_by=decision.admitted_by,
                entry_id=decision.entry_id,
            )

    def _propose_reflection(
        self, entry: ReflectionEntry, run: _RunState, *, note_id: str = ""
    ) -> None:
        """Stage the final reflection, with its next directions and open questions."""
        candidate_id = note_id or new_record_id()
        text = self._checked_candidate_text(
            reflection_candidate_text(entry), "reflection", run
        )
        if text is None:
            return
        provenance = self._memory_provenance(run, note=None)
        provenance["note_id"] = candidate_id
        candidate = MemoryCandidate(
            candidate_id=candidate_id,
            source="reflection",
            text=text,
            remember_by="rule",
            task_id=run.task_id,
            session_id="",
            provenance=provenance,
            created_at=time.time(),
            remember=True,
        )
        run.final_reflection = candidate
        self.hippo.propose(candidate)

    def _store_note(
        self,
        note: ReasoningNote,
        run: _RunState,
        *,
        source: str,
        drafts: Sequence[str] = (),
        calls: Sequence[ToolCall] = (),
    ) -> NoteStoreEntry | None:
        """Check and store one note; emit its ``note`` event."""
        text = render_note(note, executor=True)
        checked = self._check_candidate(text, source, run)
        if checked is None:
            return None
        entry = self.hippo.working.notes.add(
            "note",
            checked,
            note=note,
            stage=note.stage,
            step=note.step,
            attempt=note.attempt,
            remember=note.remember,
            remember_by=note.remember_by,
            task_id=note.task_id,
            title=note_title(note),
        )
        if entry is None:
            run.candidates_refused += 1
            return None
        run.notes += 1
        run.note_chars += len(checked)
        if note.source == "model":
            run.notes_model += 1
        elif note.source == "heuristic":
            run.notes_heuristic += 1
        if note.remember and note.remember_by == "model":
            run.remember_model += 1
        elif note.remember and note.remember_by == "heuristic":
            run.remember_heuristic += 1
        run.events.note(
            step=note.step,
            attempt=note.attempt,
            note_id=note.note_id,
            kind=note.kind,
            source=note.source,
            text=checked,
            chars=len(checked),
            drafts=list(drafts),
            remember=note.remember,
            stage=entry.stage,
        )
        if source == "reasoning_note":
            self._propose_note_candidate(note, run, calls)
        return entry

    def _store_draft(self, draft: Draft, run: _RunState) -> NoteStoreEntry | None:
        """Check and store one draft, superseding older pending ones."""
        checked = self._check_candidate(draft.text, "draft", run)
        if checked is None:
            return None
        entry = self.hippo.working.notes.add(
            "draft",
            checked,
            draft=draft,
            step=draft.step,
            attempt=run.attempts,
            task_id=draft.task_id,
            title=draft_title(draft),
        )
        if entry is None:
            run.candidates_refused += 1
            return None
        run.drafts += 1
        self.hippo.working.notes.supersede_drafts(draft.target, keep=entry.alias)
        return entry

    async def _prepare_note_turn(
        self,
        response: Message,
        step: _InflightStep,
        pairs: Sequence[tuple[ToolCall, ToolResult]],
        run: _RunState,
    ) -> tuple[str, str, NoteStoreEntry | None, NoteStoreEntry | None]:
        """Build and store a tool-call reply's draft and note; place them.

        Returns the working-memory content and thinking of the turn, and
        the stored note and draft entries (either may be ``None`` when the
        boundary or the store refuses it). The draft is marked written
        first, for this reply's writes and any earlier pending draft they
        cover, so a draft written in the same reply is never pinned.
        """
        store = self.hippo.working.notes
        pending = [
            entry.draft
            for entry in store.of_kind("draft")
            if entry.draft is not None and entry.draft.status == "pending"
        ]
        run.drafts_written += mark_drafts_written(pending, pairs)
        draft_entry = None
        draft = extract_draft(step.rest, recent_path=step.recent_path)
        if draft is not None:
            draft = dataclasses.replace(
                draft,
                draft_id=new_record_id(),
                task_id=run.task_id,
                step=run.steps,
                call_seq=self._last_call_seq(),
                created_at=time.time(),
            )
            run.drafts_written += mark_drafts_written([draft], pairs)
            draft_entry = self._store_draft(draft, run)
        note, _rest = self._reply_note(
            response,
            [result for _call, result in pairs],
            kind="step",
            task_id=run.task_id,
            run=run,
        )
        aliases = [] if draft_entry is None else [draft_entry.alias]
        note_entry = self._store_note(
            note,
            run,
            source="reasoning_note",
            drafts=aliases,
            calls=[call for call, _result in pairs],
        )
        # The note is built: these observations are earlier results for the
        # next reply's note, never this one's.
        for call, result in pairs:
            run.source_log.add(call.id, source_records([(call, result)]))
        if note_entry is not None:
            await self._restate(note, run)
        note_text = "" if note_entry is None else note_entry.text
        if step.has_model_note:
            content = response.content
        elif not step.rest.strip():
            content = note_text
        elif note_text and len(step.rest) + 2 + len(note_text) <= ASSISTANT_TEXT_SHARE:
            content = f"{step.rest}\n\n{note_text}"
        else:
            content = step.rest
        thinking = "" if self.options.reasoning == "compress" else response.thinking
        return content, thinking, note_entry, draft_entry

    def _pin_pending_draft(
        self,
        tool_msgs: list[Message],
        draft_entry: NoteStoreEntry | None,
        run: _RunState,
    ) -> bool:
        """Append the turn's pending targeted draft to its last tool result.

        Returns whether the draft was placed, so the step block skips it.
        """
        if (
            draft_entry is None
            or draft_entry.draft is None
            or draft_entry.draft.status != "pending"
            or not draft_entry.draft.target
            or not tool_msgs
        ):
            return False
        block = render_draft_block(
            draft_entry.alias,
            draft_entry.step,
            draft_entry.draft,
            text=draft_entry.text,
        )
        checked = self._check_candidate(block, "draft", run)
        if checked:
            last = tool_msgs[-1]
            tool_msgs[-1] = dataclasses.replace(
                last, content=f"{last.content}\n{checked}"
            )
            return True
        return False

    @staticmethod
    def _checked_note_copy(entry: NoteStoreEntry) -> ReasoningNote:
        """*entry*'s note with only the fields its checked text still holds.

        The digest is built from the checked text the store keeps, never
        from raw fields a redaction or a cap changed: an item whose raw
        text is not in ``entry.text`` is left out, and the rest is a copy
        that the digest may read.
        """
        note = entry.note
        assert note is not None
        text = entry.text

        def kept(items: Sequence[str]) -> list[str]:
            return [item for item in items if item and item in text]

        return dataclasses.replace(
            note,
            plan="",
            found=kept(note.found),
            decided=kept(note.decided),
            open_questions=kept(note.open_questions),
            observed=kept(note.observed),
            outcome=note.outcome if note.outcome and note.outcome in text else "",
            stage="rehearsed",
        )

    def _build_digest(
        self, query: str, outcome: str, run: _RunState, task_id: str
    ) -> str:
        """Build, check and store the digest of the attempt that just ended."""
        notes = [
            self._checked_note_copy(entry)
            for entry in self.hippo.working.notes.of_kind("note")
            if entry.attempt == run.attempts
            and entry.stage == "rehearsed"
            and entry.note is not None
        ]
        note = digest_note(
            query,
            outcome or run.status,
            notes,
            task_id=task_id,
            attempt=run.attempts,
            step=run.steps,
            calls=run.attempt_calls,
            root=run.place or "",
        )
        note = dataclasses.replace(
            note,
            note_id=new_record_id(),
            call_seq=self._last_call_seq(),
            created_at=time.time(),
        )
        checked = self._check_candidate(render_digest(note), "digest", run)
        if checked is None:
            return ""
        entry = self.hippo.working.notes.add(
            "digest",
            checked,
            note=note,
            step=note.step,
            attempt=note.attempt,
            remember=True,
            remember_by="rule",
            task_id=task_id,
            title=note_title(note),
        )
        if entry is None:
            run.candidates_refused += 1
            return ""
        run.digests += 1
        run.events.note(
            step=note.step,
            attempt=note.attempt,
            note_id=note.note_id,
            kind="digest",
            source=note.source,
            text=checked,
            chars=len(checked),
            drafts=[],
            remember=True,
            stage=entry.stage,
        )
        provenance = self._memory_provenance(run, note=note)
        # Key the digest by the run's task, the same input recall hashes: the
        # checked text may have escaped a role token or cut a long plan, and a
        # key read from it would no longer match this run's task.
        provenance["task_key"] = digest_task_key(query, run.place or "")
        candidate = MemoryCandidate(
            candidate_id=note.note_id,
            source="digest",
            text=checked,
            remember_by="rule",
            task_id=task_id,
            session_id="",
            provenance=provenance,
            created_at=note.created_at,
            remember=True,
        )
        if digest_has_evidence(checked):
            # Keep the run's last evidence-bearing attempt. A final attempt
            # without evidence must not drop an episode whose earlier attempt
            # wrote a file or passed a check.
            run.final_digest = candidate
            self.hippo.propose(candidate)
        return checked

    async def _restate(self, note: ReasoningNote, run: _RunState) -> bool:
        """Compact *note*'s new items into an older note with the same facts.

        A later note whose ``found`` or ``decided`` items match an existing
        note's by normalized text restates it: the new items join the older
        note (deduplicated, caps kept), its rendered text passes the
        boundary again, and its stage becomes ``rehearsed``. Returns
        whether a restatement happened.
        """
        items = [*note.found, *note.decided]
        if not items:
            return False
        new_keys = {" ".join(item.split()).lower() for item in items}
        for entry in self.hippo.working.notes.of_kind("note"):
            older = entry.note
            if older is None or older is note or entry.stage == "superseded":
                continue
            if not any(
                " ".join(item.split()).lower() in new_keys
                for item in (*older.found, *older.decided)
            ):
                continue
            found = list(older.found)
            decided = list(older.decided)
            old_keys = {" ".join(item.split()).lower() for item in (*found, *decided)}
            for item in note.found:
                key = " ".join(item.split()).lower()
                if key not in old_keys and len(found) < NOTE_MAX_FOUND:
                    found.append(item)
                    old_keys.add(key)
            for item in note.decided:
                key = " ".join(item.split()).lower()
                if key not in old_keys and len(decided) < NOTE_MAX_DECIDED:
                    decided.append(item)
                    old_keys.add(key)
            source_ids = list(older.source_ids)
            for source_id in note.source_ids:
                if source_id not in source_ids:
                    source_ids.append(source_id)
            source_records = list(older.source_records)
            seen = {record_id for record_id, _text in source_records}
            for record in note.source_records:
                if record[0] not in seen:
                    source_records.append(record)
                    seen.add(record[0])
            remember = older.remember or note.remember
            remember_by = older.remember_by if older.remember else note.remember_by
            updated = dataclasses.replace(
                older,
                found=found,
                decided=decided,
                source_ids=source_ids,
                source_records=source_records,
                remember=remember,
                remember_by=remember_by,
                stage="rehearsed",
                uses=older.uses + 1,
                last_used_at=time.time(),
            )
            checked = self._check_candidate(
                render_note(updated, executor=True), "reasoning_note", run
            )
            if checked is None:
                return False
            if not self.hippo.working.notes.replace_text(entry.alias, checked, updated):
                return False
            self.hippo.working.notes.touch(entry.alias, use=True, stage="rehearsed")
            run.rehearsed["restated"] += 1
            # The new note's candidate is superseded by the merged one; a
            # request would otherwise be able to admit both (review finding 8).
            self.hippo.staged.drop(note.note_id)
            self._propose_note_candidate(updated, run)
            decisions = await self.hippo.admit(updated.note_id, "rehearsal")
            self._drain_decisions(decisions, run)
            return True
        return False

    @staticmethod
    def _call_succeeded(
        call: ToolCall, result: ToolResult, exit_codes: Sequence[int]
    ) -> bool:
        """Whether a call returned without an error and with exit status 0.

        *exit_codes* are the statuses the tool recorded in ``TOOL_EXIT_CODE``
        (see :meth:`_do_execute`). A rooted ``shell_exec`` prefixes a
        non-zero status in its text; an unrooted one records none, so its
        status is unknown there and the call is not counted as a success:
        a rehearsal signal must not come from a command that may have
        failed.
        """
        if result.error.strip():
            return False
        if call.tool_name != "shell_exec":
            return True
        if result.output.startswith("[exit code "):
            return result.output.startswith("[exit code 0]")
        return bool(exit_codes) and exit_codes[-1] == 0

    @staticmethod
    def _touched_names(call: ToolCall) -> list[str]:
        """The paths and command words a successful call acted on."""
        names: list[str] = []
        if call.tool_name in ("file_read", "file_write"):
            path = call.arguments.get("path")
            if isinstance(path, str) and path.strip():
                names.append(path)
                names.extend(path[start:end] for start, end in path_spans(path))
        elif call.tool_name == "shell_exec":
            command = call.arguments.get("command")
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            if isinstance(command, str) and command.strip():
                names.append(command)
                names.extend(command[start:end] for start, end in path_spans(command))
                head = command.split()[0]
                if head:
                    names.append(head)
        return [name for name in names if len(name) >= 3]

    @staticmethod
    def _mentions(text: str, name: str) -> bool:
        """Whether *text* names *name* as a path or a whole word.

        A path or file name is compared with :func:`same_path` against
        the paths :func:`~localmelo.melo.memory.notes.path_spans` finds in
        *text*; any other name (a command word) must stand between
        non-word characters, so ``ls`` does not match ``results``.
        """
        if path_spans(name):
            return any(
                same_path(text[start:end], name) for start, end in path_spans(text)
            )
        start = 0
        while True:
            index = text.find(name, start)
            if index < 0:
                return False
            before = text[index - 1] if index else ""
            after = text[index + len(name) : index + len(name) + 1]
            if not (before.isalnum() or before == "_") and not (
                after.isalnum() or after == "_"
            ):
                return True
            start = index + 1

    async def _rehearse_acted(
        self,
        call: ToolCall,
        result: ToolResult,
        run: _RunState,
        exit_codes: Sequence[int] = (),
    ) -> None:
        """Mark the notes a successful call acted on as rehearsed."""
        if not self._call_succeeded(call, result, exit_codes):
            return
        touched = self._touched_names(call)
        if not touched:
            return
        for entry in self.hippo.working.notes.of_kind("note"):
            note = entry.note
            if note is None or entry.stage == "superseded":
                continue
            text = " ".join((*note.found, *note.decided))
            if entry.stage == "rehearsed":
                continue
            if not any(self._mentions(text, name) for name in touched):
                continue
            note.stage = "rehearsed"
            self.hippo.working.notes.touch(entry.alias, stage="rehearsed")
            run.rehearsed["acted"] += 1
            decisions = await self.hippo.admit(note.note_id, "rehearsal")
            self._drain_decisions(decisions, run)

    async def _cite_notes(self, aliases: Sequence[str], run: _RunState) -> None:
        """Mark the notes a reflection cited in ``used_notes`` as rehearsed."""
        for alias in aliases:
            entry = self.hippo.working.notes.get(alias)
            if entry is None or entry.kind != "note" or entry.stage == "superseded":
                continue
            if entry.note is not None:
                entry.note.stage = "rehearsed"
            self.hippo.working.notes.touch(alias, use=True, stage="rehearsed")
            run.rehearsed["cited"] += 1
            if entry.note is not None:
                decisions = await self.hippo.admit(entry.note.note_id, "rehearsal")
                self._drain_decisions(decisions, run)

    @staticmethod
    def _used_notes(content: str, flags: bool) -> list[str]:
        """The string aliases of the reflection reply's ``used_notes``.

        The object is the one the gate's parser reads: with ``"flags"``
        the first object with a valid ``recommended_action`` (see
        :func:`_read_reflection`), with ``"utility"`` the first JSON
        object. A non-list or non-string entries are ignored.
        """
        data: dict[str, Any] | None
        if flags:
            found = _read_reflection(_coerce_str(content))
            data = found[0] if found is not None else None
        else:
            try:
                value = _decode_reflection(_coerce_str(content))
            except ValueError:
                return []
            if not isinstance(value, dict):
                return []
            data = value
        if data is None:
            return []
        names = data.get("used_notes")
        if not isinstance(names, list):
            return []
        return [name for name in names if isinstance(name, str) and name]

    def _reflection_note_inputs(self, run: _RunState) -> tuple[list[str], list[str]]:
        """The notes lines and draft-title lines a reflection request carries."""
        notes = [
            entry
            for entry in self.hippo.working.notes.of_kind("note")
            if entry.attempt == run.attempts
            and entry.stage != "superseded"
            and entry.note is not None
        ][-REFLECTION_MAX_NOTES:]
        lines = [
            f"[{entry.alias}] note · step {entry.step} · "
            f"{entry.text[:REFLECTION_NOTE_CHARS]}"
            for entry in notes
        ]
        # A pending draft is store-global: an earlier attempt's draft that
        # was never written is still one the next step may need.
        drafts = [
            entry
            for entry in self.hippo.working.notes.of_kind("draft")
            if entry.draft is not None and entry.draft.status == "pending"
        ]
        titles: list[str] = []
        for entry in drafts:
            if entry.draft is None:
                continue
            title = (
                f"[{entry.alias}] draft · step {entry.step} · "
                f"{draft_title(dataclasses.replace(entry.draft, text=entry.text))}"
            )
            # The target is raw metadata; the line reaches a prompt only
            # after the candidate boundary.
            checked = self._check_candidate(title, "draft", run)
            if checked:
                titles.append(checked)
        return lines, titles

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
        twice, and none starts once :meth:`close` has begun. A tool-capture
        hook (see ``TOOL_ENTRY``) runs before the first provider starts,
        because a provider can spawn a child of its own.
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
                capture = TOOL_ENTRY.get()
                if capture is not None:
                    # A provider may spawn a child of its own (an MCP stdio
                    # server) before any tool call exists to mark: record the
                    # execution incomplete before the first start.
                    await capture()
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
            self._grouped_names = frozenset(
                tool.definition.name for tool in provided if tool.group
            )
            self.chat.add_system_notes(
                [note for provider in started for note in provider.system_notes()]
            )
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

    async def _fit_prompt_cap(
        self,
        others: list[Message],
        query: str,
        *,
        task_query: Message | None = None,
        query_block: str = "",
        run: _RunState | None = None,
    ) -> None:
        """Drop or strip the oldest working memory the planning prompt cannot hold.

        ``Checker.pre_plan`` refuses a step whose prompt, the working-memory
        window plus *others* (long-term recall and the reflection), is
        over its cap. The window is measured as the planning request
        carries it: once the unit that holds the task *query* is dropped,
        the request puts the query back in front (see :func:`_as_planned`),
        so dropping that unit saves nothing.

        In ``replay`` on a provider that does not require replay, the
        oldest units' thinking is cleared first (see
        :meth:`WorkingMemory.strip_thinking`), and the count that fits is
        committed. Only when no such strip fits does the whole-unit loop
        run. On a replay-required provider that loop runs directly, with
        thinking counted: a unit is dropped whole and no reasoning is ever
        cut or skipped. ``legacy`` and ``drop`` run the loop as before.
        The newest unit always stays.
        """
        units = group_turns(
            self.hippo.working.get_window(),
            empty_prefix=self.hippo.working.empty_prefix,
        )

        async def fits(kept: list[Message], target: float = 1.0) -> bool:
            sent = _as_sent(
                kept,
                self.options,
                task_query=task_query,
                requires_replay=self._requires_replay,
            )
            sent = self._tool_images.hydrate(sent)
            planned = _as_planned(sent, query, query_block)
            if run is not None and (run.inbox is not None or run.brief):
                planned = coalesce_user_messages(planned)
            check = await self.checker.pre_plan(
                others + planned,
                count_thinking=self._counts_thinking,
                cap_ratio=target,
            )
            return check.allowed

        if not await fits(self.hippo.working.get_window()):
            if (
                self.options.reasoning in ("replay", "compress+replay")
                and not self._requires_replay
            ):
                carried = {
                    id(msg)
                    for msg in _as_sent(
                        self.hippo.working.get_window(),
                        self.options,
                        task_query=task_query,
                        requires_replay=self._requires_replay,
                    )
                    if msg.role == "assistant" and msg.thinking
                }
                for first in range(1, len(units)):
                    kept = [
                        (
                            dataclasses.replace(msg, thinking="")
                            if index < first
                            and msg.role == "assistant"
                            and msg.thinking
                            and id(msg) in carried
                            else msg
                        )
                        for index, unit in enumerate(units)
                        for msg in unit
                    ]
                    if not await fits(kept):
                        continue
                    changed = self.hippo.working.strip_thinking(first, only=carried)
                    if run is not None:
                        run.replay_stripped += changed
                    logger.info(
                        "Cleared the thinking of %d working-memory message(s) "
                        "so that the planning prompt fits",
                        changed,
                    )
                    return
            for first in range(len(units)):
                kept = [msg for unit in units[first:] for msg in unit]
                if await fits(kept):
                    break
            else:
                return  # not even the newest unit fits: keep the window
            if first and self._evicts_in_chunks:
                # Drop to WINDOW_FIT_TARGET of the cap, so the next steps fit
                # without dropping again (when no such cut fits, keep *first*).
                for deeper in range(first + 1, len(units)):
                    kept = [msg for unit in units[deeper:] for msg in unit]
                    if await fits(kept, WINDOW_FIT_TARGET):
                        first = deeper
                        break
            if first:
                dropped = self.hippo.working.drop_oldest_units(first)
                logger.info(
                    "Dropped the %d oldest working-memory messages so that the "
                    "planning prompt fits",
                    dropped,
                )

    async def _do_retrieval(
        self,
        query: str,
        *,
        task_query: Message | None = None,
        run: _RunState | None = None,
    ) -> tuple[list[Message], list[Message], list[ToolDef], str | None]:
        """Stage 1+2: retrieve context, resolve tools, run boundary checks.

        With ``tool_exposure="retrieve"`` the tools are every registered
        built-in, sorted by name, then the other tools that the model's
        tool calls and BM25 resolve among themselves, also sorted by name;
        with ``"all"`` they are every registered tool, sorted by name.

        In ``drop`` and ``replay`` no long-term recall is read, so the
        system prompt has no ``[RECALL]`` section; the long-term writes are
        unchanged. Before anything reads the working-memory window, its
        oldest units are dropped or stripped when the planning prompt would
        not fit ``Checker.pre_plan``'s cap otherwise (see
        :meth:`_fit_prompt_cap`).

        Returns (long_context, short_window, tools, fail_reason).
        *fail_reason* is non-None when a check fails and the loop must break.
        """
        if self.options.reasoning == "legacy":
            long_context = await self.hippo.retrieve_context(query)
        else:
            long_context = []
        if (
            run is not None
            and not run.boundary_done
            and run.steps == 0
            and run.attempts == 1
            and self._plan.start is not None
        ):
            run.boundary_done = True
            _record, block, _hits = await self._run_recall(
                "start",
                build_start_query(query),
                run,
                boundary=True,
                placement="query",
            )
            run.boundary_block = block
        query_block = "" if run is None else run.boundary_block
        # In the window already when the layout keeps the reflection there.
        reflection_msgs = (
            []
            if self._reflection_in_window
            else _serialize_reflections(self.hippo.working.get_reflections())
        )
        await self._fit_prompt_cap(
            long_context + reflection_msgs,
            query,
            task_query=task_query,
            query_block=query_block,
            run=run,
        )
        short_window = self.hippo.working.get_window()

        tool_hints: list[str] = []
        if self.options.tool_exposure == "all":
            # The same list on every step, so the tools part of the request
            # stays byte-stable for prefix caching.
            if self.options.fixed_prefix and run is not None:
                if run.fixed_tools is None:
                    run.fixed_tools = copy.deepcopy(
                        sorted(self.hippo.list_tools(), key=lambda t: t.name)
                    )
                tools = copy.deepcopy(run.fixed_tools)
            else:
                tools = sorted(self.hippo.list_tools(), key=lambda t: t.name)
        else:
            # Every registered built-in, sorted by name, on every step. Hints
            # and BM25 choose among the other tools only, which follow,
            # sorted by name too: the list stays byte-identical for as long
            # as the chosen set does not change, whatever the hint order.
            builtins = self._builtin_names
            # A provider's tool group is offered like the built-ins, after them.
            offered = builtins | self._grouped_names
            # Hints are the model's tool calls, which live in the window; the
            # long-term and reflection messages carry none.
            hint_sources = long_context + short_window + reflection_msgs
            tool_hints = [
                name
                for name in self.hippo.extract_tool_hints(hint_sources)
                if name not in offered
            ]
            tools = sorted(
                (t for t in self.hippo.list_tools() if t.name in builtins),
                key=lambda t: t.name,
            )
            if self._grouped_names:
                tools += sorted(
                    (
                        t
                        for t in self.hippo.list_tools()
                        if t.name in self._grouped_names
                    ),
                    key=lambda t: t.name,
                )
            tools += sorted(
                self.hippo.resolve_tools(query, hints=tool_hints, exclude=offered),
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
        sent = _as_sent(
            short_window,
            self.options,
            task_query=task_query,
            requires_replay=self._requires_replay,
        )
        sent = self._tool_images.hydrate(sent)
        planned = _as_planned(sent, query, "" if run is None else run.boundary_block)
        if run is not None and (run.inbox is not None or run.brief):
            planned = coalesce_user_messages(planned)
        all_msgs = long_context + planned + reflection_msgs
        check = await self.checker.pre_plan(
            all_msgs, count_thinking=self._counts_thinking
        )
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
        *,
        task_query: Message | None = None,
        run: _RunState | None = None,
        on_ready: Callable[[list[Message]], None] | None = None,
    ) -> tuple[Message, str | None]:
        """Stage 3: LLM planning step with post-plan check.

        The request carries :func:`_as_sent`'s window. In ``replay`` the
        characters of thinking it carries are counted, and a turn skipped
        for the per-turn cap is counted too; in ``drop`` both are zero; in
        ``legacy`` neither counter is used. An explicit-mode reply's inline
        ``<think>`` block is moved into thinking before the run reads it,
        so the result and tool-hint scan see the clean answer.

        Returns (response, fail_reason).
        """
        sent, skipped = _replay_plan(
            short_window,
            self.options,
            task_query=task_query,
            requires_replay=self._requires_replay,
        )
        if run is not None:
            run.replayed_chars += sum(
                len(msg.thinking) for msg in sent if msg.role == "assistant"
            )
            run.replay_skipped += len(skipped)
        sent = self._tool_images.hydrate(sent)
        recalls = []
        recall_records: list[RecallRecord] = []
        query_block = ""
        if run is not None:
            query_block = run.boundary_block
            recalls = run.pending_recalls
            recall_records = run.pending_recall_records
            run.pending_recalls = []
            run.pending_recall_records = []
        plan_args: dict[str, Any] = {}
        if run is not None and (run.inbox is not None or run.brief):
            plan_args["coalesce_users"] = True
        if on_ready is not None:
            plan_args["on_ready"] = on_ready
        elif run is not None and run.brief and run.dropped:
            # Not at a step that carries inbox lines: they must end the request.
            plan_args["tail_note"] = render_dropped(run.dropped, sent)
        response = await self.chat.plan_step(
            context=long_context,
            short=sent,
            tools=tools,
            query=query,
            reflections=reflections,
            query_block=query_block,
            recalls=recalls,
            **plan_args,
        )
        if run is not None:
            run.plan_call = getattr(self.chat, "last_call", None)
        if run is not None and recall_records:
            self._commit_recalls(recall_records, run)
        if self.options.reasoning != "legacy":
            before = response.thinking
            response = _split_inline_reasoning(response)
            if not before and response.thinking:
                self.chat.stats.reasoning_chars += len(response.thinking)

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
        # A list for an event, and in the compress modes for the rehearsal
        # signal too; elsewhere, without a sink, the slot stays unset as
        # before.
        exit_codes: list[int] | None = [] if events.active or self._notes_mode else None
        slot = TOOL_EXIT_CODE.set(exit_codes) if exit_codes is not None else None
        gate = None
        ask_slot = None
        if run.approval is not None:
            called_tool = self.hippo.get_tool(call.tool_name)
            tier: ToolTier = (
                "write" if called_tool is None else self.checker.tier(called_tool)
            )
            gate = _ApprovalGate(
                run.approval,
                run,
                call,
                self.executor.workspace_root,
                tier,
                self.checker,
            )
            ask_slot = TOOL_ASK.set(
                _ToolAsk(run.approval, run, call, self.executor.workspace_root)
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
            captured = None
            if self.options.captured_history and run.inflight is not None:
                kept, notes = _sanitize_images(outcome.images)
                original = outcome.to_tool_result()
                original.images = kept
                result, cut = preview_result(original, notes)
                captured = capture_result(original)
                captured[1]["images_omitted"] = len(outcome.images) - len(kept)
                captured[1]["status"] = outcome.status.value
                captured[1]["error_category"] = outcome.error_category.value
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
            if ask_slot is not None:
                TOOL_ASK.reset(ask_slot)
        exported = (
            preview_result(captured[0], notes)[0] if captured is not None else result
        )
        events.tool_finished(
            call,
            outcome,
            exported,
            exit_codes=exit_codes,
            approval=None if gate is None else gate.asked,
        )
        run.note(
            lambda trace: trace.tool(
                call.tool_name, call.arguments, outcome.status.value, exported.output
            )
        )
        if not result.error.strip() and call.tool_name in ("file_read", "file_write"):
            path = call.arguments.get("path")
            if isinstance(path, str) and path.strip():
                run.recent_path = path
        if captured is not None and run.inflight is not None:
            run.inflight.captured.append(captured)
        if self._notes_mode:
            run.attempt_calls.append((call, result))
            await self._rehearse_acted(call, result, run, exit_codes or ())
        return result, cut

    async def _do_memorize(
        self, task_id: str, step: _InflightStep, run: _RunState
    ) -> None:
        """Stage 5: record the step and write checked results to memory.

        *step* holds the planning reply and, for each of its calls, the
        checked result and the characters the executor-result check cut
        from it.

        - History gets one StepRecord per executed call (the assistant
          text is attached to the first), unconditionally.
        - Long-term memory gets each record's text summary in ``legacy``,
          if the checker allows the write. The explicit modes write no
          raw summary (R4 keeps only unchecked bytes out of memory).
        - In the compress modes the reply's note and draft are built from
          the calls and results, checked, stored and placed into the
          working-memory turn; a pending targeted draft is pinned once,
          after the turn's tool results.
        - Working memory always gets the real protocol turn: the assistant
          message with its tool calls, then one ``role="tool"`` message per
          call paired by ``tool_call_id``, appended as one unit. Checked
          writes may shorten it but never leave a call or result out (see
          :meth:`_checked_tool_turn`, which gets ``step.cut_before``).

        History and working memory get the same copies of the results,
        whose images are references without the base64 payload (see
        :func:`_image_references`), so no payload outlives the step,
        except in the RAM store of the pictures that planning requests
        carry when ``tool_images`` is on (see :class:`ToolImageStore`).

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
        if any(image.data for r in step.results for image in r.images) and (
            await self._sends_images()
        ):
            await self._tool_images.prepare(step.results)
        kept = [_image_references(result) for result in step.results]
        for index, (call, result) in enumerate(zip(calls, kept, strict=True)):
            captured = step.captured[index] if index < len(step.captured) else None
            record = StepRecord(
                thought=response.content if index == 0 else "",
                tool_call=call,
                tool_result=_image_references(captured[0]) if captured else result,
                capture_metadata=captured[1] if captured else None,
            )
            step.records.append(record)
            summary = await self.hippo.store_step(task_id, record)
            if captured:
                thought = redact_secrets(record.thought)[0]
                summary = f"{thought} [called {call.tool_name}] -> {redact_secrets(result.output)[0][:200]}"
            step.stored += 1

            # Checked write: step summary -> long-term memory only, in
            # legacy. The explicit modes stop writing the raw summary.
            if self.options.reasoning == "legacy":
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

        content = response.content
        thinking = response.thinking
        if self._notes_mode:
            content, thinking, note_entry, draft_entry = await self._prepare_note_turn(
                response, step, list(zip(calls, kept, strict=True)), run
            )
            assistant_msg, tool_msgs = self._checked_tool_turn(
                response,
                kept,
                step.cut_before,
                captured=step.captured,
                content=content,
                thinking=thinking,
            )
            pinned = self._pin_pending_draft(tool_msgs, draft_entry, run)
            recall_aliases = set(run.pending_shown)
            thought_aliases = set(step.thought_aliases)
            if note_entry is not None:
                # The note belongs to the turn now in the window; its full
                # checked text need not be in the content (the model's block
                # is the plan only), so the alias marks the turn, not the
                # literal text.
                recall_aliases.add(note_entry.alias)
            if pinned and draft_entry is not None:
                recall_aliases.add(draft_entry.alias)
            # A thought chunk counts as shown while its step's turn is in the
            # window: its alias leaves with the same message, and recall
            # reaches the chunk only once that turn was evicted.
            recall_aliases.update(thought_aliases)
            block, _hits = await self._step_recall(
                run,
                note_entry,
                calls,
                fallback=step.rest or EMPTY_REPLY_LABEL,
                exclude={
                    *self.hippo.working.shown_aliases(),
                    *({note_entry.alias} if note_entry is not None else set()),
                    *(
                        {draft_entry.alias}
                        if pinned and draft_entry is not None
                        else set()
                    ),
                    *thought_aliases,
                },
            )
            if block:
                if self.options.recall_placement == "tool" and tool_msgs:
                    last = tool_msgs[-1]
                    tool_msgs[-1] = dataclasses.replace(
                        last, content=f"{last.content}\n{block}"
                    )
                else:
                    tool_msgs.append(Message(role="user", content=block))
                recall_aliases.update(hit.alias for hit in _hits if hit.kind != "long")
            self._tool_images.keep(tool_msgs)
            self.hippo.working.append_turn(
                assistant_msg, tool_msgs, aliases=recall_aliases
            )
            run.pending_shown.clear()
            return
        assistant_msg, tool_msgs = self._checked_tool_turn(
            response, kept, step.cut_before, captured=step.captured
        )
        recall_aliases = set(run.pending_shown)
        block, _hits = await self._step_recall(
            run,
            None,
            calls,
            fallback=response.content or EMPTY_REPLY_LABEL,
            exclude=self.hippo.working.shown_aliases(),
        )
        if block:
            if self.options.recall_placement == "tool" and tool_msgs:
                last = tool_msgs[-1]
                tool_msgs[-1] = dataclasses.replace(
                    last, content=f"{last.content}\n{block}"
                )
            else:
                tool_msgs.append(Message(role="user", content=block))
            recall_aliases.update(hit.alias for hit in _hits if hit.kind != "long")
        run.pending_shown.clear()
        self._tool_images.keep(tool_msgs)
        self.hippo.working.append_turn(assistant_msg, tool_msgs, aliases=recall_aliases)

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
        *,
        captured: Sequence[tuple[ToolResult, dict[str, Any]]] = (),
        content: str | None = None,
        thinking: str | None = None,
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
        if content is None:
            content = response.content
        if thinking is None:
            thinking = response.thinking
        full_text = "\n".join(part for part in (content, calls_text) if part)
        text_allowed = self._memory_write_allowed(full_text, "assistant")
        # The text's share includes the line break that joins it to the calls.
        text_share = 0
        if text_allowed and len(content) <= ASSISTANT_TEXT_SHARE:
            text_share = len(full_text) - len(calls_text)

        texts: list[str] = []
        image_notes: list[str] = []
        withheld: set[int] = set()
        for index, result in enumerate(results):
            text = _tool_result_text(result)
            if self.options.captured_history:
                text, image_note = split_image_notes(text)
                image_notes.append(image_note)
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
        note_reserve = 0
        if self.options.captured_history:
            note_reserve = sum(len(note) + bool(note) for note in image_notes)
            note_reserve += sum(
                len(describe_images(r.images)) + bool(r.images) for r in results
            )
        texts = _within_turn_budget(
            texts,
            withheld,
            len(calls_text),
            limit=MAX_MEMORY_TEXT_LEN - text_share - note_reserve,
            cut_before=cut_before,
        )
        if self.options.captured_history:
            texts = [
                "\n".join(part for part in (text, note) if part)
                for text, note in zip(texts, image_notes, strict=True)
            ]

        turn_chars = len(full_text) + sum(len(text) for text in texts)
        if turn_chars > MAX_MEMORY_TEXT_LEN or not text_allowed:
            content = ""
            if self.options.reasoning == "legacy":
                thinking = ""
        assistant_msg = Message(
            role="assistant", content=content, tool_calls=calls, thinking=thinking
        )

        def message_result(result: ToolResult, text: str, index: int) -> ToolResult:
            if not self.options.captured_history:
                return result
            if index < len(captured) and not captured[index][1].get("redactions"):
                safe_text = text
            elif index < len(captured):
                safe, safe_cut = preview_result(captured[index][0])
                safe_body = _tool_result_text(safe)
                note = image_notes[index]
                if index in withheld:
                    safe_body = "\n".join(
                        (
                            "[Tool result withheld from working memory]",
                            safe_body[:WITHHELD_PREVIEW_CHARS],
                        )
                    )
                else:
                    room = max(0, len(text) - len(note) - bool(note))
                    safe_body = _cut_to(safe_body, room, cut_before=safe_cut)
                safe_text = "\n".join(part for part in (safe_body, note) if part)
            else:
                safe_text = (
                    text
                    if result.error in (CALL_INTERRUPTED, CALL_NOT_STARTED)
                    else "[Tool result unavailable]"
                )
            return dataclasses.replace(
                result,
                output="" if result.error else safe_text,
                error=safe_text.removeprefix("Error: ") if result.error else "",
                images=[] if index in withheld else result.images,
            )

        tool_msgs = [
            Message(
                role="tool",
                content=_tool_message_content(text, result.images),
                tool_call_id=call.id,
                tool_result=message_result(result, text, index),
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

    async def _fit_reflection_messages(
        self,
        messages: list[Message],
        run: _RunState | None,
        *,
        skipped: list[int] | None = None,
        sent: list[Message] | None = None,
    ) -> list[Message] | None:
        """*messages* with the oldest reflection tail units dropped to fit.

        Keeps the system message and the final user request. ``None`` when
        even the newest tail unit alone, with them, is over the prompt cap:
        the reflection is then refused before it is sent. In ``replay``
        the thinking the kept turns carry is counted in the run's
        ``replayed_chars``.
        """
        if len(messages) <= 2:
            check = await self.checker.pre_plan(
                messages, count_thinking=self._counts_thinking
            )
            if not check.allowed:
                return None
            self._note_reflection_sent(messages, run, skipped, sent)
            return messages
        units = group_turns(
            messages[1:-1], empty_prefix=self.hippo.working.empty_prefix
        )
        for first in range(len(units)):
            kept = [msg for unit in units[first:] for msg in unit]
            candidate = [messages[0], *kept, messages[-1]]
            check = await self.checker.pre_plan(
                candidate, count_thinking=self._counts_thinking
            )
            if check.allowed:
                self._note_reflection_sent(candidate, run, skipped, sent)
                return candidate
        return None

    @staticmethod
    def _note_reflection_sent(
        candidate: list[Message],
        run: _RunState | None,
        skipped: list[int] | None,
        sent: list[Message] | None,
    ) -> None:
        """Count what a reflection request that *candidate* sends carries."""
        if run is None:
            return
        run.replayed_chars += sum(
            len(msg.thinking) for msg in candidate if msg.role == "assistant"
        )
        if skipped and sent is not None:
            kept = {id(msg) for msg in candidate}
            run.replay_skipped += sum(1 for index in skipped if id(sent[index]) in kept)

    def _store_reflection_note(
        self, response: Message, task: TaskRecord, run: _RunState
    ) -> NoteStoreEntry | None:
        """Store the heuristic note of the reflection reply that was read.

        A reply that is only the JSON object, with no thinking that names
        anything, still gets a note: its ``summary`` (else ``rationale``)
        stands in for the missing text (see :func:`note_from_reply`).
        """
        note, _rest = note_from_reply(
            response.content,
            response.thinking,
            kind="reflection",
            writer="heuristic",
            fallback=self._reflection_summary(response.content),
        )
        note = dataclasses.replace(
            note,
            note_id=new_record_id(),
            task_id=task.task_id,
            step=run.steps,
            attempt=run.attempts,
            call_seq=self._last_call_seq(),
            created_at=time.time(),
            remember=True,
            remember_by="rule",
        )
        return self._store_note(note, run, source="reflection")

    @staticmethod
    def _reflection_summary(content: str) -> str:
        """The summary (else rationale) of a readable reflection reply, or ""."""
        found = _read_reflection(content)
        if found is not None:
            data: object = found[0]
        else:
            try:
                data = _decode_reflection(_coerce_str(content))
            except ValueError:
                return ""
        if not isinstance(data, dict):
            return ""
        for key in ("summary", "rationale"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    async def _do_reflect(
        self,
        task: TaskRecord,
        attempt_id: int,
        failure_type: str,
        *,
        final: bool = False,
        replies: list[Message] | None = None,
        run: _RunState | None = None,
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
        With ``"flags"``, when that reply has JSON that the decoder rejects
        and is not a tool call, one more line follows the note and quotes
        the decoder's error (see :func:`_json_error_line`).
        What counts as a reflection, and how one whose serialized form
        exceeds :data:`MAX_REFLECTION_CHARS` is kept to that budget,
        depends on ``reflection_gate``:

        - ``"flags"``: a JSON object with a valid ``recommended_action``
          (see :func:`_read_reflection`, which repairs one common slip of
          the JSON and tells the event which), so a reply whose only JSON is
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

        A reflection request that ``fit`` refuses (the explicit modes,
        when even its newest tail unit is over the prompt cap) is a reply
        that cannot be read: with ``"flags"`` this returns ``None``, and
        with ``"utility"`` it reads ``stop`` as a reply without a JSON
        object does.
        """
        flags = self.options.reflection_gate != "utility"
        short_window = self.hippo.working.get_window()
        task_query = run.task_query_msg if run is not None else None
        sent, skipped = _replay_plan(
            short_window,
            self.options,
            task_query=task_query,
            requires_replay=self._requires_replay,
        )
        fit = None
        if self.options.reasoning != "legacy":

            async def fit(messages: list[Message]) -> list[Message] | None:
                return await self._fit_reflection_messages(
                    messages, run, skipped=skipped, sent=sent
                )

        prior_reflections = self.hippo.working.get_reflections()
        note_lines: list[str] = []
        draft_titles: list[str] = []
        if self._notes_mode and run is not None:
            note_lines, draft_titles = self._reflection_note_inputs(run)
        # A brief run asks for pointers (REFLECTION_PROMPT stays frozen).
        addendum: dict[str, Any] = (
            {"addendum": REFLECTION_ADDENDUM} if run is not None and run.brief else {}
        )
        if run is not None and run.items:
            block = await self._items_block(run, "reflection")
            if block:
                addendum["addendum"] = f"{REFLECTION_ADDENDUM}\n{block}"
        response = await self.chat.reflect(
            short=sent,
            query=task.query,
            attempt_id=attempt_id,
            failure_type=failure_type,
            prior_reflections=prior_reflections or None,
            final=final,
            fit=fit,
            notes=note_lines,
            draft_titles=draft_titles,
            **addendum,
        )
        if response is None:
            # The fit refused the request before it was sent; read it as an
            # unreadable reply (see the docstring and _parse_reflection).
            return (
                None if flags else self._parse_reflection("", attempt_id, failure_type)
            )
        if self.options.reasoning != "legacy":
            before = response.thinking
            response = _split_inline_reasoning(response)
            if not before and response.thinking:
                self.chat.stats.reasoning_chars += len(response.thinking)
        if replies is not None:
            replies.append(response)
        if run is not None:
            run.events.reflection_call(getattr(self.chat, "last_call", None))
        if flags:
            readable = _read_reflection(response.content) is not None
        else:
            readable = _holds_reflection_object(response.content)
        if not readable:
            reason = _unreadable_reason(response, flags=flags)
            if run is not None:
                run.events.reflection_unreadable(reason)
            # JSON the decoder rejected: say where, so the retry can mend it.
            hint = (
                _json_error_line(response.content)
                if flags and reason != "tool_call"
                else ""
            )
            # Often a tool call: the window ends in tool turns, and some
            # models continue them although the request offers no tools.
            # Ask once more, with those turns written out as text.
            response = await self.chat.reflect(
                short=sent,
                query=task.query,
                attempt_id=attempt_id,
                failure_type=failure_type,
                prior_reflections=prior_reflections or None,
                final=final,
                retry=True,
                retry_hint=hint,
                fit=fit,
                notes=note_lines,
                draft_titles=draft_titles,
                **addendum,
            )
            if response is None:
                return (
                    None
                    if flags
                    else self._parse_reflection("", attempt_id, failure_type)
                )
            if self.options.reasoning != "legacy":
                before = response.thinking
                response = _split_inline_reasoning(response)
                if not before and response.thinking:
                    self.chat.stats.reasoning_chars += len(response.thinking)
            if replies is not None:
                replies.append(response)
            if run is not None:
                run.events.reflection_call(getattr(self.chat, "last_call", None))
        reflection_note_id = ""
        if run is not None:
            run.reflection_thinking = response.thinking
            readable = (
                _read_reflection(response.content) is not None
                if flags
                else _holds_reflection_object(response.content)
            )
            if self._notes_mode and readable:
                await self._cite_notes(self._used_notes(response.content, flags), run)
                reflection_entry = self._store_reflection_note(response, task, run)
                if reflection_entry is not None and reflection_entry.note is not None:
                    reflection_note_id = reflection_entry.note.note_id
        if flags:
            found = _read_reflection(response.content)
            if found is None:
                return None
            data, repair = found
            if run is not None:
                run.events.reflection_repaired(repair)
            entry, decision = self._reflection_from_object(
                data, attempt_id, failure_type
            )
            if self.options.reflection_gate == "patient" or self.options.answer_on_stop:
                entry = _cap_reflection_lists(entry)
            entry = _fit_reflection(entry, MAX_REFLECTION_CHARS)
            decision.tried_memory = entry
            if run is not None and self._notes_mode:
                self._propose_reflection(entry, run, note_id=reflection_note_id)
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
        if run is not None and self._notes_mode:
            self._propose_reflection(entry, run, note_id=reflection_note_id)
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

        A reply that is only unfinished thinking (``Chat.unfinished``, #278)
        is stored and shown with all its text as thinking and no content, and
        goes the same way with an ``unfinished_answer`` notice: the model is
        asked once more, and a row that held such a reply ends the run with
        :data:`~localmelo.melo.schema.UNFINISHED_ANSWER_ERROR` instead.
        """
        fingerprints: list[tuple[str, str, str]] = []
        reflections = (
            None
            if self._reflection_in_window
            else self.hippo.working.get_reflections() or None
        )
        run.attempt_calls.clear()
        run.read_streak = 0

        for step in range(budget):
            self.chat.check_call_budget()  # before the step starts, not inside it
            if budget - step == 2:
                await self._deadline_check(run)
            reflection = run.reflection_msg
            if reflection is not None and not self.hippo.working.holds(reflection):
                # An eviction took the latest reflection: it goes back once, at
                # the end, where the eviction has already changed the prefix.
                self.hippo.working.append(reflection)
            pending = run.inbox.poll() if run.inbox is not None else []
            inserted = None
            if pending:
                inserted = Message(
                    role="user", content="\n\n".join(text for _, text in pending)
                )
                self.hippo.working.append(inserted)
                run.pinned.append(inserted)
            # Retrieval + tool resolution + boundary checks
            long_context, short_window, tools, fail = await self._do_retrieval(
                query, task_query=run.task_query_msg, run=run
            )
            if fail is not None:
                run.end("failed", error=fail)
                return step + 1, ""
            on_ready = None
            if inserted is not None:
                if not any(message is inserted for message in short_window):
                    run.end("failed", error="inbox instructions do not fit the prompt")
                    return step + 1, ""

                def on_ready(
                    messages: list[Message],
                    *,
                    inserted: Message = inserted,
                    pending: list[tuple[int, str]] = pending,
                ) -> None:
                    assert run.inbox is not None
                    if (
                        not messages
                        or messages[-1].role != "user"
                        or not messages[-1].content.endswith(inserted.content)
                    ):
                        raise ContextFitError(
                            "Inbox instructions do not fit the request"
                        )
                    for offset, text in pending:
                        run.inbox.accept(offset)
                        run.events.inbox_accepted(offset, text, run.steps, run.attempts)

            # LLM planning step (with reflection context). The step counts
            # once the planning call is made, even if it then raises.
            run.steps += 1
            run.events.step_started(run.steps, run.attempts)
            if (stream_display := STREAM_DISPLAY.get()) is not None:
                stream_display.step(run.steps)
            run.note(lambda trace: trace.step(run.steps, run.attempts))
            response, fail = await self._do_plan(
                long_context,
                short_window,
                tools,
                query,
                reflections=reflections,
                task_query=run.task_query_msg,
                run=run,
                **({"on_ready": on_ready} if on_ready is not None else {}),
            )
            # A reply that is only unfinished thinking (#278) is thinking with
            # no answer: it is stored and shown so, and goes on as empty.
            unfinished = fail is None and bool(
                getattr(self.chat, "last_unfinished", False)
            )
            if unfinished:
                before = len(response.thinking)
                response = dataclasses.replace(
                    response,
                    content="",
                    thinking=unfinished_as_thinking(
                        response.content, response.thinking
                    ),
                )
                self.chat.stats.reasoning_chars += len(response.thinking) - before
            # Also a reply the post-plan check refuses.
            run.events.assistant_message(response, run.plan_call)
            if fail is not None:
                run.end("failed", error=fail)
                return step + 1, ""
            rest = response.content
            has_model_note = False
            if self._notes_mode:
                block, rest = split_note(response.content)
                # With --note-writer heuristic the model's block is ignored,
                # so it is no model note for placement either.
                has_model_note = (
                    block is not None and self.options.note_writer == "auto"
                )
            thought_aliases: list[str] = []
            if self._notes_mode and "thought" in self._effective_kinds:
                thought_aliases = await self._store_thought_chunks(response, run)
            if rest.strip():
                run.last_text = rest
                run.last_text_thinking = response.thinking

            if not response.tool_calls:
                if rest.strip():
                    # No tool call → direct answer. In the compress modes
                    # the note is stored first and the answer is the text
                    # without its <note> block.
                    if self._notes_mode:
                        note, _ = self._reply_note(
                            response,
                            (),
                            kind="answer",
                            task_id=task.task_id,
                            run=run,
                        )
                        entry = self._store_note(note, run, source="reasoning_note")
                        if entry is not None:
                            await self._restate(note, run)
                    run.answer_thinking = response.thinking
                    retry = await self._brief_gate(run, can_retry=step + 1 < budget)
                    if retry is None:
                        run.end("completed", result=rest)
                        return step + 1, ""
                    # The answer was no end: the model reads what its check
                    # showed and plans again.
                    answer = Message(role="assistant", content=rest)
                    if self.options.reasoning not in ("legacy", "compress"):
                        answer = dataclasses.replace(answer, thinking=response.thinking)
                    self.hippo.working.append_turn(
                        answer, [Message(role="user", content=retry)]
                    )
                    continue
                # An empty reply is no answer: ask for one and plan again.
                label_content = EMPTY_REPLY_LABEL
                entry = None
                if self._notes_mode:
                    note, _ = self._reply_note(
                        response,
                        (),
                        kind="empty",
                        task_id=task.task_id,
                        run=run,
                    )
                    entry = self._store_note(note, run, source="reasoning_note")
                    if entry is not None:
                        label_content = f"{EMPTY_REPLY_LABEL}\n{entry.text}"
                run.empty_replies += 1
                run.unfinished_row = run.unfinished_row or unfinished
                if run.empty_replies >= MAX_EMPTY_REPLIES:
                    run.end(
                        "failed",
                        error=(
                            UNFINISHED_ANSWER_ERROR
                            if run.unfinished_row
                            else "empty replies"
                        ),
                    )
                    return step + 1, ""
                # The label stands in for the reply, so the nudge does not
                # follow another user message; as one unit, the two are
                # only ever dropped together.
                label = Message(role="assistant", content=label_content)
                if self.options.reasoning not in ("legacy", "compress"):
                    label = dataclasses.replace(label, thinking=response.thinking)
                nudge = EMPTY_REPLY_NUDGE
                aliases: set[str] = set(thought_aliases)
                block, _hits = await self._step_recall(
                    run,
                    entry,
                    (),
                    fallback=EMPTY_REPLY_LABEL,
                    exclude={
                        *self.hippo.working.shown_aliases(),
                        *({entry.alias} if entry is not None else set()),
                        *thought_aliases,
                    },
                )
                if block:
                    nudge = f"{nudge}\n\n{block}"
                    aliases.update(hit.alias for hit in _hits if hit.kind != "long")
                aliases.update(run.pending_shown)
                run.pending_shown.clear()
                self.hippo.working.append_turn(
                    label,
                    [Message(role="user", content=nudge)],
                    aliases=aliases,
                )
                if unfinished:
                    run.events.unfinished_answer()
                else:
                    run.events.empty_reply(run.empty_replies)
                continue
            run.empty_replies = 0
            run.unfinished_row = False

            # Execute every call in order, each through the executor boundary,
            # which runs only the tools this planning call offered. Until the
            # step is in working memory, run.inflight holds it, so that a
            # cancellation can close it (see _on_cancelled).
            allowed = frozenset(t.name for t in tools)
            calls = list(response.tool_calls)
            inflight = run.inflight = _InflightStep(
                response=response,
                rest=rest,
                has_model_note=has_model_note,
                recent_path=run.recent_path,
                thought_aliases=thought_aliases,
            )
            for call in calls:
                # Counted when handed over, so a call that timeout_s cuts
                # short counts too.
                run.n_tool += 1
                result, cut = await self._do_execute(call, allowed=allowed, run=run)
                inflight.results.append(result)
                inflight.cut_before.append(cut)
            await self._do_memorize(task.task_id, inflight, run)
            run.inflight = None

            # Stuck detection (one fingerprint per step)
            fingerprints.append(self._make_fingerprint(calls, inflight.results))
            stuck = self._detect_stuck(fingerprints)
            if stuck:
                run.events.stuck()
                run.note(Trace.stuck)
                return step + 1, stuck
            if run.brief and step + 1 < budget:
                await self._push_after_reads(run)

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
        inbox: InboxSource | None = None,
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
          :meth:`_run_reflective_attempts`). Also the end of a run whose next
          request would pass ``max_calls``: the error says ``Call budget
          reached``, and the result is the latest text the model wrote in the
          run, if any. With ``answer_on_stop``, a
          ``gate_stopped`` or ``budget_exhausted`` run with no result asks
          once more for one (see :meth:`_answer_on_stop`);
        - ``context_budget_exceeded``: the pinned request cannot fit the
          configured input budget, or the smaller overflow retry was refused;
        - ``stuck``: stuck detection fired with ``attempt_policy="none"``;
        - ``provider_error``: a model server request failed with a
          :class:`~localmelo.melo.contracts.providers.ProviderError`: a
          chat request after its retries, or an embedding request. The
          error is the provider's message; no traceback is logged. With
          ``answer_on_stop``, a request refused for a budget or rate limit
          (see :func:`~localmelo.melo.agent._call_budget.is_budget_refusal`)
          still gets the extra request for an answer, and the latest text the
          model wrote in the run is the result when that fails too;
        - ``setup_error``: the tool providers could not be set up at the
          start of the first run (see :meth:`_ensure_tools`): one failed
          to start, or a provided tool has the name of a registered tool.
          The error says which, and no provider is left running. Setup
          is not retried: every later run on this agent ends as
          ``setup_error`` too, before its first step;
        - ``timeout``: the run was still going after ``timeout_s``
          seconds. It is cancelled wherever it was: a request in flight
          is abandoned (and counts in ``llm_errors``), a shell command or
          a ``python_exec`` child is killed with every process it
          started, and tool providers still being set up are closed (so
          later runs end as ``setup_error``).
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
            task_text=query,
            events=_Emitter(
                on_event if observe is None else observe,
                task.task_id,
                interactive=observe is None,
                progress=bool(progress),
            ),
            approval=approval,
            routine=routine,
            inbox=inbox,
            brief=self.options.brief_probe is not None,
            place=place,
            limits=self._limits() if progress else None,
            refused_writes_base=(
                0 if self._confinement is None else self._confinement.refused
            ),
        )
        prefix_run = self.chat.begin_prefix()
        stats = self.chat.reset_stats()
        budget_accounting = BUDGET_ACCOUNTING.set((self.chat, self.chat._accounting))
        budget_hook = BUDGET_HOOK.set(
            (self.chat, run.events.context_budget)
            if self.options.context_window is not None
            else None
        )
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
        recall_run = RECALL_RUN.set(run)
        self._active_runs += 1
        try:
            run.events.run_started(
                query,
                max_steps=self.options.max_steps,
                workspace_root=self.executor.workspace_root,
                approval=approval is not None,
            )
            try:
                if run.inbox is not None:
                    run.inbox.validate(self.executor.workspace_root or os.getcwd())
                if run.brief:
                    assert self.options.brief_probe is not None
                    run.workspace_digest = await asyncio.to_thread(
                        self.options.brief_probe, self._workspace_for_brief()
                    )
                    run.step_digest = run.workspace_digest
                    run.check_digest = run.workspace_digest
                    self.hippo.working.on_drop = lambda gone: self._note_dropped(
                        run, gone
                    )
                    if self.options.brief_items:
                        run.items = parse_items(query)
                        if run.items:
                            run.items_baseline = await asyncio.to_thread(
                                workspace_stamps, self._workspace_for_brief()
                            )
                if self.options.brief_check is not None:
                    run.check_commands = check_commands(query)
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
                    query_msg = Message(role="user", content=query)
                    self.hippo.working.append(query_msg)
                    run.task_query_msg = query_msg
                    run.query_in_memory = True
                    if self.options.reasoning != "legacy":
                        # Only the user's own text gives a user signal: a
                        # direct statement is admitted at once, while the
                        # fact a remember request carries is staged and
                        # admitted with the digest at flush.
                        signals = classify_user_text(query)
                        for candidate in user_candidates(
                            query,
                            task_id=run.task_id,
                            typed_assertions=self.options.typed_assertions,
                            provenance={
                                "task_text": run.task_text[:300],
                                "source_ids": [],
                            },
                        ):
                            checked = self._checked_candidate_text(
                                candidate.text, "user", run
                            )
                            if checked is None:
                                continue
                            candidate.text = checked
                            self.hippo.propose(candidate)
                            decisions = await self.hippo.admit(
                                candidate.candidate_id, "user_statement"
                            )
                            self._drain_decisions(decisions, run)
                        for candidate in request_candidates(
                            query,
                            task_id=run.task_id,
                            provenance={
                                "task_text": run.task_text[:300],
                                "source_ids": [],
                            },
                        ):
                            checked = self._checked_candidate_text(
                                candidate.text, "user", run
                            )
                            if checked is None:
                                continue
                            candidate.text = checked
                            self.hippo.propose(candidate)
                        run.remember_request = signals.remember
                    await self._run_attempts(task, query, run)
                    if self.options.answer_on_stop:
                        await self._answer_on_stop(task, run)
                    self._keep_best_effort(run)
            except _ToolSetupError as exc:
                # Reported in the result, not a crash: no traceback.
                logger.info("Task %s ended with setup_error: %s", task.task_id, exc)
                run.end("setup_error", error=str(exc))
            except InboxError as exc:
                # An inbox that became unreadable mid-run is a setup failure,
                # not a crash: report its message without a class prefix.
                logger.info("Task %s ended with setup_error: %s", task.task_id, exc)
                run.end("setup_error", error=str(exc))
            except ContextFitError as exc:
                run.end("context_budget_exceeded", error=str(exc))
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
                    await self._on_cancelled(query, run, exc)
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
            if run.brief:
                self.hippo.working.on_drop = None
            PREFIX_RUN.reset(prefix_run)
            BUDGET_HOOK.reset(budget_hook)
            BUDGET_ACCOUNTING.reset(budget_accounting)
            if run.finalizer is None:
                # Else the finalizer counts the run out when it ends.
                self._active_runs -= 1
            if retries is not None:
                RETRY_HOOK.reset(retries)
            CALL_RECORDER.reset(recorder)
            RECALL_RUN.reset(recall_run)

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
        check's reason. The result itself is not changed. On a provider that
        requires reasoning replay, ``replay`` also keeps the thinking of the
        reply that gave the answer, so the next request can send it back. A
        failure is logged, not raised, so it cannot replace the run's
        outcome or skip the save.
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
            thinking = (
                run.answer_thinking
                if (
                    self.options.reasoning in ("replay", "compress+replay")
                    and self._requires_replay
                )
                else ""
            )
            self.hippo.working.append(
                Message(role="assistant", content=text, thinking=thinking)
            )
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
            if self.options.reasoning != "legacy":
                # R4 gates the promotion too: the explicit modes write
                # nothing unchecked to long-term memory. R6-lite's flush
                # admits the run's candidates through the boundary and
                # policy; its failure is logged, never raised.
                if (
                    run.final_digest is not None
                    and self.options.admit_on_outcome
                    and not run.remember_request
                ):
                    run.final_digest.provenance["final_status"] = run.status
                decisions = await self.hippo.flush(
                    run_request=run.remember_request,
                    digest=run.final_digest,
                    final_reflection=run.final_reflection,
                    task_id=run.task_id,
                )
                self._drain_decisions(decisions, run)
                self.hippo.working.clear_reflections()
            else:
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

    async def _on_cancelled(
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
            query_msg = Message(role="user", content=query)
            self.hippo.working.append(query_msg)
            run.task_query_msg = query_msg
            run.query_in_memory = True
        if run.inflight is None:
            return
        try:
            await self._close_interrupted_step(run.inflight, run.waiting_call_id, run)
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

    async def _close_interrupted_step(
        self, step: _InflightStep, waiting_call_id: str, run: _RunState
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
        if self._notes_mode:
            content, thinking, _note_entry, draft_entry = await self._prepare_note_turn(
                step.response,
                step,
                list(zip(calls, closed, strict=True)),
                run,
            )
            assistant_msg, tool_msgs = self._checked_tool_turn(
                step.response,
                closed,
                cut_before,
                captured=step.captured[: len(step.results)],
                content=content,
                thinking=thinking,
            )
            self._pin_pending_draft(tool_msgs, draft_entry, run)
            # The step's thought chunks still count as shown with this turn,
            # as in _do_memorize.
            self.hippo.working.append_turn(
                assistant_msg, tool_msgs, aliases=step.thought_aliases
            )
            return
        assistant_msg, tool_msgs = self._checked_tool_turn(
            step.response,
            closed,
            cut_before,
            captured=step.captured[: len(step.results)],
        )
        self.hippo.working.append_turn(
            assistant_msg, tool_msgs, aliases=step.thought_aliases
        )

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
                captured = step.captured[index] if index < len(step.captured) else None
                record = StepRecord(
                    thought=step.response.content if index == 0 else "",
                    tool_call=calls[index],
                    tool_result=(
                        _image_references(captured[0])
                        if captured
                        else step.closed[index]
                    ),
                    capture_metadata=captured[1] if captured else None,
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
        opts = self.options
        usage = dict(stats.usage)
        memory_stats: dict[str, Any] | None = None
        if opts.reasoning != "legacy":
            usage["reasoning_chars"] = stats.reasoning_chars
            usage.update(stats.details)
            memory_stats = {
                "replayed_chars": run.replayed_chars,
                "replay_skipped": run.replay_skipped,
                "replay_stripped": run.replay_stripped,
                "notes": run.notes,
                "notes_model": run.notes_model,
                "notes_heuristic": run.notes_heuristic,
                "note_chars": run.note_chars,
                "drafts": run.drafts,
                "drafts_written": run.drafts_written,
                "candidates": run.candidates,
                "remember_model": run.remember_model,
                "remember_heuristic": run.remember_heuristic,
                "candidates_refused": run.candidates_refused,
                "redactions": run.redactions,
                "rehearsed": dict(run.rehearsed),
                "digests": run.digests,
                "recalls": run.recalls,
                "recalls_by_trigger": dict(run.recalls_by_trigger),
                "recall_hits": run.recall_hits,
                "recall_same_task": run.recall_same_task,
                "recall_cross_task": run.recall_cross_task,
                "recall_task_notes": run.recall_task_notes,
                "recall_chars": run.recall_chars,
                "recall_ms": run.recall_ms,
                "recall_fallbacks": run.recall_fallbacks,
                "recall_skipped": run.recall_skipped,
                "recall_refused": run.recall_refused,
                "recall_empty": run.recall_empty,
                "aux_retrieve_calls": run.aux_retrieve_calls,
                "retrieve_errors": run.retrieve_errors,
                "retrieve_prompt_tokens": run.retrieve_prompt_tokens,
                "retrieve_completion_tokens": run.retrieve_completion_tokens,
                "thought_chunks": run.thought_chunks,
                "recall_log": list(run.recall_log),
                "admitted": dict(run.admitted),
                "written": dict(run.written),
                "ignored": run.ignored,
            }
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
            usage=usage,
            duration_s=duration_s,
            config=self._run_config(),
            memory_stats=memory_stats,
            context_budget=stats.context_budget,
            prefix_cache=(
                {
                    "version": PREFIX_VERSION,
                    "schema_changes": stats.schema_changes,
                    **(
                        {"reported_cached_tokens": stats.prefix_cached_tokens}
                        if stats.prefix_cached_tokens is not None
                        else {}
                    ),
                }
                if self.options.fixed_prefix
                else None
            ),
            confinement=self._confinement_record(run),
            answered_on_stop=run.answered_on_stop,
        )

    def _confinement_record(self, run: _RunState) -> dict[str, Any] | None:
        """``RunResult.confinement`` for *run* (see AgentOptions.confine_writes)."""
        if not self.options.confine_writes:
            return None
        if self._confinement is None:
            return {"writes": "not_confined", "reason": platform_note()}
        refused = self._confinement.refused - run.refused_writes_base
        return {"writes": "confined", "refused_writes": refused}

    def _run_config(self) -> dict[str, Any]:
        """The effective configuration that ``RunResult.config`` reports."""
        opts = self.options
        builtins = builtin_set_for(opts.mode, opts.builtins)
        config: dict[str, Any] = {
            "max_steps": opts.max_steps,
            "attempt_policy": opts.attempt_policy,
            "steps_per_attempt": opts.steps_per_attempt,
            "max_attempts": opts.max_attempts,
            "reflection_gate": opts.reflection_gate,
            "tool_exposure": opts.tool_exposure,
            "mode": opts.mode,
            "builtins": sorted(builtins),
            "memory": opts.memory,
            "memory_dir": self._memory_dir,
            "timeout_s": opts.timeout_s,
            "tools": sorted(tool.name for tool in self.hippo.list_tools()),
        }
        if opts.answer_on_stop:
            config["answer_on_stop"] = True
        if opts.max_calls is not None:
            config["max_calls"] = opts.max_calls
        if opts.window_eviction != "slide":
            config["window_eviction"] = opts.window_eviction
        if opts.reflect_tool_guard:
            config["reflect_tool_guard"] = True
        if opts.cache_layout != "v2":
            config["cache_layout"] = opts.cache_layout
        if opts.window_messages != SHORT_TERM_MAX:
            config["window_messages"] = opts.window_messages
        if opts.reasoning != "legacy":
            config["reasoning"] = opts.reasoning
            config["replay_turns"] = opts.replay_turns
            config["note_writer"] = opts.note_writer
            config["retriever"] = opts.retriever
            config["retriever_effective"] = dict(sorted(self._plan.as_map().items()))
            config["recall_at"] = sorted(self._plan.enabled())
            config["recall_k"] = opts.recall_k
            config["recall_kinds"] = sorted(self._effective_kinds)
            config["recall_placement"] = opts.recall_placement
            config["recall_min_score"] = opts.recall_min_score
            config["cross_task_ratio"] = opts.cross_task_ratio
            config["cross_task_cap"] = opts.cross_task_cap
            config["cross_task_cold"] = opts.cross_task_cold
            config["admit_on_outcome"] = opts.admit_on_outcome
            config["keep_decision_text"] = opts.keep_decision_text
        return config

    async def _run_attempts(self, task: TaskRecord, query: str, run: _RunState) -> None:
        """Run the attempts of *run*, as ``attempt_policy`` says.

        Two ends are handled here instead of leaving the run to its handlers
        in :meth:`run_task`, so :meth:`_answer_on_stop` can still ask: the
        request that ``max_calls`` stops (:class:`CallBudgetError`) ends the
        run ``budget_exhausted``, and, with ``answer_on_stop``, a request the
        server refuses for a budget or rate limit ends it ``provider_error``
        as before, with the run marked as refused. Any other error goes on.
        """
        try:
            if self.options.attempt_policy == "none":
                await self._run_single_attempt(task, query, run)
            else:
                await self._run_reflective_attempts(task, query, run)
        except CallBudgetError as exc:
            logger.info("Task %s stopped at its call budget: %s", task.task_id, exc)
            run.early_stop = "calls"
            run.end("budget_exhausted", error=str(exc))
        except ProviderError as exc:
            if not (self.options.answer_on_stop and is_budget_refusal(exc)):
                raise
            logger.info("Task %s was refused by the server: %s", task.task_id, exc)
            run.early_stop = "refused"
            run.end("provider_error", error=str(exc) or _exception_text(exc))

    def _keep_best_effort(self, run: _RunState) -> None:
        """A run that stopped on ``max_calls`` or a refusal, with no answer: its latest text."""
        if not run.early_stop or run.result.strip():
            return
        run.result = run.last_text
        run.answer_thinking = run.last_text_thinking

    async def _answer_on_stop(self, task: TaskRecord, run: _RunState) -> None:
        """``answer_on_stop``: one more request for the answer a stopped run lacks.

        Only for a run that ended ``gate_stopped`` or ``budget_exhausted``,
        or ``provider_error`` after the server refused a request for a budget
        or rate limit (see :meth:`_run_attempts`), with a blank result. The
        request (:meth:`Chat.final_answer`) carries the working window and the
        latest reflection. A non-blank reply becomes the run's result (its
        ``<think>`` block removed), the status stays, and
        ``RunResult.answered_on_stop`` is true. A request that fails or that
        ``max_calls`` leaves no call for, or a blank reply, leaves the run as
        it was.
        """
        stopped = run.status in ("gate_stopped", "budget_exhausted")
        if run.result.strip() or not (stopped or run.early_stop == "refused"):
            return
        # The reflection comes as its own block: leave out the request and reply
        # of its exchange and the message that carries it in the window.
        shown = {id(run.reflection_msg), *map(id, self.chat.reflect_exchange)}
        window = [
            msg for msg in self.hippo.working.get_window() if id(msg) not in shown
        ]
        try:
            reply = await self.chat.final_answer(
                window, task.query, self.hippo.working.get_reflections()
            )
        except CallBudgetError as exc:
            logger.info("No call left to answer task %s on stop: %s", task.task_id, exc)
            return
        except Exception as exc:
            logger.warning(
                "Could not ask for an answer on stop for task %s: %s",
                task.task_id,
                _exception_text(exc),
            )
            return
        reply = _split_inline_reasoning(reply)
        answer = reply.content.strip()
        if not answer:
            return
        run.result = answer
        run.answered_on_stop = True
        run.answer_thinking = reply.thinking

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
          ``Stopped after attempt <n> (stop reason: <code>)`` (``n``
          counting from 1; the code is the check that stopped the run, see
          :func:`_gate_stop_reason`) when the rationale is blank. The
          ``reflection`` event carries the code in every case.

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

        def record_digest() -> None:
            """Build the attempt's digest, then supersede its notes."""
            if not self._notes_mode:
                return
            self._build_digest(query, failure_type, run, task.task_id)
            run.attempt_calls.clear()
            self.hippo.working.notes.supersede_attempt(run.attempts)

        for attempt_id in range(opts.max_attempts):
            # Never below 1: the previous attempt ended the run otherwise.
            budget = min(opts.steps_per_attempt, opts.max_steps - total_steps)
            run.attempts += 1
            run.last_attempt = (
                run.attempts >= opts.max_attempts
                or total_steps + budget >= opts.max_steps
            )
            steps_used, failure_type = await self._run_attempt(task, query, budget, run)
            total_steps += steps_used

            if not run.running:
                record_digest()
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
                run=run,
            )
            entry = None
            decision = None
            if reflection is not None:
                entry, decision = reflection
            # None for a reply with no reflection, only with "flags".
            patient = opts.reflection_gate == "patient" and not run.patience_spent
            stop_reason = _stop_reason(
                decision, opts.reflection_gate, out_of, brief=run.brief, patient=patient
            )
            if (
                patient
                and stop_reason is None
                and decision is not None
                and not run.brief
                and not decision.next_step_is_novel
            ):
                run.patience_spent = True  # this attempt rests on the allowance
            continue_run = stop_reason is None
            verdict = GateVerdict(
                opts.reflection_gate,
                stop_reason,
                (
                    _utility(decision)
                    if decision is not None and opts.reflection_gate == "utility"
                    else None
                ),
            )
            if self._plan.attempt is not None:
                if continue_run:
                    # The new attempt's block replaces the old one; do not
                    # dedupe its long hits against the block being replaced.
                    run.boundary_block = ""
                    _record, block, _hits = await self._run_recall(
                        "attempt",
                        build_attempt_query(
                            entry,
                            failure_type=failure_type,
                            fallback=run.last_text,
                        ),
                        run,
                        boundary=True,
                        placement="query",
                        exclude=self.hippo.working.shown_aliases(),
                    )
                    run.boundary_block = block
                else:
                    run.boundary_block = ""
            record_digest()
            if reflection is None:  # only with reflection_gate="flags"
                run.events.reflection(None, continue_run=continue_run, verdict=verdict)
                if out_of:
                    logger.info(
                        "Reflection after attempt %d unreadable after a retry, "
                        "ending the run: %s",
                        run.attempts,
                        out_of,
                    )
                    run.answer_thinking = run.last_text_thinking
                    run.end("budget_exhausted", result=run.last_text, error=out_of)
                    return
                logger.info(
                    "Reflection after attempt %d unreadable after a retry, "
                    "starting attempt %d",
                    run.attempts,
                    run.attempts + 1,
                )
                await self._push_if_unchanged(run, None)
                continue
            assert entry is not None and decision is not None
            self.hippo.working.add_reflection(entry)
            exchange = self.chat.reflect_exchange
            if opts.window_eviction == "boundary" and continue_run:
                # The reflection carries the finished attempt; the next one
                # starts from the query (and what the run pinned) alone.
                self.hippo.working.drop_since(run.task_query_msg, run.pinned)
                exchange = []
            if self._reflection_in_window:
                # Once, here: the exchange (when the request shared the plan's
                # prefix) and the block that carries the reflection.
                run.reflection_msg = self.chat.reflection_message(entry)
                for message in (*exchange, run.reflection_msg):
                    self.hippo.working.append(message)

            best_effort = decision.best_effort_result
            if decision.recommended_action == "done":
                retry = await self._brief_gate(run, can_retry=not out_of)
                if retry is not None:
                    # A failing brief check overrules the done: the next
                    # attempt starts with what the check showed.
                    run.events.reflection(
                        reflection,
                        continue_run=True,
                        verdict=verdict._replace(stop_reason=None),
                    )
                    self.hippo.working.append(Message(role="user", content=retry))
                    continue
                run.events.reflection(reflection, continue_run=False, verdict=verdict)
                if best_effort.strip():
                    answer = best_effort
                    run.answer_thinking = run.reflection_thinking
                else:
                    answer = run.last_text
                    run.answer_thinking = run.last_text_thinking
                run.end("completed", result=answer)
                return
            if out_of:
                run.events.reflection(reflection, continue_run=False, verdict=verdict)
                run.answer_thinking = run.reflection_thinking
                run.end("budget_exhausted", result=best_effort, error=out_of)
                return
            if not continue_run:
                run.events.reflection(reflection, continue_run=False, verdict=verdict)
                # The rationale is the model's own words (frozen by the
                # legacy goldens); only the fallback names the check.
                error = decision.rationale.strip() or (
                    f"Stopped after attempt {run.attempts} "
                    f"(stop reason: {stop_reason})"
                )
                logger.info(
                    "Reflection gate stopped the run after attempt %d: %s",
                    run.attempts,
                    stop_reason,
                )
                run.answer_thinking = run.reflection_thinking
                run.end("gate_stopped", result=best_effort, error=error)
                return
            run.events.reflection(reflection, continue_run=True, verdict=verdict)
            await self._push_if_unchanged(run, entry)

    async def _push_if_unchanged(
        self, run: _RunState, entry: ReflectionEntry | None
    ) -> None:
        """At a brief run's boundary: ask the next attempt to start writing.

        When the workspace digest is the one from the run's start, a
        ``no_change`` notice is emitted and a user line is added to the
        window, so the next attempt's first planning request carries it.
        The line ends with the first direction of the boundary's reflection
        *entry*, when it has one. Nothing happens outside a brief run or a
        git tree.
        """
        probe = self.options.brief_probe
        if probe is None or run.workspace_digest is None:
            return
        digest = await asyncio.to_thread(probe, self._workspace_for_brief())
        if digest != run.workspace_digest:
            return
        run.events.no_change()
        line = " ".join(part for part in (NO_CHANGE_PUSH, start_with(entry)) if part)
        self.hippo.working.append(Message(role="user", content=line))

    async def _push_after_reads(self, run: _RunState) -> None:
        """In a brief run, after ``STEP_PUSH_STEPS`` steps in a row, ask to write.

        Every ``STEP_PUSH_STEPS``-th step of an attempt probes the workspace.
        A digest that is the one at the last change seen (or at the start)
        adds a ``no_change`` notice (detail ``scope: "step"``) and one user
        line to the window; a changed digest becomes the new baseline.
        """
        probe = self.options.brief_probe
        if probe is None or run.step_digest is None:
            return
        run.read_streak += 1
        if run.read_streak < STEP_PUSH_STEPS:
            return
        run.read_streak = 0
        digest = await asyncio.to_thread(probe, self._workspace_for_brief())
        if digest != run.step_digest:
            run.step_digest = digest
            return
        run.events.no_change(scope="step")
        self.hippo.working.append(Message(role="user", content=STEP_PUSH))

    @staticmethod
    def _note_dropped(run: _RunState, gone: list[Message]) -> None:
        """Keep a line for each tool result that left the window of a brief run."""
        run.dropped = [*run.dropped, *dropped_notes(gone)][-DROPPED_KEPT:]

    async def _items_block(self, run: _RunState, scope: str) -> str:
        """The block that names the brief's items no edit has touched, or "".

        Marks each item from the files git lists as changed now against those
        at the run's start, and records the marks in a ``brief_items`` notice
        when the block is shown (*scope*: where it goes).
        """
        if run.items_baseline is None:
            return ""
        now = await asyncio.to_thread(workspace_stamps, self._workspace_for_brief())
        if now is None:
            return ""
        marks = mark_items(run.items, edited_files(run.items_baseline, now))
        block = render_items(run.items, marks)
        if block:
            run.events.brief_items(
                scope,
                {
                    item.split(".", 1)[0]: mark
                    for item, mark in zip(run.items, marks, strict=True)
                },
            )
        return block

    async def _deadline_check(self, run: _RunState) -> None:
        """With two steps left in a brief run's last attempt, run the checks once.

        With ``brief_items``, the items no edit has touched yet are named in a
        user line first (joined to the gate's failure message when both come).
        The checks run only when the workspace changed since the checks last ran and the
        gate has not run in this attempt. A pass changes nothing; a failure
        adds the gate's failure message to the window, so the next planning
        request carries it, and uses one of the gate's retries.
        """
        if not run.brief or not run.last_attempt:
            return
        lines = []
        if run.items:
            block = await self._items_block(run, "deadline")
            if block:
                lines.append(block)
        if (
            run.check_commands
            and run.check_attempt != run.attempts
            and run.check_digest is not None
            and self.options.brief_probe is not None
        ):
            digest = await asyncio.to_thread(
                self.options.brief_probe, self._workspace_for_brief()
            )
            if digest != run.check_digest:
                retry = await self._brief_gate(run, can_retry=True, quiet=True)
                if retry is not None:
                    lines.append(retry)
        if lines:
            self.hippo.working.append(Message(role="user", content="\n\n".join(lines)))

    async def _brief_gate(
        self, run: _RunState, *, can_retry: bool, quiet: bool = False
    ) -> str | None:
        """Run a brief's checks before the run ends ``completed``.

        Returns ``None`` when the run may end: the brief names no check,
        every check passed, or one failed for good (a refused check, no
        retry left or *can_retry* false), which adds a ``brief_check_failed``
        notice. Otherwise it returns the user message for the model's next
        planning request, and the caller keeps the run going. The checks
        run in order and the first failure stops the gate. With *quiet*
        (the deadline check) a failure that cannot send the run back adds
        no ``brief_check_failed`` notice: the run goes on.
        """
        check = self.options.brief_check
        if check is None or not run.check_commands:
            return None
        run.check_rounds += 1
        run.check_attempt = run.attempts
        probe = self.options.brief_probe
        if probe is not None and run.brief:
            run.check_digest = await asyncio.to_thread(
                probe, self._workspace_for_brief()
            )
        for command in run.check_commands:
            result = await check(self.executor, command)
            run.events.brief_check(result, run.check_rounds)
            if result.passed:
                continue
            if (
                can_retry
                and not result.refused
                and run.check_retries < MAX_CHECK_RETRIES
            ):
                run.check_retries += 1
                return failure_message(result)
            if not quiet:
                run.events.brief_check_failed(run.check_retries)
            return None
        return None

    def _workspace_for_brief(self) -> str:
        return self.executor.workspace_root or os.getcwd()

    async def _run_single_attempt(
        self, task: TaskRecord, query: str, run: _RunState
    ) -> None:
        """``attempt_policy="none"``: one attempt with budget ``max_steps``.

        No reflection call is made. An attempt that runs out of steps ends
        the run as ``budget_exhausted``, one that gets stuck as ``stuck``.
        """
        run.attempts += 1
        run.last_attempt = True
        _, failure_type = await self._run_attempt(
            task, query, self.options.max_steps, run
        )
        if self._notes_mode:
            self._build_digest(query, failure_type, run, task.task_id)
            run.attempt_calls.clear()
            self.hippo.working.notes.supersede_attempt(run.attempts)
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
