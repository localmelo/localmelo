from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import time
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import Any, Literal

from localmelo.melo.agent._answer_format import (
    closing_line_of,
    has_closing_line,
    requested_closing_label,
)
from localmelo.melo.agent._approval_gate import _ApprovalGate, _ToolAsk
from localmelo.melo.agent._brief_aids import (
    DROPPED_KEPT,
    REFLECTION_ADDENDUM,
    STEP_PUSH,
    STEP_PUSH_STEPS,
    Dropped,
    dropped_notes,
    render_dropped,
    start_with,
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
    BLOCKED_ANSWER_NOTE,
    BUDGET_ACCOUNTING,
    BUDGET_HOOK,
    CALL_RECORDER,
    PREFIX_RUN,
    PREFIX_VERSION,
    RETRY_HOOK,
    TOOL_OVER_ANSWER_NOTE,
    CallTelemetry,
    Chat,
    RequestStats,
)
from localmelo.melo.agent._checkpoint import _CheckpointLoop
from localmelo.melo.agent._context_fit import ContextFitError, PromptCapError
from localmelo.melo.agent._earlier import check_earlier_turns, earlier_block
from localmelo.melo.agent._events import _Emitter, check_sink
from localmelo.melo.agent._notes_loop import _NotesLoop
from localmelo.melo.agent._pause import Hold
from localmelo.melo.agent._picture_refusal import names_pictures
from localmelo.melo.agent._recall import RECALL_RUN, _RecallLoop
from localmelo.melo.agent._reflection import (
    _cap_reflection_lists,
    _fit_reflection,
    _json_error_line,
    _read_reflection,
    _reflection_from_object,
    _reflection_summary,
    _stop_reason,
    _unreadable_reason,
)
from localmelo.melo.agent._step_review import RunReviews, StepReviewer, answer_on_block
from localmelo.melo.agent._task_aids import FINISH_NOTE, refusal_note
from localmelo.melo.agent._tool_images import (
    TEXT_READERS,
    USER_KEEP,
    ToolImageStore,
    withheld_note,
)
from localmelo.melo.agent.options import (
    NOTE_MODES,
    REPLAY_TURN_CAP,
    AgentOptions,
    reasoning_refusal,
)
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import (
    ExecutorResultPayload,
    MemoryWritePayload,
    ToolResolutionResult,
)
from localmelo.melo.checker.policy import builtin_set_for
from localmelo.melo.checker.tiers import (
    changes_nothing,
    tier_of,
    tier_requires_approval,
)
from localmelo.melo.checker.validators import (
    MAX_IMAGES_PER_RESULT,
    MAX_MEMORY_TEXT_LEN,
    _sanitize_images,
    image_problem,
)
from localmelo.melo.contracts.approvals import (
    RUN_ACTS,
    SELF_ASKING_TOOLS,
    TOOL_ASK,
    ApprovalPolicy,
    find_auto,
    is_cancellation,
)
from localmelo.melo.contracts.attachments import words_before_files
from localmelo.melo.contracts.events import EventSink
from localmelo.melo.contracts.folders import WRITE_TOOLS
from localmelo.melo.contracts.inbox import InboxError, InboxSource
from localmelo.melo.contracts.pause import PAUSE
from localmelo.melo.contracts.providers import (
    LLM_REQUEST_EXTRA_BODY,
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.contracts.sign_in import hide_passwords
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.contracts.tools import (
    CALL_HANDS_OFF,
    CALL_REFUSED,
    TOOL_MARKS,
    BaseToolProvider,
    ProvidedTool,
)
from localmelo.melo.executor import (
    TOOL_EXIT_CODE,
    AutoConfinement,
    Executor,
    WriteConfinement,
    confine,
    platform_note,
    register_builtins,
)
from localmelo.melo.executor.env import child_env
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.hippo import (
    KINDS,
    SIGNALS,
    classify_user_text,
    request_candidates,
    user_candidates,
)
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.notes import SourceRecordLog, note_from_reply, split_note
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
from localmelo.melo.memory.procedures.tracker import Limits, progress_state, shown_view
from localmelo.melo.memory.recall import (
    DEFAULT_TRIGGERS,
    RECALL_DATA_NOTE,
    RECALL_TOOL_DEF,
    RECALL_TOOL_TIMEOUT_S,
    RecallRecord,
    build_attempt_query,
    build_start_query,
    default_kinds,
    resolve_plan,
)
from localmelo.melo.memory.short import group_turns
from localmelo.melo.memory.short.note_store import NoteStoreEntry
from localmelo.melo.schema import (
    ANSWER_NOT_KEPT_LABEL,
    ASSISTANT_TEXT_SHARE,
    CALL_INTERRUPTED,
    CALL_NOT_STARTED,
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    EXTEND_WHEN,
    LOCAL_WINDOW_MESSAGES,
    MAX_REFLECTION_CHARS,
    PICTURES_REFUSED_ERROR,
    SHORT_TERM_MAX,
    TOOL_CALL_TEXT_ERROR,
    TOOL_CALL_TEXT_LABEL,
    TOOL_CALL_TEXT_NUDGE,
    UNFINISHED_ANSWER_ERROR,
    WINDOW_EVICT_CHUNK,
    WINDOW_FIT_TARGET,
    WINDOW_MESSAGES_MAX,
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
    WorkCheckpoint,
    answer_text,
    broken_tool_call,
    clip_session_text,
    describe_images,
    exception_text,
    has_content,
    new_record_id,
    new_tool_call_id,
    split_thinking,
    thinking_answer,
    unfinished_as_thinking,
)

logger = logging.getLogger(__name__)


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


def _answer_of(reply: Message) -> str:
    """The answer in the reply to a request without tools, or "".

    A blank reply gives none, nor does a tool call written out as text, which
    such a request can still get.
    """
    answer = reply.content.strip()
    return "" if not answer or broken_tool_call(answer) else answer


def _replay_plan(
    window: list[Message],
    options: AgentOptions,
    *,
    task_query: Message | None = None,
    requires_replay: bool = False,
) -> tuple[list[Message], list[int]]:
    """The window as a request carries it, and its cap skips.

    The first list is the window to send. In ``legacy`` it is *window*
    itself, unchanged. Otherwise it is a new list in which an assistant
    message keeps its thinking only in ``replay`` and ``compress+replay``,
    in scope (the current task from *task_query* on, or the whole window),
    among the newest ``replay_turns`` such turns, and within
    :data:`REPLAY_TURN_CAP`; a message that fails one of these is a copy
    with an empty thinking, one that passes is the original object. The
    second list holds the indexes in *window* of the assistant turns whose
    thinking was left out because it exceeds :data:`REPLAY_TURN_CAP` (empty
    on a replay-required provider, and in ``legacy``/``drop``). On a
    replay-required provider the whole window is replayed: the task scope,
    ``replay_turns`` and the cap do not apply, since a turn the server needs
    is not one the caller can leave out.
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


# Characters of a withheld tool result that stay visible to the model. A
# result left less room than this in its turn is withheld, not cut.
WITHHELD_PREVIEW_CHARS = 200

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


def _without_words(tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """*arguments* of a command or script without the model's ``why``.

    The words only describe the step: the same command with other words is the
    same step, so a refusal or a stuck loop is not dodged by rewording (#454).
    """
    if tool_name in ("shell_exec", "python_exec") and "why" in arguments:
        return {k: v for k, v in arguments.items() if k != "why"}
    return arguments


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


def _read_notes(
    inbox: InboxSource, pending: list[tuple[int, str]]
) -> tuple[list[tuple[int, str]], list[ImagePart]]:
    """The notes a step reads, and the pictures that go with them.

    An inbox that can name the pictures of a note (``pictures(offset)``, see
    :class:`~localmelo.melo.contracts.inbox.PictureInbox`) has them sent with
    its words. A step reads whole notes while their pictures stay within what
    the store keeps (:data:`USER_KEEP`): the first note is always read, and any
    after it that would pass the limit wait, unread, for the next step. A
    picture the checker would refuse is left out.
    """
    ask = getattr(inbox, "pictures", None)
    if not callable(ask):
        return pending, []
    taken: list[tuple[int, str]] = []
    shown: list[ImagePart] = []
    for offset, text in pending:
        own = [p for p in ask(offset) if p.data and not image_problem(p)][:USER_KEEP]
        if taken and len(shown) + len(own) > USER_KEEP:
            break
        taken.append((offset, text))
        shown.extend(own)
    return taken, shown


def _melos(inbox: InboxSource | None, offset: int) -> bool:
    """Whether note *offset* of *inbox* is melo's own wording (Allow once's go-ahead).

    An inbox says so with ``melos(offset)``, as one that carries pictures says so
    with ``pictures(offset)``. The run sends such a note to the model but counts
    none of it as the person's words.
    """
    ask = getattr(inbox, "melos", None)
    return callable(ask) and bool(ask(offset))


def _tool_message_content(
    text: str, images: Sequence[ImagePart], unseen: str = ""
) -> str:
    """Content of the ``role="tool"`` message that answers a call.

    *text* is the result's text (see :func:`_tool_result_text`), cut to
    fit or not, or the notice that replaced it. The message keeps only
    references to a result's images (their pixels reach the model only
    with ``tool_images`` on, see :class:`ToolImageStore`), so a result
    with images ends with a line that names them (see
    :func:`describe_images`), whether its text is kept, cut or withheld.
    When the pixels do not reach the model, *unseen* is the line that
    says so (see :func:`withheld_note`); it follows the one that names the
    images, and a result without images gets none.
    """
    lines = (text, describe_images(images), unseen if images else "")
    return "\n".join(part for part in lines if part)


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


# Empty replies in a row that end the run as failed ("empty replies"). A
# reply that writes a tool call as text counts as one. Until then, working
# memory records each one as EMPTY_REPLY_LABEL followed by EMPTY_REPLY_NUDGE
# (TOOL_CALL_TEXT_LABEL and TOOL_CALL_TEXT_NUDGE for a tool call written as
# text; see melo.schema), and the model plans again (see Agent._run_attempt).
MAX_EMPTY_REPLIES = 2
# What each call of a step that mixed a hand-off call with others reads: none ran.
HAND_OFF_ALONE = (
    "Not run: handing the rest of the work on must be the only call of its step. "
    "Nothing in this step ran; call it again alone, or keep working here."
)
# What a call to a tool that takes no more calls in this run reads (see
# ToolResult.no_more_calls): the call did not run. The first line is the one the
# person sees beside the step; the model reads both.
TOOL_OVER = (
    "This tool takes no more calls in this turn, so this call did not run.\n"
    "Do not call it again; go on without it."
)
# The answer of such a run when the request for its answer gives none.
TOOL_OVER_ANSWER = (
    "I stopped here: a step I needed takes no more tries in this turn, so I could "
    "not do all of what you asked."
)

# The line a brief run adds to the window when an attempt boundary finds the
# workspace unchanged since the run started.
NO_CHANGE_PUSH = (
    "Nothing has changed in the workspace yet. Start this attempt by writing the "
    "part of the task you can already write, then continue."
)


# SQLite directory when LOCALMELO_PERSIST_MEMORY is set without
# LOCALMELO_MEMORY_DIR. Expanded when used, not at import.
DEFAULT_MEMORY_DIR = "~/.cache/localmelo/memory"

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
    # run_task(hide_tools=): the tools this run is not offered.
    hidden: frozenset[str] = frozenset()
    # The tools whose result said they take no more calls in this run
    # (ToolResult.no_more_calls): not offered from the next step on, and a call
    # to one does not run. Empty at the start of every run, never stored.
    withdrawn: set[str] = dataclasses.field(default_factory=set)
    # Whether the run's one non-novel continue is spent.
    patience_spent: bool = False
    # answer_on_stop: whether result is the answer of the extra request.
    answered_on_stop: bool = False
    # Whether the run's last step handed the rest of its work on (hand_off).
    handed_off: bool = False
    # answer_on_stop: whether result got the closing line the task asks for.
    format_fixed: bool = False
    # How the run ended on its own limit or on a refusal: "calls" (max_calls) or
    # "refused" (a request refused for a budget or rate limit, with answer_on_stop).
    early_stop: str = ""
    # The latest non-blank text of a planning reply in this run: the
    # answer of a reflection's "done" that gives none, and the result of a
    # final boundary whose reflection is unreadable.
    last_text: str = ""
    # The thinking of the planning reply that set last_text, and of the last
    # reflection reply read; _record_answer replays the one behind the result.
    last_text_thinking: str = ""
    reflection_thinking: str = ""
    # The thinking of the reply that gave the result, once the run ends.
    answer_thinking: str = ""
    # Planning replies in a row that are no answer (neither a tool call nor
    # text, or a tool call written as text), and whether one of them was only
    # unfinished thinking (#278) or a tool call written as text.
    empty_replies: int = 0
    unfinished_row: bool = False
    broken_row: bool = False
    # The latest thinking of the row's replies that had any, which is the answer
    # when the row ends with the model's words only there (#469); whether the
    # row has used its one retry with thinking turned off, and whether the next
    # planning call is that retry.
    row_thinking: str = ""
    thinking_retry_used: bool = False
    thinking_retry_armed: bool = False
    # empty_reply_fallback, over the whole run (the row's state above resets
    # with each tool call): whether a turn was asked again with thinking off,
    # and whether the result is the marked end of a model's thinking.
    fallback_retried: bool = False
    fallback_answered: bool = False
    # The run's task id ("" for a call without a task record).
    task_id: str = ""
    # The query Message this run appended to the window, for replay scope.
    task_query_msg: Message | None = None
    # The chat's earlier turns as a block (run_task's earlier_turns), which the
    # reflection, the answer on stop and a planning request whose window lost
    # them carry; "" for a run without them.
    earlier_block: str = ""
    # Whether the window held a message before this run's query when it began:
    # only then can a planning request find the window has lost them.
    window_had_turns: bool = False

    @property
    def plan_earlier(self) -> str:
        """The earlier-turns block for planning requests (see ``Chat.plan_step``)."""
        return self.earlier_block if self.window_had_turns else ""

    # User messages the run inserted mid-run (inbox): kept at a boundary drop.
    pinned: list[Message] = dataclasses.field(default_factory=list)
    # The message that carries the latest reflection in the window, put back
    # once if an eviction takes it.
    reflection_msg: Message | None = None
    # An opt-in source of parent instructions at planning boundaries.
    inbox: InboxSource | None = None
    # Replay counters (see memory_stats on RunResult; replay_stripped is the
    # chat's RequestStats.thinking_cleared).
    replayed_chars: int = 0
    replay_skipped: int = 0
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
    # refused_steps: whether a policy check refused the call _do_execute ran
    # last, and the steps of this run that did not count for it.
    call_refused: bool = False
    free_steps: int = 0
    # How many times the run was given more steps for making progress (#547),
    # and the calls it made so far, which progress is read against.
    extensions: int = 0
    calls_made: set[str] = dataclasses.field(default_factory=set)
    refused_calls: set[str] = dataclasses.field(default_factory=set)
    # Whether this run may change anything (run_task's acts): False offers, and
    # so runs, only what changes nothing, whatever the model calls.
    acts: bool = True
    # Whether the person sent the run's query (run_task's from_person): only
    # then may a path outside the workspace be asked about (#405).
    from_person: bool = True
    # The calls the gate reviewed, the model reviews sent and auto mode's
    # blocks (#339); run.finished carries the summary when a policy ran.
    reviews: RunReviews = dataclasses.field(default_factory=RunReviews)
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
    # The work checkpoint (melo.agent._checkpoint): the cues of the tool results
    # that left the context (by call id, once each) and whether the block has
    # yet to show the newest of them; the latest checkpoint and the message
    # that carries it in the window; how many the run wrote; and the run's
    # calls of the file tools that write, as (tool, arguments, status).
    cues: list[Dropped] = dataclasses.field(default_factory=list)
    cue_ids: set[str] = dataclasses.field(default_factory=set)
    cues_pending: bool = False
    checkpoint: WorkCheckpoint | None = None
    checkpoint_msg: Message | None = None
    checkpoints: int = 0
    # The early line (EARLY_LINE): the trailing line the next planning request
    # carries, and whether this fill has had it.
    early_note: str = ""
    early_sent: bool = False
    # Where the window's fill began: the estimate the last checkpoint left, 0
    # before the first. The early line comes halfway from it to the line.
    fill_start: int = 0
    # The person's own words for this run (run_task's user_words, else the
    # person's part of the query): a work checkpoint's goal.
    words: str = ""
    writes: list[tuple[str, dict[str, Any], str]] = dataclasses.field(
        default_factory=list
    )
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
    # The person's hold on the run, such as a page they took over (contracts.pause).
    hold: Hold = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.hold = Hold(self.events.run_paused, self.events.run_resumed)

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
                "The trace of task %s stopped: %s", self.task_id, exception_text(exc)
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
                exception_text(exc),
            )
            return
        self.events.progress(state)


def _reviews_of(run: _RunState) -> dict[str, Any] | None:
    """What ``run.finished`` reports of the run's reviews: only when a policy ran."""
    return None if run.approval is None else run.reviews.summary()


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


class _ToolSetupError(Exception):
    """The tool providers could not be set up; the message is the run's error."""


class Agent(_RecallLoop, _NotesLoop, _CheckpointLoop):
    """Task-solving agent with tool use and memory.

    It is built from providers::

        agent = Agent(llm=my_llm_provider, embedding=my_emb_provider)

    The providers of a config file come from
    :func:`localmelo.support.backends.providers_for`. ``options=AgentOptions(...)``
    sets step budgets, the attempt policy, tool exposure, the built-in tools,
    memory and a run timeout; the defaults keep the behavior from before
    options existed,
    except that every step now offers all registered built-ins and runs only
    the tools it offered, and that a reflection decides on its own checks
    whether the next attempt starts, a reply without a reflection does not
    stop the run, and an over-long reflection is trimmed (see
    :meth:`_run_reflective_attempts`).
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
        *,
        llm: BaseLLMProvider,
        embedding: BaseEmbeddingProvider | None = None,
        options: AgentOptions | None = None,
        tool_providers: Sequence[BaseToolProvider] = (),
    ) -> None:
        self._llm = llm
        self._embedding = embedding

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
        folder = workspace_root or os.path.realpath(os.getcwd())
        if self.options.confine_writes:
            env = child_env(self.options.tool_env)
            self._confinement = confine(folder, env)
        # The calls auto mode decides run under a profile of their own
        # (#339): nobody approves them, so it confines their writes even
        # without confine_writes and closes melo's own gateway to them. The
        # other calls start under the Agent's confinement, as before.
        tool_env = self.options.tool_env
        self._auto_confinement = AutoConfinement(
            folder, lambda: child_env(tool_env), base=self._confinement
        )
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
        # Of those, melo's own browser tools that ask their questions themselves:
        # the only calls the approval gate does not ask a second question about.
        self._self_asking_names: frozenset[str] = frozenset()
        # Each hand-off tool and the boolean argument that makes its call one.
        self._hand_off_args: dict[str, str] = {}
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
        # The latest run's earlier-turns block, for closing_answer after it.
        self._earlier_block = ""

        # Optional persistent memory backends: options.memory_dir, or the
        # LOCALMELO_PERSIST_MEMORY / LOCALMELO_MEMORY_DIR opt-in. History and
        # the procedural store are SQLite whenever persistence is on;
        # long-term only when memory is on and the reasoning mode is not
        # legacy.
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
            # A legacy run opens no long-term store; an existing file is left
            # as it is. The explicit modes keep a text-only store.
            if self.options.memory and self.options.reasoning != "legacy":
                long_backend = SqliteLongTerm(
                    os.path.join(self._memory_dir, "long_term.db")
                )
            # Opened, and created, at its first use: a run's end.
            procedures_backend = SqliteProcedures(
                os.path.join(self._memory_dir, "procedures.db")
            )

        # memory=False runs Hippo in its no-embedding mode, which is exactly
        # the documented behavior of that option: no long-term recall, no
        # long-term writes, and reflections are still cleared.
        # self._embedding stays set so close() closes it.
        # Pictures tools return (AgentOptions.tool_images): the store whose
        # payloads planning requests carry; "auto" asks the provider once.
        self._tool_images = ToolImageStore(
            self.options.tool_images_keep, self.options.tool_images_retire
        )
        self._images_on: bool | None = {"on": True, "off": False}.get(
            self.options.tool_images
        )
        from localmelo.melo.agent._request_accounting import context_budget

        # A set budget is known now; otherwise the server is asked once, before
        # the first run (_ensure_budget).
        budget = context_budget(
            self.options.context_window,
            self.options.output_reserve,
            self.options.safety_margin,
        )
        self._budget_asked = budget is not None
        self._budget_lock = asyncio.Lock()
        self.hippo = Hippo(
            short_term_max=self._window_for(budget is not None),
            embedding=self._embedding if self.options.memory else None,
            history=history_backend,
            long=long_backend,
            procedures=procedures_backend,
            short_term_empty_prefix=self.options.reasoning != "legacy",
            short_term_evict_chunk=WINDOW_EVICT_CHUNK,
            memory=self.options.memory,
            keep_decision_text=self.options.keep_decision_text,
        )
        self.checker = Checker(mode=self.options.mode, builtins=self.options.builtins)
        self.executor = Executor(
            self.hippo,
            self.checker,
            workspace_root=workspace_root,
            read_roots=self.options.read_folders,
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
        self._retrievers = {}
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
        if self.options.reasoning in NOTE_MODES:
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
        self.chat = Chat(
            self._llm,
            system_notes=notes,
            note_instruction=instruction,
            empty_prefix=self.options.reasoning != "legacy",
            context_budget=budget,
            fit_target=WINDOW_FIT_TARGET,
            identity=self.options.identity or "",
            max_calls=self.options.max_calls,
            reserve_answer=self.options.answer_on_stop,
            # Read at each request, so a checker swapped in later is the one asked.
            prompt_check=lambda messages, **kw: self.checker.pre_plan(messages, **kw),
        )
        # The user's model reviews steps the rules leave open, under a grant
        # or in auto mode; it keeps this chat's blocks and the user's words.
        self._reviewer = StepReviewer(lambda: self.chat, repeats=self.options.repeats)

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
            confinement=self._auto_confinement,
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
            )
            self._builtin_names = self._builtin_names | {"recall"}
            self._recall_tool_registered = True

    # ── Workspace, tools and conversation ──

    @property
    def tool_providers(self) -> tuple[BaseToolProvider, ...]:
        """The tool providers this Agent was made with (read only)."""
        return self._tool_providers

    @property
    def workspace_root(self) -> str | None:
        """The real path of the agent's workspace root, or ``None``."""
        return self.executor.workspace_root

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

    def _unseen_line(self) -> str:
        """The line that ends the tool message of a picture the model is not sent.

        Empty unless :meth:`_sends_images` decided no. It names the tools that read
        a page as text among those the agent has.
        """
        if self._images_on is not False:
            return ""
        return withheld_note(
            [name for name in TEXT_READERS if self.hippo.get_tool(name) is not None]
        )

    def _window_for(self, budgeted: bool) -> int:
        """The working window's cap: ``window_messages``, else the default for a run with or without a token budget."""
        if self.options.window_messages is not None:
            return self.options.window_messages
        return WINDOW_MESSAGES_MAX if budgeted else LOCAL_WINDOW_MESSAGES

    @property
    def window_messages(self) -> int:
        """The working window's message cap in force now."""
        return self.hippo.working.max_len

    async def _ensure_budget(self) -> None:
        """Ask the provider for the model's context length once, when nothing set a budget.

        A length gives the chat its budget (see
        :func:`~localmelo.melo.agent._request_accounting.context_budget`), and an
        unset window then widens to :data:`WINDOW_MESSAGES_MAX`, so the budget's
        fit is the limit that acts. No length, or a question that fails, leaves
        the prompt cap as it was. A cancelled question is asked again.
        """
        if self._budget_asked:
            return
        async with self._budget_lock:
            if self._budget_asked:
                return
            from localmelo.melo.agent._request_accounting import context_budget

            try:
                reported = await self._llm.context_length()
                limit = self._llm.output_limit()
            except Exception:  # a failed question is an unknown length
                logger.info("Could not ask the model server for its context length")
                reported = limit = None
            self._budget_asked = True
            budget = context_budget(
                None,
                None,
                None,
                reported=reported if type(reported) is int else None,
                output_limit=limit if type(limit) is int and limit > 0 else None,
            )
            if budget is not None:
                self.chat.context_budget = budget
                self.hippo.working.widen(self._window_for(True))

    async def start(self) -> None:
        """Ask for the context budget and start the tool providers now.

        That is the setup the first run would do otherwise (see
        :meth:`_ensure_budget` and :meth:`_ensure_tools`), under the same
        locks, so it happens once: a later call, and the first run, find it
        done. A tool setup's exception propagates, also the ``RuntimeError``
        of a setup that failed earlier or of an agent that is closing. Start
        an agent whose runs a caller may cancel before its first run: a
        cancellation during the tool setup fails it for the agent's life.
        """
        await self._ensure_budget()
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

    def hide_passwords(self, text: str) -> str:
        """*text*, a message of the person's, with each password it gives replaced by its handle (#547).

        melo signs in with a password the person gives in a message, but the
        value never reaches a model, a record, a log or a memory: the handle
        (``[password-1]``) stands for it, and only the browser tool, asked by
        code that checked the site and the turn, types the value. A caller
        that stores or shows the person's message calls this first (a
        session's runner hides every part of the message, its files too, with
        the same function); :meth:`run_task` does it again, which changes
        nothing. All of *text* is read for passwords, whoever wrote it: a
        caller that put text of melo's after the person's message passes only
        the person's part here (see *own_chars* of :meth:`run_task`, which
        hides what follows it too). See :mod:`localmelo.melo.contracts.sign_in`.

        Hiding never crosses the place where a file's block starts (#626): the
        words before the first block and the rest are hidden as two parts and
        joined as they were, so a password word that ends the person's words
        cannot take the block's opener as its value, whichever order the
        caller hides and cuts in (the step review's words end at that opener).
        A text with no block is hidden as one.
        """
        words = words_before_files(text)
        return hide_passwords(words) + hide_passwords(text[len(words) :])

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
        unchanged, and a request's fit can leave out thinking before facts.
        What the step review kept of the chat (the user's words, settled
        reviews, blocks) starts over; with ``[]`` the chat also begins now,
        for what the review counts as changed in this chat.

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
        # Another chat, an undo or a reload: the step review starts over too;
        # an empty conversation is a new chat, which begins now.
        self._reviewer.reset(new_chat=not loaded)
        loaded = copy.deepcopy(loaded)
        if self.options.reasoning != "legacy":
            # Normalize legacy checkpoints once, before any request carries
            # them (and its fit can leave their thinking out).
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
                self._tools_failure = str(exc) or exception_text(exc)
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
            # Both the provider's word and melo's list: a name alone does not
            # count, and a flag on any other name does nothing.
            self._self_asking_names = frozenset(
                tool.definition.name
                for tool in provided
                if tool.asks_itself and tool.definition.name in SELF_ASKING_TOOLS
            )
            self._hand_off_args = {
                tool.definition.name: tool.hands_off
                for tool in provided
                if tool.hands_off
            }
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
                "Could not kill tool provider %s: %s", name, exception_text(exc)
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
                exception_text(exc),
            )

    # ── Stage helpers ──
    # Each helper owns one phase of the agent loop.  They return a
    # short status string when the loop should break, or None to
    # continue to the next stage.

    async def _do_retrieval(
        self, query: str, *, run: _RunState | None = None
    ) -> tuple[list[Message], list[ToolDef], str | None]:
        """Stage 1+2: resolve tools and run boundary checks.

        With ``tool_exposure="retrieve"`` the tools are every registered
        built-in, sorted by name, then the other tools that the model's
        tool calls and BM25 resolve among themselves, also sorted by name;
        with ``"all"`` they are every registered tool, sorted by name.
        Either way, a tool the run's caller hides (``hide_tools``) and a tool
        whose result said it takes no more calls in this run
        (``ToolResult.no_more_calls``, the run's ``withdrawn`` set) are left
        out. The second changes the tools part of the request once, at the
        first step after the result: the prefix a server caches ends before it
        from then on, which is accepted.

        The window is returned whole: the planning request is fitted on a
        copy when it is sent (see :class:`Chat`).

        Returns (short_window, tools, fail_reason).
        *fail_reason* is non-None when a check fails and the loop must break.
        """
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
        short_window = self.hippo.working.get_window()

        tool_hints: list[str] = []
        if self.options.tool_exposure == "all":
            # The same list on every step, so the tools part of the request
            # stays byte-stable for prefix caching; a withdrawn tool leaves it
            # once, below (see the docstring).
            if run is not None:
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
            # Hints are the model's tool calls, which live in the window.
            tool_hints = [
                name
                for name in self.hippo.extract_tool_hints(short_window)
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
        if run is not None and (gone := run.hidden | run.withdrawn):
            tools = [t for t in tools if t.name not in gone]
        if run is not None and not run.acts:
            # What the step offers is what the executor runs (TOOL_NOT_OFFERED).
            tools = [t for t in tools if changes_nothing(t)]

        resolution_check = self.checker.check_tool_resolution(
            ToolResolutionResult(
                query=query,
                hints=tool_hints,
                resolved_tool_names=[t.name for t in tools],
            )
        )
        if not resolution_check.allowed:
            return (
                short_window,
                tools,
                (f"Tool resolution failed: {resolution_check.reason}"),
            )

        return short_window, tools, None

    def _thinking_retry_body(self) -> Mapping[str, Any]:
        """The request fields that turn thinking off for one retry (#469), if any."""
        find = getattr(self.chat, "thinking_retry_body", None)
        body = find() if callable(find) else None
        return body if isinstance(body, Mapping) else {}

    async def _do_plan(
        self,
        short_window: list[Message],
        tools: list[ToolDef],
        query: str,
        *,
        task_query: Message | None = None,
        run: _RunState | None = None,
        on_ready: Callable[[list[Message]], None] | None = None,
    ) -> tuple[Message, str | None]:
        """Stage 3: LLM planning step with post-plan check.

        The request carries :func:`_replay_plan`'s window, fitted when it is
        sent (see :class:`Chat`; a request the prompt check refuses raises
        :class:`PromptCapError`). In ``replay`` the characters of thinking
        the fitted request carries are counted, and a turn it carries that
        was skipped for the per-turn cap is counted too (see
        :meth:`_count_sent`); in ``drop`` both are zero; in ``legacy``
        neither counter is used. An explicit-mode reply's inline
        ``<think>`` block is moved into thinking before the run reads it,
        so the result and tool-hint scan see the clean answer.

        Returns (response, fail_reason).
        """
        sent, skipped = self._sent_window(short_window, task_query)
        ready = on_ready
        if run is not None:
            ready = self._count_sent(run, skipped, on_ready)
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
        if ready is not None:
            plan_args["on_ready"] = ready
        if on_ready is None and run is not None:
            # Not at a step that carries inbox lines: they must end the request.
            # A brief run's drops are named once: in the work checkpoint's cue
            # list while the window holds one, else in its tail line.
            named = run.checkpoint_msg is not None and self.hippo.working.holds(
                run.checkpoint_msg
            )
            tail = [
                (
                    render_dropped(run.dropped, sent)
                    if run.brief and run.dropped and not named
                    else ""
                ),
                run.early_note,
            ]
            run.early_note = ""
            if any(tail):
                plan_args["tail_note"] = "\n\n".join(line for line in tail if line)
        response = await self.chat.plan_step(
            short=sent,
            tools=tools,
            query=query,
            query_block=query_block,
            recalls=recalls,
            earlier="" if run is None else run.plan_earlier,
            **plan_args,
        )
        if run is not None:
            run.plan_call = getattr(self.chat, "last_call", None)
            if run.plan_call is not None:
                # What the fit left out of this request is no longer in context.
                self._note_cues(run, run.plan_call.dropped)
        if run is not None and recall_records:
            self._commit_recalls(recall_records, run)
        response = self._normalize_reply(response)

        check = await self.checker.post_plan(response)
        if not check.allowed:
            return response, f"Response check failed: {check.reason}"

        return response, None

    def _sent_window(
        self, window: list[Message], task_query: Message | None
    ) -> tuple[list[Message], list[Message]]:
        """*window* as a planning request carries it, and the turns it skips.

        The window of :func:`_replay_plan` with the live images' payloads; the
        second list holds the assistant turns left without their thinking for
        the per-turn cap (see :meth:`_count_sent`).
        """
        sent, skipped = _replay_plan(
            window,
            self.options,
            task_query=task_query,
            requires_replay=self._requires_replay,
        )
        return self._tool_images.hydrate(sent), [sent[i] for i in skipped]

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
        cancellation of the run (see ``is_cancellation``), its
        ``timeout_s`` included, and ``crashed`` for any other exception. A
        cancelled call's error says that it did not start when the policy
        was still deciding about it, and that it may have partly run
        otherwise. ``tool.finished`` carries the policy's answer whenever
        the policy was asked. The call runs with a new ``TOOL_EXIT_CODE``
        list, so a tool can record its child's exit status for the event, and
        a new ``TOOL_MARKS`` dict, so a tool can mark its own call.
        """
        events = run.events
        # The run's gate for tools, beside the one in Chat._request: what the model
        # chose before the person took a page starts once they give the run back.
        await run.hold.wait()
        events.tool_started(call, self.executor.workspace_root)
        # A list for an event, and in the compress modes for the rehearsal
        # signal too; elsewhere, without a sink, the slot stays unset as
        # before.
        exit_codes: list[int] | None = [] if events.active or self._notes_mode else None
        slot = TOOL_EXIT_CODE.set(exit_codes) if exit_codes is not None else None
        # What the tool says of its own call (``allowed_local``, ``phases``), for the event.
        marks: dict[str, Any] | None = {} if events.active else None
        marks_slot = TOOL_MARKS.set(marks) if marks is not None else None
        gate = None
        ask_slot = None
        # The questions the call's tool asked itself that were refused (#431).
        refused: list[dict[str, Any]] = []
        # What the tool's own check refused (refuse_call), for refused_steps.
        refusals: list[str] = []
        refusal_slot = CALL_REFUSED.set(refusals)
        acts_slot = RUN_ACTS.set(run.acts)
        run.call_refused = False
        switch = self._auto_confinement
        if switch is not None:
            # A call auto mode decides runs confined; the gate settles it.
            switch.on = run.approval is not None and find_auto(run.approval) is not None
            switch.allow()  # the gate adds the paths of a step it lets run
            if switch.on:
                await asyncio.to_thread(switch.ready)
        if run.approval is not None:
            called_tool = self.hippo.get_tool(call.tool_name)
            tier: ToolTier = (
                "write" if called_tool is None else self.checker.tier(called_tool)
            )
            hints = {} if called_tool is None else called_tool.annotations
            gate = _ApprovalGate(
                run.approval,
                run,
                call,
                self.executor.workspace_root,
                tier,
                self.checker,
                self._reviewer,
                destructive=hints.get("destructiveHint") is True,
                confinement=switch,
                asks_itself=call.tool_name in self._self_asking_names,
            )
            ask_slot = TOOL_ASK.set(
                _ToolAsk(
                    run.approval,
                    run,
                    call,
                    self.executor.workspace_root,
                    self._reviewer,
                    refused,
                    lambda: self._images_on is True,
                )
            )
        # The goal reviews used before the call: a call that runs and succeeds
        # gives them back (#583); a consent review stays spent.
        reviewed = run.reviews.spent()
        try:
            exec_request = ExecutionRequest(
                tool_name=call.tool_name,
                arguments=dict(call.arguments),
                ask=gate is not None and gate.fresh,
            )
            outcome = await self.executor.execute_structured(
                exec_request, allowed=allowed, approve=gate
            )
            if outcome.status is ExecutionStatus.SUCCESS:
                run.reviews.refill(reviewed)

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
                    no_more_calls=outcome.no_more_calls,
                )
                # The model reads the error of a failed call, which the check
                # never cuts, and the output otherwise.
                cut = 0 if sp.error else max(len(outcome.output) - len(sp.output), 0)
            else:
                result, cut = outcome.to_tool_result(), 0
            if self.options.task_aids:
                note = refusal_note(result.error or result.output)
                if note and result.error:
                    result.error = f"{note}\n{result.error}"
                elif note:
                    result.output = f"{note}\n{result.output}"
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
                cancelled=is_cancellation(exc),
                exit_codes=exit_codes,
                not_started=run.waiting_call_id == call.id,
                approval=None if gate is None else gate.asked,
                asks=refused,
                marks=marks,
            )
            raise
        finally:
            CALL_REFUSED.reset(refusal_slot)
            if slot is not None:
                TOOL_EXIT_CODE.reset(slot)
            if marks_slot is not None:
                TOOL_MARKS.reset(marks_slot)
            if ask_slot is not None:
                TOOL_ASK.reset(ask_slot)
            RUN_ACTS.reset(acts_slot)
        run.call_refused = self._refused_free(
            call, result, outcome, gate, refusals, refused, run
        )
        if result.no_more_calls:
            run.withdrawn.add(call.tool_name)
        exported = (
            preview_result(captured[0], notes)[0] if captured is not None else result
        )
        events.tool_finished(
            call,
            outcome,
            exported,
            exit_codes=exit_codes,
            approval=None if gate is None else gate.asked,
            asks=refused,
            marks=marks,
        )
        run.note(
            lambda trace: trace.tool(
                call.tool_name, call.arguments, outcome.status.value, exported.output
            )
        )
        if call.tool_name in WRITE_TOOLS:
            run.writes.append(
                (call.tool_name, dict(call.arguments), outcome.status.value)
            )
        if not result.error.strip() and call.tool_name in (
            "file_read",
            "file_write",
            "file_edit",
        ):
            path = call.arguments.get("path")
            if isinstance(path, str) and path.strip():
                run.recent_path = path
        if captured is not None and run.inflight is not None:
            run.inflight.captured.append(captured)
        if self._notes_mode:
            run.attempt_calls.append((call, result))
            await self._rehearse_acted(call, result, run, exit_codes or ())
        return result, cut

    def _hands_off(self, call: ToolCall) -> bool:
        """Whether *call* is a hand-off call: its tool's hand-off argument is true."""
        name = self._hand_off_args.get(call.tool_name)
        return name is not None and call.arguments.get(name) is True

    def _not_run(
        self,
        call: ToolCall,
        run: _RunState,
        why: str,
        *,
        category: ErrorCategory = ErrorCategory.INVALID_ARGUMENTS,
    ) -> ToolResult:
        """*call*, refused before it ran: its events and an error result saying *why*.

        The event carries *category*, the reason the call did not run.
        """
        outcome = ExecutionOutcome(
            tool_name=call.tool_name,
            status=ExecutionStatus.ERROR,
            error=why,
            error_category=category,
        )
        result = outcome.to_tool_result()
        run.call_refused = False
        run.events.tool_started(call, self.executor.workspace_root)
        run.events.tool_finished(call, outcome, result, exit_codes=None)
        return result

    def _refused_free(
        self,
        call: ToolCall,
        result: ToolResult,
        outcome: Any,
        gate: _ApprovalGate | None,
        refusals: list[str],
        refused: list[dict[str, Any]],
        run: _RunState,
    ) -> bool:
        """Whether *call* was refused by a policy check, so its step may be free (refused_steps).

        The tool's own check (``refuse_call`` or :meth:`refused_by_tool`), a
        question it asked that a policy or Auto refused, or the approval
        policy's denial by a policy or Auto. The person's own Deny, or no
        answer in time, is theirs: asking again is never free. Neither is a
        call refused before in this run, by tool and arguments.
        """
        free = ("policy", "auto")
        denied = outcome.error_category == ErrorCategory.APPROVAL_DENIED
        asked = gate.asked if gate is not None else None
        by = str((asked or {}).get("origin") or "")
        if refused:
            policy = all(str(entry.get("origin")) in free for entry in refused)
        elif denied:
            policy = by in free
        else:
            policy = bool(refusals) or self.refused_by_tool(call, result)
        if not (policy or refused or denied):
            return False
        try:
            key = json.dumps(
                [call.tool_name, _without_words(call.tool_name, call.arguments)],
                sort_keys=True,
            )
        except (TypeError, ValueError):
            key = f"{call.tool_name}:{_without_words(call.tool_name, call.arguments)!r}"
        again = key in run.refused_calls
        run.refused_calls.add(key)
        return policy and not again

    def refused_by_tool(self, call: ToolCall, result: ToolResult) -> bool:
        """Whether *result* is the refusal of *call* by its tool's own check.

        For ``refused_steps``, beside ``refuse_call`` and the approval
        policy's denial: a subclass may read a tool's known refusal text, for
        a tool that does not mark its refusals. None here.
        """
        return False

    async def _do_memorize(
        self, task_id: str, step: _InflightStep, run: _RunState
    ) -> None:
        """Stage 5: record the step and write checked results to memory.

        *step* holds the planning reply and, for each of its calls, the
        checked result and the characters the executor-result check cut
        from it.

        - History gets one StepRecord per executed call (the assistant
          text is attached to the first), unconditionally.
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
            await self.hippo.store_step(task_id, record)
            step.stored += 1

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
        characters, half the prompt cap (``PROMPT_CAP_CHARS``), so that one
        step leaves room in the next planning prompt. The calls keep their share, and so does the assistant
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

        unseen = self._unseen_line()
        tool_msgs = [
            Message(
                role="tool",
                content=_tool_message_content(text, result.images, unseen),
                tool_call_id=call.id,
                tool_result=message_result(result, text, index),
                images=[] if index in withheld else result.images,
            )
            for index, (call, result, text) in enumerate(
                zip(calls, results, texts, strict=True)
            )
        ]
        return assistant_msg, tool_msgs

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
                json.dumps(
                    _without_words(tc.tool_name, tc.arguments),
                    sort_keys=True,
                    default=str,
                ),
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
    def _new_calls(calls: Sequence[ToolCall], run: _RunState) -> bool:
        """Whether this step made a call the run had not made before, and keep its calls.

        A call is its tool and arguments, without what the call returned: the
        same call again is no next step, whatever it reads this time.
        """
        keys = {
            hashlib.sha256(
                json.dumps(
                    [tc.tool_name, _without_words(tc.tool_name, tc.arguments)],
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()[:16]
            for tc in calls
        }
        new = not keys <= run.calls_made
        run.calls_made |= keys
        return new

    @staticmethod
    def _progressing(
        fingerprints: list[tuple[str, str, str]], fresh: list[bool]
    ) -> bool:
        """Whether the last five steps made progress: three different new steps ran with no error.

        *fresh* says of each step whether it made a call the run had not made
        before (:meth:`_new_calls`). Two steps that take turns (A, B, A, B, A)
        are a loop, not progress, and so is one call that keeps reading a page
        that changes, or a round of three calls that comes back.
        """
        done = {
            (name, step_h)
            for (name, step_h, error), new in zip(
                fingerprints[-5:], fresh[-5:], strict=True
            )
            if name and not error and new
        }
        return len(done) >= 3

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

    @staticmethod
    def _count_sent(
        run: _RunState,
        skipped: Sequence[Message],
        then: Callable[[list[Message]], None] | None = None,
    ) -> Callable[[list[Message]], None]:
        """The ``on_ready`` hook that counts the reasoning a request carries.

        Called with the request as it is sent, after its fit: the thinking
        characters it carries go to ``replayed_chars``, and each of the
        *skipped* turns (left without thinking for the per-turn cap) that
        it still carries to ``replay_skipped``. A fit works on copies, so a
        turn is matched by value. *then*, when given, is called first: a
        request it refuses is not counted.
        """

        def count(messages: list[Message]) -> None:
            if then is not None:
                then(messages)
            run.replayed_chars += sum(
                len(msg.thinking) for msg in messages if msg.role == "assistant"
            )
            run.replay_skipped += sum(1 for msg in skipped if msg in messages)

        return count

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
            fallback=_reflection_summary(response.content),
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
        When that reply has JSON that the decoder rejects and is not a tool
        call, one more line follows the note and quotes the decoder's error
        (see :func:`_json_error_line`).

        A reflection is a JSON object with a valid ``recommended_action``
        (see :func:`_read_reflection`, which repairs one common slip of the
        JSON and tells the event which), so a reply whose only JSON is
        something else, such as a tool call written out as text, is asked for
        again too. When the second reply holds no reflection either, this
        returns ``None``: the boundary has no reflection (see
        :meth:`_run_reflective_attempts`). A reflection over
        :data:`MAX_REFLECTION_CHARS` once serialized is trimmed to fit (see
        :func:`_fit_reflection`), whatever its decision: the decision and its
        rationale stay the model's, and its ``tried_memory`` is the trimmed
        entry.

        A reflection request is fitted when it is sent, as a planning one
        is (see :class:`Chat`); one that the prompt check still refuses is
        a reply that cannot be read: this returns ``None``.
        """
        short_window = self.hippo.working.get_window()
        task_query = run.task_query_msg if run is not None else None
        sent, skipped = _replay_plan(
            short_window,
            self.options,
            task_query=task_query,
            requires_replay=self._requires_replay,
        )
        on_ready = None
        if run is not None:
            on_ready = self._count_sent(run, [sent[i] for i in skipped])

        prior_reflections = self.hippo.working.get_reflections()
        note_lines: list[str] = []
        draft_titles: list[str] = []
        if self._notes_mode and run is not None:
            note_lines, draft_titles = self._reflection_note_inputs(run)
        # A brief run asks for pointers (REFLECTION_PROMPT stays frozen).
        addendum: dict[str, Any] = (
            {"addendum": REFLECTION_ADDENDUM} if run is not None and run.brief else {}
        )
        # The chat's earlier turns, which the window may have lost by now.
        earlier = run.earlier_block if run is not None else ""

        async def ask(**retry: Any) -> Message | None:
            """One reflection request; the reply, normalized and reported."""
            reply = await self.chat.reflect(
                short=sent,
                query=task.query,
                attempt_id=attempt_id,
                failure_type=failure_type,
                prior_reflections=prior_reflections or None,
                final=final,
                on_ready=on_ready,
                notes=note_lines,
                draft_titles=draft_titles,
                earlier=earlier,
                **retry,
                **addendum,
            )
            if reply is None:
                return None
            reply = self._normalize_reply(reply)
            if replies is not None:
                replies.append(reply)
            if run is not None:
                call = getattr(self.chat, "last_call", None)
                run.events.reflection_call(call)
                if call is not None:
                    self._note_cues(run, call.dropped)
            return reply

        response = await ask()
        if response is None:
            # The prompt check refused the request before it was sent; read
            # it as an unreadable reply (see the docstring).
            return None
        found = _read_reflection(response.content)
        if found is None:
            reason = _unreadable_reason(response)
            if run is not None:
                run.events.reflection_unreadable(reason)
            # JSON the decoder rejected: say where, so the retry can mend it.
            hint = _json_error_line(response.content) if reason != "tool_call" else ""
            # Often a tool call: the window ends in tool turns, and some
            # models continue them although the request offers no tools.
            # Ask once more, with those turns written out as text.
            response = await ask(retry=True, retry_hint=hint)
            if response is None:
                return None
            found = _read_reflection(response.content)
        reflection_note_id = ""
        if run is not None:
            run.reflection_thinking = response.thinking
            if self._notes_mode and found is not None:
                await self._cite_notes(self._used_notes(response.content), run)
                reflection_entry = self._store_reflection_note(response, task, run)
                if reflection_entry is not None and reflection_entry.note is not None:
                    reflection_note_id = reflection_entry.note.note_id
        if found is None:
            return None
        data, repair = found
        if run is not None:
            run.events.reflection_repaired(repair)
        entry, decision = _reflection_from_object(data, attempt_id, failure_type)
        entry = _fit_reflection(_cap_reflection_lists(entry), MAX_REFLECTION_CHARS)
        decision.tried_memory = entry
        if run is not None and self._notes_mode:
            self._propose_reflection(entry, run, note_id=reflection_note_id)
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

        A reply with neither a tool call nor text is no answer (text with no
        letter or digit, such as ``...``, is none): its step counts, working
        memory records it as :data:`EMPTY_REPLY_LABEL`, an assistant message,
        followed by :data:`EMPTY_REPLY_NUDGE` as a user message, and the
        model plans again. The label keeps the roles alternating, since some
        chat templates (Mistral, Gemma) reject two user messages in a row,
        such as the query and the nudge. Neither message goes to history,
        and neither counts as a reply of the model: the latest text, the
        empty-reply count and stuck detection only see what the model
        returned. The :data:`MAX_EMPTY_REPLIES`-th empty reply in a row in
        this run (reflections in between do not break the row) ends it as
        ``failed`` with the error ``empty replies``.

        A reply that is only unfinished thinking (``Chat.unfinished``, #278)
        is stored and shown with all its text as thinking and no content, and
        goes the same way with an ``unfinished_answer`` notice: the model is
        asked once more, and a row that held such a reply ends the run with
        :data:`~localmelo.melo.schema.UNFINISHED_ANSWER_ERROR` instead.

        A row that ends with such a reply, or an empty one, from a model that
        did think (#469) does not fail when the answer may be in the thinking
        (``AgentOptions.empty_reply_fallback``, on by default; off, the row
        ends as a failed run, as it did before). Where the provider has a
        switch for thinking (``thinking_off_body``),
        the reply that would end the row is followed by one more request sent
        with thinking off. If the row ends empty again, the last lines of the
        row's latest thinking (:func:`~localmelo.melo.schema.thinking_answer`)
        are the answer, ending with the note "(from melo's thinking)", and
        the run completes. A row with no thinking at all still fails. Each of
        the two steps is a notice (``thinking_off_retry``,
        ``answer_from_thinking``) and sets its flag in ``RunResult.empty_reply``.

        A reply with no tool call that writes one out as text
        (:func:`~localmelo.melo.schema.broken_tool_call`; a local model's
        server can leave ``text`` or ``<function_calls>`` in the content) is
        no answer either, and counts as an empty reply with a label, a nudge
        and a ``tool_call_as_text`` notice of its own. Its text is never the
        run's latest text, and a row that held one ends the run with the error
        :data:`~localmelo.melo.schema.TOOL_CALL_TEXT_ERROR` (unless the row
        also held unfinished thinking, whose error wins). The step's
        ``assistant.message`` event still has what the model sent, and the
        notice follows it even when this reply ends the run, so a display
        drops the text before ``run.finished``.
        """
        fingerprints: list[tuple[str, str, str]] = []
        fresh: list[bool] = []
        run.attempt_calls.clear()
        run.read_streak = 0

        # A step whose every call a policy check refused may not count
        # (refused_steps): the budget grows by one, so ``step`` still counts
        # the planning calls of this attempt.
        step = -1
        while step + 1 < budget:
            step += 1
            self.chat.check_call_budget()  # before the step starts, not inside it
            if (
                budget - step == EXTEND_WHEN
                and run.last_attempt
                and self.options.extend_steps > 0
                and run.extensions < self.options.max_extensions
                and self._progressing(fingerprints, fresh)
            ):
                # Still making progress near the end of the run's last steps:
                # more steps now, with the last ones still there to finish.
                run.extensions += 1
                budget += self.options.extend_steps
                logger.info(
                    "Task %s is making progress: %d more steps",
                    task.task_id,
                    self.options.extend_steps,
                )
            if budget - step == 2:
                await self._deadline_check(run)
                if self.options.task_aids and run.last_attempt:
                    self.hippo.working.append(Message(role="user", content=FINISH_NOTE))
            reflection = run.reflection_msg
            if reflection is not None and not self.hippo.working.holds(reflection):
                # An eviction took the latest reflection: it goes back once, at
                # the end, where the eviction has already changed the prefix.
                self.hippo.working.append(reflection)
            inserted, pending = self._read_inbox(run)
            # Retrieval + tool resolution + boundary checks
            short_window, tools, fail = await self._do_retrieval(query, run=run)
            if fail is not None:
                run.end("failed", error=fail)
                return step + 1, ""
            on_ready = None
            if inserted is not None:
                if not any(message is inserted for message in short_window):
                    run.end("failed", error="inbox instructions do not fit the prompt")
                    return step + 1, ""
                on_ready = self._accept_inbox(run, inserted, pending)

            # LLM planning step (with reflection context). The step counts
            # once the planning call is made, even if it then raises.
            run.steps += 1
            run.events.step_started(run.steps, run.attempts)
            if (stream_display := STREAM_DISPLAY.get()) is not None:
                stream_display.step(run.steps)
            run.note(lambda trace: trace.step(run.steps, run.attempts))
            if await self._checkpoint_step(run, query, tools):
                short_window = self.hippo.working.get_window()
            # The retry of a reply that was only thinking (#469) goes out with
            # the provider's thinking switched off, for this call only.
            retry_body = self._thinking_retry_body() if run.thinking_retry_armed else {}
            run.thinking_retry_armed = False
            retry_token = LLM_REQUEST_EXTRA_BODY.set(retry_body) if retry_body else None
            try:
                response, fail = await self._do_plan(
                    short_window,
                    tools,
                    query,
                    task_query=run.task_query_msg,
                    run=run,
                    on_ready=on_ready,
                )
            except PromptCapError as exc:
                # Over the prompt cap even fitted: nothing was sent.
                run.end("failed", error=f"Plan check failed: {exc}")
                return step + 1, ""
            finally:
                if retry_token is not None:
                    LLM_REQUEST_EXTRA_BODY.reset(retry_token)
            # A reply that is only unfinished thinking (#278) is thinking with
            # no answer: it is stored and shown so, and goes on as empty.
            unfinished = fail is None and bool(
                getattr(self.chat, "last_unfinished", False)
            )
            # What the provider reported as the model's reasoning, before an
            # unfinished reply's text is merged into it: only that is a model
            # reasoning its way to an answer (see :meth:`_answer_from_thinking`).
            reasoned = response.thinking
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
            response = self._answer_from_thinking(
                response, reasoned, run, can_retry=step + 1 < budget
            )
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
            # A tool call written as text is no answer, and never the latest
            # text: no fallback may show it as one.
            broken = not response.tool_calls and broken_tool_call(rest, tools)
            # Dots, dashes or emoji alone say nothing: the text is blank.
            answered = has_content(rest) and not broken
            if answered:
                run.last_text = rest
                run.last_text_thinking = response.thinking

            if not response.tool_calls:
                if answered:
                    ended = await self._direct_answer(
                        task, run, response, rest, can_retry=step + 1 < budget
                    )
                else:
                    ended = await self._empty_reply(
                        task,
                        run,
                        response,
                        broken=broken,
                        unfinished=unfinished,
                        thought_aliases=thought_aliases,
                    )
                if ended:
                    return step + 1, ""
                continue
            run.empty_replies = 0
            run.unfinished_row = False
            run.broken_row = False
            run.row_thinking = ""
            run.thinking_retry_used = False
            run.thinking_retry_armed = False

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
            refused_all = bool(calls)
            # A hand-off call runs only as the one call of its step: a step that
            # mixes it with others runs none of them (see ProvidedTool.hands_off).
            handing = [call for call in calls if self._hands_off(call)]
            mixed = bool(handing) and len(calls) > 1
            handed: list[str] = []
            handed_slot = CALL_HANDS_OFF.set(handed if handing and not mixed else None)
            over_again = False
            # The tools that were over before this step: the model was not offered
            # them, so a call to one ends the run with its answer. A call in the
            # step that made a tool over only does not run; the run goes on.
            over_before = frozenset(run.withdrawn)
            try:
                for call in calls:
                    # Counted when handed over, so a call that timeout_s cuts
                    # short counts too.
                    run.n_tool += 1
                    if mixed:
                        result, cut = self._not_run(call, run, HAND_OFF_ALONE), 0
                    elif call.tool_name in run.withdrawn:
                        result = self._not_run(
                            call,
                            run,
                            TOOL_OVER,
                            category=ErrorCategory.TOOL_NOT_OFFERED,
                        )
                        cut = 0
                        over_again = over_again or call.tool_name in over_before
                    else:
                        result, cut = await self._do_execute(
                            call, allowed=allowed, run=run
                        )
                    inflight.results.append(result)
                    inflight.cut_before.append(cut)
                    refused_all = refused_all and run.call_refused
            finally:
                CALL_HANDS_OFF.reset(handed_slot)
            await self._do_memorize(task.task_id, inflight, run)
            run.inflight = None
            if handed:
                # The rest of the work moved elsewhere: the run ends on its receipt.
                run.handed_off = True
                run.end("completed", result=handed[-1])
                return step + 1, ""
            if run.reviews.stop:
                # Auto mode blocked the same kind of step twice: the run
                # answers with the explanation instead of trying again.
                await self._end_with_answer(task, run)
                return step + 1, ""
            if over_again:
                # The model called a tool that took no more calls before this
                # step: its next request offers no tool and asks for the answer.
                await self._end_with_answer(
                    task,
                    run,
                    note=TOOL_OVER_ANSWER_NOTE,
                    fallback=TOOL_OVER_ANSWER,
                )
                return step + 1, ""

            if refused_all and run.free_steps < self.options.refused_steps:
                # Every call of the step was refused: it did nothing, so it
                # does not use up the run's steps (refused_steps).
                run.free_steps += 1
                budget += 1
            # Stuck detection (one fingerprint per step)
            fingerprints.append(self._make_fingerprint(calls, inflight.results))
            if self.options.extend_steps > 0:
                fresh.append(self._new_calls(calls, run))
            stuck = self._detect_stuck(fingerprints)
            if stuck:
                run.events.stuck()
                run.note(Trace.stuck)
                return step + 1, stuck
            if run.brief and step + 1 < budget:
                await self._push_after_reads(run)

        run.events.budget_used()
        return budget, "budget"

    def _read_inbox(
        self, run: _RunState
    ) -> tuple[Message | None, list[tuple[int, str]]]:
        """The notes the user sent during the run, read as one user message.

        The message goes into working memory and is pinned. The notes count
        as accepted only once a request carries them (:meth:`_accept_inbox`).
        Returns ``(None, [])`` when there are none.
        """
        pending = run.inbox.poll() if run.inbox is not None else []
        if not pending:
            return None, []
        assert run.inbox is not None
        pending, shown = _read_notes(run.inbox, pending)
        if run.from_person:
            pending = [(offset, self.hide_passwords(text)) for offset, text in pending]
        inserted = Message(
            role="user",
            content="\n\n".join(text for _, text in pending),
            images=self._tool_images.keep_user(shown),
        )
        self.hippo.working.append(inserted)
        run.pinned.append(inserted)
        for offset, text in pending:
            self._reviewer.user_message("" if _melos(run.inbox, offset) else text)
        return inserted, pending

    @staticmethod
    def _accept_inbox(
        run: _RunState, inserted: Message, pending: list[tuple[int, str]]
    ) -> Callable[[list[Message]], None]:
        """The ``on_ready`` hook that accepts *pending* once the request is built.

        It raises :class:`ContextFitError` when the request does not end with
        the inserted message.
        """

        def on_ready(messages: list[Message]) -> None:
            assert run.inbox is not None
            if (
                not messages
                or messages[-1].role != "user"
                or not messages[-1].content.endswith(inserted.content)
            ):
                raise ContextFitError("Inbox instructions do not fit the request")
            for offset, text in pending:
                by = "melo" if _melos(run.inbox, offset) else ""
                run.inbox.accept(offset)
                run.events.inbox_accepted(offset, text, run.steps, run.attempts, by)

        return on_ready

    def _answer_from_thinking(
        self, response: Message, reasoned: str, run: _RunState, *, can_retry: bool
    ) -> Message:
        """The reply, or the end of the row's thinking when that is the answer.

        A reply with thinking but no tool call and no words may have its
        answer in the thinking (#469). The row's last empty reply first asks
        once more with thinking off, where the provider has that switch (and
        *can_retry*, a step is left); if the row still ends empty, the end of
        the thinking is the answer, marked as such.
        """
        if (
            self.options.empty_reply_fallback != "on"
            or response.tool_calls
            or has_content(response.content)
        ):
            return response
        if reasoned.strip():
            run.row_thinking = reasoned
        if (
            run.empty_replies + 1 >= MAX_EMPTY_REPLIES
            and run.row_thinking.strip()
            and not run.broken_row
        ):
            if (
                not run.thinking_retry_used
                and can_retry
                and self._thinking_retry_body()
            ):
                run.thinking_retry_used = True
                run.thinking_retry_armed = True
                run.fallback_retried = True
            else:
                marked = thinking_answer(run.row_thinking)
                response = dataclasses.replace(
                    response, content=marked, thinking=run.row_thinking
                )
                if marked:  # thinking with no words gives no answer
                    run.fallback_answered = True
                    run.events.answer_from_thinking()
        return response

    async def _direct_answer(
        self,
        task: TaskRecord,
        run: _RunState,
        response: Message,
        rest: str,
        *,
        can_retry: bool,
    ) -> bool:
        """A reply with words and no tool call: the answer, unless a check objects.

        In the compress modes the note is stored first and the answer is the
        text without its ``<note>`` block (*rest*). Returns True when the run
        ended as ``completed``. When the brief's check shows the answer was no
        end, the model reads what the check showed and plans again: False.
        """
        if self._notes_mode:
            note, _ = self._reply_note(
                response, (), kind="answer", task_id=task.task_id, run=run
            )
            entry = self._store_note(note, run, source="reasoning_note")
            if entry is not None:
                await self._restate(note, run)
        run.answer_thinking = response.thinking
        retry = await self._brief_gate(run, can_retry=can_retry)
        if retry is None:
            run.end("completed", result=rest)
            return True
        answer = Message(role="assistant", content=rest)
        if self.options.reasoning not in ("legacy", "compress"):
            answer = dataclasses.replace(answer, thinking=response.thinking)
        self.hippo.working.append_turn(answer, [Message(role="user", content=retry)])
        return False

    async def _empty_reply(
        self,
        task: TaskRecord,
        run: _RunState,
        response: Message,
        *,
        broken: bool,
        unfinished: bool,
        thought_aliases: list[str],
    ) -> bool:
        """A reply with no tool call and no answer: ask for one and plan again.

        So is a tool call written as text (*broken*), with a label and nudge
        that say so; its text is no note either. Returns True when this reply
        ended the run as ``failed``.
        """
        label_text = TOOL_CALL_TEXT_LABEL if broken else EMPTY_REPLY_LABEL
        label_content = label_text
        entry = None
        if self._notes_mode:
            note, _ = self._reply_note(
                dataclasses.replace(response, content="") if broken else response,
                (),
                kind="empty",
                task_id=task.task_id,
                run=run,
            )
            entry = self._store_note(note, run, source="reasoning_note")
            if entry is not None:
                label_content = f"{label_text}\n{entry.text}"
        run.empty_replies += 1
        run.unfinished_row = run.unfinished_row or unfinished
        run.broken_row = run.broken_row or broken
        if broken:
            # Before the row's limit: the reply that ends the run is
            # dropped from the displays too, never shown as text.
            run.events.tool_call_as_text(
                run.empty_replies, final=run.empty_replies >= MAX_EMPTY_REPLIES
            )
        if run.empty_replies >= MAX_EMPTY_REPLIES and not run.thinking_retry_armed:
            run.end(
                "failed",
                error=(
                    UNFINISHED_ANSWER_ERROR
                    if run.unfinished_row
                    else (TOOL_CALL_TEXT_ERROR if run.broken_row else "empty replies")
                ),
            )
            return True
        # The label stands in for the reply, so the nudge does not
        # follow another user message; as one unit, the two are
        # only ever dropped together.
        label = Message(role="assistant", content=label_content)
        if self.options.reasoning not in ("legacy", "compress"):
            label = dataclasses.replace(label, thinking=response.thinking)
        nudge = TOOL_CALL_TEXT_NUDGE if broken else EMPTY_REPLY_NUDGE
        aliases: set[str] = set(thought_aliases)
        block, _hits = await self._step_recall(
            run,
            entry,
            (),
            fallback=label_text,
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
            label, [Message(role="user", content=nudge)], aliases=aliases
        )
        if unfinished:
            run.events.unfinished_answer()
        elif not broken:
            run.events.empty_reply(run.empty_replies)
        if run.thinking_retry_armed:
            # Armed by this reply: the next request is the retry.
            run.events.thinking_off_retry()
        return False

    # ── Main loop ──

    async def run(self, query: str) -> str:
        """Run one task and return its outcome as text.

        The text is the run's result when that is non-empty, otherwise
        ``"[<status>] <error>"``, so a run that failed never reads as an
        empty answer. :meth:`run_task` returns the whole record.
        """
        outcome = await self.run_task(query)
        return outcome.result or f"[{outcome.status}] {outcome.error}"

    @contextlib.contextmanager
    def _run_scope(self, run: _RunState, task_id: str) -> Iterator[None]:
        """Set the context variables that one run's requests and tools read.

        ``PREFIX_RUN`` (the date line of the fixed prefix), ``BUDGET_ACCOUNTING``
        and ``BUDGET_HOOK`` (the context budget), ``RETRY_HOOK`` (the retry
        notices, only with a sink), ``CALL_RECORDER`` and ``RECALL_RUN`` (the run
        a ``recall`` tool call belongs to). Leaving the scope, also when the run
        raises, resets them in the order ``PREFIX_RUN``, ``BUDGET_HOOK``,
        ``BUDGET_ACCOUNTING``, ``RETRY_HOOK``, ``CALL_RECORDER``, ``RECALL_RUN``.
        """
        prefix_run = self.chat.begin_prefix()
        budget_accounting = BUDGET_ACCOUNTING.set((self.chat, self.chat._accounting))
        budget_hook = BUDGET_HOOK.set(
            (self.chat, run.events.context_budget)
            if self.chat.context_budget is not None
            else None
        )
        # The retry notices of this run's model requests; only with a sink.
        retries = RETRY_HOOK.set(run.events.retry) if run.events.active else None
        # Every response of this run's model requests becomes a row of
        # history's llm_calls table, when the history keeps model calls,
        # with or without a sink. Read at each run, so a history set after
        # the Agent was built records too.
        history = self.hippo.history
        recorder = CALL_RECORDER.set(
            CallLog(history, task_id, run) if isinstance(history, CallHistory) else None
        )
        recall_run = RECALL_RUN.set(run)
        pause = PAUSE.set(run.hold)
        try:
            yield
        finally:
            PAUSE.reset(pause)
            PREFIX_RUN.reset(prefix_run)
            BUDGET_HOOK.reset(budget_hook)
            BUDGET_ACCOUNTING.reset(budget_accounting)
            if retries is not None:
                RETRY_HOOK.reset(retries)
            CALL_RECORDER.reset(recorder)
            RECALL_RUN.reset(recall_run)

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
        user_words: str | None = None,
        chat_started: float | None = None,
        from_person: bool = True,
        acts: bool = True,
        hide_tools: Collection[str] = (),
        earlier_turns: Sequence[tuple[str, str]] | None = None,
        images: Sequence[ImagePart] = (),
        chat_first: str | None = None,
        own_chars: int | None = None,
        recalled: str = "",
        file_handles: Collection[str] = (),
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
          that reflection was unreadable (see
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
        ``result or error``, and the reflections are cleared (an explicit
        reasoning mode first admits the run's candidates to long-term
        memory when ``memory`` is on); a failure while doing so is logged
        without a traceback and does not replace the run's own outcome.
        ``timeout_s`` bounds the run itself, not this final step, and the
        save has no bound of its own. A cancellation
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
        reflections are cleared, and no network call
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

        *user_words* are what the person typed for this run, when the query
        carries more (attachments, task news, a task's objective another
        model wrote): a step review reads only them as the user's words.
        Without it, the person's part of the query is (see *own_chars*).
        *own_chars* is how many leading characters of the query are the turn's
        own text as its caller made it: the person's message (words and
        attachments) for a chat's turn; for a task the request its code built
        (the objective, reading or card, and on a restart melo's check note
        before it). A caller that appended anything after it (recalled notes,
        other chats, context blocks, news) says so, and nothing is found by
        searching the text; a caller that gives none (a direct run, a worker)
        has all of the query counted as its own text. In a run the person
        began (*from_person*), passwords are hidden in that part, in
        *user_words*, and in all that follows the part except *recalled* when
        it stands right after the part; a password there gives a handle melo
        cannot type, because the step review takes the handles it accepts from
        the person's words only. *recalled* is the recalled-memory block the
        caller put in the query, as memory made it. A *recalled* that does not
        stand right after the person's part is not left out: all that follows
        is hidden, and the review's notes are *recalled* hidden the same way
        (fail closed). The notes a step review reads for the run's request
        are that string and nothing else, so in a run the person began a
        caller that gives none has none, whatever the query holds; a run the
        person did not begin hides nothing and leaves the review's notes as
        they were, those of the person's newest message.
        *chat_started* is when the chat began (wall-clock seconds, the
        session's stored start): a step review counts the files changed
        since then as written in this chat. With *from_person* false, the
        query is a request the person did not send (a task's question to its
        chat): a step review counts no new message of theirs, so a block or
        consent of their last message holds and the query's words allow
        nothing, and long-term memory takes no statement or remember request
        of the user's from it. With *acts* false nobody asked this run to act
        (a turn melo began in a chat): only tools that change nothing are
        offered (:func:`~localmelo.melo.checker.tiers.changes_nothing`), and the
        executor refuses any other call. *from_person* says whose words the
        query is, *acts* what the run may do: a task another agent made has no
        person's words and still does its job. *hide_tools* are not offered in
        this run.
        *earlier_turns* are the chat's newest turns before this one, each
        ``(what the user said, the answer)``, oldest first, from a caller that
        keeps the chat (a session's runner): the working window drops its
        oldest turns as tool output fills it, so the reflection, the answer on
        stop (``answer_on_stop``, and :meth:`closing_answer` after the run)
        and a planning request whose window has lost what it held before the
        query (a window that began the run empty is not) carry them as one
        data block (see
        :func:`~localmelo.melo.agent._earlier.earlier_block`). A short request
        such as "go look it up" then stays on what the chat is about. When
        the step review's window starts over (a reload), the newest of their
        user words go into it before the query's, so the person's earlier
        words still count (#465); their answers never do. ``None`` or none
        adds nothing. *chat_first* is what the person typed in the chat's
        first turn (a session's runner reads it from its store): when the
        window starts over, the step review keeps it first, as the task the
        chat began with, so a goal review still has it after a reload.

        *file_handles* are the handles the door that read the person's message
        made from values in its attached files (``hide_message``'s
        ``file_handles``), given by the caller as data: a file's text is never
        read for one. The step review lets melo type the one password they name
        when the person's words give none of their own and the set holds exactly
        one the vault still keeps (:meth:`StepReviewer.user_message`). They count
        only in a run the person began: with *from_person* false they are
        ignored, as is every note read mid-run (#626).

        *images* are pictures the person added to the query, with their data.
        The query message keeps reference copies, as a tool result does, and the
        store keeps the payloads of the newest few (see
        :meth:`~localmelo.melo.agent._tool_images.ToolImageStore.keep_user`), so
        each planning request carries them; once a payload is gone, the model
        reads a line that names the picture instead. ``ValueError`` before the
        run for a picture the checker would refuse (see
        :func:`~localmelo.melo.checker.validators.image_problem`), or more than
        :data:`~localmelo.melo.checker.validators.MAX_IMAGES_PER_RESULT`.

        Before the run starts, ``TypeError`` is raised when *on_event* or
        *observe* is not callable or is a coroutine function, when both
        are given, when *approval* has no ``async def decide``, when
        *earlier_turns* is not a sequence of pairs of ``str``, when
        *query*, *chat_first*, *routine* or *recalled* is not a ``str``, or
        *own_chars* is neither ``None`` nor an ``int`` (a ``bool`` is not one),
        and ``ValueError`` when *routine* is blank or *own_chars* is negative:
        no event is emitted, no task record is written, and working memory is
        left as it is.
        """
        check_sink(on_event)
        check_sink(observe, "observe")
        if on_event is not None and observe is not None:
            raise TypeError("pass on_event or observe, not both")
        _check_approval(approval)
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        if not isinstance(recalled, str):
            raise TypeError("recalled must be a str")
        if own_chars is not None:
            if isinstance(own_chars, bool) or not isinstance(own_chars, int):
                raise TypeError("own_chars must be an int")
            if own_chars < 0:
                raise ValueError("own_chars must not be negative")
        # The person's part of the query: all of it unless the caller says more
        # was appended. Its position comes from the caller, never from the text.
        own = query[:own_chars]
        if from_person:
            # A password the person gave is cut out before any model, record,
            # log or memory sees the message (#547): a handle stands for it.
            # In reading order: the person's part, then the block the caller
            # named in `recalled` as memory made it, only when it stands right
            # after that part (judged on the text as it came), then all the
            # rest. What follows the person's part is melo's, but it may carry
            # text a password could hide in (a task's context blocks, a note), so
            # it is hidden too; no handle made there can be typed, since
            # `StepReviewer.user_message` takes the live handles from the
            # person's words only.
            rest = query[len(own) :]
            kept = recalled if rest.startswith(recalled) else ""
            own = self.hide_passwords(own)
            query = own + kept + self.hide_passwords(rest[len(kept) :])
            if not kept:
                recalled = self.hide_passwords(recalled)  # the notes follow suit
            if isinstance(user_words, str):
                user_words = self.hide_passwords(user_words)
        # A file's text is data, never the person's words (#626): they end where
        # a file's block starts, here and in the step review.
        person_words = words_before_files(own if user_words is None else user_words)
        if routine is not None:
            if not isinstance(routine, str):
                raise TypeError("routine must be a str")
            if not routine.strip():
                raise ValueError("routine must not be blank")
        if chat_first is not None and not isinstance(chat_first, str):
            raise TypeError("chat_first must be a str")
        turns_before = (
            check_earlier_turns(earlier_turns) if earlier_turns is not None else []
        )
        pictures = list(images)
        if len(pictures) > MAX_IMAGES_PER_RESULT:
            raise ValueError(f"at most {MAX_IMAGES_PER_RESULT} pictures per message")
        for picture in pictures:
            if problem := image_problem(picture):
                raise ValueError(f"picture refused: {problem}")
        earlier = earlier_block(turns_before)
        place = self._run_place()
        started = time.perf_counter()
        task = TaskRecord(query=query)
        run = _RunState(
            task_id=task.task_id,
            task_text=query,
            words=person_words,
            events=_Emitter(
                on_event if observe is None else observe,
                task.task_id,
                interactive=observe is None,
                progress=bool(progress),
            ),
            approval=approval,
            from_person=from_person,
            acts=acts,
            routine=routine,
            inbox=inbox,
            brief=self.options.brief_probe is not None,
            place=place,
            limits=self._limits() if progress else None,
            refused_writes_base=(
                0 if self._confinement is None else self._confinement.refused
            ),
            hidden=frozenset(hide_tools),
            earlier_block=earlier,
        )
        self._earlier_block = earlier
        # The user's own words, which a step review reads (never tool output).
        # A window that starts over takes the person's earlier words first,
        # never the answers (#465).
        self._reviewer.earlier_words(
            [words for words, _ in turns_before], first=chat_first or ""
        )
        # No password is typed in a run the person did not begin (#547).
        self._reviewer.begin_run()
        if from_person:
            # The notes recalled for this message go along, apart: they count
            # only where the person's words point to them (#465).
            self._reviewer.user_message(
                person_words, recalled=recalled, files=file_handles
            )
        if chat_started is not None:
            self._reviewer.since = chat_started
        await self._ensure_budget()
        stats = self.chat.reset_stats()
        timeout_s = self.options.timeout_s
        deadline = asyncio.timeout(timeout_s)  # never expires when None
        run.hold.deadline = deadline
        with self._run_scope(run, task.task_id):
            self._active_runs += 1
            try:
                run.events.run_started(
                    query,
                    max_steps=self.options.max_steps,
                    workspace_root=self.executor.workspace_root,
                    approval=approval is not None,
                )
                # What leaves the window is kept as a cue (and a brief run's note).
                self.hippo.working.on_drop = lambda gone: self._note_dropped(run, gone)
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
                            raise _ToolSetupError(
                                str(exc) or exception_text(exc)
                            ) from exc
                        query_msg = Message(
                            role="user",
                            content=query,
                            images=self._tool_images.keep_user(pictures),
                        )
                        run.window_had_turns = any(
                            msg.role != "system"
                            for msg in self.hippo.working.get_window()
                        )
                        self.hippo.working.append(query_msg)
                        run.task_query_msg = query_msg
                        run.query_in_memory = True
                        if self.options.reasoning != "legacy" and from_person:
                            # Only the user's own text gives a user signal: a
                            # direct statement is admitted at once, while the
                            # fact a remember request carries is staged and
                            # admitted with the digest at flush. A query the
                            # person did not write gives none. The person's own
                            # words, never what a model wrote about them (a task's
                            # request carries its chat's reading after the words).
                            signals = classify_user_text(run.words)
                            for candidate in user_candidates(
                                run.words,
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
                                decisions = await self.hippo.admit(
                                    candidate.candidate_id, "user_statement"
                                )
                                self._drain_decisions(decisions, run)
                            for candidate in request_candidates(
                                run.words,
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
                            await self._restore_closing_line(task, run)
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
                    logger.info(
                        "Task %s ended with provider_error: %s", task.task_id, exc
                    )
                    error = str(exc) or exception_text(exc)
                    if self._refused_pictures(exc):
                        error = PICTURES_REFUSED_ERROR
                    run.end("provider_error", error=error)
                except Exception as exc:
                    if (
                        isinstance(exc, TimeoutError)
                        and timeout_s is not None
                        and deadline.expired()
                    ):
                        # This run's own deadline; any other TimeoutError is a crash.
                        logger.info(
                            "Task %s timed out after %g s", task.task_id, timeout_s
                        )
                        run.end("timeout", error=f"Timed out after {timeout_s:g} s")
                    else:
                        logger.exception("Task %s crashed", task.task_id)
                        run.end("crashed", error=exception_text(exc))
                except BaseException as exc:
                    if run.interactive and is_cancellation(exc):
                        # A caller stopped a run that it follows.
                        await self._on_cancelled(query, run, exc)
                    else:
                        if (
                            isinstance(exc, asyncio.CancelledError)
                            and self._exit is not None
                        ):
                            error = self._exit.reason  # cancelled for an exit
                        else:
                            error = exception_text(exc)
                        run.end("crashed", error=error)
                    raise
                finally:
                    if run.running:
                        # Every path above ends the run; this keeps the record
                        # terminal should a later change miss one.
                        run.end("crashed", error="The run ended without a status")
                    # A hold still on (the person had a page) ends with the run.
                    run.hold.end("stopped" if run.status == "cancelled" else "ended")
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
                                    await self._finish_task_for_exit(
                                        task, run, self._exit
                                    )
                                    raise
                        finally:
                            # Built here, so that run.finished carries it on every
                            # path, also when run_task raises.
                            outcome = self._run_result(
                                task, run, stats, time.perf_counter() - started
                            )
                            run.outcome = outcome
                            run.report_progress(ended=run.status)
                            run.events.run_finished(outcome, _reviews_of(run))
                await self._learn(run, query, outcome)
                return outcome
            finally:
                self.hippo.working.on_drop = None
                if run.finalizer is None:
                    # Else the finalizer counts the run out when it ends.
                    self._active_runs -= 1

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
                reason = exception_text(exc)
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
                reason = exception_text(exc)
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
        traceback: the run did not crash because of it.
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
                    exception_text(exc),
                )
            if self.options.reasoning != "legacy":
                # R4: the explicit modes write nothing unchecked to long-term
                # memory. R6-lite's flush admits the run's candidates through
                # the boundary and policy; its failure is logged, never raised.
                decisions = await self.hippo.flush(
                    run_request=run.remember_request,
                    digest=run.final_digest,
                    final_reflection=run.final_reflection,
                    task_id=run.task_id,
                )
                self._drain_decisions(decisions, run)
                self.hippo.working.clear_reflections()
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
        reflections are cleared and nothing is flushed to long-term memory: a
        flush calls the embedding server and may take longer than the exit
        allows. A failure is logged, not raised, and a cancellation during the
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
                    reason = exception_text(exc)
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

    def _refused_pictures(self, exc: ProviderError) -> bool:
        """Whether *exc* is the server refusing the pictures its request carried.

        True only for a request that carried pixels, when the error says the model
        cannot take pictures (see ``_picture_refusal.names_pictures``). A bare HTTP 400,
        an error about the length of the request or about another parameter, and an
        outage are not it, so they keep their own text.
        """
        return self._tool_images.sent_pixels and names_pictures(exc)

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
        4. The reflections are cleared: no long-term write, and no
           embedding or other network call.
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
            run.events.run_finished(run.outcome, _reviews_of(run))
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
                    exception_text(exc),
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
                    exception_text(exc),
                )

    async def _save_final_state(self, task: TaskRecord) -> None:
        """Save *task*; a failure is logged without a traceback, not raised."""
        try:
            await self.hippo.save_task(task)
        except Exception as exc:
            logger.warning(
                "Could not save the final state of task %s: %s",
                task.task_id,
                exception_text(exc),
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
                "replay_stripped": stats.thinking_cleared,
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
            prefix_cache={
                "version": PREFIX_VERSION,
                "schema_changes": stats.schema_changes,
                **(
                    {"reported_cached_tokens": stats.prefix_cached_tokens}
                    if stats.prefix_cached_tokens is not None
                    else {}
                ),
            },
            confinement=self._confinement_record(run),
            answered_on_stop=run.answered_on_stop,
            format_fixed=run.format_fixed,
            handed_off=run.handed_off,
            checkpoints=run.checkpoints,
            empty_reply=(
                {
                    "thinking_off_retry": run.fallback_retried,
                    "from_thinking": run.fallback_answered,
                }
                if run.fallback_retried or run.fallback_answered
                else None
            ),
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
        if opts.empty_reply_fallback != "on":
            config["empty_reply_fallback"] = opts.empty_reply_fallback
        if opts.refused_steps:
            config["refused_steps"] = opts.refused_steps
        if opts.max_calls is not None:
            config["max_calls"] = opts.max_calls
        config["window_eviction"] = opts.window_eviction
        if self.window_messages != SHORT_TERM_MAX:
            config["window_messages"] = self.window_messages
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
            run.end("provider_error", error=str(exc) or exception_text(exc))

    def _keep_best_effort(self, run: _RunState) -> None:
        """A run that stopped on ``max_calls`` or a refusal, with no answer: its latest text."""
        if not run.early_stop or run.result.strip():
            return
        run.result = run.last_text
        run.answer_thinking = run.last_text_thinking

    def _normalize_reply(self, response: Message) -> Message:
        """*response* with an inline ``<think>`` block moved into its thinking.

        Only in the explicit reasoning modes (see :func:`_split_inline_reasoning`).
        Thinking found in the content, and not reported by the provider, counts
        in the run's ``reasoning_chars``.
        """
        if self.options.reasoning == "legacy":
            return response
        before = response.thinking
        response = _split_inline_reasoning(response)
        if not before and response.thinking:
            self.chat.stats.reasoning_chars += len(response.thinking)
        return response

    async def _aux_reply(
        self, request: Awaitable[Message], what: str, task_id: str
    ) -> Message | None:
        """The reply to an auxiliary request, with its ``<think>`` block split off.

        ``None`` when the request failed: ``max_calls`` left no call for it, or
        the server did not answer. That is logged, with *what* the request was
        for, and never ends the run.
        """
        try:
            reply = await request
        except CallBudgetError as exc:
            logger.info("No call left to ask for %s in task %s: %s", what, task_id, exc)
            return None
        except Exception as exc:
            logger.warning(
                "Could not ask for %s in task %s: %s",
                what,
                task_id,
                exception_text(exc),
            )
            return None
        return _split_inline_reasoning(reply)

    async def _end_with_answer(
        self,
        task: TaskRecord,
        run: _RunState,
        *,
        note: str = BLOCKED_ANSWER_NOTE,
        fallback: str = "",
    ) -> None:
        """End a run that cannot go on, its answer one request without tools (#339).

        The request is :meth:`Chat.final_answer` with *note*
        (:data:`BLOCKED_ANSWER_NOTE`, for a run auto mode stopped; the
        default) over the working window, whose last tool message says why the
        run stops. A request that fails, or a blank reply, leaves *fallback*,
        or, for a block, a fixed answer that names the blocked step. The run
        ends ``completed``.
        """
        window = list(self.hippo.working.get_window())
        answer, thinking = "", ""
        reply = await self._aux_reply(
            self.chat.final_answer(
                window,
                task.query,
                None,
                note=note,
                earlier=run.earlier_block,
            ),
            "the answer of a run that cannot go on",
            task.task_id,
        )
        if reply is not None:
            answer, thinking = reply.content.strip(), reply.thinking
            if broken_tool_call(answer):  # no tools were offered: not an answer
                answer, thinking = "", ""
        if not answer:
            blocked = run.reviews.blocked
            answer = fallback or answer_on_block(
                blocked[-1]["title"] if blocked else ""
            )
        run.answer_thinking = thinking
        run.end("completed", result=answer)

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
        reply = await self._aux_reply(
            self.chat.final_answer(
                window,
                task.query,
                self.hippo.working.get_reflections(),
                earlier=run.earlier_block,
            ),
            "an answer on stop",
            task.task_id,
        )
        if reply is None:
            return
        answer = _answer_of(reply)
        if not answer:
            return
        run.result = answer
        run.answered_on_stop = True
        run.answer_thinking = reply.thinking

    async def closing_answer(self, query: str, *, note: str = "") -> str:
        """After a run: one request, without tools, for the answer to *query* from what it found.

        For a caller whose run ended with no answer it can use (a reply that
        only said what it would do, a run that got stuck): the request is
        the one ``answer_on_stop`` makes (the working window and the latest
        reflection), with *note* in place of its closing instruction when
        given. Returns the reply without its thinking, or ``""`` when it is
        blank or a tool call written as text. A failed request raises.
        """
        prefix_run = self.chat.begin_prefix()
        try:
            reply = await self.chat.final_answer(
                self.hippo.working.get_window(),
                query,
                self.hippo.working.get_reflections(),
                note=note,
                earlier=self._earlier_block,
            )
        finally:
            PREFIX_RUN.reset(prefix_run)
        return _answer_of(_split_inline_reasoning(reply))

    async def _restore_closing_line(self, task: TaskRecord, run: _RunState) -> None:
        """``answer_on_stop``: add the closing line the task asks for, when the answer lacks it.

        For a run that ended ``completed``, ``gate_stopped`` or
        ``budget_exhausted`` with an answer. The line is detected in the person's
        own words (:func:`~localmelo.melo.agent._answer_format.requested_closing_label`);
        a task that names none is left as it is. One request, without tools
        (:meth:`Chat.closing_line`), gives the line from the answer alone, and
        it is added after the answer, which stays as it was.
        ``RunResult.format_fixed`` is then true. A request that fails or that
        ``max_calls`` leaves no call for, or a reply with no such line, leaves
        the run as it was, and so does a run that handed its work on: its
        result is the receipt of that hand-off, not an answer.
        """
        if run.handed_off or run.status not in (
            "completed",
            "gate_stopped",
            "budget_exhausted",
        ):
            return
        _, answer = split_thinking(run.result)
        label = requested_closing_label(run.words) if answer else None
        if label is None or has_closing_line(answer, label):
            return
        reply = await self._aux_reply(
            self.chat.closing_line(task.query, answer, label),
            "the closing line",
            task.task_id,
        )
        if reply is None:
            return
        line = closing_line_of(reply.content, label)
        if line is None:
            return
        run.result = f"{run.result.rstrip()}\n{line}"
        run.format_fixed = True

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
        - Otherwise the continuation gate decides: the next attempt starts
          when the decision is ``continue`` and its checks hold
          (``task_still_feasible`` and ``next_step_is_concrete``; and
          ``next_step_is_novel``, except that the run lets one ``continue``
          that is not novel through). A gate that stops the run ends it as
          ``gate_stopped``, with the ``best_effort_result`` as the result and
          the decision's rationale as the error, or
          ``Stopped after attempt <n> (stop reason: <code>)`` (``n``
          counting from 1; the code is the check that stopped the run, see
          :func:`_gate_stop_reason`) when the rationale is blank. The
          ``reflection`` event carries the code in every case.

        A reply that holds no reflection even when asked again (see
        :meth:`_do_reflect`) gives no decision and adds no reflection: the
        previous one, if any, stays in working memory, and nothing of the
        reply reaches long-term memory. The next attempt starts; when none
        can start, the run ends as ``budget_exhausted`` with the boundary's
        usual error, ``Max attempts reached`` or ``Max steps reached``, and
        the latest non-blank text of a planning reply in this run as the
        result. One INFO line names the attempt.

        A reflection over :data:`MAX_REFLECTION_CHARS` keeps its decision and
        is trimmed to fit before it is kept (see :meth:`_do_reflect`).
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
            # None for a reply with no reflection.
            patient = not run.patience_spent
            stop_reason = _stop_reason(
                decision, out_of, brief=run.brief, patient=patient
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
            if reflection is None:
                run.events.reflection(
                    None, continue_run=continue_run, stop_reason=stop_reason
                )
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
                        reflection, continue_run=True, stop_reason=None
                    )
                    self.hippo.working.append(Message(role="user", content=retry))
                    continue
                run.events.reflection(
                    reflection, continue_run=False, stop_reason=stop_reason
                )
                if best_effort.strip():
                    answer = best_effort
                    run.answer_thinking = run.reflection_thinking
                else:
                    answer = run.last_text
                    run.answer_thinking = run.last_text_thinking
                run.end("completed", result=answer)
                return
            if out_of:
                run.events.reflection(
                    reflection, continue_run=False, stop_reason=stop_reason
                )
                run.answer_thinking = run.reflection_thinking
                run.end("budget_exhausted", result=best_effort, error=out_of)
                return
            if not continue_run:
                run.events.reflection(
                    reflection, continue_run=False, stop_reason=stop_reason
                )
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
            run.events.reflection(
                reflection, continue_run=True, stop_reason=stop_reason
            )
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

    def _note_dropped(self, run: _RunState, gone: list[Message]) -> None:
        """Keep a cue for each tool result that left the window, and a brief run's line."""
        self._note_cues(run, gone)
        if run.brief:
            run.dropped = [*run.dropped, *dropped_notes(gone)][-DROPPED_KEPT:]

    async def _deadline_check(self, run: _RunState) -> None:
        """With two steps left in a brief run's last attempt, run the checks once.

        The checks run only when the workspace changed since the checks last ran and the
        gate has not run in this attempt. A pass changes nothing; a failure
        adds the gate's failure message to the window, so the next planning
        request carries it, and uses one of the gate's retries.
        """
        if not run.brief or not run.last_attempt:
            return
        lines = []
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
        cleared. A provider that the cancellation
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
