from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from localmelo.melo.agent._chat import Chat, RequestStats
from localmelo.melo.agent.options import AgentOptions
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import (
    ExecutorResultPayload,
    MemoryWritePayload,
    ToolResolutionResult,
)
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import Executor, builtin_names, register_builtins
from localmelo.melo.executor.models import ExecutionRequest
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.short import group_turns
from localmelo.melo.schema import (
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
    ToolResult,
    describe_images,
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


# The decisions a reflection may recommend; any other value reads "stop".
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


def _decode_reflection(text: str) -> object:
    """Return the JSON value of a reflection reply; ``ValueError`` if none.

    A Markdown code fence around the reply is stripped first. When the
    rest is not JSON as a whole, the first balanced ``{...}`` in it that
    parses as a JSON object is used, so text around the object does not
    matter. After a ``{`` that does not start one, the search goes on
    from where the decoder gave up, so it reads the reply about once
    and never starts inside a string of the text it already read.
    Nesting deeper than the decoder's recursion limit ends the search.
    """
    body = _strip_code_fences(text)
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        pass
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            return decoder.raw_decode(body, start)[0]
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
        except RecursionError:
            break
    raise ValueError("No JSON object in the reflection response")


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


def _cut_to(text: str, room: int) -> str:
    """*text*, or its start and a truncation marker, in at most *room* chars.

    The marker says how many characters of *text* were left out (see
    :func:`_truncation_marker`). Every caller leaves at least
    ``WITHHELD_PREVIEW_CHARS`` of room, which is ample for the marker.
    """
    if len(text) <= room:
        return text
    # The marker is never longer than this one, whose count is the
    # largest possible, so the cut text never passes *room*.
    keep = max(room - len(_truncation_marker(len(text))), 0)
    return text[:keep] + _truncation_marker(len(text) - keep)


def _within_turn_budget(
    texts: Sequence[str], withheld: set[int], calls_chars: int
) -> list[str]:
    """Return *texts* with the largest cut, or replaced, until they fit.

    The rendered tool calls of one turn (*calls_chars* characters) and its
    tool results share ``MAX_MEMORY_TEXT_LEN`` characters: the calls keep
    their share and the results get the rest. While the results do not
    fit, the largest is cut to the room the others leave it, with a
    marker that says how much was left out (see :func:`_cut_to`). A
    result left less than ``WITHHELD_PREVIEW_CHARS`` of room is replaced
    by the withheld notice instead, and its index added to *withheld*.
    Indices already in *withheld* hold notices and are left alone.
    """
    fitted = list(texts)
    budget = MAX_MEMORY_TEXT_LEN - calls_chars
    total = sum(len(text) for text in fitted)
    if total <= budget:
        return fitted
    reason = (
        f"Tool calls and results too large: {calls_chars + total} chars "
        f"(max {MAX_MEMORY_TEXT_LEN})"
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
            kept = _cut_to(text, room)
        else:
            kept = _withheld_notice(text, reason)
            if len(kept) >= len(text):
                break  # only results a notice would not shorten are left
            withheld.add(index)
        total += len(kept) - len(text)
        fitted[index] = kept
    return fitted


# Working-memory message that answers a planning reply with neither a
# tool call nor text: such a reply is no final answer.
EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Continue the task: call a tool or give your "
    "final answer."
)

# Empty replies in a row that end the run as failed ("empty replies").
MAX_EMPTY_REPLIES = 2

CONTINUATION_UTILITY_THRESHOLD = 0.1


def _should_continue(decision: ReflectionDecision) -> bool:
    """Active-learning style continuation gate.

    Computes a lightweight utility score and checks structured signals.
    Returns True only if all conditions suggest another attempt is worthwhile.
    """
    if decision.recommended_action != "continue":
        return False
    if not decision.task_still_feasible:
        return False
    if not decision.next_step_is_concrete:
        return False
    if not decision.next_step_is_novel:
        return False

    utility = (
        decision.estimated_info_gain * decision.feasibility * decision.novelty
        - decision.estimated_cost
        - decision.repeat_risk
    )
    return utility >= CONTINUATION_UTILITY_THRESHOLD


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
    # answer of a reflection's "done" that gives none.
    last_text: str = ""
    # Planning replies in a row with neither a tool call nor text.
    empty_replies: int = 0

    @property
    def running(self) -> bool:
        return self.status == "running"

    def end(self, status: str, *, result: str = "", error: str = "") -> None:
        self.status = status
        self.result = result
        self.error = error


def _exception_text(exc: BaseException) -> str:
    """``"<type>: <message>"``, or the type alone when the message is empty."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


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
    the attempt policy, tool exposure, the built-in tools, memory and a
    run timeout; the defaults keep the behavior from before options
    existed, except that a step now runs only the tools it offered.
    ``tool_providers`` supply further tools, for example those of MCP
    servers: the first run starts them and registers their tools next to
    the built-ins (see :meth:`_ensure_tools`), and :meth:`close` closes
    them.

    :meth:`run_task` runs one task and returns a
    :class:`~localmelo.melo.schema.RunResult`; :meth:`run` returns its
    outcome as text. Runs on one agent share its working memory and
    request counters, so they must not overlap.
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
        self._tool_providers: tuple[BaseToolProvider, ...] = tuple(tool_providers)
        # Tool setup (_ensure_tools) runs once, under this lock. It ends
        # either ready or failed; a failure is kept as its error text.
        self._tools_lock = asyncio.Lock()
        self._tools_ready = False
        self._tools_failure: str | None = None
        # close() has begun: no tool provider is started from then on.
        self._closing = False

        # Optional persistent memory backends: options.memory_dir, or the
        # LOCALMELO_PERSIST_MEMORY / LOCALMELO_MEMORY_DIR opt-in. History is
        # SQLite whenever persistence is on; long-term only when memory is
        # on and an embedding provider exists.
        self._memory_dir = _resolve_memory_dir(self.options)
        history_backend = None
        long_backend = None
        if self._memory_dir is not None:
            from localmelo.melo.memory.history.sqlite import SqliteHistory
            from localmelo.melo.memory.long.sqlite import SqliteLongTerm

            os.makedirs(self._memory_dir, exist_ok=True)
            history_backend = SqliteHistory(
                os.path.join(self._memory_dir, "history.db")
            )
            if self._embedding and self.options.memory:
                long_backend = SqliteLongTerm(
                    os.path.join(self._memory_dir, "long_term.db")
                )

        # memory=False runs Hippo in its no-embedding mode, which is exactly
        # the documented behavior of that option: no long-term recall, no
        # long-term writes, no reflection promotion, and reflections are
        # still cleared. self._embedding stays set so close() closes it.
        self.hippo = Hippo(
            embedding=self._embedding if self.options.memory else None,
            history=history_backend,
            long=long_backend,
        )
        self.checker = Checker()
        self.executor = Executor(self.hippo, self.checker)
        self.chat = Chat(self._llm)

        register_builtins(self.executor, self.hippo, names=self.options.builtins)

    # ── Tool providers ──

    async def _ensure_tools(self) -> None:
        """Start the tool providers and register their tools, once.

        The first call starts every provider in order and collects the
        tools they return. Only when all of them have started, and no
        provided tool's name is taken by a registered tool or by another
        provided tool, are the tools registered with the tool registry and
        the executor (with each tool's ``timeout_ms``), so a failed setup
        registers nothing. A taken name raises ``ValueError``; a failure
        of a provider's ``start()`` propagates unchanged. On any failure,
        cancellation included, every provider whose ``start()`` was called
        is closed before the exception propagates, so none is left
        running. A cancellation that arrives during this clean-up waits
        for it to finish and then propagates in place of the failure (see
        :meth:`_close_providers`).

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
                    provided.extend(await provider.start())
                # close() may have begun while the last provider started.
                self._check_not_closing()
                self._check_tool_names(provided)
            except BaseException as exc:
                self._tools_failure = str(exc) or _exception_text(exc)
                await self._close_providers(started)
                raise
            for tool in provided:
                self.hippo.register_tool(tool.definition)
                self.executor.register(
                    tool.definition.name, tool.function, timeout_ms=tool.timeout_ms
                )
            self._tools_ready = True

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
        """
        cancelled: asyncio.CancelledError | None = None
        for provider in providers:
            closing = asyncio.create_task(self._close_provider(provider))
            while not closing.done():
                try:
                    # wait() neither cancels the task nor raises its outcome.
                    await asyncio.wait([closing])
                except asyncio.CancelledError as exc:
                    cancelled = exc
        if cancelled is not None:
            raise cancelled

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

    async def _fit_prompt_cap(self, others: list[Message]) -> None:
        """Drop the oldest working memory that the planning prompt has no room for.

        ``Checker.pre_plan`` refuses a step whose prompt, the working-memory
        window plus *others* (long-term recall and the reflection), is
        over its cap. Units of the window (see :func:`group_turns`) are
        dropped oldest first and whole, so a tool call keeps its results,
        until the prompt fits; the newest unit always stays. When even the
        newest unit alone does not fit, nothing is dropped and pre_plan
        refuses the step as before.

        Each turn fits half the cap (see :meth:`_checked_tool_turn`), but
        several large turns, of one run or of earlier runs on this agent,
        can pass it together. Without this, the task would fail with
        "Prompt too large", and so would each later run on this agent,
        until eviction by message count caught up.
        """
        units = group_turns(self.hippo.working.get_window())
        for first in range(len(units)):
            kept = [msg for unit in units[first:] for msg in unit]
            if (await self.checker.pre_plan(others + kept)).allowed:
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

        With ``tool_exposure="retrieve"`` the tools come from hint
        extraction plus BM25; with ``"all"`` they are every registered
        tool, sorted by name.

        Before anything reads the working-memory window, its oldest units
        are dropped when the planning prompt would not fit
        ``Checker.pre_plan``'s cap otherwise (see :meth:`_fit_prompt_cap`).

        Returns (long_context, short_window, tools, fail_reason).
        *fail_reason* is non-None when a check fails and the loop must break.
        """
        long_context = await self.hippo.retrieve_context(query)
        reflection_msgs = _serialize_reflections(self.hippo.working.get_reflections())
        await self._fit_prompt_cap(long_context + reflection_msgs)
        short_window = self.hippo.working.get_window()

        tool_hints: list[str] = []
        if self.options.tool_exposure == "all":
            # The same list on every step, so the tools part of the request
            # stays byte-stable for prefix caching.
            tools = sorted(self.hippo.list_tools(), key=lambda t: t.name)
        else:
            # Include reflection context in tool hint extraction
            hint_sources = long_context + short_window + reflection_msgs
            tool_hints = self.hippo.extract_tool_hints(hint_sources)
            tools = self.hippo.resolve_tools(query, hints=tool_hints)

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

        all_msgs = long_context + short_window + reflection_msgs
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
        self, call: ToolCall, *, allowed: Collection[str]
    ) -> ToolResult:
        """Stage 4: execute one tool call and validate the result.

        *allowed* holds the names of the tools offered in the planning
        call that produced *call*; the executor refuses any other tool.

        Uses the sanitized payload from the executor-result checker
        when truncation or modification was applied; falls back to
        the raw outcome otherwise.
        """
        exec_request = ExecutionRequest(
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
        )
        outcome = await self.executor.execute_structured(exec_request, allowed=allowed)

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
            return ToolResult(
                tool_name=sp.tool_name,
                output=sp.output,
                error=sp.error,
                duration_ms=sp.duration_ms,
                images=list(sp.images),
            )
        return outcome.to_tool_result()

    async def _do_memorize(
        self, task_id: str, response: Message, results: Sequence[ToolResult]
    ) -> None:
        """Stage 5: record the step and write checked results to memory.

        - History gets one StepRecord per executed call (the assistant
          text is attached to the first), unconditionally.
        - Long-term memory gets each record's text summary, if the checker
          allows the write.
        - Working memory always gets the real protocol turn: the assistant
          message with its tool calls, then one ``role="tool"`` message per
          call paired by ``tool_call_id``, appended as one unit. Checked
          writes may shorten it but never leave a call or result out (see
          :meth:`_checked_tool_turn`).

        History and working memory get the same copies of *results*, whose
        images are references without the base64 payload (see
        :func:`_image_references`), so no payload outlives the step.
        """
        calls = response.tool_calls
        kept = [_image_references(result) for result in results]
        for index, (call, result) in enumerate(zip(calls, kept, strict=True)):
            step = StepRecord(
                thought=response.content if index == 0 else "",
                tool_call=call,
                tool_result=result,
            )
            summary = await self.hippo.store_step(task_id, step)

            # Checked write: step summary -> long-term memory only
            mem_check = self.checker.check_memory_write(
                MemoryWritePayload(
                    text=summary,
                    role="assistant",
                    metadata={"step_id": step.step_id},
                )
            )
            if mem_check.allowed:
                await self.hippo.memorize(
                    summary, metadata={"step_id": step.step_id}, to_working=False
                )

        assistant_msg, tool_msgs = self._checked_tool_turn(response, kept)
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
        self, response: Message, results: Sequence[ToolResult]
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
        prompt. The calls keep their share; the results share the rest,
        the largest cut first to the room the others leave it, ending
        with a marker that says how many characters were left out (see
        :func:`_within_turn_budget`). This is also how a result longer
        than one memory write allows is shortened. A result left less
        than ``WITHHELD_PREVIEW_CHARS`` of room is replaced by the notice
        instead. The assistant text, which history keeps, is dropped when
        it does not fit beside them. Notices, and results shorter than
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

        texts: list[str] = []
        withheld: set[int] = set()
        for index, result in enumerate(results):
            text = _tool_result_text(result)
            # A text too long for one write is checked as the budget below
            # keeps it: cut. A memory write needs text: an image-only
            # result is checked with the line that names its images.
            checked = _cut_to(text, MAX_MEMORY_TEXT_LEN)
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
        texts = _within_turn_budget(texts, withheld, len(calls_text))

        content, thinking = response.content, response.thinking
        full_text = "\n".join(part for part in (content, calls_text) if part)
        turn_chars = len(full_text) + sum(len(text) for text in texts)
        if turn_chars > MAX_MEMORY_TEXT_LEN or not self._memory_write_allowed(
            full_text, "assistant"
        ):
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
    ) -> tuple[ReflectionEntry, ReflectionDecision]:
        """Structured reflection at attempt boundary.

        *final* says that no further attempt can start; the request then
        tells the model so (see :meth:`Chat.reflect`).

        After parsing, enforce :data:`MAX_REFLECTION_CHARS`: if the
        serialized form of the new reflection exceeds the budget, force
        ``decision.recommended_action = "decompose"`` so the next attempt
        does not start. The model is meant to keep the reflection compact
        — when it cannot, the agent stops chasing instead of carrying an
        ever-growing scratchpad. A ``done`` decision is left as it is: it
        ends the run, so no attempt would carry the reflection.
        """
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

        Parsing is tolerant: a Markdown code fence around the reply is
        stripped, and a reply that is not JSON as a whole is read from the
        first ``{...}`` object in it (see :func:`_decode_reflection`). Only
        a reply without such an object falls back to ``stop``, with the
        rationale "Could not parse reflection response"; a reply that is
        JSON but not an object stops too.
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
        counts, :data:`EMPTY_REPLY_NUDGE` is appended to working memory
        as a user message, and the model plans again. The
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
            response, fail = await self._do_plan(
                long_context,
                short_window,
                tools,
                query,
                reflections=reflections,
            )
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
                self.hippo.working.append(
                    Message(role="user", content=EMPTY_REPLY_NUDGE)
                )
                continue
            run.empty_replies = 0

            # Execute every call in order, each through the executor boundary,
            # which runs only the tools this planning call offered.
            allowed = frozenset(t.name for t in tools)
            calls = list(response.tool_calls)
            results: list[ToolResult] = []
            for call in calls:
                # Counted when handed over, so a call that timeout_s cuts
                # short counts too.
                run.n_tool += 1
                results.append(await self._do_execute(call, allowed=allowed))
            await self._do_memorize(task.task_id, response, results)

            # Stuck detection (one fingerprint per step)
            fingerprints.append(self._make_fingerprint(calls, results))
            stuck = self._detect_stuck(fingerprints)
            if stuck:
                return step + 1, stuck

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

    async def run_task(self, query: str) -> RunResult:
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
          effort, if any, is the result;
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
          ``python_exec`` code is the exception: it runs on the event
          loop, so neither ``timeout_s`` nor the executor's timeout can
          interrupt it, and the run ends only once that code returns.
          Closing the providers is not cut short, also when the deadline
          expires while a failed setup is closing them, so the run
          returns once they are closed, which can be after ``timeout_s``;
        - ``crashed``: an unexpected exception ended the run. It is
          logged and reported as the error, not raised. A cancellation
          from outside the run, or any other exception that is not an
          ``Exception``, still propagates once the task is saved.

        Whatever happens, the task record is saved with the status and
        ``result or error``, and the reflections are cleared (promoted
        to long-term memory first when ``memory`` is on); a failure while
        doing so is logged without a traceback and does not replace the
        run's own outcome. ``timeout_s`` bounds the run itself, not this
        final step. The save has no bound of its own; promotion has
        :data:`PROMOTION_TIMEOUT_S`, so an embedding server that stopped
        answering delays the return by that much at most.

        ``RunResult.config`` reports the effective configuration: the
        options, the registered built-ins (``builtins``), the resolved
        ``memory_dir`` and every tool registered once the tool providers
        are set up (``tools``), with tool names sorted.
        """
        started = time.perf_counter()
        task = TaskRecord(query=query)
        run = _RunState()
        stats = self.chat.reset_stats()
        timeout_s = self.options.timeout_s
        deadline = asyncio.timeout(timeout_s)  # never expires when None
        try:
            async with deadline:
                await self.hippo.save_task(task)
                try:
                    await self._ensure_tools()
                except Exception as exc:
                    raise _ToolSetupError(str(exc) or _exception_text(exc)) from exc
                self.hippo.working.append(Message(role="user", content=query))
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
            run.end("crashed", error=_exception_text(exc))
            raise
        finally:
            if run.running:
                # Every path above ends the run; this keeps the record
                # terminal should a later change miss one.
                run.end("crashed", error="The run ended without a status")
            await self._finish_task(task, run)
        return self._run_result(task, run, stats, time.perf_counter() - started)

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
          decides. A gate that stops the run ends it as ``gate_stopped``,
          with the ``best_effort_result`` as the result and the decision's
          rationale as the error, or ``Stopped after attempt <n>`` (``n``
          counting from 1) when the rationale is blank.
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
            entry, decision = await self._do_reflect(
                task, attempt_id, failure_type, final=bool(out_of)
            )
            self.hippo.working.add_reflection(entry)

            best_effort = decision.best_effort_result
            if decision.recommended_action == "done":
                answer = best_effort if best_effort.strip() else run.last_text
                run.end("completed", result=answer)
                return
            if out_of:
                run.end("budget_exhausted", result=best_effort, error=out_of)
                return
            if not _should_continue(decision):
                stopped = f"Stopped after attempt {run.attempts}"
                run.end(
                    "gate_stopped",
                    result=best_effort,
                    error=decision.rationale.strip() or stopped,
                )
                return

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
        """
        try:
            self._closing = True
            await self._close_providers(self._tool_providers)
            # Wait for a setup still running in a run (see above).
            async with self._tools_lock:
                pass
        finally:
            await self.hippo.aclose()
            await self._llm.close()
            if self._embedding:
                await self._embedding.close()
