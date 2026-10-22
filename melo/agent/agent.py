from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from localmelo.melo.agent._chat import Chat
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import (
    ExecutorResultPayload,
    MemoryWritePayload,
    ToolResolutionResult,
)
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.executor import Executor, register_builtins
from localmelo.melo.executor.models import ExecutionRequest
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import (
    MAX_AGENT_STEPS,
    MAX_ATTEMPTS,
    MAX_REFLECTION_CHARS,
    MIN_AGENT_STEPS,
    STEPS_PER_ATTEMPT,
    ImagePart,
    Message,
    ReflectionDecision,
    ReflectionEntry,
    StepRecord,
    TaskRecord,
    ToolCall,
    ToolDef,
    ToolResult,
    describe_images,
)

if TYPE_CHECKING:
    from localmelo.support.config import Config

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


# Characters of a withheld tool result that stay visible to the model.
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

    *text* is the result's text (see :func:`_tool_result_text`) or the
    notice that replaced it. The images of a result are not sent to the
    model, so a result with images ends with a line that names them (see
    :func:`describe_images`), whether its text is kept or withheld.
    """
    return "\n".join(part for part in (text, describe_images(images)) if part)


def _withheld_notice(text: str, reason: str) -> str:
    """Stand-in for a tool result kept out of working memory."""
    notice = f"[Tool result withheld from working memory: {reason}."
    preview = text[:WITHHELD_PREVIEW_CHARS]
    if not preview:
        return f"{notice}]"  # an image-only result: nothing to preview
    return f"{notice} First {len(preview)} characters follow.]\n{preview}"


def _within_turn_budget(
    texts: Sequence[str], withheld: set[int], calls_chars: int
) -> list[str]:
    """Return *texts* with the largest replaced by notices until they fit.

    The rendered tool calls of one turn (*calls_chars* characters) and its
    tool results share ``MAX_MEMORY_TEXT_LEN`` characters: the calls keep
    their share and the results get the rest. Indices in *withheld*
    already hold notices and are left alone; the index of each text
    replaced here is added to it.
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
        notice = _withheld_notice(fitted[index], reason)
        if len(notice) >= len(fitted[index]):
            break  # only results a notice would not shorten are left
        total += len(notice) - len(fitted[index])
        fitted[index] = notice
        withheld.add(index)
    return fitted


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


class Agent:
    """Task-solving agent with tool use and memory.

    Two construction modes:

    1. **Config-based** (gateway / full app)::

        from localmelo.support.config import load
        agent = Agent(config=load())

    2. **Direct provider injection** (testing / custom setups)::

        agent = Agent(llm=my_llm_provider, embedding=my_emb_provider)
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        llm: BaseLLMProvider | None = None,
        embedding: BaseEmbeddingProvider | None = None,
    ) -> None:
        if llm is not None:
            self._llm = llm
            self._embedding = embedding
        elif config is not None:
            self._llm, self._embedding = _providers_from_config(config)
        else:
            raise TypeError("Agent requires either config= or llm= to be provided")

        # Optional persistent memory backends (env-based opt-in).
        # LOCALMELO_PERSIST_MEMORY=1 enables SQLite-backed history and long-term.
        # LOCALMELO_MEMORY_DIR overrides the default storage directory.
        history_backend = None
        long_backend = None
        if os.environ.get("LOCALMELO_PERSIST_MEMORY"):
            from localmelo.melo.memory.history.sqlite import SqliteHistory
            from localmelo.melo.memory.long.sqlite import SqliteLongTerm

            mem_dir = os.environ.get(
                "LOCALMELO_MEMORY_DIR",
                os.path.expanduser("~/.cache/localmelo/memory"),
            )
            os.makedirs(mem_dir, exist_ok=True)
            history_backend = SqliteHistory(os.path.join(mem_dir, "history.db"))
            if self._embedding:
                long_backend = SqliteLongTerm(os.path.join(mem_dir, "long_term.db"))

        self.hippo = Hippo(
            embedding=self._embedding,
            history=history_backend,
            long=long_backend,
        )
        self.checker = Checker()
        self.executor = Executor(self.hippo, self.checker)
        self.chat = Chat(self._llm)

        register_builtins(self.executor, self.hippo)

    # ── Stage helpers ──
    # Each helper owns one phase of the agent loop.  They return a
    # short status string when the loop should break, or None to
    # continue to the next stage.

    async def _do_retrieval(
        self, query: str
    ) -> tuple[list[Message], list[Message], list[ToolDef], str | None]:
        """Stage 1+2: retrieve context, resolve tools, run boundary checks.

        Returns (long_context, short_window, tools, fail_reason).
        *fail_reason* is non-None when a check fails and the loop must break.
        """
        long_context = await self.hippo.retrieve_context(query)
        short_window = self.hippo.working.get_window()

        # Include reflection context in tool hint extraction
        reflection_msgs = _serialize_reflections(self.hippo.working.get_reflections())
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

    async def _do_execute(self, call: ToolCall) -> ToolResult:
        """Stage 4: execute one tool call and validate the result.

        Uses the sanitized payload from the executor-result checker
        when truncation or modification was applied; falls back to
        the raw outcome otherwise.
        """
        exec_request = ExecutionRequest(
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
        )
        outcome = await self.executor.execute_structured(exec_request)

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
        rejected tool result is replaced by a short notice bound to the
        same ``tool_call_id``. No call or result is ever left out, so the
        turn stays protocol-valid and the model still sees every call it
        made and what each returned. A result's images (references, see
        :func:`_image_references`) are checked with its text. A result
        replaced by a notice, by that check or by the budget below, keeps
        no images on its message, but its message still ends with the
        line that names them (see :func:`_tool_message_content`).

        The whole turn also shares one budget of ``MAX_MEMORY_TEXT_LEN``
        characters, half the 100,000 at which ``Checker.pre_plan`` fails
        the task, so that one step leaves room in the next planning
        prompt. The calls keep their share; the results share the rest,
        the largest replaced by the notice first (see
        :func:`_within_turn_budget`); and the assistant text, which history
        keeps, is dropped when it does not fit beside them. Notices, and
        results shorter than one, cannot shrink further, so a turn whose
        calls nearly fill the budget, or a turn of hundreds of calls, can
        still pass it.

        The lines that name images are added after the check and the
        budget and count toward neither, so they never withhold a result
        that fits without them. Each is at most a few hundred characters,
        so a turn can pass the budget by that much per result with images.
        """
        calls = self._checked_calls(response.tool_calls)
        calls_text = _render_tool_calls(calls)

        texts: list[str] = []
        withheld: set[int] = set()
        for index, result in enumerate(results):
            text = _tool_result_text(result)
            # A memory write needs text: an image-only result is checked
            # with the line that names its images.
            check = self.checker.check_memory_write(
                MemoryWritePayload(
                    text=text or describe_images(result.images),
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
        """Return one ``(tool_names, args_hash, errors)`` fingerprint per step.

        A step with several calls is fingerprinted as a whole: calls are
        sorted (the same batch in another order matches) and their errors
        joined. ``_detect_stuck`` therefore fires when the same batch, or
        the same errors, repeat across steps — not merely because one step
        contains several calls.
        """
        pairs = sorted(
            (tc.tool_name, json.dumps(tc.arguments, sort_keys=True, default=str))
            for tc in calls
        )
        return (
            ",".join(name for name, _ in pairs),
            hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16],
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
        # Same tool+args repeated 3 times
        keys = {(name, args_h) for name, args_h, _ in last3}
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
    ) -> tuple[ReflectionEntry, ReflectionDecision]:
        """Structured reflection at attempt boundary.

        After parsing, enforce :data:`MAX_REFLECTION_CHARS`: if the
        serialized form of the new reflection exceeds the budget, force
        ``decision.recommended_action = "decompose"`` so the next attempt
        does not start. The model is meant to keep the reflection compact
        — when it cannot, the agent stops chasing instead of carrying an
        ever-growing scratchpad.
        """
        short_window = self.hippo.working.get_window()
        prior_reflections = self.hippo.working.get_reflections()
        response = await self.chat.reflect(
            short=short_window,
            query=task.query,
            attempt_id=attempt_id,
            failure_type=failure_type,
            prior_reflections=prior_reflections or None,
        )
        entry, decision = self._parse_reflection(
            response.content, attempt_id, failure_type
        )
        serialized = _serialize_reflections([entry])
        size = serialized[0].content.__len__() if serialized else 0
        if size > MAX_REFLECTION_CHARS:
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
        """Parse LLM reflection JSON with strict coercion and conservative fallback."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
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
        if action not in ("continue", "stop", "decompose"):
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
        self, task: TaskRecord, query: str, budget: int
    ) -> tuple[int, str]:
        """Run one attempt of the agent loop.

        Returns ``(steps_used, failure_type)``.
        *failure_type* is ``""`` when the task reached a terminal state
        (completed or failed via checker), ``"budget"`` when the step
        budget was exhausted, or ``"stuck"`` when stuck detection fired.
        The method mutates *task.status* / *task.result* on terminal
        conditions (direct answer or checker failure).
        """
        fingerprints: list[tuple[str, str, str]] = []
        reflections = self.hippo.working.get_reflections() or None

        for step in range(budget):
            # Retrieval + tool resolution + boundary checks
            long_context, short_window, tools, fail = await self._do_retrieval(query)
            if fail is not None:
                task.status = "failed"
                task.result = fail
                return step + 1, ""

            # LLM planning step (with reflection context)
            response, fail = await self._do_plan(
                long_context,
                short_window,
                tools,
                query,
                reflections=reflections,
            )
            if fail is not None:
                task.status = "failed"
                task.result = fail
                return step + 1, ""

            # No tool call → direct answer
            if not response.tool_calls:
                task.status = "completed"
                task.result = response.content
                return step + 1, ""

            # Execute every call in order, each through the executor boundary
            calls = list(response.tool_calls)
            results: list[ToolResult] = []
            for call in calls:
                results.append(await self._do_execute(call))
            await self._do_memorize(task.task_id, response, results)

            # Stuck detection (one fingerprint per step)
            fingerprints.append(self._make_fingerprint(calls, results))
            stuck = self._detect_stuck(fingerprints)
            if stuck:
                return step + 1, stuck

        return budget, "budget"

    # ── Main loop ──

    async def run(self, query: str) -> str:
        task = TaskRecord(query=query)
        await self.hippo.save_task(task)
        self.hippo.working.append(Message(role="user", content=query))

        total_steps = 0

        for attempt_id in range(MAX_ATTEMPTS):
            budget = min(STEPS_PER_ATTEMPT, MAX_AGENT_STEPS - total_steps)
            if budget <= 0:
                task.status = "failed"
                task.result = "Max steps reached"
                break

            steps_used, failure_type = await self._run_attempt(task, query, budget)
            total_steps += steps_used

            if task.status in ("completed", "failed"):
                break

            # Reflect at attempt boundary
            task.attempts_completed = attempt_id + 1
            entry, decision = await self._do_reflect(task, attempt_id, failure_type)
            self.hippo.working.add_reflection(entry)

            if not _should_continue(decision):
                task.status = "failed"
                task.result = (
                    decision.best_effort_result
                    or f"Stopped after attempt {attempt_id}: {decision.rationale}"
                )
                break
        else:
            if task.status == "running":
                task.status = "failed"
                task.result = "Max attempts reached"

        await self.hippo.promote_reflections(task.task_id)
        await self.hippo.save_task(task)
        return task.result

    async def close(self) -> None:
        await self.hippo.aclose()
        await self._llm.close()
        if self._embedding:
            await self._embedding.close()
