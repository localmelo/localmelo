from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from localmelo.melo.contracts.providers import (
    LLM_REQUEST_ATTEMPT,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.memory.short import tail_turns
from localmelo.melo.schema import Message, ToolDef

if TYPE_CHECKING:
    from localmelo.melo.schema import ReflectionEntry

logger = logging.getLogger(__name__)

# Seconds to wait before each retry of a request that failed with a
# retryable ProviderError: at most two retries, 1 s and then 4 s apart.
RETRY_DELAYS_S: tuple[float, ...] = (1.0, 4.0)

SYSTEM_PROMPT = (
    "You are a task-solving agent. You can use tools to accomplish tasks.\n"
    "When you need to use a tool, respond with a tool call.\n"
    "When the task is complete, respond with your final answer directly.\n"
    "Think step by step. Be concise."
)

STEP_ESTIMATE_PROMPT = (
    "How many tool-call steps will this task need? "
    "Give a conservative upper bound — it is better to overestimate than underestimate. "
    "If the task can be answered directly without tools, say 0. "
    "Reply with ONLY a single integer."
)


def _parse_step_estimate(text: str) -> int:
    """Extract the first integer from *text*.  Returns -1 on failure."""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else -1


REFLECTION_PROMPT = (
    "The current attempt has ended without a final answer. "
    "Reflect on what happened and decide how to proceed.\n\n"
    "IMPORTANT: this reflection is the *only* state carried into the next "
    "attempt. It is compressed working state, not a log. Treat any prior "
    "reflection the same way: keep only constraints, failed paths, and "
    "evidence that are still relevant to the next concrete step. Drop the "
    "rest. The total reflection should stay under ~1500 characters when "
    "serialized — if you cannot fit it, choose 'stop' or 'decompose' "
    "instead of 'continue'.\n\n"
    "Choose 'done' only if the task is already complete, and put the final "
    "answer in best_effort_result.\n\n"
    "Respond with a JSON object (no markdown fences):\n"
    "{\n"
    '  "summary": "one-sentence summary of what was attempted",\n'
    '  "failed_hypotheses": ["approaches that did not work"],\n'
    '  "useful_evidence": ["facts discovered that may help"],\n'
    '  "recommended_avoids": ["actions to avoid next"],\n'
    '  "next_promising_directions": ["concrete next steps to try"],\n'
    '  "progress_made": true/false,\n'
    '  "task_still_feasible": true/false,\n'
    '  "new_information_gained": true/false,\n'
    '  "next_step_is_concrete": true/false,\n'
    '  "next_step_is_novel": true/false,\n'
    '  "recommended_action": "continue" | "stop" | "decompose" | "done",\n'
    '  "rationale": "why you chose this action",\n'
    '  "best_effort_result": "the final answer if done, else the best partial '
    'answer so far, or empty string",\n'
    '  "estimated_info_gain": 0.0-1.0,\n'
    '  "estimated_cost": 0.0-1.0,\n'
    '  "repeat_risk": 0.0-1.0,\n'
    '  "novelty": 0.0-1.0,\n'
    '  "feasibility": 0.0-1.0\n'
    "}"
)

# Ends the request of a reflection after which no further attempt can
# start: its decision cannot continue the run, only finalize its answer.
FINAL_REFLECTION_NOTE = (
    "No further attempt can start: choose 'done' if the task is complete, and "
    "either way put your best answer in best_effort_result."
)

# Ends the request when a reflection is asked again because the first reply
# held no JSON object. Such a reply is often a tool call: the window ends in
# tool turns, and some models continue them although the request offers no
# tools.
REFLECTION_RETRY_NOTE = (
    "Reply with only the JSON object described above. Do not call any tool: "
    "tools are not available for this reply."
)

_MEMORY_PREFIX = "[memory] "

# Most recent working-memory messages shown to the reflection call. The
# window is cut on turn boundaries, so a tool call keeps its results.
REFLECTION_WINDOW = 10


def lacks_task_query(short: Sequence[Message], query: str) -> bool:
    """Whether the working-memory window *short* no longer holds *query*.

    A planning request then puts the task query back in front of the
    window (see :meth:`Chat.plan_step`). Other user messages, such as the
    agent's note after an empty reply, do not stand in for it.
    """
    return not any(m.role == "user" and m.content == query for m in short)


def _build_system_prompt(
    context: list[Message],
    short: list[Message],
    reflections: list[ReflectionEntry] | None = None,
) -> tuple[str, list[Message]]:
    """Merge all system messages into a single system prompt.

    Memory items (prefixed with ``[memory] ``) are collected under a
    single ``[RECALL]`` section.  Reflection entries are rendered under
    a ``[REFLECTION]`` section.  Other system messages are appended
    verbatim.  Non-system messages are returned unchanged.

    Returns ``(merged_system_prompt, non_system_messages)``.
    """
    recall_items: list[str] = []
    extra_system: list[str] = []
    non_system: list[Message] = []

    for msg in (*context, *short):
        if msg.role == "system":
            text = msg.content
            if text.startswith(_MEMORY_PREFIX):
                recall_items.append(text[len(_MEMORY_PREFIX) :])
            else:
                extra_system.append(text)
        else:
            non_system.append(msg)

    parts: list[str] = [SYSTEM_PROMPT]
    for extra in extra_system:
        parts.append(extra)
    if recall_items:
        parts.append("[RECALL]")
        parts.extend(recall_items)
    if reflections:
        reflection_lines = ["[REFLECTION]"]
        for r in reflections:
            lines = [f"Attempt {r.attempt_id}: {r.summary}"]
            if r.failed_hypotheses:
                lines.append(f"  Failed: {'; '.join(r.failed_hypotheses)}")
            if r.recommended_avoids:
                lines.append(f"  Avoid: {'; '.join(r.recommended_avoids)}")
            if r.next_promising_directions:
                lines.append(f"  Try: {'; '.join(r.next_promising_directions)}")
            if r.useful_evidence:
                lines.append(f"  Evidence: {'; '.join(r.useful_evidence)}")
            reflection_lines.append("\n".join(lines))
        parts.append("\n".join(reflection_lines))

    return "\n\n".join(parts), non_system


@dataclass
class RequestStats:
    """Counts of the LLM requests a :class:`Chat` sent, and their usage.

    ``n_llm`` counts requests that returned a response; ``n_main`` and
    ``n_aux`` split them by whether the request offered tools, so
    ``n_main + n_aux == n_llm``. ``llm_errors`` counts requests that
    raised instead: every failed attempt of a retried request, and a
    request cut short by a cancellation, so ``n_llm + llm_errors`` is the
    number of requests sent. ``usage`` sums the integer usage fields of
    the responses by name.
    """

    n_llm: int = 0
    n_main: int = 0
    n_aux: int = 0
    llm_errors: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    def add_usage(self, usage: Mapping[str, object] | None) -> None:
        """Add the integer fields of one response's usage to the totals."""
        for name, value in (usage or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                self.usage[name] = self.usage.get(name, 0) + value


class Chat:
    """Builds the agent's LLM requests, retries failed ones, counts them.

    A request that fails with a retryable :class:`ProviderError` is sent
    again after each delay in *retry_delays* (default
    :data:`RETRY_DELAYS_S`), waited out with *sleep*, so tests can pass a
    *sleep* that returns at once.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.llm = llm
        self.retry_delays = tuple(retry_delays)
        self.sleep = sleep
        self.stats = RequestStats()

    def reset_stats(self) -> RequestStats:
        """Count from zero again; returns the new :attr:`stats`.

        The agent calls this at the start of each run, so the counts
        cover one run.
        """
        self.stats = RequestStats()
        return self.stats

    async def _request(
        self, messages: list[Message], tools: list[ToolDef] | None
    ) -> Message:
        """Send one request to the LLM and count it in :attr:`stats`.

        A request with tools counts as main, one without as aux; an empty
        tool list is sent as ``None`` and counts as aux.

        A retryable :class:`ProviderError` is retried after each of
        :attr:`retry_delays` in turn; each attempt runs with its 1-based
        number in ``LLM_REQUEST_ATTEMPT``. Every attempt that raises
        counts in ``llm_errors``, including one cut short by a
        cancellation. The error of the last attempt, and any other
        exception at once, propagates.
        """
        attempt = 1
        while True:
            try:
                response = await self._send(messages, tools, attempt)
                break
            except ProviderError as exc:
                self.stats.llm_errors += 1
                if not exc.retryable or attempt > len(self.retry_delays):
                    raise
                delay = self.retry_delays[attempt - 1]
                logger.info(
                    "LLM request attempt %d failed, retrying in %g s: %s",
                    attempt,
                    delay,
                    exc,
                )
            except BaseException:
                self.stats.llm_errors += 1
                raise
            await self.sleep(delay)
            attempt += 1
        self.stats.n_llm += 1
        if tools:
            self.stats.n_main += 1
        else:
            self.stats.n_aux += 1
        self.stats.add_usage(response.usage)
        return response

    async def _send(
        self, messages: list[Message], tools: list[ToolDef] | None, attempt: int
    ) -> Message:
        """Send one attempt of a request, with *attempt* in LLM_REQUEST_ATTEMPT."""
        token = LLM_REQUEST_ATTEMPT.set(attempt)
        try:
            return await self.llm.chat(messages, tools=tools or None)
        finally:
            LLM_REQUEST_ATTEMPT.reset(token)

    async def estimate_steps(self, query: str) -> int:
        """Ask the LLM to estimate how many tool-call steps *query* needs."""
        messages = [
            Message(role="system", content=STEP_ESTIMATE_PROMPT),
            Message(role="user", content=query),
        ]
        response = await self._request(messages, tools=None)
        return _parse_step_estimate(response.content)

    async def plan_step(
        self,
        context: list[Message],
        short: list[Message],
        tools: list[ToolDef],
        query: str,
        reflections: list[ReflectionEntry] | None = None,
    ) -> Message:
        system_prompt, other_msgs = _build_system_prompt(context, short, reflections)
        messages = [Message(role="system", content=system_prompt)]
        if lacks_task_query(short, query):
            # The task query is no longer in the window (evicted, or dropped
            # to fit the prompt cap). Put it back ahead of the remaining
            # turns so the conversation still opens with the request instead
            # of ending with it. The agent counts it when it checks the
            # prompt's size (see Agent._fit_prompt_cap).
            messages.append(Message(role="user", content=query))
        messages.extend(other_msgs)

        return await self._request(messages, tools=tools)

    async def reflect(
        self,
        short: list[Message],
        query: str,
        attempt_id: int,
        failure_type: str = "",
        prior_reflections: list[ReflectionEntry] | None = None,
        *,
        final: bool = False,
        retry: bool = False,
    ) -> Message:
        """Ask the LLM to reflect on the current attempt.

        The request is a user message after the newest turns of *short*.
        With *final*, no further attempt can start, and the request says
        so: the reflection can only finalize the answer. With *retry*, the
        request is asked again after a reply without the JSON object, and
        ends with :data:`REFLECTION_RETRY_NOTE`; everything before that
        note is the same as in the first request.
        """
        # Build system prompt with prior reflection context
        system_parts = [REFLECTION_PROMPT]
        if prior_reflections:
            lines = ["Prior reflections:"]
            for r in prior_reflections:
                lines.append(f"  Attempt {r.attempt_id}: {r.summary}")
                if r.recommended_avoids:
                    lines.append(f"    Avoid: {'; '.join(r.recommended_avoids)}")
                if r.useful_evidence:
                    lines.append(f"    Evidence: {'; '.join(r.useful_evidence)}")
                if r.failed_hypotheses:
                    lines.append(f"    Failed: {'; '.join(r.failed_hypotheses)}")
            system_parts.append("\n".join(lines))

        # Describe why the attempt ended
        reason_map = {
            "budget": "The step budget for this attempt was exhausted.",
            "stuck": "A repeated tool-call or error pattern was detected.",
            "error": "The attempt ended due to a failure condition.",
        }
        reason = reason_map.get(
            failure_type, "The attempt ended without a final answer."
        )
        request = (
            f"Original task: {query}\n"
            f"This was attempt {attempt_id}. {reason} Reflect and decide."
        )
        if final:
            request += f" {FINAL_REFLECTION_NOTE}"
        if retry:
            request += f"\n\n{REFLECTION_RETRY_NOTE}"

        tail = tail_turns(short, REFLECTION_WINDOW)
        if tail and tail[-1].role == "user":
            # Some chat templates (Mistral, Gemma) reject two user messages
            # in a row, so the request does not follow one: the agent's
            # nudge after an empty reply is left out, and the empty reply's
            # label, an assistant message, ends the turns instead.
            tail.pop()
        messages = [Message(role="system", content="\n\n".join(system_parts))]
        messages.extend(tail)
        messages.append(Message(role="user", content=request))
        return await self._request(messages, tools=None)
