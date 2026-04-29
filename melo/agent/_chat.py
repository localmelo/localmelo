from __future__ import annotations

import re
from typing import TYPE_CHECKING

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef

if TYPE_CHECKING:
    from localmelo.melo.schema import ReflectionEntry

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
    '  "recommended_action": "continue" or "stop" or "decompose",\n'
    '  "rationale": "why you chose this action",\n'
    '  "best_effort_result": "best partial answer so far, or empty string",\n'
    '  "estimated_info_gain": 0.0-1.0,\n'
    '  "estimated_cost": 0.0-1.0,\n'
    '  "repeat_risk": 0.0-1.0,\n'
    '  "novelty": 0.0-1.0,\n'
    '  "feasibility": 0.0-1.0\n'
    "}"
)

_MEMORY_PREFIX = "[memory] "


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


class Chat:
    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    async def estimate_steps(self, query: str) -> int:
        """Ask the LLM to estimate how many tool-call steps *query* needs."""
        messages = [
            Message(role="system", content=STEP_ESTIMATE_PROMPT),
            Message(role="user", content=query),
        ]
        response = await self.llm.chat(messages, tools=None)
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
        messages.extend(other_msgs)
        if not any(m.role == "user" for m in short):
            messages.append(Message(role="user", content=query))

        return await self.llm.chat(messages, tools=tools or None)

    async def reflect(
        self,
        short: list[Message],
        query: str,
        attempt_id: int,
        failure_type: str = "",
        prior_reflections: list[ReflectionEntry] | None = None,
    ) -> Message:
        """Ask the LLM to reflect on the current attempt."""
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

        messages = [Message(role="system", content="\n\n".join(system_parts))]
        messages.extend(short[-10:])
        messages.append(
            Message(
                role="user",
                content=(
                    f"Original task: {query}\n"
                    f"This was attempt {attempt_id}. {reason} "
                    "Reflect and decide."
                ),
            ),
        )
        return await self.llm.chat(messages, tools=None)
