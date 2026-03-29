from __future__ import annotations

import re

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef

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
    ) -> Message:
        messages = [Message(role="system", content=SYSTEM_PROMPT)]
        messages.extend(context)
        messages.extend(short)
        if not any(m.role == "user" for m in short):
            messages.append(Message(role="user", content=query))

        return await self.llm.chat(messages, tools=tools or None)
