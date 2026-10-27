"""Fakes for the tests of observe=, run.progress and run-end learning.

No real model: :class:`ScriptedLLM` answers each request with the next
scripted reply. The custom tools carry the names of the built-ins
(``file_read``, ``file_write``, ``shell_exec``), so the procedural trace
classifies their calls as it would a real run's, but they touch no file
and start no process.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import RETRY_HOOK
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor import TOOL_EXIT_CODE
from localmelo.melo.memory.procedures.normalize import (
    RoutineKey,
    key_for_query,
    length_options,
    run_place,
)
from localmelo.melo.memory.procedures.store import RunFacts
from localmelo.melo.memory.procedures.trace import Trace
from localmelo.melo.schema import Message, ToolCall, ToolDef

MODEL = "test-model-7b"
QUERY = "run pytest in the project and fix the failures"


class ScriptedLLM(BaseLLMProvider):
    """Answers each request with the next scripted reply, then waits forever.

    A scripted exception is raised instead. ``blocked`` is set once a
    request finds no reply left.
    """

    def __init__(self, replies: Iterable[Message | BaseException], model: str = MODEL):
        self.replies: deque[Message | BaseException] = deque(replies)
        self.requests: list[list[Message]] = []
        self.model = model
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        if not self.replies:
            self.blocked.set()
            await asyncio.Event().wait()
        reply = self.replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self) -> None:
        pass


class Recorder:
    """A sink that keeps every event."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    @property
    def types(self) -> list[str]:
        return [event.type for event in self.events]

    def of(self, type_: str) -> list[AgentEvent]:
        return [event for event in self.events if event.type == type_]


def _one_string(name: str) -> dict[str, Any]:
    return {"type": "object", "properties": {name: {"type": "string"}}}


@dataclass
class Tools:
    """Stand-ins for the built-ins, and ``hang``, which never returns.

    ``shell`` maps a command to its output (``"ok"`` when it is not
    there). ``slots`` and ``hooks`` keep TOOL_EXIT_CODE and RETRY_HOOK as
    each call saw them. ``hanging`` is set once ``hang`` runs.
    """

    shell: dict[str, str] = field(default_factory=dict)
    slots: list[Any] = field(default_factory=list)
    hooks: list[Any] = field(default_factory=list)
    hanging: asyncio.Event = field(default_factory=asyncio.Event)

    async def file_read(self, path: str = "") -> str:
        self._saw()
        return f"contents of {path}"

    async def file_write(self, path: str = "", content: str = "") -> str:
        self._saw()
        return f"wrote {len(content)} characters"

    async def shell_exec(self, command: str = "") -> str:
        self._saw()
        return self.shell.get(command, "ok")

    async def hang(self) -> str:
        self._saw()
        self.hanging.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    def _saw(self) -> None:
        self.slots.append(TOOL_EXIT_CODE.get())
        self.hooks.append(RETRY_HOOK.get())


DEFINITIONS = {
    "file_read": ToolDef(
        name="file_read", description="Read a file", parameters=_one_string("path")
    ),
    "file_write": ToolDef(
        name="file_write",
        description="Write a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
    ),
    "shell_exec": ToolDef(
        name="shell_exec",
        description="Run a shell command",
        parameters=_one_string("command"),
    ),
    "hang": ToolDef(name="hang", description="Never return", parameters={}),
}


async def _no_wait(delay: float) -> None:
    pass


def make_agent(
    llm: BaseLLMProvider, *, tools: Tools | None = None, **options: Any
) -> Agent:
    """An agent with the stand-in tools, offered on every step, and no waits."""
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    agent = Agent(llm=llm, options=AgentOptions(**options))
    tools = tools or Tools()
    functions: dict[str, Callable[..., Any]] = {
        "file_read": tools.file_read,
        "file_write": tools.file_write,
        "shell_exec": tools.shell_exec,
        "hang": tools.hang,
    }
    for name, definition in DEFINITIONS.items():
        agent.hippo.register_tool(definition)
        agent.executor.register(name, functions[name])
    agent.chat.sleep = _no_wait
    return agent


def answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def call(name: str, call_id: str = "call_1", **arguments: Any) -> Message:
    """A reply with one tool call."""
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name=name, arguments=arguments, id=call_id)],
    )


def read(path: str, call_id: str = "call_1") -> Message:
    return call("file_read", call_id, path=path)


def write(path: str, call_id: str = "call_1") -> Message:
    return call("file_write", call_id, path=path, content="fixed")


def shell(command: str, call_id: str = "call_1") -> Message:
    return call("shell_exec", call_id, command=command)


def fix_run(writes: int = 1) -> list[Message]:
    """Read the test, then *writes* rounds of edit and pytest, then answer.

    Each pytest but the last fails; give the agent ``passing_tools()``.
    """
    replies = [read("/p/test_parser.py")]
    for i in range(writes):
        replies.append(write("/p/parser.py", f"call_w{i}"))
        command = "pytest -q" if i == writes - 1 else f"pytest -q -x # {i}"
        replies.append(shell(command, f"call_t{i}"))
    replies.append(answer("Fixed."))
    return replies


def passing_tools() -> Tools:
    """pytest passes on its last round (see :func:`fix_run`), fails before."""
    tools = Tools()
    tools.shell["pytest -q"] = "8 passed in 0.12s"
    for i in range(10):
        tools.shell[f"pytest -q -x # {i}"] = "1 failed, 7 passed in 0.12s"
    return tools


def past_trace(steps: int) -> Trace:
    """A past fix run of *steps* steps: read, edits, a passing pytest, an answer."""
    trace = Trace()
    trace.step(1, 1)
    trace.tool("file_read", {"path": "/p/test_parser.py"}, "success", "")
    for k in range(2, steps - 1):
        trace.step(k, 1)
        trace.tool(
            "file_write", {"path": "/p/parser.py", "content": "x"}, "success", ""
        )
    trace.step(steps - 1, 1)
    trace.tool("shell_exec", {"command": "pytest -q"}, "success", "8 passed in 0.1s")
    trace.step(steps, 1)
    return trace


def query_key(agent: Agent, query: str = QUERY) -> RoutineKey:
    return key_for_query(query, run_place(agent.workspace_root))


async def remember(
    agent: Agent, *steps: int, key: RoutineKey | None = None, model: str = MODEL
) -> None:
    """Record completed past runs of *steps* steps in the agent's store."""
    for n in steps:
        await agent.hippo.procedures.learn(
            RunFacts.from_trace(
                past_trace(n),
                task_id=uuid.uuid4().hex,
                key=key or query_key(agent),
                model=model,
                options=length_options(agent._run_config()),
                status="completed",
                steps=n,
                attempts=1,
                duration_s=60.0,
            )
        )
