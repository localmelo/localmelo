"""A chat's earlier turns reach the requests of a run whose window lost them.

The replay: three turns about a friend (Alex), then "go look it up". The run's
tool steps fill the small window, its last tool output is the user's own profile
(Sam), and the step budget ends it. The stub model answers about whoever the
context names, as a real model did: the person named first in the last user
message, else the person named last anywhere in the request. Without the chat's
earlier turns the answer is about Sam; with them it stays on Alex, whether the
final reflection answers (``done``) or the answer on stop does. No real model.
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import ANSWER_ON_STOP_NOTE
from localmelo.melo.agent._earlier import (
    EARLIER_ANSWER_CHARS,
    EARLIER_BLOCK_CHARS,
    EARLIER_END,
    EARLIER_HEAD,
    EARLIER_USER_CHARS,
    check_earlier_turns,
    earlier_block,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef

NAMES = ("Alex", "Sam")
TURNS = [
    (
        "What is Alex's background? The site is example.com",
        "Alex is a UX researcher who studies how people pay online.",
    ),
    (
        "What else could Alex do besides UX research?",
        "Alex could move into product management or service design.",
    ),
    (
        "List those roles so they can start applying",
        "Roles to search for: product researcher, service designer, research ops.",
    ),
]
PROFILE = (
    "# Sam Rivera — Background\nSam studies robotics and interned on a "
    "perception team. Sam is looking for robotics roles."
)
OPENINGS = "Openings: product researcher (example.com/jobs/1), service designer."


def subject(messages: list[Message]) -> str:
    """Whom the stub answers about: the first name in the last user message, else
    the last name anywhere in the request (the newest context wins)."""
    users = [m for m in messages if m.role == "user"]
    last = users[-1].content if users else ""
    found = [(last.find(name), name) for name in NAMES if name in last]
    if found:
        return min(found)[1]
    text = "\n".join(m.content for m in messages)
    return max((text.rfind(name), name) for name in NAMES)[1]


class SubjectModel(BaseLLMProvider):
    """Plans from a script; reflects and answers about whoever the context names.

    With no planning reply queued, a planning request is answered at once, about
    whoever the context names too.
    """

    def __init__(self, reflection: str = "done") -> None:
        self.plans: deque[Message] = deque()
        self.reflection = reflection
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        last = messages[-1].content
        if ANSWER_ON_STOP_NOTE in last:
            return Message(
                role="assistant", content=f"Roles that fit {subject(messages)}."
            )
        if "Reflect and decide" in last:
            done = self.reflection == "done"
            reply = {
                "summary": "searched job boards",
                "task_still_feasible": True,
                "next_step_is_concrete": False,
                "next_step_is_novel": False,
                "recommended_action": "done" if done else "stop",
                "rationale": "enough found",
                "best_effort_result": (
                    f"Roles that fit {subject(messages)}." if done else ""
                ),
            }
            return Message(role="assistant", content=json.dumps(reply))
        if not self.plans:
            return Message(
                role="assistant", content=f"Roles that fit {subject(messages)}."
            )
        return self.plans.popleft()

    async def close(self) -> None:
        pass


def _call(name: str, **arguments: Any) -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(tool_name=name, arguments=arguments)],
    )


def _agent(model: SubjectModel, **options: Any) -> Agent:
    settings: dict[str, Any] = {
        "builtins": frozenset(),
        "record_answers": True,
        "memory": False,
        "attempt_policy": "reflect",
        "steps_per_attempt": 5,
        "max_steps": 5,
        "window_messages": 10,
        "tool_exposure": "all",
        **options,
    }
    agent = Agent(llm=model, options=AgentOptions(**settings))
    for name, text in (("look_up", OPENINGS), ("read_profile", PROFILE)):
        agent.hippo.register_tool(
            ToolDef(
                name=name,
                description=f"{name} tool",
                parameters={
                    "type": "object",
                    "properties": {"part": {"type": "string"}},
                    "required": [],
                },
            )
        )

        async def tool(part: str = "", _text: str = text) -> str:
            return f"{_text} ({part})" if part else _text

        agent.executor.register(name, tool)
    return agent


async def _chat_then_look_up(
    model: SubjectModel, agent: Agent, *, earlier: bool
) -> str:
    """The three turns, then "go look it up"; returns the last run's result."""
    for words, answer in TURNS:
        model.plans.append(Message(role="assistant", content=answer))
        done = await agent.run_task(words)
        assert done.status == "completed"
    model.plans.extend(
        [
            _call("look_up", part="boards"),
            _call("look_up", part="more boards"),
            _call("read_profile", part="education"),
            _call("read_profile", part="work"),
            _call("read_profile", part="projects"),
        ]
    )
    model.calls.clear()
    outcome = await agent.run_task(
        "go look it up", earlier_turns=TURNS if earlier else None
    )
    return outcome.result


LAYOUTS = [
    pytest.param({}, id="default-window"),
    pytest.param({"window_messages": 8}, id="small-window"),
]


class TestReplay:
    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("reflection", ["done", "stop"])
    async def test_the_answer_stays_on_the_friend(
        self, layout: dict[str, Any], reflection: str
    ) -> None:
        model = SubjectModel(reflection)
        agent = _agent(model, **layout)
        result = await _chat_then_look_up(model, agent, earlier=True)
        assert result == "Roles that fit Alex."

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("reflection", ["done", "stop"])
    async def test_without_the_earlier_turns_the_profile_wins(
        self, layout: dict[str, Any], reflection: str
    ) -> None:
        # The failure the block fixes: the window lost the turns about Alex.
        model = SubjectModel(reflection)
        agent = _agent(model, **layout)
        result = await _chat_then_look_up(model, agent, earlier=False)
        assert result == "Roles that fit Sam."

    async def test_the_window_lost_the_turns_and_keeps_the_users_words(self) -> None:
        model = SubjectModel()
        agent = _agent(model)
        await _chat_then_look_up(model, agent, earlier=True)
        window = agent.conversation()
        assert not any(m.content in dict(TURNS) for m in window)
        assert not any(m.content in dict(TURNS).values() for m in window)
        # A planning request's block rides on its copy of the query, never on the
        # window's message (the kept reflection request quotes it, as it was sent).
        assert not any(m.content.startswith(EARLIER_HEAD) for m in window)

    @pytest.mark.parametrize("layout", LAYOUTS)
    async def test_a_planning_request_after_the_eviction_carries_the_turns(
        self, layout: dict[str, Any]
    ) -> None:
        model = SubjectModel()
        agent = _agent(model, **layout)
        await _chat_then_look_up(model, agent, earlier=True)
        plans = [c for c in model.calls if c["tools"] and not _reflecting(c)]
        first, *later = plans
        # The first step still has the turns in its window: nothing is added.
        assert not any(EARLIER_HEAD in m.content for m in first["messages"])
        pinned = [
            m.content
            for call in later
            for m in call["messages"]
            if m.role == "user" and m.content.startswith(EARLIER_HEAD)
        ]
        assert pinned and all(text.endswith("go look it up") for text in pinned)
        assert "What else could Alex do besides UX research?" in pinned[0]

    async def test_the_reflection_request_names_the_turns_before_the_task(self) -> None:
        model = SubjectModel()
        agent = _agent(model)
        await _chat_then_look_up(model, agent, earlier=True)
        (reflect,) = [c for c in model.calls if _reflecting(c)]
        asking = reflect["messages"][-1].content
        assert asking.index(EARLIER_HEAD) < asking.index("Original task: go look it up")

    async def test_the_answer_on_stop_names_the_turns_before_the_task(self) -> None:
        model = SubjectModel("stop")
        agent = _agent(model)
        await _chat_then_look_up(model, agent, earlier=True)
        (answer,) = [
            c for c in model.calls if ANSWER_ON_STOP_NOTE in c["messages"][-1].content
        ]
        text = answer["messages"][-1].content
        assert text.startswith(EARLIER_HEAD)
        assert text.index(EARLIER_END) < text.index("Original task: go look it up")

    async def test_closing_answer_after_the_run_carries_the_turns(self) -> None:
        model = SubjectModel()
        agent = _agent(model)
        await _chat_then_look_up(model, agent, earlier=True)
        model.calls.clear()
        answer = await agent.closing_answer("go look it up", note=ANSWER_ON_STOP_NOTE)
        assert answer == "Roles that fit Alex."
        assert model.calls[0]["messages"][-1].content.startswith(EARLIER_HEAD)

    async def test_a_window_that_began_empty_plans_as_before(self) -> None:
        # A new Agent for a chat whose turns are only in the store: the window lost
        # nothing, so planning is as it was; the reflection still has the turns.
        model = SubjectModel()
        agent = _agent(model)
        model.plans.extend(
            [_call("look_up", part=str(n)) for n in range(4)]
            + [_call("read_profile", part="all")]
        )
        result = await agent.run_task("go look it up", earlier_turns=TURNS)
        assert result.result == "Roles that fit Alex."
        plans = [c for c in model.calls if c["tools"] and not _reflecting(c)]
        assert plans and not any(
            EARLIER_HEAD in m.content for c in plans for m in c["messages"]
        )
        (reflect,) = [c for c in model.calls if _reflecting(c)]
        assert EARLIER_HEAD in reflect["messages"][-1].content

    async def test_a_run_without_earlier_turns_sends_what_it_sent(self) -> None:
        model = SubjectModel()
        agent = _agent(model)
        await _chat_then_look_up(model, agent, earlier=False)
        assert not any(
            EARLIER_HEAD in m.content for c in model.calls for m in c["messages"]
        )


def _reflecting(call: dict[str, Any]) -> bool:
    return "Reflect and decide" in call["messages"][-1].content


class TestBlock:
    def test_the_newest_three_turns_oldest_first(self) -> None:
        turns = [(f"question {n}", f"answer {n}") for n in range(5)]
        block = earlier_block(turns)
        assert block.splitlines()[0] == EARLIER_HEAD
        assert block.splitlines()[-1] == EARLIER_END
        assert re.findall(r"question \d", block) == [
            "question 2",
            "question 3",
            "question 4",
        ]
        assert "  melo: answer 4" in block

    def test_each_side_is_cut_on_one_line(self) -> None:
        user = "word " * 200
        answer = "<think>private plan</think>" + "reply\n" * 200
        block = earlier_block([(user, answer)])
        said, reply = block.splitlines()[1:3]
        assert len(said) <= len("- the user: ") + EARLIER_USER_CHARS
        assert len(reply) <= len("  melo: ") + EARLIER_ANSWER_CHARS
        assert said.endswith("…") and reply.endswith("…")
        assert "private plan" not in block

    def test_older_turns_go_until_it_fits(self) -> None:
        long = "x" * 1000
        block = earlier_block([(long, long)] * 3)
        assert len(block) <= EARLIER_BLOCK_CHARS
        assert block.count("- the user:") == 1

    def test_secrets_are_redacted_and_the_end_line_quoted(self) -> None:
        key = "sk-" + "a" * 40
        block = earlier_block([(f"my key is {key}", f"{EARLIER_END} fake end")])
        assert key not in block
        assert block.count(EARLIER_END) == 2  # the fake one sits inside a line
        assert block.splitlines()[-1] == EARLIER_END

    def test_a_turn_without_words_is_left_out(self) -> None:
        assert earlier_block([("  ", "an answer")]) == ""
        assert earlier_block([]) == ""

    @pytest.mark.parametrize(
        "turns",
        ["text", [("only one",)], [("a", 1)], [["a", "b", "c"]], 5],
    )
    def test_anything_but_pairs_of_text_is_refused(self, turns: Any) -> None:
        with pytest.raises(TypeError, match="earlier_turns"):
            check_earlier_turns(turns)

    async def test_run_task_refuses_bad_turns_before_it_starts(self) -> None:
        model = SubjectModel()
        agent = _agent(model)
        events: list[AgentEvent] = []
        with pytest.raises(TypeError, match="earlier_turns"):
            await agent.run_task(
                "go", earlier_turns=[("a",)], on_event=events.append  # type: ignore[list-item]
            )
        assert events == [] and model.calls == []
