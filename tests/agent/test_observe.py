"""run_task(observe=...): a sink that follows a run without making it interactive.

An observer receives the events ``on_event`` would, the exit-code slot and
the retry notices included, but a cancellation still ends the run
``crashed``, as for a run without a sink (decision L-11). Also
``BaseLLMProvider.model``, the model id the procedural store keys on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from localmelo.melo.agent._chat import RETRY_HOOK
from localmelo.melo.contracts.approvals import AllowAllPolicy
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.executor import TOOL_EXIT_CODE
from localmelo.melo.schema import Message, ToolDef
from localmelo.tests.agent.procedural_helpers import (
    Recorder,
    ScriptedLLM,
    Tools,
    answer,
    call,
    make_agent,
    read,
)


def _busy() -> ProviderError:
    return ProviderError("HTTP 503: busy", status_code=503, retryable=True)


def _comparable(events: list[AgentEvent]) -> list[tuple[str, dict[str, Any]]]:
    timed = {"duration_s", "result"}
    return [
        (e.type, {k: v for k, v in e.data.items() if k not in timed}) for e in events
    ]


async def test_an_observer_receives_what_on_event_does() -> None:
    script: list[Message | BaseException] = [_busy(), read("/p/a.py"), answer("done")]
    followed, observed = Recorder(), Recorder()
    first = make_agent(ScriptedLLM(script))
    first.chat.retry_delays = (0.0,)
    await first.run_task("read a.py", on_event=followed)
    second = make_agent(ScriptedLLM(script))
    second.chat.retry_delays = (0.0,)
    result = await second.run_task("read a.py", observe=observed)
    assert result.status == "completed"
    assert _comparable(observed.events) == _comparable(followed.events)
    assert "notice" in observed.types  # the retry notice


async def test_an_observer_gets_the_exit_code_slot_and_the_retry_hook() -> None:
    tools = Tools()
    agent = make_agent(
        ScriptedLLM([read("/a"), answer("a"), read("/b"), answer("b")]), tools=tools
    )
    await agent.run_task("read a")
    await agent.run_task("read b", observe=Recorder())
    assert tools.slots == [None, []]
    assert tools.hooks[0] is None and tools.hooks[1] is not None
    assert RETRY_HOOK.get() is None
    assert TOOL_EXIT_CODE.get() is None


async def _cancel_in_a_tool(
    run: Callable[..., Coroutine[Any, Any, Any]], tools: Tools, **kwargs: Any
) -> None:
    running: asyncio.Task[Any] = asyncio.create_task(run("hang", **kwargs))
    await asyncio.wait_for(tools.hanging.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_a_cancelled_observed_run_ends_crashed() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools)
    await _cancel_in_a_tool(agent.run_task, tools, observe=recorder)
    assert recorder.types[-1] == "run.finished"
    result = recorder.events[-1].data["result"]
    assert (result["status"], result["error"]) == ("crashed", "CancelledError")
    finished = recorder.of("tool.finished")[-1].data
    assert finished["status"] == "cancelled"
    task = await agent.hippo.history.get_task(result["task_id"])
    assert task is not None and task.status == "crashed"


async def test_a_cancelled_followed_run_still_ends_cancelled() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools)
    await _cancel_in_a_tool(agent.run_task, tools, on_event=recorder)
    assert recorder.events[-1].data["result"]["status"] == "cancelled"


async def test_an_observed_run_with_a_policy_is_interactive() -> None:
    tools, recorder = Tools(), Recorder()
    agent = make_agent(ScriptedLLM([call("hang")]), tools=tools)
    await _cancel_in_a_tool(
        agent.run_task, tools, observe=recorder, approval=AllowAllPolicy()
    )
    assert recorder.events[-1].data["result"]["status"] == "cancelled"


async def test_observe_and_on_event_together_are_refused() -> None:
    llm = ScriptedLLM([answer("never")])
    agent = make_agent(llm)
    recorder = Recorder()
    with pytest.raises(TypeError, match="on_event or observe"):
        await agent.run_task("hi", on_event=recorder, observe=recorder)
    assert (recorder.events, llm.requests) == ([], [])
    assert agent.hippo.history._tasks == {}  # type: ignore[attr-defined]


@pytest.mark.parametrize("bad", [42, "sink"])
async def test_observe_must_be_callable(bad: object) -> None:
    agent = make_agent(ScriptedLLM([]))
    with pytest.raises(TypeError, match="observe must be a callable"):
        await agent.run_task("hi", observe=bad)  # type: ignore[arg-type]


async def test_observe_must_be_synchronous() -> None:
    async def sink(event: AgentEvent) -> None:
        pass

    agent = make_agent(ScriptedLLM([]))
    with pytest.raises(TypeError, match="observe must be synchronous"):
        await agent.run_task("hi", observe=sink)


class NoModel(BaseLLMProvider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return answer("hi")

    async def close(self) -> None:
        pass


def test_a_provider_without_a_model_id_reports_an_empty_one() -> None:
    assert BaseLLMProvider.model == ""
    assert NoModel().model == ""


def test_the_built_in_chat_providers_report_their_model() -> None:
    from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    openai = OpenAICompatLLM(base_url="http://127.0.0.1:9/v1", model="qwen3-8b")
    ollama = OllamaNativeChat(base_url="http://127.0.0.1:9", model="llama3")
    assert (openai.model, ollama.model) == ("qwen3-8b", "llama3")
