"""``AgentOptions.thinking_steps``: which planning steps the model thinks on (#551).

The default sends every request as the provider's own fields say. ``"first_and_failed"``
sends the plan that follows a tool step that worked with the provider's thinking switch
(``thinking_off_body``), and keeps thinking for the first plan and the plan after a step
with an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import LLM_REQUEST_EXTRA_BODY
from localmelo.melo.schema import Message, ToolDef
from localmelo.tests.agent.test_approvals import _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_mode import FakeModel

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


class Recording(FakeModel):
    """A scripted model that notes the request fields each planning call carried."""

    def __init__(self, plans: list[Message], switch: Mapping[str, Any]) -> None:
        super().__init__(plans)
        self.thinking_off_body = MappingProxyType(dict(switch))
        self.bodies: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.bodies.append(dict(LLM_REQUEST_EXTRA_BODY.get()))
        return await super().chat(messages, tools)


async def bodies_of(
    tmp_path: Path, steps: str, *, switch: Mapping[str, Any] = MappingProxyType(OFF)
) -> list[dict[str, Any]]:
    """The fields of each planning call of a run whose second tool step fails."""
    llm = Recording(
        [
            _calls(("python_exec", {"code": "ok"}, "c1")),
            _calls(("python_exec", {"code": "boom"}, "c2")),
            _calls(("python_exec", {"code": "ok"}, "c3")),
            _answer("done"),
        ],
        switch,
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset({"python_exec"}),
            tool_exposure="all",
            workspace_root=str(workspace),
            thinking_steps=steps,  # type: ignore[arg-type]
        ),
    )
    ran: list[str] = []

    async def python_exec(code: str) -> str:
        ran.append(code)
        if code == "boom":
            raise RuntimeError("boom")
        return "ok"

    agent.executor.register("python_exec", python_exec, requires_approval=False)
    agent.chat.sleep = _no_wait
    await agent.run_task("run it")
    assert ran == ["ok", "boom", "ok"]
    return llm.bodies


async def test_the_default_thinks_on_every_step(tmp_path: Path) -> None:
    assert await bodies_of(tmp_path, "all") == [{}, {}, {}, {}]


async def test_only_the_plan_after_a_step_that_worked_goes_out_with_thinking_off(
    tmp_path: Path,
) -> None:
    # First plan thinks; after ok: off; after the failed step: thinks; after ok: off.
    assert await bodies_of(tmp_path, "first_and_failed") == [{}, OFF, {}, OFF]


async def test_a_provider_with_no_switch_thinks_on_every_step_either_way(
    tmp_path: Path,
) -> None:
    assert await bodies_of(tmp_path, "first_and_failed", switch={}) == [{}] * 4


def test_any_other_value_is_refused() -> None:
    with pytest.raises(ValueError, match="thinking_steps must be one of"):
        AgentOptions(thinking_steps="never")  # type: ignore[arg-type]
