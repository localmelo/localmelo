"""What a run states about itself to a tool that asks (#551).

The browser works through a picture check only in a run the person started,
with a model that reads pictures. Both are the run's own facts, answered by the
question handle every tool call gets, never by a page or a tool result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import TOOL_ASK, AutoPolicy
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.schema import ToolDef
from localmelo.tests.agent.test_approvals import Recorder, _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_mode import FakeModel


class Asks(BaseToolProvider):
    """A tool that notes what its question handle states."""

    def __init__(self) -> None:
        self.facts: list[tuple[bool, bool]] = []

    async def start(self) -> list[ProvidedTool]:
        definition = ToolDef(
            name="facts",
            description="facts",
            parameters={"type": "object", "properties": {}, "required": []},
            annotations={"readOnlyHint": True},
        )
        return [ProvidedTool(definition=definition, function=self._facts)]

    async def _facts(self) -> str:
        ask: Any = TOOL_ASK.get()
        self.facts.append((ask.from_person(), ask.sees_pictures()))
        return "noted"

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("images", "from_person", "expected"),
    [
        ("on", True, (True, True)),
        ("on", False, (False, True)),  # a chore, a follow-up or a task's own
        ("off", True, (True, False)),
        ("auto", True, (True, False)),  # a model that is not known to read pictures
    ],
)
async def test_a_run_states_who_started_it_and_whether_its_model_reads_pictures(
    tmp_path: Path, images: str, from_person: bool, expected: tuple[bool, bool]
) -> None:
    tool = Asks()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = Agent(
        llm=FakeModel([_calls(("facts", {}, "c1")), _answer("done")]),
        options=AgentOptions(
            builtins=frozenset(),
            workspace_root=str(workspace),
            tool_images=images,  # type: ignore[arg-type]
        ),
        tool_providers=[tool],
    )
    agent.chat.sleep = _no_wait
    await agent.run_task(
        "Check in",
        on_event=Recorder(),
        approval=AutoPolicy(),
        from_person=from_person,
    )
    assert tool.facts == [expected]
