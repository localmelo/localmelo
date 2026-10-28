"""T62 end to end: a note's fact names the earlier result that established it.

The stub server plays a real ``Agent`` run: turn 1 returns records, turn 2's
note quotes one of them. Turn 2 either calls a tool whose result carries no
records or answers without a call. The run's remember request admits its
notes at the end, and the entries and decisions must name the turn-1 record
and never the unrelated one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import ToolDef
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.fixtures.stub_llm import StubServer

FACT = "Yoga starts 2026-04-11T09:00:00-04:00 at Studio Verde."
UNRELATED = "The library closes late on Friday."
QUERY = "Remember that the studio schedule is important. Check the studio schedule."


def _records() -> str:
    return json.dumps(
        [
            {"record_id": "rec-yoga", "text": FACT},
            {"record_id": "rec-unrelated", "text": UNRELATED},
        ]
    )


def _scenario(*, answer_only: bool) -> dict[str, Any]:
    note = f"<note>found: {FACT}\nremember: yes</note>"
    replies: list[dict[str, Any]] = [
        {"tool_calls": [{"name": "records", "arguments": {"records": _records()}}]}
    ]
    if answer_only:
        replies.append({"content": f"{note} The studio schedule is clear."})
    else:
        replies.append(
            {
                "content": note,
                "tool_calls": [{"name": "records", "arguments": {"records": "[]"}}],
            }
        )
        replies.append({"content": "The studio schedule is clear."})
    return {"models": ["stub-model"], "sequential": True, "replies": replies}


@pytest.fixture
def stub(request: pytest.FixtureRequest) -> Iterator[StubServer]:
    answer_only = bool(getattr(request, "param", False))
    server = StubServer(
        {"source_ids": _scenario(answer_only=answer_only)}, default="source_ids"
    ).start()
    try:
        yield server
    finally:
        server.close()


def _agent(stub: StubServer, memory_dir: Path) -> Agent:
    agent = Agent(
        llm=OpenAICompatLLM(base_url=stub.base_url, model="stub-model"),
        options=AgentOptions(
            reasoning="compress",
            builtins=frozenset(),
            attempt_policy="none",
            tool_exposure="all",
            retriever="off",
            max_steps=6,
            memory_dir=str(memory_dir),
        ),
    )

    async def _records(records: str) -> str:
        return records

    agent.hippo.register_tool(
        ToolDef(
            name="records",
            description="return the records it is given",
            parameters={
                "type": "object",
                "properties": {"records": {"type": "string"}},
                "required": ["records"],
            },
        )
    )
    agent.executor.register("records", _records)
    return agent


async def _run(
    stub: StubServer, memory_dir: Path
) -> tuple[list[Any], list[dict[str, Any]]]:
    agent = _agent(stub, memory_dir)
    try:
        result = await agent.run_task(QUERY)
        assert result.status == "completed", result
        entries = await agent.hippo.long.active_entries()
        decisions = await agent.hippo.long.decisions()
        return entries, decisions
    finally:
        await agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stub", [False, True], indirect=True)
async def test_a_fact_names_the_turn_one_record(
    stub: StubServer, tmp_path: Path
) -> None:
    entries, decisions = await _run(stub, tmp_path / "memory")
    facts = [entry for entry in entries if FACT in entry.text]
    assert facts, [entry.text for entry in entries]
    for entry in facts:
        assert entry.provenance["source_ids"] == ["rec-yoga"], entry.text
    fact_decisions = [
        decision for decision in decisions if "rec-yoga" in decision["source_ids"]
    ]
    assert fact_decisions, decisions
    assert not any("rec-unrelated" in decision["source_ids"] for decision in decisions)
    assert all(
        "rec-unrelated" not in entry.provenance["source_ids"] for entry in entries
    )
