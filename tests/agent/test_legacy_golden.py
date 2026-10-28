"""Golden fixtures of legacy runs on the wire — no real model.

T-COMPAT (``test_phase1a_compat.py``) fixes what scripted runs send and keep
at the level of ``Message`` objects. These goldens add what it does not
cover: the request bodies the real chat providers put on the wire, and the
usage a server reports beyond the three summed keys. Every run here goes
through ``Agent.run_task`` as direct mode runs it, without an event sink or
an approval policy, through the real :class:`OpenAICompatLLM` or
:class:`OllamaNativeChat` against an ``httpx.MockTransport`` server, with
SQLite history in a temporary ``memory_dir``. Its golden file,
``goldens/legacy/<run>.json``, holds:

- ``result``: ``RunResult.to_dict()`` as a whole, except ``duration_s``;
- ``requests``: every request body as sent (``json.loads`` of its
  content), reflection requests included. Headers are not recorded;
- ``window``: working memory once the run is over, by ``Message`` field;
- ``history``: the ``tasks`` and ``steps`` rows of ``history.db``, read by
  column name.

The documents are built and normalized with T-COMPAT's serializer and
compared with its ``check_golden``, so the same rules apply: generated ids
become ordinals, timestamps and durations are dropped,
``config.memory_dir`` becomes ``<memory_dir>``, a long text keeps its start,
length and digest, and ``LOCALMELO_RECORD_GOLDENS=1`` writes the files
instead of comparing them.

The goldens were recorded at ade1513, which holds C1, the reflection gate
with its default ``reflection_gate="flags"``, U1a (#37), the web workspace
(#35) and R3 (#38), before any change of R1. Every later change must leave
them as they are.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.agent.test_phase1a_compat import (
    Normalizer,
    check_golden,
    history_json,
    result_json,
    window_json,
)

GOLDEN_DIR = Path(__file__).parent / "goldens" / "legacy"

# ── The scripted server ─────────────────────────────────────────────────────


class ScriptedServer:
    """Answers each chat request with the next scripted body; keeps each body.

    A request body is kept as ``json.loads`` of its content, as the provider
    sent it.
    """

    def __init__(self, replies: Iterable[dict[str, Any]]) -> None:
        self.replies: deque[dict[str, Any]] = deque(replies)
        self.bodies: list[Any] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("no reply is scripted for this request")
        return httpx.Response(200, json=self.replies.popleft())


@dataclass(frozen=True)
class Call:
    """One tool call of a scripted reply."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    """One scripted model reply, written out per provider by :func:`_body`."""

    content: str = ""
    thinking: str = ""
    calls: tuple[Call, ...] = ()
    prompt_tokens: int = 100
    completion_tokens: int = 10
    # Usage fields an OpenAI-compatible server adds, such as
    # prompt_tokens_details; Ollama's native API has none.
    usage_details: dict[str, Any] = field(default_factory=dict)


def _openai_body(reply: Reply, call_ids: Iterable[str]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.thinking:
        message["reasoning_content"] = reply.thinking
    if reply.calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call, call_id in zip(reply.calls, call_ids, strict=True)
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if reply.calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            **reply.usage_details,
        },
    }


def _ollama_body(reply: Reply) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.thinking:
        message["thinking"] = reply.thinking
    if reply.calls:
        # The native API sends no call ids: the provider generates them.
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in reply.calls
        ]
    return {
        "model": "stub-model",
        "message": message,
        "done": True,
        "prompt_eval_count": reply.prompt_tokens,
        "eval_count": reply.completion_tokens,
    }


def _bodies(provider: str, replies: Iterable[Reply]) -> list[dict[str, Any]]:
    """The server's bodies for *replies*; OpenAI call ids are unique per run."""
    bodies: list[dict[str, Any]] = []
    count = 0
    for reply in replies:
        if provider == "ollama":
            bodies.append(_ollama_body(reply))
            continue
        ids = [f"call_{count + i + 1}" for i in range(len(reply.calls))]
        count += len(reply.calls)
        bodies.append(_openai_body(reply, ids))
    return bodies


def _llm(provider: str, transport: httpx.MockTransport) -> BaseLLMProvider:
    if provider == "openai":
        return OpenAICompatLLM(
            base_url="http://llm.test/v1", model="stub-model", transport=transport
        )
    return OllamaNativeChat(
        base_url="http://ollama.test", model="stub-model", transport=transport
    )


# ── Replies ─────────────────────────────────────────────────────────────────


def _python(code: str, *, content: str = "", thinking: str = "") -> Reply:
    return Reply(
        content=content,
        thinking=thinking,
        calls=(Call("python_exec", {"code": code}),),
    )


def _reflection(action: str, *, thinking: str = "", **fields: Any) -> Reply:
    """A reflection reply with every key REFLECTION_PROMPT asks for."""
    data: dict[str, Any] = {
        "summary": "Ran the code once.",
        "failed_hypotheses": [],
        "useful_evidence": [],
        "recommended_avoids": [],
        "next_promising_directions": [],
        "progress_made": True,
        "task_still_feasible": True,
        "new_information_gained": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "recommended_action": action,
        "rationale": "",
        "best_effort_result": "",
        "estimated_info_gain": 0.8,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    data.update(fields)
    return Reply(
        content=json.dumps(data),
        thinking=thinking,
        prompt_tokens=900,
        completion_tokens=80,
    )


# More than ASSISTANT_TEXT_SHARE (2,000) characters, so the turn's text gets
# no share of the memory budget beside its result.
LONG_DRAFT = "Here is a long draft of the reasoning behind the answer. " * 45

# A result that the executor cuts to MAX_OUTPUT_LEN (50,000): with the long
# draft, the turn passes MAX_MEMORY_TEXT_LEN, so working memory keeps neither
# its text nor its thinking.
LONG_OUTPUT_CODE = "print('x' * 60000)"


@dataclass
class Scenario:
    """One scripted run: options, the query and the model's replies.

    *status* is the status the run must end with.
    """

    status: str
    query: str
    replies: list[Reply]
    options: dict[str, Any] = field(default_factory=dict)


def _tools_empty_cut_answer() -> Scenario:
    return Scenario(
        status="completed",
        query="What is six times seven? Check it with Python.",
        replies=[
            _python(
                "print(6 * 7)",
                content="Computing it.",
                thinking="I compute six times seven with Python first.",
            ),
            Reply(prompt_tokens=180, completion_tokens=0),
            _python(
                LONG_OUTPUT_CODE,
                content=LONG_DRAFT,
                thinking="A long draft, and a long output to check it against.",
            ),
            Reply(
                content="Six times seven is 42.",
                thinking="Python printed 42.",
                prompt_tokens=400,
                completion_tokens=12,
            ),
        ],
    )


def _reflection_continues() -> Scenario:
    return Scenario(
        status="completed",
        query="What is 2 to the power of 10? Check it with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python(
                "print(2 ** 10)",
                content="Computing it.",
                thinking="Python gives the exact power.",
            ),
            _reflection(
                "continue",
                thinking="The output is known; one more step answers.",
                summary="Python printed 1024.",
                next_promising_directions=["Answer with 1024."],
                rationale="The answer is known.",
            ),
            Reply(
                content="2 to the power of 10 is 1024.",
                thinking="The reflection says the answer is 1024.",
            ),
        ],
    )


def _reflection_stops() -> Scenario:
    # A continue whose next step is not novel: the gate stops the run.
    return Scenario(
        status="gate_stopped",
        query="Find the largest prime below 10 with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python("print(7)", content="Trying a guess.", thinking="7 is prime."),
            _reflection(
                "continue",
                thinking="Running the same code again would not help.",
                summary="Printed 7 without checking it.",
                next_step_is_novel=False,
                rationale="Running the same code again would not help.",
                best_effort_result="7",
            ),
        ],
    )


def _reflection_done() -> Scenario:
    return Scenario(
        status="completed",
        query="What is 2 + 2? Check it with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python(
                "print(2 + 2)",
                content="Let me compute it.",
                thinking="A one-line program answers it.",
            ),
            _reflection(
                "done",
                thinking="Python printed 4, so the task is done.",
                summary="Python printed 4.",
                rationale="The code printed the answer.",
                best_effort_result="2 + 2 = 4",
            ),
        ],
    )


# Usage as an OpenAI-compatible server reports it for a reasoning model with
# a prefix cache.
USAGE_DETAILS: dict[str, Any] = {
    "prompt_tokens_details": {"cached_tokens": 96},
    "completion_tokens_details": {"reasoning_tokens": 28},
}


def _usage_details() -> Scenario:
    return Scenario(
        status="completed",
        query="List the files with Python.",
        replies=[
            Reply(
                thinking="I list the directory first.",
                calls=(Call("python_exec", {"code": "print('notes.txt')"}),),
                prompt_tokens=120,
                completion_tokens=40,
                usage_details=USAGE_DETAILS,
            ),
            Reply(
                content="The directory holds notes.txt.",
                thinking="The listing shows one file.",
                prompt_tokens=200,
                completion_tokens=30,
                usage_details=USAGE_DETAILS,
            ),
        ],
    )


SCENARIOS: dict[str, tuple[str, Callable[[], Scenario]]] = {
    "openai_tools_empty_cut_answer": ("openai", _tools_empty_cut_answer),
    "ollama_tools_empty_cut_answer": ("ollama", _tools_empty_cut_answer),
    "openai_reflection_continues": ("openai", _reflection_continues),
    "ollama_reflection_continues": ("ollama", _reflection_continues),
    "openai_reflection_stops": ("openai", _reflection_stops),
    "ollama_reflection_stops": ("ollama", _reflection_stops),
    "openai_reflection_done": ("openai", _reflection_done),
    "ollama_reflection_done": ("ollama", _reflection_done),
    "openai_usage_details": ("openai", _usage_details),
}


async def record_run(
    provider: str,
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any]]:
    """Run *scenario* on *provider* and return its RunResult and document."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    memory_dir = str(tmp_path / "memory")
    server = ScriptedServer(_bodies(provider, scenario.replies))
    agent = Agent(
        llm=_llm(provider, httpx.MockTransport(server.handler)),
        options=AgentOptions(memory_dir=memory_dir, **scenario.options),
    )
    try:
        result = await agent.run_task(scenario.query)
        window = window_json(agent.hippo.working.get_window())
    finally:
        await agent.close()
    # Every scripted reply was used: the script and the run agree.
    assert not server.replies, f"replies left: {list(server.replies)}"

    history = history_json(Path(memory_dir) / "history.db")
    normalize = Normalizer(
        tasks=[result.task_id, *(row["task_id"] for row in history["tasks"])],
        steps=[row["step_id"] for row in history["steps"]],
    )
    document = normalize(
        {
            "result": result_json(result, memory_dir),
            "requests": server.bodies,
            "window": window,
            "history": history,
        }
    )
    return result, document


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_legacy_run_matches_its_golden(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, build = SCENARIOS[name]
    scenario = build()
    result, document = await record_run(provider, scenario, tmp_path, monkeypatch)
    assert (result.status, result.config["reflection_gate"]) == (
        scenario.status,
        "flags",
    )
    check_golden(GOLDEN_DIR / f"{name}.json", document)


async def test_default_two_step_wire_and_result_remain_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, document = await record_run(
        "openai", _usage_details(), tmp_path, monkeypatch
    )
    assert len(document["requests"]) == 2
    assert "context_budget" not in result.to_dict()
    check_golden(GOLDEN_DIR / "openai_usage_details.json", document)


def test_every_legacy_golden_file_has_its_run() -> None:
    assert sorted(path.stem for path in GOLDEN_DIR.glob("*.json")) == sorted(SCENARIOS)
