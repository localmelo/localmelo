"""R2's JSON line: memory_stats, usage extras and config keys."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, RunResult, ToolDef


class FakeLLM(BaseLLMProvider):
    """Answers with queued replies and reports the scripted whole usage."""

    reasoning_field = "auto"

    def __init__(
        self,
        replies: list[Message] | None = None,
        *,
        full_usage: dict[str, int] | None = None,
    ) -> None:
        self.replies: deque[Message] = deque(replies or [])
        self.full_usage = full_usage
        self.last_usage: dict[str, int] | None = None

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.last_usage = self.full_usage
        if self.replies:
            return self.replies.popleft()
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


def _options(**kwargs: Any) -> AgentOptions:
    kwargs.setdefault("builtins", frozenset())
    return AgentOptions(**kwargs)


def _result(status: str = "completed", **kwargs: Any) -> RunResult:
    return RunResult(task_id="t" * 32, status=status, **kwargs)


def test_to_dict_omits_memory_stats_when_none() -> None:
    result = _result(result="ok")
    data = result.to_dict()
    assert "memory_stats" not in data
    assert list(data)[-2:] == ["duration_s", "config"]


def test_to_dict_puts_memory_stats_after_config() -> None:
    result = _result(
        result="ok",
        config={"reasoning": "replay"},
        memory_stats={"replayed_chars": 1, "replay_skipped": 2, "replay_stripped": 3},
    )
    data = result.to_dict()
    assert list(data)[-3:] == ["duration_s", "config", "memory_stats"]
    assert data["memory_stats"] == {
        "replayed_chars": 1,
        "replay_skipped": 2,
        "replay_stripped": 3,
    }


def test_to_dict_is_still_a_deep_copy_with_memory_stats() -> None:
    result = _result(memory_stats={"replayed_chars": 1})
    data = result.to_dict()
    data["memory_stats"]["replayed_chars"] = 99
    assert result.memory_stats == {"replayed_chars": 1}


def test_both_run_result_shapes_round_trip() -> None:
    legacy = _result(usage={"total_tokens": 1}, config={"tools": []})
    rich = _result(
        usage={"total_tokens": 1, "reasoning_chars": 4},
        config={"reasoning": "replay", "replay_turns": 2},
        memory_stats={"replayed_chars": 4, "replay_skipped": 0, "replay_stripped": 0},
    )
    assert RunResult(**legacy.to_dict()) == legacy
    assert RunResult(**rich.to_dict()) == rich


@pytest.mark.asyncio
async def test_legacy_usage_keeps_its_three_keys_and_config_its_eleven() -> None:
    llm = FakeLLM(
        [
            Message(
                role="assistant",
                content="ok",
                thinking="not counted",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )
        ],
        full_usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cached_tokens": 5,
            "reasoning_tokens": 1,
        },
    )
    result = await Agent(llm=llm, options=_options()).run_task("q")
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert result.memory_stats is None
    assert list(result.config)[-2:] == ["timeout_s", "tools"]
    assert "reasoning" not in result.config and "replay_turns" not in result.config


@pytest.mark.asyncio
async def test_replay_usage_adds_reasoning_and_reported_details() -> None:
    llm = FakeLLM(
        [
            Message(
                role="assistant",
                content="ok",
                thinking="draft",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )
        ],
        full_usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cached_tokens": 5,
            "reasoning_tokens": 1,
        },
    )
    result = await Agent(
        llm=llm, options=_options(reasoning="replay", replay_turns=3)
    ).run_task("q")
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "reasoning_chars": 5,
        "cached_tokens": 5,
        "reasoning_tokens": 1,
    }
    assert result.memory_stats == {
        "replayed_chars": 0,
        "replay_skipped": 0,
        "replay_stripped": 0,
        "notes": 0,
        "notes_model": 0,
        "notes_heuristic": 0,
        "note_chars": 0,
        "drafts": 0,
        "drafts_written": 0,
        "candidates": 0,
        "remember_model": 0,
        "remember_heuristic": 0,
        "candidates_refused": 0,
        "redactions": 0,
        "rehearsed": {"restated": 0, "acted": 0, "cited": 0},
        "digests": 0,
    }
    assert list(result.config)[-3:] == ["reasoning", "replay_turns", "note_writer"]
    assert (result.config["reasoning"], result.config["replay_turns"]) == (
        "replay",
        3,
    )


@pytest.mark.asyncio
async def test_details_are_absent_when_no_response_reported_them() -> None:
    llm = FakeLLM(
        [
            Message(
                role="assistant",
                content="ok",
                thinking="draft",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        ],
        full_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    result = await Agent(llm=llm, options=_options(reasoning="drop")).run_task("q")
    assert "cached_tokens" not in result.usage
    assert "reasoning_tokens" not in result.usage
    assert result.usage["reasoning_chars"] == 5


@pytest.mark.asyncio
async def test_the_record_serializes_to_one_json_line_for_replay() -> None:
    llm = FakeLLM([Message(role="assistant", content="ok", thinking="draft")])
    result = await Agent(llm=llm, options=_options(reasoning="replay")).run_task("q")
    data = result.to_dict()
    line = json.dumps(data)
    assert "\n" not in line
    assert "memory_stats" in data
