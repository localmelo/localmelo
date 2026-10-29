"""The legacy freeze (R1): what recording model calls leaves as it was.

The goldens fix most of it against R1's base: T-COMPAT
(``test_phase1a_compat.py``) the requests, RunResults, windows and tasks
and steps rows of scripted runs, and ``test_legacy_golden.py`` the request
bodies on the wire and the usage when a server reports cached and
reasoning tokens. The tests here pin the rest: the one new header, the
fields of RunResult and Message, the history calls and their order, and
the CLI's flags and exit codes. A failed row, the bench's two queries and
a newer schema are in ``test_llm_calls_recording.py`` and
``tests/memory/test_llm_calls.py``.
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.history import History
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import Message, RunResult, StepRecord, TaskRecord
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

# The headers of a chat request at R1's base (ade1513), by name.
BASE_HEADERS = {
    "accept",
    "accept-encoding",
    "connection",
    "content-length",
    "content-type",
    "host",
    "user-agent",
    "x-localmelo-attempt",
}

# The CLI's option strings at R1's base.
BASE_OPTIONS = [
    "--attempt-policy",
    "--base-url",
    "--builtins",
    "--chat-model",
    "--daemon",
    "--embedding-base-url",
    "--embedding-model",
    "--help",
    "--host",
    "--max-steps",
    "--mcp-config",
    "--mcp-only",
    "--memory",
    "--memory-dir",
    "--output-format",
    "--port",
    "--reasoning",
    "--reasoning-field",
    "--reconfigure",
    "--reflection-gate",
    "--replay-turns",
    "--request-timeout",
    "--serve",
    "--timeout",
    "--tool-exposure",
    "-h",
]

# Options added after R1, each by a stage whose spec calls for it: P3's
# progress line (progress_spec.md section 11, "New flags only"), U2a.1's
# interactive UI and U2b's session selection (u2_spec.md section 1, "Flags"),
# R4's note writer (r4_spec.md Requirement 12), R5's recall flags, R6-lite's
# admission flags, and U3 PR 4's unconfigured gateway boot (app_spec.md
# "The lead's decisions" item 3). P5 moved --workspace from the interactive
# UI to direct mode; the list is the same. #108 adds four opt-in worker
# recording/import options; the legacy exit-code assertions stay unchanged.
LATER_OPTIONS = [
    "--progress",
    "--routine",
    "--workspace",
    "--yolo",
    "-c",
    "--continue",
    "-r",
    "--resume",
    "--note-writer",
    "--retriever",
    "--recall-at",
    "--recall-k",
    "--recall-kinds",
    "--recall-placement",
    "--recall-min-score",
    "--retrieval-extra-body",
    "--admit-on-outcome",
    "--keep-decision-text",
    "--allow-unconfigured",
    "--save-session",
    "--brief",
    "--inbox",
    "--inbox-fd",
    "--parent-run",
    "--import-session",
    "--title",
    "--fixed-prefix",
    "--window-eviction",
    "--captured-history",
    "--context-window",
    "--output-reserve",
    "--safety-margin",
]


def _openai_reply(content: str = "", call: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "reasoning_content": "Thinking it over.",
    }
    if call:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "arguments": json.dumps({"code": "print(42)"}),
                },
            }
        ]
    return {
        "choices": [{"message": message}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }


def _ollama_reply(content: str = "", call: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "thinking": "Thinking it over.",
    }
    if call:
        message["tool_calls"] = [
            {"function": {"name": "python_exec", "arguments": {"code": "print(42)"}}}
        ]
    return {"message": message, "prompt_eval_count": 100, "eval_count": 20}


REPLIES: dict[str, Callable[..., dict[str, Any]]] = {
    "openai": _openai_reply,
    "ollama": _ollama_reply,
}


async def _wire_run(
    provider: str, memory_dir: Path | None
) -> tuple[RunResult, list[tuple[set[str], Any]]]:
    """A tool step and an answer; returns each request's header names and body."""
    reply = REPLIES[provider]
    replies = deque([reply(call=True), reply("42")])
    seen: list[tuple[set[str], Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((set(request.headers.keys()), json.loads(request.content)))
        return httpx.Response(200, json=replies.popleft())

    transport = httpx.MockTransport(handler)
    llm: BaseLLMProvider
    if provider == "openai":
        llm = OpenAICompatLLM(
            base_url="http://llm.test/v1", model="m", transport=transport
        )
    else:
        llm = OllamaNativeChat(
            base_url="http://ollama.test", model="m", transport=transport
        )
    options = AgentOptions(memory_dir=None if memory_dir is None else str(memory_dir))
    agent = Agent(llm=llm, options=options)
    try:
        result = await agent.run_task("What is six times seven?")
    finally:
        await agent.close()
    return result, seen


@pytest.mark.parametrize("provider", ["openai", "ollama"])
async def test_the_only_change_on_the_wire_is_the_kind_header(
    tmp_path: Path, provider: str
) -> None:
    recorded, with_rows = await _wire_run(provider, tmp_path)
    plain, without_rows = await _wire_run(provider, None)
    assert [headers for headers, _ in with_rows] == [
        BASE_HEADERS | {"x-localmelo-call-kind"}
    ] * 2
    # Recording changes no body; the legacy golden compares them with the base.
    assert [body for _, body in with_rows] == [body for _, body in without_rows]
    # RunResult.usage keeps its three keys, also with cached and reasoning
    # tokens reported.
    assert (
        recorded.usage
        == plain.usage
        == {
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "total_tokens": 240,
        }
    )


def test_run_result_and_message_keep_their_fields() -> None:
    assert [
        f.name
        for f in dataclasses.fields(RunResult)
        if f.name not in ("memory_stats", "context_budget", "prefix_cache")
    ] == [
        "task_id",
        "status",
        "result",
        "error",
        "steps",
        "attempts",
        "n_llm",
        "n_main",
        "n_aux",
        "n_tool",
        "llm_errors",
        "usage",
        "duration_s",
        "config",
    ]
    assert [f.name for f in dataclasses.fields(Message)] == [
        "role",
        "content",
        "tool_calls",
        "tool_result",
        "tool_call_id",
        "images",
        "timestamp",
        "usage",
        "thinking",
    ]


async def test_run_result_config_keeps_its_keys(tmp_path: Path) -> None:
    result, _ = await _wire_run("openai", tmp_path)
    assert list(result.config) == [
        "max_steps",
        "attempt_policy",
        "steps_per_attempt",
        "max_attempts",
        "reflection_gate",
        "tool_exposure",
        "mode",
        "builtins",
        "memory",
        "memory_dir",
        "timeout_s",
        "tools",
    ]


class LoggingHistory(History):
    """In-memory history that logs each save_task and add_step."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    async def save_task(self, task: TaskRecord) -> None:
        self.calls.append(("save_task", task.status))
        await super().save_task(task)

    async def add_step(self, task_id: str, step: StepRecord) -> None:
        self.calls.append(("add_step", step.thought))
        await super().add_step(task_id, step)


class LoggingSqliteHistory(SqliteHistory):
    """SqliteHistory that logs each save_task and add_step."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.calls: list[tuple[str, str]] = []

    async def save_task(self, task: TaskRecord) -> None:
        self.calls.append(("save_task", task.status))
        await super().save_task(task)

    async def add_step(self, task_id: str, step: StepRecord) -> None:
        self.calls.append(("add_step", step.thought))
        await super().add_step(task_id, step)


async def test_history_gets_the_same_calls_in_the_same_order(tmp_path: Path) -> None:
    async def run(history: History) -> list[tuple[str, str]]:
        replies = deque([_openai_reply("Computing.", call=True), _openai_reply("42")])
        llm = OpenAICompatLLM(
            base_url="http://llm.test/v1",
            model="m",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=replies.popleft())
            ),
        )
        agent = Agent(llm=llm)
        await agent.hippo.aclose()
        agent.hippo.history = history
        try:
            await agent.run_task("What is six times seven?")
        finally:
            await agent.close()
        calls: list[tuple[str, str]] = getattr(history, "calls")  # noqa: B009
        return calls

    in_memory = await run(LoggingHistory())
    persistent = await run(LoggingSqliteHistory(tmp_path / "history.db"))
    assert (
        persistent
        == in_memory
        == [
            ("save_task", "running"),
            ("add_step", "Computing."),
            ("save_task", "completed"),
        ]
    )


def test_no_new_flag_and_the_same_exit_codes() -> None:
    """R1 adds no flag: the options are R1's base and those of later stages."""
    assert sorted(cli._build_parser()._option_string_actions) == sorted(
        BASE_OPTIONS
        + LATER_OPTIONS
        + ["--gateway-login", "--gateway-rotate-token", "--enable-playground"]
    )
    ui_or_help = {
        "--yolo",
        "-c",
        "--continue",
        "-r",
        "--resume",
        "--help",
        "-h",
    }
    assert set(cli._build_parser(direct_only=True)._option_string_actions) == (
        set(BASE_OPTIONS + LATER_OPTIONS) - ui_or_help
    )
    assert (cli.EXIT_COMPLETED, cli.EXIT_CRASHED, cli.EXIT_NOT_COMPLETED) == (0, 1, 2)
