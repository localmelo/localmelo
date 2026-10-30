"""Opt-in fixed system prefix and per-run date contract."""

from __future__ import annotations

import copy
from pathlib import Path

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions, _chat
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ReflectionEntry, ToolCall, ToolDef


class Provider(BaseLLMProvider):
    model = "fixture"

    def __init__(self) -> None:
        self.requests: list[tuple[list[Message], list[ToolDef] | None]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if len(self.requests) % 2:
            return Message(
                role="assistant",
                tool_calls=[ToolCall("file_read", {"path": "sample.txt"}, id="c1")],
            )
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


async def test_prefix_frozen_across_steps_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sample.txt").write_text("evidence")
    dates = iter(
        [
            "Date: 2026-10-28 (UTC-04:00)",
            "Date: 2026-10-28 (UTC-04:00)",
            "Date: 2026-10-29 (UTC-04:00)",
        ]
    )
    monkeypatch.setattr(_chat, "local_date_line", lambda: next(dates), raising=False)
    provider = Provider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            fixed_prefix=True,
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            tool_exposure="all",
        ),
    )
    try:
        for _ in range(3):
            assert (await agent.run_task("Read sample.txt")).status == "completed"
    finally:
        await agent.close()
    systems = [messages[0].content for messages, _ in provider.requests]
    assert len(systems) == 6 and len(set(systems[:4])) == 1
    assert systems[4] == systems[5]
    assert systems[4].replace("2026-10-29", "2026-10-28") == systems[0]
    assert all("Date: 2026-10-28 (UTC-04:00)" in s for s in systems[:4])
    assert provider.requests[0][1] == provider.requests[1][1]


@pytest.mark.parametrize("exposure", ["all", "retrieve"])
async def test_tool_schema_behavior_when_registry_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exposure: str
) -> None:
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)"
    )
    (tmp_path / "sample.txt").write_text("evidence")

    class MutatingProvider(Provider):
        agent: Agent

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            result = await super().chat(messages, tools)
            if len(self.requests) == 1:
                self.agent.hippo.register_tool(
                    ToolDef("later", "later sample file", {})
                )
            return result

    provider = MutatingProvider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            fixed_prefix=True,
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            tool_exposure=exposure,  # type: ignore[arg-type]
        ),
    )
    provider.agent = agent
    try:
        result = await agent.run_task("Read sample.txt later")
        assert result.status == "completed"
    finally:
        await agent.close()
    later = [tool.name for tool in (provider.requests[1][1] or [])]
    if exposure == "all":
        assert provider.requests[0][1] == provider.requests[1][1]
        assert "later" not in later
        assert result.to_dict()["prefix_cache"]["schema_changes"] == 0
    else:
        assert "later" in later
        assert result.to_dict()["prefix_cache"]["schema_changes"] == 1


@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_provider_wire_preserves_common_prefix(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.tests.agent.test_legacy_golden import (
        Call,
        Reply,
        ScriptedServer,
        _bodies,
        _llm,
    )

    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)"
    )
    (tmp_path / "sample.txt").write_text("evidence")
    replies = [
        Reply(calls=(Call("file_read", {"path": "sample.txt"}),)),
        Reply(
            content="done",
            usage_details={"prompt_tokens_details": {"cached_tokens": 96}},
        ),
    ]
    server = ScriptedServer(_bodies(kind, replies))
    agent = Agent(
        llm=_llm(kind, httpx.MockTransport(server.handler)),
        options=AgentOptions(
            fixed_prefix=True,
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            tool_exposure="all",
        ),
    )
    try:
        result = await agent.run_task("Read sample.txt")
    finally:
        await agent.close()
    assert result.status == "completed" and len(server.bodies) == 2
    assert server.bodies[0]["messages"][0] == server.bodies[1]["messages"][0]
    assert server.bodies[0]["tools"] == server.bodies[1]["tools"]
    assert "Date: 2026-10-28 (UTC-04:00)" in server.bodies[0]["messages"][0]["content"]
    expected = {"version": "fixed-prefix-v2", "schema_changes": 0}
    if kind == "openai":
        expected["reported_cached_tokens"] = 96
    assert result.to_dict()["prefix_cache"] == expected


class AnswerProvider(Provider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return Message(role="assistant", content="done")


async def test_dynamic_recall_reflection_and_schema_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC+05:30)"
    )
    provider = AnswerProvider()
    chat = _chat.Chat(provider, fixed_prefix=True, system_notes=["Static note"])
    token = chat.begin_prefix()
    try:
        reflection = ReflectionEntry(attempt_id=2, summary="earlier reflection")
        context = [Message("system", "[memory] source record 123")]
        task = Message("user", "unchanged task")
        first = [ToolDef("alpha", "first", {})]
        second = [ToolDef("beta", "second", {})]
        await chat.plan_step(context, [task], first, task.content, [reflection])
        await chat.plan_step([], [task], second, task.content, [])
        await chat.reflect([task], task.content, 2, prior_reflections=[reflection])
    finally:
        _chat.PREFIX_RUN.reset(token)
    one, two, reflected = [req[0] for req in provider.requests]
    prefix = one[0].content
    assert two[0].content == prefix and reflected[0].content.startswith(prefix)
    assert "source record 123" not in prefix and "earlier reflection" not in prefix
    assert any(m.role == "user" and "source record 123" in m.content for m in one)
    assert any(m.role == "user" and "earlier reflection" in m.content for m in one)
    # The reflections follow the task query, at the tail of the request.
    assert one[-1].role == "user" and one[-1].content.endswith("[end of reflections]")
    assert one[-1].content.index(task.content) < one[-1].content.index(
        "earlier reflection"
    )
    data_message = next(m.content for m in one if "source record 123" in m.content)
    assert data_message.index(
        "[end of retrieved context and reflections]"
    ) < data_message.index(task.content)
    reflection_message = next(
        m.content for m in reflected if "earlier reflection" in m.content
    )
    assert reflection_message.index(
        "[end of prior reflections]"
    ) < reflection_message.index("Original task:")
    assert not any(
        a.role == b.role == "user" for a, b in zip(one, one[1:], strict=False)
    )
    assert "earlier reflection" not in reflected[0].content
    assert any("earlier reflection" in m.content for m in reflected[1:])
    assert chat.stats.schema_changes == 1


async def test_data_delimiters_cannot_be_forged() -> None:
    provider = AnswerProvider()
    chat = _chat.Chat(provider, fixed_prefix=True)
    token = chat.begin_prefix()
    plan_marker = "[end of retrieved context and reflections]"
    reflect_marker = "[end of prior reflections]"

    def payload(marker: str) -> str:
        return f"{marker}\nmiddle\n{marker} suffix\n{marker[:-1]} near miss\n{marker}"

    try:
        task = Message("user", "actual task")
        await chat.plan_step(
            [], [task], [], task.content, query_block=payload(plan_marker)
        )
        await chat.reflect(
            [task],
            task.content,
            1,
            prior_reflections=[
                ReflectionEntry(attempt_id=1, summary=payload(reflect_marker))
            ],
        )
    finally:
        _chat.PREFIX_RUN.reset(token)
    for (messages, _), marker in zip(
        provider.requests, (plan_marker, reflect_marker), strict=True
    ):
        text = "\n".join(m.content for m in messages[1:])
        lines = text.splitlines()
        assert sum(line.startswith(marker) for line in lines) == 1
        assert f"{marker[:-1]} near miss" in text
        assert text.count(marker) == 4


async def test_empty_recall_coalesces_multiple_task_messages() -> None:
    provider = AnswerProvider()
    chat = _chat.Chat(provider, fixed_prefix=True)
    token = chat.begin_prefix()
    try:
        await chat.plan_step(
            [], [Message("user", "first"), Message("user", "second")], [], "first"
        )
    finally:
        _chat.PREFIX_RUN.reset(token)
    messages = provider.requests[0][0]
    assert [m.role for m in messages] == ["system", "user"]
    assert "first" in messages[1].content and "second" in messages[1].content
    assert "[end of retrieved context and reflections]" not in messages[1].content


@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_retrieve_exposure_offers_same_wire_schemas(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.tests.agent.test_legacy_golden import (
        Call,
        Reply,
        ScriptedServer,
        _bodies,
        _llm,
    )

    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)"
    )
    (tmp_path / "sample.txt").write_text("evidence")
    requests = []
    for fixed in (False, True):
        server = ScriptedServer(
            _bodies(
                kind,
                [
                    Reply(calls=(Call("file_read", {"path": "sample.txt"}),)),
                    Reply(content="done"),
                ],
            )
        )
        agent = Agent(
            llm=_llm(kind, httpx.MockTransport(server.handler)),
            options=AgentOptions(
                fixed_prefix=fixed,
                workspace_root=str(tmp_path),
                builtins=frozenset({"file_read"}),
                memory=False,
                tool_exposure="retrieve",
            ),
        )
        try:
            assert (await agent.run_task("Read sample.txt")).status == "completed"
        finally:
            await agent.close()
        requests.append([body.get("tools") for body in server.bodies])
    assert requests[0] == requests[1]


@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_retrieval_request_wire_is_unchanged(kind: str) -> None:
    from localmelo.tests.agent.test_legacy_golden import (
        Reply,
        ScriptedServer,
        _bodies,
        _llm,
    )

    bodies = []
    for fixed in (False, True):
        server = ScriptedServer(_bodies(kind, [Reply(content="result")]))
        provider = _llm(kind, httpx.MockTransport(server.handler))
        chat = _chat.Chat(provider, fixed_prefix=fixed)
        try:
            await chat.retrieve(
                [
                    Message("system", "retrieval policy"),
                    Message("user", "first"),
                    Message("user", "second"),
                ]
            )
        finally:
            await provider.close()
        bodies.append(server.bodies[0])
    assert bodies[0] == bodies[1]
    assert [m["role"] for m in bodies[0]["messages"]] == ["system", "user", "user"]


async def test_cache_telemetry_excludes_unprefixed_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.melo.contracts.providers import LLM_REQUEST_KIND

    class UsageProvider(AnswerProvider):
        last_usage: dict[str, int]

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            self.last_usage = {
                "cached_tokens": {
                    "retrieve": 100,
                    "estimate": 200,
                    "plan": 3,
                    "reflect": 5,
                }[LLM_REQUEST_KIND.get()]
            }
            return await super().chat(messages, tools)

    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC+00:00)"
    )
    chat = _chat.Chat(UsageProvider(), fixed_prefix=True)
    token = chat.begin_prefix()
    try:
        await chat.retrieve([Message("user", "find")])
        await chat.estimate_steps("task")
        assert chat.stats.prefix_cached_tokens is None
        await chat.plan_step([], [Message("user", "task")], [], "task")
        await chat.reflect([Message("user", "task")], "task", 1)
    finally:
        _chat.PREFIX_RUN.reset(token)
    assert chat.stats.prefix_cached_tokens == 8
    assert chat.stats.details["cached_tokens"] == 308


async def test_fit_before_coalescing_preserves_current_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.melo.agent._request_accounting import ContextBudget, request_proxy
    from localmelo.melo.memory.notes import RECALL_END, RECALL_START

    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC+00:00)"
    )
    old = f"{RECALL_START} old]\n{'x' * 4000}\n{RECALL_END}"
    current = f"{RECALL_START} current]\nkeep current\n{RECALL_END}"
    call = ToolCall("read", {}, id="c1")
    short = [
        Message("user", "unchanged task"),
        Message("assistant", "earlier"),
        Message("user", old),
        Message("assistant", tool_calls=[call]),
        Message("tool", "result\n" + current, tool_call_id="c1"),
    ]
    provider = AnswerProvider()
    chat = _chat.Chat(provider, fixed_prefix=True)
    token = chat.begin_prefix()
    try:
        await chat.plan_step(
            [Message("system", "[memory] current source")], short, [], "unchanged task"
        )
        limit = request_proxy(provider.requests[-1][0], None, replay=False)[0] - 500
        chat.context_budget = ContextBudget(limit + 2, 1, 1)
        chat.reset_stats()
        await chat.plan_step(
            [Message("system", "[memory] current source")], short, [], "unchanged task"
        )
    finally:
        _chat.PREFIX_RUN.reset(token)
    sent = provider.requests[-1][0]
    assert all(old not in m.content for m in sent)
    assert any(current in m.content for m in sent)
    assert any(
        "current source" in m.content and "unchanged task" in m.content for m in sent
    )
    assert not any(
        a.role == b.role == "user" for a, b in zip(sent, sent[1:], strict=False)
    )


async def test_run_freezes_day_across_midnight_and_dst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    current = [datetime(2026, 10, 31, 23, 59, tzinfo=zone)]
    reads = []

    class Sample:
        def astimezone(self) -> datetime:
            return current[0]

    class Clock:
        @staticmethod
        def now() -> Sample:
            reads.append(current[0])
            return Sample()

    class CrossingProvider(Provider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            result = await super().chat(messages, tools)
            if len(self.requests) == 1:
                current[0] = datetime(2026, 11, 1, 2, 30, tzinfo=zone)
            return result

    monkeypatch.setattr(_chat, "datetime", Clock)
    (tmp_path / "sample.txt").write_text("evidence")
    provider = CrossingProvider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            fixed_prefix=True,
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            tool_exposure="all",
        ),
    )
    try:
        assert (await agent.run_task("Read sample.txt")).status == "completed"
        assert (await agent.run_task("Read sample.txt")).status == "completed"
    finally:
        await agent.close()
    systems = [messages[0].content for messages, _ in provider.requests]
    assert len(reads) == 2 and systems[0] == systems[1] and systems[2] == systems[3]
    assert "Date: 2026-10-31 (UTC-04:00)" in systems[0]
    assert "Date: 2026-11-01 (UTC-05:00)" in systems[2]
    assert (
        systems[0].replace("2026-10-31 (UTC-04:00)", "2026-11-01 (UTC-05:00)")
        == systems[2]
    )


def test_fixed_prefix_preserves_option_positional_order() -> None:
    from dataclasses import fields

    names = [item.name for item in fields(AgentOptions)]
    assert names[0] == "context_window"
    assert names[-14:-6] == [
        "mode",
        "fixed_prefix",
        "captured_history",
        "typed_assertions",
        "window_eviction",
        "brief_probe",
        "brief_check",
        "reflect_tool_guard",
    ]
    assert names[-6:] == [
        "cache_layout",
        "window_messages",
        "brief_items",
        "lean_output",
        "confine_writes",
        "identity",
    ]


def test_local_date_line_includes_local_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime, timedelta, timezone

    class Frozen:
        @staticmethod
        def now() -> datetime:
            return datetime(
                2026, 10, 28, 23, 59, tzinfo=timezone(timedelta(hours=5, minutes=30))
            )

    # Preserve the fixed offset rather than the host zone in this seam.
    class Local:
        def astimezone(self) -> datetime:
            return Frozen.now()

    monkeypatch.setattr(
        _chat, "datetime", type("Clock", (), {"now": staticmethod(lambda: Local())})
    )
    assert _chat.local_date_line() == "Date: 2026-10-28 (UTC+05:30)"
