"""``AgentOptions.window_eviction``: the window's eviction, the cap fit and the context fit.

``slide`` is today's behaviour. ``chunk`` and ``boundary`` keep the retained
history stable for several steps so a server's prompt cache can reuse it.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._context_fit import fit_request
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.short import WorkingMemory
from localmelo.melo.schema import (
    SHORT_TERM_MAX,
    WINDOW_EVICT_CHUNK,
    WINDOW_FIT_TARGET,
    Message,
    ToolCall,
    ToolDef,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

REPO = Path(__file__).resolve().parents[2]
ECHO = ToolDef(
    name="echo",
    description="Echo text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)


def _turn(n: int, text: str = "") -> tuple[Message, list[Message]]:
    call = ToolCall("echo", {"text": f"step {n}"}, id=f"call{n}")
    return (
        Message("assistant", tool_calls=[call]),
        [Message("tool", text or f"result {n}", tool_call_id=call.id)],
    )


def _fill(working: WorkingMemory, turns: int) -> list[int]:
    """Append a query and *turns* turns; the window's size after each append."""
    sizes = []
    working.append(Message("user", "query"))
    for n in range(turns):
        working.append_turn(*_turn(n))
        sizes.append(len(working.get_window()))
    return sizes


def _whole_units(window: list[Message]) -> bool:
    calls = {c.id for m in window for c in m.tool_calls}
    answered = {m.tool_call_id for m in window if m.role == "tool"}
    first_ok = not window or window[0].role != "tool"
    return calls == answered and first_ok


# ── WorkingMemory ──


def test_slide_drops_one_unit_per_step_once_full() -> None:
    sizes = _fill(WorkingMemory(SHORT_TERM_MAX), 14)
    assert sizes[:9] == [3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert sizes[9:] == [SHORT_TERM_MAX] * 5  # one unit leaves per step


def test_chunk_drops_down_to_max_minus_k_and_keeps_units_whole() -> None:
    working = WorkingMemory(SHORT_TERM_MAX, evict_chunk=WINDOW_EVICT_CHUNK)
    sizes = _fill(working, 30)
    assert max(sizes) <= SHORT_TERM_MAX
    drops = [i for i in range(1, len(sizes)) if sizes[i] < sizes[i - 1]]
    for i in drops:
        assert sizes[i] <= SHORT_TERM_MAX - WINDOW_EVICT_CHUNK
        assert sizes[i] > SHORT_TERM_MAX - WINDOW_EVICT_CHUNK - 2  # by one unit
    # Dropping happens at most once per K/2 steps.
    assert all(
        b - a >= WINDOW_EVICT_CHUNK // 2 for a, b in zip(drops, drops[1:], strict=False)
    )
    assert len(drops) >= 3
    assert _whole_units(working.get_window())


def test_chunk_never_splits_a_unit_with_several_results() -> None:
    working = WorkingMemory(10, evict_chunk=4)
    for n in range(6):
        calls = [ToolCall("echo", {"n": i}, id=f"c{n}{i}") for i in range(3)]
        results = [Message("tool", "r", tool_call_id=c.id) for c in calls]
        working.append_turn(Message("assistant", tool_calls=calls), results)
        assert _whole_units(working.get_window())
        assert len(working.get_window()) <= 10


def test_chunk_keeps_the_newest_unit_even_when_it_alone_is_too_big() -> None:
    working = WorkingMemory(4, evict_chunk=3)
    calls = [ToolCall("echo", {}, id=f"c{i}") for i in range(6)]
    big = [Message("tool", "r", tool_call_id=c.id) for c in calls]
    working.append(Message("user", "query"))
    working.append_turn(Message("assistant", tool_calls=calls), big)
    assert len(working.get_window()) == 7  # the lone newest unit stays


def test_a_negative_chunk_is_refused() -> None:
    with pytest.raises(ValueError, match="evict_chunk"):
        WorkingMemory(20, evict_chunk=-1)


def test_drop_since_keeps_the_query_and_pinned_units() -> None:
    working = WorkingMemory(40)
    old = Message("user", "earlier task")
    query = Message("user", "query")
    pinned = Message("user", "inbox note")
    for msg in (old, query):
        working.append(msg)
    working.append_turn(*_turn(0))
    working.append(pinned)
    working.append_turn(*_turn(1))
    dropped = working.drop_since(query, keep=[pinned])
    assert dropped == 4
    assert working.get_window() == [old, query, pinned]


def test_drop_since_without_its_anchor_drops_everything_but_the_pinned() -> None:
    working = WorkingMemory(40)
    working.append_turn(*_turn(0))
    pinned = Message("user", "inbox note")
    working.append(pinned)
    assert working.drop_since(None, keep=[pinned]) == 2
    assert working.get_window() == [pinned]
    assert working.drop_since(Message("user", "gone")) == 1
    assert working.get_window() == []


def test_drop_since_forgets_the_aliases_of_what_it_drops() -> None:
    working = WorkingMemory(40)
    query = Message("user", "query")
    working.append(query)
    assistant, results = _turn(0)
    working.append_turn(assistant, results, aliases=["a1"])
    assert working.shown_aliases() == {"a1"}
    working.drop_since(query)
    assert working.shown_aliases() == frozenset()


# ── The options ──


def test_the_option_is_validated_and_defaults_to_slide() -> None:
    assert AgentOptions().window_eviction == "slide"
    with pytest.raises(ValueError, match="window_eviction"):
        AgentOptions(window_eviction="sliding")  # type: ignore[arg-type]


async def _agent(mode: str) -> Agent:
    llm = OpenAICompatLLM(
        "http://stub.test/v1",
        "stub",
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    return Agent(
        llm=llm, options=AgentOptions(builtins=frozenset(), window_eviction=mode)  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("mode", ["slide", "chunk", "boundary"])
async def test_the_mode_reaches_the_window_and_the_run_config(mode: str) -> None:
    agent = await _agent(mode)
    try:
        sizes = _fill(agent.hippo.working, 12)
        # The window's size after the tenth step: full under slide, a chunk
        # less under chunk and boundary.
        assert sizes[9] == (SHORT_TERM_MAX if mode == "slide" else 12)
        config = agent._run_config()
        if mode == "slide":
            assert "window_eviction" not in config
        else:
            assert config["window_eviction"] == mode
    finally:
        await agent.close()


# ── The prompt-cap fit (X2) ──


def _bulk_window(agent: Agent, units: int, size: int) -> Message:
    query = Message("user", "query")
    agent.hippo.working.append(query)
    for n in range(units):
        agent.hippo.working.append_turn(*_turn(n, "x" * size))
    return query


def _window_chars(agent: Agent) -> int:
    return sum(len(m.content) for m in agent.hippo.working.get_window())


async def test_the_cap_fit_drops_to_just_under_the_cap_under_slide() -> None:
    agent = await _agent("slide")
    try:
        query = _bulk_window(agent, 9, 12_000)  # 108,000 chars of results
        await agent._fit_prompt_cap([], "query", task_query=query)
        assert 90_000 < _window_chars(agent) <= 100_000
    finally:
        await agent.close()


@pytest.mark.parametrize("mode", ["chunk", "boundary"])
async def test_the_cap_fit_drops_to_70_percent_under_chunk_and_boundary(
    mode: str,
) -> None:
    agent = await _agent(mode)
    try:
        query = _bulk_window(agent, 9, 12_000)
        before = agent.hippo.working.get_window()
        await agent._fit_prompt_cap([], "query", task_query=query)
        chars = _window_chars(agent)
        assert chars <= 100_000 * WINDOW_FIT_TARGET
        assert chars > 100_000 * WINDOW_FIT_TARGET - 12_100  # by one unit
        # It dropped from the front, whole units.
        window = agent.hippo.working.get_window()
        assert window == before[len(before) - len(window) :]
        assert _whole_units(window)
    finally:
        await agent.close()


async def test_the_cap_fit_keeps_the_window_when_it_fits() -> None:
    agent = await _agent("chunk")
    try:
        query = _bulk_window(agent, 5, 12_000)  # 60,000 chars
        before = agent.hippo.working.get_window()
        await agent._fit_prompt_cap([], "query", task_query=query)
        assert agent.hippo.working.get_window() == before
    finally:
        await agent.close()


async def test_the_cap_fit_falls_back_to_the_cap_when_70_percent_is_out_of_reach() -> (
    None
):
    # The newest unit alone is over 70% of the cap but under it: dropping to
    # 70% is impossible, so the window drops just far enough to fit the cap.
    agent = await _agent("chunk")
    try:
        query = Message("user", "query")
        agent.hippo.working.append(query)
        agent.hippo.working.append_turn(*_turn(0, "x" * 20_000))
        agent.hippo.working.append_turn(*_turn(1, "x" * 80_000))
        await agent._fit_prompt_cap([], "query", task_query=query)
        window = agent.hippo.working.get_window()
        assert [m.content for m in window][-1] == "x" * 80_000
        assert len(window) == 2  # only the newest unit is left
    finally:
        await agent.close()


# ── The context fit (X2) ──


def _estimate(messages: list[Message], tools: list[ToolDef] | None) -> int:
    return sum(len(m.content) + 10 for m in messages)


def _history(units: int, size: int) -> list[Message]:
    messages = [Message("system", "s"), Message("user", "query")]
    for n in range(units):
        assistant, results = _turn(n, "y" * size)
        messages += [assistant, *results]
    return messages


def _fit(
    messages: list[Message], limit: int, target: float
) -> tuple[list[Message], dict[str, int]]:
    return fit_request(
        messages,
        None,
        limit=limit,
        estimate=_estimate,
        requires_replay=False,
        drop_target=target,
    )


def test_the_context_fit_drops_just_under_the_limit_by_default() -> None:
    messages = _history(10, 200)  # about 2,200
    fitted, counts = _fit(messages, 1_500, 1.0)
    assert 1_300 < _estimate(fitted, None) <= 1_500
    assert counts["units"] == 4


def test_the_context_fit_drops_to_the_target_when_it_must_drop() -> None:
    messages = _history(10, 200)
    fitted, counts = _fit(messages, 1_500, WINDOW_FIT_TARGET)
    assert 1_050 - 230 < _estimate(fitted, None) <= 1_050
    assert counts["units"] > 4
    assert [m.content for m in fitted[:2]] == ["s", "query"]
    assert _whole_units(fitted[2:])


def test_the_context_fit_leaves_a_request_that_fits_alone() -> None:
    messages = _history(3, 200)
    fitted, counts = _fit(messages, 5_000, WINDOW_FIT_TARGET)
    assert fitted == messages and sum(counts.values()) == 0


def test_the_context_fit_settles_for_the_limit_when_the_target_is_out_of_reach() -> (
    None
):
    # Only the newest unit and the head stay; with them the request is over 70%
    # of the limit but under it.
    messages = _history(6, 200)
    pinned = _estimate(messages[:2] + messages[-2:], None)
    limit = int(pinned / 0.85)
    fitted, counts = _fit(messages, limit, WINDOW_FIT_TARGET)
    assert WINDOW_FIT_TARGET * limit < _estimate(fitted, None) <= limit
    assert counts["units"] == 5


# ── A run against a stub server with a prefix cache ──


@dataclasses.dataclass
class CacheServer:
    """A stub OpenAI-compatible server that also models a prompt cache.

    A request's ``prompt_tokens`` is its rendered size over 4, its
    ``cached_tokens`` the characters it shares with the best earlier request
    over 4 (the tools block first, then the messages, as a chat template
    renders them).
    """

    steps_per_attempt: int
    bodies: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    rendered: list[str] = dataclasses.field(default_factory=list)
    plans: int = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        text = json.dumps(body.get("tools"), sort_keys=True) + json.dumps(
            body["messages"], sort_keys=True
        )
        shared = 0
        for earlier in self.rendered:
            n = 0
            for a, b in zip(earlier, text, strict=False):
                if a != b:
                    break
                n += 1
            shared = max(shared, n)
        self.rendered.append(text)
        usage = {
            "prompt_tokens": len(text) // 4,
            "completion_tokens": 5,
            "total_tokens": len(text) // 4 + 5,
            "prompt_tokens_details": {"cached_tokens": shared // 4},
        }
        if body.get("tools"):
            self.plans += 1
            call = {
                "id": f"call_{self.plans}",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json.dumps({"text": f"step {self.plans}"}),
                },
            }
            message: dict[str, Any] = {"content": "", "tool_calls": [call]}
        else:
            decision = {
                "summary": "Echoed.",
                "failed_hypotheses": [],
                "useful_evidence": [],
                "recommended_avoids": [],
                "next_promising_directions": ["echo again"],
                "progress_made": True,
                "task_still_feasible": True,
                "new_information_gained": True,
                "next_step_is_concrete": True,
                "next_step_is_novel": True,
                "recommended_action": "continue",
                "rationale": "Once more.",
                "best_effort_result": "hi",
            }
            message = {"content": json.dumps(decision)}
        return httpx.Response(
            200, json={"choices": [{"message": message}], "usage": usage}
        )


async def _stub_run(
    mode: str, *, max_steps: int, steps_per_attempt: int
) -> tuple[list[AgentEvent], CacheServer, Agent]:
    server = CacheServer(steps_per_attempt)
    llm = OpenAICompatLLM(
        "http://stub.test/v1", "stub", transport=httpx.MockTransport(server)
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            tool_exposure="all",
            window_eviction=mode,  # type: ignore[arg-type]
            max_steps=max_steps,
            steps_per_attempt=steps_per_attempt,
        ),
    )
    agent.hippo.register_tool(ECHO)

    async def echo(text: str = "") -> str:
        return text + " " + "e" * 300

    agent.executor.register("echo", echo)
    events: list[AgentEvent] = []
    try:
        await agent.run_task("echo hi", on_event=events.append)
    finally:
        await agent.close()
    return events, server, agent


def _collapsed_steps(events: list[AgentEvent]) -> list[int]:
    """Planning steps whose request lost part of the best earlier request's history."""
    return [
        e.data["step"]
        for e in events
        if e.type == "assistant.message"
        and e.data["prefix"] is not None
        and e.data["prefix"]["messages"] < e.data["prefix"]["previous_messages"]
    ]


async def test_slide_collapses_the_shared_history_every_step_once_full() -> None:
    events, _, _ = await _stub_run("slide", max_steps=24, steps_per_attempt=24)
    collapsed = _collapsed_steps(events)
    assert collapsed == list(range(12, 25))  # the window is full after step 10


async def test_chunk_collapses_at_most_once_per_half_chunk() -> None:
    events, server, _ = await _stub_run("chunk", max_steps=24, steps_per_attempt=24)
    collapsed = _collapsed_steps(events)
    assert len(collapsed) >= 3
    assert collapsed[0] == 11  # the tenth step overflowed the window
    gaps = [b - a for a, b in zip(collapsed, collapsed[1:], strict=False)]
    assert min(gaps) >= WINDOW_EVICT_CHUNK // 2
    # No request carries more than the window's maximum, plus the system message.
    assert all(len(b["messages"]) <= SHORT_TERM_MAX + 2 for b in server.bodies)


async def test_boundary_drops_the_finished_attempt_at_the_next_attempt() -> None:
    events, server, _ = await _stub_run("boundary", max_steps=30, steps_per_attempt=12)
    plans = [e for e in events if e.type == "assistant.message"]
    starts = [1, 13, 25]
    collapsed = _collapsed_steps(events)
    assert {13, 25} <= set(collapsed)
    # Within an attempt the chunk fallback collapses at most once (at 11, 24).
    assert sorted(set(collapsed) - set(starts)) == [11, 24]
    # The first request of an attempt holds the system message, the query and the
    # reflection carried in the system text, not the finished attempt's turns.
    plan_bodies = [b for b in server.bodies if b.get("tools")]
    for step in (13, 25):
        roles = [m["role"] for m in plan_bodies[step - 1]["messages"]]
        assert roles == ["system", "user"]
    assert len(plans) == 30


async def test_slide_keeps_the_finished_attempt_across_the_boundary() -> None:
    _, server, _ = await _stub_run("slide", max_steps=30, steps_per_attempt=12)
    plan_bodies = [b for b in server.bodies if b.get("tools")]
    roles = [m["role"] for m in plan_bodies[12]["messages"]]  # step 13
    assert roles.count("assistant") > 0 and len(roles) > 10


async def test_the_run_reports_the_mode_and_slide_adds_no_key() -> None:
    _, _, agent = await _stub_run("chunk", max_steps=3, steps_per_attempt=3)
    assert agent._run_config()["window_eviction"] == "chunk"


def _report(log: Path) -> str:
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "cache_report.py"), str(log)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _cached_share(out: str) -> float:
    line = next(x for x in out.splitlines() if x.strip().startswith("total:"))
    return float(line.rsplit("cached share", 1)[1].strip().rstrip("%"))


async def test_the_modes_rank_by_cache_reuse_in_cache_report(tmp_path: Path) -> None:
    shares = {}
    for mode in ("slide", "chunk", "boundary"):
        events, _, _ = await _stub_run(mode, max_steps=30, steps_per_attempt=12)
        log = tmp_path / f"{mode}.jsonl"
        log.write_text(
            "\n".join(
                json.dumps({"v": 1, "type": "event", "event": e.to_dict()})
                for e in events
            )
        )
        out = _report(log)
        shares[mode] = _cached_share(out)
        print(mode, out.splitlines()[-1])
    assert shares["chunk"] > shares["slide"]
    assert shares["boundary"] > shares["slide"]
