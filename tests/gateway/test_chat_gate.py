"""The live gate's own tests: no FastAPI, no HTTP, a fake clock (R1)."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.schema import Message, RunResult, ToolDef
from localmelo.support.gateway import chat_gate
from localmelo.support.gateway.chat_gate import (
    ChatGate,
    GatedEmbedding,
    GatedLLM,
    _trace_source,
)
from localmelo.support.gateway.playground import SmokePlayground
from localmelo.support.scheduler import REASONS, Clock
from localmelo.support.scheduler.sim.trace import SCHEMA, read_trace

# ── Fakes ──


class ClockBox:
    """A clock the test advances by hand; both faces move together."""

    def __init__(self) -> None:
        self.mono_us = 1_000_000
        self.wall_us = 1_700_000_000_000_000

    def __call__(self) -> Clock:
        return Clock(self.mono_us, self.wall_us)

    def advance(self, seconds: float) -> None:
        step = round(seconds * 1_000_000)
        self.mono_us += step
        self.wall_us += step


class FakeLLM(BaseLLMProvider):
    """A chat provider that records its requests and answers in order."""

    def __init__(self, answer: str = "ok") -> None:
        self.model = "fake-chat"
        self.last_reasoning_key = "reasoning_content"
        self.last_usage: dict[str, int] | None = {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "cached_tokens": 2,
        }
        self.answer = answer
        self.calls: list[tuple[list[tuple[str, str]], list[ToolDef] | None]] = []
        self.closed = False
        self.error: BaseException | None = None
        self.delay = 0.0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append(([(m.role, m.content) for m in messages], tools))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return Message(role="assistant", content=self.answer)

    async def close(self) -> None:
        self.closed = True


class FakeEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text))] for text in texts]

    async def close(self) -> None:
        self.closed = True


def admitted(call: Any) -> bool:
    return bool(call.future is not None and call.future.done())


def finish(
    gate: ChatGate,
    call: Any,
    outcome: str = "ok",
    usage: dict[str, int] | None = None,
) -> None:
    gate.finish(call, outcome, usage)


def submit_high(gate: ChatGate, session_id: str, **kwargs: Any) -> Any:
    """Submit a P1 call: it is exempt from the focus cap, not from the limit."""
    gate.set_level(session_id, 1)
    return gate.submit(session_id, **kwargs)


def messages(count: int) -> list[Message]:
    return [Message(role="user", content=f"m{index}") for index in range(count)]


# ── off: observation only (R8) ──


async def test_off_admits_every_call_at_once_and_books_no_hold() -> None:
    decided: list[object] = []

    def spy(*args: object, **kwargs: object) -> Any:
        decided.append(args)
        raise AssertionError("off must not decide")

    gate = ChatGate(policy="off", decide_fn=spy)
    first = gate.submit("a")
    second = gate.submit("b")
    assert first.future is None
    assert second.future is None
    assert gate.held_seconds("a") == 0.0
    assert gate.held("a") == ""
    assert decided == []
    await gate.close()


async def test_off_sends_the_same_requests_in_the_same_order() -> None:
    direct = FakeLLM()
    for index in range(3):
        await direct.chat(messages(index + 1))

    gated_llm = FakeLLM()
    gate = ChatGate(policy="off")
    wrapper = GatedLLM(gated_llm, gate.client("a"))
    for index in range(3):
        await wrapper.chat(messages(index + 1))

    assert gated_llm.calls == direct.calls
    await gate.close()


async def test_off_copies_model_usage_and_reasoning_key_after_a_call() -> None:
    inner = FakeLLM()
    gate = ChatGate(policy="off")
    wrapper = GatedLLM(inner, gate.client("a"))
    assert wrapper.model == "fake-chat"
    inner.model = "changed-model"
    inner.last_reasoning_key = "reasoning"
    inner.last_usage = {"prompt_tokens": 5, "completion_tokens": 1}
    await wrapper.chat(messages(1))
    assert wrapper.model == "changed-model"
    assert wrapper.last_reasoning_key == "reasoning"
    assert wrapper.last_usage == {"prompt_tokens": 5, "completion_tokens": 1}

    failing = FakeLLM()
    failing.error = RuntimeError("provider down")
    broken = GatedLLM(failing, gate.client("b"))
    with pytest.raises(RuntimeError, match="provider down"):
        await broken.chat(messages(1))
    failing.error = None
    assert (await broken.chat(messages(1))).content == "ok"
    await gate.close()


async def test_off_leaves_the_run_timeout_line_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support.gateway import playground

    manager = SmokePlayground(memory_dir=tmp_path, policy="off")
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: FakeLLM())
    try:
        payload = await manager.create_session(
            chat_url="http://127.0.0.1:8000/v1", chat_model="Qwen-27B", adapter="omlx"
        )
        session_id = str(payload["session_id"])

        async def run_task(query: str, **kwargs: Any) -> RunResult:
            return RunResult(task_id="t", status="completed", result="hi")

        monkeypatch.setattr(manager._sessions[session_id].agent, "run_task", run_task)
        seen: list[tuple[object, float | None]] = []
        original = playground.asyncio.wait_for

        async def spy(awaitable: Any, timeout: float | None = None) -> Any:
            seen.append((awaitable, timeout))
            return await original(awaitable, timeout)

        monkeypatch.setattr(playground.asyncio, "wait_for", spy)
        await manager.run(session_id, "hello")
        assert len(seen) == 1
        awaitable, timeout = seen[0]
        assert timeout == 600.0
        assert isinstance(awaitable, asyncio.Future)
    finally:
        await manager.close_all()


# ── Admissibility (R11-R15) ──


async def test_fcfs_admits_in_arrival_order() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    order: list[str] = []
    first = gate.submit("a")
    assert admitted(first)
    second = gate.submit("b")
    third = gate.submit("c")
    assert not admitted(second) and not admitted(third)
    finish(gate, first)
    order.append("b" if admitted(second) else "c")
    finish(gate, second if admitted(second) else third)
    order.append("c" if admitted(third) else "b")
    assert order == ["b", "c"]
    await gate.close()


async def test_class_admits_the_higher_level_first() -> None:
    gate = ChatGate(policy="class", n_srv=5, r_f=0)
    holders = [submit_high(gate, f"h{index}") for index in range(4)]
    assert all(admitted(call) for call in holders)
    low = gate.submit("low")
    high = gate.submit("high")
    middle = gate.submit("middle")
    gate.set_level("low", 3)
    gate.set_level("high", 1)
    gate.set_level("middle", 2)
    assert not admitted(low) and not admitted(high) and not admitted(middle)
    finish(gate, holders[0])
    assert admitted(high)
    assert not admitted(middle) and not admitted(low)
    finish(gate, holders[1])
    assert admitted(middle)
    finish(gate, holders[2])
    assert admitted(low)
    await gate.close()


async def test_the_lane_bypasses_the_non_lane_limit() -> None:
    gate = ChatGate(policy="fcfs", n_srv=8, r_f=1)
    holders = [submit_high(gate, f"h{index}") for index in range(6)]
    assert all(admitted(call) for call in holders)
    background = gate.submit("background")
    assert not admitted(background)
    gate.focus("focused")
    lane = gate.submit("focused")
    assert admitted(lane)
    assert not admitted(background)
    await gate.close()


async def test_the_lane_waits_when_every_server_slot_is_busy() -> None:
    gate = ChatGate(policy="fcfs", n_srv=3, r_f=0)
    # Embeddings are never held, so they can fill every server slot.
    holders = [gate.submit(f"h{index}", kind="embed") for index in range(3)]
    assert all(admitted(call) for call in holders)
    gate.focus("focused")
    lane = gate.submit("focused")
    background = gate.submit("background")
    assert not admitted(lane)
    assert gate.held("focused") == "Resources"
    assert not admitted(background)
    finish(gate, holders[0])
    assert admitted(lane)
    assert not admitted(background)
    await gate.close()


async def test_the_limit_is_max_one_n_srv_minus_one_r_f() -> None:
    wide = ChatGate(policy="fcfs", n_srv=8, r_f=1)
    holders = [submit_high(wide, f"h{index}") for index in range(6)]
    assert all(admitted(call) for call in holders)
    assert not admitted(wide.submit("over"))
    assert "reserve_clamped" not in wide.alarms
    await wide.close()

    narrow = ChatGate(policy="fcfs", n_srv=2, r_f=1)
    first = narrow.submit("a")
    second = narrow.submit("b")
    assert admitted(first) and not admitted(second)
    assert "reserve_clamped" not in narrow.alarms
    await narrow.close()

    single = ChatGate(policy="fcfs", n_srv=1, r_f=1)
    first = single.submit("a")
    second = single.submit("b")
    assert admitted(first) and not admitted(second)
    assert single.alarms.count("reserve_clamped") == 1
    await single.close()


async def test_focus_caps_background_calls_at_n_fg() -> None:
    gate = ChatGate(policy="fcfs", n_srv=8, r_f=1, n_fg=1)
    gate.focus("focused")
    first = gate.submit("b1")
    second = gate.submit("b2")
    assert admitted(first)
    assert not admitted(second)
    assert gate.held("b2") in REASONS
    await gate.close()


async def test_without_a_heartbeat_the_cap_is_n_a() -> None:
    clock = ClockBox()
    gate = ChatGate(policy="fcfs", n_srv=8, r_f=1, n_fg=1, clock=clock)
    gate.focus("focused")
    first = gate.submit("b1")
    second = gate.submit("b2")
    assert admitted(first) and not admitted(second)
    clock.advance(31.0)
    gate.wake()
    assert admitted(second)
    await gate.close()


async def test_a_retrieve_call_is_admitted_while_the_limit_is_full() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    holder = gate.submit("a")
    assert admitted(holder)
    retrieve = gate.submit("r", kind="retrieve")
    assert admitted(retrieve)
    assert not gate.alarms
    await gate.close()


async def test_an_embedding_is_logged_with_kind_embed_and_counts_in_flight(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=tmp_path)
    embedding = gate.submit("a", kind="embed")
    assert admitted(embedding)
    gate.focus("focused")
    lane = gate.submit("focused")
    assert not admitted(lane)
    assert gate.held("focused") == "Resources"
    finish(gate, embedding)
    assert admitted(lane)
    assert gate._log.path is not None
    await gate.close()
    traces = read_trace(gate._log.path)  # type: ignore[arg-type]
    assert traces[0].calls[0].kind == "embed"
    assert traces[0].calls[0].status == "ok"


async def test_an_embedding_does_not_open_a_chat_task() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    embedding = gate.submit("only-an-embedding", kind="embed")
    assert admitted(embedding)
    assert gate._state.tasks == ()
    await gate.close()


async def test_a_focused_embedding_is_never_held() -> None:
    gate = ChatGate(policy="fcfs", n_srv=1)
    gate.focus("focused")
    holder = gate.submit("focused", kind="plan")
    assert admitted(holder)
    embedding = gate.submit("focused", kind="embed")
    # A full server holds a planning call, never an embedding (R14). The
    # lane path would hold it, so it is submitted as a chat call.
    assert embedding.meta.session_kind == "chat"
    assert admitted(embedding)
    assert not any(a.startswith(("orphan_call", "lane_answer")) for a in gate.alarms)
    await gate.close()


async def test_an_embedding_after_the_last_chat_call_is_admitted_and_not_an_orphan() -> (
    None
):
    gate = ChatGate(policy="fcfs", n_srv=2)
    chat = gate.submit("a")
    finish(gate, chat)
    embedding = gate.submit("a", kind="embed")
    assert admitted(embedding)
    assert not any(alarm.startswith("orphan_call") for alarm in gate.alarms)
    await gate.close()


async def test_an_idle_chat_produces_no_no_pending_call_alarm() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    chat = gate.submit("a")
    finish(gate, chat)
    gate.submit("b")
    assert not any(alarm.startswith("no_pending_call") for alarm in gate.alarms)
    await gate.close()


async def test_a_retry_rides_on_its_first_admission() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    holder = gate.submit("a")
    assert admitted(holder)
    retry = gate.submit("b", attempt=2)
    assert admitted(retry)
    await gate.close()


async def test_a_level_change_redistributes_the_next_admission() -> None:
    gate = ChatGate(policy="class", n_srv=2)
    holder = gate.submit("holder")
    assert admitted(holder)
    second = gate.submit("p2")
    third = gate.submit("p3")
    gate.set_level("p2", 3)
    assert not admitted(second) and not admitted(third)
    gate.set_level("p2", 1)
    finish(gate, holder)
    assert admitted(second)
    assert not admitted(third)
    await gate.close()


async def test_a_hold_records_its_reason_and_seconds(tmp_path: Path) -> None:
    clock = ClockBox()
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path, clock=clock)
    holder = gate.submit("a")
    held = gate.submit("b")
    assert not admitted(held)
    assert gate.held("b") in REASONS
    assert gate.held_since("b") is not None
    clock.advance(0.25)
    finish(gate, holder)
    assert admitted(held)
    assert gate.held_seconds("b") == pytest.approx(0.25)
    finish(gate, held)
    assert gate._log.path is not None
    await gate.close()
    traces = read_trace(gate._log.path)  # type: ignore[arg-type]
    call = {trace.task_key: trace for trace in traces}["b"].calls[0]
    assert call.submit_s is not None and call.admit_s is not None
    assert call.admit_s - call.submit_s == pytest.approx(0.25)


# ── The call log (R19, R20) ──


async def test_the_log_writes_one_canonical_line_per_call_with_mode_0600(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path)
    first = gate.submit("a")
    finish(gate, first, usage={"prompt_tokens": 100, "completion_tokens": 20})
    second = gate.submit("b")
    finish(gate, second)
    assert gate._log.path is not None
    path = gate._log.path
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    traces = read_trace(path)
    assert [trace.task_key for trace in traces] == ["a", "b"]
    assert traces[0].source == "gateway"
    assert traces[0].harness == "localmelo"
    assert traces[0].overlapping is False
    assert SCHEMA in path.read_text(encoding="utf-8")
    await gate.close()


async def test_a_refused_call_is_logged_with_no_admit_time(tmp_path: Path) -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=tmp_path)
    holder = gate.submit("a")
    held = gate.submit("b")
    assert not admitted(held)
    assert held.future is not None
    held.future.cancel()
    finish(gate, held, "refused")
    assert gate.held("b") == ""
    assert gate._log.path is not None
    await gate.close()
    call = read_trace(gate._log.path)[0].calls[0]  # type: ignore[arg-type]
    assert call.status == "refused"
    assert call.admit_s is None
    assert call.start_s == call.submit_s
    finish(gate, holder)


async def test_the_log_round_trips_and_the_call_index_counts_per_session(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path)
    for session, usage in (
        ("a", {"prompt_tokens": 3, "completion_tokens": 1}),
        ("a", {"prompt_tokens": 4, "completion_tokens": 2}),
        ("b", {"prompt_tokens": 5, "completion_tokens": 3}),
    ):
        call = gate.submit(session)
        assert admitted(call)
        finish(gate, call, usage=usage)
    assert gate._log.path is not None
    await gate.close()
    traces = read_trace(gate._log.path)  # type: ignore[arg-type]
    by_key = {trace.task_key: trace for trace in traces}
    assert [call.call_index for call in by_key["a"].calls] == [0, 1]
    assert [call.call_index for call in by_key["b"].calls] == [0]
    assert by_key["a"].calls[1].prompt_tokens == 4


async def test_a_log_write_failure_warns_once_and_does_not_fail_the_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        with caplog.at_level("WARNING"):
            gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=read_only)
            call = gate.submit("a")
            finish(gate, call)
            assert gate._log.path is None
            assert sum("Call log" in record.message for record in caplog.records) == 1
            await gate.close()
    finally:
        read_only.chmod(0o700)


async def test_every_charge_source_maps_to_the_trace_vocabulary() -> None:
    assert _trace_source("server") == "server"
    assert _trace_source("tokens") == "tokens"
    assert _trace_source("wall_share") == "wall"
    assert _trace_source("none") == "unknown"
    assert _trace_source("not-a-source") == "unknown"


# ── Failures, cancellation and shutdown (R10, R21, R22) ──


async def test_a_decide_failure_falls_back_and_still_admits() -> None:
    def broken(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("boom")

    gate = ChatGate(policy="fcfs", n_srv=2, decide_fn=broken)
    first = gate.submit("a")
    assert admitted(first)
    assert any(alarm.startswith("decide_failed") for alarm in gate.alarms)
    gate.focus("focused")
    lane = gate.submit("focused")
    assert admitted(lane)
    await gate.close()


async def test_an_advance_failure_admits_the_call_and_keeps_the_gate_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = chat_gate.advance
    state = {"failed": False}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("advance broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(chat_gate, "advance", flaky)
    gate = ChatGate(policy="fcfs", n_srv=2)
    first = gate.submit("a")
    assert admitted(first)
    assert any(alarm.startswith("sched_transition_failed") for alarm in gate.alarms)
    finish(gate, first)
    second = gate.submit("b")
    assert admitted(second)
    await gate.close()


async def test_a_gate_transition_never_yields_to_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = {"ticks": 0}
    seen = {"before": 0, "violations": 0}
    real_advance = chat_gate.advance
    real_apply = chat_gate.apply_dispatch

    def watched_advance(*args: Any, **kwargs: Any) -> Any:
        seen["before"] = counter["ticks"]
        return real_advance(*args, **kwargs)

    def watched_apply(*args: Any, **kwargs: Any) -> Any:
        # A tick between advance and apply means the transition awaited.
        if counter["ticks"] != seen["before"]:
            seen["violations"] += 1
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(chat_gate, "advance", watched_advance)
    monkeypatch.setattr(chat_gate, "apply_dispatch", watched_apply)
    gate = ChatGate(policy="fcfs", n_srv=8)

    async def ticker() -> None:
        while True:
            counter["ticks"] += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    try:
        for index in range(10):
            call = gate.submit(f"s{index % 3}")
            finish(gate, call)
            await asyncio.sleep(0)
        assert counter["ticks"] > 0, "the ticker must have had chances to run"
        assert seen["violations"] == 0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await gate.close()


async def test_close_cancels_pending_futures_and_closes_the_log(tmp_path: Path) -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=tmp_path)
    holder = gate.submit("a")
    held = gate.submit("b")
    assert not admitted(held)
    assert held.future is not None
    path = gate._log.path
    await gate.close()
    assert held.future.cancelled()
    assert gate._calls == {}
    assert path is not None
    assert path.exists()
    # A write after close is a silent no-op, and a second close is too.
    gate._log.write({"not": "a line"})
    await gate.close()
    # A call submitted after shutdown is admitted, not left pending.
    late = gate.submit("c")
    assert late.future is None
    finish(gate, late)
    finish(gate, holder)


# ── The wrappers (R18) ──


async def test_gated_embedding_forwards_embed_and_close() -> None:
    inner = FakeEmbedding()
    gate = ChatGate(policy="off")
    wrapper = GatedEmbedding(inner, gate.client("a"))
    assert await wrapper.embed(["ab", "c"]) == [[2.0], [1.0]]
    assert inner.calls == 1
    await wrapper.close()
    assert inner.closed
    await gate.close()


async def test_the_gate_survives_a_failed_provider_and_stays_usable() -> None:
    inner = FakeLLM()
    gate = ChatGate(policy="fcfs", n_srv=2)
    wrapper = GatedLLM(inner, gate.client("a"))
    inner.error = RuntimeError("down")
    with pytest.raises(RuntimeError, match="down"):
        await wrapper.chat(messages(1))
    inner.error = None
    assert (await wrapper.chat(messages(1))).content == "ok"
    await gate.close()
    assert inner.closed is False  # close_all owns the inner provider
