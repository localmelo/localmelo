"""The live gate's own tests: no FastAPI, no HTTP, a fake clock (R1)."""

from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import replace
from datetime import UTC, datetime
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
from localmelo.support.scheduler import REASONS, Clock, Machine, Params
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
        real_shield = playground.asyncio.shield
        shields: list[tuple[object, object]] = []

        async def spy(awaitable: Any, timeout: float | None = None) -> Any:
            seen.append((awaitable, timeout))
            return await original(awaitable, timeout)

        def shield_spy(inner: Any) -> Any:
            shielded = real_shield(inner)
            shields.append((inner, shielded))
            return shielded

        monkeypatch.setattr(playground.asyncio, "wait_for", spy)
        monkeypatch.setattr(playground.asyncio, "shield", shield_spy)
        await manager.run(session_id, "hello")
        assert len(seen) == 1
        awaitable, timeout = seen[0]
        assert timeout == 600.0
        assert isinstance(awaitable, asyncio.Future)
        # It is the shielded future, not the run's task itself.
        assert len(shields) == 1
        inner, shielded = shields[0]
        assert awaitable is shielded
        assert inner is not shielded
    finally:
        await manager.close_all()


# ── Admissibility (R11-R15) ──


async def test_fcfs_admits_in_arrival_order() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    gate.set_level("a", 1)
    first = gate.submit("a")
    assert admitted(first)
    gate.set_level("c", 3)
    older = gate.submit("c")
    gate.set_level("b", 2)
    newer = gate.submit("b")
    assert not admitted(older) and not admitted(newer)
    finish(gate, first)
    # fcfs is arrival order: the older P3 call goes before the newer P2 one.
    assert admitted(older)
    assert not admitted(newer)
    finish(gate, older)
    assert admitted(newer)
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


async def test_a_focused_embedding_is_never_held_and_logs_front(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=tmp_path)
    gate.focus("focused")
    holder = gate.submit("focused", kind="plan")
    assert admitted(holder)
    embedding = gate.submit("focused", kind="embed")
    # A full server holds a planning call, never an embedding (R14). The
    # lane path would hold it, so it is submitted as a chat call; the log
    # still names the focused chat's calls "front" (4.7, decision 1b).
    assert embedding.meta.session_kind == "front"
    state_call = next(
        c for c in gate._state.calls if c.call_id == embedding.meta.call_id
    )
    assert state_call.session_kind == "chat"
    assert admitted(embedding)
    assert not any(a.startswith(("orphan_call", "lane_answer")) for a in gate.alarms)
    finish(gate, embedding)
    finish(gate, holder)
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    traces = read_trace(path)
    line = next(c for c in traces[0].calls if c.call_index == embedding.meta.call_index)
    assert line.session_kind == "front"
    assert line.kind == "embed"


async def test_an_empty_kind_is_a_plan_and_is_held_under_class(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0, call_log_dir=tmp_path)
    holder = gate.submit("holder")
    assert admitted(holder)
    # R14: an unknown or empty kind reads as a plan, so it is holdable.
    empty = gate.submit("empty", kind="")
    assert empty.meta.kind == "plan"
    assert not admitted(empty)
    finish(gate, holder)
    assert admitted(empty)
    finish(gate, empty)
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    by_key = {trace.task_key: trace for trace in read_trace(path)}
    assert by_key["empty"].calls[0].kind == "plan"
    assert by_key["empty"].calls[0].status == "ok"


async def test_the_log_records_the_level_at_admission(tmp_path: Path) -> None:
    gate = ChatGate(policy="class", n_srv=1, r_f=0, call_log_dir=tmp_path)
    holder = gate.submit("holder")
    assert admitted(holder)
    low = gate.submit("low")
    assert not admitted(low)
    # Raise the held call while a full server holds it (F4's reproduction).
    gate.set_level("low", 1)
    finish(gate, holder)
    assert admitted(low)
    finish(gate, low)
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    by_key = {trace.task_key: trace for trace in read_trace(path)}
    assert by_key["low"].calls[0].level == 1
    assert by_key["holder"].calls[0].level == 2


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
    older = gate.submit("older")  # the older call stays P2
    raised = gate.submit("raised")
    assert not admitted(older) and not admitted(raised)
    gate.set_level("raised", 1)
    finish(gate, holder)
    # If a level change did nothing, arrival order would admit older.
    assert admitted(raised)
    assert not admitted(older)
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
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
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
    calls = [c for trace in read_trace(gate._log.path) for c in trace.calls]  # type: ignore[arg-type]
    # The admitted holder now also gets its shutdown line (F2); select the
    # refused one this test is about.
    call = next(c for c in calls if c.status == "refused")
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
            # The log's own dir is chmod 0700 (F9), so make its parent the
            # one that refuses the mkdir.
            gate = ChatGate(
                policy="fcfs", n_srv=2, call_log_dir=read_only / "scheduler"
            )
            call = gate.submit("a")
            finish(gate, call)
            assert gate._log.path is None
            assert sum("Call log" in record.message for record in caplog.records) == 1
            await gate.close()
    finally:
        read_only.chmod(0o700)


async def test_a_write_failure_after_startup_warns_once_and_keeps_the_chat(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ChatGate(policy="off", call_log_dir=tmp_path)
    assert gate.call_log_path is not None

    def broken(*args: Any, **kwargs: Any) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(chat_gate.os, "write", broken)
    with caplog.at_level("WARNING"):
        gate.finish(gate.submit("a"))
        assert gate.call_log_path is None
        gate.finish(gate.submit("b"))
    warnings = [r for r in caplog.records if "Call log write failed" in r.message]
    assert len(warnings) == 1
    await gate.close()


async def test_a_short_write_is_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = ChatGate(policy="off", call_log_dir=tmp_path)
    real = chat_gate.os.write
    state = {"first": True}

    def short(fd: int, data: bytes) -> int:
        # Write all but the last byte of the first line, as a file-size
        # limit or an interrupted write can (F1's reproduction).
        if state["first"]:
            state["first"] = False
            return real(fd, data[:-1])
        return real(fd, data)

    monkeypatch.setattr(chat_gate.os, "write", short)
    gate.finish(gate.submit("a"))
    gate.finish(gate.submit("b"))
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    assert [task.task_key for task in read_trace(path)] == ["a", "b"]


async def test_a_partial_write_is_truncated_and_disables_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = ChatGate(policy="off", call_log_dir=tmp_path)
    gate.finish(gate.submit("a"))
    real = chat_gate.os.write
    state = {"first": True}

    def failing(fd: int, data: bytes) -> int:
        if state["first"]:
            state["first"] = False
            real(fd, data[:-1])
            raise OSError("disk full")
        return real(fd, data)

    path = gate.call_log_path
    monkeypatch.setattr(chat_gate.os, "write", failing)
    gate.finish(gate.submit("b"))
    assert gate.call_log_path is None
    await gate.close()
    assert path is not None
    # The partial line is gone, so the earlier complete line still reads.
    assert [task.task_key for task in read_trace(path)] == ["a"]


async def test_the_reader_ignores_a_truncated_final_line(tmp_path: Path) -> None:
    gate = ChatGate(policy="off", call_log_dir=tmp_path)
    gate.finish(gate.submit("a"))
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    complete = path.read_text(encoding="utf-8")
    # A crash mid-write leaves a prefix of the next canonical line.
    path.write_text(complete + '{"admit_s":0.0,"attempt":1,"call_ind', encoding="utf-8")
    traces = read_trace(path)
    assert [task.task_key for task in traces] == ["a"]


async def test_pruning_uses_the_injected_clock(tmp_path: Path) -> None:
    clock = ClockBox()
    wall_s = clock.wall_us / 1_000_000
    runs = tmp_path / "runs"
    old = runs / "20200101T000000"
    old.mkdir(parents=True)
    os.utime(old, (wall_s - 31 * 86400, wall_s - 31 * 86400))
    keep = runs / "20200102T000000"
    keep.mkdir()
    os.utime(keep, (wall_s - 1 * 86400, wall_s - 1 * 86400))
    gate = ChatGate(policy="off", call_log_dir=tmp_path, retention_days=30, clock=clock)
    try:
        assert not old.exists()
        assert keep.exists(), "the fake clock, not the real one, sets the cutoff"
        path = gate.call_log_path
        assert path is not None
        assert path.parent.parent == runs
    finally:
        await gate.close()


async def test_two_gates_in_the_same_second_do_not_share_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Any:
            return datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(chat_gate, "datetime", FixedDatetime)
    first = ChatGate(policy="off", call_log_dir=tmp_path)
    second = ChatGate(policy="off", call_log_dir=tmp_path)
    try:
        first.finish(first.submit("a"))
        second.finish(second.submit("b"))
        assert first.call_log_path is not None
        assert second.call_log_path is not None
        assert first.call_log_path != second.call_log_path
        assert first.call_log_path.exists() and second.call_log_path.exists()
    finally:
        await first.close()
        await second.close()


async def test_the_call_log_parent_runs_and_run_dirs_are_0700(tmp_path: Path) -> None:
    base = tmp_path / "scheduler"
    gate = ChatGate(policy="off", call_log_dir=base)
    try:
        assert stat.S_IMODE(base.stat().st_mode) == 0o700
        assert stat.S_IMODE((base / "runs").stat().st_mode) == 0o700
        assert gate.call_log_path is not None
        assert stat.S_IMODE(gate.call_log_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(gate.call_log_path.stat().st_mode) == 0o600
    finally:
        await gate.close()


async def test_an_admitted_call_gets_a_failed_line_at_shutdown(tmp_path: Path) -> None:
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path)
    call = gate.submit("a")
    assert admitted(call)
    path = gate.call_log_path
    await gate.close()
    assert path is not None and path.is_file()
    traces = read_trace(path)
    assert len(traces) == 1
    line = traces[0].calls[0]
    assert line.status == "failed"
    assert line.admit_s is not None


async def test_an_apply_failure_admits_the_call_and_keeps_the_gate_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = chat_gate.ChatGate._update_holds
    state = {"failed": False}

    def flaky(self: Any, decided: Any, now: Any) -> None:
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("apply broke")
        real(self, decided, now)

    monkeypatch.setattr(chat_gate.ChatGate, "_update_holds", flaky)
    gate = ChatGate(policy="fcfs", n_srv=2)
    first = gate.submit("a")
    assert admitted(first)
    assert any(a.startswith("sched_transition_failed") for a in gate.alarms)
    finish(gate, first)
    second = gate.submit("b")
    assert admitted(second)
    await gate.close()


async def test_a_charge_failure_does_not_leak_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = chat_gate.ChatGate._charge
    state = {"failed": False}

    def flaky(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not state["failed"]:
            state["failed"] = True
            raise ArithmeticError("bad usage")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(chat_gate.ChatGate, "_charge", flaky)
    gate = ChatGate(policy="fcfs", n_srv=1)
    first = gate.submit("a")
    finish(gate, first)  # the charge raises once, but CallEnded must still apply
    second = gate.submit("b")
    assert admitted(second)
    assert not gate.held("b")
    await gate.close()


async def test_an_odd_usage_value_is_logged_as_unknown(tmp_path: Path) -> None:
    gate = ChatGate(policy="off", call_log_dir=tmp_path)
    call = gate.submit("a")
    gate.finish(
        call,
        "ok",
        {"prompt_tokens": 10**100, "completion_tokens": 1, "cached_tokens": -1},
    )
    assert gate.call_log_path is not None
    await gate.close()
    line = read_trace(gate._log.path)[0].calls[0]  # type: ignore[arg-type]
    assert line.prompt_tokens is None
    assert line.cached_tokens is None
    assert line.tokens_unknown is True


async def test_a_held_call_gets_a_refused_line_at_shutdown(tmp_path: Path) -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=tmp_path)
    holder = gate.submit("a")
    held = gate.submit("b")
    assert not admitted(held)
    path = gate.call_log_path
    await gate.close()
    assert path is not None and path.is_file()
    traces = read_trace(path)
    refused = [c for trace in traces for c in trace.calls if c.status == "refused"]
    assert len(refused) == 1
    assert refused[0].admit_s is None
    assert refused[0].start_s == refused[0].submit_s
    finish(gate, holder)


async def test_a_lane_call_is_taskless() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2)
    gate.focus("f")
    call = gate.submit("f")
    assert admitted(call)
    state_call = next(c for c in gate._state.calls if c.call_id == call.meta.call_id)
    assert state_call.session_kind == "front"
    assert state_call.task_id is None
    finish(gate, call)
    await gate.close()


async def test_every_charge_source_maps_to_the_trace_vocabulary() -> None:
    from localmelo.support.scheduler.costs import Rates, charge_s

    rates = Rates()
    sources = {
        "tokens": charge_s("ok", 10, 0, 5, None, None, None, None, rates)[1],
        "server": charge_s("failed", None, None, None, 1.0, 1.0, None, None, rates)[1],
        "wall_share": charge_s("failed", None, None, None, None, None, 2.0, 1.0, rates)[
            1
        ],
        "none": charge_s("failed", None, None, None, None, None, None, None, rates)[1],
    }
    mapped = {
        "tokens": "tokens",
        "server": "server",
        "wall_share": "wall",
        "none": "unknown",
    }
    for expected, source in sources.items():
        assert source == expected
        assert _trace_source(source) == mapped[expected]
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


async def test_the_hold_wake_is_the_focus_window() -> None:
    gate = ChatGate(policy="fcfs")
    try:
        assert gate.hold_wake_s == Params().tau_focus_s
        assert gate._params is not None
        gate._params = replace(gate._params, tau_focus_s=1.5)
        assert gate.hold_wake_s == 1.5
    finally:
        await gate.close()


async def test_a_lane_answer_alarms_admits_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = chat_gate.dispatch

    def answered(state: Any, now: Any, params: Any, **kwargs: Any) -> Any:
        # S2 has no code answer for the lane: dispatch's answer is an alarm
        # and an admit (R10). Force one for a lane call to drive the path.
        result = real(state, now, params, **kwargs)
        lane = [c for c in state.calls if c.session_kind == "front"]
        if not lane:
            return result
        return replace(
            result, lane_admit=(), lane_answer=((lane[0].call_id, "reservation"),)
        )

    monkeypatch.setattr(chat_gate, "dispatch", answered)
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path)
    gate.focus("focused")
    call = gate.submit("focused")
    assert admitted(call)
    assert "lane_answer:reservation" in gate.alarms
    finish(gate, call)
    path = gate.call_log_path
    await gate.close()
    assert path is not None
    line = read_trace(path)[0].calls[0]
    assert line.status == "ok"
    assert line.session_kind == "front"


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


# ── The machine and the lock (S2b, R5/R9-R12) ──


async def test_set_machine_under_off_updates_the_band_and_the_page_without_a_decision() -> (
    None
):
    def spy(*args: object, **kwargs: object) -> Any:
        raise AssertionError("off must not decide")

    gate = ChatGate(policy="off", decide_fn=spy, lock_gate="wait")
    gate.set_machine(Machine(lock="foreign", on_ac=False, battery_pct=55.0))
    assert gate.machine_payload()["band"] == 2
    assert gate.machine_payload()["lock"] == "foreign"
    gate.set_machine(Machine(lock="foreign", on_ac=False, battery_pct=45.0))
    assert gate.machine_payload()["band"] == 1
    gate.set_machine(Machine(lock="foreign", on_ac=False, battery_pct=10.0))
    assert gate.machine_payload()["band"] == 0
    gate.set_machine(Machine(lock="none", on_ac=True, battery_pct=10.0))
    assert gate.machine_payload()["band"] == 4
    await gate.close()


async def test_record_never_sees_the_lock() -> None:
    gate = ChatGate(policy="fcfs", n_srv=2, lock_gate="record")
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("a")
    assert admitted(call)
    assert gate.held("a") == ""
    assert gate.machine_payload()["lock"] == "foreign"
    finish(gate, call)
    await gate.close()


@pytest.mark.parametrize("policy", ["off", "fcfs"])
async def test_wait_holds_plan_reflect_and_embed_before_the_policy(
    policy: str,
) -> None:
    gate = ChatGate(policy=policy, n_srv=8, lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    calls = [
        gate.submit("a", kind="plan"),
        gate.submit("a", kind="reflect"),
        gate.submit("a", kind="embed"),
    ]
    for call in calls:
        assert not admitted(call)
        assert gate.held("a") == "Reservation"
        assert gate.held_since("a") is not None
    assert gate._state.calls == ()  # the policy never saw them
    gate.set_machine(Machine(lock="none"))
    assert all(admitted(call) for call in calls)
    assert gate.held("a") == ""
    for call in calls:
        finish(gate, call)
    await gate.close()


async def test_a_lock_wait_is_one_hold_and_one_log_line(tmp_path: Path) -> None:
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path, lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("a")
    assert gate.held("a") == "Reservation"
    finish(gate, call, "refused")
    assert gate.held("a") == ""
    assert gate._state.calls == ()
    path = gate.call_log_path
    assert path is not None
    await gate.close()
    traces = read_trace(path)
    assert traces[0].calls[0].status == "refused"
    assert traces[0].calls[0].admit_s is None


@pytest.mark.parametrize("policy", ["off", "fcfs"])
async def test_a_lock_wait_released_in_submit_order(policy: str) -> None:
    gate = ChatGate(policy=policy, n_srv=1, lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    first = gate.submit("a", kind="plan")
    second = gate.submit("b", kind="plan")
    gate.set_machine(Machine(lock="none"))
    if policy == "fcfs":
        assert admitted(first)
        assert not admitted(second)
        finish(gate, first)
        assert admitted(second)
    else:
        assert admitted(first) and admitted(second)
    finish(gate, first)
    finish(gate, second)
    await gate.close()


async def test_a_pending_call_was_not_orphaned_by_the_lock() -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, lock_gate="wait")
    holder = gate.submit("holder", kind="plan")
    assert admitted(holder)
    pending = gate.submit("a", kind="plan")
    assert not admitted(pending)
    gate.set_machine(Machine(lock="foreign"))
    assert gate.held("a") == "Reservation"
    finish(gate, holder)
    gate.set_machine(Machine(lock="none"))
    assert admitted(pending)
    assert not any(alarm.startswith("no_pending_call") for alarm in gate.alarms)
    finish(gate, pending)
    await gate.close()


async def test_the_lane_waits_under_respect() -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, lock_gate="wait")
    holder = gate.submit("holder", kind="plan")
    assert admitted(holder)
    gate.focus("a")
    lane = gate.submit("a", kind="plan")
    assert not admitted(lane)
    gate.set_machine(Machine(lock="foreign"))
    assert not any(alarm.startswith("lane_answer") for alarm in gate.alarms)
    assert gate.held("a") == "Reservation"
    finish(gate, holder)
    gate.set_machine(Machine(lock="none"))
    assert admitted(lane)
    finish(gate, lane)
    await gate.close()


async def test_a_lock_hold_does_not_flap() -> None:
    decisions: list[object] = []

    def spy(state: Any, now: Any, params: Any) -> Any:
        from localmelo.support.scheduler import decide

        decisions.append(object())
        return decide(state, now, params)

    gate = ChatGate(policy="fcfs", n_srv=2, lock_gate="wait", decide_fn=spy)
    gate.set_machine(Machine(lock="foreign"))
    assert len(decisions) == 1
    call = gate.submit("a", kind="plan")
    start = gate.held_since("a")
    for _ in range(10):
        gate.set_machine(Machine(lock="foreign"))
    assert len(decisions) == 1
    assert gate.held_since("a") == start
    assert gate.held_seconds("a") == 0.0
    gate.set_machine(Machine(lock="none"))
    finish(gate, call)
    await gate.close()


async def test_stop_during_a_lock_wait_cancels_the_future_and_keeps_the_gate_usable() -> (
    None
):
    gate = ChatGate(policy="fcfs", n_srv=2, lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("a", kind="plan")
    assert call.future is not None and not call.future.done()
    call.future.cancel()
    finish(gate, call, "refused")
    assert gate.held("a") == ""
    gate.set_machine(Machine(lock="none"))
    later = gate.submit("a", kind="plan")
    assert admitted(later)
    finish(gate, later)
    await gate.close()


async def test_close_cancels_lock_waiters_and_logs_them_refused(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="fcfs", n_srv=2, call_log_dir=tmp_path, lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    call = gate.submit("a", kind="plan")
    path = gate.call_log_path
    assert path is not None
    await gate.close()
    assert call.future is not None and call.future.cancelled()
    traces = read_trace(path)
    assert traces[0].calls[0].status == "refused"


def test_enforces_holds_is_true_for_fcfs_or_wait_and_false_for_off_record() -> None:
    assert ChatGate(policy="fcfs", lock_gate="record").enforces_holds is True
    assert ChatGate(policy="off", lock_gate="wait").enforces_holds is True
    assert ChatGate(policy="off", lock_gate="record").enforces_holds is False
