"""The machine readers: the GPU lock, the battery and the machine log (S2b)."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.machine_monitor import (
    MACHINE_SCHEMA,
    BatteryReader,
    BatterySnapshot,
    LockSnapshot,
    LockWatcher,
    MachineMonitor,
    classify,
    classify_lock,
    parse_battery,
    parse_lock,
    read_lock_file,
    read_machine,
)
from localmelo.support.scheduler import Machine
from localmelo.support.scheduler.machine_gates import battery_band

NIGHT = """\
holder: local_ai-benchmark session
purpose: Night chain: Core AI llm-server gate, M8 (llama.cpp, oMLX oQ8+MTP, lithos-metal, vLLM-Metal; 6 scenarios), MLC-LLM text-only
start: 2026-10-20T23:30:09-0400
eta: 2026-10-21T12:00:09-0400 (750 min)
"""

OVERNIGHT = """\
holder: agent-benchmark session (personal-agent-bench)
purpose: real-web UI eval: 27B oQ5 then Flash-Next oQ4e (untimed, port 8510)
start: 2026-10-27 03:06:08 EDT
eta: 330 min
"""


def parse(text: str, now: float, self_holder: str = "") -> LockSnapshot:
    return parse_lock(text, now_wall=now, self_holder=self_holder)


def at(text: str) -> float:
    """The Unix time of an ISO instant, for a lock fixture's ``now``."""
    return datetime.fromisoformat(text).timestamp()


async def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


# ── Parsing the lock (section 3.3) ──


def test_parse_lock_reads_the_local_ai_form() -> None:
    snapshot = parse(NIGHT, at("2026-10-20T23:30:09-04:00"))
    assert snapshot.present and snapshot.readable
    assert snapshot.holder == "local_ai-benchmark session"
    assert snapshot.purpose.startswith("Night chain:")
    assert snapshot.start == "2026-10-20T23:30:09-0400"
    assert snapshot.eta == "2026-10-21T12:00:09-0400 (750 min)"
    assert snapshot.run == ""
    assert snapshot.stale is False


def test_parse_lock_reads_the_agent_benchmark_form() -> None:
    start = datetime(2026, 10, 27, 3, 6, 8).timestamp()
    snapshot = parse(OVERNIGHT, start)
    assert snapshot.holder == "agent-benchmark session (personal-agent-bench)"
    assert snapshot.start == "2026-10-27 03:06:08 EDT"
    assert snapshot.eta == "330 min"
    assert snapshot.stale is False


def test_parse_lock_reads_run_and_mem_gib_and_ignores_unknown_lines() -> None:
    text = (
        "holder: local_ai-benchmark session\n"
        "purpose: one run\n"
        "run: model-eval-42\n"
        "mem_gib: 70\n"
        "note: this key is ignored\n"
    )
    snapshot = parse(text, at("2026-10-20T23:30:09-04:00"))
    assert snapshot.run == "model-eval-42"
    assert snapshot.mem_gib == 70.0
    assert snapshot.present and snapshot.readable


def test_parse_lock_first_occurrence_wins_and_crlf_is_stripped() -> None:
    text = (
        "holder: first\r\n"
        "holder: second\r\n"
        "purpose: p\r\n"
        "run: r1\r\n"
        "run: r2\r\n"
    )
    snapshot = parse(text, at("2026-10-20T23:30:09-04:00"))
    assert snapshot.holder == "first"
    assert snapshot.run == "r1"
    assert "lock_duplicate_holder" in snapshot.alarms


def test_an_absent_file_is_none_and_an_empty_file_is_unreadable(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.lock"
    snapshot = read_lock_file(missing, now_wall=0.0)
    assert snapshot.present is False
    assert classify_lock(snapshot, "any") == "none"
    empty = tmp_path / "empty.lock"
    empty.write_bytes(b"")
    snapshot = read_lock_file(empty, now_wall=0.0)
    assert snapshot.present is True
    assert snapshot.readable is False
    assert classify_lock(snapshot, "any") == "foreign"


def test_bad_utf8_a_directory_a_permission_error_and_an_oversized_file_are_unreadable(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.lock"
    bad.write_bytes(b"holder: caf\xe9\n")
    assert read_lock_file(bad, now_wall=0.0).readable is False
    directory = tmp_path / "dir.lock"
    directory.mkdir()
    assert read_lock_file(directory, now_wall=0.0).readable is False
    huge = tmp_path / "huge.lock"
    huge.write_bytes(b"holder: x\n" + b"a" * (64 * 1024))
    assert read_lock_file(huge, now_wall=0.0).readable is False
    if os.geteuid() == 0:
        pytest.skip("root can read a chmod 0 file")
    denied = tmp_path / "denied.lock"
    denied.write_text("holder: x\n", encoding="utf-8")
    denied.chmod(0)
    try:
        assert read_lock_file(denied, now_wall=0.0).readable is False
    finally:
        denied.chmod(0o600)


def test_eta_forms_and_staleness() -> None:
    start = "2026-10-20T23:30:09+00:00"
    eta = "2026-10-20T23:40:09+00:00"
    text = f"holder: h\nstart: {start}\neta: {eta}\n"
    target = at(eta)
    assert parse(text, target + 599.0).stale is False
    assert parse(text, target + 600.0).stale is True
    with_minutes = f"holder: h\nstart: {start}\neta: {eta} (10 min)\n"
    assert parse(with_minutes, target + 600.0).stale is True
    relative = f"holder: h\nstart: {start}\neta: 10 min\n"
    assert parse(relative, target + 600.0).stale is True
    missing_start = "holder: h\neta: 10 min\n"
    assert parse(missing_start, target + 10_000.0).stale is False
    bad_eta = f"holder: h\nstart: {start}\neta: not a date\n"
    snapshot = parse(bad_eta, target + 10_000.0)
    assert snapshot.stale is False
    assert snapshot.eta == "not a date"


def test_a_huge_relative_eta_is_not_an_error(tmp_path: Path) -> None:
    text = (
        "holder: evil\n"
        "start: 2026-01-01T00:00:00\n"
        "eta: 99999999999999999999999999 min\n"
    )
    snapshot = parse(text, at("2026-01-01T00:00:00"))
    assert snapshot.readable and snapshot.stale is False
    assert snapshot.eta == "99999999999999999999999999 min"
    path = tmp_path / "gpu.lock"
    path.write_text(text, encoding="utf-8")

    async def start_stop() -> None:
        monitor = MachineMonitor(
            lock_path=path, use_kqueue=False, lock_poll_s=0.01, settle_s=0.01
        )
        await monitor.start()
        try:
            payload = monitor.payload()
            assert payload is not None
            assert payload["lock"] == "foreign" and payload["readable"] is True
        finally:
            await monitor.stop()

    asyncio.run(start_stop())


def test_lock_self_exact_and_holder_run_forms() -> None:
    snapshot = parse("holder: local_ai-benchmark session\nrun: r1\n", 0.0)
    assert classify_lock(snapshot, "local_ai-benchmark session") == "self"
    assert classify_lock(snapshot, "local_ai-benchmark session:r1") == "self"
    assert classify_lock(snapshot, "local_ai-benchmark session:r2") == "foreign"
    assert classify_lock(snapshot, "local_ai-benchmark") == "foreign"
    assert classify_lock(snapshot, "LOCAL_AI-BENCHMARK SESSION") == "foreign"
    assert classify_lock(snapshot, "") == "foreign"
    near = parse("holder: local_ai-benchmark session (copy)\n", 0.0)
    assert classify_lock(near, "local_ai-benchmark session") == "foreign"
    colon_holder = parse("holder: bench:run\nrun: task\n", 0.0)
    assert classify_lock(colon_holder, "bench:run:task") == "self"
    carried = parse_lock(
        "holder: bench\nrun: r1\n", now_wall=0.0, self_holder="bench:r1"
    )
    assert classify(carried) == "self"


def test_classify_lock_is_foreign_for_unreadable_and_self_for_self() -> None:
    assert classify_lock(LockSnapshot.absent(), "x") == "none"
    assert classify_lock(LockSnapshot.unreadable("boom"), "x") == "foreign"
    assert classify_lock(parse("holder: x\n", 0.0), "x") == "self"


# ── Parsing the battery (section 5) ──


def test_parse_battery_ac_charged_battery_discharging_charging_no_battery_garbage() -> (
    None
):
    ac = parse_battery(
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=36307043)\t100%; charged; 0:00 remaining present: true\n"
    )
    assert (ac.on_ac, ac.pct, ac.state, ac.known) == (True, 100.0, "charged", True)
    discharging = parse_battery(
        "Now drawing from 'Battery Power'\n"
        " -InternalBattery-0 (id=1)\t74%; discharging; 3:03 remaining present: true\n"
    )
    assert (discharging.on_ac, discharging.pct, discharging.state) == (
        False,
        74.0,
        "discharging",
    )
    charging = parse_battery(
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=1)\t74%; charging; 3:03 remaining\n"
    )
    assert (charging.on_ac, charging.pct, charging.state) == (True, 74.0, "charging")
    no_battery = parse_battery("Now drawing from 'AC Power'\n")
    assert (no_battery.on_ac, no_battery.pct) == (True, None)
    garbage = parse_battery("something else\n")
    assert (garbage.on_ac, garbage.pct, garbage.known) == (None, None, True)
    assert parse_battery("").known is False


def test_a_battery_band_matches_s1_hysteresis() -> None:
    def band(pct: float, previous: int) -> int:
        return battery_band(Machine(on_ac=False, battery_pct=pct), previous)

    # Entry: 59 is band 2 from any higher band; 60 enters 4 but hysteresis
    # keeps an existing lower band until 80.
    assert band(59.0, 4) == 2
    assert band(60.0, 2) == 2
    assert band(80.0, 2) == 4
    assert band(50.0, 4) == 2
    assert band(20.0, 2) == 1
    assert band(10.0, 1) == 0
    assert band(50.0, 0) == 1
    assert band(20.0, 0) == 0
    assert battery_band(Machine(on_ac=True, battery_pct=10.0), 0) == 4
    assert battery_band(Machine(on_ac=False, battery_pct=None), 0) == 4


async def test_the_battery_reader_keeps_the_last_good_state_on_a_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    readings: list[object] = [
        "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t55%; discharging; 3:00 remaining\n",
        RuntimeError("pmset failed"),
        "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining\n",
    ]

    def runner() -> str:
        value = readings.pop(0)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, str)
        return value

    seen: list[BatterySnapshot] = []
    reader = BatteryReader(runner, seen.append, poll_s=0.01)
    first = await reader.poll_once()
    assert (first.on_ac, first.pct) == (False, 55.0)
    with caplog.at_level("WARNING"):
        second = await reader.poll_once()
    assert second == first
    assert sum("battery unknown" in record.message for record in caplog.records) == 1
    third = await reader.poll_once()
    assert (third.on_ac, third.pct) == (True, 100.0)
    assert [snapshot.pct for snapshot in seen] == [55.0, 100.0]


async def test_the_battery_reader_rejects_garbage_and_keeps_the_last_good(
    caplog: pytest.LogCaptureFixture,
) -> None:
    readings = [
        "Now drawing from 'Battery Power'\n"
        " -InternalBattery-0 (id=1)\t55%; discharging; 3:00 remaining\n",
        "something else\n",
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining\n",
    ]

    def runner() -> str:
        return readings.pop(0)

    reader = BatteryReader(runner, lambda snapshot: None, poll_s=0.01)
    first = await reader.poll_once()
    with caplog.at_level("WARNING"):
        second = await reader.poll_once()
    assert second == first
    assert sum("battery unknown" in record.message for record in caplog.records) == 1
    third = await reader.poll_once()
    assert (third.on_ac, third.pct) == (True, 100.0)


# ── The lock watcher (section 4) ──


async def test_the_lock_watcher_emits_one_change_per_state_and_none_for_the_same_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gpu.lock"
    seen: list[LockSnapshot] = []
    watcher = LockWatcher(
        path, seen.append, poll_s=0.01, settle_s=0.01, use_kqueue=False
    )
    await watcher.start()
    try:
        await wait_until(lambda: len(seen) == 1)
        assert seen[0].present is False
        path.write_text("holder: a\n", encoding="utf-8")
        await wait_until(lambda: len(seen) == 2)
        assert seen[1].holder == "a"
        path.write_text("holder: a\n", encoding="utf-8")
        await asyncio.sleep(0.05)
        assert len(seen) == 2
        path.unlink()
        await wait_until(lambda: len(seen) == 3)
        assert seen[2].present is False
    finally:
        await watcher.stop()


async def test_the_lock_watcher_survives_an_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    path.write_text("holder: a\n", encoding="utf-8")
    seen: list[LockSnapshot] = []
    watcher = LockWatcher(
        path, seen.append, poll_s=0.01, settle_s=0.01, use_kqueue=False
    )
    await watcher.start()
    try:
        await wait_until(lambda: len(seen) == 1 and seen[0].holder == "a")
        replacement = tmp_path / "new.lock"
        replacement.write_text("holder: b\n", encoding="utf-8")
        os.replace(replacement, path)
        await wait_until(lambda: len(seen) >= 2 and seen[-1].holder == "b")
        assert [snapshot.holder for snapshot in seen] == ["a", "b"]
        assert all(snapshot.present for snapshot in seen)
    finally:
        await watcher.stop()


async def test_the_lock_watcher_holds_a_two_step_write_to_one_state_or_two_changes_but_never_a_foreign_none_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gpu.lock"
    path.write_bytes(b"")
    seen: list[LockSnapshot] = []
    watcher = LockWatcher(
        path, seen.append, poll_s=0.01, settle_s=0.01, use_kqueue=False
    )
    await watcher.start()
    try:
        await wait_until(lambda: len(seen) >= 1)
        path.write_text("holder: a\n", encoding="utf-8")
        await wait_until(lambda: len(seen) >= 2 and seen[-1].holder == "a")
        states = [classify_lock(snapshot, "") for snapshot in seen]
        assert "self" not in states
        for first, second in zip(states, states[1:], strict=False):
            assert not (first == "foreign" and second == "none")
    finally:
        await watcher.stop()


async def test_a_kqueue_watcher_sees_a_create_and_a_delete(tmp_path: Path) -> None:
    if not hasattr(__import__("select"), "kqueue"):
        pytest.skip("kqueue is not available")
    path = tmp_path / "gpu.lock"
    seen: list[LockSnapshot] = []
    watcher = LockWatcher(
        path,
        seen.append,
        poll_s=0.2,
        settle_s=0.01,
        guard_poll_s=0.2,
        use_kqueue=True,
    )
    await watcher.start()
    try:
        await wait_until(lambda: len(seen) == 1)
        path.write_text("holder: a\n", encoding="utf-8")
        await wait_until(lambda: bool(seen) and seen[-1].holder == "a", timeout=0.5)
        path.unlink()
        await wait_until(lambda: bool(seen) and not seen[-1].present, timeout=0.5)
    finally:
        await watcher.stop()


async def test_the_polling_fallback_sees_the_same(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    seen: list[LockSnapshot] = []
    watcher = LockWatcher(
        path, seen.append, poll_s=0.01, settle_s=0.01, use_kqueue=False
    )
    await watcher.start()
    try:
        await wait_until(lambda: len(seen) == 1)
        path.write_text("holder: a\n", encoding="utf-8")
        await wait_until(lambda: seen[-1].holder == "a")
        path.unlink()
        await wait_until(lambda: not seen[-1].present)
    finally:
        await watcher.stop()


# ── machine.jsonl (section 6.4) ──


async def test_machine_jsonl_lines_are_canonical_0600_and_read_back(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "gpu.lock"
    gate = ChatGate(call_log_dir=tmp_path / "scheduler", lock_gate="wait")
    on_battery = False

    def runner() -> str:
        if on_battery:
            return (
                "Now drawing from 'Battery Power'\n"
                " -InternalBattery-0 (id=1)\t55%; discharging; 3:00 remaining\n"
            )
        return (
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining\n"
        )

    monitor = MachineMonitor(
        lock_path=lock,
        lock_gate="wait",
        write=gate.write_machine,
        on_change=lambda snapshot: gate.set_machine(
            snapshot.machine, payload=snapshot.payload()
        ),
        battery_runner=runner,
        use_kqueue=False,
        lock_poll_s=0.01,
        battery_poll_s=0.01,
        settle_s=0.01,
    )
    await monitor.start()
    try:
        lock.write_text("holder: a\n", encoding="utf-8")
        await wait_until(
            lambda: (monitor.payload() or {}).get("lock") == "foreign", timeout=0.5
        )
        on_battery = True
        await wait_until(
            lambda: (monitor.payload() or {}).get("battery_pct") == 55.0,
            timeout=0.5,
        )
        assert gate.machine_payload()["band"] == 2
    finally:
        await monitor.stop()
        await gate.close()
    path = gate.machine_log_path
    assert path is not None and path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = read_machine(path)
    assert [line["event"] for line in lines] == ["start", "lock", "battery", "stop"]
    assert all(line["schema"] == MACHINE_SCHEMA for line in lines)
    assert lines[1]["holder"] == "a" and lines[1]["lock"] == "foreign"
    assert lines[2]["battery_pct"] == 55.0 and lines[2]["band"] == 2
    assert lines[0]["at_s"] >= 0.0 and lines[0]["wall_s"] > 0.0


def test_a_machine_log_write_failure_warns_once_and_does_not_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gate = ChatGate(call_log_dir=tmp_path)
    path = gate.machine_log_path
    assert path is not None
    # Replace the machine fd with one that refuses writes (a closed fd).
    assert gate._log._machine_fd is not None
    os.close(gate._log._machine_fd)
    with caplog.at_level("WARNING"):
        gate.write_machine({"event": "lock"})
        assert gate.machine_log_path is None
    assert sum("Machine log write failed" in r.message for r in caplog.records) == 1
    assert gate.call_log_path is not None  # the call log is untouched
    # The call log still works.
    gate._log.write({"task_key": "a"})
    assert gate.call_log_path is not None


def test_read_machine_rejects_a_wrong_schema_or_version(tmp_path: Path) -> None:
    import json

    path = tmp_path / "machine.jsonl"
    for line in (
        {"schema": "localmelo.sched.trace", "version": 1, "event": "start"},
        {"schema": MACHINE_SCHEMA, "version": 2, "event": "start"},
        {"schema": MACHINE_SCHEMA, "version": True, "event": "start"},
    ):
        path.write_text(json.dumps(line) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_machine(path)


def test_no_code_writes_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    original = "holder: local_ai-benchmark session\nrun: r1\n"
    path.write_text(original, encoding="utf-8")
    snapshot = read_lock_file(path, now_wall=0.0)
    assert snapshot.holder == "local_ai-benchmark session"
    assert path.read_text(encoding="utf-8") == original

    # A grep-style check over both modules: the lock reader opens read-only;
    # nothing creates, writes, renames or removes a lock.
    for name in ("machine_monitor.py", "chat_gate.py"):
        source = (Path("support/gateway") / name).read_text(encoding="utf-8")
        for pattern in (
            'open(path, "w"',
            'open(target, "w"',
            "write_text",
            "write_bytes",
            "unlink(",
            "os.remove",
            "os.rename",
            "os.replace",
        ):
            assert pattern not in source, f"{name} uses {pattern}"
