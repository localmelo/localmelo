"""melo-procedures: episodes, replay, export and forget.

Every test runs ``main([...])`` in process with files under ``tmp_path``.
The replay tests use ``tests/memory/procedures/fixtures/bench_episodes.jsonl``:
20 bench episodes in four groups of five, reduced to synthetic ids and
keys, step counts, statuses, and class sequences with pass markers. Their
expected values come from an independent implementation of the reference
rules below, not from the code under test.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import statistics
import tomllib
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.procedures.normalize import key_for_query
from localmelo.melo.memory.procedures.store import RunFacts, SqliteProcedures
from localmelo.melo.memory.procedures.trace import Segment
from localmelo.melo.memory.retrieval import RawCall
from localmelo.melo.schema import ToolCall, ToolResult
from localmelo.support.procedures import main
from localmelo.tests.memory.history_fixture import write_history_sync

FIXTURE = (
    Path(__file__).parents[1]
    / "memory"
    / "procedures"
    / "fixtures"
    / "bench_episodes.jsonl"
)
CAP = 0.95


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines()]


# ── flags ───────────────────────────────────────────────────────────────────


def test_help_and_usage(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--help")
    assert code == 0 and out.startswith("usage: melo-procedures")
    code, _, err = run(capsys)
    assert code == 2 and "required" in err
    code, _, err = run(capsys, "replay", str(FIXTURE), "--protocol", "nope")
    assert code == 2 and "invalid choice" in err
    code, _, err = run(capsys, "replay", str(FIXTURE), "--spread", "-1")
    assert code == 2 and "must be a number > 0" in err
    code, _, err = run(capsys, "forget")
    assert code == 2 and "one of the arguments --routine --all is required" in err


def test_the_entry_point() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["melo-procedures"] == "localmelo.support.procedures:main"


# ── episodes ────────────────────────────────────────────────────────────────


def shell(call_id: str, command: str) -> ToolCall:
    return ToolCall(tool_name="shell_exec", arguments={"command": command}, id=call_id)


def result(output: str) -> ToolResult:
    return ToolResult(tool_name="shell_exec", output=output)


CALLS = [
    RawCall(
        task_id="t1",
        seq=0,
        step=1,
        content="Look first.",
        tool_calls=(
            ToolCall(
                tool_name="file_read", arguments={"path": "/w/test_a.py"}, id="c1"
            ),
            ToolCall(tool_name="file_read", arguments={"path": "/w/a.py"}, id="c2"),
        ),
        tool_results=(
            ToolResult(tool_name="file_read", output="tests"),
            ToolResult(tool_name="file_read", output="code"),
        ),
        outcome="tools",
        timestamp=100.0,
    ),
    RawCall(
        task_id="t1",
        seq=1,
        step=2,
        content="Look first.",
        tool_calls=(shell("c3", "pytest -q"),),
        tool_results=(result("1 failed, 2 passed in 0.1s"),),
        outcome="tools",
        timestamp=110.0,
    ),
    RawCall(
        task_id="t1",
        seq=2,
        kind="reflect",
        step=2,
        attempt=1,
        outcome="aux",
        timestamp=115.0,
    ),
    RawCall(
        task_id="t1",
        seq=3,
        step=3,
        attempt=2,
        content="Fix.",
        tool_calls=(
            ToolCall(
                tool_name="file_write",
                arguments={"path": "/w/a.py", "content": "x"},
                id="c4",
            ),
        ),
        tool_results=(ToolResult(tool_name="file_write", output="ok"),),
        outcome="tools",
        timestamp=120.0,
    ),
    RawCall(
        task_id="t1",
        seq=4,
        step=4,
        attempt=2,
        content="Test.",
        tool_calls=(shell("c5", "pytest -q | tail -1"),),
        tool_results=(result("3 passed in 0.1s"),),
        outcome="tools",
        timestamp=130.0,
    ),
    RawCall(
        task_id="t1",
        seq=5,
        step=5,
        attempt=2,
        content="Done.",
        outcome="answer",
        timestamp=140.0,
    ),
]

RESULT = {
    "task_id": "t1",
    "status": "completed",
    "steps": 5,
    "attempts": 2,
    "n_tool": 5,
    "duration_s": 42.5,
    "config": {
        "max_steps": 30,
        "attempt_policy": "reflect",
        "steps_per_attempt": 10,
        "max_attempts": 3,
        "reflection_gate": "flags",
        "timeout_s": None,
    },
    "localmelo_version": "0.1.0",
}


def write_result(tmp_path: Path, **changes: Any) -> str:
    path = tmp_path / "result.json"
    path.write_text(
        "Some text output\n" + json.dumps({**RESULT, **changes}) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_episodes_with_llm_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_history_sync(tmp_path / "history.db", CALLS)
    code, out, err = run(
        capsys,
        "episodes",
        "--history",
        str(tmp_path),
        "--model",
        "model-a",
        "--result-json",
        write_result(tmp_path),
    )
    assert code == 0, err
    routine, episode = lines(out)
    assert routine["table"] == "routines" and episode["table"] == "episodes"
    key = key_for_query("fixture task t1", "")
    assert routine["key"] == key.key and routine["query_text"] == "fixture task t1"
    assert episode["routine_id"] == routine["routine_id"]
    assert episode["model"] == "model-a" and episode["harness_version"] == "0.1.0"
    assert (episode["status"], episode["steps"], episode["attempts"]) == (
        "completed",
        5,
        2,
    )
    assert (episode["n_tool"], episode["duration_s"]) == (5, 42.5)
    assert (episode["first_pass_step"], episode["cycles"]) == (4, 1)
    segments = [Segment.from_dict(s) for s in json.loads(episode["segments_json"])]
    assert [
        (s.cls, s.label, s.steps, s.calls, s.attempt, s.passed) for s in segments
    ] == [
        ("R", "Read test_a.py", (1, 1), 1, 1, False),
        ("R", "Read a.py", (1, 1), 1, 1, False),
        ("T", "pytest", (2, 2), 1, 1, False),
        ("W", "Edit a.py", (3, 3), 1, 2, False),
        ("T", "pytest", (4, 4), 1, 2, True),
    ]
    # The template id hashes the model and the result's length options.
    assert episode["template_id"].startswith(routine["routine_id"] + ":")


def test_episodes_without_llm_calls_group_rows_by_thought(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_history_sync(tmp_path / "history.db", CALLS, llm_calls=False)
    code, out, err = run(capsys, "episodes", "--history", str(tmp_path / "history.db"))
    assert code == 0, err
    _, episode = lines(out)
    segments = [Segment.from_dict(s) for s in json.loads(episode["segments_json"])]
    # "Look first." starts step 1; the second call's "" is another thought,
    # so it starts step 2; the third repeats "Look first." after "" (step 3).
    assert [s.steps for s in segments] == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert episode["steps"] == 5
    assert episode["model"] == "" and episode["harness_version"] == "unknown"
    assert episode["status"] == "running"


def test_episodes_with_a_key_and_a_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_history_sync(tmp_path / "history.db", CALLS)
    code, out, _ = run(
        capsys, "episodes", "--history", str(tmp_path), "--key", "abc123"
    )
    assert code == 0
    routine, _ = lines(out)
    assert routine["key"] == "abc123"
    code, out, _ = run(capsys, "episodes", "--history", str(tmp_path), "--place", "/p")
    assert lines(out)[0]["key"] == key_for_query("fixture task t1", "/p").key


def test_episodes_input_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(capsys, "episodes", "--history", str(tmp_path / "none.db"))
    assert code == 2 and "does not exist" in err
    write_history_sync(tmp_path / "history.db", CALLS)
    code, _, err = run(
        capsys,
        "episodes",
        "--history",
        str(tmp_path),
        "--result-json",
        write_result(tmp_path, task_id="t9"),
    )
    assert code == 2 and "task t9 is not in the history" in err
    bad = tmp_path / "bad.json"
    bad.write_text("not json\n", encoding="utf-8")
    code, _, err = run(
        capsys, "episodes", "--history", str(tmp_path), "--result-json", str(bad)
    )
    assert code == 2 and "does not end with a JSON object" in err
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not sqlite" * 100)
    code, _, err = run(capsys, "episodes", "--history", str(junk))
    assert code == 2 and "melo-procedures: error:" in err


def test_episodes_feed_the_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = []
    for index in range(3):
        directory = tmp_path / f"run{index}"
        directory.mkdir()
        calls = [RawCall(**{**c.__dict__, "task_id": f"r{index}"}) for c in CALLS]
        write_history_sync(directory / "history.db", calls)
        code, out, _ = run(
            capsys,
            "episodes",
            "--history",
            str(directory),
            "--model",
            "m",
            "--key",
            "same",
            "--result-json",
            write_result(directory, task_id=f"r{index}"),
        )
        assert code == 0
        rows.extend(out.splitlines())
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(rows) + "\n", encoding="utf-8")
    code, out, _ = run(capsys, "replay", str(episodes))
    (summary,) = lines(out)
    assert summary["runs"] == 3
    assert summary["levels"] == {"0": 1, "1": 1, "2": 1}


# ── replay on the fixture ───────────────────────────────────────────────────


def fixture_groups() -> list[list[dict[str, Any]]]:
    rows = [
        json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    ]
    episodes = sorted(
        (r for r in rows if r["table"] == "episodes"), key=lambda r: r["created_at"]
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault(episode["template_id"], []).append(episode)
    return list(groups.values())


def reference_error(
    n: int, window: list[int], *, rule: str = "dynamic"
) -> tuple[float, float, bool]:
    """Section 6.3's bar (or the median one) on a run of *n* steps, frozen
    when the run outlasts the window: (bar error, share at the cap, longer)."""
    m = statistics.median(window)
    shown, frozen, errors, capped = 0.0, False, [], 0
    for k in range(1, n + 1):
        frozen = frozen or k > max(window)
        if not frozen:
            if rule == "median" or k < m:
                raw = k / m
            else:
                longer = [s - k for s in window if s > k]
                raw = k / (k + statistics.median(longer)) if longer else k / (k + 1)
            shown = min(CAP, max(shown, raw))
        errors.append(abs(shown - k / n))
        capped += shown >= CAP
    return statistics.fmean(errors), capped / n, frozen


def reference_online(drop: str) -> tuple[list[float], list[float]]:
    """The designed online gate (lead's rule): shown and held bar errors."""
    shown: list[float] = []
    held: list[float] = []
    for runs in fixture_groups():
        for i, episode in enumerate(runs):
            prior = [r for r in runs[:i] if r["status"] == "completed"]
            if len(prior) < 2:
                continue
            first = runs[0]
            window = prior[-8:]
            later = [r["steps"] for r in prior if r is not first]
            if (
                first in window
                and len(later) >= 2
                and (
                    drop == "unconditional"
                    or (drop == "conditional" and first["steps"] > max(later))
                )
            ):
                window = [r for r in window if r is not first]
            steps = [r["steps"] for r in window]
            m = statistics.median(steps)
            gate = len(steps) >= 2 and (max(steps) - min(steps)) / m <= 0.5
            (shown if gate else held).append(
                reference_error(episode["steps"], steps)[0]
            )
    return shown, held


@pytest.mark.parametrize("drop", ["conditional", "off", "unconditional"])
def test_online_replay_matches_the_reference(
    capsys: pytest.CaptureFixture[str], drop: str
) -> None:
    code, out, _ = run(capsys, "replay", str(FIXTURE), "--drop", drop)
    assert code == 0
    (summary,) = lines(out)
    shown, held = reference_online(drop)
    assert summary["level2"]["runs"] == len(shown)
    assert summary["held"]["runs"] == len(held)
    assert summary["level2"]["bar_error"] == pytest.approx(
        statistics.fmean(shown), abs=1e-4
    )
    assert summary["held"]["bar_error"] == pytest.approx(
        statistics.fmean(held), abs=1e-4
    )
    assert summary["level2"]["backward_moves"] == 0
    assert summary["params"]["drop"] == drop


def test_online_replay_pinned_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "replay", str(FIXTURE))
    (summary,) = lines(out)
    assert summary["protocol"] == "online" and summary["arm"] == "default"
    assert summary["runs"] == 20
    # Level 0: each group's first run, and the second of a group whose first
    # run did not complete; level 2: four runs whose window passed the gate.
    assert summary["levels"] == {"0": 5, "1": 11, "2": 4}
    assert summary["level2"]["runs"] == 4 and summary["held"]["runs"] == 6
    assert summary["model_demoted"] == 0
    assert set(summary["thresholds"]) == {
        "bar_error",
        "separation",
        "backward_moves",
        "cap_share",
        "end_jump",
        "longer_rate",
        "phase_agreement",
    }
    assert summary["level2_by_prior_runs"]
    assert 0 <= summary["phase_agreement"] <= 1


def test_loo_replay_matches_the_reference(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "replay", str(FIXTURE), "--protocol", "loo")
    assert code == 0
    by_rule = {line["denominator"]: line for line in lines(out)}
    assert list(by_rule) == ["median", "dynamic", "milestone", "hybrid"]
    for rule in ("median", "dynamic"):
        results = []
        for runs in fixture_groups():
            for i, episode in enumerate(runs):
                others = [r["steps"] for j, r in enumerate(runs) if j != i]
                results.append(reference_error(episode["steps"], others, rule=rule))
        assert by_rule[rule]["runs"] == 20
        assert by_rule[rule]["bar_error"] == pytest.approx(
            statistics.fmean(e for e, _, _ in results), abs=1e-4
        )
        assert by_rule[rule]["cap_share"] == pytest.approx(
            statistics.fmean(c for _, c, _ in results), abs=1e-4
        )


def test_first_k_replay_matches_the_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _ = run(
        capsys,
        "replay",
        str(FIXTURE),
        "--protocol",
        "first-k",
        "--denominator",
        "median",
    )
    (line,) = lines(out)
    shown: list[float] = []
    held: list[float] = []
    for runs in fixture_groups():
        prior = [r["steps"] for r in runs[:2]]
        gate = (max(prior) - min(prior)) / statistics.median(prior) <= 0.5
        for episode in runs[2:]:
            (shown if gate else held).append(
                reference_error(episode["steps"], prior, rule="median")[0]
            )
    assert line["shown"]["runs"] == len(shown) and line["held"]["runs"] == len(held)
    assert line["shown"]["bar_error"] == pytest.approx(
        statistics.fmean(shown), abs=1e-4
    )
    assert line["held"]["bar_error"] == pytest.approx(statistics.fmean(held), abs=1e-4)


def test_the_arms(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "replay", str(FIXTURE), "--arms")
    assert code == 0
    assert [line["arm"] for line in lines(out)] == [
        "default",
        "denominator=median",
        "denominator=milestone",
        "denominator=hybrid",
        "spread=0.4",
        "spread=0.6",
        "drop=off",
        "drop=unconditional",
        "stats=normal",
    ]
    for line in lines(out):
        assert line["level2"].get("backward_moves", 0) == 0


def test_the_build_in_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = FIXTURE.read_text(encoding="utf-8").splitlines()
    builds = []
    for number, line in enumerate(rows):
        row = json.loads(line)
        if row["table"] == "episodes":
            # The four groups interleave: each group alternates builds.
            row["harness_version"] = f"build-{number // 4 % 2}"
        builds.append(json.dumps(row))
    path = tmp_path / "builds.jsonl"
    path.write_text("\n".join(builds), encoding="utf-8")
    _, without, _ = run(capsys, "replay", str(path))
    _, within, _ = run(capsys, "replay", str(path), "--key-build")
    assert lines(without)[0]["levels"] == {"0": 5, "1": 11, "2": 4}
    assert lines(within)[0]["key_build"] is True
    assert lines(within)[0]["levels"]["0"] > 5


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "1: "),
        ('{"table": "logs"}', "unknown table 'logs'"),
        ('{"table": "episodes", "task_id": "x"}', "1: 'routine_id'"),
        ("[1, 2]", "not a JSON object"),
    ],
)
def test_replay_input_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(content + "\n", encoding="utf-8")
    code, _, err = run(capsys, "replay", str(path))
    assert code == 2
    assert err.startswith(f"melo-procedures: error: {path}:") and message in err


def test_replay_refuses_duplicates_and_empty_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode = next(
        line
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if '"episodes"' in line
    )
    twice = tmp_path / "twice.jsonl"
    twice.write_text(f"{episode}\n{episode}\n", encoding="utf-8")
    code, _, err = run(capsys, "replay", str(twice))
    assert code == 2 and "appears twice" in err
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    code, _, err = run(capsys, "replay", str(empty))
    assert code == 2 and "no episodes" in err
    code, _, err = run(capsys, "replay", str(tmp_path / "missing.jsonl"))
    assert code == 2 and "cannot read" in err


# ── export and forget ───────────────────────────────────────────────────────


async def fill(path: Path) -> list[str]:
    store = SqliteProcedures(path)
    ids = []
    try:
        for index, text in enumerate(["first task", "second task"]):
            result = await store.learn(
                RunFacts(
                    task_id=f"t{index}",
                    key=key_for_query(text, "/p"),
                    model="m",
                    options={},
                    status="completed",
                    steps=4,
                )
            )
            ids.append(result.routine.routine_id)
    finally:
        await store.aclose()
    return ids


def test_export_and_forget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "procedures.db"
    first, _ = asyncio.run(fill(path))
    code, out, _ = run(capsys, "export", "--db", str(path))
    assert code == 0
    assert [row["table"] for row in lines(out)] == ["routines"] * 2 + [
        "episodes"
    ] * 2 + ["templates"] * 2
    code, out, _ = run(capsys, "forget", "--routine", first, "--db", str(path))
    assert code == 0 and lines(out) == [{"deleted": 3}]
    code, out, _ = run(capsys, "forget", "--all", "--db", str(path))
    assert lines(out) == [{"deleted": 3}]
    code, out, _ = run(capsys, "export", "--db", str(path))
    assert out == ""


def test_the_memory_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(fill(tmp_path / "procedures.db"))
    code, out, _ = run(capsys, "export", "--memory-dir", str(tmp_path))
    assert code == 0 and len(lines(out)) == 6
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path))
    code, out, _ = run(capsys, "export")
    assert code == 0 and len(lines(out)) == 6
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path / "elsewhere"))
    code, _, err = run(capsys, "forget", "--all")
    assert code == 2 and "procedures.db does not exist" in err
    assert not (tmp_path / "elsewhere").exists()


def test_export_refuses_a_file_that_is_not_a_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "procedures.db"
    path.write_bytes(b"garbage" * 1000)
    code, _, err = run(capsys, "export", "--db", str(path))
    assert code == 2 and f"cannot read {path}" in err
    os.chmod(path, 0o600)


def test_separation_is_null_without_held_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = []
    for index, steps in enumerate([6, 7, 6, 7], start=1):
        segments = [{"cls": "R", "label": "x", "steps": [1, steps]}]
        rows.append(
            {
                "table": "episodes",
                "task_id": f"t{index}",
                "routine_id": "r",
                "template_id": "r:t",
                "model": "m",
                "status": "completed",
                "steps": steps,
                "segments_json": json.dumps(segments),
                "created_at": index,
            }
        )
    path = tmp_path / "steady.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    code, out, _ = run(capsys, "replay", str(path))
    (summary,) = lines(out)
    assert summary["held"] == {"runs": 0}
    assert summary["separation"] is None
    assert summary["thresholds"]["separation"] is None
    assert summary["thresholds"]["bar_error"] is True


@pytest.mark.parametrize(
    "corruption",
    [
        "UPDATE llm_calls SET step_ids_json = '{broken' WHERE seq = 0",
        "UPDATE steps SET tool_call_json = 'not json' WHERE seq = 0",
        "UPDATE steps SET tool_result_json = '[1, 2]' WHERE seq = 0",
        "DROP TABLE steps",
    ],
)
def test_episodes_refuse_a_damaged_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], corruption: str
) -> None:
    path = tmp_path / "history.db"
    write_history_sync(path, CALLS)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute(corruption)
        conn.commit()
    code, out, err = run(capsys, "episodes", "--history", str(path))
    assert code == 2 and out == ""
    assert err.startswith(f"melo-procedures: error: cannot read {path}: ")


def test_a_result_without_a_task_needs_a_history_of_one_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    two = [*CALLS, RawCall(task_id="t2", seq=0, step=1, content="x", timestamp=1.0)]
    write_history_sync(tmp_path / "history.db", two)
    result = write_result(tmp_path)
    data = json.loads(Path(result).read_text(encoding="utf-8").splitlines()[-1])
    del data["task_id"]
    Path(result).write_text(json.dumps(data) + "\n", encoding="utf-8")
    code, _, err = run(
        capsys, "episodes", "--history", str(tmp_path), "--result-json", result
    )
    assert code == 2
    assert "names no task_id, and the history holds 2 tasks" in err
    code, out, _ = run(capsys, "episodes", "--history", str(tmp_path))
    assert code == 0 and len(lines(out)) == 4


@pytest.mark.parametrize("command", [["export"], ["forget", "--all"]])
def test_export_and_forget_leave_a_foreign_file_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: list[str]
) -> None:
    path = tmp_path / "procedures.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE notes (text TEXT)")
        conn.execute("INSERT INTO notes VALUES ('mine')")
        conn.commit()
    before = path.read_bytes()
    code, out, err = run(capsys, *command, "--db", str(path))
    assert code == 2 and out == ""
    assert "is not a procedures store" in err
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["procedures.db"]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def episode(
    task_id: str,
    steps: int,
    created_at: float,
    *,
    routine_id: str = "r",
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "table": "episodes",
        "task_id": task_id,
        "routine_id": routine_id,
        "template_id": f"{routine_id}:opts",
        "model": "m",
        "status": status,
        "steps": steps,
        "segments_json": json.dumps([{"cls": "R", "label": "x", "steps": [1, steps]}]),
        "created_at": created_at,
    }


def test_the_end_jump_counts_completed_runs_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        episode("t1", 6, 1),
        episode("t2", 6, 2),
        episode("t3", 6, 3),
        episode("t4", 3, 4, status="budget_exhausted"),
    ]
    code, out, _ = run(capsys, "replay", write_rows(tmp_path / "e.jsonl", rows))
    (summary,) = lines(out)
    assert summary["level2"]["runs"] == 2
    # t3 ends at 6/7 and jumps to 1; t4 stops at 3/6 and does not jump.
    assert summary["level2"]["end_jump"] == pytest.approx(1 - 6 / 7, abs=1e-4)


def test_level2_error_by_number_of_prior_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [episode(f"t{i:02d}", 5, i) for i in range(12)]
    code, out, _ = run(capsys, "replay", write_rows(tmp_path / "e.jsonl", rows))
    (summary,) = lines(out)
    # Ten runs had 2 to 11 earlier completed runs; the window holds 8.
    assert list(summary["level2_by_prior_runs"]) == [str(n) for n in range(2, 12)]
    assert summary["level2_share"] == pytest.approx(10 / 11, abs=1e-4)


def test_a_similar_task_shows_level_1_in_the_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    texts = [
        "summarize today's unread mail into the daily notes file",
        "run pytest in the project and fix the failing tests",
        "convert the sales csv export into a json report",
        "copy the photos folder to the backup drive",
        "delete log files older than a week from the logs directory",
        "summarize the unread mail of today into the daily notes",
    ]
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        routine = f"r{index}"
        rows.append(
            {
                "table": "routines",
                "routine_id": routine,
                "key": f"k{index}",
                "query_text": text,
                "place": "/p",
                "created_at": 0,
            }
        )
        runs = 2 if index < 5 else 1
        for run_index in range(runs):
            rows.append(
                episode(
                    f"{routine}-{run_index}",
                    5,
                    10 * index + run_index,
                    routine_id=routine,
                )
            )
    code, out, _ = run(capsys, "replay", write_rows(tmp_path / "e.jsonl", rows))
    (summary,) = lines(out)
    # Five first runs at level 0, five second runs at 1, the reworded run at 1.
    assert summary["levels"] == {"0": 5, "1": 6, "2": 0}


@pytest.mark.parametrize("protocol", ["loo", "first-k"])
@pytest.mark.parametrize("flag", [["--drop", "off"], ["--stats", "normal"], ["--arms"]])
def test_online_flags_are_refused_with_other_protocols(
    capsys: pytest.CaptureFixture[str], protocol: str, flag: list[str]
) -> None:
    code, _, err = run(capsys, "replay", str(FIXTURE), "--protocol", protocol, *flag)
    assert code == 2
    assert f"{flag[0]} applies only to --protocol online" in err
