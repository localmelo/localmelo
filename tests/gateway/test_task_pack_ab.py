"""The task-pack A/B script builds both arms' first requests with the real code.

Two of its cases, on a stand-in model: before the handover the task's request
does not say whom the work is about; after, it does, and the stand-in's reply
is judged right. No model server; everything lives in a temporary folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "task_pack_ab.py"


@pytest.fixture(scope="module")
def ab() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task_pack_ab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["task_pack_ab"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["resume", "deadline"])
async def test_the_after_arm_holds_the_subject_and_the_before_arm_does_not(
    ab: ModuleType, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    # The script puts its recording model in the workers' builder: undone after the test.
    monkeypatch.setattr(
        "localmelo.support.gateway.workers.build_session_agent",
        ab.workers_module.build_session_agent,
    )
    (case,) = [case for case in ab.CASES if case.name == name]
    before = await ab.first_request(case, tmp_path / "before", after=False)
    after = await ab.first_request(case, tmp_path / "after", after=True)
    assert not before["holds_right"] and after["holds_right"]
    assert after["message_chars"] > before["message_chars"]
    assert not (before["spec"]["chat_turns"] or before["spec"]["chat_picked"])
    for arm, expected in ((before, "unclear"), (after, "right")):
        reply = await ab.Stub(case).chat(arm["messages"], arm["tools"])
        assert ab.verdict(case, reply) == expected


def _args(ab: ModuleType, **values: object) -> argparse.Namespace:
    base = {
        "stub": True,
        "base_url": "",
        "model": "",
        "reps": 2,
        "jobs": 8,
        "request_timeout": 5.0,
        "cases": ["resume", "deadline", "look-up-zh"],
        "out": "",
        "dump": "",
    }
    return argparse.Namespace(**{**base, **values})


async def test_many_slots_give_the_rows_one_slot_gives(
    ab: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.gateway.workers.build_session_agent",
        ab.workers_module.build_session_agent,
    )
    one = await ab.run(_args(ab, jobs=1))
    four = await ab.run(_args(ab, jobs=4))
    assert four["cases"] == one["cases"]
    # The cases keep the script's own order.
    assert [row["case"] for row in one["cases"]] == ["resume", "look-up-zh", "deadline"]
    assert four["summary"] == one["summary"]


async def test_a_run_again_reuses_the_calls_it_finished(
    ab: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.gateway.workers.build_session_agent",
        ab.workers_module.build_session_agent,
    )
    args = _args(ab, out=str(tmp_path / "results.json"), cases=["resume"], reps=1)
    kept = ab.calls_file(args)
    # A first run stopped after one call, and a last line cut short.
    entry = {"case": "resume", "arm": "after", "rep": 0}
    kept.write_text(
        json.dumps(ab.calls_key(args))
        + "\n"
        + json.dumps({**entry, "verdict": "right", "reply": "kept from before"})
        + '\n{"case": "resume", "arm": "bef',
        encoding="utf-8",
    )
    rows = (await ab.run(args))["cases"]
    assert rows[0]["arms"]["after"]["replies"] == ["kept from before"]
    assert rows[0]["arms"]["before"]["replies"] == [
        "First I look into your own profile."
    ]
    # Another model starts the file anew.
    other = _args(ab, out=args.out, cases=["resume"], reps=1, stub=False, model="m")
    assert ab.done_calls(kept, ab.calls_key(other)) == {}
    assert kept.read_text("utf-8").splitlines() == [json.dumps(ab.calls_key(other))]
