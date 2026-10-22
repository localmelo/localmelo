"""RunResult — the record of one finished run."""

from __future__ import annotations

import dataclasses
import json

from localmelo.melo.schema import RunResult

# The fields in the order of the Phase 1a spec (B4). The CLI's JSON output
# prints RunResult.to_dict() and relies on this order.
SPEC_FIELDS = [
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


class TestRunResult:
    def test_fields_are_in_spec_order(self) -> None:
        assert [f.name for f in dataclasses.fields(RunResult)] == SPEC_FIELDS

    def test_defaults(self) -> None:
        assert RunResult(task_id="t1", status="completed").to_dict() == {
            "task_id": "t1",
            "status": "completed",
            "result": "",
            "error": "",
            "steps": 0,
            "attempts": 0,
            "n_llm": 0,
            "n_main": 0,
            "n_aux": 0,
            "n_tool": 0,
            "llm_errors": 0,
            "usage": {},
            "duration_s": 0.0,
            "config": {},
        }

    def test_to_dict_keeps_field_order_and_round_trips_through_json(self) -> None:
        result = RunResult(
            task_id="t2",
            status="budget_exhausted",
            error="Max steps reached",
            steps=3,
            attempts=2,
            n_llm=4,
            n_main=3,
            n_aux=1,
            n_tool=3,
            usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            duration_s=1.25,
            config={
                "max_steps": 3,
                "builtins": ["file_read"],
                "memory_dir": None,
                "timeout_s": None,
            },
        )
        data = result.to_dict()
        assert list(data) == SPEC_FIELDS
        assert json.loads(json.dumps(data)) == data

    def test_to_dict_is_a_deep_copy(self) -> None:
        result = RunResult(
            task_id="t3",
            status="completed",
            usage={"total_tokens": 1},
            config={"tools": ["echo"]},
        )
        data = result.to_dict()
        data["usage"]["total_tokens"] = 99
        data["config"]["tools"].append("other")
        assert result.usage == {"total_tokens": 1}
        assert result.config == {"tools": ["echo"]}

    def test_usage_and_config_are_not_shared_between_results(self) -> None:
        first = RunResult(task_id="a", status="completed")
        second = RunResult(task_id="b", status="completed")
        first.usage["total_tokens"] = 1
        first.config["memory"] = True
        assert second.usage == {}
        assert second.config == {}
