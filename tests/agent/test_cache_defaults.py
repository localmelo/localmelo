"""The default request: a scripted run's requests are frozen in a golden.

The golden is one run with the defaults (a Python call, a reflection that continues,
an answer) through the OpenAI-compatible provider. A change to the requests a default
run sends shows up as a difference here, so it is a decision, not an accident.
``LOCALMELO_RECORD_GOLDENS=1`` records the golden again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions, _chat
from localmelo.melo.agent.agent import NO_CHANGE_PUSH
from localmelo.support.config import AgentConfig
from localmelo.tests.agent.test_brief_protocol import _digest
from localmelo.tests.agent.test_brief_protocol import _run as _brief_run
from localmelo.tests.agent.test_reflection_gate import _answer, _echo_call, _reflection
from localmelo.tests.agent.wire_runs import (
    check_golden,
    record_run,
    reflection_continues,
)

GOLDEN = Path(__file__).parent / "goldens" / "cache_defaults"
DEFAULTS: dict[str, Any] = {"window_eviction": "chunk", "window_messages": 64}
SENTENCE = "Answer with the JSON object only; do not call any tool."


def test_the_defaults_of_the_options_and_of_the_config() -> None:
    options = AgentOptions()
    assert {name: getattr(options, name) for name in DEFAULTS} == DEFAULTS
    config = AgentConfig()
    # The config leaves the window unset: the backend decides it (see
    # tests/cli/test_window_messages_config.py).
    assert {name: getattr(config, name) for name in DEFAULTS} == {
        **DEFAULTS,
        "window_messages": None,
    }


async def _run_default_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)", raising=False
    )
    _, document = await record_run(
        "openai", reflection_continues(), tmp_path, monkeypatch
    )
    return document


async def test_the_default_run_matches_its_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _run_default_scenario(tmp_path, monkeypatch)
    check_golden(GOLDEN / "openai_reflection_continues.json", document)
    requests: Any = document["requests"]
    assert len(requests) == 3
    reflect, after = requests[1], requests[2]
    assert after["messages"][: len(reflect["messages"])] == reflect["messages"]
    result: Any = document["result"]
    assert result["config"]["window_messages"] == 64


async def test_the_default_reflection_request_offers_the_tools_and_asks_for_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _run_default_scenario(tmp_path, monkeypatch)
    plan, reflect = document["requests"][0], document["requests"][1]
    assert reflect["tools"] == plan["tools"]
    assert reflect["messages"][-1]["content"].endswith(SENTENCE)
    system = plan["messages"][0]["content"]
    # The date line closes the fixed section; the layout's name is metadata
    # (``prefix_cache.version``), not a line the model reads.
    assert system.endswith("\nDate: 2026-10-28 (UTC-04:00)")
    assert "Prefix:" not in system and "fixed-prefix" not in system


async def test_a_brief_run_under_the_defaults_still_gets_the_no_change_line_once() -> (
    None
):
    # The fixed prefix merges the line with the reflection into one user message.
    result, llm, _ = await _brief_run(
        [_echo_call("a"), _reflection("continue"), _answer("42")],
        max_attempts=3,
        brief_probe=_digest("same"),
    )
    assert result.status == "completed"
    carrying = [
        index
        for index, call in enumerate(llm.calls)
        if any(
            m.role == "user" and NO_CHANGE_PUSH in m.content for m in call["messages"]
        )
    ]
    assert carrying == [2]  # plan, reflect, then the next attempt's first plan
    assert "[REFLECTION]" in llm.calls[2]["messages"][-1].content
