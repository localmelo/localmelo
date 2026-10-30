"""The cache settings' defaults: the new default request, and the opt-outs that keep the old ones.

``fixed_prefix``, ``window_eviction``, ``reflect_tool_guard``, ``cache_layout`` and
``window_messages`` default to ``True``, ``"chunk"``, ``True``, ``"v3"`` and 64 (a
local server's window). The legacy goldens run with the opt-out values; the goldens
here are the same scripted run with the defaults, and with only the layout and the
window opted out.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions, _chat
from localmelo.melo.agent.agent import NO_CHANGE_PUSH
from localmelo.support.config import AgentConfig
from localmelo.tests.agent.test_brief_protocol import _digest
from localmelo.tests.agent.test_brief_protocol import _run as _brief_run
from localmelo.tests.agent.test_legacy_golden import (
    GOLDEN_DIR as LEGACY_GOLDEN_DIR,
)
from localmelo.tests.agent.test_legacy_golden import (
    SCENARIOS,
    record_run,
)
from localmelo.tests.agent.test_phase1a_compat import check_golden
from localmelo.tests.agent.test_reflection_gate import _answer, _echo_call, _reflection
from localmelo.tests.legacy_options import LEGACY_CACHE_OPTIONS

GOLDEN = Path(__file__).parent / "goldens" / "cache_defaults"
DEFAULTS: dict[str, Any] = {
    "fixed_prefix": True,
    "window_eviction": "chunk",
    "reflect_tool_guard": True,
    "cache_layout": "v3",
    "window_messages": 64,
}
SENTENCE = "Answer with the JSON object only; do not call any tool."


def test_the_defaults_of_the_options_and_of_the_config() -> None:
    options = AgentOptions()
    assert {name: getattr(options, name) for name in DEFAULTS} == DEFAULTS
    config = AgentConfig()
    # The config leaves the window unset: the backend decides it (see
    # tests/cli/test_cache_layout_config.py).
    assert {name: getattr(config, name) for name in DEFAULTS} == {
        **DEFAULTS,
        "window_messages": None,
    }


def test_the_opt_out_values_are_the_earlier_defaults() -> None:
    assert LEGACY_CACHE_OPTIONS == {
        "fixed_prefix": False,
        "window_eviction": "slide",
        "reflect_tool_guard": False,
        "cache_layout": "v2",
        "window_messages": 20,
    }
    options = AgentOptions(**LEGACY_CACHE_OPTIONS)
    assert dataclasses.replace(options, **DEFAULTS) == AgentOptions()


async def _run_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **options: object
) -> dict[str, object]:
    provider, build = SCENARIOS["openai_reflection_continues"]
    scenario = build()
    scenario.options.update(options)
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)", raising=False
    )
    _, document = await record_run(provider, scenario, tmp_path, monkeypatch)
    return document


async def test_the_default_run_matches_its_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _run_scenario(tmp_path, monkeypatch, **DEFAULTS)
    check_golden(GOLDEN / "openai_reflection_continues.json", document)
    requests: Any = document["requests"]
    assert len(requests) == 3
    reflect, after = requests[1], requests[2]
    assert after["messages"][: len(reflect["messages"])] == reflect["messages"]
    result: Any = document["result"]
    assert (result["config"]["cache_layout"], result["config"]["window_messages"]) == (
        "v3",
        64,
    )


async def test_the_layout_and_window_opt_outs_match_their_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt_outs = {"cache_layout": "v2", "window_messages": 20}
    document = await _run_scenario(tmp_path, monkeypatch, **{**DEFAULTS, **opt_outs})
    check_golden(GOLDEN / "openai_reflection_continues_v2.json", document)
    result: Any = document["result"]
    assert "cache_layout" not in result["config"]
    assert "window_messages" not in result["config"]


async def test_the_default_reflection_request_offers_the_tools_and_asks_for_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _run_scenario(tmp_path, monkeypatch, **DEFAULTS)
    plan, reflect = document["requests"][0], document["requests"][1]  # type: ignore[index]
    assert reflect["tools"] == plan["tools"]
    assert reflect["messages"][-1]["content"].endswith(SENTENCE)
    system = plan["messages"][0]["content"]
    # The date line closes the fixed section; the layout's name is metadata
    # (``prefix_cache.version``), not a line the model reads.
    assert system.endswith("\nDate: 2026-10-28 (UTC-04:00)")
    assert "Prefix:" not in system and "fixed-prefix" not in system


async def test_the_opt_out_values_send_the_requests_of_the_earlier_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opted_out = await _run_scenario(tmp_path, monkeypatch, **LEGACY_CACHE_OPTIONS)
    earlier = json.loads(
        (LEGACY_GOLDEN_DIR / "openai_reflection_continues.json").read_text()
    )
    assert opted_out["requests"] == earlier["requests"]
    assert opted_out["result"] == earlier["result"]
    assert json.dumps(opted_out["requests"]) != json.dumps(
        json.loads((GOLDEN / "openai_reflection_continues.json").read_text())[
            "requests"
        ]
    )


async def test_a_brief_run_under_the_defaults_still_gets_the_no_change_line_once() -> (
    None
):
    # The fixed prefix merges the line with the reflection into one user message.
    result, llm, _ = await _brief_run(
        [_echo_call("a"), _reflection("continue"), _answer("42")],
        max_attempts=3,
        brief_probe=_digest("same"),
        **DEFAULTS,
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
