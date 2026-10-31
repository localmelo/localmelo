"""AgentOptions.lean_output: fewer output tokens in direct runs; off, main's requests.

The run is the scripted one of the legacy goldens (a Python call, a reflection that
continues, an answer) through the OpenAI-compatible provider, with the cache defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions, _chat
from localmelo.melo.agent._chat import (
    LEAN_REFLECTION_PROMPT,
    LEAN_SYSTEM_PROMPT,
    REFLECTION_PROMPT,
    SYSTEM_PROMPT,
)
from localmelo.tests.agent.test_legacy_golden import SCENARIOS, Reply, record_run
from localmelo.tests.agent.test_reflection_gate import FakeLLM

DEFAULTS = {
    "fixed_prefix": True,
    "window_eviction": "chunk",
    "reflect_tool_guard": True,
    "cache_layout": "v3",
    "window_messages": 64,
}
# The sha256 of the run's three request bodies (plan, reflection, plan) as main sends
# them, and with the option. Recorded at 139992b5; re-recorded when the fixed system
# section lost its ``Prefix: fixed-prefix-v2`` line: with that line put back,
# the digests of 139992b5 come out again. Re-recorded again when file_read gained
# start_line and end_line: with its earlier description and parameters put back
# in the tool list, the digests before this change come out.
OFF = (
    "e2494bc9076754585383caeb7c38e9a44c44e58ca594307405c5d3daba25c660",
    "87e6a20e821aed81a210b0cd191f85dbb8676b2b0ff00d70841836bd7a9b51a8",
    "7df9238803ac5730f08f70d58e65a59cf636d827958798ad3ffb0e6c44b05255",
)
ON = (
    "d467130bd3b93739da67550f021f41d7fb0dc99e90d7fc52597d883b8ebe8050",
    "1405dc19ec94a164689bde35b8a8b6a66bba830849e5d417b555fd9a95d4189a",
    "229e4791ac17cefec0fd1eb4948504f66e64defb8bba2f85c9d05ed7f256fa73",
)
GATE = {
    "recommended_action",
    "task_still_feasible",
    "next_step_is_concrete",
    "next_step_is_novel",
}
DROPPED = {
    "progress_made",
    "new_information_gained",
    "estimated_info_gain",
    "estimated_cost",
    "repeat_risk",
    "novelty",
    "feasibility",
}


def _keys(prompt: str) -> set[str]:
    return set(re.findall(r'^  "(\w+)":', prompt, re.M))


async def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short: bool = False, **options: Any
) -> tuple[Any, list[dict[str, Any]], tuple[str, ...]]:
    """The result, the request bodies and their sha256 of one run.

    *short*: the reflection reply holds only the keys of ``LEAN_REFLECTION_PROMPT``.
    """
    provider, build = SCENARIOS["openai_reflection_continues"]
    scenario = build()
    if short:
        long = json.loads(scenario.replies[1].content)
        keep = {k: v for k, v in long.items() if k in _keys(LEAN_REFLECTION_PROMPT)}
        scenario.replies[1] = Reply(content=json.dumps(keep))
    scenario.options.update(DEFAULTS | options)
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)", raising=False
    )
    raw: list[str] = []
    run_dir = Path(tempfile.mkdtemp(dir=tmp_path))
    result, _ = await record_run(provider, scenario, run_dir, monkeypatch, raw)
    digests = tuple(hashlib.sha256(body.encode()).hexdigest() for body in raw)
    return result, [json.loads(body) for body in raw], digests


async def test_without_the_option_every_request_is_the_one_main_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert AgentOptions().lean_output is False
    assert (await _run(tmp_path, monkeypatch))[2] == OFF
    assert (await _run(tmp_path, monkeypatch, lean_output=False))[2] == OFF


async def test_with_the_option_the_system_and_reflection_prompts_are_the_lean_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bodies, digests = await _run(tmp_path, monkeypatch, lean_output=True)
    assert digests == ON
    system = bodies[0]["messages"][0]["content"]
    assert system.startswith(LEAN_SYSTEM_PROMPT) and "step by step" not in system
    asking = bodies[1]["messages"][-1]["content"]
    assert asking.startswith(LEAN_REFLECTION_PROMPT)
    assert not DROPPED & _keys(asking)


async def test_the_fixed_prefix_is_the_same_on_every_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bodies, _ = await _run(tmp_path, monkeypatch, lean_output=True)
    for later in bodies[1:]:  # the reflection request and the next planning request
        assert later["messages"][: len(bodies[0]["messages"])] == bodies[0]["messages"]
        assert later["tools"] == bodies[0]["tools"]


async def test_the_earlier_layout_also_sends_the_lean_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options: dict[str, Any] = {"lean_output": True, "fixed_prefix": False}
    _, bodies, _ = await _run(tmp_path, monkeypatch, **options)
    assert bodies[0]["messages"][0]["content"].startswith(LEAN_SYSTEM_PROMPT)
    assert bodies[1]["messages"][0]["content"].startswith(LEAN_REFLECTION_PROMPT)


async def test_the_utility_gate_keeps_the_full_reflection_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options: dict[str, Any] = {"lean_output": True, "reflection_gate": "utility"}
    _, bodies, _ = await _run(tmp_path, monkeypatch, **options)
    assert bodies[1]["messages"][-1]["content"].startswith(REFLECTION_PROMPT)
    assert bodies[0]["messages"][0]["content"].startswith(LEAN_SYSTEM_PROMPT)


async def test_the_compress_modes_keep_the_system_prompt_for_their_notes() -> None:
    options = AgentOptions(lean_output=True, reasoning="compress")
    agent = Agent(llm=FakeLLM([]), options=options)
    try:
        assert agent.chat.system_prompt == SYSTEM_PROMPT
        assert agent.chat.reflection_prompt == LEAN_REFLECTION_PROMPT
    finally:
        await agent.close()


def test_the_lean_prompts_keep_what_the_loop_reads() -> None:
    assert LEAN_SYSTEM_PROMPT.startswith(SYSTEM_PROMPT.rpartition("\n")[0] + "\n")
    for hint in ("no text between tool calls", "three lines", "relative paths"):
        assert hint in LEAN_SYSTEM_PROMPT
    assert "sys.path" in LEAN_SYSTEM_PROMPT and "read them first" in LEAN_SYSTEM_PROMPT
    full, lean = _keys(REFLECTION_PROMPT), _keys(LEAN_REFLECTION_PROMPT)
    assert GATE <= lean <= full and full - lean == DROPPED
    both = os.path.commonprefix([REFLECTION_PROMPT, LEAN_REFLECTION_PROMPT])
    assert both.startswith(_chat._REFLECTION_INTRO)


@pytest.mark.parametrize("short", [True, False])
async def test_a_short_and_a_long_reflection_reply_both_continue_the_run(
    short: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, bodies, _ = await _run(tmp_path, monkeypatch, short, lean_output=True)
    # Read at once: no retry request, and the next attempt ran.
    assert (result.status, result.attempts, len(bodies)) == ("completed", 2, 3)


def test_the_option_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="lean_output"):
        AgentOptions(lean_output="on")  # type: ignore[arg-type]
