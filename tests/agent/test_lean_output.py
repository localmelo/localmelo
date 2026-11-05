"""AgentOptions.lean_output: fewer output tokens in direct runs; off, main's requests.

The run is the scripted one of the default golden (a Python call, a reflection that
continues, an answer) through the OpenAI-compatible provider, with the defaults.
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
from localmelo.tests.agent.test_reflection_gate import FakeLLM
from localmelo.tests.agent.wire_runs import Reply, record_run, reflection_continues

# The sha256 of the run's three request bodies (plan, reflection, plan) as main sends
# them, and with the option. Recorded at 139992b5; re-recorded when the fixed system
# section lost its ``Prefix: fixed-prefix-v2`` line: with that line put back,
# the digests of 139992b5 come out again. Re-recorded again when file_read gained
# start_line and end_line: with its earlier description and parameters put back
# in the tool list, the digests before this change come out. Re-recorded again when
# shell_exec and python_exec gained the optional ``why`` (and a last sentence in their
# descriptions): with the earlier tool definitions put back, the digests before this
# change come out.
OFF = (
    "5ccaf2488d634fd83628cf6db020d3b930be27752f8d97733af26ea4b154939f",
    "393e376869c7f673b8d96d5ce9382beaed6cedddd99bfb46da6f9d55d2cd8a25",
    "d14a13703fd4482f2e7d47a156fbee047a396ae5ee5df6c2151be27543375d86",
)
ON = (
    "8b86af148262bcd8cb3b62407bfeb466e28fe2bd107ba32e8d801369cd6ba3b1",
    "4701abf403c54784409028e113a4b850a082257d7f47356d1bf1979f68e0c407",
    "27b6c90dbc5da39370677ace0ff21176bb723bb093d3bed2912861ecf5994bf2",
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
    scenario = reflection_continues()
    if short:
        long = json.loads(scenario.replies[1].content)
        keep = {k: v for k, v in long.items() if k in _keys(LEAN_REFLECTION_PROMPT)}
        scenario.replies[1] = Reply(content=json.dumps(keep))
    scenario.options.update(options)
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)", raising=False
    )
    raw: list[str] = []
    run_dir = Path(tempfile.mkdtemp(dir=tmp_path))
    result, _ = await record_run("openai", scenario, run_dir, monkeypatch, raw)
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
