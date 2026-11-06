"""A run that is still making progress when its steps run low is given more by itself (#547).

The decision is made a few steps before the end, so the steps left still finish
the work and report. A run that makes no progress, one that already used its
extensions, and one with the option off end at ``max_steps`` as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import MAX_EXTENSIONS, Message, RunResult
from localmelo.support import config
from localmelo.support.sessions.agents import configured_attempt_options
from localmelo.tests.agent.test_approvals import _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_mode import FakeModel

STEPS, MORE = 8, 5


class Tool:
    """``python_exec``: ok each time, or an error that differs each time."""

    def __init__(self, fails: bool) -> None:
        self.fails, self.n = fails, 0

    async def python_exec(self, code: str) -> str:
        self.n += 1
        if self.fails:
            raise RuntimeError(f"error number {self.n}")
        return f"ok {self.n}"


def _plans(calls: int) -> list[Message]:
    work = [
        _calls(("python_exec", {"code": f"print({n})"}, f"call_{n}"))
        for n in range(calls)
    ]
    return [*work, _answer("all done")]


async def _run(
    tmp_path: Path, calls: int, *, extend: int, fails: bool = False
) -> tuple[RunResult, Tool]:
    (tmp_path / "ws").mkdir()
    agent = Agent(
        llm=FakeModel(_plans(calls)),
        options=AgentOptions(
            builtins=frozenset({"python_exec"}),
            tool_exposure="all",
            workspace_root=str(tmp_path / "ws"),
            attempt_policy="none",
            max_steps=STEPS,
            extend_steps=extend,
        ),
    )
    tool = Tool(fails)
    agent.executor.register("python_exec", tool.python_exec, requires_approval=False)
    agent.chat.sleep = _no_wait
    return await agent.run_task("do the work"), tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("calls", "extend", "fails", "status", "steps"),
    [
        # Progress: the 12 steps of work and the answer need the extension.
        (12, MORE, False, "completed", 13),
        # Off: it ends at max_steps, as before.
        (12, 0, False, "budget_exhausted", STEPS),
        # No progress (every step errs): no extension.
        (12, MORE, True, "budget_exhausted", STEPS),
        # Progress, but past the extensions it was given: it ends.
        (40, MORE, False, "budget_exhausted", STEPS + MORE * MAX_EXTENSIONS),
    ],
    ids=["progress", "off", "no progress", "extensions used up"],
)
async def test_a_run_making_progress_is_given_more_steps_and_nothing_else_is(
    tmp_path: Path, calls: int, extend: int, fails: bool, status: str, steps: int
) -> None:
    result, _ = await _run(tmp_path, calls, extend=extend, fails=fails)
    assert (result.status, result.steps) == (status, steps)


def test_a_persons_session_has_it_on_and_a_caller_keeps_its_own() -> None:
    cfg = config.Config()
    assert configured_attempt_options(cfg, AgentOptions()).extend_steps > 0
    assert (
        configured_attempt_options(cfg, AgentOptions(extend_steps=7)).extend_steps == 7
    )
    kept = configured_attempt_options(cfg, AgentOptions(), {"extend_steps"})
    assert kept.extend_steps == 0 and AgentOptions().extend_steps == 0
    # A step budget the person typed (--max-steps) is exact.
    typed = configured_attempt_options(cfg, AgentOptions(max_steps=3), {"max_steps"})
    assert typed.extend_steps == 0
    with pytest.raises(ValueError, match="extend_steps"):
        AgentOptions(extend_steps=-1)


A, B, C, D = ("run", "a", ""), ("run", "b", ""), ("run", "c", ""), ("run", "d", "")


@pytest.mark.parametrize(
    ("steps", "progressing"),
    [
        ([A, B, C, D, A], True),
        ([A, A, A, B, C], True),
        # Two steps that take turns are a loop, and so is one that repeats.
        ([A, B, A, B, A], False),
        ([A, A, A, A, A], False),
        # Errors are no progress.
        (
            [A, ("run", "x", "boom"), ("run", "y", "boom"), ("run", "z", "boom"), B],
            False,
        ),
    ],
    ids=["new steps", "three kinds", "alternating", "repeating", "errors"],
)
def test_two_steps_that_take_turns_are_no_progress(
    steps: list[tuple[str, str, str]], progressing: bool
) -> None:
    assert Agent._progressing(steps) is progressing
