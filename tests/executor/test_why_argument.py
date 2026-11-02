"""shell_exec and python_exec take an optional ``why`` for the approval card (#454).

It is one short sentence of the model's own words. The tools accept it, whichever
way they are built, and ignore it: it changes nothing about what runs, how the
step is keyed or how the rules review it. The schema text is pinned here, as the
wire goldens pin the requests that carry it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import step_key
from localmelo.melo.executor import (
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    register_builtins,
)
from localmelo.melo.executor.builtins import BUILTINS, WHY_NOTE, WHY_TEXT
from localmelo.melo.memory.coordinator import Hippo

SENTENCE = (
    "One short sentence, in the person's language, saying what this step does "
    "and why (shown on approval cards)"
)
CALLS = [
    ("shell_exec", "command", "echo hello"),
    ("python_exec", "code", "print('hello')"),
]


def _tool(name: str) -> Any:
    return next(tool for tool, _ in BUILTINS if tool.name == name)


@pytest.mark.parametrize(("name", "required", "_code"), CALLS)
def test_the_schema_offers_an_optional_why(
    name: str, required: str, _code: str
) -> None:
    tool = _tool(name)
    properties = tool.parameters["properties"]
    assert properties["why"] == {"type": "string", "description": SENTENCE}
    assert WHY_TEXT == SENTENCE
    assert tool.parameters["required"] == [required]
    assert tool.description.endswith(". " + WHY_NOTE)
    assert WHY_NOTE == "Give why for a step that changes anything."
    assert list(properties) == [required, "why"]


def _executor(root: Path | None) -> Executor:
    hippo = Hippo()
    executor = Executor(
        hippo,
        Checker(mode="interactive"),
        workspace_root=None if root is None else str(root),
    )
    register_builtins(
        executor,
        hippo,
        names={"shell_exec", "python_exec"},
        mode="interactive",
        workspace_root=None if root is None else str(root),
    )
    return executor


@pytest.mark.asyncio
@pytest.mark.parametrize("rooted", [False, True], ids=["unrooted", "rooted"])
@pytest.mark.parametrize(("name", "key", "code"), CALLS)
async def test_the_tools_accept_a_why_and_run_the_same(
    tmp_path: Path, rooted: bool, name: str, key: str, code: str
) -> None:
    executor = _executor(tmp_path if rooted else None)
    plain = await executor.execute_structured(
        ExecutionRequest(tool_name=name, arguments={key: code})
    )
    said = await executor.execute_structured(
        ExecutionRequest(
            tool_name=name, arguments={key: code, "why": "Say hello to the log"}
        )
    )
    assert plain.status == said.status == ExecutionStatus.SUCCESS
    assert plain.output == said.output
    assert "hello" in said.output


@pytest.mark.asyncio
async def test_a_why_that_is_not_text_is_refused_like_any_argument() -> None:
    outcome = await _executor(None).execute_structured(
        ExecutionRequest(
            tool_name="shell_exec", arguments={"command": "echo hi", "why": 7}
        )
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert "'why' must be of type string" in outcome.error


@pytest.mark.parametrize(("name", "key", "code"), CALLS)
def test_a_why_never_changes_the_step(name: str, key: str, code: str) -> None:
    """The same step is the same step: the key, the review and the verdict ignore it."""
    plain, said = {key: code}, {key: code, "why": "Something else entirely"}
    assert step_key(name, plain, "/work") == step_key(name, said, "/work")
    checker = Checker()

    def review(arguments: dict[str, Any]) -> Any:
        payload = ActionReviewPayload(name, arguments, "/work")
        return checker.review_action(payload)

    assert review(plain) == review(said)


def test_a_why_cannot_soften_a_risky_step() -> None:
    risky = {"command": "sudo rm -rf /opt/app", "why": "Print the date"}
    review = Checker().review_action(ActionReviewPayload("shell_exec", risky, "/work"))
    assert review.verdict == "high"
