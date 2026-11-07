"""A model's reason on an approval card is shown only in the person's own script (#576).

The check is one condition where the model's words are read (``step_gist``): the
words and the person's words either both have Chinese, Japanese or Korean
characters or neither does. A reason that does not match is not shown as what
melo says; the card keeps its own sentence of what the code does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.checker.review.gist import request_gist, step_gist
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.support.gateway.workers import WORKER_GUIDE
from localmelo.support.sessions.identity import ACTING_RULE, LANGUAGE_RULE
from localmelo.tests.agent.test_approvals import _answer, _calls
from localmelo.tests.agent.test_auto_mode import FakeModel, _agent

ENGLISH = "Check current time to confirm whether the check-in is from today"
CHINESE = "查看当前时间，确认今天的签到是不是刚刚完成的"
CODE = "import time\nprint(time.time())"


@pytest.mark.parametrize(
    "why, cjk, by",
    [
        # The person writes Chinese: an English reason is not shown as melo's words.
        (ENGLISH, True, "code"),
        (CHINESE, True, "model"),
        # The person writes English: a Chinese reason is not either.
        (CHINESE, False, "code"),
        (ENGLISH, False, "model"),
        # Nothing is known of the person's words: the reason stands as it did.
        (ENGLISH, None, "model"),
        (CHINESE, None, "model"),
    ],
)
def test_a_reason_in_another_script_than_the_person_s_is_not_shown(
    why: str, cjk: bool | None, by: str
) -> None:
    gist = step_gist("python_exec", {"code": CODE, "why": why}, cjk=cjk)
    assert gist is not None and gist.by == by
    if by == "code":
        # The card says what the code does, in its own words, with no note of the reason.
        assert gist.id == "script" and not gist.note and not gist.warn


def test_a_comment_in_another_script_is_not_shown_either() -> None:
    """The first comment of a script reads as the model's words too; code comments are mostly English."""
    code = f"# {ENGLISH}\n{CODE}"
    assert step_gist("python_exec", {"code": code}, cjk=True).by == "code"  # type: ignore[union-attr]
    assert step_gist("python_exec", {"code": code}, cjk=False).by == "model"  # type: ignore[union-attr]


def test_a_task_is_told_the_language_to_write_in_like_every_chat() -> None:
    """A task's worker had no language rule: the model wrote its reasons in English (#576)."""
    assert LANGUAGE_RULE in WORKER_GUIDE
    assert WORKER_GUIDE.endswith(" " + ACTING_RULE)


class Asked:
    """A policy that records each request and allows it."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision(True, origin="user")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tasks, cjk, by",
    [
        (("看一下现在几点",), True, "code"),
        (("What time is it now?",), False, "model"),
        # A short Latin reply in a Chinese chat changes nothing: the recent words count.
        (("看一下现在几点", "ok"), True, "code"),
        (("What time is it now?", "好的"), True, "code"),
    ],
)
async def test_the_gate_tells_each_request_the_script_the_person_writes_in(
    tmp_path: Path, tasks: tuple[str, ...], cjk: bool, by: str
) -> None:
    call = _calls(("python_exec", {"code": CODE, "why": ENGLISH}, "call_1"))
    plans = [call, _answer("done")] * len(tasks)
    agent, _ = _agent(FakeModel(plans), tmp_path / "ws")

    async def python_exec(code: str, why: str = "") -> str:
        return "ok"

    agent.executor.register("python_exec", python_exec, requires_approval=True)
    policy = Asked()
    for task in tasks:
        await agent.run_task(task, approval=policy)
    request = policy.requests[-1]
    assert request.person_cjk is cjk
    gist = request_gist(request)
    assert gist is not None and gist.by == by
