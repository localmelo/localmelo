"""``app_control`` is never looser than ``system`` was, in any approval mode.

The same step is run once with each kind: a script that presses keys through
System Events (``system``) and one that quits Safari (``app_control``). Under
auto mode, a session grant, the user's own words, an Allow once and the
model's review, both must end the same way: only a click lets either run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import _step_review
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
)
from localmelo.melo.contracts.review import (
    MODEL_CAN_ALLOW,
    WORDS_CAN_ALLOW,
    ActionReview,
    ModelAnswer,
    step_key,
)
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.agent.test_approvals import Recorder
from localmelo.tests.agent.test_auto_mode import (
    FakeModel,
    _agent,
    _approvals,
    _plans,
    verdict,
)

SYSTEM = (
    "import subprocess\nsubprocess.run(['osascript', '-e', "
    '\'tell application "System Events" to keystroke "a"\'])'
)
APP = (
    "import subprocess\nsubprocess.run(['osascript', '-e', "
    "'tell application \"Safari\" to quit'])"
)
KINDS = [("system", SYSTEM), ("app_control", APP)]


def ids(kinds: list[tuple[str, str]]) -> list[str]:
    return [name for name, _ in kinds]


@pytest.mark.asyncio
@pytest.mark.parametrize(("category", "code"), KINDS, ids=ids(KINDS))
async def test_auto_blocks_it_and_the_users_words_do_not_allow_it(
    tmp_path: Path, category: str, code: str
) -> None:
    """Even a message that names the step and asks for it does not run it."""
    llm = FakeModel(
        _plans([code], answer="explained"),
        # Were the model asked, it would say the user asked for this.
        [verdict(verdict="ask", category="none", user_asked="yes")],
    )
    agent, fake = _agent(llm, tmp_path / "ws")
    recorder = Recorder()

    result = await agent.run_task(
        "quit Safari for me, go ahead", on_event=recorder, approval=AutoPolicy()
    )

    assert (result.result, fake.ran) == ("explained", [])
    assert llm.review_requests == []  # no consent review for a click-only kind
    [approval] = _approvals(recorder)
    assert (approval["origin"], approval["allowed"], approval["mode"]) == (
        "auto",
        False,
        "auto",
    )
    assert approval["review"]["verdict"] == "high"
    assert approval["review"]["categories"] == [category]
    assert "allow this step once from the chat" in llm.tool_messages()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(("category", "code"), KINDS, ids=ids(KINDS))
async def test_a_block_is_allowed_once_by_a_click_and_only_once(
    tmp_path: Path, category: str, code: str
) -> None:
    llm = FakeModel(_plans([code, code], answer="done"))
    agent, fake = _agent(llm, tmp_path / "ws")
    policy = AutoPolicy()
    key = step_key("python_exec", {"code": code}, str(tmp_path / "ws"))
    policy.allow_once(key, [category], "python_exec")
    recorder = Recorder()

    await agent.run_task("do it", on_event=recorder, approval=policy)

    assert fake.ran == [code]  # the first ran on the click, the second did not
    first, second = _approvals(recorder)
    assert (first["origin"], first["allowed"], first["allowance"]) == (
        "user",
        True,
        "once",
    )
    assert (second["origin"], second["allowed"]) == ("auto", False)


class ChatApprover:
    """A person who answers every question with "Allow for this chat"."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision(True, remember=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(("category", "code"), KINDS, ids=ids(KINDS))
async def test_a_session_grant_never_covers_it(
    tmp_path: Path, category: str, code: str
) -> None:
    """A grant made on a routine call still asks every time for this step."""
    llm = FakeModel(_plans(["print(1)", code, code], answer="done"))
    agent, fake = _agent(llm, tmp_path / "ws")
    approver = ChatApprover()
    policy = RememberingPolicy(approver)

    await agent.run_task("do it", approval=policy)

    assert [r.review.verdict if r.review else "" for r in approver.asked] == [
        "routine",
        "high",
        "high",
    ]
    assert [r.review.categories for r in approver.asked[1:] if r.review] == [
        (category,),
        (category,),
    ]
    assert llm.review_requests == []  # the user's model is not asked about it
    assert fake.ran == ["print(1)", code, code]  # each click allowed one run


# ── The model's review and the user's words, one level down ──


def reviewer() -> _step_review.StepReviewer:
    return _step_review.StepReviewer(lambda: None)  # type: ignore[arg-type, return-value]


@pytest.mark.parametrize("mode", ["ask", "silent", "auto"])
def test_no_mode_asks_the_model_for_consent_to_either_kind(mode: Any) -> None:
    system = ActionReview("high", ("system",), ("Does it.",))
    app = ActionReview("high", ("app_control",), ("Does it.",))
    pending = reviewer()
    assert pending._purpose(app, mode, {"safari"}) == pending._purpose(
        system, mode, {"safari"}
    )
    assert pending._purpose(app, mode, {"safari"}) is None


@pytest.mark.parametrize("category", ["system", "app_control"])
@pytest.mark.parametrize("purpose", ["undecided", "consent"])
def test_the_models_answer_keeps_either_kind_blocked(
    category: str, purpose: Any
) -> None:
    review = ActionReview("high", (category,), ("Does it.",))
    answer = ModelAnswer(
        verdict="routine",
        category=category,
        fits_request="yes",
        user_asked="yes",
        reason="The user asked.",
    )
    settled = _step_review._settle(review, answer, purpose)
    assert settled.verdict == "high"
    assert settled.categories[0] == category


def test_the_two_kinds_are_alike_to_every_gate() -> None:
    for gate in (MODEL_CAN_ALLOW, WORDS_CAN_ALLOW):
        assert ("system" in gate) == ("app_control" in gate)


# ── The block's record carries the sentence the card starts with ──


@pytest.mark.asyncio
async def test_a_block_keeps_the_sentence_and_the_kind_for_the_card(
    tmp_path: Path,
) -> None:
    code = APP + "\n"
    llm = FakeModel(
        _plans(["# Close the extra Safari tabs\n" + code], answer="explained")
    )
    agent, fake = _agent(llm, tmp_path / "ws")
    recorder = Recorder()

    await agent.run_task("tidy up", on_event=recorder, approval=AutoPolicy())

    [approval] = _approvals(recorder)
    assert fake.ran == []
    review = approval["review"]
    assert review["categories"] == ["app_control"]
    assert review["gist"] == {
        "say": {"id": "model", "params": {"text": "Close the extra Safari tabs"}},
        "by": "model",
        "apps": "Safari",
    }


@pytest.mark.asyncio
async def test_a_block_without_words_gets_a_sentence_built_by_code(
    tmp_path: Path,
) -> None:
    llm = FakeModel(_plans([APP], answer="explained"))
    agent, _ = _agent(llm, tmp_path / "ws")
    recorder = Recorder()

    await agent.run_task("tidy up", on_event=recorder, approval=AutoPolicy())

    [approval] = _approvals(recorder)
    assert approval["review"]["gist"] == {
        "say": {"id": "script_apps", "params": {"apps": "Safari"}},
        "by": "code",
        "apps": "Safari",
    }
