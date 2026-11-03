"""Auto mode's card (#465): a stopped step becomes the person's choice.

In Auto, a step the review stops that a person may allow asks the chat's
card (the policy that asks a person) instead of coming back to the model as a
block. A step of any other kind, and every step of a run with nobody to ask,
is blocked as before. The model and the tools are the fakes of
``test_auto_mode``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    Chosen,
)
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.agent.test_approvals import Recorder, _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_mode import (
    BLOCKED,
    POST,
    VENV,
    FakeModel,
    OpensSite,
    _agent,
    _approvals,
    _plans,
    verdict,
)

KEYS = "import os\nprint(open(os.path.expanduser('~/.ssh/id_rsa')).read())"
NOT_ASKED = verdict(verdict="ask", category="beyond_request", fits_request="no")


class Card:
    """The person at the chat's card: each request gets the next answer."""

    def __init__(self, *answers: ApprovalDecision) -> None:
        self.answers = list(answers)
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        return self.answers.pop(0)


def _auto(card: Card) -> AutoPolicy:
    return AutoPolicy(asking=RememberingPolicy(card))


@pytest.mark.asyncio
async def test_a_step_stopped_as_not_asked_for_gives_a_card_and_no_block(
    tmp_path: Path,
) -> None:
    llm = FakeModel(_plans([POST]), [NOT_ASKED])
    agent, fake = _agent(llm, tmp_path / "ws")
    card = Card(ApprovalDecision(True))
    recorder = Recorder()

    await agent.run_task("test this API", on_event=recorder, approval=_auto(card))

    assert fake.ran == [POST]
    [asked] = card.asked
    assert asked.review is not None
    assert asked.review.categories == ("beyond_request",)
    [approval] = _approvals(recorder)
    assert (approval["origin"], approval["allowed"]) == ("user", True)
    assert recorder.of("run.finished")[-1].data["reviews"]["blocked"] == []
    assert not any(BLOCKED in text for text in llm.tool_messages())


@pytest.mark.asyncio
async def test_a_high_impact_step_words_can_allow_asks_too(tmp_path: Path) -> None:
    llm = FakeModel(_plans([VENV], answer="charted"))
    agent, fake = _agent(llm, tmp_path / "ws")
    card = Card(ApprovalDecision(True))

    result = await agent.run_task("chart cashflow.csv", approval=_auto(card))

    # No model review: the rules found it, and the person is asked anyway.
    assert (result.result, fake.ran, llm.review_requests) == ("charted", [VENV], [])
    [asked] = card.asked
    assert asked.review is not None and asked.review.verdict == "high"


@pytest.mark.asyncio
async def test_a_step_of_a_kind_only_a_click_allows_stays_blocked(
    tmp_path: Path,
) -> None:
    llm = FakeModel(_plans([KEYS], answer="explained"))
    agent, fake = _agent(llm, tmp_path / "ws")
    card = Card()
    recorder = Recorder()

    await agent.run_task("read my notes", on_event=recorder, approval=_auto(card))

    assert (fake.ran, card.asked) == ([], [])
    [approval] = _approvals(recorder)
    assert (approval["origin"], approval["allowed"]) == ("auto", False)
    assert llm.tool_messages()[0].startswith(f"Error: {BLOCKED}")


@pytest.mark.asyncio
async def test_a_run_with_nobody_to_ask_still_blocks(tmp_path: Path) -> None:
    llm = FakeModel(_plans([POST], answer="explained"), [NOT_ASKED])
    agent, fake = _agent(llm, tmp_path / "ws")
    recorder = Recorder()

    await agent.run_task("test this API", on_event=recorder, approval=AutoPolicy())

    assert fake.ran == []
    [approval] = _approvals(recorder)
    assert (approval["origin"], approval["allowed"]) == ("auto", False)
    assert llm.tool_messages()[0].startswith(f"Error: {BLOCKED}")


@pytest.mark.asyncio
async def test_a_site_stopped_as_not_asked_for_asks_on_the_card(
    tmp_path: Path,
) -> None:
    opener = OpensSite()
    plans = [_calls(("open_site", {"site": "news.example"}, "call_1")), _answer("ok")]
    llm = FakeModel(plans, [NOT_ASKED])
    (tmp_path / "ws").mkdir()
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            tool_exposure="all",
            workspace_root=str(tmp_path / "ws"),
        ),
        tool_providers=[opener],
    )
    agent.chat.sleep = _no_wait
    card = Card(ApprovalDecision(True))

    await agent.run_task("what is the weather?", approval=_auto(card))

    [answer] = opener.answers
    assert isinstance(answer, Chosen) and answer.allow
    [asked] = card.asked
    assert asked.tool_name == "browser:https://news.example"


# ── No typed permissions: Don't allow is the person's choice, for the turn ──

PIP = "import subprocess\nsubprocess.run(['pip', 'install', 'requests'])"
PIP3 = "import subprocess\nsubprocess.run(['pip3', 'install', '-q', 'requests'])"


@pytest.mark.asyncio
async def test_dont_allow_hands_the_model_the_choice_and_the_note(
    tmp_path: Path,
) -> None:
    llm = FakeModel(_plans([POST, POST], answer="explained"), [NOT_ASKED])
    agent, fake = _agent(llm, tmp_path / "ws")
    card = Card(ApprovalDecision(False, "not this API"))
    recorder = Recorder()

    result = await agent.run_task(
        "test this API", on_event=recorder, approval=_auto(card)
    )

    assert (result.result, fake.ran, len(card.asked)) == ("explained", [], 1)
    first, second = llm.tool_messages()
    assert "The user chose Don't allow for this step" in first
    assert "Their note: not this API." in first
    assert "melo will not ask them again" in first
    # The same step again in the turn: refused unasked, and not reviewed again.
    assert "earlier in this turn" in second and "not asked again" in second
    assert len(llm.review_requests) == 1
    one, two = _approvals(recorder)
    assert (one["origin"], one["allowed"], one["reason"]) == (
        "user",
        False,
        "not this API",
    )
    assert (two["origin"], two["allowed"], two["declined"]) == ("user", False, True)
    assert "call_key" not in two  # a refused repeat is no new denial
    for text in (first, second):
        assert BLOCKED not in text and "go ahead" not in text.lower()


@pytest.mark.asyncio
async def test_a_step_alike_is_not_asked_again_in_the_turn_but_is_next_turn(
    tmp_path: Path,
) -> None:
    llm = FakeModel(
        [*_plans([PIP, PIP3], answer="explained"), *_plans([PIP], answer="installed")]
    )
    agent, fake = _agent(llm, tmp_path / "ws")
    card = Card(ApprovalDecision(False), ApprovalDecision(True))
    policy = _auto(card)

    await agent.run_task("chart sales.csv", approval=policy)

    assert (fake.ran, len(card.asked)) == ([], 1)
    assert "earlier in this turn" in llm.tool_messages()[1]

    # A new message is a new turn: the person is asked again.
    result = await agent.run_task("try again", approval=policy)

    assert (result.result, fake.ran, len(card.asked)) == ("installed", [PIP], 2)
    assert llm.review_requests == []


@pytest.mark.asyncio
async def test_a_site_declined_on_the_card_is_not_asked_about_again_in_the_turn(
    tmp_path: Path,
) -> None:
    opener = OpensSite()
    plans = [
        _calls(("open_site", {"site": "news.example"}, "call_1")),
        _calls(("open_site", {"site": "news.example"}, "call_2")),
        _answer("ok"),
    ]
    llm = FakeModel(plans, [NOT_ASKED])
    (tmp_path / "ws").mkdir()
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            tool_exposure="all",
            workspace_root=str(tmp_path / "ws"),
        ),
        tool_providers=[opener],
    )
    agent.chat.sleep = _no_wait
    card = Card(ApprovalDecision(False, "not that site"))

    await agent.run_task("what is the weather?", approval=_auto(card))

    first, second = opener.answers
    assert isinstance(first, Chosen) and not first.again
    assert isinstance(second, Chosen) and second.again
    assert second.reason == "not that site"
    assert len(card.asked) == 1 and len(llm.review_requests) == 1
