"""The steps that reach macOS or destroy work are reviewed in every mode.

Each class used to read as ``routine``: Auto ran it without a question and Ask
showed it as plain code. Now:

- **Auto** blocks each one as its category says: ``system`` (a disk erased,
  other processes stopped, a link of a custom scheme, a script that reaches
  macOS without naming an app) can only be allowed by a click, never by the
  user's words or the model; ``delete`` (uncommitted work discarded) can also
  be allowed by the user's own words naming what it acts on.
- **Ask** asks every time, never offers a grant that covers it, and the card
  names what the step does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
)
from localmelo.melo.contracts.review import (
    WORDS_CAN_ALLOW,
    covered_by_grant,
    offers_grant,
    review_title,
)
from localmelo.support.gateway.console.shapes import approval_json
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.agent.test_approvals import Recorder
from localmelo.tests.agent.test_auto_mode import (
    FakeModel,
    _agent,
    _approvals,
    _plans,
    verdict,
)

RUN = "import subprocess\nsubprocess.run({argv!r})"
KILL = RUN.format(argv=["pkill", "-f", "dev-server"])
DISK = RUN.format(argv=["diskutil", "eraseDisk", "JHFS+", "Scratch", "disk2"])
LINK = RUN.format(argv=["open", "shortcuts://run-shortcut?name=Foo"])
BRIDGE = RUN.format(
    argv=[
        "osascript",
        "-e",
        'use framework "Foundation"',
        "-e",
        "set t to current application's NSTask",
    ]
)
DISCARD = RUN.format(argv=["git", "checkout", "--", "notes.txt"])

SYSTEM = [("kill", KILL), ("disk", DISK), ("link", LINK), ("bridge", BRIDGE)]
EVERY = [*SYSTEM, ("discard", DISCARD)]


def names(rows: list[tuple[str, str]]) -> list[str]:
    return [name for name, _ in rows]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "code"), EVERY, ids=names(EVERY))
async def test_auto_blocks_each_class_without_asking_the_model(
    tmp_path: Path, name: str, code: str
) -> None:
    llm = FakeModel(_plans([code], answer="explained"))
    agent, fake = _agent(llm, tmp_path / "ws")
    recorder = Recorder()

    result = await agent.run_task(
        "tidy up the project", on_event=recorder, approval=AutoPolicy()
    )

    assert (result.result, fake.ran) == ("explained", [])
    assert llm.review_requests == []
    [approval] = _approvals(recorder)
    assert (approval["origin"], approval["allowed"]) == ("auto", False)
    review = approval["review"]
    assert review["verdict"] == "high"
    expected = "delete" if name == "discard" else "system"
    assert expected in review["categories"]
    assert "allow this step once from the chat" in llm.tool_messages()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "code"), SYSTEM, ids=names(SYSTEM))
async def test_the_users_words_never_allow_a_system_step(
    tmp_path: Path, name: str, code: str
) -> None:
    llm = FakeModel(
        _plans([code], answer="explained"),
        [verdict(verdict="ask", category="none", user_asked="yes")],
    )
    agent, fake = _agent(llm, tmp_path / "ws")

    await agent.run_task(
        "please stop dev-server and open the Foo shortcut, go ahead",
        approval=AutoPolicy(),
    )

    assert fake.ran == []
    assert llm.review_requests == []  # click-only: the model is not asked either


@pytest.mark.asyncio
async def test_the_users_words_can_allow_discarding_what_they_name(
    tmp_path: Path,
) -> None:
    """``delete`` is a category words can allow, as ``git reset --hard`` always was."""
    assert "delete" in WORDS_CAN_ALLOW
    llm = FakeModel(
        _plans([DISCARD]),
        [verdict(verdict="ask", category="delete", user_asked="yes")],
    )
    agent, fake = _agent(llm, tmp_path / "ws")

    await agent.run_task("throw away my edits to notes.txt", approval=AutoPolicy())

    assert fake.ran == [DISCARD]
    [request] = llm.review_requests
    assert "throw away my edits to notes.txt" in request[1].content


@pytest.mark.asyncio
async def test_discarding_is_blocked_when_the_user_named_nothing_of_it(
    tmp_path: Path,
) -> None:
    llm = FakeModel(_plans([DISCARD], answer="explained"))
    agent, fake = _agent(llm, tmp_path / "ws")

    await agent.run_task("run the tests", approval=AutoPolicy())

    assert fake.ran == []
    assert llm.review_requests == []


class ChatApprover:
    """A person who answers every question with "Allow for this chat"."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision(True, remember=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "code"), EVERY, ids=names(EVERY))
async def test_ask_asks_every_time_and_the_card_names_the_step(
    tmp_path: Path, name: str, code: str
) -> None:
    llm = FakeModel(_plans(["print(1)", code, code], answer="done"))
    agent, fake = _agent(llm, tmp_path / "ws")
    approver = ChatApprover()

    await agent.run_task("do it", approval=RememberingPolicy(approver))

    reviews = [r.review for r in approver.asked]
    assert [r.verdict if r else "" for r in reviews] == ["routine", "high", "high"]
    assert fake.ran == ["print(1)", code, code]  # each click allowed one run
    card = approver.asked[1]
    assert card.review is not None
    assert covered_by_grant(card.review) is False
    assert offers_grant(card.review) is False
    expected = "delete" if name == "discard" else "system"
    assert expected in card.review.categories
    shown = approval_json(card)
    assert shown["review"]["categories"][0] in ("system", "delete")
    assert shown["review"]["title"] == review_title(card.review, "ws")
    assert shown["gist"] is not None
    assert shown["gist"]["by"] == "code"


def test_each_class_has_a_card_title_and_a_plain_sentence() -> None:
    from localmelo.melo.checker import Checker
    from localmelo.melo.checker.payloads import ActionReviewPayload
    from localmelo.melo.checker.review.gist import step_gist

    def card(command: str) -> tuple[str, str]:
        arguments = {"command": command}
        found = Checker().review_action(
            ActionReviewPayload("shell_exec", arguments, "/tmp/ws", home="/tmp")
        )
        gist = step_gist("shell_exec", arguments, found)
        assert gist is not None
        return review_title(found, "ws"), gist.id

    assert card("pkill -f dev-server") == ("Change system settings?", "command")
    assert card("diskutil eraseDisk JHFS+ Scratch disk2") == (
        "Change system settings?",
        "effect_delete",
    )
    assert card("git checkout -- .") == (
        "Delete files that cannot be restored?",
        "effect_delete",
    )
    assert card("open 'shortcuts://run-shortcut?name=Foo'") == (
        "Change system settings?",
        "command",
    )
