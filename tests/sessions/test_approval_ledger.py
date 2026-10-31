"""A task's approvals go through its chat: recorded approvals matched by code, never by a model."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    ALLOWED_ONCE,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    find_auto,
    review_mode,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.sessions.approvals import (
    ApprovalLedger,
    ApprovalUse,
    OwnedTaskPolicy,
    RememberingPolicy,
    request_step,
    reusable,
)

ROUTINE = ActionReview("routine")
UNDECIDED = ActionReview("undecided", (), ("melo could not tell.",))
HIGH = ActionReview("high", ("install",), ("Installs requests with pip.",))
SELF = ActionReview("high", ("self",), ("Changes localmelo's settings.",))
CLICK = ActionReview("high", ("credentials",), ("Reads your keys.",))
WORDS_ALLOWED = ActionReview("routine", ("download",), ("Downloads a file.",))


def request(
    folder: str,
    *,
    code: str = "print(1)",
    tool: str = "python_exec",
    review: ActionReview | None = ROUTINE,
    arguments: Mapping[str, Any] | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=uuid.uuid4().hex,
        run_id="task-1",
        call_id="call-1",
        tool_name=tool,
        arguments=dict(arguments) if arguments is not None else {"code": code},
        preview=code,
        workspace_root=folder,
        tier="exec",
        review=review,
    )


class Asker:
    """The person: answers each ask with *decision*, counting the asks."""

    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, asked: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(asked)
        return self.decision


@pytest.fixture()
def folder(tmp_path: Path) -> str:
    path = tmp_path / "ws"
    path.mkdir()
    return str(path)


def owned(
    ledger: ApprovalLedger,
    inner: Any,
    task: str = "task-1",
    chat: str | None = "chat-1",
    uses: list[ApprovalUse] | None = None,
) -> OwnedTaskPolicy:
    async def owner() -> str | None:
        return chat

    async def on_use(use: ApprovalUse) -> None:
        if uses is not None:
            uses.append(use)

    return OwnedTaskPolicy(inner, ledger, task, owner, on_use)


def test_reusable_only_for_reviewed_calls_with_no_category() -> None:
    assert reusable(ROUTINE) and reusable(UNDECIDED)
    for review in (None, HIGH, SELF, CLICK, WORDS_ALLOWED):
        assert not reusable(review)


def test_the_ledger_matches_the_exact_step_in_the_same_folder(
    folder: str, tmp_path: Path
) -> None:
    ledger = ApprovalLedger(clock=lambda: 1000.0)
    first = request(folder)
    entry = ledger.record("chat-1", first)
    assert entry is not None and entry.step_key == request_step(first)
    assert ledger.match("chat-1", request(folder)) == entry
    # A comment does not change the step; one argument does; another folder does.
    assert ledger.match("chat-1", request(folder, code="print(1)  # same")) == entry
    assert ledger.match("chat-1", request(folder, code="print(2)")) is None
    other = tmp_path / "other"
    other.mkdir()
    assert ledger.match("chat-1", request(str(other))) is None
    # Another chat's approvals are not this chat's.
    assert ledger.match("chat-2", request(folder)) is None
    # The same call is another step for another tool.
    shell = request(folder, tool="file_write", arguments={"path": "a", "content": "b"})
    assert ledger.match("chat-1", shell) is None


def test_a_runner_whose_unreadable_changed_files_changed_asks_again(
    folder: str,
) -> None:
    # The step key is the command; the files a runner runs that melo could not read
    # are compared as they were when the person allowed it.
    def pytest_step(changed: tuple[str, ...]) -> ApprovalRequest:
        review = ActionReview("undecided", (), ("Runs lib.so.",), changed=changed)
        return request(folder, code="import pytest; pytest.main()", review=review)

    ledger = ApprovalLedger()
    entry = ledger.record("chat-1", pytest_step(("lib.so:100:4",)))
    assert ledger.match("chat-1", pytest_step(("lib.so:100:4",))) == entry
    assert ledger.match("chat-1", pytest_step(("lib.so:200:9",))) is None
    assert ledger.match("chat-1", pytest_step(())) is None


def test_high_impact_steps_are_never_recorded_or_matched(folder: str) -> None:
    ledger = ApprovalLedger()
    for review in (HIGH, SELF, CLICK, WORDS_ALLOWED, None):
        assert ledger.record("chat-1", request(folder, review=review)) is None
    assert ledger.entries("chat-1") == []
    ledger.record("chat-1", request(folder))
    # The same step reviewed as high-impact now (another rules version) asks.
    assert ledger.match("chat-1", request(folder, review=HIGH)) is None


def test_revoke_drop_and_entries(folder: str) -> None:
    ledger = ApprovalLedger()
    one = ledger.record("chat-1", request(folder))
    two = ledger.record("chat-1", request(folder, code="print(2)"))
    assert one is not None and two is not None
    assert [e.entry_id for e in ledger.entries("chat-1")] == [
        two.entry_id,
        one.entry_id,
    ]
    assert ledger.revoke("chat-1", one.entry_id) is True
    assert ledger.revoke("chat-1", one.entry_id) is False
    assert ledger.revoke("chat-9", two.entry_id) is False
    assert ledger.match("chat-1", request(folder)) is None
    ledger.drop("chat-1")
    assert ledger.entries("chat-1") == []
    # Recording the same step again replaces its entry.
    ledger.record("chat-1", request(folder))
    ledger.record("chat-1", request(folder))
    assert len(ledger.entries("chat-1")) == 1


async def test_an_identical_step_runs_on_the_earlier_approval(folder: str) -> None:
    ledger = ApprovalLedger()
    person = Asker(ApprovalDecision(True))
    uses: list[ApprovalUse] = []
    first = owned(ledger, RememberingPolicy(person), uses=uses)
    assert (await first.decide(request(folder))).origin == "user"
    assert len(person.asked) == 1 and len(ledger.entries("chat-1")) == 1
    second = owned(ledger, RememberingPolicy(person), task="task-2", uses=uses)
    again = request(folder)
    decision = await second.decide(again)
    assert decision.allow and decision.origin == "earlier"
    assert len(person.asked) == 1
    (use,) = uses
    assert (use.task_id, use.request_id) == ("task-2", again.request_id)
    assert ledger.uses_of("task-2")[0]["live"] is True
    # Revoked: the step asks again, and the step that ran is still listed.
    ledger.revoke("chat-1", use.entry_id)
    assert ledger.uses_of("task-2")[0]["live"] is False
    assert (await second.decide(request(folder))).origin == "user"
    assert len(person.asked) == 2


async def test_a_step_that_differs_asks(folder: str, tmp_path: Path) -> None:
    ledger = ApprovalLedger()
    person = Asker(ApprovalDecision(True))
    policy = owned(ledger, RememberingPolicy(person))
    await policy.decide(request(folder))
    await policy.decide(request(folder, code="print(1, 2)"))
    other = tmp_path / "other"
    other.mkdir()
    await policy.decide(request(str(other)))
    assert len(person.asked) == 3


async def test_a_high_impact_identical_step_asks_every_time(folder: str) -> None:
    ledger = ApprovalLedger()
    person = Asker(ApprovalDecision(True))
    policy = owned(ledger, RememberingPolicy(person))
    for _ in range(3):
        decision = await policy.decide(request(folder, review=HIGH))
        assert decision.origin == "user"
    assert len(person.asked) == 3
    assert ledger.entries("chat-1") == []


async def test_a_step_nothing_reviewed_asks_every_time_and_is_never_recorded(
    folder: str,
) -> None:
    ledger = ApprovalLedger()
    entry = ledger.record("chat-1", request(folder))
    person = Asker(ApprovalDecision(True))
    policy = owned(ledger, RememberingPolicy(person))
    for _ in range(2):
        decision = await policy.decide(request(folder, review=None))
        assert (decision.allow, decision.origin) == (True, "user")
    assert len(person.asked) == 2
    assert ledger.entries("chat-1") == [entry]


async def test_a_denial_or_a_task_without_a_chat_records_nothing(folder: str) -> None:
    ledger = ApprovalLedger()
    no = owned(ledger, RememberingPolicy(Asker(ApprovalDecision(False, "no"))))
    assert (await no.decide(request(folder))).allow is False
    assert ledger.entries("chat-1") == []
    alone = owned(ledger, RememberingPolicy(Asker(ApprovalDecision(True))), chat=None)
    assert (await alone.decide(request(folder))).origin == "user"
    assert ledger.entries("chat-1") == []


async def test_auto_allow_once_stays_one_step_once(folder: str) -> None:
    ledger = ApprovalLedger()
    auto = AutoPolicy(task=True)
    policy = owned(ledger, auto)
    blocked = request(folder, review=UNDECIDED)
    assert (await policy.decide(blocked)).origin == "auto"
    auto.allow_once(request_step(blocked), [], "python_exec")
    allowed = await policy.decide(request(folder, review=UNDECIDED))
    assert allowed.allow and allowed.reason == ALLOWED_ONCE
    # Not recorded: the same step is blocked again.
    assert ledger.entries("chat-1") == []
    assert (await policy.decide(request(folder, review=UNDECIDED))).allow is False
    assert find_auto(policy) is auto
    assert review_mode(policy, blocked) == "auto"


async def test_an_earlier_approval_runs_a_blocked_auto_step(folder: str) -> None:
    ledger = ApprovalLedger()
    ledger.record("chat-1", request(folder, review=UNDECIDED))
    policy = owned(ledger, AutoPolicy(task=True))
    decision = await policy.decide(request(folder, review=UNDECIDED))
    assert decision.origin == "earlier"


async def test_in_auto_what_auto_runs_runs_as_it_would(folder: str) -> None:
    ledger = ApprovalLedger()
    ledger.record("chat-1", request(folder))
    blocked = request(folder, code="print(2)", review=UNDECIDED)
    ledger.record("chat-1", blocked)
    auto = AutoPolicy(task=True)
    uses: list[ApprovalUse] = []
    policy = owned(ledger, auto, uses=uses)
    # A routine step Auto runs anyway uses no approval and shows no row.
    assert (await policy.decide(request(folder))).origin == "auto"
    # A step allowed once on the task's page runs on that allowance, once.
    auto.allow_once(request_step(blocked), [], "python_exec")
    once = await policy.decide(request(folder, code="print(2)", review=UNDECIDED))
    assert (once.origin, once.reason) == ("user", ALLOWED_ONCE)
    assert uses == []
    # Then only the person's earlier approval runs it.
    again = await policy.decide(request(folder, code="print(2)", review=UNDECIDED))
    assert again.origin == "earlier" and len(uses) == 1
    assert len(ledger.entries("chat-1")) == 2


async def test_the_owner_chats_grant_covers_its_tasks_routine_calls(
    folder: str,
) -> None:
    ledger = ApprovalLedger()
    granted: dict[str, frozenset[str]] = {"chat-1": frozenset({"python_exec"})}
    ledger.grants_of = lambda chat: granted.get(chat, frozenset())
    person = Asker(ApprovalDecision(True))
    policy = owned(ledger, RememberingPolicy(person))
    routine = await policy.decide(request(folder, code="print(3)"))
    assert routine.origin == "session" and person.asked == []
    # The gate lets the user's model review an undecided call first, as under a grant.
    assert review_mode(policy, request(folder, review=UNDECIDED)) == "silent"
    assert review_mode(policy, request(folder, tool="file_write")) == "ask"
    # A high-impact call asks every time.
    assert (await policy.decide(request(folder, review=HIGH))).origin == "user"
    assert len(person.asked) == 1
    # Another chat's grant covers nothing here.
    granted.clear()
    granted["chat-2"] = frozenset({"python_exec"})
    assert (await policy.decide(request(folder, code="print(4)"))).origin == "user"


async def test_a_failing_row_does_not_stop_the_step(folder: str) -> None:
    ledger = ApprovalLedger()
    ledger.record("chat-1", request(folder))

    async def owner() -> str | None:
        return "chat-1"

    async def broken(use: ApprovalUse) -> None:
        raise RuntimeError("store closed")

    policy = OwnedTaskPolicy(
        RememberingPolicy(Asker(ApprovalDecision(False))), ledger, "t", owner, broken
    )
    assert (await policy.decide(request(folder))).origin == "earlier"


def test_the_ledger_compares_resolved_folders(folder: str, tmp_path: Path) -> None:
    link = tmp_path / "link"
    os.symlink(folder, link)
    ledger = ApprovalLedger()
    ledger.record("chat-1", request(folder))
    assert ledger.match("chat-1", request(str(link))) is not None
