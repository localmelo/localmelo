"""Session grants and the approval broker."""

from __future__ import annotations

import asyncio
import dataclasses
import uuid

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.sessions.approvals import (
    RESOLVED_IDS_KEPT,
    ApprovalBroker,
    RememberingPolicy,
    RoutineOnlyPolicy,
)


def request(
    tool_name: str = "shell_exec", request_id: str | None = None
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id or uuid.uuid4().hex,
        run_id="run-1",
        call_id="call_1",
        tool_name=tool_name,
        arguments={"command": "ls"},
        preview="ls",
        workspace_root="/tmp",
    )


# ── Grants ──────────────────────────────────────────────────────────────────


async def test_grants_once_for_the_session_and_forget() -> None:
    answers = [
        ApprovalDecision(True),
        ApprovalDecision(True, remember=True),
        ApprovalDecision(False, "not that one"),
        ApprovalDecision(True),
    ]
    asked: list[str] = []

    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        asked.append(req.tool_name)
        return answers.pop(0)

    policy = RememberingPolicy(ask)
    assert isinstance(policy, ApprovalPolicy)
    # Allowed once: the next call asks again.
    assert (await policy.decide(request())).allow
    assert policy.allowed_tools == frozenset()
    # Allowed for the session: the next call of the tool does not ask.
    assert (await policy.decide(request())).allow
    assert policy.allowed_tools == {"shell_exec"}
    granted = await policy.decide(request())
    assert granted == ApprovalDecision(True, origin="session")
    assert asked == ["shell_exec", "shell_exec"]
    # Another tool still asks.
    denied = await policy.decide(request("file_write"))
    assert not denied.allow and asked[-1] == "file_write"
    # forget takes the grant back.
    policy.forget("shell_exec")
    assert policy.allowed_tools == frozenset()
    assert (await policy.decide(request())).allow
    assert asked[-1] == "shell_exec"


async def test_forget_without_a_name_takes_every_grant_back() -> None:
    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, remember=True)

    policy = RememberingPolicy(ask)
    await policy.decide(request("shell_exec"))
    await policy.decide(request("python_exec"))
    assert policy.allowed_tools == {"shell_exec", "python_exec"}
    policy.forget()
    assert policy.allowed_tools == frozenset()


# ── The broker ──────────────────────────────────────────────────────────────


async def test_an_answer_from_another_task_resolves_the_request() -> None:
    broker = ApprovalBroker()
    ask = broker.asker("s1")
    req = request()
    waiting = asyncio.ensure_future(ask(req))
    await asyncio.sleep(0)  # registered before the first await
    assert broker.pending("s1") == [req]
    assert broker.pending("s2") == []

    async def answer() -> None:
        assert broker.answer(req.request_id, ApprovalDecision(True)) == "resolved"

    await asyncio.create_task(answer())
    assert (await waiting).allow
    assert broker.pending("s1") == []
    assert broker.answer(req.request_id, ApprovalDecision(False)) == "already_resolved"
    assert broker.answer("never-seen", ApprovalDecision(False)) == "unknown"
    with pytest.raises(TypeError):
        broker.answer(req.request_id, "yes")  # type: ignore[arg-type]


async def test_synchronous_registration_cleans_up_if_cancelled_before_start() -> None:
    broker = ApprovalBroker()
    req = request()
    task = broker.begin("s1", req)
    assert broker.pending("s1") == [req]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.pending("s1") == []
    assert broker.answer(req.request_id, ApprovalDecision(True)) == "already_resolved"


async def test_pending_lists_the_oldest_first() -> None:
    broker = ApprovalBroker()
    ask = broker.asker("s1")
    first, second = request(), request()
    tasks = [asyncio.ensure_future(ask(first)), asyncio.ensure_future(ask(second))]
    await asyncio.sleep(0)
    assert broker.pending("s1") == [first, second]
    assert broker.withdraw_all("s1", reason="bye") == 2
    for task in tasks:
        assert not (await task).allow


async def test_the_first_answer_wins() -> None:
    broker = ApprovalBroker()
    req = request()
    task = asyncio.ensure_future(broker.asker("s1")(req))
    await asyncio.sleep(0)
    assert broker.answer(req.request_id, ApprovalDecision(False, "no")) == "resolved"
    assert broker.answer(req.request_id, ApprovalDecision(True)) == "already_resolved"
    decision = await task
    assert not decision.allow and decision.reason == "no"


async def test_a_timeout_denies_with_its_text() -> None:
    broker = ApprovalBroker(timeout_s=0.05)
    decision = await broker.asker("s1")(request())
    assert decision == ApprovalDecision(
        False, "No answer within 0.05 s", origin="timeout"
    )
    assert broker.pending("s1") == []
    with pytest.raises(ValueError):
        ApprovalBroker(timeout_s=0)


async def test_a_cancellation_removes_the_request_and_propagates() -> None:
    broker = ApprovalBroker()
    req = request()
    task = asyncio.ensure_future(broker.asker("s1")(req))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.pending("s1") == []
    assert broker.answer(req.request_id, ApprovalDecision(True)) == "already_resolved"


async def test_withdraw_all_denies_one_session_or_every_one() -> None:
    broker = ApprovalBroker()
    ones = [request(), request()]
    other = request()
    tasks = [asyncio.ensure_future(broker.asker("s1")(r)) for r in ones]
    tasks.append(asyncio.ensure_future(broker.asker("s2")(other)))
    await asyncio.sleep(0)
    assert broker.withdraw_all("s1", reason="The session closed") == 2
    for task in tasks[:2]:
        decision = await task
        assert decision == ApprovalDecision(
            False, "The session closed", origin="policy"
        )
    assert broker.pending("s2") == [other]
    assert broker.withdraw_all(reason="The gateway is stopping") == 1
    assert (await tasks[2]).reason == "The gateway is stopping"
    assert broker.withdraw_all(reason="nothing left") == 0


async def test_the_broker_remembers_the_last_resolved_ids() -> None:
    broker = ApprovalBroker()
    ids = []
    for _ in range(RESOLVED_IDS_KEPT + 1):
        req = request()
        task = asyncio.ensure_future(broker.asker("s1")(req))
        await asyncio.sleep(0)
        broker.answer(req.request_id, ApprovalDecision(True))
        await task
        ids.append(req.request_id)
    assert broker.answer(ids[0], ApprovalDecision(True)) == "unknown"  # forgotten
    assert broker.answer(ids[-1], ApprovalDecision(True)) == "already_resolved"


# ── Grants cover routine calls only (#279) ──────────────────────────────────

HIGH = ActionReview("high", ("install",), ("Installs x with pip.",))
UNDECIDED = ActionReview("undecided", (), ("Runs a command melo could not read.",))


def reviewed(review: ActionReview | None) -> ApprovalRequest:
    return dataclasses.replace(request("python_exec"), review=review)


async def test_a_grant_covers_routine_calls_and_never_a_high_impact_one() -> None:
    asked: list[ApprovalRequest] = []

    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        asked.append(req)
        return ApprovalDecision(True, remember=True)

    policy = RememberingPolicy(ask)
    first = await policy.decide(reviewed(ActionReview("routine")))
    assert first == ApprovalDecision(True, remember=True)
    assert policy.allowed_tools == {"python_exec"}
    routine = await policy.decide(reviewed(ActionReview("routine")))
    assert routine == ApprovalDecision(True, origin="session")
    assert await policy.decide(reviewed(None)) == routine  # nothing reviewed it
    # A high-impact call asks under the grant, and its "remember" is dropped.
    assert await policy.decide(reviewed(HIGH)) == ApprovalDecision(True)
    # An undecided call asks too (no model review yet); its grant is kept.
    assert await policy.decide(reviewed(UNDECIDED)) == ApprovalDecision(
        True, remember=True
    )
    assert [r.review for r in asked] == [ActionReview("routine"), HIGH, UNDECIDED]
    assert policy.allowed_tools == {"python_exec"}


async def test_a_high_impact_answer_never_grants_the_tool() -> None:
    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, remember=True)

    policy = RememberingPolicy(ask)
    await policy.decide(reviewed(HIGH))
    assert policy.allowed_tools == frozenset()


async def test_routine_only_runs_routine_calls_and_asks_about_the_rest() -> None:
    asked: list[ApprovalRequest] = []

    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        asked.append(req)
        return ApprovalDecision(False, "no")

    policy = RoutineOnlyPolicy(RememberingPolicy(ask))
    allowed = ApprovalDecision(True, origin="policy")
    assert await policy.decide(reviewed(ActionReview("routine"))) == allowed
    assert await policy.decide(reviewed(None)) == allowed
    assert await policy.decide(reviewed(HIGH)) == ApprovalDecision(False, "no")
    assert await policy.decide(reviewed(UNDECIDED)) == ApprovalDecision(False, "no")
    assert [r.review for r in asked] == [HIGH, UNDECIDED]
