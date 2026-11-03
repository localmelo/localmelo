"""Session grants and the approval broker."""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import Awaitable, Callable

import pytest

from localmelo.melo.contracts.approvals import (
    AUTO_ASK_WAIT_S,
    LEFT_UNANSWERED,
    NO_ANSWER,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    Withdrawal,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.sessions.approvals import (
    CARD_LEFT_S,
    RESOLVED_IDS_KEPT,
    ApprovalBroker,
    CardWatch,
    RememberingPolicy,
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
        # Withdrawn unanswered: no one denied the step itself (#411).
        assert decision == Withdrawal(False, "The session closed", origin="policy")
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


@pytest.mark.parametrize(
    "name", ["browser-submit:https://example.com", "browser-type:https://example.com"]
)
async def test_sending_a_form_and_typing_are_never_granted_for_the_session(
    name: str,
) -> None:
    asked: list[str] = []

    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        asked.append(req.tool_name)
        return ApprovalDecision(True, remember=True)

    policy = RememberingPolicy(ask)
    send = request(name)
    # Whatever a client sends, the answer allows this send and nothing after it.
    assert await policy.decide(send) == ApprovalDecision(True)
    assert policy.allowed_tools == frozenset()
    assert await policy.decide(send) == ApprovalDecision(True)
    assert asked == [name] * 2
    # A click or a choice keeps its one answer for the site.
    assert await policy.decide(request("browser-act:https://example.com")) == (
        ApprovalDecision(True, remember=True)
    )
    assert policy.allowed_tools == frozenset({"browser-act:https://example.com"})


async def test_a_grant_is_silent_only_for_its_tool() -> None:
    # Under a grant the gate lets the user's model review an undecided call
    # first (the routine-only policy of the old terminal auto mode is gone:
    # auto mode is AutoPolicy now, tested in tests/agent/test_auto_mode.py).
    async def ask(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, remember=True)

    policy = RememberingPolicy(ask)
    assert not policy.silent_for(reviewed(UNDECIDED))
    await policy.decide(reviewed(ActionReview("routine")))
    assert policy.silent_for(reviewed(UNDECIDED))
    assert not policy.silent_for(request("shell_exec"))


# ── Auto's card: who sees it and how long it waits (#465) ──────────────────


class FakeTime:
    """A clock that moves only when the code under test sleeps; *at* runs a step at a time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.at: dict[float, Callable[[], object]] = {}

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        step = self.at.pop(self.now, None)
        if step is not None:
            step()
        await asyncio.sleep(0)


def counted(viewers: list[int]) -> Callable[[], Awaitable[int]]:
    async def count() -> int:
        return viewers[0]

    return count


def watched_card(
    viewers: list[int], time_: FakeTime, broker: ApprovalBroker
) -> CardWatch:
    return CardWatch(
        counted(viewers), broker.ends, clock=time_.clock, sleep=time_.sleep
    )


async def test_a_card_nobody_answers_in_a_minute_ends_as_a_timeout() -> None:
    broker, time_, viewers = ApprovalBroker(), FakeTime(), [1]
    asked = request()

    decided = await asyncio.wait_for(
        watched_card(viewers, time_, broker).ask(
            asked, RememberingPolicy(broker.asker("s"))
        ),
        5,
    )

    assert decided == ApprovalDecision(False, NO_ANSWER, origin="timeout")
    assert time_.now == AUTO_ASK_WAIT_S == 60.0  # not a second earlier
    assert broker.pending("s") == []
    assert broker.answer(asked.request_id, ApprovalDecision(True)) == "already_resolved"


async def test_an_answer_before_the_minute_is_the_persons() -> None:
    broker, time_, viewers = ApprovalBroker(), FakeTime(), [1]
    asked = request()
    time_.at[AUTO_ASK_WAIT_S - 1] = lambda: broker.answer(
        asked.request_id, ApprovalDecision(True)
    )

    decided = await asyncio.wait_for(
        watched_card(viewers, time_, broker).ask(
            asked, RememberingPolicy(broker.asker("s"))
        ),
        5,
    )

    assert decided == ApprovalDecision(True)


async def test_the_last_viewer_leaving_ends_the_card() -> None:
    broker, time_, viewers = ApprovalBroker(), FakeTime(), [2]
    asked = request()
    time_.at[10.0] = lambda: viewers.__setitem__(0, 0)

    decided = await asyncio.wait_for(
        watched_card(viewers, time_, broker).ask(
            asked, RememberingPolicy(broker.asker("s"))
        ),
        5,
    )

    assert decided == ApprovalDecision(False, LEFT_UNANSWERED, origin="timeout")
    # A full grace from the first check that found nobody.
    assert time_.now == 10.0 + CARD_LEFT_S < AUTO_ASK_WAIT_S


async def test_a_reload_of_the_page_does_not_end_the_card() -> None:
    broker, time_, viewers = ApprovalBroker(), FakeTime(), [1]
    asked = request()
    # The stream closes and opens again within the grace, then the person answers.
    time_.at[10.0] = lambda: viewers.__setitem__(0, 0)
    time_.at[12.0] = lambda: viewers.__setitem__(0, 1)
    time_.at[30.0] = lambda: broker.answer(asked.request_id, ApprovalDecision(False))

    decided = await asyncio.wait_for(
        watched_card(viewers, time_, broker).ask(
            asked, RememberingPolicy(broker.asker("s"))
        ),
        5,
    )

    assert decided == ApprovalDecision(False)  # the person's Deny, not a timeout


async def test_a_card_not_up_yet_is_ended_once_it_is() -> None:
    time_, tries = FakeTime(), []
    answer: asyncio.Future[ApprovalDecision] = (
        asyncio.get_running_loop().create_future()
    )

    def end(request_id: str, decision: ApprovalDecision) -> bool:
        tries.append(request_id)
        if len(tries) < 3:
            return False  # its event is still being written
        answer.set_result(decision)
        return True

    class Waits:
        async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
            return await answer

    watch = CardWatch(counted([1]), end, clock=time_.clock, sleep=time_.sleep)
    decided = await asyncio.wait_for(watch.ask(request(), Waits()), 5)

    assert decided.origin == "timeout" and len(tries) == 3


async def test_a_count_that_fails_watches_nobody() -> None:
    async def broken() -> int:
        raise RuntimeError("the hub is closed")

    def ends(request_id: str, decision: ApprovalDecision) -> bool:
        return True

    assert not await CardWatch(broken, ends).watched()
    assert not await CardWatch(counted([0]), ends).watched()
    assert await CardWatch(counted([1]), ends).watched()
