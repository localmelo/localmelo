"""``ask``: the one way a policy is asked for a decision, and how it fails closed."""

from __future__ import annotations

import asyncio

import pytest

from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest, ask

NO_DECISION = "the approval policy returned {}, not an ApprovalDecision"
STRAY = "the approval policy raised CancelledError while the run was not cancelled"


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        request_id="0" * 32,
        run_id="run",
        call_id="call_1",
        tool_name="shell_exec",
        arguments={"command": "ls"},
        preview="ls",
        workspace_root=None,
    )


class Record:
    """What the caller's record is told: each ``(origin, error)``."""

    def __init__(self) -> None:
        self.told: list[tuple[str, BaseException | None]] = []

    def __call__(self, origin: str, error: BaseException | None) -> None:
        self.told.append((origin, error))


async def test_a_decision_passes_through_and_nothing_is_recorded() -> None:
    request, record = _request(), Record()
    decision = ApprovalDecision(True, "fine", remember=True)
    seen: list[ApprovalRequest] = []

    async def decide(asked: ApprovalRequest) -> ApprovalDecision:
        seen.append(asked)
        return decision

    assert await ask(decide, request, record=record) is decision
    assert seen == [request]
    assert record.told == []


@pytest.mark.parametrize("answer", [None, "yes", True, {"allow": True}])
async def test_a_return_that_is_no_decision_is_refused_and_recorded(
    answer: object,
) -> None:
    record = Record()

    async def decide(asked: ApprovalRequest) -> object:
        return answer

    with pytest.raises(TypeError) as caught:
        await ask(decide, _request(), record=record)

    assert str(caught.value) == NO_DECISION.format(type(answer).__name__)
    assert record.told == [("error", caught.value)]


async def test_an_exception_is_recorded_and_raised_as_it_is() -> None:
    record = Record()
    failure = RuntimeError("the policy is down")

    async def decide(asked: ApprovalRequest) -> ApprovalDecision:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        await ask(decide, _request(), record=record)

    assert caught.value is failure
    assert record.told == [("error", failure)]


async def test_a_cancelled_error_while_nothing_cancels_the_run_is_a_refusal() -> None:
    record = Record()

    async def decide(asked: ApprovalRequest) -> ApprovalDecision:
        raise asyncio.CancelledError()

    with pytest.raises(RuntimeError) as caught:
        await ask(decide, _request(), record=record)

    assert str(caught.value) == STRAY
    assert caught.value.__suppress_context__
    assert record.told == [("error", caught.value)]


async def test_a_cancellation_of_the_caller_propagates_as_a_withdrawal() -> None:
    record = Record()
    waiting = asyncio.Event()

    async def decide(asked: ApprovalRequest) -> ApprovalDecision:
        waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(ask(decide, _request(), record=record))
    await waiting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert record.told == [("withdrawn", None)]
