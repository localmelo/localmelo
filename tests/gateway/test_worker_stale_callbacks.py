"""Generation-one callbacks must not mutate a replacement worker's fences."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.melo.contracts.providers import LLM_REQUEST_KIND
from localmelo.melo.schema import Message
from localmelo.support.gateway import workers as worker_module
from localmelo.support.gateway.worker_evidence import SafePointEvidence
from localmelo.support.gateway.workers import TaskWorkers, _WorkerLLM

TASK = "task"


def seam() -> Any:
    workers = TaskWorkers(Mock(writer="gateway"), Mock(), Mock(), Mock())
    workers.tasks = Mock()
    workers.tasks.park_identity = AsyncMock(return_value=(2, "generation-2"))
    workers.tasks.get_task = AsyncMock(return_value=SimpleNamespace(level=1))
    workers.parks = Mock()
    workers.parks.hold = AsyncMock()
    workers.parks.resume = AsyncMock()
    workers._ready[TASK] = asyncio.Event()
    return workers


def generation(name: str) -> tuple[Any, Any, tuple, tuple]:
    runner = SimpleNamespace(current_turn=SimpleNamespace(id=name))
    call = object()
    evidence = SafePointEvidence(name, "approval", 1, 1, 1, name)
    return runner, call, (call, runner, name), (runner, name, evidence)


def assert_unchanged(workers: Any, call_point: tuple, approval_point: tuple) -> None:
    assert workers._points[TASK] is call_point
    assert workers._approval_points[TASK] is approval_point
    assert not workers._ready[TASK].is_set()
    workers.parks.resume.assert_not_awaited()
    workers.parks.hold.assert_not_awaited()


@pytest.mark.parametrize("site", ["call", "approval"])
async def test_blocked_old_release_cannot_resume_new_generation(site: str) -> None:
    workers = seam()
    old_runner, old_call, old_call_point, old_approval = generation("turn-1")
    new_runner, _, new_call_point, new_approval = generation("turn-2")
    workers._runners[TASK] = old_runner
    workers._points[TASK] = old_call_point
    workers._approval_points[TASK] = old_approval

    # The old callback is waiting at the same coordinator lock as a replacement.
    await workers._lock.acquire()
    if site == "call":
        callback = asyncio.create_task(
            workers._release_call(TASK, old_call, old_runner, "turn-1")
        )
    else:
        callback = asyncio.create_task(workers._release_approval(TASK, old_approval))
    await asyncio.sleep(0)
    assert not callback.done()
    workers._runners[TASK] = new_runner
    workers._points[TASK] = new_call_point
    workers._approval_points[TASK] = new_approval
    workers._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(callback, 3)
    assert_unchanged(workers, new_call_point, new_approval)


async def test_blocked_old_approval_hold_cannot_replace_new_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = seam()
    old_runner, _, old_call_point, old_approval = generation("turn-1")
    new_runner, _, new_call_point, new_approval = generation("turn-2")
    workers._runners[TASK] = old_runner
    workers._points[TASK] = old_call_point
    workers._approval_points[TASK] = old_approval
    entered, unblock = asyncio.Event(), asyncio.Event()

    async def evidence(*args: object, **kwargs: object) -> SafePointEvidence:
        entered.set()
        await unblock.wait()
        return cast(SafePointEvidence, old_approval[2])

    monkeypatch.setattr(worker_module, "committed_evidence", evidence)
    request = ApprovalRequest("request-1", TASK, "call-1", "write", {}, "", None)
    callback = asyncio.create_task(workers._hold_approval(TASK, old_runner, request))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        workers._runners[TASK] = new_runner
        workers._points[TASK] = new_call_point
        workers._approval_points[TASK] = new_approval
        unblock.set()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(callback, 3)
        assert_unchanged(workers, new_call_point, new_approval)
        assert callback.cancelled()
    finally:
        unblock.set()
        if not callback.done():
            callback.cancel()
            await asyncio.gather(callback, return_exceptions=True)


@pytest.mark.parametrize("site", ["call", "approval"])
async def test_old_hold_after_replacement_cannot_close_new_fence(site: str) -> None:
    workers = seam()
    old_runner, old_call, _, _ = generation("turn-1")
    new_runner, _, new_call_point, new_approval = generation("turn-2")
    workers._runners[TASK] = new_runner
    workers._points[TASK] = new_call_point
    workers._approval_points[TASK] = new_approval
    workers._ready[TASK].set()
    with pytest.raises(asyncio.CancelledError):
        if site == "call":
            await workers._hold_call(TASK, old_call, old_runner, "turn-1", "plan")
        else:
            request = ApprovalRequest("old", TASK, "call", "write", {}, "", None)
            await workers._hold_approval(TASK, old_runner, request)
    assert workers._ready[TASK].is_set()
    assert workers._points[TASK] is new_call_point
    assert workers._approval_points[TASK] is new_approval
    workers.parks.hold.assert_not_awaited()


@pytest.mark.parametrize("change", ["runner", "turn"])
async def test_blocked_old_llm_never_starts_after_generation_changes(
    change: str,
) -> None:
    workers = seam()
    old_runner, _, old_call_point, old_approval = generation("turn-1")
    new_runner, _, new_call_point, new_approval = generation("turn-2")
    workers._runners[TASK] = old_runner
    workers._points[TASK] = old_call_point
    workers._approval_points[TASK] = old_approval
    call = SimpleNamespace(meta=SimpleNamespace(admitted_mono_us=1))
    gate = Mock()
    gate.submit.return_value = call
    gate.acquire = AsyncMock()
    client: Any = SimpleNamespace(gate=gate, session_id=TASK)
    provider = Mock()
    provider.chat = AsyncMock(return_value=Message(role="assistant", content="bad"))
    entered, unblock = asyncio.Event(), asyncio.Event()

    async def hold(*args: object) -> None:
        entered.set()
        await unblock.wait()

    release = AsyncMock()
    started = Mock()
    llm = _WorkerLLM(
        provider,
        client,
        workers._ready[TASK],
        lambda: workers._runners[TASK],
        hold,
        release,
        started,
    )
    token = LLM_REQUEST_KIND.set("plan")
    try:
        callback = asyncio.create_task(llm.chat([]))
        await asyncio.wait_for(entered.wait(), 3)
        workers._points[TASK] = new_call_point
        workers._approval_points[TASK] = new_approval
        if change == "runner":
            workers._runners[TASK] = new_runner
        else:
            old_runner.current_turn = new_runner.current_turn
        # A generation-two opening must not allow the stale call through.
        workers._ready[TASK].set()
        unblock.set()
        with pytest.raises(RuntimeError, match="Worker turn changed"):
            await asyncio.wait_for(callback, 3)
        started.assert_not_called()
        provider.chat.assert_not_awaited()
        gate.finish.assert_called_once_with(call, "refused")
        workers._ready[TASK].clear()
        assert_unchanged(workers, new_call_point, new_approval)
    finally:
        unblock.set()
        if not callback.done():
            callback.cancel()
            await asyncio.gather(callback, return_exceptions=True)
        LLM_REQUEST_KIND.reset(token)
