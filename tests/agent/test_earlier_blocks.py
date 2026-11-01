"""A step blocked earlier in a task is refused again before any review (#411).

A task its chat continued on its own runs with an
:class:`~localmelo.melo.contracts.approvals.EarlierBlocks` policy: the agent's
gate refuses a call with the same key as a blocked one, says who blocked it,
and never asks the inner policy or reviews the call.
"""

from __future__ import annotations

import pytest

from localmelo.melo.contracts.approvals import (
    AllowAllPolicy,
    ApprovalDecision,
    AutoPolicy,
    DenyAllPolicy,
    EarlierBlocks,
    call_key,
    find_auto,
    find_earlier,
)

from .test_approvals import Policy, Recorder, ScriptedLLM, _agent, _answer, _deploy


@pytest.mark.asyncio
async def test_a_call_blocked_earlier_is_refused_before_the_policy_is_asked() -> None:
    llm = ScriptedLLM([_deploy(), _deploy("call_2", "staging"), _answer("Done.")])
    inner = Policy(ApprovalDecision(True, origin="user"))
    agent, tools = _agent(llm)
    root = agent.executor.workspace_root
    blocked = call_key("deploy", {"target": "prod"}, root)
    policy = EarlierBlocks(inner, {blocked: ("Auto mode", "")})
    recorder = Recorder()

    result = await agent.run_task("deploy", on_event=recorder, approval=policy)

    assert result.status == "completed"
    # The same call is refused, a different one is asked as usual.
    assert tools.ran == [("deploy", {"target": "staging"})]
    assert [request.arguments for request in inner.requests] == [{"target": "staging"}]
    [refused] = llm.tool_messages(1)
    assert "Auto mode blocked this exact step earlier in this task" in refused
    assert "find another way" in refused
    first, second = [event.data for event in recorder.of("tool.finished")]
    assert (first["status"], first["approval"]["origin"]) == ("blocked", "policy")
    assert first["approval"]["earlier"] == "Auto mode"
    assert first["approval"]["call_key"] == blocked
    assert "review" not in first["approval"], "refused before any review"
    assert second["approval"]["allowed"] is True


@pytest.mark.asyncio
async def test_a_denial_keeps_its_call_key_and_an_allowance_lifts_the_block() -> None:
    llm = ScriptedLLM([_deploy(), _answer("Not deploying.")])
    agent, _ = _agent(llm)
    recorder = Recorder()
    await agent.run_task("deploy", on_event=recorder, approval=DenyAllPolicy("no"))
    [finished] = recorder.of("tool.finished")
    key = call_key("deploy", {"target": "prod"}, agent.executor.workspace_root)
    assert finished.data["approval"]["call_key"] == key

    auto = AutoPolicy(task=True)
    policy = EarlierBlocks(auto, {key: ("Auto mode", "step-1")})
    assert find_earlier(policy) is policy and find_auto(policy) is auto
    root = agent.executor.workspace_root
    assert policy.blocked("deploy", {"target": "prod"}, root) == "Auto mode"
    # The person allowed that step once since: it is asked again, not refused.
    auto.allow_once("step-1", ["install"], "deploy")
    assert policy.blocked("deploy", {"target": "prod"}, root) is None
    assert find_earlier(AllowAllPolicy()) is None
