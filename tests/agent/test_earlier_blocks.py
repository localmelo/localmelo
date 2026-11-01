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
    Withdrawal,
    call_key,
    find_auto,
    find_earlier,
)
from localmelo.melo.contracts.review import loose_keys

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
    assert "call_key" not in first["approval"], "a refusal of a repeat is no new block"
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


@pytest.mark.asyncio
async def test_a_withdrawn_question_is_no_block_of_the_step() -> None:
    # The gateway stopping withdraws the question: nobody denied the step itself.
    llm = ScriptedLLM([_deploy(), _answer("Stopped.")])
    agent, _ = _agent(llm)
    recorder = Recorder()
    withdrawn = Policy(Withdrawal(False, "The gateway is stopping", origin="policy"))
    await agent.run_task("deploy", on_event=recorder, approval=withdrawn)
    [finished] = recorder.of("tool.finished")
    assert finished.data["approval"]["allowed"] is False
    assert "call_key" not in finished.data["approval"]
    with pytest.raises(ValueError):
        Withdrawal(True, origin="policy")


PIP = 'import subprocess\nsubprocess.run(["pip", "install", "requests"], check=True)'


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("shell_exec", {"command": "pip  install requests "}),
        ("shell_exec", {"command": "pip3 install 'requests'"}),
        ("shell_exec", {"command": "python -m pip install -q requests"}),
        ("shell_exec", {"command": "python3 -m pip install --quiet requests"}),
        ("python_exec", {"code": 'import os\nos.system("pip install requests")'}),
        ("shell_exec", {"command": "pip3 install --user requests"}),
    ],
)
def test_a_step_alike_is_refused_too(tool: str, arguments: dict[str, str]) -> None:
    blocked = call_key("python_exec", {"code": PIP}, "/w")
    loose = {blocked: loose_keys("python_exec", {"code": PIP}, ("install",))}
    policy = EarlierBlocks(AllowAllPolicy(), {blocked: ("Auto mode", "")}, loose)
    assert policy.match("python_exec", {"code": PIP}, "/w") == ("Auto mode", True)
    reflowed = {"code": PIP.replace("check=True", "check = True")}
    assert policy.match("python_exec", reflowed, "/w") == ("Auto mode", True)
    assert policy.match(tool, arguments, "/w") == ("Auto mode", False)
    # Another program, or another package, is another step.
    assert policy.match("shell_exec", {"command": "pip install numpy"}, "/w") is None
    assert policy.match("shell_exec", {"command": "ls -la"}, "/w") is None


def test_an_allowance_of_the_blocked_step_lifts_its_steps_alike() -> None:
    auto = AutoPolicy(task=True)
    blocked = call_key("python_exec", {"code": PIP}, "/w")
    policy = EarlierBlocks(
        auto,
        {blocked: ("Auto mode", "step-1")},
        {blocked: loose_keys("python_exec", {"code": PIP}, ("install",))},
    )
    alike = {"command": "pip3 install requests"}
    assert policy.match("shell_exec", alike, "/w") == ("Auto mode", False)
    auto.allow_once("step-1", ["install"], "python_exec")
    assert policy.match("shell_exec", alike, "/w") is None


def _blocked(
    tool: str, arguments: dict[str, str], categories: tuple[str, ...]
) -> EarlierBlocks:
    key = call_key(tool, arguments, "/w")
    return EarlierBlocks(
        AllowAllPolicy(),
        {key: ("Auto mode", "")},
        {key: loose_keys(tool, arguments, categories)},
    )


TWO_CALLS = (
    "import subprocess\n"
    'subprocess.run(["pip", "install", "numpy"])\n'
    'subprocess.run(["python", "fetch.py"])'
)


@pytest.mark.parametrize(
    ("tool", "arguments", "workaround"),
    [
        (
            "shell_exec",
            {"command": "pip install numpy && python fetch.py"},
            ("shell_exec", {"command": "python fetch.py"}),
        ),
        (
            "shell_exec",
            {"command": "cd /w/app && pip install numpy"},
            ("shell_exec", {"command": "cd /w/app && pytest -q"}),
        ),
        (
            "python_exec",
            {"code": TWO_CALLS},
            (
                "python_exec",
                {"code": 'import subprocess\nsubprocess.run(["python", "fetch.py"])'},
            ),
        ),
    ],
)
def test_only_the_blocked_part_of_a_compound_step_is_refused_again(
    tool: str, arguments: dict[str, str], workaround: tuple[str, dict[str, str]]
) -> None:
    policy = _blocked(tool, arguments, ("install",))
    # The way around it runs; the install alone, however it is spelt, does not.
    assert policy.match(*workaround, "/w") is None
    alone = {"command": "pip3 install -q numpy"}
    assert policy.match("shell_exec", alone, "/w") == ("Auto mode", False)


def test_a_block_of_another_kind_keys_the_whole_command_only() -> None:
    policy = _blocked("shell_exec", {"command": "rm -rf build && make"}, ("delete",))
    assert policy.match("shell_exec", {"command": "make"}, "/w") is None
    spaced = {"command": "rm  -rf build  &&  make"}
    assert policy.match("shell_exec", spaced, "/w") == ("Auto mode", False)
