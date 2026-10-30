"""The action review in a run (#279): the gate, the self denial, grants.

Both cases of #279 run call by call through the agent's approval gate with
a fake approver that answers every question with "Allow for this chat".
``python_exec`` is a fake that records the code instead of running it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.review import SELF_DENIAL, ActionReview
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.agent.test_approvals import (
    ENDING,
    Policy,
    Recorder,
    ScriptedLLM,
    _answer,
    _calls,
    _no_wait,
)

PROBE = "import subprocess, sys\n"
ASK_INSTEAD = "If the task needs it, say in your answer what for and ask the user."
WITHOUT_IT = (
    "Finish what you can without it, and say in your answer what you would need."
)

# Case A: a question about a daily forum sign-in (sites made generic).
CASE_A = [
    ("A2", "import urllib.request\nurllib.request.urlopen('https://forum.example/')"),
    (
        "A5",
        PROBE
        + "subprocess.run(['curl', '-s', '-o', '/dev/null', 'https://forum.example/'])",
    ),
    ("A7", PROBE + "subprocess.run([sys.executable, '-m', 'pip', '--version'])"),
    (
        "A10",
        PROBE + "subprocess.run(['/usr/bin/python3', '-m', 'venv', '.signin_venv'])\n"
        "subprocess.run(['.signin_venv/bin/python', '-m', 'pip', 'install', '-U', 'pip'])",
    ),
    (
        "A11",
        PROBE
        + "subprocess.run(['.signin_venv/bin/python', '-m', 'pip', 'install', 'playwright'])",
    ),
    (
        "A12",
        PROBE
        + "subprocess.run(['.signin_venv/bin/python', '-m', 'playwright', 'install', 'chromium'])",
    ),
    (
        "A13",
        "import os\nprint(os.listdir(os.path.expanduser('~/Library/Caches/ms-playwright')))",
    ),
]
# Case B: a chart, two turns after a grant from a stock-price question.
CASE_B = [
    (
        "grant",
        "import urllib.request\nurllib.request.urlopen('https://quotes.example/q?s=X')",
    ),
    ("B1", "try:\n    import matplotlib\nexcept ImportError:\n    print('missing')"),
    (
        "B2",
        PROBE
        + "subprocess.run([sys.executable, '-m', 'pip', 'install', 'matplotlib'])",
    ),
    ("B5", PROBE + "subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'])"),
    (
        "B6",
        "import urllib.request\n"
        "data = urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py').read()\n"
        "open('get-pip.py', 'wb').write(data)",
    ),
    (
        "B7",
        PROBE + "subprocess.run([sys.executable, 'get-pip.py', '--user', '--quiet'])",
    ),
    (
        "B8",
        PROBE
        + "subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'matplotlib'])",
    ),
    ("B9", "import matplotlib.pyplot as plt\nplt.plot([1])\nplt.savefig('chart.png')"),
]


class FakePython:
    """``python_exec`` that records the code it would run."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def python_exec(self, code: str) -> str:
        self.ran.append(code)
        return "ok"


def _agent(llm: ScriptedLLM, root: Path) -> tuple[Agent, FakePython]:
    options = AgentOptions(
        builtins=frozenset({"python_exec"}),
        tool_exposure="all",
        workspace_root=str(root),
        steps_per_attempt=12,
    )
    agent = Agent(llm=llm, options=options)
    fake = FakePython()
    agent.executor.register("python_exec", fake.python_exec, requires_approval=True)
    agent.chat.sleep = _no_wait
    return agent, fake


def _script(calls: list[tuple[str, str]]) -> ScriptedLLM:
    replies = [_calls(("python_exec", {"code": code}, name)) for name, code in calls]
    return ScriptedLLM([*replies, _answer("done")])


class ChatApprover:
    """A person who answers every question with "Allow for this chat"."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision(True, remember=True)


async def _run(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> tuple[ChatApprover, RememberingPolicy, FakePython, ScriptedLLM]:
    (tmp_path / "ws").mkdir()
    llm = _script(calls)
    agent, fake = _agent(llm, tmp_path / "ws")
    approver = ChatApprover()
    policy = RememberingPolicy(approver)
    result = await agent.run_task("help me", approval=policy)
    assert result.status == "completed"
    return approver, policy, fake, llm


@pytest.mark.asyncio
async def test_case_a_asks_at_the_environment_install_and_browser_steps(
    tmp_path: Path,
) -> None:
    approver, policy, fake, _ = await _run(tmp_path, CASE_A)

    # A2 asks and is granted for the chat; the grant covers only routine calls.
    assert [r.call_id for r in approver.asked] == ["A2", "A10", "A11", "A12"]
    assert [r.review.categories for r in approver.asked[1:] if r.review] == [
        ("install", "env"),
        ("install",),
        ("download",),
    ]
    assert len(fake.ran) == len(CASE_A)  # the person allowed each one
    assert policy.allowed_tools == {"python_exec"}


@pytest.mark.asyncio
async def test_case_b_asks_at_each_install_and_never_lets_melos_python_change(
    tmp_path: Path,
) -> None:
    approver, _, fake, llm = await _run(tmp_path, CASE_B)

    assert [r.call_id for r in approver.asked] == ["grant", "B6", "B7", "B8"]
    ran = {name for name, code in CASE_B if code in fake.ran}
    assert ran == {"grant", "B1", "B6", "B7", "B8", "B9"}
    # B2 and B5 would change the Python that runs localmelo: denied unasked.
    # The model reads a refused call's text as an error.
    denials = [
        m for m in llm.requests[-1] if m.role == "tool" and "denied" in m.content
    ]
    assert [m.content for m in denials] == [
        f"Error: The approval policy denied this call: {SELF_DENIAL}. It installs matplotlib "
        f"into the Python that runs localmelo. {ENDING} {WITHOUT_IT}",
        f"Error: The approval policy denied this call: {SELF_DENIAL}. It installs pip into "
        f"the Python that runs localmelo. {ENDING} {WITHOUT_IT}",
    ]


@pytest.mark.asyncio
async def test_the_request_carries_the_review_and_a_denial_says_why(
    tmp_path: Path,
) -> None:
    code = (
        PROBE + "subprocess.run(['.venv/bin/python', '-m', 'pip', 'install', 'httpx'])"
    )
    llm = _script([("call_1", code)])
    agent, fake = _agent(llm, tmp_path)
    policy = Policy(ApprovalDecision(False, "not now"))
    recorder = Recorder()

    await agent.run_task("add httpx", on_event=recorder, approval=policy)

    [request] = policy.requests
    assert request.review == ActionReview(
        "high", ("install",), ("Installs httpx with pip into .venv.",)
    )
    assert fake.ran == []
    assert llm.tool_messages(1) == [
        "Error: The user denied this call: not now. melo asked because it installs httpx "
        f"with pip into .venv. {ENDING} {ASK_INSTEAD}"
    ]


@pytest.mark.asyncio
async def test_a_self_call_is_denied_without_asking_and_recorded(
    tmp_path: Path,
) -> None:
    code = f"import subprocess\nsubprocess.run([{sys.executable!r}, '-m', 'pip', 'install', 'x'])"
    llm = _script([("call_1", code)])
    agent, fake = _agent(llm, tmp_path)
    policy = Policy(ApprovalDecision(True))
    recorder = Recorder()

    await agent.run_task("install x", on_event=recorder, approval=policy)

    assert policy.requests == [] and fake.ran == []
    [finished] = recorder.of("tool.finished")
    approval = finished.data["approval"]
    assert (approval["allowed"], approval["origin"]) == (False, "policy")
    assert approval["reason"] == SELF_DENIAL


@pytest.mark.asyncio
async def test_a_routine_call_reads_the_plain_denial(tmp_path: Path) -> None:
    llm = _script([("call_1", "print(1)")])
    agent, _ = _agent(llm, tmp_path)
    policy = Policy(ApprovalDecision(False))

    await agent.run_task("print", approval=policy)

    assert policy.requests[0].review == ActionReview("routine")
    assert llm.tool_messages(1) == [f"Error: The user denied this call. {ENDING}"]
