"""Auto mode, part 2 (#339): a chat's switch, its confinement, and a stopped turn's site question.

A chat's policy (:class:`ChatPolicy`) reads its mode at every step, so a
switch applies from the chat's next step while a call already waiting for
a person keeps waiting. A call auto mode decides runs under the write
sandbox even when the Agent confines nothing else; an Ask call does not.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import (
    STOPPED_NOTE,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
)
from localmelo.support.sessions.approvals import ChatPolicy
from localmelo.tests.agent.test_approvals import Recorder, _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_mode import (
    FakeModel,
    _agent,
    _approvals,
    verdict,
)

DELETE = "import shutil\nshutil.rmtree('.git', ignore_errors=True)"
ROUTINE = "print(sorted(__import__('os').listdir('.')))"


@pytest.mark.asyncio
async def test_a_switch_to_auto_applies_from_the_chat_s_next_step(
    tmp_path: Path,
) -> None:
    asked: list[str] = []
    policy: ChatPolicy

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.call_id)
        policy.mode = "auto"  # the person switches while this card waits
        return ApprovalDecision(True, origin="user")

    policy = ChatPolicy(ask)
    plans = [
        _calls(("python_exec", {"code": ROUTINE}, "call_1")),
        _calls(("python_exec", {"code": DELETE}, "call_2")),
        _answer("done"),
    ]
    agent, fake = _agent(FakeModel(plans), tmp_path / "ws")
    recorder = Recorder()

    await agent.run_task("tidy the notes", on_event=recorder, approval=policy)

    first, second = _approvals(recorder)
    # The card that waited is answered by the person; the next step is Auto's.
    assert asked == ["call_1"]
    assert (first["origin"], first["allowed"]) == ("user", True)
    assert (second["origin"], second["allowed"], second["mode"]) == (
        "auto",
        False,
        "auto",
    )
    assert fake.ran == [ROUTINE]


@pytest.mark.asyncio
async def test_a_switch_back_to_ask_asks_again_and_keeps_the_allowances(
    tmp_path: Path,
) -> None:
    asked: list[str] = []

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.call_id)
        return ApprovalDecision(False, "no", origin="user")

    policy = ChatPolicy(ask, "auto")
    policy.auto.allow_once("k", ["install"])
    policy.mode = "ask"
    agent, fake = _agent(
        FakeModel(_plans_of(DELETE)),
        tmp_path / "ws",
    )

    await agent.run_task("tidy the notes", approval=policy)

    assert asked == ["call_1"] and fake.ran == []
    assert "k" in policy.auto.given


def _plans_of(code: str) -> list:  # type: ignore[type-arg]
    return [_calls(("python_exec", {"code": code}, "call_1")), _answer("done")]


@pytest.mark.asyncio
async def test_a_waiting_card_stays_when_the_chat_switches(tmp_path: Path) -> None:
    waiting = asyncio.Event()
    answer: asyncio.Future[ApprovalDecision] = (
        asyncio.get_running_loop().create_future()
    )

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        waiting.set()
        return await answer

    policy = ChatPolicy(ask)
    agent, fake = _agent(FakeModel(_plans_of(DELETE)), tmp_path / "ws")
    run = asyncio.create_task(agent.run_task("delete .git", approval=policy))
    await waiting.wait()
    policy.mode = "auto"
    answer.set_result(ApprovalDecision(True, origin="user"))
    await run

    assert fake.ran == [DELETE]


# ── The browser's site question after a stop (gap d) ──


@pytest.mark.asyncio
async def test_the_site_question_of_a_stopped_turn_is_refused_without_a_review(
    tmp_path: Path,
) -> None:
    from localmelo.melo.agent._step_review import RunReviews
    from localmelo.melo.agent.agent import _RunState, _ToolAsk
    from localmelo.melo.schema import ToolCall

    llm = FakeModel([], [verdict()])
    agent, _ = _agent(llm, tmp_path / "ws")
    run = _RunState.__new__(_RunState)
    run.task_id = "t"
    run.reviews = RunReviews(stop=True)
    ask = _ToolAsk(
        AutoPolicy(),
        run,
        ToolCall(tool_name="open_site", arguments={}, id="call_9"),
        str(tmp_path / "ws"),
        agent._reviewer,
    )

    decision = await ask(
        "browser:https://news.example",
        {"site": "news.example", "origin": "https://news.example"},
        "open news.example",
    )

    assert (decision.allow, decision.origin, decision.reason) == (
        False,
        "auto",
        STOPPED_NOTE,
    )
    assert llm.review_requests == []


# ── Auto's calls run confined; Ask's do not ──


@pytest.mark.asyncio
async def test_only_a_call_auto_decides_is_confined(tmp_path: Path) -> None:
    seen: list[bool] = []
    agent, _ = _agent(
        FakeModel(_plans_of(ROUTINE) + _plans_of(ROUTINE)), tmp_path / "ws"
    )
    switch = agent._auto_confinement
    assert switch is not None and not switch.on

    async def python_exec(code: str) -> str:
        seen.append(switch.on)
        return "ok"

    agent.executor.register("python_exec", python_exec, requires_approval=True)

    async def allow(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, origin="user")

    policy = ChatPolicy(allow)
    await agent.run_task("list the files", approval=policy)
    policy.mode = "auto"
    await agent.run_task("list the files", approval=policy)

    assert seen == [False, True]


def test_an_agent_that_confines_its_runs_has_no_second_switch(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("writes are confined on macOS")
    (tmp_path / "ws").mkdir()
    agent = Agent(
        llm=FakeModel([]),
        options=AgentOptions(workspace_root=str(tmp_path / "ws"), confine_writes=True),
    )
    assert agent._auto_confinement is None


@pytest.mark.skipif(sys.platform != "darwin", reason="writes are confined on macOS")
@pytest.mark.asyncio
async def test_an_auto_call_cannot_write_outside_its_folder_even_unreviewed(
    tmp_path: Path,
) -> None:
    # Outside the user's temporary folder, which stays writable: this tree.
    outside = Path(__file__).resolve().parents[2] / ".cache"
    outside.mkdir(exist_ok=True)
    target = outside / f"t308-confine-{uuid.uuid4().hex}"
    code = "import os\nopen(os.environ['TARGET'], 'w').write('x')\nprint('wrote')"
    try:
        for mode, wrote in (("auto", False), ("ask", True)):

            async def allow(request: ApprovalRequest) -> ApprovalDecision:
                return ApprovalDecision(True, origin="user")

            # The path comes from the environment, so the rules cannot see it:
            # the review lets it run, and only the sandbox can stop it.
            llm = FakeModel(_plans_of(code), [verdict()])
            ws = tmp_path / f"ws-{mode}"
            ws.mkdir()
            agent = Agent(
                llm=llm,
                options=AgentOptions(
                    builtins=frozenset({"python_exec"}),
                    tool_exposure="all",
                    workspace_root=str(ws),
                    tool_env={"TARGET": str(target)},
                ),
            )
            agent.chat.sleep = _no_wait
            recorder = Recorder()
            await agent.run_task(
                "write the note",
                on_event=recorder,
                approval=ChatPolicy(allow, mode),  # type: ignore[arg-type]
            )
            [finished] = recorder.of("tool.finished")
            assert target.exists() is wrote, (mode, finished.data)
            if not wrote:
                assert "not allowed in auto mode" in str(finished.data)
            target.unlink(missing_ok=True)
            await agent.close()
    finally:
        target.unlink(missing_ok=True)
        if outside.exists() and not any(outside.iterdir()):
            os.rmdir(outside)
