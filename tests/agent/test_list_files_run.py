"""A chat's first look at the folder needs no approval; direct mode never has it.

Runs are scripted (no model). The interactive and gateway surfaces offer
``list_files``; its tier is ``read``, so the approval policy of either one is
never asked. Direct mode and the library's default keep their three built-ins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.support.cli.approvals import Approvals
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.agent.test_approvals import ScriptedLLM, _answer, _calls


class Panel:
    """An ask function that records every request (none is expected)."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision(False)


def _agent(llm: ScriptedLLM, root: Path, mode: str) -> Agent:
    (root / "src").mkdir()
    (root / "README.md").write_text("# Demo\n")
    options = AgentOptions(mode=mode, workspace_root=str(root))  # type: ignore[arg-type]
    return Agent(llm=llm, options=options)


@pytest.mark.parametrize("mode", ["interactive", "gateway"])
async def test_a_chat_lists_the_folder_without_being_asked(
    tmp_path: Path, mode: str
) -> None:
    llm = ScriptedLLM([_calls(("list_files", {}, "c1")), _answer("A small project.")])
    panel = Panel()
    # The terminal wraps its panel in Approvals; the console gives its
    # broker's ask function to the same RememberingPolicy.
    policy: Any = (
        Approvals(panel, "ask").policy()
        if mode == "interactive"
        else RememberingPolicy(panel)
    )

    result = await _agent(llm, tmp_path, mode).run_task(
        "What is this?", approval=policy
    )

    assert result.status == "completed"
    assert panel.asked == []
    assert "list_files" in llm.offered[0]
    assert llm.tool_messages(1) == ["src/\nREADME.md  7 B"]


@pytest.mark.parametrize("mode", ["direct", "python"])
async def test_direct_mode_and_the_library_default_never_offer_it(
    tmp_path: Path, mode: str
) -> None:
    llm = ScriptedLLM([_calls(("list_files", {}, "c1")), _answer("done")])

    await _agent(llm, tmp_path, mode).run_task("What is this?")

    assert llm.offered[0] == ["file_read", "file_write", "python_exec"]
    assert "list_files is not available in this run" in llm.tool_messages(1)[0]
