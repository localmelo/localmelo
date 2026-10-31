"""Auto mode's contract (#339): the step key, the policy, the note the model reads."""

from __future__ import annotations

import dataclasses

import pytest

from localmelo.melo.contracts.approvals import (
    ALLOW_ONCE_S,
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    block_note,
    denial_text,
    find_auto,
    review_mode,
)
from localmelo.melo.contracts.review import ActionReview, ModelAnswer, step_key

INSTALL = ActionReview(
    "high",
    ("install", "env"),
    (
        "Installs matplotlib with pip into .venv.",
        "Creates a Python environment in .venv.",
    ),
)


def _request(
    code: str = "print(1)", review: ActionReview | None = None
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r1",
        run_id="run-1",
        call_id="call_1",
        tool_name="python_exec",
        arguments={"code": code},
        preview=code,
        workspace_root="/ws",
        review=review,
    )


def test_the_step_key_ignores_layout_and_comments_but_not_the_folder() -> None:
    a = step_key("python_exec", {"code": "import os\nos.listdir('.')"}, "/ws")
    assert a == step_key(
        "python_exec", {"code": "import os  # x\n\nos.listdir( '.' )"}, "/ws"
    )
    assert a != step_key("python_exec", {"code": "import os\nos.listdir('..')"}, "/ws")
    assert a != step_key(
        "python_exec", {"code": "import os\nos.listdir('.')"}, "/other"
    )
    shell = step_key("shell_exec", {"command": "pip  install x # why"}, "/ws")
    assert shell == step_key("shell_exec", {"command": ["pip", "install", "x"]}, "/ws")
    write = step_key("file_write", {"path": "/ws/a.txt", "content": "1"}, "/ws")
    assert write != step_key("file_write", {"path": "/ws/a.txt", "content": "2"}, "/ws")
    assert len(a) == 32


@pytest.mark.asyncio
async def test_auto_runs_routine_calls_and_blocks_the_rest() -> None:
    policy = AutoPolicy()
    assert await policy.decide(
        _request(review=ActionReview("routine"))
    ) == ApprovalDecision(True, origin="auto")
    assert await policy.decide(_request(review=None)) == ApprovalDecision(
        True, origin="auto"
    )
    assert await policy.decide(_request("pip", INSTALL)) == ApprovalDecision(
        False, "installs software", origin="auto"
    )


@pytest.mark.asyncio
async def test_an_allowance_runs_its_step_once_until_it_runs_out() -> None:
    now = [100.0]
    policy = AutoPolicy(clock=lambda: now[0])
    request = _request("pip", INSTALL)
    key = step_key("python_exec", {"code": "pip"}, "/ws")
    policy.allow_once(key)
    assert policy.has_allowances()
    assert await policy.decide(request) == ApprovalDecision(
        True, "allowed once", origin="user"
    )
    assert not policy.has_allowances()
    assert (await policy.decide(request)).allow is False
    policy.allow_once(key)
    now[0] += ALLOW_ONCE_S
    assert (await policy.decide(request)).allow is False


def test_the_review_mode_of_a_policy_and_of_its_wrappers() -> None:
    class Grants:
        def __init__(self, tools: set[str]) -> None:
            self.tools = tools

        def silent_for(self, request: ApprovalRequest) -> bool:
            return request.tool_name in self.tools

    class Wrapper:
        def __init__(self, inner: object) -> None:
            self.inner = inner

    auto = AutoPolicy()
    assert review_mode(Wrapper(auto), _request()) == "auto"
    assert find_auto(Wrapper(auto)) is auto
    assert review_mode(Grants({"python_exec"}), _request()) == "silent"
    assert review_mode(Wrapper(Grants(set())), _request()) == "ask"
    assert review_mode(AllowAllPolicy(), _request()) == "ask"
    assert find_auto(AllowAllPolicy()) is None


def test_the_note_tells_the_model_to_explain_and_not_to_retry() -> None:
    note = block_note(INSTALL)
    assert note.split("\n") == [
        "Auto mode blocked this step, so it did not run. It installs matplotlib with "
        "pip into .venv; creates a Python environment in .venv.",
        "In your answer, tell the user in plain words:",
        "- what you wanted to do, and why melo blocked it;",
        "- why the task needed it;",
        "- other ways to reach the goal: without this step, or with something the "
        "user can do;",
        "- that they can allow this step once from the chat, or say so in their next "
        "message.",
        "Do not reach the same result another way (another command, a script, another "
        "tool): every step gets the same review, and a step that does the same thing "
        "is blocked too. Go on with any work that does not need it.",
    ]


def test_the_note_s_variants() -> None:
    off_goal = dataclasses.replace(
        INSTALL, model=ModelAnswer("ask", "install", "no", "no", "Installs x.")
    )
    assert "in .venv. The user did not ask for it.\n" in block_note(off_goal)
    assert "from the task's page; the task goes on without it." in block_note(
        INSTALL, task=True
    )
    self_review = ActionReview("high", ("self",), ("Installs x into melo's Python.",))
    assert "this step cannot be allowed" in block_note(self_review, folder="demo")
    assert "Offer a way that works inside demo." in block_note(
        self_review, folder="demo"
    )
    unreviewed = ActionReview("high", ("unreviewed",), ("melo could not review it.",))
    assert "It could not be reviewed, so it was blocked." in block_note(unreviewed)
    stop = block_note(INSTALL, second=True)
    assert stop.startswith(
        "Auto mode blocked this step too: it installs matplotlib with pip into .venv;"
    )
    assert "Your next reply is your answer to the user" in stop
    assert block_note(INSTALL, changed=True).endswith(
        "Run exactly the step that was blocked, or explain."
    )


def test_a_denial_by_auto_mode_reads_its_note() -> None:
    note = block_note(INSTALL)
    assert denial_text(ApprovalDecision(False, note, origin="auto"), INSTALL) == note
    # A bare policy's short reason becomes the plain note.
    assert (
        denial_text(
            ApprovalDecision(False, "installs software", origin="auto"), INSTALL
        )
        == note
    )
