"""A task's aids: two steps kept for finishing, and the environment's refusal named.

Runs are scripted (no model). With ``task_aids``, the planning request two
steps before the end of the run's last attempt carries ``FINISH_NOTE``, so a
model that reads it commits in the steps that are left; and a tool's text
whose errors share one cause the environment gives starts with a line that
names it. Without the option (direct runs, the default) neither happens.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._task_aids import FINISH_NOTE, refusal_note
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef

COMMIT = (
    "import subprocess\n"
    "for argv in (['git', 'add', '-A'], ['git', '-c', 'user.name=t', '-c',"
    " 'user.email=t@example.com', 'commit', '-q', '-m', 'Finish']):\n"
    "    subprocess.run(argv, check=True)\n"
    "print('committed')"
)


class Finisher(BaseLLMProvider):
    """Reads a file on every step until told two steps are left, then commits."""

    def __init__(self) -> None:
        self.requests: list[list[Message]] = []
        self.committed = False

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        n = len(self.requests)
        if self.committed:
            return Message(role="assistant", content="Done and committed.")
        if any(m.role == "user" and m.content == FINISH_NOTE for m in messages):
            self.committed = True
            call = ToolCall(
                tool_name="python_exec", arguments={"code": COMMIT}, id=f"c{n}"
            )
        else:
            # A new file each step, so the run is never stuck.
            path = f"f{n}.txt"
            call = ToolCall(tool_name="file_read", arguments={"path": path}, id=f"c{n}")
        return Message(role="assistant", content="", tool_calls=[call])

    async def close(self) -> None:
        pass


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for n in range(1, 6):
        (root / f"f{n}.txt").write_text(f"note {n}\n")
    return root


def _log(root: Path) -> str:
    return subprocess.run(
        ["git", "log", "--oneline"], cwd=root, capture_output=True, text=True
    ).stdout


async def _run(root: Path, *, aids: bool) -> tuple[Finisher, str]:
    llm = Finisher()
    options = AgentOptions(
        workspace_root=str(root),
        attempt_policy="none",
        max_steps=5,
        task_aids=aids,
        memory=False,
    )
    result = await Agent(llm=llm, options=options).run_task("Fix it and commit.")
    return llm, result.status


async def test_the_commit_happens_in_the_kept_steps(repo: Path) -> None:
    llm, status = await _run(repo, aids=True)
    assert status == "completed"
    # Steps 1-3 read; the request of step 4 says two are left; it commits.
    noted = [
        i
        for i, r in enumerate(llm.requests)
        if any(m.content == FINISH_NOTE for m in r)
    ]
    assert noted and noted[0] == 3
    assert "Finish" in _log(repo)


async def test_without_the_aids_no_request_says_so(repo: Path) -> None:
    llm, status = await _run(repo, aids=False)
    assert status == "budget_exhausted"
    assert not any(m.content == FINISH_NOTE for r in llm.requests for m in r)
    assert _log(repo) == ""


PYTEST_REFUSED = "\n".join(
    [
        f"ERROR tests/test_{i}.py::test_x - RuntimeError: refusing the real HOME '/u/{i}'"
        for i in range(8)
    ]
    + ["FAILED tests/test_z.py::test_y - AssertionError: 1 != 2", "8 errors, 1 failed"]
)


def test_one_environmental_cause_of_most_errors_is_named() -> None:
    note = refusal_note(PYTEST_REFUSED)
    assert note.startswith("The test environment refused: RuntimeError: refusing")
    assert "(8 of 9 errors have this one cause" in note


@pytest.mark.parametrize(
    "text",
    [
        # Too few errors.
        "E   PermissionError: [Errno 1] Operation not permitted",
        # Many errors of the code itself, not of the environment.
        "\n".join(f"E   AssertionError: {i} != 0" for i in range(9)),
        "\n".join("E   AssertionError: same" for _ in range(9)),
        # No one cause has most of them.
        "\n".join(f"E   PermissionError: denied {i}" for i in range(4))
        + "\n"
        + "\n".join(f"E   ValueError: bad {i}" for i in range(4)),
        "all 120 tests passed",
    ],
)
def test_no_note_without_one_environmental_cause(text: str) -> None:
    assert refusal_note(text) == ""


class Tester(BaseLLMProvider):
    """Runs a tool whose output is a refused test run, then answers."""

    def __init__(self) -> None:
        self.tool_texts: list[str] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        tool = [m.content for m in messages if m.role == "tool"]
        if tool:
            self.tool_texts = tool
            return Message(role="assistant", content="The environment refused.")
        code = f"print({PYTEST_REFUSED!r})"
        call = ToolCall(tool_name="python_exec", arguments={"code": code}, id="c1")
        return Message(role="assistant", content="", tool_calls=[call])

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("aids", [True, False])
async def test_the_tool_text_starts_with_the_note_only_with_the_aids(
    tmp_path: Path, aids: bool
) -> None:
    llm = Tester()
    options = AgentOptions(workspace_root=str(tmp_path), task_aids=aids, memory=False)
    await Agent(llm=llm, options=options).run_task("Run the tests.")
    [text] = llm.tool_texts
    assert text.startswith("The test environment refused:") is aids
    assert "ERROR tests/test_0.py::test_x" in text
