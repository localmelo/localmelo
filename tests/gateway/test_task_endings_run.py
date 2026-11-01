"""A real worker on a scripted model ends a task in a temporary git repository.

The model edits ``calc.py``, runs (or only claims to run) its test, commits
or not, and ends its report with its proposal. The task's ending is what code
decides from the run's own tool calls; a draft finishes with one tap.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.task_endings import Baseline
from localmelo.support.gateway.task_store import TaskStoreError
from localmelo.support.gateway.workers import TaskInbox, TaskWorkers
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore

Reply = Callable[[], Message]
IDENTITY = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
TEST = (
    "import subprocess, sys\n"
    "sys.exit(subprocess.run([sys.executable, '-m', 'unittest', '-q']).returncode)\n"
)
COMMIT = (
    "import subprocess\n"
    "subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', "
    "'-c', 'commit.gpgsign=false', "
    "'commit', '-qam', 'Fix add'], check=True)\n"
)


def call(name: str, arguments: dict[str, Any]) -> Reply:
    def reply() -> Message:
        tool = ToolCall(
            tool_name=name, arguments=arguments, id=f"{name}-{id(arguments)}"
        )
        return Message(role="assistant", content="", tool_calls=[tool])

    return reply


def says(text: str) -> Reply:
    return lambda: Message(role="assistant", content=text)


EDIT = call("file_edit", {"path": "calc.py", "old": "a - b", "new": "a + b"})
RUN_TEST = call("python_exec", {"code": TEST})
RUN_COMMIT = call("python_exec", {"code": COMMIT})


class Scripted(BaseLLMProvider):
    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        return self.replies.pop(0)() if self.replies else says("Done.\nEnding: ready")()

    async def close(self) -> None:
        pass


Setup = tuple[TaskWorkers, Scripted, Path]


@pytest.fixture
async def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Setup]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", *IDENTITY, "-c", "commit.gpgsign=false", "commit", "-qm", "Start"],
        cwd=repo,
        check=True,
    )
    cfg = Config()
    cfg.gateway.workspace = str(repo)
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    llm = Scripted()

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](llm),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        yield workers, llm, repo
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def ended(workers: TaskWorkers, task_id: str, turns: int = 1) -> dict[str, Any]:
    """Allow every step the task asks for until its *turns*-th run has ended."""

    async def finished() -> bool:
        for request in workers.broker.pending(task_id):
            workers.broker.answer(request.request_id, ApprovalDecision(True))
        record = await workers.tasks.get_task(task_id)
        count = len(await workers.store.list_turns(task_id, limit=10))
        return (
            record.state == "done" and count == turns and task_id not in workers._runs
        )

    await until(finished)
    return await workers.get(task_id)


async def until(check: Callable[[], Awaitable[bool]]) -> None:
    async with asyncio.timeout(30):
        while not await check():
            await asyncio.sleep(0.01)


async def start(workers: TaskWorkers, repo: Path, objective: str) -> str:
    row, _ = await workers.create({"objective": objective, "workspace": str(repo)})
    return str(row["task_id"])


async def test_a_fix_tested_and_committed_is_ready(setup: Setup) -> None:
    workers, llm, repo = setup
    llm.replies = [EDIT, RUN_TEST, RUN_COMMIT, says("Fixed add.\nEnding: ready")]
    task = await ended(
        workers, await start(workers, repo, "Fix add in calc.py and commit")
    )
    ending = task["ending"]
    assert (ending["state"], ending["reasons"]) == ("ready", [])
    assert [t["outcome"] for t in ending["tests"]] == ["passed"]
    assert ending["tests"][0]["exit_code"] == 0
    assert "unittest" in ending["tests"][0]["command"]


async def test_a_claimed_test_that_never_ran_leaves_a_draft_that_finishes(
    setup: Setup,
) -> None:
    workers, llm, repo = setup
    claim = "Fixed add. Ran python -m unittest: OK (1 test).\nEnding: ready"
    llm.replies = [EDIT, says(claim)]
    task_id = await start(workers, repo, "Fix add in calc.py and commit")
    task = await ended(workers, task_id)
    ending = task["ending"]
    assert ending["state"] == "draft"
    assert ending["reasons"] == ["tests_missing", "commit_missing"]
    assert ending["tests"] == [] and ending["uncommitted"] == ["calc.py"]
    # One tap: the person's Finish it, through the continue path.
    llm.replies = [RUN_TEST, RUN_COMMIT, says("Tested and committed.\nEnding: ready")]
    body, how = await workers.continue_stopped(task_id)
    assert (how, body["state"], body["ending"]) == ("continued", "queued", None)
    finished = await ended(workers, task_id, turns=2)
    assert finished["ending"]["state"] == "ready"
    turns = await workers.store.list_turns(task_id, limit=10)
    assert turns[-1].user_text.startswith("Finish it: run the tests")
    with pytest.raises(TaskStoreError):
        await workers.continue_stopped(task_id)


async def test_a_failing_test_and_a_question(setup: Setup) -> None:
    workers, llm, repo = setup
    (repo / "test_calc.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_no(self):\n        self.fail('no')\n"
    )
    subprocess.run(
        ["git", *IDENTITY, "-c", "commit.gpgsign=false", "commit", "-qam", "Red"],
        cwd=repo,
        check=True,
    )
    llm.replies = [EDIT, RUN_TEST, says("The test still fails.\nEnding: ready")]
    task = await ended(workers, await start(workers, repo, "Fix add in calc.py"))
    assert (task["ending"]["state"], task["ending"]["reasons"]) == (
        "draft",
        ["tests_failed"],
    )
    assert task["ending"]["tests"][0]["exit_code"] == 1
    llm.replies = [says("Ending: blocked: Should add round floats?")]
    asked = await ended(workers, await start(workers, repo, "Look at calc.py"))
    assert asked["ending"]["state"] == "blocked"
    assert asked["ending"]["question"] == "Should add round floats?"


async def test_a_stopped_task_is_dropped(setup: Setup) -> None:
    workers, llm, repo = setup
    held = asyncio.Event()

    async def slow(
        messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        held.set()
        await asyncio.sleep(30)
        return says("never")()

    llm.chat = slow  # type: ignore[method-assign]
    task_id = await start(workers, repo, "Fix add in calc.py")
    await asyncio.wait_for(held.wait(), 10)
    await workers.update(task_id, state="stop")

    async def stopped() -> bool:
        return (await workers.tasks.get_task(task_id)).state == "cancelled"

    await until(stopped)
    assert (await workers.get(task_id))["ending"] == {
        "state": "dropped",
        "reasons": ["stopped"],
    }


# A failing test whose wrapper swallows the failure with try/except around check=True.
SWALLOWED = (
    "import subprocess, sys\n"
    "try:\n"
    "    subprocess.run([sys.executable, '-m', 'unittest', '-q'], check=True)\n"
    "except subprocess.CalledProcessError as e:\n"
    "    print('FAILED', e.returncode)\n"
)
COMMIT_NAMING_TESTS = (
    "import subprocess\n"
    "subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', "
    "'-c', 'commit.gpgsign=false', "
    "'commit', '-qam', 'Fix add; unittest still red'], check=True)\n"
)
BAD_EDIT = call("file_edit", {"path": "calc.py", "old": "a - b", "new": "a * b"})


async def test_a_swallowed_failure_is_not_ready(setup: Setup) -> None:
    workers, llm, repo = setup
    llm.replies = [
        BAD_EDIT,
        call("python_exec", {"code": SWALLOWED}),
        says("Done.\nEnding: ready"),
    ]
    task = await ended(workers, await start(workers, repo, "Fix add in calc.py"))
    assert (task["ending"]["state"], task["ending"]["reasons"]) == (
        "draft",
        ["tests_unverified"],
    )


async def test_a_commit_message_naming_the_tests_is_not_a_passing_test(
    setup: Setup,
) -> None:
    workers, llm, repo = setup
    llm.replies = [
        BAD_EDIT,
        RUN_TEST,
        call("python_exec", {"code": COMMIT_NAMING_TESTS}),
        says("Committed.\nEnding: ready"),
    ]
    task = await ended(
        workers, await start(workers, repo, "Fix add in calc.py and commit")
    )
    assert (task["ending"]["state"], task["ending"]["reasons"]) == (
        "draft",
        ["tests_failed"],
    )


async def test_a_message_the_run_read_counts_as_asked(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, _, repo = setup
    record = await workers.tasks.get_task(
        await start(workers, repo, "Fix add in calc.py")
    )
    seen: list[bool] = []

    async def decide(*args: Any, asks_commit: bool, **kwargs: Any) -> None:
        seen.append(asks_commit)

    monkeypatch.setattr("localmelo.support.gateway.workers.decide_ending", decide)
    inbox = TaskInbox()
    inbox.put("then commit it, please")
    inbox.accept(inbox.poll()[0][0])
    assert inbox.read == ["then commit it, please"]
    workers._inboxes[record.task_id] = inbox
    base = Baseline(None)
    await workers._ending(record.task_id, "done", None, str(repo), base, record)
    assert seen == [True]


README = call("file_write", {"path": "README.md", "content": "# Calc\n"})


@pytest.mark.parametrize(
    ("edit", "objective", "owed"),
    [
        (EDIT, "Fix add in calc.py", (["tests_missing"], ["calc.py"])),
        (README, "Write a README and commit it", (["commit_missing"], ["README.md"])),
    ],
    ids=["code", "docs"],
)
async def test_work_a_stopped_run_left_is_owed_by_the_run_that_continues_it(
    setup: Setup, edit: Reply, objective: str, owed: tuple[list[str], list[str]]
) -> None:
    workers, llm, repo = setup
    held = asyncio.Event()
    script = [edit]

    async def edit_then_hang(
        messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if script:
            return script.pop(0)()
        held.set()
        await asyncio.sleep(30)
        return says("never")()

    plain = llm.chat
    llm.chat = edit_then_hang  # type: ignore[method-assign]
    task_id = await start(workers, repo, objective)

    async def editing() -> bool:
        for request in workers.broker.pending(task_id):
            workers.broker.answer(request.request_id, ApprovalDecision(True))
        return held.is_set()

    await until(editing)
    await workers.update(task_id, state="stop")

    async def stopped() -> bool:
        return (await workers.tasks.get_task(task_id)).state == "cancelled"

    await until(stopped)
    llm.chat = plain  # type: ignore[method-assign]
    llm.replies = [says("Nothing left to do.\nEnding: ready")]
    _, how = await workers.message(task_id, "Go on")
    assert how == "continued"
    ending = (await ended(workers, task_id, turns=2))["ending"]
    assert (ending["state"], ending["reasons"], ending["uncommitted"]) == (
        "draft",
        *owed,
    )


@pytest.mark.parametrize("how", ["unreadable", "raises"])
async def test_a_folder_unreadable_at_the_start_is_not_kept(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    workers, llm, repo = setup
    from localmelo.support.gateway import task_endings

    calls: list[str | None] = []

    async def flaky(folder: str | None) -> Baseline:
        calls.append(folder)
        if len(calls) == 1:  # the first run's start only
            if how == "raises":
                raise OSError("disk went away")
            return Baseline(None, git=True, error=True)
        return await task_endings.baseline(folder)

    monkeypatch.setattr("localmelo.support.gateway.workers.baseline", flaky)
    llm.replies = [says("Looked.\nEnding: ready")]
    task_id = await start(workers, repo, "Look at calc.py")
    first = (await ended(workers, task_id))["ending"]
    assert (first["state"], first["reasons"]) == ("draft", ["unchecked"])
    # The next run reads the folder again: nothing changed, so it is ready.
    llm.replies = [says("Looked again.\nEnding: ready")]
    await workers.continue_stopped(task_id)
    second = (await ended(workers, task_id, turns=2))["ending"]
    assert (second["state"], second["reasons"]) == ("ready", [])


STASH = call(
    "python_exec", {"code": "import subprocess\nsubprocess.run(['git', 'stash'])\n"}
)


async def test_work_a_run_stashed_stays_unchecked_in_the_next_run(
    setup: Setup,
) -> None:
    workers, llm, repo = setup
    llm.replies = [EDIT, STASH, says("Fixed add.\nEnding: ready")]
    task_id = await start(workers, repo, "Fix add in calc.py")
    first = (await ended(workers, task_id))["ending"]
    assert (first["state"], first["reasons"]) == ("draft", ["unchecked"])
    # The next run fixes and passes its test, but the first run's work is still stashed.
    llm.replies = [EDIT, RUN_TEST, says("Tested.\nEnding: ready")]
    await workers.continue_stopped(task_id)
    second = (await ended(workers, task_id, turns=2))["ending"]
    assert (second["state"], second["reasons"]) == ("draft", ["unchecked"])


@pytest.mark.parametrize("how", ["stop", "shutdown"])
async def test_a_stop_while_the_ending_is_read_leaves_the_run_as_it_ended(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    workers, llm, repo = setup
    from localmelo.support.gateway import workers as module

    reading, go_on = asyncio.Event(), asyncio.Event()
    real = module.decide_ending

    async def slow(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        reading.set()
        await go_on.wait()
        return await real(*args, **kwargs)

    monkeypatch.setattr(module, "decide_ending", slow)
    llm.replies = [says("Looked.\nEnding: ready")]
    task_id = await start(workers, repo, "Look at calc.py")
    await asyncio.wait_for(reading.wait(), 10)
    # The finished turn's runner stays registered until the run is finalized;
    # cancelling it stops nothing, and shutdown waits for the ending.
    assert task_id in workers._runners
    if how == "stop":
        await workers.update(task_id, state="stop")
        go_on.set()
    else:
        closing = asyncio.create_task(workers.aclose())
        await asyncio.sleep(0.05)
        assert not closing.done()
        go_on.set()
        await asyncio.wait_for(closing, 10)

    async def finished() -> bool:
        return task_id not in workers._runs

    await until(finished)
    record = await workers.tasks.get_task(task_id)
    assert record.state == "done"
    assert record.ending is not None and record.ending["state"] == "ready"
