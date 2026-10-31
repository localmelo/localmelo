"""How a task ended is decided by code from its own run, never from its report.

The run is a stand-in store (one turn, its tool events) over a real temporary
git repository: each test changes the repository the way the run's tool calls
say they did, then reads the ending :func:`decide` gives.
"""

from __future__ import annotations

import sqlite3
import subprocess
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.front_rules import asks_commit
from localmelo.support.gateway.task_endings import (
    Baseline,
    baseline,
    decide,
    proposal,
    python_test,
    report_text,
    shown,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore, StoredEvent

# A test run whose exit code is the call's own, and one whose wrapper hides it.
PASSES_ON = (
    "import subprocess, sys\n"
    "r = subprocess.run([sys.executable, '-m', 'pytest', '-q'])\n"
    "sys.exit(r.returncode)\n"
)
HIDES = (
    "import subprocess, sys\n"
    "r = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True)\n"
    "print(r.stdout)\n"
    "print('exit code', r.returncode)\n"
)
COMMIT = (
    "import subprocess\nsubprocess.run(['git', 'commit', '-am', 'Fix'], check=True)\n"
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    folder = tmp_path / "repo"
    folder.mkdir()
    git(folder, "init", "-q")
    (folder / "app.py").write_text("x = 1\n")
    (folder / "README.md").write_text("# App\n")
    git(folder, "add", ".")
    git(folder, "commit", "-qm", "Start")
    yield folder


class Run:
    """A stand-in store with one turn of a task: its answer and its tool events."""

    def __init__(self, answer: str = "Done.\nEnding: ready", run_id: str = "run-2"):
        self.answer = answer
        self.run_id = run_id
        self.events: list[StoredEvent] = []

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        status: str = "success",
        exit_code: int | None = None,
        call_id: str | None = None,
        run_id: str | None = None,
        finished: bool = True,
    ) -> Run:
        key = call_id or f"call-{len(self.events)}"
        run = run_id or self.run_id
        self._add(
            "tool.started", run, {"call_id": key, "name": name, "arguments": arguments}
        )
        if finished:
            data = {
                "call_id": key,
                "name": name,
                "status": status,
                "exit_code": exit_code,
            }
            self._add("tool.finished", run, data)
        return self

    def edit(self, path: str) -> Run:
        return self.call("file_edit", {"path": path, "old": "a", "new": "b"})

    def test(self, code: str = PASSES_ON, exit_code: int = 0) -> Run:
        status = "success" if exit_code == 0 else "error"
        return self.call(
            "python_exec", {"code": code}, status=status, exit_code=exit_code
        )

    def _add(self, kind: str, run_id: str, data: dict[str, Any]) -> None:
        event = {"type": kind, "run_id": run_id, "data": data}
        self.events.append(StoredEvent(len(self.events) + 1, "task", 2, event))

    async def list_turns(
        self, session_id: str, *, limit: int | None = None
    ) -> list[Any]:
        return [SimpleNamespace(turn=2, run_id=self.run_id, answer=self.answer)]

    async def list_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        turns: tuple[int, int] | None = None,
    ) -> list[StoredEvent]:
        found = [e for e in self.events if (e.id or 0) > after_id]
        return found[:limit]


async def end(
    run: Run,
    repo: Path | None,
    base: Baseline,
    *,
    phase: str = "done",
    reason: str = "",
    commit: bool = False,
) -> dict[str, Any]:
    ending = await decide(
        run,  # type: ignore[arg-type]
        "task",
        phase=phase,
        failure_reason=reason,
        folder=str(repo) if repo else None,
        base=base,
        asks_commit=commit,
    )
    assert ending is not None
    return ending


# ── Each downgrade ──


async def test_a_change_tested_after_it_and_committed_is_ready(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Fix")
    run = Run().edit("app.py").test().call("python_exec", {"code": COMMIT})
    ending = await end(run, repo, base, commit=True)
    assert ending["state"] == "ready"
    assert ending["reasons"] == []
    assert ending["proposed"] == "ready"
    assert [t["outcome"] for t in ending["tests"]] == ["passed"]
    assert "pytest" in ending["tests"][0]["command"]


async def test_the_last_test_failing_leaves_a_draft(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit("app.py").test(exit_code=0).test(exit_code=1)
    ending = await end(run, repo, base)
    assert ending["state"] == "draft"
    assert ending["reasons"] == ["tests_failed"]
    assert ending["tests"][-1] == {
        "command": ending["tests"][-1]["command"],
        "exit_code": 1,
        "outcome": "failed",
    }


async def test_no_test_after_a_code_change_leaves_a_draft(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    ending = await end(Run().edit("app.py"), repo, base)
    assert (ending["state"], ending["reasons"]) == ("draft", ["tests_missing"])


@pytest.mark.parametrize(
    "steps",
    [("test", "edit"), ("edit", "test", "edit")],
    ids=["pass-edit", "edit-pass-edit"],
)
async def test_a_test_before_the_latest_change_does_not_count(
    repo: Path, steps: tuple[str, ...]
) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    run = Run()
    for step in steps:
        run.edit("app.py") if step == "edit" else run.test()
    ending = await end(run, repo, base)
    assert ending["reasons"] == ["tests_missing"]


async def test_a_wrapper_that_exits_zero_proves_nothing(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    ending = await end(Run().edit("app.py").test(HIDES), repo, base)
    assert (ending["state"], ending["reasons"]) == ("draft", ["tests_unverified"])
    assert ending["tests"][0]["outcome"] == "unknown"


async def test_a_call_that_edits_and_tests_does_not_test_its_own_edit(
    repo: Path,
) -> None:
    base = await baseline(str(repo), None)
    both = "open('app.py', 'w').write('x = 3\\n')\n" + PASSES_ON
    (repo / "app.py").write_text("x = 3\n")
    ending = await end(Run().test(both), repo, base)
    assert ending["reasons"] == ["tests_missing"]


async def test_uncommitted_work_when_a_commit_was_asked_is_a_draft(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit("app.py").test()
    ending = await end(run, repo, base, commit=True)
    assert (ending["state"], ending["reasons"]) == ("draft", ["commit_missing"])
    assert ending["uncommitted"] == ["app.py"]
    # The same work, when no commit was asked, is ready.
    assert (await end(run, repo, base))["state"] == "ready"


async def test_a_question_in_the_report_makes_it_blocked(repo: Path) -> None:
    base = await baseline(str(repo), None)
    asked = await end(Run("Ending: blocked: Which port should it use?"), repo, base)
    assert (asked["state"], asked["reasons"]) == ("blocked", ["question"])
    assert asked["question"] == "Which port should it use?"
    prose = await end(Run("Added the flag.\nShould it be on by default?"), repo, base)
    assert prose["state"] == "blocked"
    assert prose["proposed"] is None
    assert prose["question"] == "Should it be on by default?"


# ── Plain endings ──


async def test_a_plain_ready_outside_git_and_without_a_proposal(tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    base = await baseline(str(folder), None)
    (folder / "poem.py").write_text("print('hi')\n")
    ending = await end(Run("Wrote it.").edit("poem.py"), folder, base)
    assert (ending["state"], ending["reasons"], ending["proposed"]) == (
        "ready",
        [],
        None,
    )


async def test_a_docs_only_change_needs_no_test(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "README.md").write_text("# App\nMore.\n")
    git(repo, "commit", "-qam", "Docs")
    run = Run().edit("README.md").call("python_exec", {"code": COMMIT})
    assert (await end(run, repo, base, commit=True))["state"] == "ready"


async def test_failed_and_dropped_proposals_stand(repo: Path) -> None:
    base = await baseline(str(repo), None)
    for said in ("failed", "dropped"):
        ending = await end(Run(f"Could not.\nEnding: {said}").test(), repo, base)
        assert ending["state"] == said


async def test_a_run_out_of_steps_is_not_finished(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    run = Run("I was about to run the tests.").edit("app.py")
    ending = await end(
        run, repo, base, phase="failed", reason=OUT_OF_STEPS, commit=True
    )
    assert ending["state"] == "draft"
    assert ending["reasons"] == ["out_of_steps", "tests_missing", "commit_missing"]


async def test_failed_and_cancelled_runs_take_the_state_default(repo: Path) -> None:
    base = await baseline(str(repo), None)
    for phase in ("failed", "cancelled"):
        ending = await decide(
            Run(),  # type: ignore[arg-type]
            "task",
            phase=phase,
            failure_reason="",
            folder=str(repo),
            base=base,
            asks_commit=False,
        )
        assert ending is None
    record = SimpleNamespace(
        state="cancelled", wait_reason="", finished_at=5.0, ending=None
    )
    assert shown(record) == {"state": "dropped", "reasons": ["stopped"]}  # type: ignore[arg-type]
    record.state = "failed"
    assert shown(record) == {"state": "failed", "reasons": ["run_failed"]}  # type: ignore[arg-type]
    record.wait_reason = OUT_OF_STEPS
    assert shown(record)["state"] == "draft"  # type: ignore[arg-type,index]


def test_an_ending_shows_only_for_the_transition_it_was_decided_at() -> None:
    ending = {"state": "draft", "reasons": ["tests_missing"], "at": 7.0}
    record = SimpleNamespace(
        state="done", wait_reason="", finished_at=7.0, ending=ending
    )
    assert shown(record) == ending  # type: ignore[arg-type]
    record.state = "queued"  # continued: no ending until it ends again
    assert shown(record) is None  # type: ignore[arg-type]
    record.state, record.finished_at = "failed", 9.0  # another path ended it
    assert shown(record) == {"state": "failed", "reasons": ["run_failed"]}  # type: ignore[arg-type]
    record.state = "done"  # done before endings were kept
    assert shown(record) is None  # type: ignore[arg-type]


# ── Claims and scope ──


async def test_tests_a_report_claims_are_ignored(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    claim = "Fixed it. Ran `pytest -q`: 12 passed.\nEnding: ready"
    run = (
        Run(claim)
        .edit("app.py")
        .call("python_exec", {"code": "print('12 passed (pytest)')"})
    )
    ending = await end(run, repo, base)
    assert (ending["state"], ending["reasons"], ending["tests"]) == (
        "draft",
        ["tests_missing"],
        [],
    )


async def test_tests_of_another_run_do_not_count(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit("app.py")
    run.call("python_exec", {"code": PASSES_ON}, exit_code=0, run_id="run-1")
    ending = await end(run, repo, base)
    assert (ending["reasons"], ending["tests"]) == (["tests_missing"], [])


async def test_a_denied_or_unfinished_test_is_not_a_pass(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    denied = (
        Run().edit("app.py").call("python_exec", {"code": PASSES_ON}, status="blocked")
    )
    assert (await end(denied, repo, base))["reasons"] == ["tests_missing"]
    cut = Run().edit("app.py").call("python_exec", {"code": PASSES_ON}, finished=False)
    assert (await end(cut, repo, base))["reasons"] == ["tests_unverified"]


async def test_repeated_call_ids_pair_in_order(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit("app.py")
    run.call(
        "python_exec", {"code": PASSES_ON}, exit_code=1, status="error", call_id="c0"
    )
    run.call("python_exec", {"code": PASSES_ON}, exit_code=0, call_id="c0")
    ending = await end(run, repo, base)
    assert [t["outcome"] for t in ending["tests"]] == ["failed", "passed"]
    assert ending["state"] == "ready"


async def test_work_that_was_there_before_the_run_is_not_blamed(repo: Path) -> None:
    (repo / "other.py").write_text("y = 1\n")  # the person's own uncommitted work
    base = await baseline(str(repo), None)
    (repo / "README.md").write_text("# App\nMore.\n")
    git(repo, "commit", "-qm", "Docs", "README.md")
    run = Run().edit("README.md").call("python_exec", {"code": COMMIT})
    ending = await end(run, repo, base, commit=True)
    assert (ending["state"], ending["uncommitted"]) == ("ready", [])


async def test_a_change_then_committed_still_needs_its_test(repo: Path) -> None:
    base = await baseline(str(repo), None)
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Fix")
    run = Run().edit("app.py").call("python_exec", {"code": COMMIT})
    ending = await end(run, repo, base, commit=True)
    assert (ending["state"], ending["reasons"]) == ("draft", ["tests_missing"])


async def test_a_continued_draft_still_owes_its_tests_and_commit(repo: Path) -> None:
    (repo / "app.py").write_text("x = 2\n")  # the earlier run's change
    before = {
        "state": "draft",
        "reasons": ["tests_missing", "commit_missing"],
        "uncommitted": ["app.py"],
    }
    base = await baseline(str(repo), before)
    only_commit = await end(Run(), repo, base, commit=True)
    assert only_commit["reasons"] == ["tests_missing", "commit_missing"]
    git(repo, "commit", "-qam", "Fix")
    finished = Run().test().call("python_exec", {"code": COMMIT})
    assert (await end(finished, repo, base, commit=True))["state"] == "ready"


# ── The report ──


def test_the_proposal_is_the_last_ending_line() -> None:
    assert proposal("Done.\nEnding: ready") == ("ready", "")
    assert proposal("**Ending:** `blocked` — need the key name") == (
        "blocked",
        "need the key name",
    )
    assert proposal("Ending: ready\nmore\nEnding: failed") == ("failed", "")
    assert proposal("All done, tests pass.") == (None, "")
    assert proposal("Ending: finished") == (None, "")


def test_the_person_reads_the_report_without_its_ending_line() -> None:
    assert report_text("Fixed the parser.\n\nEnding: ready\n") == "Fixed the parser."
    assert report_text("No line here.") == "No line here."


# ── What counts as a test whose result is known ──


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['pytest', '-q'], check=True)",
        "import subprocess\nsubprocess.check_call(['npm', 'test'])",
        "import subprocess, sys\nsys.exit(subprocess.call(['cargo', 'test']))",
        "import pytest\nraise SystemExit(pytest.main(['-q']))",
        "import subprocess\nr = subprocess.run(['pytest'])\nassert r.returncode == 0",
        "import subprocess, sys\nr = subprocess.run(['pytest'])\nif r.returncode:\n    sys.exit(1)",
        "import subprocess\nsubprocess.run(['make', 'test']).check_returncode()",
        "import subprocess, sys\nrc = subprocess.run(['go', 'test', './...']).returncode\nsys.exit(rc)",
    ],
)
def test_a_test_that_passes_its_exit_code_on(code: str) -> None:
    command, passes = python_test(code)
    assert command and passes


@pytest.mark.parametrize(
    "code,is_test",
    [
        (HIDES, True),
        ("import subprocess\nsubprocess.run(['pytest'])\nprint('ok')", True),
        ("print('ran pytest: 3 passed')", False),
        ("import subprocess\nsubprocess.run(['git', 'status'], check=True)", False),
        ("def broken(:\n  pytest", False),
    ],
)
def test_a_result_that_is_not_passed_on_is_unknown(code: str, is_test: bool) -> None:
    command, passes = python_test(textwrap.dedent(code))
    assert bool(command) is is_test
    assert not passes


# ── Where the ending is kept ──


async def test_the_store_keeps_an_ending_with_its_transition(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        record, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "Fix it"}),
            workspace=str(tmp_path),
            origin="user",
        )
        await tasks.change(record.task_id, state="running", actor="scheduler")
        with pytest.raises(ValueError):
            await tasks.change(
                record.task_id, state="queued", ending={"state": "ready"}
            )
        draft = {"state": "draft", "reasons": ["tests_missing"]}
        done = await tasks.change(record.task_id, state="done", ending=draft)
        assert done.ending == {**draft, "at": done.finished_at}
        assert shown(done) == done.ending
        again = await tasks.reopen(record.task_id, "Finish it: run the tests")
        # Kept for the next run to read what is owed; shown only once it ends again.
        assert again.ending == done.ending and shown(again) is None
    finally:
        await store.aclose()


async def test_an_older_store_gains_the_ending_column(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    record, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "Old"}),
        workspace=str(tmp_path),
        origin="user",
    )
    await store.aclose()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE tasks DROP COLUMN ending")
    store = SessionStore(path, writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        assert (await tasks.get_task(record.task_id)).ending is None
    finally:
        await store.aclose()


# ── What asks for a commit ──


@pytest.mark.parametrize(
    "text,asks",
    [
        ("Fix add in calc.py, run the tests and commit", True),
        ("修复 calc.py 然后提交", True),
        ("push it when done", True),
        ("Fix add in calc.py; don't commit yet", False),
        ("修复 calc.py，先不要提交", False),
        ("The log says `git commit` failed; explain why", False),
        ("Look at the committed history", False),
    ],
)
def test_a_commit_is_asked_only_in_plain_words(text: str, asks: bool) -> None:
    assert asks_commit(text) is asks
