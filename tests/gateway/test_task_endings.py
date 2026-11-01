"""How a task ended is decided by code from its own run, never from its report.

The run is a stand-in store (one turn, its tool events) over a real temporary
git repository: each test changes the repository the way the run's tool calls
say they did, then reads the ending :func:`decide` gives. The rule is fail
closed: ready needs positive proof, any doubt leaves a draft with its reason.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.support.gateway import task_endings
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.front_rules import asks_commit, asks_tests
from localmelo.support.gateway.task_endings import (
    MAX_STAMPS,
    Baseline,
    _test_marker,
    baseline,
    can_finish,
    decide,
    finish_message,
    finishes,
    proposal,
    python_test,
    report_text,
    shown,
    stash_floor,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore, StoredEvent

# The one shape of a test whose exit code is the call's own; a wrapper that hides it.
PASSES_ON = (
    "import subprocess, sys\n"
    "sys.exit(subprocess.run([sys.executable, '-m', 'pytest', '-q']).returncode)\n"
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
# Wrappers that exit 0 whatever the nested test runner said.
SWALLOW = {
    "try_except_check_true": (
        "import subprocess, sys\n"
        "try:\n    subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)\n"
        "except subprocess.CalledProcessError as e:\n    print('tests failed', e.returncode)\n"
    ),
    "check_output_caught": (
        "import subprocess\n"
        "try:\n    print(subprocess.check_output(['pytest', '-q'], text=True))\n"
        "except subprocess.CalledProcessError as e:\n    print(e.output)\n"
    ),
    "pipe_tail_passed_on": (
        "import subprocess, sys\n"
        "sys.exit(subprocess.run('python -m pytest -q 2>&1 | tail -20', shell=True)"
        ".returncode)\n"
    ),
    "or_true_check": (
        "import subprocess\nsubprocess.run('pytest -q || true', shell=True, check=True)\n"
    ),
    "unrelated_check_call": (
        "import subprocess\nsubprocess.check_call(['git', 'status'])\n"
        "r = subprocess.run(['pytest', '-q'])\nprint(r.returncode)\n"
    ),
    "unrelated_if_raise": (
        "import subprocess\ng = subprocess.run(['git', 'diff', '--quiet'])\n"
        "if g.returncode > 1:\n    raise SystemExit(2)\n"
        "r = subprocess.run(['pytest', '-q'])\nprint(r.returncode)\n"
    ),
    "assert_in": (
        "import subprocess\nr = subprocess.run(['pytest', '-q'])\n"
        "assert r.returncode in (0, 1)\n"
    ),
    "output_text": (
        "import subprocess, sys\n"
        "r = subprocess.run(['pytest', '-q'], capture_output=True, text=True)\n"
        "sys.exit(0 if 'passed' in r.stdout else 1)\n"
    ),
    "popen_background": (
        "import subprocess, sys\np = subprocess.Popen(['pytest', '-q'])\n"
        "sys.exit(p.returncode)\n"
    ),
    "in_process": "import pytest, sys\nsys.exit(pytest.main(['-q']))\n",
    "renamed_import": (
        "import subprocess as sys, sys as subprocess\n"
        "sys.exit(subprocess.run(['pytest', '-q']).returncode)\n"
    ),
    "other_folder": (
        "import subprocess\nsubprocess.run(['pytest', '/somewhere/else/tests'], check=True)\n"
    ),
    "other_cwd": (
        "import subprocess\nsubprocess.run(['pytest', '-q'], check=True, cwd='/elsewhere')\n"
    ),
}
# Calls that name a test runner but run no tests.
NOT_TESTS = {
    "commit_message": (
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-am', 'Fix the flaky pytest fixture'], check=True)\n"
    ),
    "pytest_version": (
        "import subprocess\n"
        "print(subprocess.check_output(['python', '-m', 'pytest', '--version']))\n"
    ),
    "cat_ini": "import subprocess\nsubprocess.run(['cat', 'pytest.ini'], check=True)\n",
    "collect_only": (
        "import subprocess\nsubprocess.run(['pytest', '--collect-only', '-q'], check=True)\n"
    ),
    "pip_install": (
        "import subprocess\nsubprocess.run(['pip', 'install', 'pytest'], check=True)\n"
    ),
    "printed": "print('ran pytest: 3 passed')\n",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
        + ["-c", "commit.gpgsign=false", *args],
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
    (folder / "CMakeLists.txt").write_text("project(x)\n")
    (folder / "tests").mkdir()
    (folder / "tests" / "test_app.py").write_text("def test_x():\n    pass\n")
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
        started = {"call_id": key, "name": name, "arguments": arguments}
        self._add("tool.started", run, started)
        if finished:
            done = {"call_id": key, "name": name, "status": status}
            self._add("tool.finished", run, {**done, "exit_code": exit_code})
        return self

    def edit(self, path: str = "app.py") -> Run:
        return self.call("file_edit", {"path": path, "old": "a", "new": "b"})

    def test(self, code: str = PASSES_ON, exit_code: int = 0) -> Run:
        return self.py(code, exit_code)

    def py(self, code: str, exit_code: int = 0) -> Run:
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
    tests: bool = False,
    finishing: bool = False,
) -> dict[str, Any]:
    ending = await decide(
        run,  # type: ignore[arg-type]
        "task",
        phase=phase,
        failure_reason=reason,
        folder=str(repo) if repo else None,
        base=base,
        asks_commit=commit,
        asks_tests=tests,
        finishing=finishing,
    )
    assert ending is not None
    return ending


def why(ending: dict[str, Any]) -> tuple[str, list[str]]:
    return ending["state"], ending["reasons"]


# ── Each downgrade ──


async def test_a_change_tested_after_it_and_committed_is_ready(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Fix")
    run = Run().edit().test().py(COMMIT)
    ending = await end(run, repo, base, commit=True)
    assert why(ending) == ("ready", [])
    assert ending["proposed"] == "ready"
    assert [t["outcome"] for t in ending["tests"]] == ["passed"]
    assert "pytest" in ending["tests"][0]["command"]


async def test_the_last_test_failing_leaves_a_draft(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    ending = await end(Run().edit().test().test(exit_code=1), repo, base)
    assert why(ending) == ("draft", ["tests_failed"])
    assert ending["tests"][-1]["exit_code"] == 1
    assert ending["tests"][-1]["outcome"] == "failed"


async def test_a_failing_last_test_is_never_ready_even_without_a_change(
    repo: Path, tmp_path: Path
) -> None:
    base = await baseline(str(repo))
    run = Run("test_x fails: the fixture uses UTC.\nEnding: ready").test(exit_code=1)
    still = await end(run, repo, base)
    assert why(still) == ("draft", ["tests_still_fail"])
    assert not can_finish(still), "nothing was changed: there is nothing to finish"
    # Outside git the rules of the test calls still apply.
    plain = tmp_path / "plain"
    plain.mkdir()
    outside = await baseline(str(plain))
    (plain / "x.py").write_text("x = 1\n")
    failing = Run().edit("x.py").test(exit_code=1)
    assert why(await end(failing, plain, outside)) == ("draft", ["tests_failed"])
    unknown = Run().test(HIDES)
    assert why(await end(unknown, plain, outside)) == ("draft", ["tests_unverified"])


async def test_no_test_after_a_code_change_leaves_a_draft(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    assert why(await end(Run().edit(), repo, base)) == ("draft", ["tests_missing"])


@pytest.mark.parametrize(
    "steps",
    [("test", "edit"), ("edit", "test", "edit")],
    ids=["pass-edit", "edit-pass-edit"],
)
async def test_a_test_before_the_latest_change_does_not_count(
    repo: Path, steps: tuple[str, ...]
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    run = Run()
    for step in steps:
        run.edit() if step == "edit" else run.test()
    assert (await end(run, repo, base))["reasons"] == ["tests_missing"]


async def test_a_wrapper_that_exits_zero_proves_nothing(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    ending = await end(Run().edit().test(HIDES), repo, base)
    assert why(ending) == ("draft", ["tests_unverified"])
    assert ending["tests"][0]["outcome"] == "unknown"


@pytest.mark.parametrize("name", list(SWALLOW))
async def test_a_swallowed_failure_is_never_a_pass(repo: Path, name: str) -> None:
    command, passes = python_test(SWALLOW[name])
    assert command and not passes, name
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    # The child exited 0 because its wrapper swallowed the runner's failure.
    ending = await end(Run().edit().test(SWALLOW[name]), repo, base)
    assert why(ending) == ("draft", ["tests_unverified"]), name


@pytest.mark.parametrize("name", list(NOT_TESTS))
def test_a_call_that_runs_no_tests_is_not_a_test(name: str) -> None:
    assert python_test(NOT_TESTS[name]) == ("", False), name


async def test_a_commit_message_naming_the_tests_does_not_hide_a_failure(
    repo: Path,
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Fix the flaky pytest fixture")
    run = Run().edit().test(exit_code=1).py(NOT_TESTS["commit_message"])
    assert why(await end(run, repo, base, commit=True)) == ("draft", ["tests_failed"])


async def test_a_call_that_edits_and_tests_proves_nothing(repo: Path) -> None:
    base = await baseline(str(repo))
    both = "open('app.py', 'w').write('x = 3\\n')\n" + PASSES_ON
    (repo / "app.py").write_text("x = 3\n")
    assert why(await end(Run().test(both), repo, base)) == (
        "draft",
        ["tests_unverified"],
    )


@pytest.mark.parametrize(
    "write",
    [
        "import subprocess\n"
        "subprocess.run(\"printf 'x = 9\\\\n' > app.py\", shell=True, check=True)\n",
        "from pathlib import Path\n"
        "with Path('app.py').open('w') as f:\n    f.write('x = 9\\n')\n",
    ],
    ids=["shell-redirect", "path-open"],
)
async def test_a_write_no_file_tool_names_after_the_test_counts(
    repo: Path, write: str
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 9\n")
    run = Run().edit().test().py(write)
    assert why(await end(run, repo, base)) == ("draft", ["tests_missing"])


async def test_a_change_no_call_names_then_committed_needs_a_test(repo: Path) -> None:
    base = await baseline(str(repo))
    write = (
        "import subprocess\n"
        "subprocess.run(\"cat > app.py <<'EOF'\\nx = 5\\nEOF\", shell=True, check=True)\n"
    )
    (repo / "app.py").write_text("x = 5\n")
    git(repo, "commit", "-qam", "Fix")
    run = Run().py(write).py(COMMIT)
    assert why(await end(run, repo, base, commit=True)) == ("draft", ["tests_missing"])


async def test_uncommitted_work_when_a_commit_was_asked_is_a_draft(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit().test()
    ending = await end(run, repo, base, commit=True)
    assert why(ending) == ("draft", ["commit_missing"])
    assert ending["uncommitted"] == ["app.py"]
    # The same work, when no commit was asked, is ready.
    assert (await end(run, repo, base))["state"] == "ready"


async def test_a_partial_commit_leaves_the_new_file_owed(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    (repo / "new.py").write_text("y = 1\n")
    git(repo, "commit", "-qam", "Fix")  # -a skips the untracked new.py
    write = {"path": "new.py", "content": "y"}
    run = Run().edit().call("file_write", write).test().py(COMMIT)
    ending = await end(run, repo, base, commit=True)
    assert why(ending) == ("draft", ["commit_missing"])
    assert ending["uncommitted"] == ["new.py"]


async def test_a_question_in_the_report_makes_it_blocked(repo: Path) -> None:
    base = await baseline(str(repo))
    asked = await end(Run("Ending: blocked: Which port should it use?"), repo, base)
    assert why(asked) == ("blocked", ["question"])
    assert asked["question"] == "Which port should it use?"
    prose = await end(Run("Added the flag.\nShould it be on by default?"), repo, base)
    assert prose["state"] == "blocked"
    assert prose["proposed"] is None
    assert prose["question"] == "Should it be on by default?"


async def test_a_stash_that_grows_hides_the_work(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "stash", "-q")
    stash = "import subprocess\nsubprocess.run(['git', 'stash'])\n"
    run = Run().edit().test(exit_code=0).py(stash)
    assert why(await end(run, repo, base, commit=True)) == ("draft", ["unchecked"])


async def test_a_folder_git_cannot_read_is_unchecked(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = await baseline(str(repo))
    (repo / ".git" / "HEAD").write_text("garbage\n")  # git refuses the repository now
    # A test after the edit: Finish it could do nothing more, so it is not offered.
    ending = await end(Run().edit().test(), repo, base)
    assert why(ending) == ("draft", ["unreadable"])
    assert not can_finish(ending)
    # No test after the edit: Finish it can run one, once.
    owed = await end(Run().edit(), repo, base)
    assert why(owed) == ("draft", ["unchecked"]) and can_finish(owed)
    again = await end(Run().edit(), repo, base, finishing=True)
    assert why(again) == ("draft", ["unreadable"]) and not can_finish(again)
    broken = Baseline(None, git=True, error=True)
    assert why(await end(Run(), repo, broken)) == ("draft", ["unreadable"])


# ── Plain endings ──


async def test_a_plain_ready_outside_git_and_without_a_proposal(tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    base = await baseline(str(folder))
    (folder / "poem.md").write_text("# Hi\n")
    ending = await end(Run("Wrote it.").edit("poem.md"), folder, base)
    assert (*why(ending), ending["proposed"]) == ("ready", [], None)
    tested = await end(Run("Wrote it.").edit("poem.py").test(), folder, base)
    assert why(tested) == ("ready", [])


async def test_code_edited_outside_git_still_owes_a_test(tmp_path: Path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "test_x.py").write_text("def test_x():\n    pass\n")
    base = await baseline(str(folder))
    assert (base.git, base.error, base.tests) == (False, False, True)
    (folder / "x.py").write_text("print(1)\n")
    for run in (
        Run().edit("x.py"),
        Run().py("open('x.py', 'w').write('print(1)')\n"),
        Run().edit("x.py").test().edit("x.py"),
    ):
        assert why(await end(run, folder, base)) == ("draft", ["tests_missing"])
    assert why(await end(Run().edit("x.py").test(), folder, base)) == ("ready", [])
    unchanged = await baseline(str(folder))
    assert why(await end(Run().py("print(1)\n"), folder, unchanged)) == ("ready", [])


async def test_a_docs_only_change_needs_no_test_and_a_build_file_does(
    repo: Path,
) -> None:
    base = await baseline(str(repo))
    (repo / "README.md").write_text("# App\nMore.\n")
    git(repo, "commit", "-qam", "Docs")
    run = Run().edit("README.md").py(COMMIT)
    assert (await end(run, repo, base, commit=True))["state"] == "ready"
    build = await baseline(str(repo))
    (repo / "CMakeLists.txt").write_text("project(y)\n")
    ending = await end(Run().edit("CMakeLists.txt"), repo, build)
    assert why(ending) == ("draft", ["tests_missing"])


async def test_failed_and_dropped_proposals_stand_and_keep_their_reasons(
    repo: Path,
) -> None:
    base = await baseline(str(repo))
    for said in ("failed", "dropped"):
        ending = await end(Run(f"Could not.\nEnding: {said}").test(), repo, base)
        assert why(ending) == (said, [])
        assert not can_finish(ending)
    (repo / "app.py").write_text("x = 2\n")
    gave_up = await end(Run("Moot now.\nEnding: dropped").edit(), repo, base)
    assert why(gave_up) == ("dropped", ["tests_missing"])
    assert can_finish(gave_up), "work it left untested can still be finished"


async def test_a_run_out_of_steps_is_not_finished(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    run = Run("I was about to run the tests.").edit()
    ending = await end(
        run, repo, base, phase="failed", reason=OUT_OF_STEPS, commit=True
    )
    assert why(ending) == ("draft", ["out_of_steps", "tests_missing", "commit_missing"])


async def test_failed_and_cancelled_runs_take_the_state_default(repo: Path) -> None:
    base = await baseline(str(repo))
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
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    claim = "Fixed it. Ran `pytest -q`: 12 passed.\nEnding: ready"
    run = Run(claim).edit().py(NOT_TESTS["printed"])
    ending = await end(run, repo, base)
    assert (*why(ending), ending["tests"]) == ("draft", ["tests_missing"], [])


async def test_tests_of_another_run_do_not_count(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit()
    run.call("python_exec", {"code": PASSES_ON}, exit_code=0, run_id="run-1")
    ending = await end(run, repo, base)
    assert (ending["reasons"], ending["tests"]) == (["tests_missing"], [])


async def test_a_denied_or_unfinished_test_is_not_a_pass(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    denied = Run().edit().call("python_exec", {"code": PASSES_ON}, status="blocked")
    assert (await end(denied, repo, base))["reasons"] == ["tests_missing"]
    cut = Run().edit().call("python_exec", {"code": PASSES_ON}, finished=False)
    assert (await end(cut, repo, base))["reasons"] == ["tests_unverified"]


async def test_repeated_call_ids_pair_in_order(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    run = Run().edit()
    code = {"code": PASSES_ON}
    run.call("python_exec", code, exit_code=1, status="error", call_id="c0")
    run.call("python_exec", code, exit_code=0, call_id="c0")
    ending = await end(run, repo, base)
    assert [t["outcome"] for t in ending["tests"]] == ["failed", "passed"]
    assert ending["state"] == "ready"


async def test_a_shell_test_counts_only_as_one_plain_command(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    plain = (
        Run().edit().call("shell_exec", {"command": "uv run pytest -q"}, exit_code=0)
    )
    assert why(await end(plain, repo, base)) == ("ready", [])
    piped = (
        Run().edit().call("shell_exec", {"command": "pytest -q | tail"}, exit_code=0)
    )
    assert why(await end(piped, repo, base)) == ("draft", ["tests_unverified"])
    red = Run().edit().call("shell_exec", {"command": "npm test"}, exit_code=1)
    assert why(await end(red, repo, base)) == ("draft", ["tests_failed"])


async def test_work_that_was_there_before_the_run_is_not_blamed(repo: Path) -> None:
    (repo / "other.py").write_text("y = 1\n")  # the person's own uncommitted work
    base = await baseline(str(repo))
    (repo / "README.md").write_text("# App\nMore.\n")
    git(repo, "commit", "-qm", "Docs", "README.md")
    run = Run().edit("README.md").py(COMMIT)
    ending = await end(run, repo, base, commit=True)
    assert (ending["state"], ending["uncommitted"]) == ("ready", [])


async def test_a_change_then_committed_still_needs_its_test(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Fix")
    run = Run().edit().py(COMMIT)
    assert why(await end(run, repo, base, commit=True)) == ("draft", ["tests_missing"])


async def test_work_left_by_an_earlier_run_is_still_owed(repo: Path) -> None:
    # The baseline is taken once, before the first run since the last ready ending;
    # every later run (after a stop, a failed claim, a crash) is judged against it.
    kept = (await baseline(str(repo))).to_json()
    (repo / "app.py").write_text("x = 2\n")  # the earlier run's change; it said failed
    base = (await baseline(str(repo))).since(kept)
    only_commit = await end(Run(), repo, base, commit=True)
    assert why(only_commit) == ("draft", ["tests_missing", "commit_missing"])
    git(repo, "commit", "-qam", "Fix")
    finished = Run().test().py(COMMIT)
    assert (await end(finished, repo, base, commit=True))["state"] == "ready"


async def test_the_stored_test_command_has_no_secrets(repo: Path) -> None:
    base = await baseline(str(repo))
    token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    code = (
        "import subprocess, sys\n"
        f"sys.exit(subprocess.run(['pytest', '-q', '--token={token}']).returncode)\n"
    )
    ending = await end(Run().test(code), repo, base)
    assert token not in ending["tests"][0]["command"]


# ── Normal flows stay ready; anything that could write is a writer ──

# Ways a worker commits or reads git after its test: none of them writes code.
COMMITS = {
    "add+commit": (
        "import subprocess\nsubprocess.run(['git', 'add', '-A'], check=True)\n"
        "subprocess.run(['git', 'commit', '-m', 'Fix add'], check=True)\n"
    ),
    "cwd": (
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-am', 'Fix add'], check=True, cwd='.')\n"
    ),
    "msg-var": (
        "import subprocess\nmsg = 'Fix add'\n"
        "subprocess.run(['git', 'commit', '-am', msg], check=True)\n"
    ),
    "if-check": (
        "import subprocess\n"
        "r = subprocess.run(['git', 'commit', '-am', 'Fix'], capture_output=True, "
        "text=True)\nif r.returncode:\n    print(r.stderr)\n"
    ),
    "log-decode": (
        "import subprocess\n"
        "print(subprocess.check_output(['git', 'log', '--oneline', '-1']).decode())\n"
    ),
    "shell": (
        "import subprocess\n"
        "subprocess.run('git commit -am \"Fix add\"', shell=True, check=True)\n"
    ),
    "import-os": (
        "import subprocess, os\n"
        "subprocess.run(['git', 'commit', '-am', 'Fix add'], check=True)\n"
    ),
    "identity": (
        "import subprocess\nWHO = ['-c', 'user.name=T', '-c', 'user.email=t@e.invalid']\n"
        "subprocess.run(['git', *['-c', 'user.name=T'], 'commit', '-qam', 'x'])\n"
        "print(open('app.py').read().strip(), __name__)\n"
    ),
}
# Calls that could change a file: each makes the test before it stale.
WRITES = {
    "open-w": "open('app.py', 'w').write('x = 9')\n",
    "path": "from pathlib import Path\nPath('app.py').write_text('x = 9')\n",
    "os-remove": "import os\nos.remove('app.py')\n",
    "module-alias": "import os\nx = os\nx.replace('a.py', 'app.py')\n",
    "sys-modules": "import sys\nsys.modules['os'].replace('a.py', 'app.py')\n",
    "rebound-reader": "import os\nprint = os.remove\nprint('app.py')\n",
    "attribute-set": "import os, json\njson.loads = os.remove\njson.loads('app.py')\n",
    "dunder": "print.__self__.open('app.py', 'w')\n",
    "getattr": "import os\ngetattr(os, 'remove')('app.py')\n",
    "other-module": "import shutil\nshutil.copy('a.py', 'app.py')\n",
    "diff-output": (
        "import subprocess\nsubprocess.run(['git', 'diff', '--output=app.py'])\n"
    ),
    "diff-o": "import subprocess\nsubprocess.run(['git', 'diff', '-o', 'app.py'])\n",
    "diff-computed": (
        "import subprocess\nopt = '--output=app.py'\n"
        "subprocess.run(['git', 'diff', opt])\n"
    ),
    "checkout": "import subprocess\nsubprocess.run(['git', 'checkout', 'app.py'])\n",
    "git-config-runs": (
        "import subprocess\n"
        "subprocess.run(['git', '-c', 'core.fsmonitor=touch app.py', 'status'])\n"
    ),
    "git-env": (
        "import subprocess\nsubprocess.run(['git', 'diff'], env={'GIT_EXTERNAL_DIFF': 'x'})\n"
    ),
    "shell-redirect": (
        "import subprocess\nsubprocess.run('git diff > app.py', shell=True)\n"
    ),
    "module-name-rebound": "import os\njson = os\njson.replace('a.py', 'app.py')\n",
}
SHELL_WRITES = ["sed -i s/1/9/ app.py", "git show HEAD:app.py > app.py", "make build"]


@pytest.mark.parametrize("code", list(COMMITS.values()), ids=list(COMMITS))
async def test_a_tested_change_then_committed_any_usual_way_is_ready(
    repo: Path, code: str
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Fix")
    ending = await end(Run().edit().test().py(code), repo, base, commit=True)
    assert why(ending) == ("ready", [])


@pytest.mark.parametrize("code", list(WRITES.values()), ids=list(WRITES))
async def test_a_call_that_could_write_after_the_test_makes_it_stale(
    repo: Path, code: str
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 9\n")
    ending = await end(Run().edit().test().py(code), repo, base)
    assert why(ending) == ("draft", ["tests_missing"])


@pytest.mark.parametrize("command", SHELL_WRITES)
async def test_a_shell_call_that_could_write_makes_the_test_stale(
    repo: Path, command: str
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 9\n")
    run = Run().edit().test().call("shell_exec", {"command": command}, exit_code=0)
    assert why(await end(run, repo, base)) == ("draft", ["tests_missing"])
    reads = Run().edit().test().call("shell_exec", {"command": "git status --short"})
    assert why(await end(reads, repo, base)) == ("ready", [])


async def test_an_unknown_tool_after_the_test_could_have_written(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 9\n")
    unknown = Run().edit().test().call("project_writer", {"path": "app.py"})
    assert why(await end(unknown, repo, base)) == ("draft", ["tests_missing"])
    known = Run().edit().test().call("file_read", {"path": "app.py"})
    assert why(await end(known, repo, base)) == ("ready", [])


# ── Runners, and the flags that make them run nothing ──


@pytest.mark.parametrize(
    "argv",
    [
        "['uv', 'run', '--with', 'pytest-xdist', 'pytest', '-n', 'auto']",
        "['uv', 'run', '--frozen', 'python', '-m', 'unittest']",
        "['poetry', 'run', 'pytest']",
        "['make', 'test']",
        "['tox']",
        "['tox', '-e', 'py312']",
        "['nox', '-s', 'tests']",
        "['npx', 'jest']",
        "['npx', 'vitest', 'run']",
        "['bun', 'test']",
        "['xcodebuild', 'test', '-scheme', 'App']",
        "['xcodebuild', '-scheme', 'App', 'test']",
        "['swift', 'test']",
        "['node', '--test']",
    ],
)
def test_more_runners_pass_their_exit_code_on(argv: str) -> None:
    command, passes = python_test(
        f"import subprocess\nsubprocess.run({argv}, check=True)\n"
    )
    assert command and passes, argv


@pytest.mark.parametrize(
    "argv",
    [
        "['pytest', '-V']",
        "['pytest', '--setup-plan']",
        "['pytest', '--fixtures']",
        "['pytest', '--pyargs', 'json']",
        "['pytest', '--co']",
        "['cargo', 'test', '--no-run']",
        "['go', 'test', '-run', '^$', './...']",
        "['go', 'test', '-run=^$', './...']",
        "['go', 'test', '-list', '.']",
        "['npx', 'vitest', 'list']",
        "['tox', '-l']",
        "['xcodebuild', 'test', '-list']",
        "['uv', 'run', 'pytest', '--collect-only']",
        "['uv', 'run', 'npm', 'test']",
    ],
)
def test_a_runner_that_runs_no_test_is_not_a_test(argv: str) -> None:
    code = f"import subprocess\nsubprocess.run({argv}, check=True)\n"
    assert python_test(code) == ("", False), argv


@pytest.mark.parametrize(
    "argv",
    [
        "['cargo', 'test', '--manifest-path=/elsewhere/Cargo.toml']",
        "['uv', 'run', '--project', '../other', 'pytest']",
        "['pytest', '--rootdir=~/elsewhere']",
    ],
)
def test_a_runner_pointed_elsewhere_has_an_unknown_result(argv: str) -> None:
    code = f"import subprocess\nsubprocess.run({argv}, check=True)\n"
    command, passes = python_test(code)
    assert command and not passes, argv


async def test_a_newline_in_a_shell_test_is_an_operator(repo: Path) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    for command in ("pytest -q\ntrue", "pytest -q\r\ntrue"):
        run = Run().edit().call("shell_exec", {"command": command}, exit_code=0)
        assert why(await end(run, repo, base)) == ("draft", ["tests_unverified"])


# ── What a change is, and what is kept ──


@pytest.mark.parametrize("path", ["pkg/docs/views.py", "docs/conf.py", "notes.txt"])
async def test_only_doc_extensions_are_docs(repo: Path, path: str) -> None:
    base = await baseline(str(repo))
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text("x = 1\n")
    ending = await end(Run().edit(path), repo, base)
    assert why(ending) == ("draft", ["tests_missing"]), path


async def test_a_stash_the_person_made_between_runs_is_not_the_run(repo: Path) -> None:
    (repo / "app.py").write_text("x = 7\n")  # the earlier run's change
    kept = (await baseline(str(repo))).to_json()
    git(repo, "stash", "-q")  # the person stashes it before the next run
    base = (await baseline(str(repo))).since(kept)
    ending = await end(Run().test(), repo, base)
    assert "unchecked" not in ending["reasons"], ending


async def test_the_kept_baseline_holds_changed_files_but_never_the_stash(
    repo: Path,
) -> None:
    (repo / "app.py").write_text("x = 2\n")
    (repo / "README.md").write_text("# More\n")
    git(repo, "stash", "-q")
    (repo / "app.py").write_text("x = 3\n")
    (repo / "README.md").write_text("# Other\n")
    now = await baseline(str(repo))
    assert (now.git, now.error, now.stash) == (True, False, 1)
    kept = now.to_json()
    assert set(kept) == {"git", "head", "stamps", "tests"}, "never the stash"
    assert sorted(kept["stamps"]) == ["README.md", "app.py"]
    assert kept["tests"] is True


async def test_docs_an_earlier_run_left_uncommitted_are_still_owed(repo: Path) -> None:
    kept = (await baseline(str(repo))).to_json()
    # The earlier run edited the README, a commit was asked, and it was stopped.
    (repo / "README.md").write_text("# New\n")
    base = (await baseline(str(repo))).since(kept)
    ending = await end(Run(), repo, base, commit=True)
    assert why(ending) == ("draft", ["commit_missing"])
    assert ending["uncommitted"] == ["README.md"]


async def test_work_a_run_stashed_stays_unchecked_until_the_stash_is_back(
    repo: Path,
) -> None:
    kept = (await baseline(str(repo))).to_json()
    first = (await baseline(str(repo))).since(kept)
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "stash", "-q")  # the earlier run stashed its work: it ended unchecked
    assert why(await end(Run().edit(), repo, first)) == ("draft", ["unchecked"])
    kept["stash_floor"] = await stash_floor(str(repo), first)
    assert kept["stash_floor"] == 0
    later = (await baseline(str(repo))).since(kept)
    assert await stash_floor(str(repo), later) is None, "this run grew nothing"
    ending = await end(Run().test(), repo, later, commit=True)
    assert why(ending) == ("draft", ["stashed"]), "a test ran: Finish it cannot help"
    assert not can_finish(ending)
    git(repo, "stash", "pop", "-q")  # the work is back in sight: the usual rules
    back = (await baseline(str(repo))).since(kept)
    pop = Run().call("shell_exec", {"command": "git stash pop"}, exit_code=0).test()
    assert why(await end(pop, repo, back)) == ("ready", [])


async def test_a_folder_with_too_many_changes_is_unchecked(repo: Path) -> None:
    for n in range(MAX_STAMPS + 1):
        (repo / f"f{n}.py").write_text("")
    assert (await baseline(str(repo))).error
    base = await baseline(str(repo))
    ending = await end(Run(), repo, base)
    assert why(ending) == ("draft", ["too_large"]) and not can_finish(ending)
    # Asked to commit: Finish it can commit the changes, and asks to.
    owed = await end(Run(), repo, base, commit=True)
    assert why(owed) == ("draft", ["commit_missing", "unchecked"])
    assert can_finish(owed) and finish_message(owed) == finish_message(
        {"reasons": ["commit_missing"]}
    )
    again = await end(Run(), repo, base, commit=True, finishing=True)
    assert why(again) == ("draft", ["too_large"]) and not can_finish(again)


def nested(repo: Path) -> Path:
    folder = repo / "sub"
    folder.mkdir()
    (folder / "app.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Sub")
    return folder


async def test_a_file_already_changed_in_a_nested_task_folder_is_still_seen(
    repo: Path,
) -> None:
    folder = nested(repo)
    (folder / "app.py").write_text("x = 2\n")  # changed before the task started
    base = await baseline(str(folder))
    assert list(base.stamps or {}) == ["app.py"]
    assert (base.stamps or {})["app.py"][0] == len("x = 2\n"), "its own stamp"
    (folder / "app.py").write_text("x = 333\n")  # the run's edit
    assert why(await end(Run().edit(), folder, base)) == ("draft", ["tests_missing"])


async def test_a_commit_in_another_folder_of_the_repository_is_not_the_tasks(
    repo: Path,
) -> None:
    folder = nested(repo)
    base = await baseline(str(folder))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "Elsewhere")
    assert why(await end(Run(), folder, base, commit=True)) == ("ready", [])


async def test_a_task_folder_inside_a_repository_sees_only_itself(repo: Path) -> None:
    folder = repo / "sub"
    folder.mkdir()
    (folder / "app.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Sub")
    base = await baseline(str(folder))
    (repo / "app.py").write_text("x = 2\n")  # outside the task's folder
    (folder / "app.py").write_text("x = 2\n")
    ending = await end(Run().edit(), folder, base, commit=True)
    assert why(ending) == ("draft", ["tests_missing", "commit_missing"])
    assert ending["uncommitted"] == ["app.py"]
    git(repo, "commit", "-qm", "Fix", "sub/app.py")
    committed = await end(Run().edit().test(), folder, base, commit=True)
    assert why(committed) == ("ready", [])


# ── Outside git: a snapshot of the folder; tests only where they apply ──

# A run that only fetched a page: it ran commands, none of which wrote a file.
FETCH = "import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'print(1)'])\n"


def folder_of(tmp_path: Path, *, tests: bool) -> Path:
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "notes.md").write_text("# Notes\n")
    if tests:
        (folder / "test_x.py").write_text("def test_x():\n    pass\n")
    return folder


@pytest.mark.parametrize("tests", [False, True], ids=["no-tests", "tests"])
async def test_a_run_that_only_visited_a_website_outside_git_is_ready(
    tmp_path: Path, tests: bool
) -> None:
    folder = folder_of(tmp_path, tests=tests)
    base = await baseline(str(folder))
    shell = {"command": "curl -sL https://example.invalid/ -o /dev/null"}
    run = Run("The site's title is Example.\nEnding: ready").py(FETCH)
    run.call("shell_exec", shell, exit_code=0)
    ending = await end(run, folder, base)
    assert (*why(ending), ending["notes"]) == ("ready", [], [])


async def test_code_changed_where_no_test_setup_was_found_still_owes_a_test(
    tmp_path: Path,
) -> None:
    folder = folder_of(tmp_path, tests=False)
    base = await baseline(str(folder))
    assert (base.git, base.error, base.tests) == (False, False, False)
    (folder / "page.py").write_text("print('hi')\n")
    ending = await end(Run().edit("page.py"), folder, base)
    assert (*why(ending), ending["notes"]) == ("draft", ["tests_missing"], [])
    assert can_finish(ending)
    # Finish it found no runner: it ends Ready, saying no tests apply.
    done = await end(Run(), folder, base, tests=True, finishing=True)
    assert (*why(done), done["notes"]) == ("ready", [], ["no_tests"])
    assert not can_finish(done)


async def test_a_write_by_a_shell_command_outside_git_is_seen(tmp_path: Path) -> None:
    folder = folder_of(tmp_path, tests=True)
    base = await baseline(str(folder))
    (folder / "x.py").write_text("x = 9\n")  # what the command below wrote
    run = Run().call("shell_exec", {"command": "printf 'x = 9' > x.py"}, exit_code=0)
    assert why(await end(run, folder, base)) == ("draft", ["tests_missing"])
    gone = await baseline(str(folder))
    (folder / "x.py").unlink()  # a deletion is a change too
    removed = Run().call("shell_exec", {"command": "rm x.py"}, exit_code=0)
    assert why(await end(removed, folder, gone)) == ("draft", ["tests_missing"])
    edited = Run().call("shell_exec", {"command": "make"}, exit_code=0).test()
    assert why(await end(edited, folder, base)) == ("ready", [])


@pytest.mark.parametrize("ran", ["nothing", "pytest-collects-nothing"])
async def test_finish_it_with_no_test_that_applies_ends_ready(
    tmp_path: Path, ran: str
) -> None:
    folder = folder_of(tmp_path, tests=False)
    kept = (await baseline(str(folder))).to_json()
    (folder / "page.py").write_text("print('hi')\n")  # an earlier run's change
    base = (await baseline(str(folder))).since(kept)
    run = Run("Nothing to test here.\nEnding: ready")
    if ran != "nothing":
        run.test(exit_code=5)  # pytest exits 5 when it collects no test
    ending = await end(run, folder, base, tests=True, finishing=True)
    assert (*why(ending), ending["notes"], ending["tests"]) == (
        "ready",
        [],
        ["no_tests"],
        [],
    )
    # The same run, not Finish it's: the change still owes its test.
    owed = await end(run, folder, base, tests=True)
    assert why(owed) == ("draft", ["tests_missing"])


async def test_tests_asked_for_are_owed_where_a_setup_applies(repo: Path) -> None:
    base = await baseline(str(repo))
    assert why(await end(Run(), repo, base, tests=True)) == ("draft", ["tests_missing"])
    assert why(await end(Run().test(), repo, base, tests=True)) == ("ready", [])
    assert why(await end(Run(), repo, base)) == ("ready", [])


async def test_a_git_folder_with_no_test_setup_owes_a_test_only_for_changed_code(
    repo: Path,
) -> None:
    git(repo, "rm", "-rq", "tests")
    git(repo, "commit", "-qm", "No tests")
    base = await baseline(str(repo))
    assert base.tests is False
    asked = await end(Run(), repo, base, tests=True)
    assert (*why(asked), asked["notes"]) == ("ready", [], ["no_tests"])
    (repo / "app.py").write_text("x = 2\n")
    ending = await end(Run().edit(), repo, base)
    assert (*why(ending), ending["notes"]) == ("draft", ["tests_missing"], [])
    done = await end(Run(), repo, base, tests=True, finishing=True)
    assert (*why(done), done["notes"]) == ("ready", [], ["no_tests"])


async def test_removing_the_tests_during_the_run_does_not_lift_them(
    repo: Path, tmp_path: Path
) -> None:
    base = await baseline(str(repo))
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "rm", "-rq", "tests")
    assert why(await end(Run().edit(), repo, base)) == ("draft", ["tests_missing"])
    outside = folder_of(tmp_path, tests=True)
    plain = await baseline(str(outside))
    (outside / "test_x.py").unlink()
    (outside / "x.py").write_text("x = 1\n")
    assert why(await end(Run().edit("x.py"), outside, plain)) == (
        "draft",
        ["tests_missing"],
    )


async def test_a_snapshot_too_large_or_unreadable_is_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = folder_of(tmp_path, tests=False)
    monkeypatch.setattr(task_endings, "MAX_STAMPS", 3)
    monkeypatch.setattr(task_endings, "MAX_SEEN", 8)
    for n in range(3):
        (folder / f"f{n}.py").write_text("")
    assert not (await baseline(str(folder))).error, "three code files fit"
    (folder / "f3.py").write_text("")
    assert (await baseline(str(folder))).error, "a fourth code file does not"
    (folder / "f3.py").unlink()
    for n in range(6):
        (folder / f"d{n}.md").write_text("")
    assert (await baseline(str(folder))).error, "nor ten files in all"
    for n in range(6):
        (folder / f"d{n}.md").unlink()
    base = await baseline(str(folder))
    venv = folder / ".venv" / "lib"
    venv.mkdir(parents=True)
    for n in range(20):
        (venv / f"m{n}.py").write_text("")
    assert not (await baseline(str(folder))).error, "environments are not read"
    assert why(await end(Run(), folder, base)) == ("ready", [])
    missing = Baseline(None, error=True, tests=False)
    assert why(await end(Run(), folder, missing)) == ("draft", ["unreadable"])
    large = Baseline(None, error=True, why="too_large")
    ending = await end(Run().py(FETCH), folder, large)
    assert why(ending) == ("draft", ["unchecked"]), "Finish it can run a test once"
    again = await end(Run().py(FETCH).test(), folder, large, finishing=True)
    assert why(again) == ("draft", ["too_large"]) and not can_finish(again)
    tested = await end(Run().py(FETCH).test(), folder, large)
    assert why(tested) == ("draft", ["too_large"]), "the test already ran"


async def test_a_task_kept_before_snapshots_is_judged_from_its_own_start(
    tmp_path: Path,
) -> None:
    folder = folder_of(tmp_path, tests=True)
    (folder / "x.py").write_text("x = 1\n")
    old = {"git": False, "head": "", "stamps": {}}  # how a folder outside git was kept
    base = (await baseline(str(folder))).since(old)
    assert why(await end(Run(), folder, base)) == ("ready", [])


async def test_a_saved_page_or_picture_is_not_code(tmp_path: Path) -> None:
    folder = folder_of(tmp_path, tests=True)
    base = await baseline(str(folder))
    for name in ("page.html", "logo.png", "rows.csv", "paper.pdf", "notes.txt"):
        (folder / name).write_text("<html></html>")
    fetch = (
        "import subprocess\nsubprocess.run(['curl', '-o', 'page.html', 'https://x'])\n"
    )
    ending = await end(Run().py(fetch), folder, base)
    assert why(ending) == ("ready", [])
    (folder / "build.sh").write_text("make\n")  # a script is code
    assert why(await end(Run().py(fetch), folder, base)) == ("draft", ["tests_missing"])


async def test_a_file_tool_edit_outside_the_folder_counts_outside_git(
    tmp_path: Path,
) -> None:
    folder = folder_of(tmp_path, tests=True)
    other = tmp_path / "proj"
    other.mkdir()
    (other / "lib.py").write_text("z = 1\n")
    base = await baseline(str(folder))
    (other / "lib.py").write_text("z = 2\n")
    edit = Run().edit(str(other / "lib.py"))
    assert why(await end(edit, folder, base)) == ("draft", ["tests_missing"])
    page = Run().call("file_write", {"path": str(other / "page.html"), "content": "x"})
    assert why(await end(page, folder, base)) == ("ready", []), "not code"
    refused = Run().call(
        "file_edit", {"path": str(other / "lib.py")}, status="error", exit_code=None
    )
    assert why(await end(refused, folder, base)) == ("ready", [])


async def test_what_a_snapshot_cannot_see(tmp_path: Path) -> None:
    """The snapshot's known limits outside git, for unknown commands (docs: for-agents)."""
    folder = folder_of(tmp_path, tests=True)
    (folder / "app.py").write_text("x = 1\n")
    (folder / "venv").mkdir()
    (folder / "venv" / "lib.py").write_text("y = 1\n")
    real = tmp_path / "real.py"
    real.write_text("a = 1\n")
    (folder / "link.py").symlink_to(real)
    base = await baseline(str(folder))
    stat = (folder / "app.py").stat()
    (folder / "app.py").write_text("x = 2\n")  # same size, time put back
    os.utime(folder / "app.py", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    (folder / "venv" / "lib.py").write_text("y = 22\n")  # a skipped folder
    real.write_text("a = 222\n")  # through a symlink: the link is read, not its file
    ending = await end(Run().py("open('app.py', 'w').write('x = 2')\n"), folder, base)
    assert why(ending) == ("ready", [])
    # A file tool names its file, so the same edits through it count.
    assert why(await end(Run().edit("venv/lib.py"), folder, base)) == (
        "draft",
        ["tests_missing"],
    )


@pytest.mark.parametrize(
    ("name", "text", "marks"),
    [
        ("tests.py", "", True),
        ("parser_tests.py", "", True),
        ("test.py", "", True),
        ("util_test.ts", "", True),
        ("util_test.js", "", True),
        ("util_test.go", "", True),
        ("manage.py", "", True),
        ("phpunit.xml", "", True),
        ("mix.exs", "", True),
        ("CMakeLists.txt", "project(x)\nenable_testing()\n", True),
        ("CMakeLists.txt", "project(x)\nadd_test(NAME t COMMAND t)\n", True),
        ("CMakeLists.txt", "project(x)\n", False),
        ("pyproject.toml", "[tool.pytest.ini_options]\naddopts = '-q'\n", True),
        ("pyproject.toml", "[project]\nname = 'x'\n", False),
        ("setup.cfg", "[tool:pytest]\naddopts = -q\n", True),
        ("setup.cfg", "[metadata]\nname = x\n", False),
        (
            "App.Tests.csproj",
            '<PackageReference Include="Microsoft.NET.Test.Sdk" />',
            True,
        ),
        ("App.csproj", '<Project Sdk="Microsoft.NET.Sdk" />', False),
        ("package.json", '{"scripts": {"test": "vitest run"}}', True),
        (
            "package.json",
            '{\n  "scripts": {\n    "test": "echo \\"Error: no test specified\\" && exit 1"\n  }\n}\n',
            False,
        ),
        ("package.json", '{"scripts": {"build": "tsc"}}', False),
        ("package.json", "{not json", True),
    ],
)
def test_what_says_a_test_runner_applies(
    tmp_path: Path, name: str, text: str, marks: bool
) -> None:
    (tmp_path / name).write_text(text)
    assert _test_marker(str(tmp_path), name, False) is marks


@pytest.mark.parametrize("build", ["build.gradle", "pom.xml"])
def test_a_jvm_build_marks_tests_only_with_its_test_sources(
    tmp_path: Path, build: str
) -> None:
    (tmp_path / build).write_text("")
    assert not _test_marker(str(tmp_path), build, False)
    (tmp_path / "src" / "test").mkdir(parents=True)
    assert _test_marker(str(tmp_path), build, False)


async def test_an_untested_change_in_a_django_project_is_owed(repo: Path) -> None:
    git(repo, "rm", "-rq", "tests")
    (repo / "shop").mkdir()
    (repo / "manage.py").write_text("x")
    (repo / "shop" / "views.py").write_text("v = 1\n")
    (repo / "shop" / "tests.py").write_text("from django.test import TestCase\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Shop")
    base = await baseline(str(repo))
    assert base.tests
    (repo / "shop" / "views.py").write_text("v = 2\n")
    ending = await end(Run().edit("shop/views.py"), repo, base)
    assert (*why(ending), ending["notes"]) == ("draft", ["tests_missing"], [])
    # Finish it's run in a folder where tests apply still owes a test that ran.
    finished = await end(Run(), repo, base, tests=True, finishing=True)
    assert why(finished) == ("draft", ["tests_missing"])


@pytest.mark.parametrize(
    "text",
    [
        "Finish it: run the tests that cover your changes, fix what fails, and end "
        "your report with its Ending line.",
        "收尾：运行覆盖你改动的测试，修好失败的地方，提交改动，报告最后写上 Ending 行。",
        "Fix add in calc.py\n\nFinish it: run the tests that cover your changes, fix "
        "what fails, and end your report with its Ending line.",
    ],
)
def test_finish_its_own_message_is_known_in_every_language(text: str) -> None:
    assert finishes(text)
    assert finishes(finish_message({"reasons": ["commit_missing"]}))


@pytest.mark.parametrize(
    "text",
    ["Run the tests", "Finish it", "", "收尾：跑测试", "Finish it: run the tests"],
)
def test_other_words_are_not_finish_its_message(text: str) -> None:
    assert not finishes(text)


@pytest.mark.parametrize(
    ("text", "asks"),
    [
        ("Fix the parser and run the tests", True),
        ("改完跑一下测试", True),
        ("run pytest on it", True),
        ("Finish it: run the tests that cover your changes", True),
        ("收尾：运行覆盖你改动的测试，修好失败的地方", True),
        ("Don't run the tests, just commit", False),
        ("不用跑测试", False),
        ("测试一下这个网站能不能打开", False),
        ("Visit example.com and tell me its title", False),
        ("The log says `pytest` failed; why?", False),
        (
            "Check the parser handles empty input, and don't forget to run the tests.",
            True,
        ),
        ("Don't forget the tests", True),
        ("Make sure you don't skip the tests.", True),
        ("Make sure the tests pass.", True),
        ("Make sure all the tests still pass", True),
        ("别忘了跑测试", True),
        ("不要忘记运行测试", True),
        ("确保测试都通过", True),
        ("Don't forget to commit", False),
        ("别忘了提交", False),
        ("Make sure the site loads", False),
        ("先不跑测试", False),
    ],
)
def test_tests_are_asked_in_plain_words(text: str, asks: bool) -> None:
    assert asks_tests(text) is asks


# ── git that cannot run ──


def fake_git(tmp_path: Path, err: str, code: int) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "git"
    script.write_text(f"#!/bin/sh\necho '{err}' >&2\nexit {code}\n")
    script.chmod(0o755)
    return bin_dir


@pytest.mark.parametrize(
    "where",
    ["none", "stand-in"],
)
async def test_git_that_cannot_run_is_not_a_git_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    if where == "none":
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    else:
        shim = "xcrun: error: invalid active developer path (/Library/Developer)"
        monkeypatch.setenv("PATH", str(fake_git(tmp_path, shim, 1)))
    base = await baseline(str(folder))
    assert (base.git, base.error) == (False, False)
    ending = await end(Run().edit("x.py").test(), folder, base)
    assert why(ending) == ("ready", [])


async def test_git_that_runs_but_fails_is_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    monkeypatch.setenv("PATH", str(fake_git(tmp_path, "fatal: unsafe repository", 128)))
    base = await baseline(str(folder))
    assert (base.git, base.error) == (True, True)


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


# ── The one shape a passing test has ──


@pytest.mark.parametrize(
    "code",
    [
        PASSES_ON,
        "import subprocess\nsubprocess.run(['pytest', '-q'], check=True)",
        "import subprocess\nsubprocess.check_call(['npm', 'test'])",
        "import subprocess\nsubprocess.run(['uv', 'run', 'pytest'], check=True, timeout=600)",
        "import subprocess, sys\n"
        "sys.exit(subprocess.run(['go', 'test', './...'], capture_output=True).returncode)",
        "import subprocess\nsubprocess.run(['npm', 'run', 'test'], check=True)",
        "import subprocess\nsubprocess.check_call(['cargo', 'test'])",
    ],
)
def test_a_test_that_passes_its_exit_code_on(code: str) -> None:
    command, passes = python_test(code)
    assert command and passes


# ── Where the ending and the baseline are kept ──


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
        first = await tasks.keep_work_base(record.task_id, {"git": True, "head": "a"})
        again = await tasks.keep_work_base(record.task_id, {"git": True, "head": "b"})
        assert first == again == {"git": True, "head": "a"}, "the first run's stays"
        await tasks.change(record.task_id, state="running", actor="scheduler")
        with pytest.raises(ValueError):
            await tasks.change(
                record.task_id, state="queued", ending={"state": "ready"}
            )
        draft = {"state": "draft", "reasons": ["tests_missing"]}
        done = await tasks.change(record.task_id, state="done", ending=draft)
        assert done.ending == {**draft, "at": done.finished_at}
        assert shown(done) == done.ending
        probe = {"git": True, "head": "c"}
        owed = await tasks.keep_work_base(record.task_id, probe)
        assert owed == {"git": True, "head": "a"}, "owed until ready"
        await tasks.keep_stash_floor(record.task_id, 2)
        await tasks.keep_stash_floor(record.task_id, 3)
        held = await tasks.keep_work_base(record.task_id, probe)
        assert held == {"git": True, "head": "a", "stash_floor": 2}, "the lowest stays"
        assert not hasattr(done, "work_base"), "lists and reads never carry it"
        reopened = await tasks.reopen(record.task_id, "Finish it: run the tests")
        assert shown(reopened) is None
        await tasks.change(record.task_id, state="running", actor="scheduler")
        ready = await tasks.change(
            record.task_id, state="done", ending={"state": "ready"}
        )
        assert ready.ending is not None
        other, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "Other"}),
            workspace=str(tmp_path),
            origin="user",
        )
        await tasks.keep_stash_floor(other.task_id, 1)  # no baseline kept: nothing
        assert await tasks.keep_work_base(other.task_id, probe) == probe
        fresh = await tasks.keep_work_base(record.task_id, probe)
        assert fresh == probe, "a ready ending owes nothing: the next run starts anew"
    finally:
        await store.aclose()


async def test_an_older_store_gains_the_ending_columns(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    spec = TaskSpec.from_payload({"objective": "Old"})
    record, _ = await tasks.create_task(spec, workspace=str(tmp_path), origin="user")
    await store.aclose()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE tasks DROP COLUMN ending")
        conn.execute("ALTER TABLE tasks DROP COLUMN work_base")
    store = SessionStore(path, writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        again = await tasks.get_task(record.task_id)
        assert again.ending is None
        kept = await tasks.keep_work_base(record.task_id, {"git": False})
        assert kept == {"git": False}
    finally:
        await store.aclose()


# ── What asks for a commit ──


@pytest.mark.parametrize(
    "text,asks",
    [
        ("Fix add in calc.py, run the tests and commit", True),
        ("修复 calc.py 然后提交", True),
        ("push it when done", True),
        ("Fix the parser and push it", True),
        ("Fix the parser and open a PR", True),
        ("Fix the parser, don't push, just commit", True),
        ("Fix the parser. Do not push. Commit when tests pass.", True),
        ("修好 parser，不要推送，提交就行", True),
        ("Fix the parser and check it in", True),
        ("Fix add in calc.py; don't commit yet", False),
        ("修复 calc.py，先不要提交", False),
        ("The log says `git commit` failed; explain why", False),
        ("Look at the committed history", False),
    ],
)
def test_a_commit_is_asked_in_plain_words(text: str, asks: bool) -> None:
    assert asks_commit(text) is asks


def test_the_main_chat_reads_how_a_task_ended_in_plain_words() -> None:
    from localmelo.support.gateway.console.front import _row, answer_head

    ending = {"state": "draft", "reasons": ["tests_missing"], "at": 5.0}
    record = SimpleNamespace(
        task_id="t1",
        title="Fix",
        state="done",
        wait_reason="",
        level=2,
        finished_at=5.0,
    )
    record.ending = ending
    assert _row(record) == (  # type: ignore[arg-type]
        "t1 · Fix · done, not finished (No test ran after the last change) · P2"
    )
    record.ending = {
        "state": "blocked",
        "reasons": ["question"],
        "question": "Which port?",
        "at": 5.0,
    }
    assert _row(record) == "t1 · Fix · needs you (Which port?) · P2"  # type: ignore[arg-type]
    record.ending = {"state": "ready", "reasons": [], "at": 5.0}
    assert _row(record) == "t1 · Fix · ready · P2"  # type: ignore[arg-type]
    record.ending = {"state": "ready", "reasons": [], "notes": ["no_tests"], "at": 5.0}
    assert _row(record) == "t1 · Fix · ready (No tests apply) · P2"  # type: ignore[arg-type]
    record.ending = {
        "state": "dropped",
        "proposed": "dropped",
        "reasons": [],
        "at": 5.0,
    }
    assert _row(record) == "t1 · Fix · not needed · P2"  # type: ignore[arg-type]
    assert answer_head("Fixed the report.\nEnding: ready") == "Fixed the report."
