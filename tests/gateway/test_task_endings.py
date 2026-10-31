"""How a task ended is decided by code from its own run, never from its report.

The run is a stand-in store (one turn, its tool events) over a real temporary
git repository: each test changes the repository the way the run's tool calls
say they did, then reads the ending :func:`decide` gives. The rule is fail
closed: ready needs positive proof, any doubt leaves a draft with its reason.
"""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.front_rules import asks_commit
from localmelo.support.gateway.task_endings import (
    MAX_STAMPS,
    Baseline,
    baseline,
    can_finish,
    decide,
    proposal,
    python_test,
    report_text,
    shown,
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
    ending = await end(Run().edit().test(), repo, base)
    assert why(ending) == ("draft", ["unchecked"])
    broken = Baseline(None, git=True, error=True)
    assert why(await end(Run(), repo, broken)) == ("draft", ["unchecked"])


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
    base = await baseline(str(folder))
    (folder / "x.py").write_text("print(1)\n")
    for run in (
        Run().edit("x.py"),
        Run().py("open('x.py', 'w').write('print(1)')\n"),
        Run().edit("x.py").test().edit("x.py"),
    ):
        assert why(await end(run, folder, base)) == ("draft", ["tests_missing"])
    assert why(await end(Run().py("print(1)\n"), folder, base)) == ("ready", [])


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


async def test_the_kept_baseline_holds_code_paths_only(repo: Path) -> None:
    (repo / "app.py").write_text("x = 2\n")
    (repo / "README.md").write_text("# More\n")
    git(repo, "stash", "-q")
    (repo / "app.py").write_text("x = 3\n")
    (repo / "README.md").write_text("# Other\n")
    now = await baseline(str(repo))
    assert (now.git, now.error, now.stash) == (True, False, 1)
    kept = now.to_json()
    assert set(kept) == {"git", "head", "stamps"}, "never the stash"
    assert list(kept["stamps"]) == ["app.py"]


async def test_a_folder_with_too_many_changes_is_unchecked(repo: Path) -> None:
    for n in range(MAX_STAMPS + 1):
        (repo / f"f{n}.py").write_text("")
    assert (await baseline(str(repo))).error
    assert why(await end(Run(), repo, await baseline(str(repo)))) == (
        "draft",
        ["unchecked"],
    )


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
        assert not hasattr(done, "work_base"), "lists and reads never carry it"
        reopened = await tasks.reopen(record.task_id, "Finish it: run the tests")
        assert shown(reopened) is None
        await tasks.change(record.task_id, state="running", actor="scheduler")
        ready = await tasks.change(
            record.task_id, state="done", ending={"state": "ready"}
        )
        assert ready.ending is not None
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
    record.ending = {
        "state": "dropped",
        "proposed": "dropped",
        "reasons": [],
        "at": 5.0,
    }
    assert _row(record) == "t1 · Fix · not needed · P2"  # type: ignore[arg-type]
    assert answer_head("Fixed the report.\nEnding: ready") == "Fixed the report."
