"""The review notes a step that goes by the exit status of a pipeline.

The status of ``pytest | tail`` is tail's, so a step that commits when it is
0 commits after failed tests. Without ``pipefail`` such a step is undecided,
which asks in Ask mode and is blocked in Auto mode, and the reason says why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import ActionReview

PIPE_NOTE = "Relies on the exit status of a pipeline"


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    folder = tmp_path / "home" / "ws"
    folder.mkdir(parents=True)
    return folder


def _review(ws: Path, tool: str, **arguments: Any) -> ActionReview:
    return Checker().review_action(
        ActionReviewPayload(
            tool,
            arguments,
            str(ws),
            home=str(ws.parent),
            executable="/opt/melo/bin/python3",
            prefix="/opt/melo",
        )
    )


def _py(ws: Path, command: str, tail: str = "print(r.returncode)\n") -> ActionReview:
    code = f"import subprocess\nr = subprocess.run({command!r}, shell=True)\n{tail}"
    return _review(ws, "python_exec", code=code)


def test_a_pipe_into_tail_whose_status_is_read_is_undecided(ws: Path) -> None:
    review = _py(ws, "pytest -q | tail -5")

    assert review.verdict == "undecided"
    assert review.reasons[0].startswith(PIPE_NOTE)
    assert "(tail)" in review.reasons[0] and "pipefail" in review.reasons[0]


def test_pipefail_makes_the_status_the_tests(ws: Path) -> None:
    assert _py(ws, "set -o pipefail; pytest -q | tail -5").verdict == "routine"
    assert _py(ws, "set -eo pipefail\npytest -q | tail -5").verdict == "routine"
    assert _py(ws, "pytest -q | tail -5; echo ${PIPESTATUS[0]}").verdict == "routine"


def test_a_plain_command_has_its_own_status(ws: Path) -> None:
    assert _py(ws, "pytest -q").verdict == "routine"


def test_a_pipe_whose_status_nothing_reads_is_routine(ws: Path) -> None:
    review = _py(ws, "pytest -q | tail -5", tail="print(r.stdout)\n")

    assert review.verdict == "routine"


@pytest.mark.parametrize(
    "call",
    [
        "subprocess.run('pytest | tail -5', shell=True, check=True)",
        "subprocess.check_call('pytest | tail -5', shell=True)",
        "subprocess.check_output('pytest | tail -5', shell=True)",
        "print(subprocess.call('pytest | tail -5', shell=True))",
        "print(os.system('pytest | tail -5'))",
    ],
)
def test_other_ways_to_go_by_the_status(ws: Path, call: str) -> None:
    review = _review(ws, "python_exec", code=f"import os, subprocess\n{call}\n")

    assert review.verdict == "undecided"
    assert review.reasons[0].startswith(PIPE_NOTE)


def test_a_status_dropped_on_the_floor_is_not_relied_on(ws: Path) -> None:
    code = "import os, subprocess\nos.system('pytest | tail -5')\n"

    assert _review(ws, "python_exec", code=code).verdict == "routine"


def test_a_command_that_goes_on_after_a_piped_test_run_is_undecided(ws: Path) -> None:
    pipe = _review(ws, "shell_exec", command="pytest -q | tail -5 && git commit -m x")
    either = _review(ws, "shell_exec", command="make test | tee log || echo failed")

    for review in (pipe, either):
        assert review.verdict == "undecided"
        assert review.reasons[0].startswith(PIPE_NOTE)


def test_a_shell_command_that_is_safe_stays_routine(ws: Path) -> None:
    commands = [
        "set -o pipefail; pytest -q | tail -5 && git commit -m x",
        "pytest -q && git commit -m x",
        "pytest -q | tail -5",
        "cat a.txt | grep x && echo found",
        "pytest -q | tail -5; git status",
    ]

    for command in commands:
        assert _review(ws, "shell_exec", command=command).verdict == "routine", command
