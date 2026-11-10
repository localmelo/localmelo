"""The stage table: each row once, and the priority between two true facts."""

from typing import Any

import pytest

from localmelo.support.task_stage import CALL_REASON, Stage, task_stage

APPROVAL = {"pending_approvals": [{"request_id": "r1"}]}


@pytest.mark.parametrize(
    ("task", "stopping", "stage"),
    [
        # Each row once.
        (
            {"state": "done", "ending": {"state": "draft"}},
            False,
            Stage("finished", ending="draft"),
        ),
        ({"state": "done"}, False, Stage("finished")),
        (
            {"state": "done", "ending": {"state": "blocked"}},
            False,
            Stage("finished", "person", "review", "blocked"),
        ),
        (
            {"state": "failed", "out_of_steps": True},
            False,
            Stage("finished", "person", "continue", "dropped"),
        ),
        ({"state": "expired"}, False, Stage("finished", ending="dropped")),
        ({"state": "running"}, True, Stage("stopping")),
        (
            {"state": "running", **APPROVAL},
            False,
            Stage("approval", "person", "approval"),
        ),
        (
            {"state": "running", "ask": {"on": "person"}},
            False,
            Stage("answer", "person", "answer"),
        ),
        (
            {"state": "running", "ask": {"on": "chat"}},
            False,
            Stage("asked", "model", "answer"),
        ),
        (
            {"state": "held", "wait_reason": "approval"},
            False,
            Stage("approval", "person", "approval"),
        ),
        (
            {"state": "held", "wait_reason": "Resources"},
            False,
            Stage("slot", "slot", "Resources"),
        ),
        (
            {"state": "queued", "wait_reason": "Battery"},
            False,
            Stage("slot", "time", "Battery"),
        ),
        ({"state": "held", "wait_reason": "scheduler"}, False, Stage("model", "model")),
        ({"state": "queued"}, False, Stage("queued", "slot")),
        ({"state": "parked"}, False, Stage("paused", "person", "resume")),
        (
            {"state": "interrupted", "wait_reason": CALL_REASON},
            False,
            Stage("resume", "person", "resume", why="call"),
        ),
        (
            {"state": "interrupted", "wait_reason": "park not saved"},
            False,
            Stage("resume", "person", "resume"),
        ),
        ({"state": "running"}, False, Stage("running")),
        # Fails closed: an unknown state is the plain running stage, never empty.
        ({"state": "someday"}, False, Stage("running")),
        # Priority: an approval or a question over running and a hold, a Stop
        # over every wait, finished over everything.
        (
            {"state": "held", "wait_reason": "Resources", **APPROVAL},
            False,
            Stage("approval", "person", "approval"),
        ),
        (
            {"state": "held", "ask": {"on": "person"}},
            False,
            Stage("answer", "person", "answer"),
        ),
        ({"state": "held", **APPROVAL}, True, Stage("stopping")),
        (
            {"state": "done", **APPROVAL, "ask": {"on": "person"}},
            True,
            Stage("finished"),
        ),
    ],
)
def test_the_first_row_that_holds_decides_the_stage(
    task: dict[str, Any], stopping: bool, stage: Stage
) -> None:
    assert task_stage(task, stopping=stopping) == stage
    assert task_stage(task, stopping=stopping).label
