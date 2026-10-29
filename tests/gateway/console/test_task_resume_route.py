"""HTTP shape of the controlled Resume route and its error codes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.support.gateway.console import Console
from localmelo.support.gateway.resume_evidence import (
    ConfirmationRequiredError,
    RecoveryUnknownError,
    ResumeRefusedError,
    StaleConfirmationError,
)
from localmelo.support.gateway.resume_tokens import ResumeBinding
from localmelo.support.gateway.task_store import TaskNotFoundError


class FakeResumeWorkers:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    async def resume(
        self,
        task_id: str,
        *,
        operation_id: str,
        confirmation_token: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((task_id, operation_id, confirmation_token))
        if task_id == "missing":
            raise TaskNotFoundError("No task")
        if operation_id == "preview":
            raise ConfirmationRequiredError(
                ResumeBinding(task_id, "exec-1", "evidence", 4, None),
                token="single-use-token",
            )
        if operation_id == "stale":
            raise StaleConfirmationError("The recovery evidence changed.")
        if operation_id == "unknown":
            raise RecoveryUnknownError()
        if operation_id == "blocked":
            raise ResumeRefusedError(
                "Resume unavailable: recovery information is missing. "
                "Start again as a new task.",
                reason="missing_card",
            )
        return {"task_id": task_id, "state": "queued"}


@pytest.fixture
def workers(
    console: Console, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeResumeWorkers]:
    fake = FakeResumeWorkers()
    original = console.workers
    monkeypatch.setattr(console, "workers", fake)
    yield fake
    console.workers = original


def test_resume_accepted_returns_202_task_and_operation(
    client: TestClient, workers: FakeResumeWorkers
) -> None:
    response = client.post(
        "/api/tasks/task1/resume",
        json={"operation_id": "op-1", "confirmation_token": "tok"},
    )
    assert response.status_code == 202
    assert response.json() == {
        "task": {"task_id": "task1", "state": "queued"},
        "operation_id": "op-1",
    }
    assert workers.calls[-1] == ("task1", "op-1", "tok")


def test_versioned_alias_reaches_the_same_handler(
    client: TestClient, workers: FakeResumeWorkers
) -> None:
    response = client.post("/api/v2/tasks/task1/resume", json={"operation_id": "op-1"})
    assert response.status_code == 202
    assert workers.calls[-1] == ("task1", "op-1", None)


def test_confirmation_required_carries_one_token_and_the_warning(
    client: TestClient, workers: FakeResumeWorkers
) -> None:
    response = client.post("/api/tasks/task1/resume", json={"operation_id": "preview"})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "confirmation_required"
    assert error["details"]["confirmation_token"] == "single-use-token"
    assert error["details"]["expires_in_s"] == 300
    assert error["details"]["reason"] == "unverified_processes"
    assert "may still be running" in error["details"]["warning"]


@pytest.mark.parametrize(
    "operation_id,status,code,reason",
    [
        ("stale", 409, "stale_confirmation", None),
        ("unknown", 409, "recovery_unknown", "unresolved_operation"),
        ("blocked", 409, "invalid_transition", "missing_card"),
        ("missing", 404, "not_found", None),
    ],
)
def test_resume_errors_use_the_api_envelope(
    client: TestClient,
    workers: FakeResumeWorkers,
    operation_id: str,
    status: int,
    code: str,
    reason: str | None,
) -> None:
    response = client.post(
        f"/api/tasks/{'missing' if operation_id == 'missing' else 'task1'}/resume",
        json={"operation_id": operation_id},
    )
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["message"]
    if reason is not None:
        assert error["details"]["reason"] == reason


@pytest.mark.parametrize(
    "body",
    [{}, {"operation_id": 1}, {"operation_id": "op", "extra": True}],
)
def test_invalid_resume_body_is_400(
    client: TestClient, workers: FakeResumeWorkers, body: dict[str, Any]
) -> None:
    response = client.post("/api/tasks/task1/resume", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert not workers.calls
