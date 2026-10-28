"""HTTP task routes use the console's error and response shapes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.errors import ApiError
from localmelo.support.gateway.task_policy import WorkspaceError
from localmelo.support.gateway.task_store import TaskConflictError, TaskNotFoundError


class FakeWorkers:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.changes: list[tuple[str, Any, Any]] = []

    async def create(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        self.payloads.append(payload)
        if payload.get("workspace") == "/":
            raise WorkspaceError("invalid workspace")
        if payload.get("request_key") == "conflict":
            raise TaskConflictError("different task")
        return {"task_id": "task1", "state": "queued"}, payload.get(
            "request_key"
        ) == "repeat"

    async def list(
        self, *, state: str | None, level: int | None
    ) -> list[dict[str, Any]]:
        return [{"state": state, "level": level}]

    async def get(self, task_id: str) -> dict[str, Any]:
        if task_id == "missing":
            raise TaskNotFoundError("No task")
        return {"task_id": task_id}

    async def update(
        self, task_id: str, *, level: object = None, state: str | None = None
    ) -> dict[str, Any]:
        self.changes.append((task_id, level, state))
        return {"task_id": task_id, "level": level, "state": state}


@pytest.fixture
def workers(console: Console, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeWorkers]:
    fake = FakeWorkers()
    original = console.workers
    monkeypatch.setattr(console, "workers", fake)
    yield fake
    console.workers = original


def test_create_task_and_idempotent_response(
    client: TestClient, workers: FakeWorkers
) -> None:
    response = client.post("/api/tasks", json={"prompt": "summarize", "level": "P1"})
    assert response.status_code == 202
    assert response.json() == {"task": {"task_id": "task1", "state": "queued"}}
    assert workers.payloads[-1] == {"prompt": "summarize", "level": "P1"}
    repeated = client.post(
        "/api/tasks", json={"prompt": "summarize", "request_key": "repeat"}
    )
    assert repeated.status_code == 200 and repeated.json()["repeated"] is True


def test_get_and_filter_tasks(client: TestClient, workers: FakeWorkers) -> None:
    assert client.get("/api/tasks?state=running&level=1").json() == {
        "tasks": [{"state": "running", "level": 1}]
    }
    assert client.get("/api/tasks/task1").json() == {"task": {"task_id": "task1"}}
    missing = client.get("/api/tasks/missing")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("state", ["pause", "resume", "stop"])
def test_patch_task_control(
    client: TestClient, workers: FakeWorkers, state: str
) -> None:
    level = "P1" if state == "resume" else None
    body = {"state": state, **({"level": level} if level else {})}
    response = client.patch("/api/tasks/task1", json=body)
    assert response.status_code == 200
    assert workers.changes[-1] == ("task1", level, state)


async def test_task_sessions_cannot_be_run_or_deleted_as_chats(
    console: Console, workspace: str, client: TestClient
) -> None:
    session = await console.store.create_session(kind="task", workspace=workspace)
    async with console.sessions.lock(session.id):
        with pytest.raises(ApiError, match="task"):
            await console.sessions.open(session.id)
    with pytest.raises(ApiError, match="task"):
        await console.sessions.delete(session.id)
    with pytest.raises(ApiError, match="task"):
        await console.sessions.change_workspace(session.id, "/")
    for body in ({"workspace": "/"}, {"title": "other"}):
        response = client.patch(f"/api/sessions/{session.id}", json=body)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "task_session_read_only"
    info = await console.store.get_session(session.id)
    assert info is not None and info.workspace == workspace


def test_real_task_api_idempotency_and_stop(
    client: TestClient, workspace: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    payload = {
        "objective": "later",
        "workspace": workspace,
        "level": "P5",
        "request_key": "real",
    }
    created = client.post("/api/tasks", json=payload)
    assert created.status_code == 202
    task_id = created.json()["task"]["task_id"]
    repeated = client.post("/api/tasks", json=payload)
    assert repeated.status_code == 200 and repeated.json()["task"]["task_id"] == task_id
    assert client.get(f"/api/tasks/{task_id}").json()["task"]["state"] == "parked"
    stopped = client.patch(f"/api/tasks/{task_id}", json={"state": "stop"})
    assert stopped.status_code == 200 and stopped.json()["task"]["state"] == "cancelled"


def test_real_task_control_errors(
    client: TestClient,
    console: Console,
    workspace: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    async def no_schedule() -> None:
        pass

    monkeypatch.setattr(console.workers, "_schedule", no_schedule)
    created = client.post(
        "/api/tasks", json={"objective": "later", "workspace": workspace, "level": "P2"}
    )
    assert created.status_code == 202
    task_id = created.json()["task"]["task_id"]
    url = f"/api/tasks/{task_id}"
    for body, status, code in (
        ({"state": "resume"}, 409, "not_paused"),
        ({"level": "first"}, 400, "invalid_request"),
        ({"state": "nonsense"}, 400, "invalid_request"),
    ):
        response = client.patch(url, json=body)
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
    assert client.patch(url, json={"state": "stop"}).status_code == 200
    response = client.patch(url, json={"state": "stop"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_transition"


def test_real_task_workspace_symlink_swap(
    client: TestClient, workspace: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = Path(workspace)
    created = client.post(
        "/api/tasks", json={"objective": "later", "workspace": workspace, "level": "P5"}
    )
    assert created.status_code == 202
    task_id = created.json()["task"]["task_id"]
    path.rename(tmp_path / "original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    path.symlink_to(replacement, target_is_directory=True)
    response = client.patch(f"/api/tasks/{task_id}", json={"level": "P2"})
    assert response.status_code == 200
    assert response.json()["task"]["state"] == "failed"
    assert response.json()["task"]["wait_reason"] == "invalid_workspace"


def test_task_errors_use_api_shape(client: TestClient, workers: FakeWorkers) -> None:
    workspace = client.post("/api/tasks", json={"prompt": "x", "workspace": "/"})
    assert (
        workspace.status_code == 400
        and workspace.json()["error"]["code"] == "invalid_workspace"
    )
    conflict = client.post(
        "/api/tasks", json={"prompt": "x", "request_key": "conflict"}
    )
    assert (
        conflict.status_code == 409
        and conflict.json()["error"]["code"] == "submission_conflict"
    )
    for body in ({}, {"level": True}, {"state": [], "other": 1}):
        response = client.patch("/api/tasks/task1", json=body)
        assert response.status_code == 400 and "error" in response.json()
