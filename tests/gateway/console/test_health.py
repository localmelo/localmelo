"""``GET /api/health``: shape, the store's state and the cached model probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.schema import SESSION_TEXT_CHARS
from localmelo.support.gateway.console import Console, ConsoleOptions

from .conftest import FakeBuilder, FakeProbe, make_cfg, open_session, seed_turn


def _client(console: Console) -> TestClient:
    app = FastAPI()
    console.mount(app)
    return TestClient(app, base_url="http://127.0.0.1:8401")


async def test_health_shape(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["ok"] is True
    assert body["api"] == 2
    assert body["events_schema"] == 1
    assert isinstance(body["version"], str) and body["version"]
    assert body["configured"] is True
    assert body["config_errors"] == []
    assert body["model"] == {
        "backend": "mlc",
        "backend_label": "MLC-LLM",
        "name": "stub-model",
        "embedding": "none",
    }
    assert body["model_server"] is None
    assert body["limits"] == {
        "text_chars": MAX_QUERY_LEN,
        "attachment_budget_chars": SESSION_TEXT_CHARS,
    }
    assert body["sessions"] == {
        "open": 0,
        "running": 0,
        "pending_approvals": 0,
        "store": "ok",
        "store_error": "",
        "resume_mode": "pairs",
    }
    assert body["uptime_s"] >= 0


async def test_health_reports_the_cached_probe(
    client: TestClient, probe: FakeProbe
) -> None:
    probe.state = {
        "url_label": "http://127.0.0.1:9/v1",
        "reachable": False,
        "error": "connection refused",
        "checked_s_ago": 4,
    }
    body = client.get("/api/health").json()
    assert body["model_server"] == probe.state


async def test_health_does_not_require_the_store(
    client: TestClient, console: Console
) -> None:
    console.store._failed = RuntimeError("The disk is gone")  # type: ignore[attr-defined]
    body = client.get("/api/health").json()
    assert body["sessions"]["store"] == "failed"
    assert body["sessions"]["store_error"] == "The disk is gone"


async def test_health_counts_open_sessions(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    await open_session(console.sessions, turn.session_id)
    body = client.get("/api/health").json()
    assert body["sessions"]["open"] == 1
    assert body["sessions"]["running"] == 0


async def test_a_store_that_could_not_open_is_unavailable(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    first = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), builder=builder, probe=probe
        ),
    )
    second = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), builder=builder, probe=probe
        ),
    )
    await first.start()
    await second.start()
    try:
        client = _client(second)
        health = client.get("/api/health")
        sessions = client.get("/api/sessions")
        assert health.status_code == 200
        body: dict[str, Any] = health.json()
        assert body["sessions"]["store"] == "unavailable"
        assert body["sessions"]["store_error"]
        assert sessions.status_code == 503
        assert sessions.json()["error"]["code"] == "store_unavailable"
    finally:
        await second.aclose()
        await first.aclose()
