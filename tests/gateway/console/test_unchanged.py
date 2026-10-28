"""The legacy surface stays as it was: ``/`` and the ``/v1`` routes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localmelo.support.gateway import GatewayServer
from localmelo.support.gateway.console import ConsoleOptions
from localmelo.support.gateway.session import Session

from .conftest import make_cfg, quiet_gate


class QuietProbe:
    def snapshot(self) -> None:
        return None

    async def aclose(self) -> None:
        pass


class QuietAgent:
    """A manager session's Agent, for the routes that only close it."""

    async def run(self, query: str) -> str:
        return ""

    async def close(self) -> None:
        pass


def build(
    tmp_path: Path, *, host: str = "127.0.0.1", playground: bool = False
) -> GatewayServer:
    cfg = make_cfg()
    cfg.gateway.host = host
    cfg._enable_playground = playground
    return GatewayServer(
        cfg,
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            probe=QuietProbe(),
            gate=quiet_gate(),
        ),
    )


def test_the_workspace_page_is_the_same_file(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    resp = client.get("/")
    assert resp.status_code == 200
    static = Path(__file__).parents[3] / "support" / "gateway" / "static" / "index.html"
    assert resp.content == static.read_bytes()


def test_v1_health_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fixed clock makes the no-lifespan client's uptime exactly 0.0, so
    # the whole body can be compared as the bytes a client receives.
    monkeypatch.setattr("localmelo.support.gateway.time.time", lambda: 1000.0)
    server = build(tmp_path)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    assert client.get("/v1/health").content == (
        b'{"status":"ok","uptime":0.0,"sessions":{"active":0,"busy":0,"max":50},'
        b'"chat_backend":"mlc","embedding_backend":"none"}'
    )


def test_v1_sessions_is_byte_identical(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    assert client.get("/v1/sessions").content == b'{"sessions":[]}'


def test_v1_write_rejections_keep_the_legacy_bodies(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    # FastAPI's own 422 body, not API v2's envelope.
    refused = client.post(
        "/v1/agent/run",
        content="not json",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert refused.status_code == 422
    assert refused.content == (
        b'{"detail":[{"type":"dict_type","loc":["body"],"msg":"Input should be '
        b'a valid dictionary","input":"not json"}]}'
    )
    assert client.delete("/v1/sessions/nope").content == (
        b'{"error":"session not found"}'
    )


def test_v1_agent_run_rejection_is_byte_identical(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    # The checker's own 400 body, not API v2's envelope.
    assert client.post("/v1/agent/run", json={"query": ""}).content == (
        b'{"error":"Query must be a non-empty string"}'
    )


def test_v1_close_is_byte_identical(tmp_path: Path) -> None:
    server = build(tmp_path)
    server.sessions._sessions["a1b2c3d4e5f6"] = Session(agent=QuietAgent())
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )

    assert client.delete("/v1/sessions/a1b2c3d4e5f6").content == (
        b'{"closed":"a1b2c3d4e5f6"}'
    )
    assert client.get("/v1/sessions").content == b'{"sessions":[]}'


def test_v1_playground_routes_are_byte_identical(tmp_path: Path) -> None:
    server = build(tmp_path, playground=True)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    # The scenario list is static, so its digest pins the playground's bytes.
    scenarios = client.get("/v1/playground/scenarios")
    assert hashlib.sha256(scenarios.content).hexdigest() == (
        "691a4c50b0b6c5b08cac8129ca1b0622627789618b1f1111dfc3ce618f70d752"
    )
    assert client.get("/v1/playground/session/nope").content == (
        b'{"error":"session not found"}'
    )


def test_the_api_description_pages_are_off(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_a_non_loopback_gateway_has_no_api(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        build(tmp_path, host="0.0.0.0")
