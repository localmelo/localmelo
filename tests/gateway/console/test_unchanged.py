"""The legacy surface stays as it was: ``/`` and the ``/v1`` routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localmelo.support.gateway import GatewayServer
from localmelo.support.gateway.console import ConsoleOptions

from .conftest import make_cfg, quiet_gate


class QuietProbe:
    def snapshot(self) -> None:
        return None

    async def aclose(self) -> None:
        pass


def build(tmp_path: Path, *, host: str = "127.0.0.1") -> GatewayServer:
    cfg = make_cfg()
    cfg.gateway.host = host
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
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
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
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    assert client.get("/v1/health").content == (
        b'{"status":"ok","uptime":0.0,"sessions":{"active":0,"busy":0,"max":50},'
        b'"chat_backend":"mlc","embedding_backend":"none"}'
    )


def test_v1_sessions_is_byte_identical(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    assert client.get("/v1/sessions").content == b'{"sessions":[]}'


def test_v1_write_rejections_keep_the_legacy_bodies(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
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


def test_the_api_description_pages_are_off(tmp_path: Path) -> None:
    server = build(tmp_path)
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_a_non_loopback_gateway_has_no_api(tmp_path: Path) -> None:
    server = build(tmp_path, host="0.0.0.0")
    assert server.console is None
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    assert client.get("/api/health").status_code == 404
    assert client.get("/v1/health").status_code == 200
