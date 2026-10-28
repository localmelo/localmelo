from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway import GatewayServer


def _server(enabled: bool) -> tuple[GatewayServer, TestClient]:
    cfg = Config(chat_backend="mlc", embedding_backend="none")
    cfg.mlc = LocalBackendConfig(chat_url="http://127.0.0.1:1/v1", chat_model="saved")
    cfg._enable_playground = enabled
    server = GatewayServer(cfg)
    token = server.auth.read_token().decode("ascii")
    return server, TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + token},
    )


def test_playground_is_disabled_without_creating_manager() -> None:
    server, client = _server(False)
    assert not hasattr(server.app.state, "smoke_playground")
    assert client.get("/v1/playground/scenarios").status_code == 404
    assert client.get("/").status_code == 200
    server.auth.close()


def test_enabled_playground_requires_auth_and_saved_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, client = _server(True)
    manager = server.app.state.smoke_playground
    captured = {}

    async def discover(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"models": []}

    monkeypatch.setattr(manager, "discover", discover)
    unauthenticated = TestClient(server.app, base_url="http://127.0.0.1:8401")
    assert unauthenticated.get("/v1/playground/scenarios").status_code == 401
    assert (
        client.post(
            "/v1/playground/discover", json={"chat_url": "http://attacker.invalid"}
        ).status_code
        == 400
    )
    assert not captured
    assert client.post("/v1/playground/discover", json={}).status_code == 200
    assert captured["chat_url"] == "http://127.0.0.1:1/v1"
    assert (
        client.post(
            "/v1/playground/session", json={"chat_model": "override"}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/v1/playground/scenarios", headers={"Host": "attacker.invalid"}
        ).status_code
        == 421
    )
    server.auth.close()
