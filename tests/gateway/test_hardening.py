from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway import GatewayServer


def test_the_gateway_has_no_playground_routes() -> None:
    cfg = Config(chat_backend="mlc", embedding_backend="none")
    cfg.mlc = LocalBackendConfig(chat_url="http://127.0.0.1:1/v1", chat_model="saved")
    server = GatewayServer(cfg)
    token = server.auth.read_token().decode("ascii")
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + token},
    )
    assert client.get("/v1/playground/scenarios").status_code == 404
    assert client.post("/v1/playground/session", json={}).status_code == 404
    assert client.get("/").status_code == 200
    server.auth.close()


def test_the_removed_playground_flag_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from localmelo.__main__ import _build_parser, _check_args

    parser = _build_parser()
    _check_args(parser, parser.parse_args(["--serve", "--enable-playground"]))
    assert "--enable-playground was removed" in capsys.readouterr().err
