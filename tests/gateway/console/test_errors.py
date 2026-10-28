"""The API v2 error envelope, and the legacy routes it leaves alone."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.errors import INTERNAL_ERROR


def test_an_unknown_api_route_is_the_envelope(client: TestClient) -> None:
    resp = client.get("/api/no/such/route")
    assert resp.status_code == 404
    assert resp.headers["cache-control"] == "no-store"
    assert resp.json() == {
        "error": {"code": "not_found", "message": "Not found.", "details": {}}
    }


def test_the_api_root_is_the_envelope(client: TestClient) -> None:
    resp = client.get("/api")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_a_bad_query_parameter_is_a_400(client: TestClient) -> None:
    resp = client.get("/api/sessions?limit=999")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "limit" in resp.json()["error"]["message"]
    assert "999" not in resp.json()["error"]["message"]


def test_a_bad_kind_is_a_400(client: TestClient) -> None:
    resp = client.get("/api/sessions?kind=nope")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "kind" in resp.json()["error"]["message"]


def test_an_unhandled_api_error_is_the_500_envelope(console: Console) -> None:
    app = FastAPI()

    @app.get("/api/boom")
    async def boom() -> None:
        raise RuntimeError("a secret value")

    console.mount(app)
    client = TestClient(app, base_url="http://127.0.0.1:8401")
    resp = client.get("/api/boom")
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {
            "code": "internal_error",
            "message": INTERNAL_ERROR,
            "details": {},
        }
    }
    assert "a secret value" not in resp.text


def test_a_known_path_with_the_wrong_method_is_the_envelope(client: TestClient) -> None:
    resp = client.post("/api/health")
    assert resp.status_code == 405
    assert resp.json() == {
        "error": {
            "code": "method_not_allowed",
            "message": "Method not allowed.",
            "details": {},
        }
    }
    assert "GET" in resp.headers["allow"]


@pytest.mark.parametrize("method", ["TRACE", "CONNECT"])
def test_an_unusual_method_on_a_known_path_is_the_envelope(
    client: TestClient, method: str
) -> None:
    resp = client.request(method, "/api/health")
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in resp.headers["allow"]


def test_an_unknown_path_with_any_method_is_404(client: TestClient) -> None:
    assert client.post("/api/no/such/route").status_code == 404
    assert client.post("/api/no/such/route").json()["error"]["code"] == "not_found"


def test_a_legacy_404_keeps_fastapis_body(client: TestClient) -> None:
    resp = client.get("/no/such/route")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_a_legacy_422_keeps_fastapis_body(console: Console) -> None:
    from localmelo.support.gateway import GatewayServer
    from localmelo.support.gateway.console import ConsoleOptions

    from .conftest import make_cfg

    server = GatewayServer(
        make_cfg(),
        console_options=ConsoleOptions(
            store_path=console.store.path,
            builder=console.options.builder,
            probe=console.options.probe,
        ),
    )
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    resp = client.post("/v1/agent/run", json=["not", "an", "object"])
    assert resp.status_code == 422
    assert "detail" in resp.json()
    assert "error" not in resp.json()


@pytest.mark.parametrize(
    "path",
    [
        "/api/sessions",
        "/api/sessions/0123456789ab",
        "/api/sessions/0123456789ab/turns",
        "/api/sessions/0123456789ab/events",
        "/api/sessions/0123456789ab/blobs/" + "0" * 64,
        "/api/sessions/0123456789ab/submissions/req-1",
        "/api/submissions/req-1",
    ],
)
async def test_every_store_route_is_503_when_the_store_failed(
    client: TestClient, console: Console, path: str
) -> None:
    console.store._failed = RuntimeError("The disk is gone")  # type: ignore[attr-defined]
    resp = client.get(path)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "store_unavailable"
    assert resp.json()["error"]["message"]
