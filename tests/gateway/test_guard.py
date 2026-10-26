"""The gateway refuses requests that are not addressed to it.

A web page can reach a gateway on 127.0.0.1 through the browser: through
DNS rebinding, with the page's own name in the Host header, or as a
cross-origin request, with the page's Origin. These tests send such
requests to the real gateway app, whose sessions and playground are
fakes that record what runs, and send what a test client cannot (no
Host, a repeated header) to LoopbackGuard directly.

Requires fastapi + uvicorn (gateway extra). Collection-safe without either.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import FastAPI, WebSocket  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.types import Message, Receive, Scope, Send  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from localmelo.support.config import (  # noqa: E402
    Config,
    GatewayConfig,
    LocalBackendConfig,
)
from localmelo.support.gateway import GatewayServer  # noqa: E402
from localmelo.support.gateway.guard import LoopbackGuard, is_loopback  # noqa: E402
from localmelo.support.gateway.session import AgentLike, SessionManager  # noqa: E402

# mount() registers its shutdown hook with FastAPI's deprecated on_event.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:\s*on_event is deprecated:DeprecationWarning"
)

PORT = 8401
GATEWAY = f"http://127.0.0.1:{PORT}"
FOREIGN_HOST = f"evil.example:{PORT}"
FOREIGN_ORIGIN = f"http://evil.example:{PORT}"

HOST_ERROR = {
    "error": (
        f"Host header must be 127.0.0.1:{PORT}, localhost:{PORT} or [::1]:{PORT}"
    ),
    "code": "misdirected_request",
}
ORIGIN_ERROR = {
    "error": (
        f"Cross-origin request refused: only pages at http://127.0.0.1:{PORT}, "
        f"http://localhost:{PORT} or http://[::1]:{PORT} may send it"
    ),
    "code": "foreign_origin",
}

# A task that RecordingAgent records, then fails, so the route answers 500.
FAILING_TASK = "fail"

# Every route of the gateway, requests that would end in 404, 422 or 500,
# and FastAPI's API description.
ROUTES = [
    pytest.param("GET", "/", None, id="web-ui"),
    pytest.param("GET", "/v1/health", None, id="health"),
    pytest.param("GET", "/v1/sessions", None, id="sessions"),
    pytest.param("DELETE", "/v1/sessions/s1", None, id="close-session"),
    pytest.param("POST", "/v1/agent/run", {"query": "ls"}, id="agent-run"),
    pytest.param("GET", "/v1/playground/scenarios", None, id="playground-scenarios"),
    pytest.param(
        "POST",
        "/v1/playground/discover",
        {"chat_url": "http://evil.example/v1"},
        id="playground-discover",
    ),
    pytest.param(
        "POST",
        "/v1/playground/session",
        {"chat_url": "http://evil.example/v1", "chat_model": "m"},
        id="playground-session",
    ),
    pytest.param(
        "POST", "/v1/playground/session/pg1/run", {"query": "ls"}, id="playground-run"
    ),
    pytest.param("DELETE", "/v1/playground/session/pg1", None, id="playground-close"),
    pytest.param("GET", "/no/such/route", None, id="404"),
    pytest.param("POST", "/v1/agent/run", ["not", "an", "object"], id="422"),
    pytest.param("POST", "/v1/agent/run", {"query": FAILING_TASK}, id="500"),
    pytest.param("GET", "/openapi.json", None, id="openapi"),
]

ORIGIN_CHECKED_METHODS = ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingAgent:
    """Records each task and echoes it; contacts no model server."""

    def __init__(self, runs: list[str]) -> None:
        self._runs = runs

    async def run(self, query: str) -> str:
        self._runs.append(query)
        if query == FAILING_TASK:
            raise RuntimeError("boom")
        return f"echo: {query}"

    async def close(self) -> None:
        pass


class RecordingSessions(SessionManager):
    """SessionManager whose agents record the tasks they run."""

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[str] = []

    def _create_agent(self) -> AgentLike:
        return RecordingAgent(self.runs)


class RecordingPlayground:
    """Stands in for SmokePlayground: records each call, contacts no server."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def scenarios_payload(self) -> list[dict[str, Any]]:
        self.calls.append("scenarios")
        return []

    async def discover(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("discover")
        return {"adapter": "mlc", "chat_models": [], "embedding_models": []}

    async def create_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("create_session")
        return {"session_id": "pg1"}

    async def run(self, session_id: str, query: str) -> dict[str, Any]:
        self.calls.append("run")
        return {"result": f"echo: {query}"}

    async def close(self, session_id: str) -> bool:
        self.calls.append("close")
        return True

    async def close_all(self) -> None:
        pass


@dataclass
class Gateway:
    server: GatewayServer
    sessions: RecordingSessions
    playground: RecordingPlayground
    client: TestClient

    def ran_nothing(self) -> bool:
        return not self.sessions.runs and not self.playground.calls


def _build_gateway(
    monkeypatch: pytest.MonkeyPatch, *, host: str = "127.0.0.1", port: int = PORT
) -> Gateway:
    """The real gateway app with the web UI, its sessions and playground faked."""
    playground = RecordingPlayground()
    monkeypatch.setattr(
        "localmelo.support.gateway.webapp.SmokePlayground", lambda: playground
    )
    cfg = Config(
        chat_backend="mlc",
        embedding_backend="mlc",
        mlc=LocalBackendConfig(
            chat_url="http://llm.test/v1",
            chat_model="m",
            embedding_url="http://llm.test/v1",
            embedding_model="e",
        ),
        gateway=GatewayConfig(port=port, host=host),
    )
    server = GatewayServer(cfg)
    sessions = RecordingSessions()
    server.sessions = sessions
    client = TestClient(
        server.app, base_url=f"http://127.0.0.1:{port}", raise_server_exceptions=False
    )
    return Gateway(server, sessions, playground, client)


@pytest.fixture()
def gateway(monkeypatch: pytest.MonkeyPatch) -> Gateway:
    return _build_gateway(monkeypatch)


class InnerApp:
    """An ASGI app that answers 200 and records the scopes it gets."""

    def __init__(self) -> None:
        self.scopes: list[Scope] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.scopes.append(scope)
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})


async def _call_guard(
    scope: Scope, *, port: int = PORT
) -> tuple[list[Message], InnerApp]:
    """Send *scope* through a LoopbackGuard; return what it sent and its app."""
    inner = InnerApp()
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await LoopbackGuard(inner, port=port)(scope, receive, send)
    return sent, inner


def _http_scope(headers: list[tuple[bytes, bytes]], *, method: str = "POST") -> Scope:
    return {
        "type": "http",
        "method": method,
        "path": "/v1/agent/run",
        "headers": headers,
    }


# ---------------------------------------------------------------------------
# is_loopback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "localhost", "LocalHost", "::1", "[::1]"]
)
def test_loopback_hosts(host: str) -> None:
    assert is_loopback(host)


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.168.1.20", "example.com", "localhost.example", ""]
)
def test_other_hosts_are_not_loopback(host: str) -> None:
    assert not is_loopback(host)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
def test_a_foreign_host_is_refused_on_every_route(
    gateway: Gateway, method: str, path: str, body: Any
) -> None:
    resp = gateway.client.request(
        method, path, json=body, headers={"Host": FOREIGN_HOST}
    )
    assert resp.status_code == 421
    assert resp.json() == HOST_ERROR
    assert gateway.ran_nothing()


@pytest.mark.parametrize(
    "host",
    [f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}", f"LOCALHOST:{PORT}"],
)
def test_the_gateway_host_passes(gateway: Gateway, host: str) -> None:
    assert gateway.client.get("/v1/health", headers={"Host": host}).status_code == 200
    resp = gateway.client.post(
        "/v1/agent/run", json={"query": "hi"}, headers={"Host": host}
    )
    assert resp.status_code == 200
    assert gateway.sessions.runs == ["hi"]


def test_error_responses_come_after_the_guard(gateway: Gateway) -> None:
    # The same requests as in ROUTES that a foreign Host gets 421 for.
    assert gateway.client.get("/no/such/route").status_code == 404
    resp = gateway.client.post("/v1/agent/run", json=["not", "an", "object"])
    assert resp.status_code == 422
    resp = gateway.client.post("/v1/agent/run", json={"query": FAILING_TASK})
    assert resp.status_code == 500
    assert resp.json()["error"] == "boom"


@pytest.mark.parametrize(
    "host",
    [
        f"127.0.0.1:{PORT + 1}",
        "127.0.0.1",
        "localhost",
        f"0.0.0.0:{PORT}",
        f"evil.localhost:{PORT}",
        f"localhost.evil.example:{PORT}",
        f"::1:{PORT}",
    ],
)
def test_another_host_or_port_is_refused(gateway: Gateway, host: str) -> None:
    resp = gateway.client.get("/v1/health", headers={"Host": host})
    assert resp.status_code == 421
    assert resp.json() == HOST_ERROR


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"host", b"")],
        [(b"host", b"127.0.0.1:8401"), (b"host", b"127.0.0.1:8401")],
        [(b"host", b"127.0.0.1:8401"), (b"host", b"evil.example:8401")],
    ],
    ids=["missing", "empty", "repeated", "repeated-foreign"],
)
async def test_a_missing_empty_or_repeated_host_is_refused(
    headers: list[tuple[bytes, bytes]],
) -> None:
    sent, inner = await _call_guard(_http_scope(headers))
    assert sent[0]["status"] == 421
    assert json.loads(sent[1]["body"]) == HOST_ERROR
    assert inner.scopes == []


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]", "LOCALHOST"])
def test_port_80_also_accepts_the_bare_host(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    gateway = _build_gateway(monkeypatch, port=80)
    for value in (host, f"{host}:80"):
        resp = gateway.client.get("/v1/health", headers={"Host": value})
        assert resp.status_code == 200
    resp = gateway.client.get("/v1/health", headers={"Host": f"{host}:{PORT}"})
    assert resp.status_code == 421


def test_the_host_is_checked_before_the_origin(gateway: Gateway) -> None:
    resp = gateway.client.post(
        "/v1/agent/run",
        json={"query": "ls"},
        headers={"Host": FOREIGN_HOST, "Origin": FOREIGN_ORIGIN},
    )
    assert resp.status_code == 421
    assert resp.json()["code"] == "misdirected_request"


# ---------------------------------------------------------------------------
# Origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ORIGIN_CHECKED_METHODS)
@pytest.mark.parametrize(
    "origin",
    [
        FOREIGN_ORIGIN,
        "null",
        f"http://127.0.0.1:{PORT + 1}",
        f"https://127.0.0.1:{PORT}",
        "http://localhost",
        f"http://localhost.evil.example:{PORT}",
    ],
)
def test_a_foreign_origin_is_refused(
    gateway: Gateway, method: str, origin: str
) -> None:
    resp = gateway.client.request(
        method, "/v1/agent/run", json={"query": "ls"}, headers={"Origin": origin}
    )
    assert resp.status_code == 403
    assert resp.json() == ORIGIN_ERROR
    assert gateway.ran_nothing()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/", None),
        ("/v1/playground/session", {"chat_url": "http://evil.example/v1"}),
        ("/v1/playground/session/pg1/run", {"query": "ls"}),
    ],
)
def test_a_foreign_origin_is_refused_on_any_path(
    gateway: Gateway, path: str, body: Any
) -> None:
    resp = gateway.client.post(path, json=body, headers={"Origin": FOREIGN_ORIGIN})
    assert resp.status_code == 403
    assert resp.json() == ORIGIN_ERROR
    assert gateway.ran_nothing()


@pytest.mark.parametrize(
    "origin",
    [
        GATEWAY,
        f"http://localhost:{PORT}",
        f"http://[::1]:{PORT}",
        f"http://LOCALHOST:{PORT}",
        None,
    ],
)
def test_the_gateway_origin_and_no_origin_pass(
    gateway: Gateway, origin: str | None
) -> None:
    headers = {} if origin is None else {"Origin": origin}
    resp = gateway.client.post("/v1/agent/run", json={"query": "hi"}, headers=headers)
    assert resp.status_code == 200
    assert gateway.sessions.runs == ["hi"]
    for method in ORIGIN_CHECKED_METHODS:
        # Methods without a route get 405 from the router, past the guard.
        resp = gateway.client.request(method, "/v1/health", headers=headers)
        assert resp.status_code == 405


@pytest.mark.parametrize("origin", [FOREIGN_ORIGIN, "null"])
def test_a_get_with_a_foreign_origin_passes_the_origin_rule(
    gateway: Gateway, origin: str
) -> None:
    for path in ("/", "/v1/health"):
        resp = gateway.client.get(path, headers={"Origin": origin})
        assert resp.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        [(b"origin", b"")],
        [(b"origin", GATEWAY.encode()), (b"origin", GATEWAY.encode())],
    ],
    ids=["empty", "repeated"],
)
async def test_an_empty_or_repeated_origin_is_foreign(
    headers: list[tuple[bytes, bytes]],
) -> None:
    host = [(b"host", f"127.0.0.1:{PORT}".encode())]
    sent, inner = await _call_guard(_http_scope(host + headers))
    assert sent[0]["status"] == 403
    assert json.loads(sent[1]["body"]) == ORIGIN_ERROR
    assert inner.scopes == []


def test_port_80_accepts_an_origin_without_the_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _build_gateway(monkeypatch, port=80)
    for origin in ("http://localhost", "http://127.0.0.1:80", "http://[::1]"):
        resp = gateway.client.post(
            "/v1/agent/run", json={"query": "hi"}, headers={"Origin": origin}
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Fetch metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/v1/health", None),
        ("GET", "/v1/sessions", None),
        ("POST", "/v1/agent/run", {"query": "ls"}),
        ("GET", "/v1/playground/scenarios", None),
    ],
)
def test_a_request_from_another_site_is_refused_under_v1(
    gateway: Gateway, site: str, method: str, path: str, body: Any
) -> None:
    resp = gateway.client.request(
        method, path, json=body, headers={"Sec-Fetch-Site": site}
    )
    assert resp.status_code == 403
    assert resp.json() == ORIGIN_ERROR
    assert gateway.ran_nothing()


@pytest.mark.parametrize("site", ["same-origin", "none"])
def test_a_request_from_the_gateway_page_or_the_user_passes(
    gateway: Gateway, site: str
) -> None:
    assert gateway.client.get("/v1/health", headers={"Sec-Fetch-Site": site}).is_success
    resp = gateway.client.post(
        "/v1/agent/run",
        json={"query": "hi"},
        headers={"Sec-Fetch-Site": site, "Origin": GATEWAY},
    )
    assert resp.status_code == 200
    assert gateway.sessions.runs == ["hi"]


def test_a_link_from_another_site_opens_the_web_ui(gateway: Gateway) -> None:
    resp = gateway.client.get(
        "/",
        headers={
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        },
    )
    assert resp.status_code == 200
    assert "localmelo playground" in resp.text


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "headers"),
    [
        ("GET", "/v1/health", {"Origin": GATEWAY}),
        ("POST", "/v1/agent/run", {"Origin": GATEWAY}),
        (
            "OPTIONS",
            "/v1/agent/run",
            {
                "Origin": GATEWAY,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        ),
        (
            "OPTIONS",
            "/v1/agent/run",
            {"Origin": FOREIGN_ORIGIN, "Access-Control-Request-Method": "POST"},
        ),
        ("POST", "/v1/agent/run", {"Origin": FOREIGN_ORIGIN}),
        (
            "GET",
            "/v1/health",
            {"Origin": FOREIGN_ORIGIN, "Sec-Fetch-Site": "cross-site"},
        ),
        ("GET", "/", {"Origin": FOREIGN_ORIGIN}),
        ("GET", "/v1/health", {"Host": FOREIGN_HOST}),
    ],
)
def test_no_response_carries_cors_headers(
    gateway: Gateway, method: str, path: str, headers: dict[str, str]
) -> None:
    body = {"query": "hi"} if method == "POST" else None
    resp = gateway.client.request(method, path, json=body, headers=headers)
    assert not [h for h in resp.headers if h.lower().startswith("access-control-")]


# ---------------------------------------------------------------------------
# Where the guard is installed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_a_loopback_host_installs_the_guard(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, host: str
) -> None:
    with caplog.at_level(logging.WARNING, logger="localmelo.gateway"):
        gateway = _build_gateway(monkeypatch, host=host)
    assert [m.cls for m in gateway.server.app.user_middleware] == [LoopbackGuard]
    assert not caplog.records


def test_a_non_loopback_host_has_no_guard_and_logs_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="localmelo.gateway"):
        gateway = _build_gateway(monkeypatch, host="0.0.0.0")

    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert "0.0.0.0" in message
    assert "no authentication" in message
    assert "DNS rebinding" in message
    assert 'host = "127.0.0.1"' in message

    assert gateway.server.app.user_middleware == []
    foreign = {
        "Host": FOREIGN_HOST,
        "Origin": FOREIGN_ORIGIN,
        "Sec-Fetch-Site": "cross-site",
    }
    assert gateway.client.get("/v1/health", headers=foreign).status_code == 200
    resp = gateway.client.post("/v1/agent/run", json={"query": "hi"}, headers=foreign)
    assert resp.status_code == 200


def test_the_guarded_gateway_starts_and_stops(gateway: Gateway) -> None:
    # Entering the client runs the app's lifespan through the guard.
    with gateway.client as client:
        assert gateway.server._started_at is not None
        assert client.get("/v1/health").status_code == 200


async def test_a_lifespan_scope_passes_through() -> None:
    sent, inner = await _call_guard({"type": "lifespan"})
    assert [scope["type"] for scope in inner.scopes] == ["lifespan"]
    assert sent == []


# ---------------------------------------------------------------------------
# Websockets
# ---------------------------------------------------------------------------


def _websocket_client() -> TestClient:
    """A client for a guarded app whose websocket at /v1/ws says hello."""
    app = FastAPI()

    @app.websocket("/v1/ws")
    async def hello(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return TestClient(LoopbackGuard(app, port=PORT))


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({"Host": FOREIGN_HOST}, "misdirected_request"),
        ({"Origin": FOREIGN_ORIGIN}, "foreign_origin"),
        ({"Origin": "null"}, "foreign_origin"),
        ({"Sec-Fetch-Site": "cross-site"}, "foreign_origin"),
    ],
)
def test_a_refused_websocket_handshake_is_closed(
    headers: dict[str, str], code: str
) -> None:
    client = _websocket_client()
    url = f"ws://127.0.0.1:{PORT}/v1/ws"
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect(url, headers=dict(headers)),
    ):
        pass
    assert refused.value.code == 1008
    assert refused.value.reason == code


@pytest.mark.parametrize(
    "headers", [{}, {"Origin": GATEWAY, "Sec-Fetch-Site": "same-origin"}]
)
def test_a_websocket_from_the_gateway_page_or_a_script_is_accepted(
    headers: dict[str, str],
) -> None:
    client = _websocket_client()
    with client.websocket_connect(
        f"ws://127.0.0.1:{PORT}/v1/ws", headers=dict(headers)
    ) as websocket:
        assert websocket.receive_text() == "hello"
