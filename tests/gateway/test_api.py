"""Gateway HTTP-level tests exercising real production routes.

Requires fastapi + uvicorn (gateway extra). Collection-safe without either.
"""

from __future__ import annotations

pytest = __import__("pytest")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from localmelo.support.config import (  # noqa: E402
    Config,
    GatewayConfig,
    LocalBackendConfig,
)
from localmelo.support.gateway import GatewayServer  # noqa: E402
from localmelo.support.gateway.session import SessionManager  # noqa: E402

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal agent stand-in that echoes queries."""

    def __init__(self, *, fail: bool = False) -> None:
        self.closed = False
        self.run_count = 0
        self._fail = fail

    async def run(self, query: str) -> str:
        self.run_count += 1
        if self._fail:
            raise RuntimeError("boom")
        return f"echo: {query}"

    async def close(self) -> None:
        self.closed = True


class FakeSessionManager(SessionManager):
    """SessionManager that injects FakeAgents."""

    def __init__(
        self,
        *,
        fail_agent: bool = False,
        max_sessions: int = 50,
        idle_ttl: float = 3600.0,
    ) -> None:
        super().__init__(max_sessions=max_sessions, idle_ttl=idle_ttl)
        self._fail_agent = fail_agent

    def _create_agent(self) -> FakeAgent:  # type: ignore[override]
        return FakeAgent(fail=self._fail_agent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_cfg() -> Config:
    """Minimal valid Config for constructing a GatewayServer."""
    return Config(
        chat_backend="mlc",
        embedding_backend="mlc",
        mlc=LocalBackendConfig(
            chat_url="http://127.0.0.1:8400/v1",
            chat_model="Qwen3-1.7B",
            embedding_url="http://127.0.0.1:8400/v1",
            embedding_model="nomic-embed",
        ),
        gateway=GatewayConfig(port=8401, host="127.0.0.1"),
    )


def _build_server(
    sessions: FakeSessionManager | None = None,
) -> GatewayServer:
    """Build a GatewayServer with real routes but no webapp."""
    if sessions is None:
        sessions = FakeSessionManager()

    cfg = _test_cfg()

    with patch("localmelo.support.gateway.webapp.mount", lambda app: None):
        server = GatewayServer(cfg)

    # Replace the real SessionManager with our fake
    server.sessions = sessions
    return server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sessions() -> FakeSessionManager:
    return FakeSessionManager()


@pytest.fixture()
def server(sessions: FakeSessionManager) -> GatewayServer:
    return _build_server(sessions)


@pytest.fixture()
def client(server: GatewayServer) -> TestClient:
    # raise_server_exceptions=False so we can assert on 500 responses
    return TestClient(server.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /v1/agent/run
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_valid_query_returns_200(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={"query": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"] == "echo: hello"
        assert "session_id" in data

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={"query": ""})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_missing_query_returns_400(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={})
        assert resp.status_code == 400

    def test_whitespace_only_query_returns_400(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={"query": "   "})
        assert resp.status_code == 400

    def test_invalid_session_id_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/agent/run",
            json={"query": "hi", "session_id": "bad id!"},
        )
        assert resp.status_code == 400
        assert "session_id" in resp.json()["error"].lower()

    def test_too_long_session_id_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/agent/run",
            json={"query": "hi", "session_id": "a" * 65},
        )
        assert resp.status_code == 400

    def test_valid_session_id_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/agent/run",
            json={"query": "hi", "session_id": "my-session_01"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session_01"

    def test_query_whitespace_stripped(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={"query": "  hello  "})
        assert resp.status_code == 200
        assert resp.json()["result"] == "echo: hello"


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime" in data
        assert "sessions" in data
        assert "llm" not in data
        assert "chat_backend" in data
        assert data["chat_backend"] == "mlc"
        assert "embedding_backend" in data
        assert data["embedding_backend"] == "mlc"

    def test_health_session_counts(self, client: TestClient) -> None:
        client.post("/v1/agent/run", json={"query": "seed"})
        resp = client.get("/v1/health")
        data = resp.json()
        assert data["sessions"]["active"] == 1


# ---------------------------------------------------------------------------
# GET /v1/sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_empty_list(self, client: TestClient) -> None:
        resp = client.get("/v1/sessions")
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []

    def test_after_run(self, client: TestClient) -> None:
        client.post("/v1/agent/run", json={"query": "hi"})
        resp = client.get("/v1/sessions")
        listing = resp.json()["sessions"]
        assert len(listing) == 1
        assert listing[0]["status"] == "idle"


# ---------------------------------------------------------------------------
# DELETE /v1/sessions/{id}
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_existing_returns_200(self, client: TestClient) -> None:
        client.post(
            "/v1/agent/run",
            json={"query": "hi", "session_id": "to-delete"},
        )
        resp = client.delete("/v1/sessions/to-delete")
        assert resp.status_code == 200
        assert resp.json()["closed"] == "to-delete"

    def test_delete_missing_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/v1/sessions/nonexistent")
        assert resp.status_code == 404

    def test_deleted_session_gone_from_list(self, client: TestClient) -> None:
        client.post(
            "/v1/agent/run",
            json={"query": "hi", "session_id": "gone"},
        )
        client.delete("/v1/sessions/gone")
        resp = client.get("/v1/sessions")
        ids = {s["session_id"] for s in resp.json()["sessions"]}
        assert "gone" not in ids


# ---------------------------------------------------------------------------
# Session reuse
# ---------------------------------------------------------------------------


class TestSessionReuse:
    def test_same_session_id_reuses_session(
        self, sessions: FakeSessionManager, client: TestClient
    ) -> None:
        r1 = client.post(
            "/v1/agent/run",
            json={"query": "first", "session_id": "reuse-me"},
        )
        r2 = client.post(
            "/v1/agent/run",
            json={"query": "second", "session_id": "reuse-me"},
        )
        assert r1.json()["session_id"] == "reuse-me"
        assert r2.json()["session_id"] == "reuse-me"

        listing = client.get("/v1/sessions").json()["sessions"]
        assert len(listing) == 1

        s = sessions.get("reuse-me")
        assert s is not None
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        assert agent.run_count == 2


# ---------------------------------------------------------------------------
# Session lifecycle: idle -> running -> idle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_idle_after_successful_run(
        self, sessions: FakeSessionManager, client: TestClient
    ) -> None:
        resp = client.post(
            "/v1/agent/run",
            json={"query": "hello", "session_id": "lifecycle"},
        )
        assert resp.status_code == 200
        s = sessions.get("lifecycle")
        assert s is not None
        assert s.status == "idle"

    def test_idle_after_failed_run(self) -> None:
        """Agent.run() raises -> 500, session returns to idle."""
        fail_sessions = FakeSessionManager(fail_agent=True)
        fail_server = _build_server(fail_sessions)
        fail_client = TestClient(fail_server.app, raise_server_exceptions=False)

        resp = fail_client.post(
            "/v1/agent/run",
            json={"query": "trigger error", "session_id": "err-sess"},
        )
        assert resp.status_code == 500
        assert "boom" in resp.json()["error"]

        s = fail_sessions.get("err-sess")
        assert s is not None
        assert s.status == "idle"

    def test_session_usable_after_error(self) -> None:
        """After an error, the same session can be reused."""
        fail_sessions = FakeSessionManager(fail_agent=True)
        fail_server = _build_server(fail_sessions)
        fail_client = TestClient(fail_server.app, raise_server_exceptions=False)

        r1 = fail_client.post(
            "/v1/agent/run",
            json={"query": "fail", "session_id": "recover"},
        )
        assert r1.status_code == 500

        # Switch agent to non-failing
        s = fail_sessions.get("recover")
        assert s is not None
        s.agent = FakeAgent(fail=False)  # type: ignore[assignment]

        r2 = fail_client.post(
            "/v1/agent/run",
            json={"query": "ok now", "session_id": "recover"},
        )
        assert r2.status_code == 200
        assert r2.json()["result"] == "echo: ok now"


# ---------------------------------------------------------------------------
# GatewayServer delegates inference to external backends
# ---------------------------------------------------------------------------


class TestNoEmbeddedInference:
    """GatewayServer must not embed inference processes."""

    def test_no_llm_attribute(self) -> None:
        """GatewayServer must not have an llm attribute."""
        server = _build_server()
        assert not hasattr(server, "llm")

    def test_no_start_llm_server_method(self) -> None:
        """GatewayServer must not have _start_llm_server."""
        assert not hasattr(GatewayServer, "_start_llm_server")
