"""Gateway HTTP-level tests exercising real production routes.

Requires fastapi + uvicorn (gateway extra). Collection-safe without either.
"""

from __future__ import annotations

pytest = __import__("pytest")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from typing import Any  # noqa: E402
from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from localmelo.melo.checker.validators import MAX_QUERY_LEN  # noqa: E402
from localmelo.support.config import (  # noqa: E402
    Config,
    GatewayConfig,
    LocalBackendConfig,
)
from localmelo.support.gateway import GatewayServer  # noqa: E402
from localmelo.support.gateway.chat_gate import ChatGate  # noqa: E402
from localmelo.support.gateway.console import ConsoleOptions  # noqa: E402
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

    options = ConsoleOptions(
        # A gate with no call log: a test must not write the user's
        # ~/.cache/localmelo/scheduler.
        gate=ChatGate(policy="off", call_log_dir=None),
    )
    with patch("localmelo.support.gateway.webapp.mount", lambda app, cfg=None: None):
        server = GatewayServer(cfg, console_options=options)

    # Replace the real SessionManager with our fake
    server.sessions = sessions
    return server


def _client(server: GatewayServer) -> TestClient:
    """A client for *server*, addressed to it as the gateway's guard requires.

    Errors in routes are returned as 500 responses, not raised.
    """
    return TestClient(
        server.app,
        base_url=f"http://127.0.0.1:{server.port}",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
        raise_server_exceptions=False,
    )


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
    return _client(server)


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

    @pytest.mark.parametrize("session_id", [12345, ["abc"], {"id": "abc"}, True])
    def test_non_string_session_id_returns_400(
        self, client: TestClient, session_id: object
    ) -> None:
        resp = client.post(
            "/v1/agent/run",
            json={"query": "hi", "session_id": session_id},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == f"Invalid session_id format: {session_id!r}"

    def test_query_at_trimmed_limit_accepted(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={"query": "x" * MAX_QUERY_LEN})
        assert resp.status_code == 200
        assert resp.json()["result"] == "echo: " + "x" * MAX_QUERY_LEN

    def test_query_over_trimmed_limit_returns_400(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/run", json={"query": "x" * (MAX_QUERY_LEN + 1)})
        assert resp.status_code == 400
        assert resp.json() == {
            "error": f"Query too large: {MAX_QUERY_LEN + 1} chars "
            f"(max {MAX_QUERY_LEN})"
        }

    def test_query_limit_ignores_surrounding_whitespace(
        self, client: TestClient
    ) -> None:
        # Raw text is over the limit, but the trimmed query the agent runs fits.
        resp = client.post(
            "/v1/agent/run",
            json={"query": "  " + "x" * MAX_QUERY_LEN + "  "},
        )
        assert resp.status_code == 200
        assert resp.json()["result"] == "echo: " + "x" * MAX_QUERY_LEN

    def test_padded_query_over_trimmed_limit_returns_400(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/v1/agent/run",
            json={"query": "  " + "x" * (MAX_QUERY_LEN + 1) + "  "},
        )
        assert resp.status_code == 400
        assert resp.json() == {
            "error": f"Query too large: {MAX_QUERY_LEN + 1} chars "
            f"(max {MAX_QUERY_LEN})"
        }


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


@pytest.mark.asyncio
async def test_delete_busy_session_returns_409_and_keeps_it() -> None:
    import asyncio

    import httpx

    sessions = FakeSessionManager()
    server = _build_server(sessions)
    session = await sessions.create("busy-delete")
    agent: FakeAgent = session.agent  # type: ignore[assignment]
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(query: str) -> str:
        started.set()
        await release.wait()
        return f"echo: {query}"

    agent.run = blocked  # type: ignore[method-assign]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url=f"http://127.0.0.1:{server.port}",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    ) as client:
        run = asyncio.create_task(
            client.post(
                "/v1/agent/run", json={"session_id": "busy-delete", "query": "wait"}
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        resp = await client.delete("/v1/sessions/busy-delete")
        assert resp.status_code == 409
        body = resp.json()
        assert "busy-delete" in body["error"]
        assert body["session_id"] == "busy-delete"
        # The session and its run are untouched.
        assert sessions.get("busy-delete") is session
        assert not agent.closed
        assert session.status == "running"

        release.set()
        run_resp = await run
        assert run_resp.status_code == 200
        assert run_resp.json()["result"] == "echo: wait"
        assert session.status == "idle"
        assert await sessions.close("busy-delete")
        assert agent.closed


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
        fail_client = _client(fail_server)

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
        fail_client = _client(fail_server)

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


@pytest.mark.asyncio
async def test_cancelled_request_returns_session_to_idle() -> None:
    import asyncio

    import httpx

    sessions = FakeSessionManager()
    server = _build_server(sessions)
    session = await sessions.create("cancelled")
    started = asyncio.Event()

    async def blocked(query: str) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    session.agent.run = blocked  # type: ignore[method-assign]
    # Addressed to the gateway, as its guard requires (see _client); an
    # AsyncClient, unlike TestClient, can cancel the request it is sending.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url=f"http://127.0.0.1:{server.port}",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/agent/run", json={"session_id": "cancelled", "query": "wait"}
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        assert session.status == "running"
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert session.status == "idle"
        assert not session.is_busy


def test_gateway_lifespan_closes_playground(monkeypatch: Any) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    server = _build_server()
    order: list[str] = []

    async def sessions() -> None:
        order.append("sessions")

    async def gate_close() -> None:
        order.append("gate")

    gate = SimpleNamespace(close=AsyncMock(side_effect=gate_close))
    server.app.state.smoke_playground = SimpleNamespace(
        close_sessions=AsyncMock(side_effect=sessions), gate=gate
    )
    console = server.console
    assert console is not None
    aclose = console.aclose

    async def recording() -> None:
        order.append("console")
        await aclose()

    monkeypatch.setattr(console, "aclose", recording)
    with TestClient(server.app):
        assert order == []
    # The one shared gate closes last: after both surfaces' sessions.
    assert order == ["sessions", "console", "gate"]


def test_a_session_agent_keeps_its_answers() -> None:
    with patch("localmelo.support.gateway.session.Agent") as agent:
        SessionManager(config=_test_cfg())._create_agent()
    assert agent.call_args.kwargs["options"].record_answers
