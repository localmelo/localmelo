from __future__ import annotations

import asyncio
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localmelo.support.config import Config
from localmelo.support.gateway import GatewayServer
from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway.guard import LoopbackGuard


def test_login_codes_are_single_use_persistent_and_rotated(tmp_path: Path) -> None:
    path = tmp_path / "auth"
    auth = GatewayAuth(path)
    other = GatewayAuth(path)
    try:
        token = auth.read_token()
        assert len(token) == 64
        assert (path.stat().st_mode & 0o777) == 0o700
        assert ((path / "token").stat().st_mode & 0o777) == 0o600
        code = auth.issue_login(1234, now=100)
        assert auth.consume_login("².1234." + "a" * 32 + ".x", 1234, now=101) is None
        assert auth.consume_login("100.1234." + "a" * 32 + ".é", 1234, now=101) is None
        assert other.consume_login(code, 1235, now=101) is None
        cookie = other.consume_login(code, 1234, now=101)
        assert cookie and other.valid_cookie(cookie, 1234, now=102)
        assert not other.valid_cookie(cookie, 1235, now=102)
        assert cookie.encode() not in [
            p.read_bytes() for p in path.iterdir() if p.is_file()
        ]
        assert auth.consume_login(code, 1234, now=102) is None
        assert not other.valid_cookie(cookie, 1234, now=101 + 30 * 86400)
        expired = auth.issue_login(1234, now=100)
        assert other.consume_login(expired, 1234, now=160) is None
        # A code is bound to the instance it was issued for: the app's launch
        # identifier, or none for a gateway the user started directly.
        bound = auth.issue_login(1234, instance="f" * 32, now=100)
        assert other.consume_login(bound, 1234, now=101) is None
        assert other.consume_login(bound, 1234, instance="e" * 32, now=101) is None
        assert other.consume_login(bound, 1234, instance="f" * 32, now=101)
        auth.rotate()
        assert not other.valid_cookie(cookie, 1234, now=102)
        assert not auth.valid_bearer(token.decode())
    finally:
        auth.close()
        other.close()


def test_app_login_requires_the_launch_instance_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = "0123456789abcdef" * 2
    monkeypatch.setenv("LOCALMELO_APP_INSTANCE", instance)
    server = GatewayServer(Config())
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    try:
        assert client.get("/").headers["x-localmelo-instance"] == instance
        code = server.auth.issue_login(8401, instance=instance)
        origin = {"Origin": "http://127.0.0.1:8401"}
        # A code minted without the instance (as for a directly started
        # gateway) is not accepted by the app's gateway.
        unbound = server.auth.issue_login(8401)
        response = client.post(
            "/api/auth/login",
            json={"code": unbound},
            headers={**origin, "X-LocalMelo-Instance": instance},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        for headers in (
            origin,
            {**origin, "X-LocalMelo-Instance": "f" * 32},
        ):
            response = client.post(
                "/api/auth/login", json={"code": code}, headers=headers
            )
            assert response.status_code == 421
            assert response.json()["error"]["code"] == "misdirected_request"
        # A refused exchange must not consume the code.
        response = client.post(
            "/api/auth/login",
            json={"code": code},
            headers={**origin, "X-LocalMelo-Instance": instance},
        )
        assert response.status_code == 204
        assert client.get("/api/health").status_code == 200
    finally:
        server.auth.close()


def test_gateway_login_cookie_and_rotation() -> None:
    server = GatewayServer(Config())
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    page = client.get("/")
    assert page.status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/v1/health").status_code == 401
    assert client.get("/api/health").status_code == 401
    assert (
        client.get("/api/health", headers={"host": "foreign.example:8401"}).status_code
        == 421
    )
    code = server.auth.issue_login(8401)
    assert (
        client.post(
            "/api/auth/login",
            json={"code": code},
            headers={"Origin": "http://localhost:8401"},
        ).status_code
        == 403
    )
    response = client.post(
        "/api/auth/login",
        json={"code": code},
        headers={"Origin": "http://127.0.0.1:8401"},
    )
    assert response.status_code == 204
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert client.get("/api/health").status_code == 200
    assert (
        client.post(
            "/api/auth/login",
            json={"code": code},
            headers={"Origin": "http://127.0.0.1:8401"},
        ).status_code
        == 401
    )
    assert client.post("/api/auth/rotate", json={}).status_code == 401
    token = server.auth.read_token().decode("ascii")
    assert (
        client.post(
            "/api/auth/rotate", headers={"Authorization": "Bearer " + token}
        ).status_code
        == 204
    )
    assert client.get("/api/health").status_code == 401
    assert (
        client.get(
            "/api/health", headers={"Authorization": "Bearer " + token}
        ).status_code
        == 401
    )
    server.auth.close()


@pytest.mark.asyncio
async def test_unauthenticated_request_never_reads_body(tmp_path: Path) -> None:
    auth = GatewayAuth(tmp_path / "auth")
    called = False

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        raise AssertionError("app reached")

    async def receive():  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("body read")

    sent = []

    async def send(message):  # type: ignore[no-untyped-def]
        sent.append(message)

    await LoopbackGuard(app, port=8401, auth=auth)(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/health",
            "headers": [(b"host", b"127.0.0.1:8401")],
        },
        receive,
        send,
    )
    assert sent[0]["status"] == 401
    assert not called
    auth.close()


@pytest.mark.asyncio
async def test_rotation_ends_open_stream(tmp_path: Path) -> None:
    auth = GatewayAuth(tmp_path / "auth")
    started, closed = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def receive():  # type: ignore[no-untyped-def]
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):  # type: ignore[no-untyped-def]
        pass

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/sessions/a/events",
        "headers": [
            (b"host", b"127.0.0.1:8401"),
            (b"authorization", b"Bearer " + auth.read_token()),
        ],
    }
    task = asyncio.create_task(
        LoopbackGuard(app, port=8401, auth=auth)(scope, receive, send)
    )
    await asyncio.wait_for(started.wait(), 1)
    auth.rotate()
    await asyncio.wait_for(task, 1)
    assert closed.is_set()
    auth.close()


@pytest.mark.asyncio
async def test_rotation_revokes_two_gateways(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg1, cfg2 = Config(), Config()
    cfg1.gateway.port, cfg2.gateway.port = 8401, 8402
    first, second = GatewayServer(cfg1), GatewayServer(cfg2)
    started = [asyncio.Event(), asyncio.Event()]
    closed = [asyncio.Event(), asyncio.Event()]
    tasks = []
    cookies = []
    release = asyncio.Event()
    sent = []
    for index, server in enumerate((first, second)):
        port = server.port
        code = server.auth.issue_login(port)
        cookie = server.auth.consume_login(code, port)
        assert cookie is not None
        cookies.append(cookie)

        async def app(scope, receive, send, i=index):  # type: ignore[no-untyped-def]
            try:
                started[i].set()
                await release.wait()
                await send({"type": "http.response.body", "body": b"private event"})
                await asyncio.Event().wait()
            finally:
                closed[i].set()

        async def receive():  # type: ignore[no-untyped-def]
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):  # type: ignore[no-untyped-def]
            sent.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/sessions/a/events",
            "headers": [
                (b"host", f"127.0.0.1:{port}".encode()),
                (b"cookie", f"melo_session_{port}={cookie}".encode()),
            ],
        }
        tasks.append(
            asyncio.create_task(
                LoopbackGuard(app, port=port, auth=server.auth)(scope, receive, send)
            )
        )
    try:
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started)), 1)
        first.auth.rotate()
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert not sent
        assert all(e.is_set() for e in closed)
        assert not first.auth.valid_cookie(cookies[0], 8401)
        assert not second.auth.valid_cookie(cookies[1], 8402)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        first.auth.close()
        second.auth.close()


@pytest.mark.asyncio
async def test_rotation_during_body_read_rechecks_auth(tmp_path: Path) -> None:
    auth = GatewayAuth(tmp_path / "auth")
    reached = False

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        nonlocal reached
        reached = True

    async def receive():  # type: ignore[no-untyped-def]
        auth.rotate()
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):  # type: ignore[no-untyped-def]
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/events",
        "headers": [
            (b"host", b"127.0.0.1:8401"),
            (b"authorization", b"Bearer " + auth.read_token()),
        ],
    }
    await LoopbackGuard(app, port=8401, auth=auth)(scope, receive, send)
    assert sent[0]["status"] == 401
    assert not reached
    auth.close()


def test_login_cli_refuses_non_terminal(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "localmelo", "--gateway-login"],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert not result.stdout
    assert "only to a terminal" in result.stderr
    assert not (tmp_path / ".cache/localmelo/gateway/auth/token").exists()


def test_rotation_cli_changes_private_token_offline(tmp_path: Path) -> None:
    auth = GatewayAuth(tmp_path / ".cache/localmelo/gateway/auth")
    before = auth.read_token()
    result = subprocess.run(
        [sys.executable, "-m", "localmelo", "--gateway-rotate-token"],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert not auth.valid_bearer(before.decode("ascii"))
    assert before.decode("ascii") not in result.stdout + result.stderr
    auth.close()


def test_non_loopback_gateway_refused() -> None:
    cfg = Config()
    cfg.gateway.host = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        GatewayServer(cfg)


def _startup_token(path: str) -> bytes:
    auth = GatewayAuth(Path(path))
    try:
        return auth.read_token()
    finally:
        auth.close()


def _startup_or_rotate(args: tuple[str, bool]) -> None:
    path, rotate = args
    auth = GatewayAuth(Path(path), allow_corrupt=rotate)
    try:
        if rotate:
            auth.rotate()
        else:
            auth.read_token()
    finally:
        auth.close()


def test_concurrent_first_start_and_rotation(tmp_path: Path) -> None:
    path = str(tmp_path / "gateway/auth")
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_startup_or_rotate, [(path, i % 2 == 0) for i in range(24)])
    auth = GatewayAuth(Path(path))
    assert len(auth.read_token()) == 64
    auth.close()


def test_concurrent_first_start_and_empty_recovery(tmp_path: Path) -> None:
    path = tmp_path / "gateway" / "auth"
    for empty in (False, True):
        if empty:
            (path / "token").write_bytes(b"")
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            tokens = pool.map(_startup_token, [str(path)] * 24)
        assert len(set(tokens)) == 1
        assert len(tokens[0]) == 64


def test_rotation_invalidates_streams_even_if_directory_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = GatewayAuth(tmp_path / "auth")
    old_token = auth.read_token()
    event = auth.rotation
    fsync = os.fsync

    def fail_directory(fd: int) -> None:
        if fd == auth._directory:
            raise OSError("directory sync failed")
        fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory sync failed"):
        auth.rotate()
    assert event.is_set()
    assert auth.read_token() != old_token
    auth.close()


def test_corrupt_token_requires_explicit_cli_rotation(tmp_path: Path) -> None:
    path = tmp_path / ".cache/localmelo/gateway/auth"
    auth = GatewayAuth(path)
    auth.close()
    (path / "token").write_bytes(b"corrupt-secret")
    with pytest.raises(ValueError) as error:
        GatewayAuth(path)
    assert str(path / "token") in str(error.value)
    assert "melo --gateway-rotate-token" in str(error.value)
    assert "corrupt-secret" not in str(error.value)
    result = subprocess.run(
        [sys.executable, "-m", "localmelo", "--gateway-rotate-token"],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "corrupt-secret" not in result.stdout + result.stderr
    auth = GatewayAuth(path)
    assert len(auth.read_token()) == 64
    auth.close()


def test_symlink_prefix_allowed_but_gateway_and_auth_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    prefix = tmp_path / "prefix"
    prefix.symlink_to(real, target_is_directory=True)
    path = prefix / "gateway" / "auth"
    auth = GatewayAuth(path)
    auth.close()
    assert (real / "gateway/auth/token").exists()
    for boundary in ("gateway", "auth"):
        root = tmp_path / boundary
        root.mkdir()
        target = tmp_path / (boundary + "-target")
        target.mkdir()
        gateway = root / "gateway"
        if boundary == "gateway":
            gateway.symlink_to(target, target_is_directory=True)
        else:
            gateway.mkdir()
            (gateway / "auth").symlink_to(target, target_is_directory=True)
        with pytest.raises(OSError, match="symlinks are not allowed"):
            GatewayAuth(gateway / "auth")


def test_auth_refuses_unsafe_directory_and_token(tmp_path: Path) -> None:
    path = tmp_path / "auth"
    path.mkdir(mode=0o755)
    os.chmod(path, 0o755)
    with pytest.raises(PermissionError):
        GatewayAuth(path)
    os.chmod(path, 0o700)
    (path / "token").symlink_to(tmp_path / "not-a-token")
    with pytest.raises(OSError):
        GatewayAuth(path)
    with pytest.raises(OSError):
        GatewayAuth(path, allow_corrupt=True)
    (path / "token").unlink()
    (path / "token").write_bytes(b"invalid")
    os.chmod(path / "token", 0o644)
    with pytest.raises(PermissionError):
        GatewayAuth(path, allow_corrupt=True)
    os.chmod(path / "token", 0o600)
    os.link(path / "token", path / "another-link")
    with pytest.raises(PermissionError):
        GatewayAuth(path, allow_corrupt=True)
