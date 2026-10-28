"""Host and origin precede auth; body metadata is checked only after auth."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.types import Message, Receive, Scope, Send

from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway.guard import LoopbackGuard


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"2000000"), (b"content-type", b"application/json")],
        [(b"content-length", b"1"), (b"content-type", b"text/plain")],
    ],
)
async def test_unauthenticated_body_headers_do_not_reveal_limits(
    tmp_path: Path, headers: list[tuple[bytes, bytes]]
) -> None:
    auth = GatewayAuth(tmp_path / "auth")
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError("route reached")

    async def receive() -> Message:
        raise AssertionError("body read")

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/health",
        "headers": [(b"host", b"127.0.0.1:8401"), *headers],
    }
    try:
        await LoopbackGuard(app, port=8401, auth=auth)(scope, receive, send)
        assert sent[0]["status"] == 401
    finally:
        auth.close()
