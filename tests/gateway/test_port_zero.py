"""``melo --serve --port 0``: the gateway binds a free port and uses it everywhere.

Only loopback sockets on ports the system picks are bound; nothing listens.
"""

from __future__ import annotations

import socket
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import localmelo.__main__ as cli
from localmelo.support.config import Config
from localmelo.support.gateway import GatewayServer, bind_free_port


def _cfg(port: int = 0, host: str = "127.0.0.1", *, ephemeral: bool = True) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    cfg.vllm.chat_model = "m"
    cfg.gateway.port = port
    cfg.gateway.host = host
    cfg._ephemeral_port = ephemeral
    return cfg


def _port_errors(cfg: Config) -> list[str]:
    return [error for error in cfg.validate() if "[gateway] port" in error]


def test_port_zero_is_valid_only_when_asked_for() -> None:
    assert _port_errors(_cfg(0)) == []
    assert _port_errors(_cfg(0, ephemeral=False)) == [
        "[gateway] port=0 is out of range (1-65535)"
    ]
    assert _port_errors(_cfg(65536)) != []
    assert _port_errors(_cfg(-1)) != []


@pytest.mark.parametrize("host", ["127.0.0.1", "[::1]"])
def test_bind_free_port_holds_the_port_it_reports(host: str) -> None:
    cfg = _cfg(0, host)
    try:
        sock = bind_free_port(cfg)
    except OSError:
        pytest.skip("no IPv6 loopback")
    try:
        assert cfg.gateway.port == sock.getsockname()[1] > 0
        assert cfg.gateway_port_ok
    finally:
        sock.close()


def test_bind_free_port_refuses_a_non_loopback_host() -> None:
    with pytest.raises(ValueError, match="loopback"):
        bind_free_port(_cfg(0, "0.0.0.0"))


def test_the_gateway_signs_in_on_the_bound_port() -> None:
    cfg = _cfg()
    sock = bind_free_port(cfg)
    port = cfg.gateway.port
    server = GatewayServer(cfg)
    try:
        assert server.port == port
        client = TestClient(server.app, base_url=f"http://127.0.0.1:{port}")
        code = server.auth.issue_login(port)
        response = client.post(
            "/api/auth/login",
            json={"code": code},
            headers={"Origin": f"http://127.0.0.1:{port}"},
        )
        assert response.status_code == 204
        assert f"melo_session_{port}" in response.headers["set-cookie"]
        assert client.get("/api/health").status_code == 200
    finally:
        server.auth.close()
        sock.close()


def test_start_gateway_hands_the_bound_socket_to_the_server() -> None:
    cfg = _cfg()
    seen: dict[str, object] = {}

    def run(self: GatewayServer, sockets: list[socket.socket] | None = None) -> None:
        seen["port"] = self.port
        seen["sockets"] = sockets

    with (
        mock.patch.object(GatewayServer, "run", run),
        mock.patch("builtins.print") as out,
    ):
        cli._start_gateway(cfg)
    sockets = seen["sockets"]
    assert isinstance(sockets, list) and len(sockets) == 1
    (sock,) = sockets
    try:
        port = sock.getsockname()[1]
        assert seen["port"] == port == cfg.gateway.port
    finally:
        sock.close()
    printed = "\n".join(str(call.args[0]) for call in out.call_args_list if call.args)
    assert f"  Gateway: http://127.0.0.1:{port}" in printed
    assert f"  Web UI:  http://127.0.0.1:{port}/" in printed
    assert f"melo --gateway-login --port {port} in another terminal" in printed
    assert f"--gateway http://127.0.0.1:{port}" in printed
    assert ":0" not in printed


def test_a_fixed_port_binds_nothing_first() -> None:
    cfg = _cfg(8401, ephemeral=False)
    start = mock.MagicMock()
    gateway = mock.MagicMock(start_gateway=start, bind_free_port=mock.MagicMock())
    with (
        mock.patch.dict(sys.modules, {"localmelo.support.gateway": gateway}),
        mock.patch("builtins.print"),
    ):
        cli._start_gateway(cfg)
    gateway.bind_free_port.assert_not_called()
    start.assert_called_once_with(cfg)


def _run(*argv: str, cfg: Config) -> mock.MagicMock:
    with (
        mock.patch.object(cli, "_register_backends"),
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch("localmelo.support.config.load", return_value=cfg),
        mock.patch.object(cli, "_start_gateway") as start,
    ):
        cli.main()
    return start


def test_serve_port_zero_reaches_the_gateway_as_a_request_for_a_free_port() -> None:
    cfg = _cfg(8401, ephemeral=False)
    start = _run("--serve", "--port", "0", cfg=cfg)
    start.assert_called_once_with(cfg)
    assert cfg.gateway.port == 0 and cfg.gateway_port_ok


def test_serve_with_a_saved_port_zero_is_still_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _cfg(0, ephemeral=False)
    with pytest.raises(SystemExit) as exc:
        _run("--serve", cfg=cfg)
    assert exc.value.code == 1
    assert "[gateway] port=0 is out of range (1-65535)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--daemon", "install", "--port", "0"],
        ["--gateway-login", "--port", "0"],
        ["--gateway-rotate-token", "--port", "0"],
    ],
)
def test_port_zero_is_refused_where_a_fixed_port_is_needed(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch.object(cli, "_start_gateway") as start,
        mock.patch("localmelo.support.gateway.daemon.install") as install,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    start.assert_not_called()
    install.assert_not_called()
    assert "--port 0 picks a free port for --serve only" in capsys.readouterr().err
