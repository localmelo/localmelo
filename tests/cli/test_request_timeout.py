"""--request-timeout: the HTTP timeout of each chat request, for one run.

With --base-url it goes to the OpenAI-compatible provider; without it, it
replaces the configured request_timeout_s in memory, and the config file
is not written. The config file is always a temporary one, never
~/.cache/localmelo/config.toml. The last tests send real requests to a
slow local server, to show that the timeout fires and is retried.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import RunResult
from localmelo.support.config import Config, LocalBackendConfig, load, save
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

ANSWER = {"choices": [{"message": {"role": "assistant", "content": "42"}}]}


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


@pytest.fixture
def config_path(tmp_path: Path) -> Iterator[Path]:
    """config.toml in *tmp_path*, which load() and save() use meanwhile."""
    path = tmp_path / "config.toml"
    with (
        mock.patch("localmelo.support.config.CONFIG_PATH", str(path)),
        mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
    ):
        yield path


def _args(*argv: str) -> Any:
    """The parsed flags of ``melo *argv "q"``."""
    return cli._build_parser().parse_args([*argv, "q"])


def _chat_timeout(agent: Agent) -> httpx.Timeout:
    llm = agent.chat.llm
    assert isinstance(llm, OpenAICompatLLM | OllamaNativeChat)
    return llm._http.timeout


def _config(backend: str, request_timeout_s: Any = None) -> Config:
    """A valid config with *backend* for chat and no embedding."""
    cfg = Config(
        chat_backend=backend,
        embedding_backend="none",
        request_timeout_s=request_timeout_s,
    )
    setattr(
        cfg, backend, LocalBackendConfig(chat_url="http://127.0.0.1:9", chat_model="m")
    )
    return cfg


# ── --base-url ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "seconds"),
    [
        ((), 120.0),
        (("--request-timeout", "600"), 600.0),
        (("--request-timeout", "0.5"), 0.5),
    ],
    ids=["default", "600", "0.5"],
)
async def test_base_url_gives_the_provider_the_flag_or_its_default(
    config_path: Path, argv: tuple[str, ...], seconds: float
) -> None:
    """A configured request_timeout_s does not apply with --base-url."""
    # --base-url does not read the config file, so its 900 s is not used.
    save(_config("vllm", 900))
    before = config_path.read_bytes()
    agent = cli._build_direct_mode_agent(
        _args("--base-url", "http://127.0.0.1:9/v1", *argv)
    )
    try:
        assert _chat_timeout(agent) == httpx.Timeout(seconds)
    finally:
        await agent.close()
    assert config_path.read_bytes() == before


# ── The config file's backends ──────────────────────────────────────────────

BACKENDS = pytest.mark.parametrize(
    ("backend", "provider_default"), [("vllm", 120.0), ("ollama", 300.0)]
)


@BACKENDS
@pytest.mark.parametrize("configured", [None, 900], ids=["unset", "configured"])
async def test_without_the_flag_the_configured_timeout_or_the_default(
    config_path: Path, backend: str, provider_default: float, configured: int | None
) -> None:
    save(_config(backend, configured))
    agent = cli._build_direct_mode_agent(_args())
    try:
        expected = provider_default if configured is None else configured
        assert _chat_timeout(agent) == httpx.Timeout(expected)
    finally:
        await agent.close()


@BACKENDS
@pytest.mark.parametrize("configured", [None, 900], ids=["unset", "configured"])
async def test_the_flag_overrides_the_config_for_the_run_only(
    config_path: Path, backend: str, provider_default: float, configured: int | None
) -> None:
    save(_config(backend, configured))
    before = config_path.read_bytes()
    with mock.patch(
        "localmelo.support.config.save", side_effect=AssertionError("saved")
    ):
        agent = cli._build_direct_mode_agent(_args("--request-timeout", "600"))
    try:
        assert _chat_timeout(agent) == httpx.Timeout(600.0)
    finally:
        await agent.close()
    assert config_path.read_bytes() == before
    assert load().request_timeout_s == configured


def test_a_bad_configured_value_is_a_config_error(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    save(_config("vllm", "600"))
    with pytest.raises(SystemExit) as exited:
        cli._build_direct_mode_agent(_args())
    assert exited.value.code == 1
    assert (
        "request_timeout_s must be a finite number of seconds > 0, got '600'"
        in capsys.readouterr().err
    )


async def test_the_flag_replaces_a_bad_configured_value(config_path: Path) -> None:
    """The run uses the flag's value, so the configured one is not checked."""
    save(_config("vllm", "600"))
    agent = cli._build_direct_mode_agent(_args("--request-timeout", "60"))
    try:
        assert _chat_timeout(agent) == httpx.Timeout(60.0)
    finally:
        await agent.close()


# ── A slow model server ─────────────────────────────────────────────────────


class SlowChatServer:
    """``POST /chat/completions`` on 127.0.0.1 that answers ``42`` after a delay.

    ``attempts`` keeps the ``X-Localmelo-Attempt`` header of every request.
    """

    def __init__(self, delay_s: float) -> None:
        self.attempts: list[str | None] = []
        self._closing = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.attempts.append(self.headers.get("X-Localmelo-Attempt"))
                server._closing.wait(delay_s)
                body = json.dumps(ANSWER).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # the client stopped waiting

            def log_message(self, format: str, *args: Any) -> None:
                pass  # no request log in the test output

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def close(self) -> None:
        self._closing.set()  # requests still waiting answer at once
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class Sleeps:
    """Records the waits before retries instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def _run_against(
    delay_s: float, request_timeout: str
) -> tuple[RunResult, list[str | None], list[float], float]:
    """Run one query against a server that answers after *delay_s*.

    Returns the RunResult, the attempt headers the server saw, the waits
    before retries and the run's wall-clock time.
    """
    server = SlowChatServer(delay_s)
    try:
        agent = cli._build_direct_mode_agent(
            _args("--base-url", server.base_url, "--request-timeout", request_timeout),
            options=AgentOptions(builtins=frozenset()),
        )
        sleeps = Sleeps()
        agent.chat.sleep = sleeps
        started = time.monotonic()
        try:
            result = await agent.run_task("What is 6*7?")
        finally:
            await agent.close()
        elapsed = time.monotonic() - started
    finally:
        server.close()
    return result, server.attempts, sleeps.delays, elapsed


@pytest.fixture
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests to 127.0.0.1 go straight to it, whatever proxy is set."""
    for name in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.mark.usefixtures("_no_proxy")
async def test_a_timed_out_request_is_retried_with_the_full_timeout() -> None:
    # The server would answer after 5 s; it is released when the test ends.
    result, attempts, delays, elapsed = await _run_against(5.0, "0.25")
    assert result.status == "provider_error"
    assert result.error.startswith("POST /chat/completions failed: ReadTimeout")
    assert attempts == ["1", "2", "3"]
    assert delays == [1.0, 4.0]  # the usual retries
    assert (result.n_llm, result.llm_errors) == (0, 3)
    # Each attempt waited the whole 0.25 s and no longer: a timeout of 1 s
    # or more would have taken at least 3 s.
    assert 3 * 0.25 <= elapsed < 3.0


@pytest.mark.usefixtures("_no_proxy")
async def test_a_reply_within_the_timeout_completes_the_run() -> None:
    result, attempts, delays, _ = await _run_against(0.3, "10")
    assert (result.status, result.result) == ("completed", "42")
    assert attempts == ["1"]
    assert delays == []
