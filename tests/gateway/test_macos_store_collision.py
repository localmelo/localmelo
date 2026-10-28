"""Exercise the session-store collision seen by the macOS app's health check."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parents[2]


def _available_port() -> int:
    for _ in range(20):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in range(8400, 8800) and port not in (8955, 11434, 11500):
            return port
    raise RuntimeError("Unable to choose a non-reserved test port")


def _health(port: int, child: subprocess.Popen[bytes]) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise AssertionError(f"Gateway exited with status {child.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=1
            ) as response:
                return cast(dict[str, Any], json.load(response))
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise AssertionError("Gateway did not become ready")


def test_second_gateway_reports_owned_session_store(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "LOCALMELO_DETECT_URLS": "",
        "LOCALMELO_APP_INSTANCE": "0123456789abcdef" * 2,
    }
    children: list[subprocess.Popen[bytes]] = []
    try:
        for _ in range(2):
            port = _available_port()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "localmelo",
                    "--serve",
                    "--allow-unconfigured",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT.parent,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            children.append(child)
            health = _health(port, child)
            if len(children) == 1:
                assert health["sessions"]["store"] == "ok"
            else:
                assert health["sessions"]["store"] == "unavailable"
                assert "Another melo gateway" in health["sessions"]["store_error"]
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=10)
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
