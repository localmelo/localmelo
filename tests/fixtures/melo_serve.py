"""A fake melo_serve state file, in the shape of its ``melo_serve.state/1`` schema."""

from __future__ import annotations

import json
import shutil
import socketserver
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

URL = "http://127.0.0.1:8321/v1"


def role(model: str, backend: str = "splash", name: str = "Splash") -> dict[str, Any]:
    return {
        "pick": model.lower(),
        "short": "m",
        "name": model,
        "model": model,
        "backend": backend,
        "backend_name": name,
        "status": "ready",
    }


def write_state(
    home: Path,
    *,
    chat: dict[str, Any] | None = None,
    embedding: dict[str, Any] | None = None,
    status: str = "ready",
    login: bool = True,
) -> None:
    """Write ``state.json`` in *home*. It says nothing is running until a socket listens."""
    state = {
        "schema": "melo_serve.state/1",
        "generation": 1,
        "status": status,
        "front_door": {"url": URL, "pid": 1, "auth": "none"},
        "service": {"login": login},
        "roles": {"chat": chat, "embedding": embedding},
    }
    (home / "state.json").write_text(json.dumps(state), encoding="utf-8")


class _Hangup(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        """A connection is all a client asks for."""


@contextmanager
def listening() -> Iterator[Path]:
    """A control socket that accepts connections: what makes a state running.

    It lives in a short folder under ``/tmp``, because macOS allows 104 bytes
    for a socket path, which pytest's ``tmp_path`` exceeds. Both are removed
    on exit.
    """
    folder = Path(tempfile.mkdtemp(dir="/tmp"))
    server = socketserver.ThreadingUnixStreamServer(
        str(folder / "control.sock"), _Hangup
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield folder / "control.sock"
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(folder, ignore_errors=True)
