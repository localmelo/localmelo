"""A fake melo_serve state file, in the shape of its ``melo_serve.state/1`` schema."""

from __future__ import annotations

import json
import os
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
    pid: int | None = None,
    login: bool = True,
) -> None:
    """Write ``state.json`` in *home*; the pid is this process's unless given."""
    state = {
        "schema": "melo_serve.state/1",
        "generation": 1,
        "status": status,
        "front_door": {"url": URL, "pid": pid or os.getpid(), "auth": "none"},
        "service": {"login": login},
        "roles": {"chat": chat, "embedding": embedding},
    }
    (home / "state.json").write_text(json.dumps(state), encoding="utf-8")
