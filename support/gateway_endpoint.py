"""Where the running gateway listens: ``~/.cache/localmelo/gateway/endpoint.json``.

The gateway that holds the main chat's file writes it once it listens (its
pid, host, port, the app's instance and a random ``id``) and removes it when
it shuts down. The terminal and the macOS app read it before they start a
gateway of their own, so whichever starts first, the other attaches. A file
whose process is gone, or whose port answers without the same ``id`` in the
``X-LocalMelo-Gateway`` header of ``GET /``, is stale: it is ignored, and the
next gateway replaces it. The file names a port, not a credential.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Final

import httpx

ENDPOINT_PATH: Final = "~/.cache/localmelo/gateway/endpoint.json"
GATEWAY_HEADER: Final = "X-LocalMelo-Gateway"
# How long `melo serve stop` waits for the gateway to end.
STOP_S: Final = 10.0


@dataclass(frozen=True)
class Endpoint:
    pid: int
    host: str
    port: int
    instance: str
    id: str
    started: float

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


def _path(path: str | None) -> str:
    return os.path.expanduser(path or ENDPOINT_PATH)


def write(endpoint: Endpoint, path: str | None = None) -> None:
    """Replace the file with *endpoint* at once (a temporary file, then a rename), 0600."""
    target = _path(path)
    folder = os.path.dirname(target)
    os.makedirs(folder, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".endpoint-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(asdict(endpoint), file)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def read(path: str | None = None) -> Endpoint | None:
    """The file's endpoint, or None when there is none or it does not parse."""
    try:
        with open(_path(path), encoding="utf-8") as file:
            data = json.load(file)
        found = Endpoint(
            pid=int(data["pid"]),
            host=str(data["host"]),
            port=int(data["port"]),
            instance=str(data.get("instance") or ""),
            id=str(data["id"]),
            started=float(data.get("started") or 0),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return found if found.pid > 0 and 0 < found.port <= 65535 and found.id else None


def remove(gateway_id: str, path: str | None = None) -> None:
    """Remove the file if it still names *gateway_id* (a newer gateway's stays)."""
    found = read(path)
    if found is not None and found.id == gateway_id:
        with contextlib.suppress(OSError):
            os.unlink(_path(path))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def confirmed(endpoint: Endpoint, timeout_s: float = 2.0) -> bool:
    """Whether the gateway at *endpoint* answers ``GET /`` with the same ``id``."""
    try:
        response = httpx.get(
            endpoint.url + "/",
            timeout=timeout_s,
            trust_env=False,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        return False
    return bool(response.headers.get(GATEWAY_HEADER) == endpoint.id)


def live(
    path: str | None = None, *, confirm: Callable[[Endpoint], bool] | None = None
) -> Endpoint | None:
    """The running gateway the file names; None when the file is missing or stale."""
    found = read(path)
    if found is None or not alive(found.pid) or not (confirm or confirmed)(found):
        return None
    return found


def stop(path: str | None = None, *, wait_s: float = STOP_S) -> tuple[int, str]:
    """``melo serve stop``: SIGTERM to the running gateway, then wait; (exit code, line)."""
    found = live(path)
    if found is None:
        return 0, "No gateway is running."
    with contextlib.suppress(ProcessLookupError):
        os.kill(found.pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while alive(found.pid):
        if time.monotonic() >= deadline:
            return (
                1,
                f"The gateway on port {found.port} did not stop in {wait_s:.0f} s.",
            )
        time.sleep(0.1)
    return 0, f"Stopped the gateway on port {found.port}."


__all__ = [
    "ENDPOINT_PATH",
    "GATEWAY_HEADER",
    "Endpoint",
    "alive",
    "confirmed",
    "live",
    "read",
    "remove",
    "stop",
    "write",
]
