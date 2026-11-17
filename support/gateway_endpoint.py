"""Where the running gateway listens: ``~/.cache/localmelo/gateway/endpoint.json``.

The gateway that holds the main chat's file writes it once it listens (its
pid, host, port, the app's instance and a random ``id``) and removes it when
it shuts down. The terminal and the macOS app read it before they start a
gateway of their own, so whichever starts first, the other attaches. A file
whose process is gone, or whose port answers without the same ``id`` in the
``X-LocalMelo-Gateway`` header of ``GET /``, is stale: it is ignored, and the
next gateway replaces it. The file names a port, not a credential.

A gateway also says which build it runs and whether anything runs in it. After
an update the gateway of the build before would otherwise keep serving: one
that is idle is stopped by the terminal or the app that finds it, which then
starts its own; one that is busy is used as it is and replaced at a later start.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import httpx

ENDPOINT_PATH: Final = "~/.cache/localmelo/gateway/endpoint.json"
GATEWAY_HEADER: Final = "X-LocalMelo-Gateway"
# The build a gateway runs, and "1" while a chat, task or run is in it ("0" when idle), on ``GET /``.
BUILD_HEADER: Final = "X-LocalMelo-Build"
BUSY_HEADER: Final = "X-LocalMelo-Busy"
# The build stamp, beside the package's ``__init__.py``; the app's build writes it.
BUILD_FILE: Final = Path(__file__).resolve().parents[1] / "BUILD"
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


def build() -> str:
    """The build of this code, "" for one without a stamp (a source checkout)."""
    try:
        return BUILD_FILE.read_text(encoding="utf-8").strip()[:64]
    except (OSError, ValueError):
        return ""


def answer(endpoint: Endpoint, timeout_s: float = 2.0) -> Mapping[str, str] | None:
    """The headers of ``GET /`` at *endpoint* when it answers with the same ``id``, else None."""
    try:
        response = httpx.get(
            endpoint.url + "/",
            timeout=timeout_s,
            trust_env=False,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        return None
    if response.headers.get(GATEWAY_HEADER) != endpoint.id:
        return None
    return response.headers


def confirmed(endpoint: Endpoint, timeout_s: float = 2.0) -> bool:
    """Whether the gateway at *endpoint* answers ``GET /`` with the same ``id``."""
    return answer(endpoint, timeout_s) is not None


def replaceable(headers: Mapping[str, str], own: str) -> bool:
    """Whether the gateway that sent *headers* is an idle one of another build than *own*.

    Code without a stamp replaces nothing, and a gateway that does not say it
    is idle (a busy one, or one from before it said so) is kept: ending it
    would end what runs in it.
    """
    return (
        bool(own)
        and headers.get(BUILD_HEADER, "") != own
        and headers.get(BUSY_HEADER) == "0"
    )


def live(
    path: str | None = None, *, confirm: Callable[[Endpoint], bool] | None = None
) -> Endpoint | None:
    """The running gateway the file names; None when the file is missing or stale."""
    found = read(path)
    if found is None or not alive(found.pid) or not (confirm or confirmed)(found):
        return None
    return found


def current(path: str | None = None) -> Endpoint | None:
    """The running gateway to use: :func:`live`, after an idle one of another build was stopped.

    What attaches calls this, so the caller that gets None starts a gateway of its own build.
    """
    found = read(path)
    if found is None or not alive(found.pid):
        return None
    headers = answer(found)
    if headers is None:
        return None
    if replaceable(headers, build()) and _end(found, STOP_S):
        return None
    return found


def _end(found: Endpoint, wait_s: float) -> bool:
    """SIGTERM to the gateway *found* names, then wait; whether it ended in time."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(found.pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while alive(found.pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def stop(path: str | None = None, *, wait_s: float = STOP_S) -> tuple[int, str]:
    """``melo serve stop``: SIGTERM to the running gateway, then wait; (exit code, line)."""
    found = live(path)
    if found is None:
        return 0, "No gateway is running."
    if not _end(found, wait_s):
        return (
            1,
            f"The gateway on port {found.port} did not stop in {wait_s:.0f} s.",
        )
    return 0, f"Stopped the gateway on port {found.port}."


__all__ = [
    "BUILD_HEADER",
    "BUSY_HEADER",
    "ENDPOINT_PATH",
    "GATEWAY_HEADER",
    "Endpoint",
    "alive",
    "answer",
    "build",
    "confirmed",
    "current",
    "live",
    "read",
    "remove",
    "replaceable",
    "stop",
    "write",
]
