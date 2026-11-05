"""A client of melo_serve, the per-user daemon that serves local models.

melo_serve serves the chat and the embedding model behind one OpenAI-compatible
address and switches them in one command (``melo-serve use <pick>``). It owns
the runtimes; this module only reads what it says and, for ``melo serve``, asks
it to start. Two things are read:

- the **state file**, ``$MELO_SERVE_HOME/state.json`` (default
  ``~/.cache/melo_serve``): what is served now, replaced whole on every change;
- the **control socket**, ``$MELO_SERVE_SOCKET`` (default
  ``$MELO_SERVE_HOME/control.sock``): plain HTTP, for :class:`Control`.

Both paths are resolved at call time. The state file is read at a provider
build and at a banner, never per request.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Final

import httpx

HOME_ENV: Final = "MELO_SERVE_HOME"
SOCKET_ENV: Final = "MELO_SERVE_SOCKET"
SCHEMA_PREFIX: Final = "melo_serve.state/"
COMMAND: Final = "melo-serve"
DEFAULT_URL: Final = "http://127.0.0.1:8321/v1"
USE_HINT: Final = f"{COMMAND} use <pick>"
INSTALL_HINT: Final = "uv tool install git+https://github.com/localmelo/melo_serve"
# How long ``melo serve`` waits for a melo_serve it started to say it runs.
UP_WAIT_S: Final = 15.0
CONTROL_TIMEOUT_S: Final = 10.0
ROLES: Final = ("chat", "embedding")


def home() -> Path:
    value = os.environ.get(HOME_ENV)
    return Path(value).expanduser() if value else Path.home() / ".cache" / "melo_serve"


def state_path() -> Path:
    return home() / "state.json"


def socket_path() -> Path:
    value = os.environ.get(SOCKET_ENV)
    return Path(value).expanduser() if value else home() / "control.sock"


def read_state() -> dict[str, Any] | None:
    """The state file as melo_serve wrote it; None if it is missing or not its state."""
    try:
        state = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    schema = state.get("schema") if isinstance(state, dict) else None
    if not isinstance(schema, str) or not schema.startswith(SCHEMA_PREFIX):
        return None
    return state  # type: ignore[no-any-return]


def served(state: dict[str, Any] | None, role: str = "chat") -> dict[str, Any] | None:
    """The *role* (``chat`` or ``embedding``) as the state says it is served, else None."""
    roles = state.get("roles") if state else None
    entry = roles.get(role) if isinstance(roles, dict) else None
    return entry if isinstance(entry, dict) and entry.get("model") else None


def front_door_url(state: dict[str, Any] | None) -> str:
    door = state.get("front_door") if state else None
    url = door.get("url") if isinstance(door, dict) else None
    return url if isinstance(url, str) and url else DEFAULT_URL


def backend_name(entry: dict[str, Any]) -> str:
    """The display name of a role's backend (``Splash``), else its id (``splash``)."""
    return str(entry.get("backend_name") or entry.get("backend") or "")


def running(state: dict[str, Any] | None) -> bool:
    """Whether the state says melo_serve runs and the process it names is alive."""
    if not state or state.get("status") == "stopped":
        return False
    door = state.get("front_door")
    pid = door.get("pid") if isinstance(door, dict) else None
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: the process exists, under another user
        return True
    return True


class MeloServeError(Exception):
    """melo_serve refused a call, or could not be reached.

    *status* is the HTTP status (0: no answer), *type* the ``error.type`` of its
    reply (``unknown_pick``, ``busy``, ...; ``not_running`` without an answer)
    and *data* the reply's other error fields (``suggestions``, ``generation``).
    """

    def __init__(
        self, status: int, type: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.type, self.message = status, type, message
        self.data = data or {}


class Control:
    """The control socket: ``picks()``, ``switch()``, ``up()`` and ``stop()``.

    *transport* replaces the socket in tests. Every method raises
    :class:`MeloServeError`.
    """

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def _call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float | None = CONTROL_TIMEOUT_S,
    ) -> dict[str, Any]:
        transport = self._transport or httpx.AsyncHTTPTransport(uds=str(socket_path()))
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://melo_serve", timeout=timeout
            ) as client:
                response = await client.request(method, path, json=body)
        except httpx.TransportError as exc:
            raise MeloServeError(
                0,
                "not_running",
                f"melo_serve did not answer ({type(exc).__name__}); "
                f"start it with `{COMMAND} up`",
            ) from exc
        except httpx.HTTPError as exc:
            raise MeloServeError(
                0, "bad_reply", f"the reply could not be read ({exc})"
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code >= 400:
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            data = {k: v for k, v in error.items() if k not in ("type", "message")}
            raise MeloServeError(
                response.status_code,
                str(error.get("type") or "http_error"),
                str(error.get("message") or f"HTTP {response.status_code}"),
                data,
            )
        if not isinstance(payload, dict):
            raise MeloServeError(
                response.status_code, "bad_reply", "the reply is not a JSON object"
            )
        return payload

    async def picks(self) -> dict[str, Any]:
        """``{"memory": ..., "picks": [...]}``: what can be served."""
        return await self._call("GET", "/picks")

    async def switch(
        self,
        pick: str,
        *,
        expected_generation: int | None = None,
        drain_s: float | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Serve *pick* (an id, alias or served name); answers once it is ready.

        There is no timeout: a switch drains requests and loads a model, which
        can take minutes.
        """
        body: dict[str, Any] = {"pick": pick, "force": force}
        if expected_generation is not None:
            body["expected_generation"] = expected_generation
        if drain_s is not None:
            body["drain_s"] = drain_s
        return await self._call("POST", "/switch", body, timeout=None)

    async def up(self, role: str = "chat") -> dict[str, Any]:
        """Load *role*'s pick if it is not ready, and answer the state."""
        return await self._call("POST", "/up", {"role": role}, timeout=None)

    async def stop(self) -> dict[str, Any]:
        """Stop the models and melo_serve."""
        return await self._call("POST", "/stop")


def role_text(entry: dict[str, Any] | None) -> str:
    """``model · backend · status`` of a served role, or ``none``."""
    if entry is None:
        return "none"
    parts = (entry["model"], backend_name(entry), str(entry.get("status") or ""))
    return " · ".join(part for part in parts if part)


def status_lines(state: dict[str, Any] | None) -> list[str]:
    """The ``melo serve status`` block: running or not, address, roles, login item."""
    if not running(state):
        how = f"{COMMAND} up" if shutil.which(COMMAND) else INSTALL_HINT
        return ["melo_serve:", "  running: False", f"  start: {how}"]
    assert state is not None  # running() is False without one
    login = (state.get("service") or {}).get("login")
    return [
        "melo_serve:",
        f"  running: True ({state.get('status', 'ready')})",
        f"  address: {front_door_url(state)}",
        f"  chat: {role_text(served(state, 'chat'))}",
        f"  embedding: {role_text(served(state, 'embedding'))}",
        f"  login item: {'installed' if login else 'not installed'}",
    ]


def _line(state: dict[str, Any], role: str) -> str:
    entry = served(state, role)
    where = f"({front_door_url(state)})"
    if entry is None:
        return f"melo_serve: no {role} model yet; run `{USE_HINT}` {where}"
    return f"melo_serve: {entry['model']} on {backend_name(entry)} {where}"


def ensure_running(role: str = "chat") -> str:
    """Make sure melo_serve runs for ``melo serve``, and say so in one line.

    Running: the line names what it serves for *role*. Not running: ``melo-serve
    up`` starts it (through launchd when its login item is installed, otherwise
    detached) and returns once it is up; the state is then waited for up to
    :data:`UP_WAIT_S`. Not installed: the line says how to install it. It never
    raises, so the gateway starts either way.
    """
    state = read_state()
    if running(state):
        assert state is not None
        return _line(state, role)
    command = shutil.which(COMMAND)
    if command is None:
        return f"melo_serve is not installed: {INSTALL_HINT}"
    try:
        subprocess.run(
            [command, "up"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=UP_WAIT_S * 2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"melo_serve did not start: {exc}"
    deadline = time.monotonic() + UP_WAIT_S
    while True:
        state = read_state()
        if running(state):
            assert state is not None
            return _line(state, role)
        if time.monotonic() >= deadline:
            return f"melo_serve did not start in {UP_WAIT_S:g} s; try `{COMMAND} up`"
        time.sleep(0.2)
