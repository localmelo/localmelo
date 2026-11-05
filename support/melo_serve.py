"""A client of melo_serve, the per-user daemon that serves local models.

melo_serve serves the chat and the embedding model behind one OpenAI-compatible
address and switches them in one command (``melo-serve use <pick>``). It owns
the runtimes; this module only reads what it says and, for ``melo serve``, asks
it to start. What it says is read as a :class:`Snapshot`:

- the **state file**, ``$MELO_SERVE_HOME/state.json`` (default
  ``~/.cache/melo_serve``): what is served now, replaced whole on every change;
- the **control socket**, ``$MELO_SERVE_SOCKET`` (default
  ``$MELO_SERVE_HOME/control.sock``): melo_serve runs only while it accepts a
  connection, so a state left by a crash or a stop says nothing.

Both paths are resolved at call time, and every surface (a banner, a provider
build, a status line) takes one snapshot and reads everything from it. The
state file is data from another process: a field that is not what it should be
counts as absent, and :func:`snapshot` never raises.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

HOME_ENV: Final = "MELO_SERVE_HOME"
SOCKET_ENV: Final = "MELO_SERVE_SOCKET"
SCHEMA_PREFIX: Final = "melo_serve.state/"
COMMAND: Final = "melo-serve"
DEFAULT_URL: Final = "http://127.0.0.1:8321/v1"
USE_HINT: Final = f"{COMMAND} use <pick>"
INSTALL_HINT: Final = "uv tool install git+https://github.com/localmelo/melo_serve"
NOT_INSTALLED: Final = f"melo_serve is not installed: {INSTALL_HINT}"
# The whole budget of ``melo serve`` for a melo_serve that is not running:
# ``melo-serve up`` and the wait for the state it writes.
UP_WAIT_S: Final = 15.0
# A state file is a few hundred bytes; a larger one is not read.
MAX_STATE_BYTES: Final = 256 * 1024
CONNECT_TIMEOUT_S: Final = 0.2
LOOPBACK_HOSTS: Final = ("127.0.0.1", "localhost", "::1")
ROLES: Final = ("chat", "embedding")


def home() -> Path:
    value = os.environ.get(HOME_ENV)
    return Path(value).expanduser() if value else Path.home() / ".cache" / "melo_serve"


def state_path() -> Path:
    return home() / "state.json"


def socket_path() -> Path:
    value = os.environ.get(SOCKET_ENV)
    return Path(value).expanduser() if value else home() / "control.sock"


@dataclass(frozen=True)
class Role:
    """A served model: its id, the pick's name, its backend and that backend's status."""

    model: str
    name: str = ""
    backend_name: str = ""
    status: str = ""


@dataclass(frozen=True)
class Snapshot:
    """What melo_serve says now. Not running is the empty one: no URL, no roles."""

    running: bool = False
    url: str = ""
    status: str = ""
    login: bool = False
    roles: dict[str, Role] = field(default_factory=dict)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _read_state() -> dict[str, Any] | None:
    """The state file as melo_serve wrote it; None if it is missing or not its state."""
    try:
        with state_path().open("rb") as handle:
            raw = handle.read(MAX_STATE_BYTES + 1)
        if len(raw) > MAX_STATE_BYTES:
            return None
        state = json.loads(raw)
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    if not isinstance(state, dict) or not _text(state.get("schema")).startswith(
        SCHEMA_PREFIX
    ):
        return None
    return state


def _loopback(url: str) -> bool:
    """Whether *url* is http(s) on this computer: the only address melo_serve listens on."""
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018  (a port that is not a number raises)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and parts.hostname in LOOPBACK_HOSTS


def _answers() -> bool:
    """Whether something accepts a connection on the control socket."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(CONNECT_TIMEOUT_S)
            probe.connect(str(socket_path()))
    except (OSError, ValueError):  # no socket, refused, timed out; a NUL in the path
        return False
    return True


def _role(entry: object) -> Role | None:
    """The role *entry* names, or None when it is not a dict with a model."""
    if not isinstance(entry, dict) or not _text(entry.get("model")):
        return None
    return Role(
        entry["model"],
        _text(entry.get("name")),
        _text(entry.get("backend_name")) or _text(entry.get("backend")),
        _text(entry.get("status")),
    )


def snapshot() -> Snapshot:
    """What melo_serve says now; the empty snapshot when it does not run. Never raises.

    It runs when the state file is its own, does not say ``stopped``, names a
    loopback address and the control socket accepts a connection. Only then
    does it serve its roles (those with a model).
    """
    state = _read_state()
    if state is None:
        return Snapshot()
    status = _text(state.get("status")) or "ready"
    door = state.get("front_door")
    url = (_text(door.get("url")) if isinstance(door, dict) else "") or DEFAULT_URL
    if status == "stopped" or not _loopback(url) or not _answers():
        return Snapshot()
    service, entries = state.get("service"), state.get("roles")
    entries = entries if isinstance(entries, dict) else {}
    roles = {name: _role(entries.get(name)) for name in ROLES}
    return Snapshot(
        running=True,
        url=url,
        status=status,
        login=isinstance(service, dict) and service.get("login") is True,
        roles={name: role for name, role in roles.items() if role is not None},
    )


class MeloServeError(Exception):
    """melo_serve cannot answer for a model; the message says what to run."""


def command() -> str | None:
    """The installed ``melo-serve`` command, or None."""
    return shutil.which(COMMAND)


def start_hint() -> str:
    """How to get a melo_serve that is not running: start it, or install it first."""
    return f"{COMMAND} up" if command() else INSTALL_HINT


def unavailable(snap: Snapshot, role: str) -> MeloServeError:
    """The error for a *role* that *snap* does not serve, with the command that fixes it."""
    if snap.running:
        return MeloServeError(
            f"melo_serve serves no {role} model now. Run `{USE_HINT}` to choose one."
        )
    if command():
        return MeloServeError(f"melo_serve is not running. Run `{COMMAND} up`.")
    return MeloServeError(NOT_INSTALLED)


def _clean(text: str) -> str:
    """*text* as a terminal shows it; a state's words never move the cursor."""
    from localmelo.support.sessions.display import clean_output

    return clean_output(text)


def role_text(role: Role | None) -> str:
    """``model · backend · status`` of a served role, or ``none``."""
    if role is None:
        return "none"
    return " · ".join(
        part for part in (role.model, role.backend_name, role.status) if part
    )


def status_lines(snap: Snapshot) -> list[str]:
    """The ``melo serve status`` block: running or not, address, roles, login item."""
    if not snap.running:
        return ["melo_serve:", "  running: False", f"  start: {start_hint()}"]
    lines = [
        "melo_serve:",
        f"  running: True ({snap.status})",
        f"  address: {snap.url}",
        f"  chat: {role_text(snap.roles.get('chat'))}",
        f"  embedding: {role_text(snap.roles.get('embedding'))}",
        f"  login item: {'installed' if snap.login else 'not installed'}",
    ]
    return [_clean(line) for line in lines]


def _line(snap: Snapshot, role: str) -> str:
    where = f"({snap.url})"
    served = snap.roles.get(role)
    if served is None:
        return _clean(f"melo_serve: no {role} model yet; run `{USE_HINT}` {where}")
    return _clean(f"melo_serve: {served.model} on {served.backend_name} {where}")


def ensure_running(role: str = "chat") -> str:
    """Make sure melo_serve runs for ``melo serve``, and say so in one line.

    Running: the line names what it serves for *role*. Not running: ``melo-serve
    up`` starts it (through launchd when its login item is installed, otherwise
    detached) and the state is then waited for, all within one budget of
    :data:`UP_WAIT_S` (15 s); a non-zero exit of ``up`` returns at once with the
    last line it printed (its stderr, else its stdout). Not installed: the line
    says how to install it. It never raises, so the gateway starts either way.
    """
    snap = snapshot()
    if snap.running:
        return _line(snap, role)
    path = command()
    if path is None:
        return NOT_INSTALLED
    late = f"melo_serve did not start in {UP_WAIT_S:g} s; try `{COMMAND} up`"
    deadline = time.monotonic() + UP_WAIT_S
    try:
        up = subprocess.run(
            [path, "up"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=deadline - time.monotonic(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return late
    except (OSError, subprocess.SubprocessError) as exc:
        return _clean(f"melo_serve did not start: {exc}")
    if up.returncode != 0:
        for text in (up.stderr, up.stdout):
            lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
            if lines:
                return _clean(f"melo_serve did not start: {lines[-1]}")
        return f"melo_serve did not start: {COMMAND} up exited {up.returncode}"
    while True:
        snap = snapshot()
        if snap.running:
            return _line(snap, role)
        if time.monotonic() >= deadline:
            return late
        time.sleep(0.2)
