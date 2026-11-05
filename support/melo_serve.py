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

The control socket is also the one way to change what is served: :func:`picks`
reads the chat models melo_serve offers and :func:`switch` loads one. Every
surface goes through them (the gateway for the console and the app, the
terminal when no gateway runs), and every surface words a pick with
:func:`pick_words`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import httpx

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
# A reply of the control socket is a few kilobytes; a larger one is not read.
MAX_REPLY_BYTES: Final = 256 * 1024
MAX_PICKS: Final = 64
# ``GET /picks`` measures free memory first; ``POST /switch`` answers once the
# model is ready, which can take minutes, so only its connection is bounded.
PICKS_TIMEOUT_S: Final = 30.0
CONTROL_CONNECT_S: Final = 2.0
# The fit verdicts melo_serve gives; only ``too_big`` is refused.
TOO_BIG: Final = "too_big"
# The fit word a verdict shows (none for ``unknown``), and whether it is tight.
_FIT_WORDS: Final = {"fits": "fits", "estimate": "fits", "alone": "tight"}


def _path(value: str) -> Path:
    """*value* as a path, ``~`` expanded; ``~nosuchuser`` stays as it is, never an error."""
    return Path(os.path.expanduser(value))


def home() -> Path:
    return _path(os.environ.get(HOME_ENV) or "~/.cache/melo_serve")


def state_path() -> Path:
    return home() / "state.json"


def socket_path() -> Path:
    value = os.environ.get(SOCKET_ENV)
    return _path(value) if value else home() / "control.sock"


@dataclass(frozen=True)
class Role:
    """A served model: its id, the pick's name, its backend and that backend's status."""

    model: str
    name: str = ""
    backend_name: str = ""
    status: str = ""


@dataclass(frozen=True)
class Snapshot:
    """What melo_serve says now. Not running is the empty one: no URL, no roles.

    *generation* goes up each time a switch loads a model; a switch names the
    one it was chosen at, so a switch made meanwhile is not undone.
    """

    running: bool = False
    url: str = ""
    status: str = ""
    login: bool = False
    roles: dict[str, Role] = field(default_factory=dict)
    generation: int = 0


@dataclass(frozen=True)
class Pick:
    """A chat model melo_serve offers (one row of ``GET /picks``).

    *need_gb* is the memory it needs here and *free_gb* the memory there is for
    it (free now, plus what the model it replaces gives back): the two figures
    that decide whether it fits. *verdict* is melo_serve's fit word and
    *problem* why it cannot load at all (not downloaded, ...), else ``""``.
    """

    id: str
    short: str = ""
    name: str = ""
    backend_name: str = ""
    model: str = ""
    current: bool = False
    problem: str = ""
    verdict: str = ""
    need_gb: float | None = None
    free_gb: float | None = None
    disk_gb: float | None = None
    load_s: float | None = None

    @property
    def loadable(self) -> bool:
        """Whether a switch may load it: nothing stops it and it fits in memory."""
        return not self.problem and self.verdict != TOO_BIG


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _number(value: object) -> float | None:
    """*value* as a finite number; a bool, a string or NaN is no number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _whole(value: object) -> int:
    """*value* as a whole number that is not negative, else 0."""
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _own(info: os.stat_result) -> bool:
    """Whether the file or folder *info* describes is this user's, and no one else can write it."""
    return info.st_uid == os.getuid() and not info.st_mode & 0o022


def _read_state() -> dict[str, Any] | None:
    """The state file as melo_serve wrote it; None if it is missing or not its state.

    It is opened without following a link, and must be a regular file that
    this user owns, in a folder only this user can write.
    """
    path = state_path()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not (
                stat.S_ISREG(info.st_mode) and _own(info) and _own(os.stat(path.parent))
            ):
                return None
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


def _loopback(url: str) -> str | None:
    """*url* rebuilt from its parts when it is http(s) on this computer, else None.

    melo_serve listens on loopback only, and its URL carries no user info. A URL
    with whitespace or a control character is refused, so the text kept is clean.
    """
    if any(ch.isspace() or not ch.isprintable() for ch in url):
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname
    if (
        parts.scheme not in ("http", "https")
        or host not in LOOPBACK_HOSTS
        or parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
    ):
        return None
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def _private(path: Path) -> bool:
    """Whether *path* is a socket and, with its folder, this user's own (see :func:`_own`).

    ``lstat`` does not follow a link, so a symlink is not a socket. A socket
    that another user could have put there is not melo_serve's.
    """
    try:
        sock, folder = os.lstat(path), os.stat(path.parent)
    except (OSError, ValueError):
        return False
    return stat.S_ISSOCK(sock.st_mode) and _own(sock) and _own(folder)


def _answers() -> bool:
    """Whether something accepts a connection on the control socket.

    Nothing connects to a socket that :func:`_private` refuses.
    """
    path = socket_path()
    if not _private(path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(CONNECT_TIMEOUT_S)
            probe.connect(str(path))
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
    url = _loopback(url) or ""
    if status == "stopped" or not url or not _answers():
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
        generation=_whole(state.get("generation")),
    )


class MeloServeError(Exception):
    """melo_serve cannot answer for a model; the message says what to run.

    A refusal of the control socket keeps its ``error.type`` as *kind*
    (``stale_generation``, ``busy``, ...) and, for ``stale_generation``, the
    *generation* melo_serve is at now.
    """

    def __init__(
        self, message: str, kind: str = "", generation: int | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.generation = generation


def _check(path: str | None) -> str | None:
    """Why *path* is not run: None when nothing is there, ``""`` when it may be run.

    A reason is a clause that follows ``is not run:``. *path* must be absolute,
    and the file it leads to a regular file that the user owns and can run.
    That file, and the folders of *path* and of the file, must pass :func:`_own`.
    Links are followed (``uv tool install`` makes ``~/.local/bin/melo-serve`` a
    link into its own folder).
    """
    if not path or not os.path.lexists(path):
        return None
    if not os.path.isabs(path):
        return "its path is relative"
    try:
        real = os.path.realpath(path)
        info = os.stat(real)
        folders = (os.path.dirname(path), os.path.dirname(real))
        infos = [os.stat(folder) for folder in folders]
    except OSError:
        return "it cannot be read"
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & stat.S_IXUSR:
        return "it is not a regular executable file"
    if info.st_uid != os.getuid():
        return "another user owns it"
    if info.st_mode & 0o022:
        return "others can write it"
    for folder, found in zip(folders, infos, strict=True):
        if found.st_uid != os.getuid():
            return f"another user owns its folder {folder}"
        if not _own(found):
            return f"others can write its folder {folder}"
    return ""


def _candidates() -> tuple[str | None, str]:
    """Where ``melo-serve`` may be: on PATH, then in the uv tool folder."""
    folder = (
        os.environ.get("UV_TOOL_BIN_DIR")
        or os.environ.get("XDG_BIN_HOME")
        or "~/.local/bin"
    )
    return shutil.which(COMMAND), os.path.join(os.path.expanduser(folder), COMMAND)


def _find_command() -> str | None:
    """Where ``melo-serve`` is: the first of PATH and the uv tool folder that is the user's own.

    A launchd login item has no ``~/.local/bin`` on its PATH, where
    ``uv tool install`` puts it: ``$UV_TOOL_BIN_DIR``, else ``$XDG_BIN_HOME``,
    else ``~/.local/bin``. A candidate that :func:`_check` refuses is passed
    over, so a stale file in a system folder does not hide the installed one.
    The path returned is the file the links lead to, which is what was checked
    and what is run.
    """
    for path in _candidates():
        if path is not None and _check(path) == "":
            return os.path.realpath(path)
    return None


def command() -> str | None:
    """The installed ``melo-serve`` command, or None."""
    return _find_command()


def refusal() -> str | None:
    """The line for a ``melo-serve`` that was found but is not run, or None.

    None when one runs, and when none was found (then it is not installed).
    """
    checked = [(path, _check(path)) for path in _candidates()]
    if any(why == "" for _, why in checked):
        return None
    for path, why in checked:
        if why:
            return _clean(f"melo_serve: {path} is not run: {why}")
    return None


def start_hint() -> str:
    """How to get a melo_serve that is not running: start it, or install it first."""
    return f"{COMMAND} up" if command() else refusal() or INSTALL_HINT


def unavailable(snap: Snapshot, role: str) -> MeloServeError:
    """The error for a *role* that *snap* does not serve, with the command that fixes it."""
    if snap.running:
        return MeloServeError(
            f"melo_serve serves no {role} model now. Run `{USE_HINT}` to choose one."
        )
    if command():
        return MeloServeError(f"melo_serve is not running. Run `{COMMAND} up`.")
    return MeloServeError(refusal() or NOT_INSTALLED)


def _clean(text: str) -> str:
    """*text* as a terminal shows it: the cursor stays and the home folder is ``~``."""
    from localmelo.support.sessions.display import clean_output, home_labelled

    return home_labelled(clean_output(text))


def _words(key: str, language: str, **params: Any) -> str:
    """The catalog's words of *key* in *language* (the console's catalogs, read without the gateway)."""
    from localmelo.support.words import t_in

    return t_in(language, key, **params)


# ── The control socket ──


def _refusal(status: int, reply: object) -> MeloServeError:
    """The error a refusal of the control socket carries, in melo_serve's words."""
    error = reply.get("error") if isinstance(reply, dict) else None
    error = error if isinstance(error, dict) else {}
    message = _text(error.get("message")) or f"it answered {status}"
    generation = error.get("generation")
    return MeloServeError(
        _clean(f"melo_serve: {message}"),
        _text(error.get("type")),
        _whole(generation) if isinstance(generation, int) else None,
    )


async def _call(
    method: str, path: str, *, body: dict[str, Any] | None = None, timeout: float | None
) -> Any:
    """One request to the control socket: its JSON reply, or MeloServeError.

    Nothing connects to a socket that :func:`_private` refuses; the error then
    says how to start melo_serve. A reply is read up to :data:`MAX_REPLY_BYTES`
    and must be JSON; a refusal raises with melo_serve's words and its kind.
    """
    where = socket_path()
    if not _private(where):
        raise unavailable(Snapshot(), "chat")
    raw = bytearray()
    try:
        async with (
            httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=str(where)),
                base_url="http://melo_serve",
                timeout=httpx.Timeout(timeout, connect=CONTROL_CONNECT_S),
                trust_env=False,
            ) as client,
            client.stream(method, path, json=body) as response,
        ):
            async for chunk in response.aiter_bytes():
                raw += chunk
                if len(raw) > MAX_REPLY_BYTES:
                    raise MeloServeError(f"melo_serve's answer to {path} is too large")
            status = response.status_code
    except httpx.HTTPError as exc:
        said = str(exc) or type(exc).__name__
        raise MeloServeError(_clean(f"melo_serve did not answer: {said}")) from None
    try:
        reply = json.loads(raw)
    except (ValueError, RecursionError):
        raise MeloServeError(f"melo_serve's answer to {path} is not JSON") from None
    if status >= 400:
        raise _refusal(status, reply)
    return reply


def _pick(entry: object, total_gb: float | None, free_pct: float | None) -> Pick | None:
    """A chat row of ``GET /picks``; None for any other row, or one with no usable id."""
    if not isinstance(entry, dict) or entry.get("role") != "chat":
        return None
    pick_id = _text(entry.get("id"))
    if not pick_id or not pick_id.isprintable():
        return None
    fit = entry.get("fit")
    fit = fit if isinstance(fit, dict) else {}
    need, after = _number(fit.get("need_gb")), _number(fit.get("after_pct"))
    free = None
    if total_gb is not None and need is not None and after is not None:
        free = max(0.0, need + after * total_gb / 100)  # the memory there is for it
    elif total_gb is not None and free_pct is not None:
        free = free_pct * total_gb / 100
    return Pick(
        id=pick_id,
        short=_clean(_text(entry.get("short"))),
        name=_clean(_text(entry.get("name")) or pick_id),
        backend_name=_clean(
            _text(entry.get("backend_name")) or _text(entry.get("backend"))
        ),
        model=_clean(_text(entry.get("model"))),
        current=entry.get("current") is True,
        problem=_clean(_text(entry.get("problem"))),
        verdict=_text(fit.get("verdict")),
        need_gb=need,
        free_gb=free,
        disk_gb=_number(entry.get("disk_gb")),
        load_s=_number(entry.get("load_s")),
    )


async def picks() -> list[Pick]:
    """The chat models melo_serve offers, in its order; MeloServeError when it cannot say.

    A reply that is not an object with a list of picks is refused. A row that
    is not what it should be is left out, and a field of one counts as absent,
    as in the state file.
    """
    reply = await _call("GET", "/picks", timeout=PICKS_TIMEOUT_S)
    rows = reply.get("picks") if isinstance(reply, dict) else None
    if not isinstance(rows, list):
        raise MeloServeError("melo_serve's answer to /picks lists no picks")
    memory = reply.get("memory")
    memory = memory if isinstance(memory, dict) else {}
    total, free_pct = _number(memory.get("total_gb")), _number(memory.get("free_pct"))
    found = (_pick(entry, total, free_pct) for entry in rows[:MAX_PICKS])
    return [pick for pick in found if pick is not None]


async def switch(pick: str, *, expected_generation: int) -> dict[str, Any]:
    """Load *pick* for its role, unless melo_serve has left *expected_generation* meanwhile.

    It answers once the model is ready, which can take minutes, so only the
    connection is bounded. A refusal raises MeloServeError whose kind is
    melo_serve's ``error.type``: ``stale_generation`` (with the generation it
    is at now), ``busy``, ``does_not_fit``, ``unknown_pick``, ...
    """
    reply = await _call(
        "POST",
        "/switch",
        body={
            "pick": pick,
            "expected_generation": expected_generation,
            "initiator": "localmelo",
        },
        timeout=None,
    )
    if not isinstance(reply, dict):
        raise MeloServeError("melo_serve's answer to /switch is not an object")
    return reply


def _gb(value: float, digits: int = 1) -> str:
    """A size as the rows write it: ``21`` for 21.0, ``2.5``."""
    return f"{round(value, digits):g}"


def pick_words(
    pick: Pick, language: str = "en", *, developer: bool = False
) -> tuple[str, str]:
    """The line under a pick's name, the same on every surface, and its fit word.

    ``Splash · needs 21 GB · fits``. A pick that cannot load has its reason
    where the fit word would be: ``Splash · needs 48 GB, 31 GB free``, or what
    melo_serve says is wrong (``not downloaded yet ...``). It names one memory
    figure, the one that decides the fit; the size on disk only for a
    developer. The fit word (``""`` when none shows) ends the line, so a
    surface can colour it.
    """
    parts = [pick.backend_name] if pick.backend_name else []
    fit = ""
    if pick.problem:
        parts.append(pick.problem)
    elif pick.verdict == TOO_BIG and pick.need_gb is not None:
        need = _gb(pick.need_gb)
        parts.append(
            _words("serve.pick.noRoom", language, need=need)
            if pick.free_gb is None
            else _words(
                "serve.pick.short", language, need=need, free=_gb(pick.free_gb, 0)
            )
        )
    else:
        if pick.need_gb is not None:
            parts.append(_words("serve.pick.needs", language, need=_gb(pick.need_gb)))
        if developer and pick.disk_gb is not None:
            parts.append(_words("serve.pick.disk", language, disk=_gb(pick.disk_gb)))
        word = _FIT_WORDS.get(pick.verdict)
        if word:
            fit = _words(f"serve.fit.{word}", language)
            parts.append(fit)
    return " · ".join(parts), fit


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
    last line it printed (its stderr, else its stdout), the home folder shown as
    ``~``. Not installed: the line says how to install it; a ``melo-serve`` that is
    found but not run (see :func:`refusal`) says why instead. It never raises, so
    the gateway starts either way.
    """
    snap = snapshot()
    if snap.running:
        return _line(snap, role)
    path = command()
    if path is None:
        return refusal() or NOT_INSTALLED
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
