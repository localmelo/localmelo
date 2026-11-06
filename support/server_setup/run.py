"""The first-run setup as one job, for the terminal and the gateway to run.

:func:`plan` asks melo_serve what this Mac can run (``melo-serve plan --json``,
read-only) and :class:`SetupJob` does it after the user's one OK: install
melo_serve, download the picks, register the login services, load the models and
save the config. It has no screen. A UI passes a callback and gets plain events
(:class:`ItemEvent`, then one :class:`Finished`); the line of an event is the
sentence to show.

Every step that runs ends in one plain line: ``installed X into Y``, ``already installed
(V), skipped``, ``updated melo_serve A → B``, ``skipped (you chose to skip it)``
or ``failed: <reason>``, which says what is left when something was already
done (``nothing was changed`` only while that is true); lines that melo_serve
prints itself (the services, the load, a download that stopped) are passed on.

Everything runs as an argv, never through a shell. What the children print is
data: the plan is read with a size cap and every field is type-checked as
``melo_serve.snapshot`` does it, and an event line that is not what it should
be is dropped. :func:`plan` never raises, and :meth:`SetupJob.run` always ends
with one :class:`Finished`: an error is a ``failed`` job, not an exception.
No child outlives a cancel or a timeout for long: it gets SIGTERM, then SIGKILL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any, Final, Literal

from localmelo.support import melo_serve
from localmelo.support.melo_serve import as_count, as_number, as_text

PLAN_SCHEMA: Final = "melo_serve.plan/"
UV_HINT: Final = "https://docs.astral.sh/uv/"
UV_FOLDERS: Final = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin")
NO_UV: Final = f"uv is not installed ({UV_HINT})"
PLAN_TIMEOUT_S: Final = 120.0
VERSION_TIMEOUT_S: Final = 15.0
INSTALL_TIMEOUT_S: Final = 600.0
# melo-serve install may wait 180 s for a detached daemon to stop, then 20 s for the service.
SERVICE_TIMEOUT_S: Final = 240.0
LOAD_TIMEOUT_S: Final = 900.0
# One event line is a few hundred bytes.
MAX_LINE_BYTES: Final = 16 * 1024
MAX_TEXT_CHARS: Final = 300
STDERR_TAIL_BYTES: Final = 4096
# After SIGTERM a child has this long before SIGKILL: ``melo-serve get`` gives its
# own children 30 s, and may be detaching a daemon.
CANCEL_GRACE_S: Final = 60.0
# Once a child has ended, its output pipes get this long to run dry.
PIPE_GRACE_S: Final = 2.0
# An id goes into an argv: no leading dash, nothing but these characters.
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}")
_VERSION: Final = re.compile(r"(?<!\d)\d{1,9}\.\d{1,9}\.\d{1,9}(?!\d)")

# The rows that are localmelo's own (a plan's items have their own ids).
MELO_SERVE: Final = "melo_serve"
SERVICE_MELO_SERVE: Final = "service:melo_serve"
SERVICE_GATEWAY: Final = "service:gateway"
DOWNLOAD: Final = "download"
LOAD_CHAT: Final = "load:chat"
LOAD_EMBEDDING: Final = "load:embedding"
SAVE: Final = "save"

SKIPPED: Final = "skipped (you chose to skip it)"
NOTHING: Final = "nothing was changed"
LOADED: Final = "the models are loaded; nothing was saved"
STOPPED: Final = "stopped (you cancelled it)"
STATES: Final = ("running", "done", "skipped", "failed")


def _clip(text: str) -> str:
    """*text* as one line for a terminal: nothing it could act on, the home folder as ``~``."""
    one = " ".join(melo_serve.clean(text).split())[:MAX_TEXT_CHARS]
    return one.encode("utf-8", "replace").decode()  # a lone surrogate cannot be printed


def _failed(reason: str, left: str = "") -> str:
    """``failed: <reason>``, and *left*: what is true now, such as ``nothing was changed``."""
    return f"failed: {reason}; {left}" if left else f"failed: {reason}"


# ── Running a child ──


@dataclass(frozen=True)
class _Ran:
    """How a child ended. ``why`` is empty, or what stopped it (not its exit code)."""

    name: str
    code: int | None = None
    out: bytes = b""
    err: str = ""
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.why

    @property
    def reason(self) -> str:
        """One line for why the command failed."""
        if self.why:
            return self.why
        if self.err:
            return self.err
        return _last_line(self.out.decode("utf-8", errors="replace")) or (
            f"{self.name} exited {self.code}"
        )


def _last_line(text: str) -> str:
    """The last line of *text* that has anything in it, as :func:`_clip` leaves it."""
    return next((line for line in map(_clip, reversed(text.splitlines())) if line), "")


def _lines(stream: IO[bytes]) -> Iterator[bytes | None]:
    """The lines of *stream*; None for a line over the cap (the rest of it is dropped)."""
    while raw := stream.readline(MAX_LINE_BYTES + 1):
        if len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n"):
            while (rest := stream.readline(MAX_LINE_BYTES + 1)) and not rest.endswith(
                b"\n"
            ):
                pass
            yield None
        else:
            yield raw


def _stop(child: subprocess.Popen[bytes]) -> None:
    """SIGTERM to *child* now, and SIGKILL after :data:`CANCEL_GRACE_S` if it is still there."""
    child.terminate()
    timer = threading.Timer(CANCEL_GRACE_S, child.kill)
    timer.daemon = True
    timer.start()


def _chunks(stream: IO[bytes]) -> Iterator[bytes]:
    """What arrives on *stream*, as it arrives, up to its end."""
    while chunk := os.read(stream.fileno(), 65536):
        yield chunk


def _run(
    argv: list[str],
    *,
    timeout: float | None,
    hold: Callable[[subprocess.Popen[bytes] | None], None] | None = None,
    on_line: Callable[[bytes | None], None] | None = None,
) -> _Ran:
    """Run *argv* with no input and its output read in a bounded way.

    Without *on_line*, stdout is kept up to ``melo_serve.MAX_STATE_BYTES``; more
    stops the child. With it, each stdout line (None: one over the cap) goes to
    the callback, from a thread of its own, and nothing is kept; the callback is
    not called once this returns. Stderr is kept in memory as its last
    :data:`STDERR_TAIL_BYTES`; its last line is the reason of a failure.
    *timeout* (None: none) kills the child. *hold* is told the child while it
    runs, and None when it is gone.

    The child ends when it ends, not when its pipes do: a process it left
    behind that holds a pipe is waited for :data:`PIPE_GRACE_S` and then
    ignored. Ctrl-C waits for the child, which got the same SIGINT; any other
    way out (an error in *hold*) stops it with SIGTERM, then SIGKILL.
    """
    name = os.path.basename(argv[0])
    try:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return _Ran(name, why=_clip(f"cannot run {name}: {exc}"))
    assert child.stdout is not None and child.stderr is not None
    stdout, stderr = child.stdout, child.stderr
    why = ""
    out = b""
    tail = b""
    open_ = threading.Lock()  # held while on_line runs, and to close the gate
    closed = False
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        child.kill()

    def pump_stdout() -> None:
        nonlocal out, why
        with stdout:
            if on_line is None:
                for chunk in _chunks(stdout):
                    out += chunk
                    if len(out) > melo_serve.MAX_STATE_BYTES:
                        why = f"{name} printed more than 256 KB"
                        child.kill()
                        break
                return
            for line in _lines(stdout):
                with open_:
                    if closed:
                        return
                    on_line(line)

    def pump_stderr() -> None:
        nonlocal tail
        with stderr:
            for chunk in _chunks(stderr):
                tail = (tail + chunk)[-STDERR_TAIL_BYTES:]

    pumps = [threading.Thread(target=f) for f in (pump_stdout, pump_stderr)]
    timer = threading.Timer(timeout, expire) if timeout is not None else None
    for thread in (*pumps, *([timer] if timer else [])):
        thread.daemon = True  # one that is left behind must not hold the process
        thread.start()
    try:
        if hold:
            hold(child)
        code = child.wait()
    except KeyboardInterrupt:  # the terminal gave the child the same SIGINT
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(CANCEL_GRACE_S)
        raise
    finally:
        if timer:
            timer.cancel()
        if child.poll() is None:
            _stop(child)
            child.wait()
        if hold:
            hold(None)
        deadline = time.monotonic() + PIPE_GRACE_S
        for pump in pumps:
            pump.join(max(0.0, deadline - time.monotonic()))
        with open_:
            closed = True
    if expired.is_set():
        why = f"{name} did not finish in {timeout:g} s"
    return _Ran(
        name, code, out, _last_line(tail.decode("utf-8", errors="replace")), why
    )


# ── uv and melo-serve on this Mac ──


def find_uv() -> str | None:
    """The user's own ``uv``: next to this interpreter, on PATH, then in :data:`UV_FOLDERS`.

    It is the lookup of :func:`melo_serve.command`, so a file that is not the
    user's, or that others can write, is passed over. An app has no shell PATH,
    hence the folders. The path returned is the file the links lead to (what was
    checked), so a shim that acts on the name it is run by would see another.
    """
    return melo_serve.find_executable("uv", **_uv_places())


def _uv_places() -> dict[str, tuple[str, ...]]:
    return {"before": (str(Path(sys.executable).parent),), "after": UV_FOLDERS}


def uv_problem() -> tuple[str, str]:
    """Why there is no uv to run, as ``(code, line)``: found but refused (``uv_refused``), else not installed (``uv``)."""
    why = melo_serve.refusal("uv", **_uv_places())
    return ("uv_refused", why) if why else ("uv", NO_UV)


def _version(path: str) -> str:
    """The version ``melo-serve --version`` says, or ``""``."""
    ran = _run([path, "--version"], timeout=VERSION_TIMEOUT_S)
    found = (
        _VERSION.search(ran.out.decode("utf-8", errors="replace")) if ran.ok else None
    )
    return found.group(0) if found else ""


def _numbers(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _current(version: str) -> bool:
    """Whether *version* is at least :data:`melo_serve.MIN_VERSION`."""
    return bool(version) and _numbers(version) >= _numbers(melo_serve.MIN_VERSION)


# ── The plan ──


@dataclass(frozen=True)
class Machine:
    chip: str = ""
    ram_gb: int = 0
    macos: str = ""
    macos_build: str = ""
    free_gb: float = 0.0
    hf_home: str = ""
    hf_home_from: str = ""
    homebrew: bool = False


@dataclass(frozen=True)
class Pick:
    id: str
    name: str
    backend: str


@dataclass(frozen=True)
class Item:
    """One thing the plan would set up. ``state`` is ``done``, ``partial`` or ``todo``."""

    id: str
    kind: str
    name: str
    state: str
    via: str = ""
    dest: str = ""
    role: str = ""
    size_bytes: int = 0
    cached_bytes: int = 0
    sha256: str = ""
    backend: str = (
        ""  # a model's backend, as ``backend:<this>`` names it; "" if not said
    )


@dataclass(frozen=True)
class Plan:
    """What :func:`plan` found.

    ``available`` is False when there is nothing to offer, and ``reason`` says
    why in one line: no uv, no network, an answer that is not a plan, or a Mac
    that no pick runs on (``hint`` may say what would help). ``code`` names the
    cause for a screen that words it itself: ``uv`` (not installed),
    ``uv_refused`` (found, not run: ``reason`` says why), ``unsupported``
    (melo_serve says no pick runs here: ``reason`` and ``hint`` are its words)
    or ``failed``. Otherwise ``pick`` is the recommended chat pick.
    """

    available: bool
    reason: str = ""
    hint: str = ""
    code: str = ""
    machine: Machine = field(default_factory=Machine)
    pick: Pick | None = None
    items: tuple[Item, ...] = ()
    alternatives: tuple[Pick, ...] = ()
    download_bytes: int = 0
    enough_disk: bool = True


def _unavailable(reason: str, code: str = "failed", **kwargs: Any) -> Plan:
    return Plan(False, _clip(reason), code=code, **kwargs)


def _id(value: object) -> str:
    text = as_text(value)
    return text if _ID.fullmatch(text) else ""


def _text(entry: dict[str, Any], key: str) -> str:
    return _clip(as_text(entry.get(key)))


def _pick(entry: object) -> Pick | None:
    if not isinstance(entry, dict):
        return None
    pick = Pick(
        _id(entry.get("id")),
        _text(entry, "name"),
        _id(entry.get("backend")),
    )
    return pick if pick.id and pick.backend else None


def _item(entry: object) -> Item | None:
    if not isinstance(entry, dict) or not _id(entry.get("id")):
        return None
    return Item(
        _id(entry["id"]),
        _text(entry, "kind"),
        _text(entry, "name"),
        _text(entry, "state"),
        _text(entry, "via"),
        _text(entry, "dest"),
        _text(entry, "role"),
        as_count(entry.get("size_bytes")),
        as_count(entry.get("cached_bytes")),
        _text(entry, "sha256"),
        _id(entry.get("backend")),
    )


def _machine(entry: object) -> Machine:
    if not isinstance(entry, dict):
        return Machine()
    return Machine(
        _text(entry, "chip"),
        as_count(entry.get("ram_gb")),
        _text(entry, "macos"),
        _text(entry, "macos_build"),
        as_number(entry.get("free_gb")),
        _text(entry, "hf_home"),
        _text(entry, "hf_home_from"),
        entry.get("homebrew") is True,
    )


def _parse(data: object) -> Plan:
    """The plan *data* says, or an unavailable one. *data* is a ``melo_serve.plan/1`` object."""
    if not isinstance(data, dict) or not as_text(data.get("schema")).startswith(
        PLAN_SCHEMA
    ):
        return _unavailable("melo-serve plan did not answer with a plan")
    machine = _machine(data.get("machine"))
    reason = _clip(as_text(data.get("reason")))
    hint = _clip(as_text(data.get("reason_hint")))
    pick = _pick(data.get("pick"))
    if data.get("supported") is not True or pick is None:
        return _unavailable(
            reason or "no model can run on this Mac",
            "unsupported",
            hint=hint,
            machine=machine,
        )
    entries = data.get("items")
    alternatives = data.get("alternatives")
    return Plan(
        True,
        machine=machine,
        pick=pick,
        items=tuple(
            item
            for item in map(_item, entries if isinstance(entries, list) else [])
            if item
        ),
        alternatives=tuple(
            alt
            for alt in map(
                _pick, alternatives if isinstance(alternatives, list) else []
            )
            if alt
        ),
        download_bytes=as_count(data.get("download_bytes")),
        enough_disk=data.get("enough_disk") is not False,
    )


def installed_version() -> str:
    """The version of the ``melo-serve`` on this Mac when it is at least the minimum, else ``""``."""
    path = melo_serve.command()
    have = _version(path) if path else ""
    return have if _current(have) else ""


def plan(timeout: float = PLAN_TIMEOUT_S, pick: str = "") -> Plan:
    """What melo_serve would set up on this Mac; never raises.

    It asks the installed ``melo-serve`` when that is at least the minimum
    version (an older one has no ``plan``), else ``uv tool run`` fetches
    melo_serve's pinned source into uv's cache and runs it there: nothing is
    installed or started. With *pick* (a chat pick's id) that pick is planned
    instead of the recommended one. Any failure is an unavailable plan with a
    one-line reason.
    """
    try:
        return _plan(timeout, pick)
    except Exception as exc:  # this is a promise to a screen: no exception
        return _unavailable(f"melo-serve plan failed: {exc}")


def _plan(timeout: float, pick: str) -> Plan:
    if pick and not _id(pick):
        return _unavailable("that is not a model melo_serve knows")
    path = melo_serve.command()
    if path and _current(_version(path)):
        argv = [path, "plan", "--json"]
    else:
        uv = find_uv()
        if uv is None:
            code, why = uv_problem()
            return _unavailable(why, code)
        argv = [
            uv,
            "tool",
            "run",
            "--python",
            sys.executable,
            "--from",
            melo_serve.ARCHIVE_URL,
            "melo-serve",
            "plan",
            "--json",
        ]
    if pick:
        argv += ["--pick", pick]
    ran = _run(argv, timeout=timeout)
    if not ran.ok:
        return _unavailable(ran.reason)
    try:
        data = json.loads(ran.out)
    except (ValueError, RecursionError):
        return _unavailable("melo-serve plan did not answer with JSON")
    return _parse(data)


def embedding_item(plan: Plan) -> Item | None:
    """The plan's embedding model, when its id can go into an argv."""
    item = next((i for i in plan.items if i.role == "embedding"), None)
    return item if item and _id(item.id.removeprefix("model:")) else None


def role_items(plan: Plan, chat: Pick) -> tuple[frozenset[str], frozenset[str]]:
    """The ids of the items that *chat* needs, and those that the embedding model needs."""
    embedding = embedding_item(plan)
    return (
        frozenset({f"model:{chat.id}", f"backend:{chat.backend}"}),
        frozenset(
            {
                embedding.id,
                *([f"backend:{embedding.backend}"] if embedding.backend else []),
            }
            if embedding
            else ()
        ),
    )


# ── The job ──


@dataclass(frozen=True)
class ItemEvent:
    """One row of the setup: its id, ``running``, ``done``, ``skipped`` or ``failed``, bytes and the line to show."""

    item: str
    state: str
    line: str = ""
    done_bytes: int = 0
    total_bytes: int = 0


@dataclass(frozen=True)
class Finished:
    """The last event: ``done``, ``failed`` or ``cancelled``, and for a failure its line."""

    state: Literal["done", "failed", "cancelled"]
    error: str = ""


Event = ItemEvent | Finished


def _parse_event(raw: bytes | None) -> ItemEvent | None:
    """The event a line of ``melo-serve get --json`` says; None when it is not one."""
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    item, state = _id(data.get("item")), data.get("state")
    if not item or state not in STATES:
        return None
    return ItemEvent(
        item,
        state,
        _clip(as_text(data.get("line"))),
        as_count(data.get("done_bytes")),
        as_count(data.get("total_bytes")),
    )


class SetupJob:
    """Sets up what *plan* describes for *pick*, after the user's consent.

    *skip* holds the ids of the items the user turned off: the ids of the
    plan's items and the localmelo rows (``melo_serve``, ``service:melo_serve``,
    ``service:gateway``). ``melo-serve get`` leaves out the skipped items of this
    run, the model and the backend of *pick* and of the plan's embedding model;
    others are not passed (``get`` refuses an item that is not part of its run).
    A pick whose model or backend is skipped is not loaded. Then a chat pick
    leaves the config unsaved and an embedding pick leaves memory by words only.
    *gateway_service* is whether to register the gateway's login service (the
    app passes False: it runs its own gateway, and so must any job that runs
    inside it, since installing the service restarts it).

    The object is single-use: :meth:`run` blocks, so the gateway runs it in a
    thread, and a second call raises. *on_event* may be called from two threads,
    though never after :meth:`run` has returned: the thread of :meth:`run`, and,
    while ``melo-serve get`` runs, a thread of its own that reads its progress.
    It must be thread-safe and quick, and an exception it raises is ignored.
    :meth:`cancel` may be called from any thread. ``state`` is ``pending``,
    ``running``, ``done``, ``failed`` or ``cancelled``, and ``items`` holds the
    last event of each row (a new dict on every event, so it can be read from
    another thread).
    """

    def __init__(
        self,
        plan: Plan,
        pick: str,
        skip: Iterable[str] = (),
        *,
        gateway_service: bool = False,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        chat = next(
            (p for p in (plan.pick, *plan.alternatives) if p and p.id == pick), None
        )
        if chat is None:
            raise ValueError(
                f"{pick!r} is not the plan's pick or one of its alternatives"
            )
        self.pick = chat
        self.skip = frozenset(skip)
        self.gateway_service = gateway_service
        self.state = "pending"
        self.items: dict[str, ItemEvent] = {}
        self._on_event = on_event
        self._lock = threading.RLock()
        self._child: subprocess.Popen[bytes] | None = None
        self._cancelled = False
        self._killed = False
        self._path = ""  # the melo-serve command, once the first step has found it
        embedding = embedding_item(plan)
        picked = embedding.id.removeprefix("model:") if embedding else ""
        chat_items, embedding_items = role_items(plan, chat)
        self._picks = [chat.id, *([picked] if embedding else [])]
        # The pick to load for each role; None once its model or backend is skipped.
        self._chat = None if self.skip & chat_items else chat.id
        self._embedding = (
            None if self.skip & embedding_items or not embedding else picked
        )
        self._get_skips = sorted(self.skip & (chat_items | embedding_items))

    @property
    def steps(self) -> tuple[str, ...]:
        """The rows this job adds after the plan's items: each model it loads, then the config it saves."""
        return (
            *([LOAD_CHAT] if self._chat is not None else []),
            *([LOAD_EMBEDDING] if self._embedding is not None else []),
            *([SAVE] if self._chat is not None else []),
        )

    # control

    def cancel(self) -> None:
        """End the job ``cancelled``: SIGTERM to the running child, the partial files kept.

        ``melo-serve get`` stops on SIGTERM and keeps what it downloaded. A child
        that is still there :data:`CANCEL_GRACE_S` later gets SIGKILL.
        """
        with self._lock:
            self._cancelled = True
            if self._child is not None and self._child.poll() is None:
                _stop(self._child)

    def kill(self) -> None:
        """Cancel, and SIGKILL the running child now: for a child that ignores :meth:`cancel`."""
        with self._lock:
            self._cancelled = self._killed = True
            if self._child is not None and self._child.poll() is None:
                self._child.kill()

    def _hold(self, child: subprocess.Popen[bytes] | None) -> None:
        with self._lock:
            self._child = child
            if child is not None and self._killed:
                child.kill()
            elif child is not None and self._cancelled:
                _stop(child)

    def run(self) -> str:
        """Do the steps, in order, until one fails or the job is cancelled; the final state.

        Each step returns the line of its failure, or None. An error of any
        kind ends the job ``failed`` and Ctrl-C ends it ``cancelled``; either
        way :class:`Finished` is the last event and nothing is raised. A second
        call raises: make a new job.
        """
        with self._lock:
            if self.state != "pending":
                raise RuntimeError("a SetupJob runs once; make a new one")
            self.state = "running"
        steps = (
            self._melo_serve,
            self._download,
            self._services,
            self._load,
            self._save,
        )
        state: Literal["done", "failed", "cancelled"] = "done"
        error = ""
        try:
            for step in steps:
                if self._cancelled:
                    state = "cancelled"
                    break
                error = step() or ""
                if error:
                    state = "cancelled" if self._cancelled else "failed"
                    break
        except KeyboardInterrupt:  # the child got it too, and has been waited for
            state = "cancelled"
        except Exception as exc:
            state, error = "failed", f"failed: {_clip(str(exc)) or type(exc).__name__}"
        return self._finish(state, error if state == "failed" else "")

    # events

    def _emit(self, event: Event) -> None:
        if isinstance(event, ItemEvent):
            self.items = {**self.items, event.item: event}
        if self._on_event:
            # A screen that went away stops no download.
            with contextlib.suppress(Exception):
                self._on_event(event)

    def _item(self, item: str, state: str, line: str = "") -> None:
        self._emit(ItemEvent(item, state, line))

    def _finish(
        self, state: Literal["done", "failed", "cancelled"], error: str = ""
    ) -> str:
        self.state = state
        self._emit(Finished(state, error))
        return state

    def _fail(self, item: str, line: str) -> str:
        """*line* as the failure of *item*; on a cancel the row is ``skipped``, not failed."""
        if self._cancelled:
            self._item(item, "skipped", STOPPED)
        else:
            self._item(item, "failed", line)
        return line

    def _skipped(self, item: str) -> bool:
        if item in self.skip:
            self._item(item, "skipped", SKIPPED)
            return True
        return False

    # steps

    def _melo_serve(self) -> str | None:
        """Install melo_serve, or update it below the minimum version, or leave it."""
        path = melo_serve.command()
        have = _version(path) if path else ""
        if self._skipped(MELO_SERVE):
            if path is None:
                return self._fail(
                    MELO_SERVE, _failed("melo_serve is not installed", NOTHING)
                )
            if not _current(have):
                return self._fail(
                    MELO_SERVE,
                    _failed(
                        f"{_clip(path)} is {have or 'of unknown version'}, but {melo_serve.MIN_VERSION} or later is needed",
                        NOTHING,
                    ),
                )
            self._path = path
            return None
        if path and _current(have):
            self._path = path
            self._item(MELO_SERVE, "done", f"already installed ({have}), skipped")
            return None
        uv = find_uv()
        if uv is None:
            return self._fail(MELO_SERVE, _failed(uv_problem()[1], NOTHING))
        self._item(MELO_SERVE, "running")
        argv = [uv, "tool", "install", "--managed-python", melo_serve.ARCHIVE_URL]
        if path:
            argv.insert(3, "--force")
        ran = _run(argv, timeout=INSTALL_TIMEOUT_S, hold=self._hold)
        if not ran.ok:
            return self._fail(MELO_SERVE, _failed(ran.reason, NOTHING))
        found = melo_serve.command()
        if found is None:
            why = melo_serve.refusal() or "uv put no melo-serve where it is found"
            return self._fail(MELO_SERVE, _failed(_clip(why), "uv has installed it"))
        now = _version(found)
        if not _current(now):  # an older one comes first on PATH
            return self._fail(
                MELO_SERVE,
                _failed(
                    f"{_clip(found)} is {now or 'of unknown version'} and is found "
                    f"first, but {melo_serve.MIN_VERSION} or later is needed; remove it",
                    "uv has installed the new one",
                ),
            )
        self._path = found
        if path:
            line = f"updated melo_serve {have or 'unknown'} → {now}"
        else:
            line = f"installed melo_serve {now} into {_clip(str(Path(found).parent.parent))}"
        self._item(MELO_SERVE, "done", line)
        return None

    def _download(self) -> str | None:
        """``melo-serve get`` without the skipped items; its events become the job's."""
        failed: ItemEvent | None = None

        def on_line(raw: bytes | None) -> None:
            nonlocal failed
            event = _parse_event(raw)
            if event is None:
                return
            if event.state == "failed":  # ``get`` reports a stop as a failure too
                failed = event  # its exit code says which, below
            else:
                self._emit(event)

        ran = _run(
            [
                self._path, "get", *self._picks,
                *(arg for item in self._get_skips for arg in ("--skip", item)),
                "--json",
            ],  # fmt: skip
            timeout=None,
            hold=self._hold,
            on_line=on_line,
        )
        if ran.code in (130, -signal.SIGTERM):  # stopped by a TERM or an INT
            self._cancelled = True
        if failed:
            state = "skipped" if self._cancelled else "failed"
            self._emit(replace(failed, state=state))
        if ran.ok:
            return None
        if failed:
            return failed.line
        if ran.code == 2 and not ran.why:  # cannot start: nothing was touched
            return self._fail(DOWNLOAD, _failed(ran.reason, NOTHING))
        return self._fail(DOWNLOAD, _failed(ran.reason, "what was downloaded is kept"))

    def _services(self) -> str | None:
        """Register the login services: melo_serve's, and the gateway's when asked."""
        if not self._skipped(SERVICE_MELO_SERVE):
            self._item(SERVICE_MELO_SERVE, "running")
            ran = _run(
                [self._path, "install"], timeout=SERVICE_TIMEOUT_S, hold=self._hold
            )
            if not ran.ok:
                return self._fail(SERVICE_MELO_SERVE, _failed(ran.reason))
            self._item(
                SERVICE_MELO_SERVE,
                "done",
                _last_line(ran.out.decode("utf-8", errors="replace"))
                or "installed melo_serve into its login item",
            )
        if self._cancelled:
            return None
        if self.gateway_service and not self._skipped(SERVICE_GATEWAY):
            self._item(SERVICE_GATEWAY, "running")
            try:
                from localmelo.support.gateway import daemon

                installed = daemon.install()
            except (
                ImportError,
                RuntimeError,
                OSError,
                subprocess.SubprocessError,
            ) as exc:
                return self._fail(SERVICE_GATEWAY, _failed(_clip(str(exc))))
            self._item(SERVICE_GATEWAY, "done", _clip(installed.line))
        return None

    def _load(self) -> str | None:
        """``melo-serve use`` each pick: melo_serve loads it and checks that it answers."""
        for item, picked in (
            (LOAD_CHAT, self._chat),
            (LOAD_EMBEDDING, self._embedding),
        ):
            if picked is None:
                continue
            self._item(item, "running")
            ran = _run(
                [self._path, "use", picked], timeout=LOAD_TIMEOUT_S, hold=self._hold
            )
            if not ran.ok:
                serving = (
                    "the chat model is serving; "
                    if item == LOAD_EMBEDDING and self._chat
                    else ""
                )
                return self._fail(
                    item, _failed(ran.reason, f"{serving}nothing was saved")
                )
            line = _last_line(ran.out.decode("utf-8", errors="replace"))
            self._item(item, "done", line or f"loaded {picked}")
        return None

    def _save(self) -> str | None:
        """Save melo_serve's front door as the chat server, with its embedding model unless skipped."""
        if self._chat is None:
            self._item(SAVE, "skipped", SKIPPED)
            return None
        self._item(SAVE, "running")
        try:
            saved = asyncio.run(self._save_async())
        except Exception as exc:  # a config.toml that cannot be read or written
            return self._fail(
                SAVE, _failed(f"could not save the config: {_clip(str(exc))}", LOADED)
            )
        if not saved:
            return self._fail(SAVE, _failed("melo_serve is not running", LOADED))
        self._item(SAVE, "done", "saved melo_serve as the chat server")
        return None

    async def _save_async(self) -> bool:
        from localmelo.support import config
        from localmelo.support.backends import discovery

        # With melo_serve running this reads its state and marks its address as
        # melo_serve's, with no request; nothing is asked of the other URLs.
        found = await discovery.detect_local_server(())
        if found is None:
            return False
        cfg = config.load()
        cfg.use_chat_server(
            found.url, "", embedding_model="" if self._embedding else None
        )
        if not self._embedding:  # memory goes by words only
            cfg.embedding_backend = "none"
        # Not onboard's save_chat_server: a job has no screen to print on.
        config.save(cfg)
        return True
