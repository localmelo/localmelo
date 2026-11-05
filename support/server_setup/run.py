"""The first-run setup as one job, for the terminal and the gateway to run.

:func:`plan` asks melo_serve what this Mac can run (``melo-serve plan --json``,
read-only) and :class:`SetupJob` does it after the user's one OK: install
melo_serve, download the picks, register the login services, load the models and
save the config. It has no screen. A UI passes a callback and gets plain events
(:class:`ItemEvent`, then one :class:`Finished`); the line of an event is the
sentence to show.

Every step ends in one plain line: ``installed X into Y``, ``already installed
(V), skipped``, ``updated melo_serve A → B``, ``skipped (you chose to skip it)``
or ``failed: <reason>; nothing was changed``; lines that melo_serve prints
itself (the services, the load, a download that stopped) are passed on.

Everything runs as an argv, never through a shell. What the children print is
data: the plan is read with a size cap and every field is type-checked as
``melo_serve.snapshot`` does it, and an event line that is not what it should
be is dropped.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Final, Literal

from localmelo.support import melo_serve
from localmelo.support.melo_serve import as_count, as_number, as_text
from localmelo.support.sessions.display import home_labelled

PLAN_SCHEMA: Final = "melo_serve.plan/"
UV_HINT: Final = "https://docs.astral.sh/uv/"
NO_UV: Final = f"uv is not installed ({UV_HINT})"
PLAN_TIMEOUT_S: Final = 120.0
VERSION_TIMEOUT_S: Final = 15.0
INSTALL_TIMEOUT_S: Final = 600.0
SERVICE_TIMEOUT_S: Final = 120.0
LOAD_TIMEOUT_S: Final = 900.0
# One event line is a few hundred bytes.
MAX_LINE_BYTES: Final = 16 * 1024
MAX_TEXT_CHARS: Final = 300
STDERR_TAIL_BYTES: Final = 4096
# An id goes into an argv: no leading dash, nothing but these characters.
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}")
_VERSION: Final = re.compile(r"\d+\.\d+\.\d+")

# The rows that are localmelo's own (a plan's items have their own ids).
MELO_SERVE: Final = "melo_serve"
SERVICE_MELO_SERVE: Final = "service:melo_serve"
SERVICE_GATEWAY: Final = "service:gateway"
DOWNLOAD: Final = "download"
LOAD_CHAT: Final = "load:chat"
LOAD_EMBEDDING: Final = "load:embedding"
SAVE: Final = "save"

SKIPPED: Final = "skipped (you chose to skip it)"
STATES: Final = ("running", "done", "skipped", "failed")


def _clip(text: str) -> str:
    """*text* as one line for a terminal: nothing it could act on, the home folder as ``~``."""
    return home_labelled(melo_serve.clean(text)).strip()[:MAX_TEXT_CHARS]


def _failed(reason: str) -> str:
    return f"failed: {reason}; nothing was changed"


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
    lines = [_clip(line) for line in melo_serve.clean(text).splitlines()]
    return next((line for line in reversed(lines) if line), "")


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
    the callback and nothing is kept. Stderr goes to a file whose last line is
    the reason of a failure. *timeout* (None: none) kills the child. *hold* is
    told the child while it runs, and None when it is gone.
    """
    name = os.path.basename(argv[0])
    with tempfile.TemporaryFile() as err:
        try:
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=err,
            )
        except OSError as exc:
            return _Ran(name, why=_clip(f"cannot run {name}: {exc}"))
        assert child.stdout is not None
        why = ""
        out = b""
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            child.kill()

        timer = threading.Timer(timeout, expire) if timeout is not None else None
        if timer:
            timer.start()
        if hold:
            hold(child)
        try:
            if on_line is None:
                out = child.stdout.read(melo_serve.MAX_STATE_BYTES + 1)
                if len(out) > melo_serve.MAX_STATE_BYTES:
                    why = f"{name} printed more than 256 KB"
                    child.kill()
            else:
                for line in _lines(child.stdout):
                    on_line(line)
            code = child.wait()
        finally:
            if timer:
                timer.cancel()
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
            if hold:
                hold(None)
        if expired.is_set():
            why = f"{name} did not finish in {timeout:g} s"
        err.seek(0, os.SEEK_END)
        err.seek(max(0, err.tell() - STDERR_TAIL_BYTES))
        tail = _last_line(err.read().decode("utf-8", errors="replace"))
    return _Ran(name, code, out, tail, why)


# ── uv and melo-serve on this Mac ──


def find_uv() -> str | None:
    """The user's own ``uv``: next to this interpreter (the app bundles one), on PATH, then in the usual folders.

    It is the lookup of :func:`melo_serve.command`, so a file that is not the
    user's, or that others can write, is passed over. An app has no shell PATH,
    hence the folders.
    """
    return melo_serve.find_executable(
        "uv",
        before=(str(Path(sys.executable).parent),),
        after=("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin"),
    )


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


@dataclass(frozen=True)
class Plan:
    """What :func:`plan` found.

    ``available`` is False when there is nothing to offer, and ``reason`` says
    why in one line: no uv, no network, an answer that is not a plan, or a Mac
    that no pick runs on (``hint`` may say what would help). Otherwise ``pick``
    is the recommended chat pick.
    """

    available: bool
    reason: str = ""
    hint: str = ""
    machine: Machine = field(default_factory=Machine)
    pick: Pick | None = None
    items: tuple[Item, ...] = ()
    alternatives: tuple[Pick, ...] = ()
    download_bytes: int = 0
    enough_disk: bool = True


def _unavailable(reason: str, **kwargs: Any) -> Plan:
    return Plan(False, _clip(reason), **kwargs)


def _id(value: object) -> str:
    text = as_text(value)
    return text if _ID.fullmatch(text) else ""


def _pick(entry: object) -> Pick | None:
    if not isinstance(entry, dict):
        return None
    pick = Pick(
        _id(entry.get("id")),
        _clip(as_text(entry.get("name"))),
        _id(entry.get("backend")),
    )
    return pick if pick.id and pick.backend else None


def _item(entry: object) -> Item | None:
    if not isinstance(entry, dict) or not _id(entry.get("id")):
        return None
    text = lambda key: _clip(as_text(entry.get(key)))  # noqa: E731
    return Item(
        _id(entry["id"]),
        text("kind"),
        text("name"),
        text("state"),
        text("via"),
        text("dest"),
        text("role"),
        as_count(entry.get("size_bytes")),
        as_count(entry.get("cached_bytes")),
        text("sha256"),
    )


def _machine(entry: object) -> Machine:
    if not isinstance(entry, dict):
        return Machine()
    text = lambda key: _clip(as_text(entry.get(key)))  # noqa: E731
    return Machine(
        text("chip"),
        as_count(entry.get("ram_gb")),
        text("macos"),
        text("macos_build"),
        as_number(entry.get("free_gb")),
        text("hf_home"),
        text("hf_home_from"),
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
            reason or "no model can run on this Mac", hint=hint, machine=machine
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


def plan(timeout: float = PLAN_TIMEOUT_S) -> Plan:
    """What melo_serve would set up on this Mac; never raises.

    It asks the installed ``melo-serve`` when that is at least the minimum
    version (an older one has no ``plan``), else ``uv tool run`` fetches
    melo_serve's pinned source into uv's cache and runs it there: nothing is
    installed or started. Any failure is an unavailable plan with a one-line
    reason.
    """
    path = melo_serve.command()
    if path and _current(_version(path)):
        argv = [path, "plan", "--json"]
    else:
        uv = find_uv()
        if uv is None:
            return _unavailable(NO_UV)
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
    ran = _run(argv, timeout=timeout)
    if not ran.ok:
        return _unavailable(ran.reason)
    try:
        data = json.loads(ran.out)
    except (ValueError, RecursionError):
        return _unavailable("melo-serve plan did not answer with JSON")
    return _parse(data)


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

    *skip* holds the ids of the items the user turned off: a plan's item ids
    and the localmelo rows (``melo_serve``, ``service:melo_serve``,
    ``service:gateway``). A pick is downloaded unless its model or its backend
    is skipped (the plan lists the embedding model's backend only when it is
    not the chat's). A skipped chat pick is not loaded and the config is not
    saved; a skipped embedding pick leaves memory by words only.
    *gateway_service* is whether to register the gateway's login service (the
    app passes False: it runs its own gateway).

    :meth:`run` blocks, so the gateway runs it in a thread; *on_event* is called
    from that thread. :meth:`cancel` may be called from any thread. ``state`` is
    ``pending``, ``running``, ``done``, ``failed`` or ``cancelled``, and
    ``items`` holds the last event of each row.
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
        self.skip = frozenset(skip)
        self.gateway_service = gateway_service
        self.state = "pending"
        self.items: dict[str, ItemEvent] = {}
        self._on_event = on_event
        self._lock = threading.Lock()
        self._child: subprocess.Popen[bytes] | None = None
        self._cancelled = False
        chat_backend = f"backend:{chat.backend}"
        other = {i.id for i in plan.items if i.kind == "backend"} - {chat_backend}
        embedding = next((i for i in plan.items if i.role == "embedding"), None)
        # (model item id, pick id) of each pick to load; None once skipped.
        self._chat: tuple[str, str] | None = (f"model:{chat.id}", chat.id)
        self._embedding: tuple[str, str] | None = (
            (embedding.id, embedding.id.removeprefix("model:")) if embedding else None
        )
        self._dropped: list[str] = []  # model item ids of picks that are not fetched
        if self._chat[0] in self.skip or chat_backend in self.skip:
            self._dropped.append(self._chat[0])
            self._chat = None
        if self._embedding and (
            self._embedding[0] in self.skip
            or self.skip & other
            or (chat_backend in self.skip and not other)
        ):
            self._dropped.append(self._embedding[0])
            self._embedding = None
        self._skipped_ids = [
            *(i.id for i in plan.items if i.id in self.skip),
            *(i for i in self._dropped if i not in self.skip),
        ]

    # control

    def cancel(self) -> None:
        """End the job ``cancelled``: SIGTERM to the running child, the partial files kept.

        ``melo-serve get`` stops on SIGTERM and keeps what it downloaded.
        """
        with self._lock:
            self._cancelled = True
            if self._child is not None and self._child.poll() is None:
                self._child.terminate()

    def _hold(self, child: subprocess.Popen[bytes] | None) -> None:
        with self._lock:
            self._child = child
            if child is not None and self._cancelled:
                child.terminate()

    def run(self) -> str:
        """Do the steps, in order, until one fails or the job is cancelled; the final state.

        Each step returns the line of its failure, or None.
        """
        self.state = "running"
        steps = (
            self._melo_serve,
            self._download,
            self._services,
            self._load,
            self._save,
        )
        for step in steps:
            if self._cancelled:
                return self._finish("cancelled")
            error = step()
            if error:
                if self._cancelled:
                    return self._finish("cancelled")
                return self._finish("failed", error)
        return self._finish("done")

    # events

    def _emit(self, event: Event) -> None:
        if isinstance(event, ItemEvent):
            self.items[event.item] = event
        if self._on_event:
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
        """*line* as the failure of *item* (a child ended by a cancel is no failure)."""
        if not self._cancelled:
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
        if self._skipped(MELO_SERVE):
            if melo_serve.command():
                return None
            return self._fail(MELO_SERVE, _failed("melo_serve is not installed"))
        path = melo_serve.command()
        have_path = path is not None
        have = _version(path) if path else ""
        if path and _current(have):
            self._item(MELO_SERVE, "done", f"already installed ({have}), skipped")
            return None
        uv = find_uv()
        if uv is None:
            return self._fail(MELO_SERVE, _failed(NO_UV))
        self._item(MELO_SERVE, "running")
        argv = [uv, "tool", "install", "--managed-python", melo_serve.ARCHIVE_URL]
        if have_path:
            argv.insert(3, "--force")
        ran = _run(argv, timeout=INSTALL_TIMEOUT_S, hold=self._hold)
        if not ran.ok:
            return self._fail(MELO_SERVE, _failed(ran.reason))
        path = melo_serve.command()
        if path is None:
            return self._fail(
                MELO_SERVE,
                _failed(
                    _clip(
                        melo_serve.refusal() or "uv put no melo-serve where it is found"
                    )
                ),
            )
        now = _version(path) or "unknown"
        if have_path:
            line = f"updated melo_serve {have or 'unknown'} → {now}"
        else:
            line = f"installed melo_serve {now} into {_clip(str(Path(path).parent.parent))}"
        self._item(MELO_SERVE, "done", line)
        return None

    def _download(self) -> str | None:
        """``melo-serve get`` for the picks not skipped; its events become the job's."""
        for item in self._skipped_ids:
            self._item(item, "skipped", SKIPPED)
        picks = [pick[1] for pick in (self._chat, self._embedding) if pick]
        if not picks:
            return None
        path = melo_serve.command()
        if path is None:
            return self._fail(DOWNLOAD, _failed("melo_serve is not installed"))
        failure = ""

        def on_line(raw: bytes | None) -> None:
            nonlocal failure
            event = _parse_event(raw)
            if event:
                self._emit(event)
                if event.state == "failed":
                    failure = event.line

        ran = _run(
            [path, "get", *picks, "--json"],
            timeout=None,
            hold=self._hold,
            on_line=on_line,
        )
        if ran.ok:
            return None
        if ran.code in (130, -signal.SIGTERM):  # stopped by a TERM or an INT
            self._cancelled = True
        if failure:
            return failure
        if ran.code == 2 and not ran.why:  # cannot start: nothing was touched
            return self._fail(DOWNLOAD, _failed(ran.reason))
        return self._fail(
            DOWNLOAD, f"failed: {ran.reason}; what was downloaded is kept"
        )

    def _services(self) -> str | None:
        """Register the login services: melo_serve's, and the gateway's when asked."""
        if not self._skipped(SERVICE_MELO_SERVE):
            path = melo_serve.command()
            if path is None:
                return self._fail(
                    SERVICE_MELO_SERVE, _failed("melo_serve is not installed")
                )
            self._item(SERVICE_MELO_SERVE, "running")
            ran = _run([path, "install"], timeout=SERVICE_TIMEOUT_S, hold=self._hold)
            if not ran.ok:
                return self._fail(SERVICE_MELO_SERVE, _failed(ran.reason))
            self._item(
                SERVICE_MELO_SERVE,
                "done",
                _last_line(ran.out.decode("utf-8", errors="replace"))
                or "installed melo_serve into its login item",
            )
        if self.gateway_service and not self._skipped(SERVICE_GATEWAY):
            self._item(SERVICE_GATEWAY, "running")
            try:
                from localmelo.support.gateway import daemon

                where = daemon.install()
            except (
                ImportError,
                RuntimeError,
                OSError,
                subprocess.SubprocessError,
            ) as exc:
                return self._fail(SERVICE_GATEWAY, _failed(_clip(str(exc))))
            self._item(
                SERVICE_GATEWAY,
                "done",
                f"installed gateway service into {_clip(where)}",
            )
        return None

    def _load(self) -> str | None:
        """``melo-serve use`` each pick: melo_serve loads it and checks that it answers."""
        for item, picked in (
            (LOAD_CHAT, self._chat),
            (LOAD_EMBEDDING, self._embedding),
        ):
            if picked is None:
                continue
            path = melo_serve.command()
            if path is None:
                return self._fail(item, _failed("melo_serve is not installed"))
            self._item(item, "running")
            ran = _run(
                [path, "use", picked[1]], timeout=LOAD_TIMEOUT_S, hold=self._hold
            )
            if not ran.ok:
                return self._fail(item, _failed(ran.reason))
            line = _last_line(ran.out.decode("utf-8", errors="replace"))
            self._item(item, "done", line or f"loaded {picked[1]}")
        return None

    def _save(self) -> str | None:
        """Save melo_serve's front door as the chat server, with its embedding model unless skipped."""
        if self._chat is None:
            self._item(SAVE, "skipped", SKIPPED)
            return None
        self._item(SAVE, "running")
        try:
            saved = asyncio.run(self._save_async())
        except OSError as exc:
            return self._fail(SAVE, _failed(_clip(str(exc))))
        if not saved:
            return self._fail(
                SAVE,
                _failed("melo_serve is not running or the config cannot be written"),
            )
        self._item(SAVE, "done", "saved melo_serve as the chat server")
        return None

    async def _save_async(self) -> bool:
        from localmelo.support import onboard
        from localmelo.support.backends import discovery

        # With melo_serve running this reads its state and marks its address as
        # melo_serve's, with no request; nothing is asked of the other URLs.
        found = await discovery.detect_local_server(())
        if found is None:
            return False
        return await onboard.save_chat_server(
            found.url,
            "",
            ollama=False,
            embedding_model="" if self._embedding else None,
        )
