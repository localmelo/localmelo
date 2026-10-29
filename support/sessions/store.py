"""The SQLite session store: sessions, turns, submissions and their events.

One implementation serves two files, each with one kind of writer:

- the gateway's file (``writer="gateway"``, :data:`GATEWAY_SESSIONS_DB`),
  which one gateway process owns for the store's life through an advisory
  lock on ``<path>.owner``;
- the CLI file (``writer="cli"``, :data:`CLI_SESSIONS_DB`), which every
  terminal ``melo`` of the user shares, with a lease per running turn.

A file records its writer when it is created, and a write-mode open of the
other writer's file, or of a file that is not a sessions file, is refused
before anything is written to it.

Reliability comes first:

- every state a client can observe is a committed row, and a listener
  hears of an event only after its transaction has committed;
- every write is one transaction (``BEGIN IMMEDIATE`` for more than one
  statement), and a failure or a cancellation rolls it back;
- a process that dies at any point leaves rows that the next open turns
  into a definite status: :meth:`SessionStore.open` recovers the turns
  whose process is gone before it returns;
- every table has a bound: retention clears thinking, old window
  checkpoints and old withdrawals, and deleting a session removes the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import math
import os
import re
import socket
import sqlite3
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

import aiosqlite

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.contracts.events import (
    EVENT_FIELDS,
    EVENT_SCHEMA_VERSION,
    OPTIONAL_EVENT_FIELDS,
    json_safe_payload,
)
from localmelo.melo.memory._sqlite import (
    THINKING_RETENTION_DAYS as THINKING_RETENTION_DAYS,
)
from localmelo.melo.memory._sqlite import apply_pragmas
from localmelo.melo.memory._sqlite import (
    thinking_retention_days as core_thinking_retention_days,
)
from localmelo.melo.schema import (
    ImagePart,
    Message,
    answer_text,
    decode_base64,
    split_thinking,
)
from localmelo.support.sessions.attachments import (
    TextAttachment,
    compose_user_text,
    decode_attachments,
    encode_attachments,
)
from localmelo.support.sessions.display import home_label
from localmelo.support.sessions.submissions import (
    WITHDRAWN_ERROR,
    SubmissionConflict,
    check_client_request_id,
    check_submission,
    text_sha256,
)
from localmelo.support.sessions.window import (
    WINDOW_CODEC_VERSION,
    WINDOW_MAX_CHARS,
    decode_window,
    encode_window,
    has_think_block,
    window_thinking_since,
)

logger = logging.getLogger(__name__)

Writer = Literal["gateway", "cli"]
GATEWAY_SESSIONS_DB: Final = "~/.cache/localmelo/gateway/sessions.db"
CLI_SESSIONS_DB: Final = "~/.cache/localmelo/cli/sessions.db"

# THINKING_RETENTION_DAYS, imported above, is the core's default: the days
# after which retention clears thinking, <think> blocks, window checkpoints
# and withdrawals, as history.db clears its model calls' reasoning.
# Seconds between the gateway's apply_retention() calls.
RETENTION_SWEEP_S: Final = 86_400.0
# Seconds a turn's lease lasts; the runner refreshes it every LEASE_S / 3.
LEASE_S: Final = 30.0
SESSION_KINDS: Final = frozenset({"chat", "front", "front_bg", "task", "worker"})
# Event types the scheduler owns; only a gateway file stores them.
RESERVED_EVENT_PREFIXES: Final = ("task.", "sched.", "attention.")
Cause = Literal["run", "user_stop", "closed", "shutdown", "crash", "takeover", "park"]
CAUSES: Final = frozenset(get_args(Cause))
STALE_ERROR: Final = "The process running this turn stopped before it finished."
# The error of a turn that another host's expired lease left (revision 48).
TAKEOVER_ERROR: Final = "The session was taken over by another process"
RECOVERED_APPROVAL_REASON: Final = "melo stopped before this approval was answered."
# The reason of a request that a turn's end found unanswered: the batch that
# held its approval.resolved was not stored.
ENDED_APPROVAL_REASON: Final = (
    "The turn ended before this approval's answer was stored."
)
MAX_WITHDRAWN_PER_SCOPE: Final = 1_000
# Events per append_events transaction (the runner's writer batches).
WRITER_BATCH_MAX: Final = 200
# The version of the tables below, in PRAGMA user_version.
SCHEMA_VERSION: Final = 2

# The statuses a RunResult can have (docs/run-results.md, "Statuses").
RUN_STATUSES: Final = frozenset(
    {
        "completed",
        "failed",
        "gate_stopped",
        "budget_exhausted",
        "context_budget_exceeded",
        "stuck",
        "provider_error",
        "setup_error",
        "timeout",
        "crashed",
        "cancelled",
    }
)

# The data keys of each session event, as EVENT_FIELDS pins the core
# events'. Within schema 1, types and keys are only ever added.
SESSION_EVENT_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "turn.started": frozenset(
            {"turn", "turn_id", "user_text", "client_request_id"}
        ),
        "turn.finished": frozenset({"turn", "turn_id", "status", "error", "cause"}),
        "turn.undone": frozenset({"turn", "turn_id"}),
        "session.title": frozenset({"title"}),
        "session.updated": frozenset({"model", "workspace", "workspace_label", "meta"}),
        "approval.requested": frozenset(
            {
                "request_id",
                "call_id",
                "tool_name",
                "arguments",
                "preview",
                "workspace_root",
            }
        ),
        "approval.resolved": frozenset(
            {
                "request_id",
                "call_id",
                "tool_name",
                "outcome",
                "origin",
                "reason",
                "remember",
            }
        ),
    }
)
# The session events that a run's writer may store with its batches; the
# store writes the others itself.
APPROVAL_EVENT_TYPES: Final = frozenset({"approval.requested", "approval.resolved"})
# The grammar of an event type with a reserved prefix.
RESERVED_EVENT_TYPE_RE: Final = re.compile(r"[a-z]+(\.[a-z_]+)+")

CLOSED_ERROR: Final = "The session store is closed"
FAILED_ERROR: Final = "The session store could not record a change; restart the gateway"

_TITLE_CHARS: Final = 60
_MAX_TITLE_CHARS: Final = 200
_DAY_S: Final = 86_400.0
# The owner each writer's file names in the "belongs to" error.
_OWNERS: Final[Mapping[str, str]] = MappingProxyType(
    {"gateway": "gateway", "cli": "terminal melo"}
)
_HOST_ID_PATH: Final = "~/.cache/localmelo/cli/host_id"
# How long, and how often, a busy journal-mode change is tried again.
_PRAGMA_RETRY_S: Final = 5.0
_PRAGMA_RETRY_DELAY_S: Final = 0.05

_SCHEMA: Final = (
    """CREATE TABLE IF NOT EXISTS store_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""",
    """CREATE TABLE IF NOT EXISTS schema_components (
    name    TEXT PRIMARY KEY,
    version INTEGER NOT NULL
)""",
    """CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL DEFAULT 'chat',
    title        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    model        TEXT NOT NULL DEFAULT '',
    workspace    TEXT,
    meta         TEXT NOT NULL DEFAULT '{}',
    lease_holder TEXT,
    lease_until  REAL
)""",
    "CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS sessions_workspace ON sessions(workspace, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS sessions_kind ON sessions(kind, updated_at DESC)",
    """CREATE TABLE IF NOT EXISTS turns (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn        INTEGER NOT NULL,
    id          TEXT NOT NULL UNIQUE,
    run_id      TEXT NOT NULL DEFAULT '',
    user_text   TEXT NOT NULL,
    typed_text  TEXT NOT NULL DEFAULT '',
    attachments TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    cause       TEXT NOT NULL DEFAULT '',
    answer      TEXT NOT NULL DEFAULT '',
    result      TEXT,
    holder      TEXT NOT NULL,
    undone      INTEGER NOT NULL DEFAULT 0,
    started_at  REAL NOT NULL,
    finished_at REAL,
    note        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, turn)
)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS turns_running ON turns(session_id) "
    "WHERE status = 'running'",
    """CREATE TABLE IF NOT EXISTS submissions (
    scope             TEXT NOT NULL,
    client_request_id TEXT NOT NULL,
    text_sha256       TEXT NOT NULL,
    state             TEXT NOT NULL,
    session_id        TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    turn              INTEGER,
    error             TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
    PRIMARY KEY (scope, client_request_id),
    FOREIGN KEY (session_id, turn) REFERENCES turns(session_id, turn) ON DELETE CASCADE,
    CHECK ((state = 'accepted' AND session_id IS NOT NULL AND turn IS NOT NULL)
        OR (state = 'withdrawn' AND turn IS NULL))
)""",
    "CREATE INDEX IF NOT EXISTS submissions_turn ON submissions(session_id, turn)",
    "CREATE INDEX IF NOT EXISTS submissions_withdrawn ON submissions(scope, created_at) "
    "WHERE state = 'withdrawn'",
    """CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn       INTEGER,
    type       TEXT NOT NULL,
    event      TEXT NOT NULL,
    FOREIGN KEY (session_id, turn) REFERENCES turns(session_id, turn) ON DELETE CASCADE
)""",
    "CREATE INDEX IF NOT EXISTS events_session ON events(session_id, id)",
    "CREATE INDEX IF NOT EXISTS events_turn ON events(session_id, turn, id)",
    """CREATE TABLE IF NOT EXISTS blobs (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sha256     TEXT NOT NULL,
    mime_type  TEXT NOT NULL,
    data       BLOB NOT NULL,
    PRIMARY KEY (session_id, sha256)
)""",
    """CREATE TABLE IF NOT EXISTS windows (
    session_id     TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    turn           INTEGER NOT NULL,
    version        INTEGER NOT NULL,
    data           TEXT NOT NULL,
    created_at     REAL NOT NULL,
    thinking_since REAL,
    FOREIGN KEY (session_id, turn) REFERENCES turns(session_id, turn) ON DELETE CASCADE
)""",
)


class SessionStoreError(Exception):
    """The session store cannot do what was asked: a file it cannot open, and so on."""


class UnknownSessionError(SessionStoreError):
    """The session was deleted or does not exist."""


class StoreOwnedError(SessionStoreError):
    """The gateway's file has another owner."""


class SessionBusyError(Exception):
    """The session is running a turn."""


# ── Records ──


@dataclass(frozen=True)
class SessionInfo:
    """A session as it reads now.

    ``busy`` says that a turn runs: in a CLI file under a live lease, in the
    gateway's file whatever the lease says, since only its owner runs turns
    there. ``turns`` counts the turns that are neither running nor undone.
    """

    id: str
    kind: str
    title: str
    created_at: float
    updated_at: float
    model: str
    workspace: str | None
    meta: dict[str, Any]
    busy: bool
    turns: int


@dataclass(frozen=True)
class TurnInfo:
    """A turn as it reads now.

    ``status`` is ``"running"`` or a RunResult status, and ``cause`` says
    what ended the turn (``""`` while it runs). ``answer`` and ``result``
    are the stored, redacted copies. ``client_request_id`` is the key of
    the submission that started the turn, or ``None``.
    """

    id: str
    session_id: str
    turn: int
    run_id: str
    user_text: str
    status: str
    cause: str
    answer: str
    result: dict[str, Any] | None
    holder: str
    undone: bool
    started_at: float
    finished_at: float | None
    client_request_id: str | None
    typed_text: str
    attachments: tuple[TextAttachment, ...] | None = None


def check_attachment_submission(
    info: SubmissionInfo | None,
    attachments: Sequence[TextAttachment],
    typed_text: str | None,
) -> None:
    """Disambiguate equal composed messages with different structured snapshots."""
    if info is None or info.turn is None or info.turn.attachments is None:
        return
    saved = info.turn
    if saved.attachments != tuple(attachments) or (
        attachments and saved.typed_text != typed_text
    ):
        raise SubmissionConflict(info)


@dataclass(frozen=True)
class SubmissionInfo:
    """A stored submission: ``accepted`` with its turn, or ``withdrawn``.

    ``turn`` is the accepted turn, read in the same statement, and
    ``turn_id``, ``status`` and ``cause`` are its fields; all four are
    ``None`` for a withdrawal.
    """

    client_request_id: str
    scope: str
    state: str
    session_id: str | None
    text_sha256: str
    turn_id: str | None
    status: str | None
    cause: str | None
    turn: TurnInfo | None
    error: str
    created_at: float


@dataclass(frozen=True)
class NewSession:
    """The session that a first turn creates (see :meth:`SessionStore.begin_turn`)."""

    model: str = ""
    workspace: str | None = None
    meta: Mapping[str, Any] | None = None
    kind: str = "chat"


@dataclass(frozen=True)
class StoredWindow:
    """A session's window checkpoint: taken at the end of ``turn``."""

    turn: int
    version: int
    data: str


@dataclass(frozen=True)
class StoredEvent:
    """One event of a session, with its global id.

    ``event`` is ``AgentEvent.to_dict()`` for a core event (a
    ``run.finished`` with its answer redacted) and the same envelope for a
    session event, with ``seq`` ``None`` and ``run_id`` ``""``. ``turn`` is
    ``None`` for an event that belongs to no turn. ``id`` is ``None`` only
    for an event that a runner forwards without having stored it.
    """

    id: int | None
    session_id: str
    turn: int | None
    event: dict[str, Any]


# ── Helpers ──


def _encodable(value: str | None, name: str) -> None:
    """Refuse a string SQLite cannot store, before it fails mid-turn.

    Python carries a lone surrogate (a JSON body can escape one into a
    request) and sqlite cannot write it. The API refuses it at the body;
    this is the store's own assertion, so no other surface can wedge a turn.
    """
    if value is None:
        return
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc


def title_from_text(text: str) -> str:
    """*text* with its whitespace collapsed, in at most 60 characters.

    A longer text keeps its first 59 characters and ends with ``…``.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= _TITLE_CHARS:
        return collapsed
    return collapsed[: _TITLE_CHARS - 1] + "…"


def check_title(title: object) -> str:
    """*title* stripped, or ``ValueError`` unless it is 1 to 200 characters.

    A surface checks a title with this before it writes anything else, so a
    refused rename changes nothing.
    """
    stripped = title.strip() if isinstance(title, str) else ""
    if not 1 <= len(stripped) <= _MAX_TITLE_CHARS:
        raise ValueError(f"title must be 1 to {_MAX_TITLE_CHARS} characters")
    _encodable(stripped, "title")
    return stripped


def default_path(writer: Writer) -> str:
    """The default file of *writer*, with ``~`` expanded."""
    if writer not in _OWNERS:
        raise ValueError(f"writer must be 'gateway' or 'cli', got {writer!r}")
    return os.path.expanduser(
        GATEWAY_SESSIONS_DB if writer == "gateway" else CLI_SESSIONS_DB
    )


_HOST_IDS: dict[str, str] = {}
# This process's holder suffix, by pid, so that a forked child gets its own.
_HOLDER_SUFFIXES: dict[int, str] = {}


def cli_host_id() -> str:
    """This machine's id for CLI leases, kept in ``~/.cache/localmelo/cli/host_id``.

    A random hex id, made once (mode ``0600``, in a ``0700`` directory),
    which, unlike the host name, does not change with the network. When the
    file cannot be read or made, the host name stands in for it.
    """
    path = os.path.expanduser(_HOST_ID_PATH)
    known = _HOST_IDS.get(path)
    if known is not None:
        return known
    try:
        host_id = _read_or_make_host_id(path)
    except OSError as exc:
        logger.warning("Cannot keep a host id in %s: %s", path, exc)
        host_id = socket.gethostname()
    _HOST_IDS[path] = host_id
    return host_id


def _read_or_make_host_id(path: str) -> str:
    for _ in range(3):
        with contextlib.suppress(FileNotFoundError):
            with open(path, encoding="ascii") as file:
                host_id = file.read().strip()
            if host_id:
                return host_id
        _make_directories(os.path.dirname(path))
        candidate = uuid.uuid4().hex
        temporary = f"{path}.{os.getpid()}.{candidate[:8]}"
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        try:
            os.write(fd, f"{candidate}\n".encode("ascii"))
        finally:
            os.close(fd)
        try:
            # A link never replaces a file: of two processes, the first wins.
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary)
    raise OSError(f"{path} stays empty")


def process_holder(writer: Writer) -> str:
    """This process's lease holder for *writer*'s file, the same for every runner.

    ``"<host>:<pid>:<8 hex>"``, where the host is :func:`cli_host_id` for a
    CLI file and the host name for the gateway's, whose owner lock proves
    liveness instead.
    """
    pid = os.getpid()
    suffix = _HOLDER_SUFFIXES.setdefault(pid, uuid.uuid4().hex[:8])
    host = cli_host_id() if writer == "cli" else socket.gethostname()
    return f"{host}:{pid}:{suffix}"


def _parse_holder(holder: str) -> tuple[str, int] | None:
    """The host and pid of *holder*, or ``None`` when it has no such parts."""
    parts = holder.rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    return (parts[0], pid) if pid > 0 else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: another user's live process
        return True
    return True


def redacted_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of *event* as the store stores it: a ``run.finished`` answer redacted."""
    copied = dict(event)
    if copied.get("type") == "run.finished":
        data = dict(copied.get("data") or {})
        result = data.get("result")
        if isinstance(result, Mapping):
            data["result"] = _redacted_result(result)
        copied["data"] = data
    return copied


def _redacted_result(result: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(result))
    if isinstance(copied.get("result"), str):
        copied["result"] = redact_secrets(copied["result"])[0]
    return copied


async def _rows(
    conn: aiosqlite.Connection, sql: str, args: Sequence[Any] = ()
) -> list[Any]:
    """The rows of one query."""
    return list(await conn.execute_fetchall(sql, args))


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _make_directories(directory: str) -> bool:
    """Create the missing directories of *directory*, each with mode ``0700``.

    Returns whether *directory* itself was missing.
    """
    missing: list[str] = []
    current = directory
    while current and not os.path.isdir(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    for path in reversed(missing):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            continue
        os.chmod(path, 0o700)  # whatever the umask
    return bool(missing)


def _session_event(type_: str, data: Mapping[str, Any], ts: float) -> dict[str, Any]:
    return {
        "v": EVENT_SCHEMA_VERSION,
        "type": type_,
        "run_id": "",
        "seq": None,
        "ts": ts,
        "data": json_safe_payload(data),
    }


def _check_kinds(kinds: Collection[str] | None) -> tuple[str, ...] | None:
    if kinds is None:
        return None
    if isinstance(kinds, str):
        raise ValueError("kinds must be a collection of session kinds, not a str")
    checked = tuple(kinds)
    unknown = sorted(set(checked) - SESSION_KINDS)
    if unknown:
        raise ValueError(f"unknown session kind(s): {', '.join(map(repr, unknown))}")
    return checked


def _check_kind(kind: object) -> str:
    if kind not in SESSION_KINDS:
        raise ValueError(
            f"kind must be one of {', '.join(sorted(SESSION_KINDS))}, got {kind!r}"
        )
    return str(kind)


def _meta_text(meta: Mapping[str, Any] | None) -> str:
    if meta is None:
        return "{}"
    if not isinstance(meta, Mapping):
        raise ValueError(f"meta must be a mapping, got {type(meta).__name__}")
    return _dumps(json_safe_payload(meta))


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _not_a_database(exc: BaseException) -> bool:
    """Whether *exc* says that the file is not an SQLite database."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    return code in (sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_CORRUPT) or (
        type(exc) is sqlite3.DatabaseError
    )


_SESSION_COLUMNS: Final = (
    "s.id, s.kind, s.title, s.created_at, s.updated_at, s.model, s.workspace, "
    "s.meta, s.lease_holder, s.lease_until, "
    "EXISTS (SELECT 1 FROM turns r WHERE r.session_id = s.id AND r.status = 'running'), "
    "(SELECT COUNT(*) FROM turns c WHERE c.session_id = s.id "
    "AND c.status != 'running' AND c.undone = 0)"
)
_TURN_COLUMNS: Final = (
    "t.id, t.session_id, t.turn, t.run_id, t.user_text, t.status, t.cause, "
    "t.answer, t.result, t.holder, t.undone, t.started_at, t.finished_at, "
    "(SELECT a.client_request_id FROM submissions a WHERE a.session_id = t.session_id "
    "AND a.turn = t.turn AND a.state = 'accepted'), t.typed_text, t.attachments"
)
_SUBMISSION_COLUMNS: Final = (
    "u.client_request_id, u.scope, u.state, u.session_id, u.text_sha256, u.error, "
    "u.created_at, u.turn"
)


def _turn_info(row: Sequence[Any]) -> TurnInfo:
    return TurnInfo(
        id=row[0],
        session_id=row[1],
        turn=row[2],
        run_id=row[3],
        user_text=row[4],
        status=row[5],
        cause=row[6],
        answer=row[7],
        result=None if row[8] is None else json.loads(row[8]),
        holder=row[9],
        undone=bool(row[10]),
        started_at=row[11],
        finished_at=row[12],
        client_request_id=row[13],
        typed_text=row[14],
        attachments=decode_attachments(row[15]),
    )


class SessionStore:
    """Sessions, turns, submissions and events in one SQLite file.

    *writer* names the surface whose file this is and has no default, so no
    surface opens the other's file by accident; *path* is that file,
    :func:`default_path` when ``None``. *clock* gives every time the store
    records, and *thinking_retention_days* the age after which
    :meth:`apply_retention` clears thinking. Without it, the store keeps
    thinking as long as ``history.db`` does: the days of
    ``LOCALMELO_THINKING_RETENTION_DAYS`` when it is set, else
    :data:`THINKING_RETENTION_DAYS` (see
    :func:`localmelo.melo.memory._sqlite.thinking_retention_days`), read
    when the store is made; an invalid value of the variable raises
    ``ValueError`` naming it. Nothing touches the disk until the first
    awaited call, which opens the store (see :meth:`open`).

    One ``asyncio.Lock`` serializes the store's statements, reads included,
    so a read sees only committed rows. Listeners (see :meth:`add_listener`)
    hear of each event after its transaction has committed.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        writer: Writer,
        clock: Callable[[], float] = time.time,
        thinking_retention_days: float | None = None,
    ) -> None:
        if writer not in _OWNERS:
            raise ValueError(f"writer must be 'gateway' or 'cli', got {writer!r}")
        if thinking_retention_days is None:
            # The core's value, so sessions.db, history.db and its model
            # calls always agree.
            thinking_retention_days = core_thinking_retention_days()
        if (
            isinstance(thinking_retention_days, bool)
            or not isinstance(thinking_retention_days, int | float)
            or not math.isfinite(thinking_retention_days)
            or thinking_retention_days <= 0
        ):
            raise ValueError(
                "thinking_retention_days must be a finite number > 0, "
                f"got {thinking_retention_days!r}"
            )
        self._writer: Writer = writer
        self._default_dir = os.path.dirname(default_path(writer))
        raw = (
            default_path(writer)
            if path is None
            else os.path.expanduser(os.fspath(path))
        )
        self._path = os.path.abspath(raw)
        self._clock = clock
        self._retention_s = float(thinking_retention_days) * _DAY_S
        self._lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None
        self._owner_fd: int | None = None
        self._failed: BaseException | None = None
        self._closed = False
        self._closing: asyncio.Future[None] | None = None
        self._listeners: list[Callable[[StoredEvent], None]] = []
        self._retention_hooks: list[Callable[[], Awaitable[int]]] = []

    def add_retention_hook(self, hook: Callable[[], Awaitable[int]]) -> None:
        """Run *hook* after every :meth:`apply_retention` sweep.

        A component keeps its own retention in its own transaction; the hook
        returns the rows it cleared. One hook is registered at most once.
        """
        if hook not in self._retention_hooks:
            self._retention_hooks.append(hook)

    @property
    def writer(self) -> Writer:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    @property
    def failed(self) -> BaseException | None:
        """The first write failure of a running turn, in the gateway's file.

        Once set, the store refuses new work until the gateway restarts,
        while reads keep working (see :meth:`append_events`).
        """
        return self._failed

    def retention_cutoff(self) -> float:
        """The wall-clock instant before which retention clears stored rows."""
        return self._clock() - self._retention_s

    # ── Opening and closing ──

    async def open(self) -> None:
        """Open the file, refuse a foreign one, retain and recover.

        Idempotent; every other method awaits it first. In order: create
        the missing directories (``0700``) and the file (``0600``); read the
        version and the writer marker on a read-only connection and refuse a
        foreign file, the other writer's file or a newer schema, having
        written nothing; take the owner lock (gateway only); connect and
        apply the pragmas; create the schema in a new file; run
        :meth:`apply_retention`; and recover the turns whose process is
        gone, which in the gateway's file is every running turn. Each write
        is its own transaction, so a failure rolls back only the step it
        stopped, and the turns recovered before it stay recovered.

        A failure raises :class:`SessionStoreError` (a held owner lock
        :class:`StoreOwnedError`), with an ``OSError`` or ``sqlite3.Error``
        chained, closes what it opened, releases the owner lock, and leaves
        the next call free to try again. After :meth:`aclose` it raises
        ``SessionStoreError("The session store is closed")``.
        """
        async with self._lock:
            await self._ready()

    async def aclose(self) -> None:
        """Close the connection, then release the owner lock; final and idempotent.

        Every later call raises ``SessionStoreError("The session store is
        closed")`` and never opens the file again. The close waits for the
        call under way, an :meth:`open` among them, which then stops at its
        next step; a cancelled ``aclose()`` still closes the store, in the
        background.
        """
        self._closed = True
        if self._closing is None or self._closing.cancelled():
            # A close that was cancelled before it ended, as at a loop's
            # shutdown, is begun again.
            self._closing = asyncio.ensure_future(self._close())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        async with self._lock:
            try:
                await self._discard_connection()
            finally:
                self._release_owner()

    def _stop_if_closed(self) -> None:
        if self._closed:
            raise SessionStoreError(CLOSED_ERROR)

    async def _ready(self) -> aiosqlite.Connection:
        """The open connection; opens the store first. Call it under the lock."""
        self._stop_if_closed()
        if self._conn is not None:
            return self._conn
        try:
            await self._open_steps()
            self._stop_if_closed()  # aclose() came while the last step ran
        except SessionStoreError:
            await self._abandon_open()
            raise
        except (OSError, sqlite3.Error) as exc:
            await self._abandon_open()
            if _not_a_database(exc):
                raise SessionStoreError(
                    f"{self._path} is not a localmelo sessions file"
                ) from exc
            raise SessionStoreError(f"Cannot open {self._path}: {exc}") from exc
        except BaseException:
            await self._abandon_open()
            raise
        assert self._conn is not None
        return self._conn

    async def _abandon_open(self) -> None:
        try:
            await self._discard_connection()
        finally:
            self._release_owner()

    async def _discard_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            await conn.close()

    def _release_owner(self) -> None:
        fd, self._owner_fd = self._owner_fd, None
        if fd is not None:
            os.close(fd)  # releases the flock

    async def _open_steps(self) -> None:
        # 1. Directories and the file.
        self._prepare_directories()
        if not os.path.exists(self._path):
            with contextlib.suppress(FileExistsError):
                fd = os.open(
                    self._path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                )
                os.close(fd)
        # 2. and 3. The checks, on a read-only connection: nothing is written
        # to a file they refuse.
        new_file = await self._check_read_only()
        self._stop_if_closed()
        # 4. The owner lock, before the first write.
        if self._writer == "gateway":
            self._take_owner_lock()
        # The connection that writes; busy_timeout is local to it.
        conn = await aiosqlite.connect(self._path, isolation_level=None)
        self._conn = conn
        await conn.execute("PRAGMA busy_timeout = 5000")
        self._stop_if_closed()
        # 5. Pragmas, on a new file or one with this store's marker only.
        await self._apply_pragmas(conn)
        if self._writer == "gateway":
            await conn.execute("PRAGMA synchronous = FULL")
        await conn.execute("PRAGMA journal_size_limit = 67108864")
        # 6. The schema (T-1), its checks repeated inside the transaction.
        if new_file:
            self._stop_if_closed()
            await self._create_schema(conn)
        else:
            async with self._transaction(conn, "migrate"):
                version = await self._user_version(conn)
                if version == 1:
                    await conn.execute("ALTER TABLE turns ADD COLUMN attachments TEXT")
                    await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        # 7. Retention (T-2) and 8. recovery (T-3), before anything serves.
        self._stop_if_closed()
        await self._retain(conn)
        self._stop_if_closed()
        await self._recover(conn, None, at_open=True)

    async def _check_read_only(self) -> bool:
        """Step 3 on a ``mode=ro`` connection; whether the file is new.

        A read-write connection that closed last would checkpoint a WAL that
        another writer left into the file it refuses; a read-only one never
        writes to the file, though SQLite may create the ``-wal`` and
        ``-shm`` files beside one in WAL mode.
        """
        uri = f"{Path(self._path).as_uri()}?mode=ro"
        conn = await aiosqlite.connect(uri, uri=True, isolation_level=None)
        try:
            await conn.execute("PRAGMA busy_timeout = 5000")
            return await self._check_file(conn)
        except sqlite3.DatabaseError as exc:
            if _not_a_database(exc):
                raise SessionStoreError(
                    f"{self._path} is not a localmelo sessions file"
                ) from exc
            raise
        finally:
            await conn.close()

    @staticmethod
    async def _apply_pragmas(conn: aiosqlite.Connection) -> None:
        """``apply_pragmas``, retried while another connection holds the file.

        ``PRAGMA journal_mode=WAL`` reports SQLITE_BUSY at once, without the
        busy handler, while another process converts the same new file, as
        two terminals that open it at one moment do. The retries stop after
        ``busy_timeout``'s 5 s.
        """
        deadline = time.monotonic() + _PRAGMA_RETRY_S
        while True:
            try:
                await apply_pragmas(conn)
                return
            except sqlite3.OperationalError as exc:
                busy = getattr(exc, "sqlite_errorcode", None) in (
                    sqlite3.SQLITE_BUSY,
                    sqlite3.SQLITE_LOCKED,
                )
                if not busy or time.monotonic() >= deadline:
                    raise
            await asyncio.sleep(_PRAGMA_RETRY_DELAY_S)

    def _prepare_directories(self) -> None:
        parent = os.path.dirname(self._path)
        created = _make_directories(parent)
        if created:
            return
        mode = os.stat(parent)
        loose = mode.st_mode & 0o077
        if parent == self._default_dir:
            if mode.st_uid != os.getuid():
                raise SessionStoreError(
                    f"{parent} belongs to another user; the session store needs a "
                    "directory of its own"
                )
            if loose:
                os.chmod(parent, 0o700)
                logger.info("Set the mode of %s to 0700: it holds transcripts", parent)
        elif loose:
            logger.warning(
                "%s can be read by other users; %s holds transcripts and tool output",
                parent,
                os.path.basename(self._path),
            )

    async def _check_file(self, conn: aiosqlite.Connection) -> bool:
        """Refuse a file this store must not write; return whether it is new."""
        version = await self._user_version(conn)
        objects = await _rows(conn, "SELECT type, name FROM sqlite_master")
        if version == 0:
            if objects:
                raise SessionStoreError(
                    f"{self._path} is not a localmelo sessions file"
                )
            return True
        if version < 0:
            raise SessionStoreError(f"{self._path} is not a localmelo sessions file")
        if version > SCHEMA_VERSION:
            raise SessionStoreError(
                f"{self._path} has schema version {version}; this localmelo reads "
                f"version {SCHEMA_VERSION}"
            )
        await self._check_marker(conn, objects)
        return False

    @staticmethod
    async def _user_version(conn: aiosqlite.Connection) -> int:
        rows = await _rows(conn, "PRAGMA user_version")
        return int(rows[0][0])

    async def _check_marker(
        self, conn: aiosqlite.Connection, objects: Collection[Sequence[Any]]
    ) -> None:
        marker: str | None = None
        if any(row[0] == "table" and row[1] == "store_info" for row in objects):
            rows = await _rows(
                conn, "SELECT value FROM store_info WHERE key = 'writer'"
            )
            marker = rows[0][0] if rows else None
        if marker is None or marker not in _OWNERS:
            raise SessionStoreError(f"{self._path} has no valid writer marker")
        if marker != self._writer:
            raise SessionStoreError(
                f"{self._path} belongs to the {_OWNERS[marker]} (writer '{marker}'); "
                f"open it with writer='{marker}' or use the default path"
            )

    def _take_owner_lock(self) -> None:
        try:
            import fcntl
        except ImportError:  # not POSIX
            raise SessionStoreError(
                "The gateway session store needs POSIX file locks"
            ) from None
        owner_path = f"{self._path}.owner"
        fd = os.open(owner_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = "unknown"
            with contextlib.suppress(OSError, ValueError, IndexError):
                pid = str(int(os.pread(fd, 64, 0).split()[0]))
            os.close(fd)
            raise StoreOwnedError(
                f"Another melo gateway (pid {pid}) is using {self._path}. "
                "Stop it to use the console here."
            ) from None
        except BaseException:
            os.close(fd)
            raise
        self._owner_fd = fd
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()} {socket.gethostname()} {stamp}\n".encode(), 0)

    async def _create_schema(self, conn: aiosqlite.Connection) -> None:
        """T-1: the schema, the writer marker and the version, in one transaction.

        Two processes may both find the file new; the one that gets the
        write lock second finds version 1 here, creates nothing and checks
        the marker instead.
        """
        async with self._transaction(conn, "create"):
            version = await self._user_version(conn)
            objects = await _rows(conn, "SELECT type, name FROM sqlite_master")
            if version == 0 and not objects:
                for statement in _SCHEMA:
                    await conn.execute(statement)
                now = self._clock()
                await conn.execute(
                    "INSERT INTO store_info (key, value) VALUES ('writer', ?), "
                    "('created_at', ?)",
                    (self._writer, repr(now)),
                )
                await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                return
        # Another process created the file first: check it as an existing one.
        await self._check_file(conn)

    # ── Components ──

    @staticmethod
    async def _component_version(conn: aiosqlite.Connection, name: str) -> int | None:
        rows = await _rows(
            conn, "SELECT version FROM schema_components WHERE name = ?", (name,)
        )
        return int(rows[0][0]) if rows else None

    async def ensure_component(
        self, name: str, version: int, statements: Sequence[str]
    ) -> int:
        """Create a named component at *version*, atomically, or check it.

        The gateway's file only: a terminal ``melo`` file never holds the
        scheduler's tables. The read of ``schema_components``, the creation
        of *statements* and the version row are one ``BEGIN IMMEDIATE``
        transaction, so two opens cannot half-create a component. A file
        whose component is newer than *version* is refused with
        :class:`SessionStoreError`, the way a newer ``user_version`` is;
        this release has no migration from an older one. Returns *version*.
        """
        if self._writer != "gateway":
            raise SessionStoreError(
                f"The {name} component needs the gateway's store, not the "
                f"{_OWNERS[self._writer]}'s"
            )
        if not isinstance(name, str) or not name:
            raise ValueError("a component's name must be a non-empty string")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("a component's version must be an integer >= 1")
        if isinstance(statements, str) or not all(
            isinstance(statement, str) and statement.strip() for statement in statements
        ):
            raise ValueError("a component's statements must be SQL strings")
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, f"component_{name}"):
                current = await self._component_version(conn, name)
                if current is None:
                    for statement in statements:
                        await conn.execute(statement)
                    await conn.execute(
                        "INSERT INTO schema_components (name, version) VALUES (?, ?)",
                        (name, version),
                    )
                    return version
                if current > version:
                    raise SessionStoreError(
                        f"{self._path} has {name} schema version {current}; this "
                        f"localmelo reads version {version}"
                    )
                if current < version:
                    raise SessionStoreError(
                        f"{self._path} has {name} schema version {current}; this "
                        f"localmelo cannot migrate it to version {version}"
                    )
                return current

    @contextlib.asynccontextmanager
    async def component_transaction(
        self, label: str
    ) -> AsyncIterator[aiosqlite.Connection]:
        """One ``BEGIN IMMEDIATE`` write transaction for a component's tables.

        The store's lock and write checks apply, so a component's rows commit
        or roll back with the session rows they belong to. The transaction
        stores no session event; a component that needs one appends it after
        this block.
        """
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, label):
                yield conn

    @contextlib.asynccontextmanager
    async def component_event_transaction(
        self, label: str
    ) -> AsyncIterator[tuple[aiosqlite.Connection, list[StoredEvent]]]:
        """Commit component rows and session events in one transaction."""
        if self._writer != "gateway":
            raise SessionStoreError("Out-of-turn events need the gateway's store")
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, label) as stored:
                yield conn, stored

    async def append_component_event(
        self,
        conn: aiosqlite.Connection,
        stored: list[StoredEvent],
        session_id: str,
        event_type: str,
        data: Mapping[str, Any],
    ) -> None:
        """Append an event inside a component event transaction."""
        if not (
            isinstance(event_type, str)
            and event_type.startswith(RESERVED_EVENT_PREFIXES)
            and RESERVED_EVENT_TYPE_RE.fullmatch(event_type)
        ):
            raise ValueError("invalid out-of-turn event type")
        await self._require_session(conn, session_id)
        await self._insert_event(
            conn,
            stored,
            session_id,
            None,
            _session_event(event_type, data, self._clock()),
        )

    # ── Transactions ──

    def _before_commit(self, label: str) -> None:
        """Called by each multi-statement transaction just before its COMMIT.

        A no-op: a seam for the crash tests, which end the process here.
        """

    @contextlib.asynccontextmanager
    async def _transaction(
        self, conn: aiosqlite.Connection, label: str
    ) -> AsyncIterator[list[StoredEvent]]:
        """One ``BEGIN IMMEDIATE`` transaction; yields the list of its events.

        ``BEGIN`` is awaited inside the ``try``, so a ``BEGIN`` that fails or
        is cancelled is rolled back too: aiosqlite still runs it after its
        caller is cancelled, and the rollback queued behind it ends the
        transaction it opens. A rollback that fails is logged, and the
        exception that ended the transaction, a cancellation included,
        propagates. The events appended to the list reach the listeners once
        the transaction has committed.
        """
        stored: list[StoredEvent] = []
        try:
            await conn.execute("BEGIN IMMEDIATE")
            yield stored
            self._before_commit(label)
        except BaseException:
            await self._roll_back(conn, label)
            raise
        await self._commit(conn, stored, label)

    @staticmethod
    async def _roll_back(conn: aiosqlite.Connection, label: str) -> None:
        """ROLLBACK after a failed or cancelled transaction; never raises an Exception.

        The caller re-raises what ended the transaction, which a failed
        rollback must not replace.
        """
        try:
            await conn.rollback()
        except Exception as exc:
            logger.warning(
                "Could not roll back the %s transaction of the session store: %s",
                label,
                exc,
            )

    async def _commit(
        self, conn: aiosqlite.Connection, stored: list[StoredEvent], label: str
    ) -> None:
        commit = asyncio.ensure_future(conn.commit())
        try:
            await asyncio.shield(commit)
        except BaseException:
            # aiosqlite runs a queued COMMIT whether or not this task still
            # waits for it. The ROLLBACK queued behind it ends a transaction
            # that a failed COMMIT leaves open, and does nothing after one
            # that succeeded; the listeners still hear of committed rows.
            try:
                await self._roll_back(conn, label)
            finally:
                if commit.done():
                    self._dispatch_if_committed(commit, stored)
                else:
                    commit.add_done_callback(
                        lambda done: self._dispatch_if_committed(done, stored)
                    )
            raise
        self._dispatch(stored)

    def _dispatch_if_committed(
        self, commit: asyncio.Future[None], stored: list[StoredEvent]
    ) -> None:
        if not commit.cancelled() and commit.exception() is None:
            self._dispatch(stored)

    def _dispatch(self, stored: Sequence[StoredEvent]) -> None:
        for event in stored:
            for listener in list(self._listeners):
                try:
                    listener(event)
                except (Exception, asyncio.CancelledError):
                    logger.exception("A session store listener failed")

    def add_listener(self, fn: Callable[[StoredEvent], None]) -> None:
        """Call *fn* with each stored event, after its transaction commits.

        Once per event, in id order. *fn* must not block; one that raises
        is logged.
        """
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[StoredEvent], None]) -> None:
        """Stop calling *fn*; a listener that was not added is ignored."""
        with contextlib.suppress(ValueError):
            self._listeners.remove(fn)

    async def _insert_event(
        self,
        conn: aiosqlite.Connection,
        stored: list[StoredEvent],
        session_id: str,
        turn: int | None,
        event: dict[str, Any],
    ) -> None:
        cursor = await conn.execute(
            "INSERT INTO events (session_id, turn, type, event) VALUES (?, ?, ?, ?)",
            (session_id, turn, event["type"], _dumps(event)),
        )
        stored.append(StoredEvent(cursor.lastrowid, session_id, turn, event))

    def _check_writable(self) -> None:
        if self._failed is not None:
            raise SessionStoreError(FAILED_ERROR)

    def _fail(self, exc: BaseException) -> None:
        """Fail the gateway's store closed after a lost write of a running turn."""
        if self._writer == "gateway" and self._failed is None:
            self._failed = exc
            logger.error(
                "The session store could not record a running turn's change; it "
                "refuses new work until the gateway restarts: %s",
                exc,
            )

    # ── Leases ──

    def _lease_free(
        self, lease_holder: str | None, lease_until: float | None, now: float
    ) -> bool:
        """Whether a CLI file's lease is free.

        It is free when nobody holds it, when its holder is on this host and
        its process is gone, and otherwise when it has expired. A holder on
        this host whose process lives keeps its lease, whatever
        ``lease_until`` says: after a long sleep the clock passes the lease
        while the process still runs the turn.
        """
        if lease_holder is None:
            return True
        parsed = _parse_holder(lease_holder)
        if parsed is not None and parsed[0] == cli_host_id():
            return not _pid_alive(parsed[1])
        return lease_until is None or lease_until < now

    def _running_is_live(
        self, lease_holder: str | None, lease_until: float | None, now: float
    ) -> bool:
        """Whether a running turn of a session with this lease is still live."""
        if self._writer == "gateway":
            return True  # the owner lock proves it
        return not self._lease_free(lease_holder, lease_until, now)

    async def refresh_lease(self, session_id: str, *, holder: str) -> bool:
        """Extend *holder*'s lease by :data:`LEASE_S`, while its turn still runs.

        Returns whether it did: ``False`` once another process took the
        lease over or recovery ended the turn. A lease that lapsed while
        nobody took it over is extended. One statement (T-7).
        """
        async with self._lock:
            conn = await self._ready()
            cursor = await conn.execute(
                "UPDATE sessions SET lease_until = ? WHERE id = ? AND lease_holder = ? "
                "AND EXISTS (SELECT 1 FROM turns WHERE session_id = sessions.id "
                "AND status = 'running' AND holder = ?)",
                (self._clock() + LEASE_S, session_id, holder, holder),
            )
            return bool(cursor.rowcount)

    # ── Reads ──

    def _session_info(self, row: Sequence[Any], now: float) -> SessionInfo:
        running = bool(row[10])
        return SessionInfo(
            id=row[0],
            kind=row[1],
            title=row[2],
            created_at=row[3],
            updated_at=row[4],
            model=row[5],
            workspace=row[6],
            meta=json.loads(row[7]),
            busy=running and self._running_is_live(row[8], row[9], now),
            turns=int(row[11]),
        )

    async def _get_session(
        self, conn: aiosqlite.Connection, session_id: str
    ) -> SessionInfo | None:
        rows = await _rows(
            conn,
            f"SELECT {_SESSION_COLUMNS} FROM sessions s WHERE s.id = ?",
            (session_id,),
        )
        return self._session_info(rows[0], self._clock()) if rows else None

    async def _require_session(
        self, conn: aiosqlite.Connection, session_id: str
    ) -> tuple[str | None, float | None]:
        """The lease of *session_id*; ``SessionStoreError`` when it is unknown."""
        rows = await _rows(
            conn,
            "SELECT lease_holder, lease_until FROM sessions WHERE id = ?",
            (session_id,),
        )
        if not rows:
            raise UnknownSessionError(f"Unknown session: {session_id}")
        return rows[0][0], rows[0][1]

    async def _get_turn(
        self, conn: aiosqlite.Connection, session_id: str, turn: int
    ) -> TurnInfo | None:
        rows = await _rows(
            conn,
            f"SELECT {_TURN_COLUMNS} FROM turns t WHERE t.session_id = ? AND t.turn = ?",
            (session_id, turn),
        )
        return _turn_info(rows[0]) if rows else None

    async def get_session(self, session_id: str) -> SessionInfo | None:
        async with self._lock:
            conn = await self._ready()
            return await self._get_session(conn, session_id)

    async def change_counters(self, session_id: str, *, holder: str) -> tuple[int, int]:
        """Foreign newest finished turn and undo count, without decoding turn payloads."""
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT COALESCE(MAX(CASE WHEN holder != ? AND status != 'running' "
                "THEN turn ELSE 0 END), 0), COALESCE(SUM(undone), 0) "
                "FROM turns WHERE session_id = ?",
                (holder, session_id),
            )
            await self._require_session(conn, session_id)
            return int(rows[0][0]), int(rows[0][1])

    async def find_sessions(
        self, ref: str, *, kinds: Collection[str] | None = ("chat",)
    ) -> list[SessionInfo]:
        """The sessions *ref* names, newest first.

        The session whose id is *ref*, of any kind; else the sessions whose
        id starts with *ref*, when it has at least 4 characters; else those
        whose title equals *ref*, ignoring case. *kinds* ``None`` means every
        kind.
        """
        checked = _check_kinds(kinds)
        ref = ref.strip()
        if not ref:
            return []
        async with self._lock:
            conn = await self._ready()
            now = self._clock()
            exact = await self._get_session(conn, ref)
            if exact is not None:
                return [exact]
            kind_sql, kind_args = self._kind_filter(checked)
            if len(ref) >= 4:
                # An exact, case-sensitive prefix: LIKE would ignore case.
                rows = await _rows(
                    conn,
                    f"SELECT {_SESSION_COLUMNS} FROM sessions s WHERE "
                    f"substr(s.id, 1, ?) = ? {kind_sql} ORDER BY s.updated_at DESC",
                    (len(ref), ref, *kind_args),
                )
                if rows:
                    return [self._session_info(row, now) for row in rows]
            folded = ref.casefold()
            rows = await _rows(
                conn,
                f"SELECT {_SESSION_COLUMNS} FROM sessions s WHERE s.title != '' "
                f"{kind_sql} ORDER BY s.updated_at DESC",
                kind_args,
            )
            return [
                self._session_info(row, now)
                for row in rows
                if str(row[2]).casefold() == folded
            ]

    @staticmethod
    def _kind_filter(kinds: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]]:
        if kinds is None:
            return "", ()
        if not kinds:
            return "AND 0", ()
        return f"AND s.kind IN ({', '.join('?' * len(kinds))})", kinds

    async def list_sessions(
        self,
        *,
        limit: int = 50,
        query: str = "",
        workspace: str | None = None,
        min_turns: int = 0,
        kinds: Collection[str] | None = ("chat",),
    ) -> list[SessionInfo]:
        """Sessions, the most recently updated first.

        *query* matches the title or any turn's user text (``LIKE``, with
        ``%`` and ``_`` taken literally); *workspace* the stored workspace
        exactly; *min_turns* keeps the sessions with at least that many
        turns, running ones included. *kinds* ``None`` means every kind.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be an int >= 1, got {limit!r}")
        checked = _check_kinds(kinds)
        clauses: list[str] = []
        args: list[Any] = []
        kind_sql, kind_args = self._kind_filter(checked)
        if query:
            pattern = f"%{_escape_like(query)}%"
            clauses.append(
                "AND (s.title LIKE ? ESCAPE '\\' OR EXISTS (SELECT 1 FROM turns q "
                "WHERE q.session_id = s.id AND q.user_text LIKE ? ESCAPE '\\'))"
            )
            args += [pattern, pattern]
        if workspace is not None:
            clauses.append("AND s.workspace = ?")
            args.append(workspace)
        if min_turns > 0:
            # Any turn counts, a running or an undone one too: a session
            # with a turn is a session the list shows, from the moment its
            # first turn's row exists.
            clauses.append(
                "AND (SELECT COUNT(*) FROM turns m WHERE m.session_id = s.id) >= ?"
            )
            args.append(min_turns)
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                f"SELECT {_SESSION_COLUMNS} FROM sessions s WHERE 1 {kind_sql} "
                f"{' '.join(clauses)} ORDER BY s.updated_at DESC LIMIT ?",
                (*kind_args, *args, limit),
            )
            now = self._clock()
            return [self._session_info(row, now) for row in rows]

    async def list_turns(
        self, session_id: str, *, limit: int | None = None, before: int | None = None
    ) -> list[TurnInfo]:
        """A session's turns, oldest first.

        With *limit*, the newest *limit* of them; with *before*, only those
        numbered below it (still the newest *limit* of those).
        """
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError(f"limit must be None or an int >= 1, got {limit!r}")
        sql = f"SELECT {_TURN_COLUMNS} FROM turns t WHERE t.session_id = ?"
        args: list[Any] = [session_id]
        if before is not None:
            sql += " AND t.turn < ?"
            args.append(before)
        sql += " ORDER BY t.turn DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(conn, sql, args)
        return [_turn_info(row) for row in reversed(rows)]

    async def list_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        turns: tuple[int, int] | None = None,
    ) -> list[StoredEvent]:
        """A session's events in id order, after *after_id*, at most *limit* (1..1000).

        With ``turns=(a, b)``, only the events whose turn is in ``a..b``,
        and none that belongs to no turn.
        """
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError(f"limit must be an int from 1 to 1000, got {limit!r}")
        sql = "SELECT id, turn, event FROM events WHERE session_id = ? AND id > ?"
        args: list[Any] = [session_id, after_id]
        if turns is not None:
            sql += " AND turn BETWEEN ? AND ?"
            args += [turns[0], turns[1]]
        sql += " ORDER BY id LIMIT ?"
        args.append(limit)
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(conn, sql, args)
        return [
            StoredEvent(row[0], session_id, row[1], json.loads(row[2])) for row in rows
        ]

    async def latest_event_id(self, session_id: str) -> int:
        """The id of the session's newest stored event, or 0 when it has none.

        A one-shot event read snapshots this id at its request time, then
        serves only events at or below it, so a turn that starts while the
        response pages does not add frames to it.
        """
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT COALESCE(MAX(id), 0) FROM events WHERE session_id = ?",
                (session_id,),
            )
        return int(rows[0][0])

    async def get_blob(self, session_id: str, sha256: str) -> tuple[str, bytes] | None:
        """The MIME type and bytes of an image a turn's tool returned, or ``None``."""
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT mime_type, data FROM blobs WHERE session_id = ? AND sha256 = ?",
                (session_id, sha256),
            )
        return (rows[0][0], bytes(rows[0][1])) if rows else None

    async def get_window(self, session_id: str) -> StoredWindow | None:
        """The session's window checkpoint, or ``None``."""
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT turn, version, data FROM windows WHERE session_id = ?",
                (session_id,),
            )
        return StoredWindow(rows[0][0], rows[0][1], rows[0][2]) if rows else None

    # ── Sessions ──

    async def create_session(
        self,
        *,
        model: str = "",
        workspace: str | None = None,
        meta: Mapping[str, Any] | None = None,
        kind: str = "chat",
    ) -> SessionInfo:
        """A new session without turns (T-4). *workspace* is stored as given."""
        kind = _check_kind(kind)
        meta_text = _meta_text(meta)
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            session_id = uuid.uuid4().hex[:12]
            now = self._clock()
            await conn.execute(
                "INSERT INTO sessions (id, kind, created_at, updated_at, model, "
                "workspace, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, kind, now, now, model, workspace, meta_text),
            )
            info = await self._get_session(conn, session_id)
        assert info is not None
        return info

    async def rename_session(self, session_id: str, title: str) -> bool:
        """Set the title; ``False`` for an unknown session.

        The title is stripped and must have 1 to 200 characters. A rename
        writes ``session.title`` and is never replaced by the automatic
        title (T-12).
        """
        title = check_title(title)
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "rename_session") as stored:
                now = self._clock()
                cursor = await conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, session_id),
                )
                if not cursor.rowcount:
                    return False
                await self._insert_event(
                    conn,
                    stored,
                    session_id,
                    None,
                    _session_event("session.title", {"title": title}, now),
                )
            return True

    async def update_session(
        self,
        session_id: str,
        *,
        model: str | None = None,
        workspace: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> SessionInfo:
        """Change the given fields; ``None`` leaves a field as it is (T-12).

        *meta* replaces the whole object. Writes ``session.updated`` with the
        values after the update.
        """
        meta_text = None if meta is None else _meta_text(meta)
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "update_session") as stored:
                await self._require_session(conn, session_id)
                now = self._clock()
                await conn.execute(
                    "UPDATE sessions SET model = COALESCE(?, model), "
                    "workspace = COALESCE(?, workspace), meta = COALESCE(?, meta), "
                    "updated_at = ? WHERE id = ?",
                    (model, workspace, meta_text, now, session_id),
                )
                info = await self._get_session(conn, session_id)
                assert info is not None
                await self._insert_event(
                    conn,
                    stored,
                    session_id,
                    None,
                    _session_event(
                        "session.updated",
                        {
                            "model": info.model,
                            "workspace": info.workspace,
                            "workspace_label": (
                                ""
                                if info.workspace is None
                                else home_label(info.workspace)
                            ),
                            "meta": info.meta,
                        },
                        now,
                    ),
                )
            return info

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and everything of it; ``False`` for an unknown one.

        :class:`SessionBusyError` while a turn runs (in a CLI file, under a
        live lease). The delete cascades to its turns, events, blobs,
        submissions and window (T-12).
        """
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "delete_session"):
                rows = await _rows(
                    conn,
                    "SELECT s.lease_holder, s.lease_until, r.holder FROM sessions s "
                    "LEFT JOIN turns r ON r.session_id = s.id AND r.status = 'running' "
                    "WHERE s.id = ?",
                    (session_id,),
                )
                if not rows:
                    return False
                lease_holder, lease_until, running = rows[0]
                if running is not None and self._running_is_live(
                    lease_holder, lease_until, self._clock()
                ):
                    raise SessionBusyError(f"Session {session_id} is running a turn")
                await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

    # ── Turns ──

    async def begin_turn(
        self,
        session_id: str | None,
        user_text: str,
        *,
        holder: str,
        title: str | None = None,
        client_request_id: str | None = None,
        new_session: NewSession | None = None,
        create_scope: bool = False,
        attachments: Sequence[TextAttachment] = (),
        typed_text: str | None = None,
    ) -> TurnInfo:
        """Start a turn of *user_text* under *holder*, in one transaction (T-5).

        Takes *session_id*, or *new_session* to create the session with the
        turn. *create_scope* stores the turn's key in the create scope, as
        one begun with *new_session* is stored: that is what a client that
        lost the response to a create looks up without a session id.
        With *client_request_id*, the key decides first: a stored
        submission of it raises ``SubmissionConflict`` (another text),
        ``SubmissionWithdrawn`` or ``SubmissionRepeated`` (with the turn it
        started), and nothing is written. Then a running turn raises
        :class:`SessionBusyError` (in a CLI file, one under a live lease; one
        under a free lease is recovered instead). The turn gets the next
        number, the lease, the session's first title (*title* when it has
        text, else the user text), the submission, and the events
        ``turn.started`` and, with that title, ``session.title``.
        """
        if (session_id is None) == (new_session is None):
            raise ValueError("begin_turn takes a session_id or a new_session, not both")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("text must not be blank")
        _encodable(user_text, "text")
        if len(user_text) > MAX_QUERY_LEN:
            raise ValueError(f"text must be at most {MAX_QUERY_LEN} characters")
        if client_request_id is not None:
            check_client_request_id(client_request_id)
        if not isinstance(holder, str) or not holder:
            raise ValueError("holder must be a non-empty str")
        kind = None if new_session is None else _check_kind(new_session.kind)
        meta_text = None if new_session is None else _meta_text(new_session.meta)
        attachment_data = encode_attachments(attachments)
        if attachments and (
            typed_text is None
            or compose_user_text(typed_text, attachments) != user_text
        ):
            raise ValueError("attachment snapshots must match the composed user text")
        typed = (
            typed_text
            if typed_text is not None
            else (title if isinstance(title, str) and title.strip() else "")
        )
        _encodable(typed, "title")
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "begin_turn") as stored:
                now = self._clock()
                # 1. The key decides before the lease. A create's key is
                # judged in the create scope, where step 8 stores it: a
                # withdrawal recorded while the session's Agent started must
                # answer here, not collide with the insert below.
                if client_request_id is not None:
                    lookup = (
                        None if new_session is not None or create_scope else session_id
                    )
                    found = await self._submission(conn, lookup, client_request_id)
                    check_attachment_submission(found, attachments, typed_text)
                    check_submission(found, user_text)
                # 2. The session.
                if new_session is not None:
                    session_id = uuid.uuid4().hex[:12]
                    await conn.execute(
                        "INSERT INTO sessions (id, kind, created_at, updated_at, model, "
                        "workspace, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            session_id,
                            kind,
                            now,
                            now,
                            new_session.model,
                            new_session.workspace,
                            meta_text,
                        ),
                    )
                assert session_id is not None
                lease_holder, lease_until = await self._require_session(
                    conn, session_id
                )
                # 3. Busy, and 4. recovery of a stale turn (CLI file only).
                running = await _rows(
                    conn,
                    "SELECT turn, holder FROM turns WHERE session_id = ? "
                    "AND status = 'running'",
                    (session_id,),
                )
                if running:
                    if self._running_is_live(lease_holder, lease_until, now):
                        other = running[0][1]
                        if other == holder:
                            raise SessionBusyError(
                                f"Session {session_id} is already running a turn"
                            )
                        raise SessionBusyError(
                            f"Session {session_id} is running a turn in another "
                            f"process ({other})"
                        )
                    await self._recover_turn(conn, stored, session_id, running[0][0])
                # 5. The turn.
                rows = await _rows(
                    conn,
                    "SELECT COALESCE(MAX(turn), 0) + 1, (SELECT title FROM sessions "
                    "WHERE id = ?) FROM turns WHERE session_id = ?",
                    (session_id, session_id),
                )
                number, current_title = int(rows[0][0]), rows[0][1]
                turn_id = uuid.uuid4().hex
                await conn.execute(
                    "INSERT INTO turns (session_id, turn, id, user_text, typed_text, "
                    "attachments, holder, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        number,
                        turn_id,
                        user_text,
                        typed,
                        attachment_data,
                        holder,
                        now,
                    ),
                )
                # 6. The lease, and 7. the first title.
                new_title = None
                if not current_title:
                    new_title = title_from_text(typed or user_text)
                await conn.execute(
                    "UPDATE sessions SET lease_holder = ?, lease_until = ?, "
                    "updated_at = ?, title = COALESCE(?, title) WHERE id = ?",
                    (holder, now + LEASE_S, now, new_title or None, session_id),
                )
                # 8. The submission.
                if client_request_id is not None:
                    await conn.execute(
                        "INSERT INTO submissions (scope, client_request_id, text_sha256, "
                        "state, session_id, turn, created_at) "
                        "VALUES (?, ?, ?, 'accepted', ?, ?, ?)",
                        (
                            (
                                ""
                                if new_session is not None or create_scope
                                else session_id
                            ),
                            client_request_id,
                            text_sha256(user_text),
                            session_id,
                            number,
                            now,
                        ),
                    )
                # 9. The events.
                await self._insert_event(
                    conn,
                    stored,
                    session_id,
                    number,
                    _session_event(
                        "turn.started",
                        {
                            "turn": number,
                            "turn_id": turn_id,
                            "user_text": user_text,
                            "client_request_id": client_request_id,
                        },
                        now,
                    ),
                )
                if new_title:
                    await self._insert_event(
                        conn,
                        stored,
                        session_id,
                        number,
                        _session_event("session.title", {"title": new_title}, now),
                    )
                info = await self._get_turn(conn, session_id, number)
        assert info is not None
        return info

    def _prepared_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """A run's event as :meth:`append_events` stores it, or ``ValueError``."""
        if not isinstance(event, Mapping):
            raise ValueError(f"an event must be a mapping, got {type(event).__name__}")
        type_ = event.get("type")
        data = event.get("data")
        if not isinstance(type_, str) or not isinstance(data, Mapping):
            raise ValueError("an event needs a str 'type' and a mapping 'data'")
        pinned = EVENT_FIELDS.get(type_)
        if pinned is None and type_ in APPROVAL_EVENT_TYPES:
            pinned = SESSION_EVENT_FIELDS[type_]
        if pinned is not None:
            optional = OPTIONAL_EVENT_FIELDS.get(type_, frozenset())
            if not pinned - optional <= set(data) <= pinned:
                raise ValueError(f"a {type_} event has the keys {sorted(pinned)}")
        elif type_.startswith(
            RESERVED_EVENT_PREFIXES
        ) and RESERVED_EVENT_TYPE_RE.fullmatch(type_):
            if self._writer != "gateway":
                raise ValueError(f"{type_} events need the gateway's store")
        else:
            raise ValueError(f"a run's writer cannot store {type_!r} events")
        seq = event.get("seq")
        ts = event.get("ts")
        prepared = redacted_event(
            {
                "v": event.get("v", EVENT_SCHEMA_VERSION),
                "type": type_,
                "run_id": str(event.get("run_id") or ""),
                "seq": (
                    seq if isinstance(seq, int) and not isinstance(seq, bool) else None
                ),
                "ts": ts if isinstance(ts, int | float) else self._clock(),
                "data": dict(data),
            }
        )
        try:
            _dumps(prepared)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"a {type_} event is not JSON-safe: {exc}") from None
        return prepared

    async def _runs_under(
        self, conn: aiosqlite.Connection, session_id: str, turn: int, holder: str
    ) -> bool:
        rows = await _rows(
            conn,
            "SELECT 1 FROM turns WHERE session_id = ? AND turn = ? "
            "AND status = 'running' AND holder = ?",
            (session_id, turn, holder),
        )
        return bool(rows)

    async def append_events(
        self,
        session_id: str,
        turn: int,
        events: Sequence[Mapping[str, Any]],
        *,
        holder: str,
        images: Sequence[ImagePart] = (),
    ) -> list[StoredEvent]:
        """Store one batch of a running turn's events, in one transaction (T-6).

        At most :data:`WRITER_BATCH_MAX` events: core events, the
        ``approval.*`` events and, in the gateway's file, types with a
        reserved prefix. *images* go to ``blobs``, once per digest;
        ``run.started`` sets the turn's ``run_id``; a ``run.finished``
        answer is stored redacted, and so is the copy returned. Returns
        ``[]``, with a WARNING, when the turn no longer runs under *holder*,
        after a takeover or a recovery; an unknown session raises
        :class:`SessionStoreError`. A failed write of the gateway's store
        sets :attr:`failed`.
        """
        batch = list(events)
        if len(batch) > WRITER_BATCH_MAX:
            raise ValueError(f"append_events takes at most {WRITER_BATCH_MAX} events")
        prepared = [self._prepared_event(event) for event in batch]
        async with self._lock:
            conn = await self._ready()
            try:
                async with self._transaction(conn, "append_events") as stored:
                    if not await self._runs_under(conn, session_id, turn, holder):
                        await self._require_session(conn, session_id)
                        logger.warning(
                            "Dropped %d event(s) of turn %d of session %s: it no "
                            "longer runs under %s",
                            len(prepared),
                            turn,
                            session_id,
                            holder,
                        )
                        return []
                    for event in prepared:
                        await self._insert_event(conn, stored, session_id, turn, event)
                        if event["type"] == "run.started" and event["run_id"]:
                            await conn.execute(
                                "UPDATE turns SET run_id = ? WHERE session_id = ? "
                                "AND turn = ?",
                                (event["run_id"], session_id, turn),
                            )
                    for image in images:
                        raw = decode_base64(image.data) if image.data else None
                        if raw is None:
                            continue
                        await conn.execute(
                            "INSERT OR IGNORE INTO blobs (session_id, sha256, mime_type, "
                            "data) VALUES (?, ?, ?, ?)",
                            (session_id, image.sha256, image.mime_type, raw),
                        )
            except (sqlite3.Error, OSError) as exc:
                self._fail(exc)
                raise
            return list(stored)

    async def finish_turn(
        self,
        session_id: str,
        turn: int,
        *,
        holder: str,
        status: str,
        cause: Cause,
        result: Mapping[str, Any] | None,
        answer: str,
        window: Sequence[Message] | None = None,
        error: str | None = None,
        note: str = "",
        on_stored: Callable[[StoredEvent], None] | None = None,
    ) -> TurnInfo:
        """End a turn that runs under *holder*, in one transaction (T-8).

        Stores *status*, *cause*, the redacted *answer* and *result* and the
        time; writes ``approval.resolved`` (``withdrawn``) for each request
        of the turn left unanswered, as when the batch that held its answer
        was lost, then ``turn.finished``, with *error* (by default the
        result's); releases the lease; and, given *window*, replaces the
        session's window checkpoint (thinking older than the retention
        cleared). A checkpoint longer than ``WINDOW_MAX_CHARS`` is not
        written, with a WARNING, and the older one stays. A turn that no
        longer runs under *holder* is left as it is, with a WARNING, and its
        stored row is returned; an unknown session raises
        :class:`SessionStoreError`. A failed write of the gateway's store
        sets :attr:`failed`.
        """
        if cause not in CAUSES:
            raise ValueError(
                f"cause must be one of {', '.join(sorted(CAUSES))}, got {cause!r}"
            )
        if status not in RUN_STATUSES:
            raise ValueError(f"status must be a RunResult status, got {status!r}")
        result_copy = None if result is None else _redacted_result(result)
        result_text = None if result_copy is None else _dumps(result_copy)
        if error is None:
            error = str(result_copy.get("error") or "") if result_copy else ""
        stored_answer = redact_secrets(answer)[0]
        stored_events: list[StoredEvent] = []
        async with self._lock:
            conn = await self._ready()
            try:
                async with self._transaction(conn, "finish_turn") as stored:
                    if not await self._runs_under(conn, session_id, turn, holder):
                        await self._require_session(conn, session_id)
                        info = await self._get_turn(conn, session_id, turn)
                        if info is None:
                            raise SessionStoreError(
                                f"Unknown turn {turn} of session {session_id}"
                            )
                        logger.warning(
                            "Did not finish turn %d of session %s: it no longer runs "
                            "under %s",
                            turn,
                            session_id,
                            holder,
                        )
                        return info
                    now = self._clock()
                    await self._end_turn(
                        conn,
                        stored,
                        session_id,
                        turn,
                        holder=holder,
                        status=status,
                        cause=cause,
                        answer=stored_answer,
                        note=note,
                        result=result_text,
                        error=error,
                        now=now,
                        approval_reason=ENDED_APPROVAL_REASON,
                    )
                    await conn.execute(
                        "UPDATE sessions SET updated_at = ? WHERE id = ?",
                        (now, session_id),
                    )
                    if window is not None:
                        await self._store_window(conn, session_id, turn, window, now)
                    info = await self._get_turn(conn, session_id, turn)
                stored_events = list(stored)
            except (sqlite3.Error, OSError) as exc:
                self._fail(exc)
                raise
        assert info is not None
        if on_stored is not None:
            for event in stored_events:
                on_stored(event)
        return info

    async def _end_turn(
        self,
        conn: aiosqlite.Connection,
        stored: list[StoredEvent],
        session_id: str,
        turn: int,
        *,
        holder: str,
        status: str,
        cause: str,
        answer: str,
        note: str,
        result: str | None,
        error: str,
        now: float,
        approval_reason: str,
    ) -> None:
        """The end of a running turn: its outcome, ``turn.finished``, the lease.

        First, each ``approval.requested`` of the turn without an
        ``approval.resolved`` gets one, ``withdrawn`` with *approval_reason*,
        so every request is answered once (S-3) also when the batch that
        held its answer was lost.
        """
        rows = await _rows(
            conn,
            "SELECT id FROM turns WHERE session_id = ? AND turn = ?",
            (session_id, turn),
        )
        events = await _rows(
            conn,
            "SELECT type, event FROM events WHERE session_id = ? AND turn = ? "
            "AND type IN ('approval.requested', 'approval.resolved') ORDER BY id",
            (session_id, turn),
        )
        open_requests: dict[str, Mapping[str, Any]] = {}
        for type_, text in events:
            try:
                data = json.loads(text)["data"]
            except (ValueError, KeyError, TypeError):
                continue
            if type_ == "approval.requested":
                open_requests[str(data.get("request_id"))] = data
            else:
                open_requests.pop(str(data.get("request_id")), None)
        for request_id, request in open_requests.items():
            await self._insert_event(
                conn,
                stored,
                session_id,
                turn,
                _session_event(
                    "approval.resolved",
                    {
                        "request_id": request_id,
                        "call_id": request.get("call_id", ""),
                        "tool_name": request.get("tool_name", ""),
                        "outcome": "withdrawn",
                        "origin": "withdrawn",
                        "reason": approval_reason,
                        "remember": False,
                    },
                    now,
                ),
            )
        await conn.execute(
            "UPDATE turns SET status = ?, cause = ?, answer = ?, result = ?, "
            "note = ?, finished_at = ? WHERE session_id = ? AND turn = ? "
            "AND status = 'running'",
            (status, cause, answer, result, note, now, session_id, turn),
        )
        await self._insert_event(
            conn,
            stored,
            session_id,
            turn,
            _session_event(
                "turn.finished",
                {
                    "turn": turn,
                    "turn_id": rows[0][0],
                    "status": status,
                    "error": error,
                    "cause": cause,
                },
                now,
            ),
        )
        await conn.execute(
            "UPDATE sessions SET lease_holder = NULL, lease_until = NULL "
            "WHERE id = ? AND lease_holder = ?",
            (session_id, holder),
        )

    async def _store_window(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        turn: int,
        window: Sequence[Message],
        now: float,
    ) -> None:
        cutoff = now - self._retention_s
        try:
            data = encode_window(window, thinking_cutoff=cutoff)
        except Exception:
            logger.exception(
                "Could not encode the window of turn %d of session %s", turn, session_id
            )
            return
        if len(data) > WINDOW_MAX_CHARS:
            logger.warning(
                "Did not store the window of turn %d of session %s: %d characters, "
                "over %d",
                turn,
                session_id,
                len(data),
                WINDOW_MAX_CHARS,
            )
            return
        await conn.execute(
            "INSERT OR REPLACE INTO windows (session_id, turn, version, data, "
            "created_at, thinking_since) VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn,
                WINDOW_CODEC_VERSION,
                data,
                now,
                window_thinking_since(window, thinking_cutoff=cutoff),
            ),
        )

    async def undo_last_turn(self, session_id: str) -> TurnInfo | None:
        """Mark the newest finished turn that is not undone as undone (T-9).

        Returns it, or ``None`` when there is none. Writes ``turn.undone``
        and deletes the session's window checkpoint when it covers that
        turn. :class:`SessionBusyError` while a turn runs.
        """
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "undo_last_turn") as stored:
                lease_holder, lease_until = await self._require_session(
                    conn, session_id
                )
                now = self._clock()
                running = await _rows(
                    conn,
                    "SELECT turn FROM turns WHERE session_id = ? AND status = 'running'",
                    (session_id,),
                )
                if running:
                    if self._running_is_live(lease_holder, lease_until, now):
                        raise SessionBusyError(
                            f"Session {session_id} is already running a turn"
                        )
                    await self._recover_turn(conn, stored, session_id, running[0][0])
                rows = await _rows(
                    conn,
                    "SELECT turn, id FROM turns WHERE session_id = ? AND status != "
                    "'running' AND undone = 0 ORDER BY turn DESC LIMIT 1",
                    (session_id,),
                )
                if not rows:
                    return None
                number, turn_id = rows[0]
                await conn.execute(
                    "UPDATE turns SET undone = 1 WHERE session_id = ? AND turn = ?",
                    (session_id, number),
                )
                # turn.undone belongs to no turn: no event of a turn follows
                # its turn.finished.
                await self._insert_event(
                    conn,
                    stored,
                    session_id,
                    None,
                    _session_event(
                        "turn.undone", {"turn": number, "turn_id": turn_id}, now
                    ),
                )
                await conn.execute(
                    "DELETE FROM windows WHERE session_id = ? AND turn >= ?",
                    (session_id, number),
                )
                await conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )
                info = await self._get_turn(conn, session_id, number)
            return info

    # ── Submissions ──

    async def _submission(
        self, conn: aiosqlite.Connection, session_id: str | None, client_request_id: str
    ) -> SubmissionInfo | None:
        """The submission of a key by the lookup rule, with its turn, or ``None``.

        ``None`` for *session_id* reads the create scope; a session id reads
        its own scope and the create-scope row whose session it is.
        """
        args: tuple[str, ...]
        if session_id is None:
            where = "u.scope = '' AND u.client_request_id = ?"
            args = (client_request_id,)
        else:
            where = (
                "u.client_request_id = ? AND (u.scope = ? OR (u.scope = '' "
                "AND u.session_id = ?))"
            )
            args = (client_request_id, session_id, session_id)
        rows = await _rows(
            conn,
            f"SELECT {_SUBMISSION_COLUMNS}, {_TURN_COLUMNS} FROM submissions u "
            "LEFT JOIN turns t ON t.session_id = u.session_id AND t.turn = u.turn "
            f"WHERE {where}",
            args,
        )
        if not rows:
            return None
        row = rows[0]
        turn = (
            _turn_info(row[8:]) if row[7] is not None and row[8] is not None else None
        )
        return SubmissionInfo(
            client_request_id=row[0],
            scope=row[1],
            state=row[2],
            session_id=row[3],
            text_sha256=row[4],
            turn_id=None if turn is None else turn.id,
            status=None if turn is None else turn.status,
            cause=None if turn is None else turn.cause,
            turn=turn,
            error=row[5],
            created_at=row[6],
        )

    async def get_submission(
        self, session_id: str | None, client_request_id: str
    ) -> SubmissionInfo | None:
        """The submission of *client_request_id*, with its turn, or ``None`` (unknown).

        *session_id* ``None`` reads the create scope; a session id also
        finds the create-scope submission that created that session. Raises
        ``ValueError`` for a malformed key, and :class:`SessionStoreError`
        for an unknown session. Never writes.
        """
        check_client_request_id(client_request_id)
        async with self._lock:
            conn = await self._ready()
            if session_id is not None:
                await self._require_session(conn, session_id)
            return await self._submission(conn, session_id, client_request_id)

    async def withdraw_submission(
        self, session_id: str | None, client_request_id: str, *, user_text: str
    ) -> SubmissionInfo:
        """Withdraw a message that was not accepted, in one transaction (T-10).

        A stored submission of the key with another text raises
        ``SubmissionConflict``; otherwise it is returned as it is, accepted
        or withdrawn, and the caller stops an accepted turn that still runs.
        Without one, a withdrawal is stored, with no turn and no event, and
        the scope's oldest withdrawals beyond
        :data:`MAX_WITHDRAWN_PER_SCOPE` are deleted; a later
        :meth:`begin_turn` with the key raises ``SubmissionWithdrawn``.
        """
        check_client_request_id(client_request_id)
        digest = text_sha256(user_text)
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "withdraw_submission"):
                found = await self._submission(conn, session_id, client_request_id)
                if found is not None:
                    if found.text_sha256 != digest:
                        raise SubmissionConflict(found)
                    return found
                if session_id is not None:
                    await self._require_session(conn, session_id)
                scope = session_id or ""
                await conn.execute(
                    "INSERT INTO submissions (scope, client_request_id, text_sha256, "
                    "state, session_id, turn, error, created_at) "
                    "VALUES (?, ?, ?, 'withdrawn', ?, NULL, ?, ?)",
                    (
                        scope,
                        client_request_id,
                        digest,
                        session_id,
                        WITHDRAWN_ERROR,
                        self._clock(),
                    ),
                )
                await conn.execute(
                    "DELETE FROM submissions WHERE rowid IN (SELECT rowid FROM "
                    "submissions WHERE scope = ? AND state = 'withdrawn' "
                    "ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?)",
                    (scope, MAX_WITHDRAWN_PER_SCOPE),
                )
                withdrawn = await self._submission(conn, session_id, client_request_id)
        assert withdrawn is not None
        return withdrawn

    # ── Out-of-turn events ──

    async def append_session_event(
        self, session_id: str, type: str, data: Mapping[str, Any]
    ) -> StoredEvent:
        """Store one event that belongs to no turn, for the scheduler (T-11).

        *type* must start with one of :data:`RESERVED_EVENT_PREFIXES` and
        match ``[a-z]+(\\.[a-z_]+)+``, else ``ValueError``; only the gateway's
        file stores them. *data* passes ``json_safe_payload``. Takes no lease
        and does not change ``updated_at``.
        """
        if not (
            isinstance(type, str)
            and type.startswith(RESERVED_EVENT_PREFIXES)
            and RESERVED_EVENT_TYPE_RE.fullmatch(type)
        ):
            raise ValueError(
                "an out-of-turn event's type must start with one of "
                f"{', '.join(RESERVED_EVENT_PREFIXES)}, got {type!r}"
            )
        if self._writer != "gateway":
            raise SessionStoreError("Out-of-turn events need the gateway's store")
        event = _session_event(type, data, self._clock())
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            async with self._transaction(conn, "append_session_event") as stored:
                await self._require_session(conn, session_id)
                await self._insert_event(conn, stored, session_id, None, event)
            return stored[0]

    # ── Retention ──

    async def apply_retention(self) -> int:
        """Clear what is older than ``thinking_retention_days``; return the rows changed.

        In one transaction (T-2): the ``thinking`` and inline ``<think>``
        block of ``assistant.message`` events; the ``<think>`` block of the
        final answer's copies (``run.finished`` events, ``turns.answer`` and
        ``turns.result``); window checkpoints, and in the remaining ones the
        thinking of messages older than the cutoff; and withdrawals.
        :meth:`open` runs it too.
        """
        async with self._lock:
            conn = await self._ready()
            self._check_writable()
            changed = await self._retain(conn)
        # A component's hook opens its own transaction, so it runs after the
        # lock is free and a failure names that component, not this sweep.
        for hook in list(self._retention_hooks):
            changed += await hook()
        return changed

    async def _retain(self, conn: aiosqlite.Connection) -> int:
        cutoff = self._clock() - self._retention_s
        changed = 0
        async with self._transaction(conn, "apply_retention"):
            # 1. Thinking of assistant.message events, by their ts.
            rows = await _rows(
                conn,
                "SELECT id, json_extract(event, '$.data.content'), "
                "json_extract(event, '$.data.thinking') FROM events "
                "WHERE type = 'assistant.message' AND json_extract(event, '$.ts') < ? "
                "AND (json_extract(event, '$.data.thinking') != '' "
                "OR json_extract(event, '$.data.content') LIKE '%think>%')",
                (cutoff,),
            )
            for event_id, content, thinking in rows:
                # The LIKE above also finds a "think>" that is no block.
                cleared = content
                if isinstance(content, str) and has_think_block(content):
                    cleared = split_thinking(content)[1]
                if cleared == content and not thinking:
                    continue
                await conn.execute(
                    "UPDATE events SET event = json_set(event, '$.data.thinking', '', "
                    "'$.data.content', ?) WHERE id = ?",
                    (cleared, event_id),
                )
                changed += 1
            # 2. The final answer's copies: run.finished by ts, turns by finished_at.
            rows = await _rows(
                conn,
                "SELECT id, json_extract(event, '$.data.result.result') FROM events "
                "WHERE type = 'run.finished' AND json_extract(event, '$.ts') < ? "
                "AND json_extract(event, '$.data.result.result') LIKE '%think>%'",
                (cutoff,),
            )
            for event_id, answer in rows:
                if isinstance(answer, str) and has_think_block(answer):
                    await conn.execute(
                        "UPDATE events SET event = json_set(event, "
                        "'$.data.result.result', ?) WHERE id = ?",
                        (split_thinking(answer)[1], event_id),
                    )
                    changed += 1
            rows = await _rows(
                conn,
                "SELECT session_id, turn, status, answer, "
                "json_extract(result, '$.result') FROM turns WHERE finished_at < ? "
                "AND (answer LIKE '%think>%' OR json_extract(result, '$.result') "
                "LIKE '%think>%')",
                (cutoff,),
            )
            for session_id, turn, status, answer, result_answer in rows:
                touched = False
                if has_think_block(answer):
                    cleared = answer_text(status, split_thinking(answer)[1])
                    if cleared != answer:
                        await conn.execute(
                            "UPDATE turns SET answer = ? WHERE session_id = ? "
                            "AND turn = ?",
                            (cleared, session_id, turn),
                        )
                        touched = True
                if isinstance(result_answer, str) and has_think_block(result_answer):
                    await conn.execute(
                        "UPDATE turns SET result = json_set(result, '$.result', ?) "
                        "WHERE session_id = ? AND turn = ?",
                        (split_thinking(result_answer)[1], session_id, turn),
                    )
                    touched = True
                changed += touched
            # 3. Window checkpoints: old rows go, newer ones lose old thinking.
            cursor = await conn.execute(
                "DELETE FROM windows WHERE created_at < ?", (cutoff,)
            )
            changed += max(cursor.rowcount, 0)
            rows = await _rows(
                conn,
                "SELECT session_id, data, thinking_since FROM windows "
                "WHERE thinking_since < ?",
                (cutoff,),
            )
            for session_id, data, since in rows:
                messages = decode_window(data)
                if messages is None:
                    await conn.execute(
                        "DELETE FROM windows WHERE session_id = ?", (session_id,)
                    )
                    changed += 1
                    continue
                new_data = encode_window(messages, thinking_cutoff=cutoff)
                new_since = window_thinking_since(messages, thinking_cutoff=cutoff)
                if (new_data, new_since) == (data, since):
                    continue
                await conn.execute(
                    "UPDATE windows SET data = ?, thinking_since = ? "
                    "WHERE session_id = ?",
                    (new_data, new_since, session_id),
                )
                changed += 1
            # 4. Withdrawals.
            cursor = await conn.execute(
                "DELETE FROM submissions WHERE state = 'withdrawn' AND created_at < ?",
                (cutoff,),
            )
            changed += max(cursor.rowcount, 0)
        return changed

    # ── Recovery ──

    async def recover_stale_turns(self, session_id: str | None = None) -> int:
        """End the stale running turns, of one session or of all; return how many.

        A turn is stale in a CLI file under a free lease; in the gateway's
        file only inside :meth:`open`, which runs this for the whole file,
        so here none is (a call at startup is harmless). One transaction per
        turn (T-3): ``approval.resolved`` (``withdrawn``) for each request
        nobody answered, the outcome, ``turn.finished``, and the lease
        cleared. Idempotent: a second call changes nothing.
        """
        async with self._lock:
            conn = await self._ready()
            return await self._recover(conn, session_id, at_open=False)

    async def _recover(
        self, conn: aiosqlite.Connection, session_id: str | None, *, at_open: bool
    ) -> int:
        if self._writer == "gateway" and not at_open:
            return 0  # every running turn is this process's own
        sql = "SELECT session_id, turn FROM turns WHERE status = 'running'"
        args: tuple[str, ...] = ()
        if session_id is not None:
            sql += " AND session_id = ?"
            args = (session_id,)
        rows = await _rows(conn, sql + " ORDER BY session_id, turn", args)
        ended = 0
        for stale_session, turn in rows:
            async with self._transaction(conn, "recover") as stored:
                lease = await _rows(
                    conn,
                    "SELECT lease_holder, lease_until FROM sessions WHERE id = ?",
                    (stale_session,),
                )
                if (
                    self._writer == "cli"
                    and lease
                    and not self._lease_free(lease[0][0], lease[0][1], self._clock())
                ):
                    continue
                if await self._recover_turn(conn, stored, stale_session, turn):
                    ended += 1
        return ended

    async def _recover_turn(
        self,
        conn: aiosqlite.Connection,
        stored: list[StoredEvent],
        session_id: str,
        turn: int,
    ) -> bool:
        """End a stale running turn; ``False`` when it no longer runs.

        The approvals nobody answered are withdrawn (in :meth:`_end_turn`).
        A stored ``run.finished`` gives the outcome; else, in a CLI file, a
        lease of another host that expired is a takeover (``cancelled``,
        cause ``takeover``), and any other stale turn is a crash
        (``crashed``, cause ``crash``).
        """
        rows = await _rows(
            conn,
            "SELECT t.holder, s.lease_holder FROM turns t JOIN sessions s "
            "ON s.id = t.session_id WHERE t.session_id = ? AND t.turn = ? "
            "AND t.status = 'running'",
            (session_id, turn),
        )
        if not rows:
            return False
        holder, lease_holder = rows[0]
        now = self._clock()
        finished: Mapping[str, Any] | None = None
        for (text,) in await _rows(
            conn,
            "SELECT event FROM events WHERE session_id = ? AND turn = ? "
            "AND type = 'run.finished' ORDER BY id",
            (session_id, turn),
        ):
            with contextlib.suppress(ValueError, KeyError, TypeError):
                result = json.loads(text)["data"]["result"]
                if isinstance(result, Mapping):
                    finished = result
        status = None if finished is None else finished.get("status")
        outcome: tuple[str, str, str, str | None, str]
        if finished is not None and status in RUN_STATUSES:
            assert isinstance(status, str)
            answer = finished.get("result")
            outcome = (
                status,
                "crash" if status == "cancelled" else "run",
                answer_text(status, answer if isinstance(answer, str) else ""),
                _dumps(dict(finished)),
                str(finished.get("error") or ""),
            )
        elif self._another_hosts(lease_holder):
            outcome = (
                "cancelled",
                "takeover",
                answer_text("cancelled", ""),
                None,
                TAKEOVER_ERROR,
            )
        else:
            outcome = (
                "crashed",
                "crash",
                answer_text("crashed", ""),
                None,
                STALE_ERROR,
            )
        status, cause, answer, result_text, error = outcome
        await self._end_turn(
            conn,
            stored,
            session_id,
            turn,
            holder=holder,
            status=status,
            cause=cause,
            answer=answer,
            note="",
            result=result_text,
            error=error,
            now=now,
            approval_reason=RECOVERED_APPROVAL_REASON,
        )
        logger.info(
            "Recovered turn %d of session %s (%s, cause %s)",
            turn,
            session_id,
            status,
            cause,
        )
        return True

    def _another_hosts(self, lease_holder: str | None) -> bool:
        """Whether a CLI file's lease is held under another host id."""
        if self._writer != "cli" or lease_holder is None:
            return False
        parsed = _parse_holder(lease_holder)
        return parsed is not None and parsed[0] != cli_host_id()
