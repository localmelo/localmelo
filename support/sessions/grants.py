"""Every "allow ... for this chat" in one table: who allowed what, where and when.

The ``grants`` component of a sessions file keeps one row per grant: the
session it was given in and that session's kind, its scope, what it allows
(its kind and the exact key a check matches), its words for the page, who
decided, when, the request it came from, and when it was revoked.

The scope rule:

- what **the person** allows in the main chat is ``global``: every session
  uses it (side chats, the tasks the person started, the terminal);
- what the person allows anywhere else (a side chat, the terminal's own
  chat, a task) is ``session``: it holds in that session only;
- what Auto keeps for a chat by itself (a site it let the chat read) is
  ``session`` everywhere, the main chat included: only the person's word
  counts everywhere.

The person's grant of a site question (opening, clicking and choosing) covers
the other hosts of its site where it holds (#577), and their grant of a human
check covers every known provider's check on every site (#583): a row keeps the
name it was given and :meth:`GrantBook.person_covers` matches at lookup. Auto's read stays its
exact origin.

Nothing that must ask every time is ever written: the questions of
:data:`~localmelo.melo.contracts.approvals.NEVER_GRANTED` (typing, sending
a form, a click at a point, a press in another app), a step reviewed as a
card-only kind (paying, an account's settings, opening an app in auto mode)
and an Allow once. In ask mode a page's request to open an app
(``handoff:<scheme>``) is the person's own card: their "allow for this chat"
is kept for that kind of address, as the card says. The write path
refuses them with :class:`GrantRefusedError`; the read path ignores any row that
breaks a rule (a forged one too: a ``global`` row from a side chat, Auto's
row that claims more than a site read, a folder that is never offered), so
no row can widen access.

:class:`GrantBook` keeps the live rows in memory. Every check reads that
cache; every write or revoke changes it at once, so a revoke takes effect in
running sessions before its row is committed, and the commits run in order.
A book without a store keeps its rows for its own life only (a library
caller, a test). Deleting a session deletes its rows (``ON DELETE
CASCADE``); revoked rows go at the store's retention.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal, get_args

from localmelo.melo.contracts.approvals import (
    HUMAN_CHECK,
    SITE_GRANTS,
    STAYS_FOR_CHAT,
    ApprovalDecision,
    ApprovalRequest,
    check_scope,
    grant_offered,
    kept_covers,
    never_granted,
    never_kept,
    question_site_key,
    stays_for_chat,
    whole_site,
)
from localmelo.melo.contracts.folders import above_home, never_offered
from localmelo.melo.contracts.review import url_site
from localmelo.support.sessions.display import home_label
from localmelo.support.sessions.store import SessionStore, SessionStoreError

logger = logging.getLogger(__name__)

COMPONENT: Final = "grants"
GRANTS_VERSION: Final = 1

Scope = Literal["global", "session"]
SessionKind = Literal["main", "side", "terminal", "task"]
GrantKind = Literal["tool", "site", "folder", "window", "check"]
DecidedBy = Literal["person", "auto"]
SCOPES: Final = frozenset(get_args(Scope))
SESSION_KINDS: Final = frozenset(get_args(SessionKind))
GRANT_KINDS: Final = frozenset(get_args(GrantKind))
DECIDERS: Final = frozenset(get_args(DecidedBy))

# The question about another app's window the person may allow for the chat.
WINDOW_PREFIX: Final = "window-read:"
# The longest key and words a row keeps.
KEY_CHARS: Final = 2048
LABEL_CHARS: Final = 300

# The stored kind of a session, as a grant's origin names it.
STORED_KINDS: Final[dict[str, SessionKind]] = {
    "front": "main",
    "chat": "side",
    "task": "task",
    "front_bg": "task",
}

_SCHEMA: Final = (
    """CREATE TABLE IF NOT EXISTS grants (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  session_kind TEXT NOT NULL,
  scope        TEXT NOT NULL CHECK (scope IN ('global', 'session')),
  kind         TEXT NOT NULL,
  key          TEXT NOT NULL,
  label        TEXT NOT NULL DEFAULT '',
  target       TEXT NOT NULL DEFAULT '',
  decided_by   TEXT NOT NULL CHECK (decided_by IN ('person', 'auto')),
  granted_at   REAL NOT NULL,
  request_id   TEXT NOT NULL DEFAULT '',
  revoked_at   REAL)""",
    "CREATE INDEX IF NOT EXISTS grants_session ON grants(session_id)",
    "CREATE INDEX IF NOT EXISTS grants_live ON grants(scope) WHERE revoked_at IS NULL",
)
_COLUMNS: Final = (
    "id, session_id, session_kind, scope, kind, key, label, target, decided_by, "
    "granted_at, request_id, revoked_at"
)

# A page's request to open an app: the grant covers that kind of address.
HANDOFF_KEY: Final = "handoff:"
# A tool grant in words, for the page and the terminal.
TOOL_WORDS: Final = {
    "shell_exec": "Run shell commands",
    "python_exec": "Run Python code",
    "file_write": "Write files",
    "file_edit": "Edit files",
}


class GrantRefusedError(ValueError):
    """A grant that may never be kept: it must ask every time."""


@dataclass(frozen=True)
class Grant:
    """One row of the table."""

    id: str
    session_id: str
    session_kind: SessionKind
    scope: Scope
    kind: GrantKind
    key: str
    label: str
    target: str
    decided_by: DecidedBy
    granted_at: float
    request_id: str = ""
    revoked_at: float | None = None

    @property
    def action(self) -> str:
        """What it lets melo do, in one word: ``tool``, ``open``, ``act``, ``read``, ``folder``, ``window`` or ``check``."""
        if self.kind == "site":
            if self.key.startswith("browser-act:"):
                return "act"
            return "read" if self.decided_by == "auto" else "open"
        return self.kind

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "session_id": self.session_id,
            "session_kind": self.session_kind,
            "kind": self.kind,
            "key": self.key,
            "label": self.label,
            "action": self.action,
            "target": self.target,
            "by": self.decided_by,
            "granted_at": self.granted_at,
        }


def _plain(text: object, limit: int) -> str:
    """*text* on one line, without control characters, at most *limit* characters."""
    if not isinstance(text, str):
        return ""
    flat = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(flat.split())[:limit]


def grant_kind(key: object) -> GrantKind | None:
    """The kind of a grant of the question *key*: ``tool``, ``site``, ``window`` or ``check``.

    ``None`` for a key that may never be kept (:func:`never_granted`), an
    empty one, a prefix with nothing after it, one with control characters,
    and a human check's whose name :func:`check_scope` does not read. A
    folder's grant is a kind of its own, named by its caller.
    """
    if (
        not isinstance(key, str)
        or not key
        or len(key) > KEY_CHARS
        or not key.isprintable()
        or key != key.strip()
        or never_granted(key)
    ):
        return None
    if key.startswith(HUMAN_CHECK):
        return "check" if check_scope(key) is not None else None
    for prefix in (*SITE_GRANTS, WINDOW_PREFIX):
        if key.startswith(prefix):
            if len(key) == len(prefix):
                return None
            return "window" if prefix == WINDOW_PREFIX else "site"
    return "tool"


def _site(key: str) -> str:
    for prefix in SITE_GRANTS:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            return url_site(rest) or rest
    return key


def describe(
    kind: GrantKind, key: str, by: DecidedBy, request: ApprovalRequest | None = None
) -> tuple[str, str]:
    """A grant's words and what it is about: (``label``, ``target``)."""
    if kind == "folder":
        target = home_label(key)
        return f"Read files in {target}", target
    if kind == "site":
        # What the person keeps covers the whole site (#577); Auto's read, its host.
        site = (whole_site(key) if by == "person" else None) or _site(key)
        if key.startswith("browser-act:"):
            return f"Click and choose on {site}", site
        return (f"Read {site}" if by == "auto" else f"Open {site}"), site
    if kind == "check":
        return "Press human checks on every site", ""
    if kind == "window":
        app = ""
        if request is not None:
            app = _plain(request.arguments.get("app"), 80)
        app = app or key[len(WINDOW_PREFIX) :]
        return f"See {app} windows", app
    if key.startswith(HANDOFF_KEY):
        # It covers that kind of address from any page, not only the page that asked.
        scheme = key[len(HANDOFF_KEY) :]
        app = _plain(request.arguments.get("app"), 80) if request is not None else ""
        return f"Let pages open {app or 'its app'} ({scheme}: addresses)", app or scheme
    words = TOOL_WORDS.get(key)
    if words is None and request is not None and ":" in key:
        words = _plain(request.preview, LABEL_CHARS)
    return words or f"Use the tool {key}", key


def check_grant(
    kind: GrantKind,
    key: str,
    by: DecidedBy,
    *,
    request: ApprovalRequest | None = None,
    decision: ApprovalDecision | None = None,
) -> None:
    """Raise :class:`GrantRefusedError` unless a grant of *key* may be kept.

    Never a question of :data:`NEVER_GRANTED`, a step of a card-only kind,
    an Allow once (*decision* without ``remember`` or ``always``; Auto's
    :class:`~localmelo.melo.contracts.approvals.Allowance` can carry
    neither), a high-impact step, or a path outside the workspace as a
    tool's grant. Auto keeps a site read only
    (:func:`stays_for_chat`). A folder must be one a chat may be granted.
    """
    if kind not in GRANT_KINDS or by not in DECIDERS:
        raise GrantRefusedError(f"not a grant: {kind!r} by {by!r}")
    if kind == "folder":
        if not _folder_ok(key):
            raise GrantRefusedError("this folder is never kept for a chat")
        if request is not None and (
            request.outside is None or not request.outside.grantable
        ):
            raise GrantRefusedError("this folder is never kept for a chat")
    elif grant_kind(key) != kind:
        raise GrantRefusedError(f"{key!r} is never kept for a chat")
    if decision is not None and (
        not decision.allow or not (decision.remember or decision.always)
    ):
        raise GrantRefusedError("an Allow once is never kept")
    if request is not None:
        if never_kept(request):
            raise GrantRefusedError(
                "money, account and app steps, typing and forms ask every time"
            )
        if kind != "folder" and request.outside is not None:
            raise GrantRefusedError("a folder outside the workspace has its own grant")
        if by == "person" and kind != "folder" and not grant_offered(request):
            raise GrantRefusedError("a high-impact step asks every time")
        if by == "auto" and not stays_for_chat(request):
            raise GrantRefusedError("Auto keeps only a site it let the chat read")
    if by == "auto" and not (kind == "site" and key.startswith(STAYS_FOR_CHAT)):
        raise GrantRefusedError("Auto keeps only a site it let the chat read")


def _folder_ok(key: str) -> bool:
    if not isinstance(key, str) or not key.startswith("/") or len(key) > KEY_CHARS:
        return False
    if not key.isprintable() or key.rstrip("/") == "":
        return False
    try:
        return not above_home(key) and never_offered(key) is None
    except (OSError, ValueError):
        return False


def trusted(grant: Grant) -> bool:
    """Whether a row read back may be used: it breaks none of the rules a write keeps.

    A revoked row, a ``global`` row from anywhere but the person in the main
    chat, Auto's row that is more than a site read or not ``session``, a key
    that may never be kept or that is not of its kind, and a folder that is
    never offered are all ignored.
    """
    if grant.revoked_at is not None:
        return False
    if (
        grant.scope not in SCOPES
        or grant.decided_by not in DECIDERS
        or grant.session_kind not in SESSION_KINDS
        or grant.kind not in GRANT_KINDS
    ):
        return False
    if grant.scope == "global" and (
        grant.session_kind != "main" or grant.decided_by != "person"
    ):
        return False
    if grant.decided_by == "auto" and (
        grant.scope != "session"
        or grant.kind != "site"
        or not grant.key.startswith(STAYS_FOR_CHAT)
    ):
        return False
    if grant.kind == "folder":
        return _folder_ok(grant.key)
    return grant_kind(grant.key) == grant.kind


def scope_for(session_kind: SessionKind, by: DecidedBy) -> Scope:
    """The scope rule: the person's word in the main chat counts everywhere."""
    return "global" if session_kind == "main" and by == "person" else "session"


def _row(row: Iterable[Any]) -> Grant | None:
    try:
        gid, sid, skind, scope, kind, key, label, target, by, at, rid, revoked = row
        return Grant(
            str(gid),
            str(sid),
            skind,
            scope,
            kind,
            str(key),
            _plain(label, LABEL_CHARS),
            _plain(target, LABEL_CHARS),
            by,
            float(at),
            str(rid or ""),
            None if revoked is None else float(revoked),
        )
    except (TypeError, ValueError):
        return None


class MainChatGrants:
    """The live ``global`` grants of the gateway's file, current at every check.

    For the terminal's own chat, which works without the gateway: what the
    person allowed in the main chat holds there too, and a revoke the gateway
    committed holds from the very next check, in the middle of a turn too.
    Each call asks ``PRAGMA data_version`` on a read-only connection it keeps
    open (one cheap statement, no write) and reads the rows again only when
    another connection committed since, or the file was replaced.

    It fails closed: a missing file or table gives no grant, and so does a
    file that cannot be read or a ``grants`` component newer than this
    localmelo reads (with one warning each).
    """

    def __init__(self, path: str | os.PathLike[str], *, busy_s: float = 0.5) -> None:
        self.path = os.path.expanduser(os.fspath(path))
        self._busy_s = busy_s
        self._conn: sqlite3.Connection | None = None
        self._identity: tuple[int, int] | None = None
        self._version: int | None = None
        self._grants: tuple[Grant, ...] = ()
        self._warned = ""

    def _warn(self, message: str) -> None:
        if message != self._warned:
            self._warned = message
            logger.warning("The main chat's grants do not hold here: %s", message)

    def close(self) -> None:
        conn, self._conn = self._conn, None
        self._identity = self._version = None
        self._grants = ()
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    def __call__(self) -> tuple[Grant, ...]:
        try:
            stat = os.stat(self.path)
        except OSError:
            self.close()
            return ()
        identity = (stat.st_dev, stat.st_ino)
        try:
            if self._conn is None or identity != self._identity:
                self.close()
                self._conn = sqlite3.connect(
                    f"{Path(self.path).as_uri()}?mode=ro",
                    uri=True,
                    timeout=self._busy_s,
                    isolation_level=None,
                    check_same_thread=False,
                )
                self._identity = identity
            version = int(self._conn.execute("PRAGMA data_version").fetchone()[0])
            if version != self._version:
                self._grants = self._read(self._conn)
                self._version = version
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            self.close()
            self._warn(f"{self.path} could not be read ({exc})")
            return ()
        return self._grants

    def _read(self, conn: sqlite3.Connection) -> tuple[Grant, ...]:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('schema_components', 'grants')"
            )
        }
        if tables != {"schema_components", "grants"}:
            return ()
        found = conn.execute(
            "SELECT version FROM schema_components WHERE name = ?", (COMPONENT,)
        ).fetchone()
        if found is None:
            return ()
        if int(found[0]) > GRANTS_VERSION:
            self._warn(
                f"{self.path} has grants schema version {found[0]}; "
                f"this localmelo reads version {GRANTS_VERSION}"
            )
            return ()
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM grants "
            "WHERE scope = 'global' AND revoked_at IS NULL ORDER BY granted_at"
        ).fetchall()
        return tuple(g for g in map(_row, rows) if g is not None and trusted(g))


def listed(grants: Iterable[Grant]) -> list[Grant]:
    """*grants* in the order a list shows them, which its numbers follow.

    Every ``global`` one first, newest first; then the ``session`` ones by
    session, the session with the newest grant first, newest first in it.
    """
    rows = list(grants)
    everywhere = sorted(
        (g for g in rows if g.scope == "global"), key=lambda g: (-g.granted_at, g.id)
    )
    by_session: dict[str, list[Grant]] = {}
    for grant in rows:
        if grant.scope != "global":
            by_session.setdefault(grant.session_id, []).append(grant)
    groups = sorted(
        by_session.values(), key=lambda group: -max(g.granted_at for g in group)
    )
    return everywhere + [
        g
        for group in groups
        for g in sorted(group, key=lambda g: (-g.granted_at, g.id))
    ]


class GrantBook:
    """The grants of one sessions file, kept in memory for every check.

    *store* is the file (``None``: the rows live as long as the book).
    :meth:`open` creates the component and reads the live rows; until it
    did, the book is empty and keeps what is granted in memory, written
    once it opens. *outside* gives the ``global`` grants this book does not
    own, and is asked at every check (the terminal's own chat reads the main
    chat's through :class:`MainChatGrants`, so a revoke there holds at once).
    Every change calls the functions in :attr:`listeners`.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        *,
        clock: Callable[[], float] = time.time,
        outside: Callable[[], Sequence[Grant]] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock
        self._outside = outside
        self._rows: dict[str, Grant] = {}
        self._opened = False
        self._open_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._pending: set[asyncio.Task[None]] = set()
        # Queued writes that failed, and the newest failure: a wait reports
        # those that fail while it waits, whoever dropped them.
        self._failed = 0
        self._failure: BaseException | None = None
        # Take-backs the file refused, grant id to when: the next write tries.
        self._owed: dict[str, float] = {}
        # Rows granted before the store opened, written when it does.
        self._unsaved: list[Grant] = []
        self.listeners: list[Callable[[], None]] = []

    @property
    def store(self) -> SessionStore | None:
        return self._store

    @property
    def opened(self) -> bool:
        return self._opened

    # ── Opening ──

    async def open(self) -> None:
        """Create or check the component, then read the live rows; idempotent."""
        if self._store is None:
            self._opened = True
            return
        async with self._open_lock:
            if self._opened:
                return
            async with self._store.component_transaction("component_grants") as conn:
                rows = await conn.execute_fetchall(
                    "SELECT version FROM schema_components WHERE name = ?",
                    (COMPONENT,),
                )
                found = list(rows)
                current = int(found[0][0]) if found else None
                if current is None:
                    for statement in _SCHEMA:
                        await conn.execute(statement)
                    await conn.execute(
                        "INSERT INTO schema_components (name, version) VALUES (?, ?)",
                        (COMPONENT, GRANTS_VERSION),
                    )
                elif current > GRANTS_VERSION:
                    raise SessionStoreError(
                        f"{self._store.path} has grants schema version {current}; "
                        f"this localmelo reads version {GRANTS_VERSION}"
                    )
                else:
                    # Every statement is IF NOT EXISTS: a half-made file is completed.
                    for statement in _SCHEMA:
                        await conn.execute(statement)
                live = await conn.execute_fetchall(
                    f"SELECT {_COLUMNS} FROM grants WHERE revoked_at IS NULL "
                    "ORDER BY granted_at"
                )
            loaded = {}
            for row in live:
                grant = _row(row)
                if grant is None or not trusted(grant):
                    if grant is not None:
                        logger.warning("Ignored grant %s: it breaks a rule", grant.id)
                    continue
                loaded[grant.id] = grant
            # Grants given before the open stay; the file's rows join them.
            self._rows = {**loaded, **self._rows}
            self._opened = True
            self._store.add_retention_hook(self.apply_retention)
            unsaved, self._unsaved = self._unsaved, []
            for grant in unsaved:
                if grant.id in self._rows:
                    self._queue(self._insert(grant), "grant_add")
        self._changed()

    def outside(self) -> tuple[Grant, ...]:
        """The ``global`` grants of :attr:`outside` as they are now (none: no source, or it failed)."""
        if self._outside is None:
            return ()
        try:
            found = self._outside()
        except Exception:
            logger.warning("Could not read the main chat's grants", exc_info=True)
            return ()
        return tuple(g for g in found if trusted(g) and g.scope == "global")

    async def apply_retention(self) -> int:
        """Delete revoked rows older than the store's retention; how many went."""
        if self._store is None:
            return 0
        cutoff = self._store.retention_cutoff()
        async with self._store.component_transaction("grants_retain") as conn:
            cursor = await conn.execute(
                "DELETE FROM grants WHERE revoked_at IS NOT NULL AND revoked_at < ?",
                (cutoff,),
            )
            return max(cursor.rowcount, 0)

    # ── Reading ──

    def live(self) -> list[Grant]:
        """Every live grant of this book, oldest first (not :attr:`outside`'s)."""
        return sorted(self._rows.values(), key=lambda g: (g.granted_at, g.id))

    def get(self, grant_id: str) -> Grant | None:
        return self._rows.get(grant_id)

    def visible(self, session_id: str | None, *, inherit: bool = True) -> list[Grant]:
        """The grants that hold in *session_id*: its own, and with *inherit* the global ones."""
        found = [
            g
            for g in (*self.outside(), *self._rows.values())
            if (inherit and g.scope == "global")
            or (session_id is not None and g.session_id == session_id)
        ]
        return sorted(found, key=lambda g: (g.granted_at, g.id))

    def allowed_for(self, session_id: str | None) -> frozenset[str]:
        """The names a session holds a grant of (tools, sites, windows; Auto's site reads too)."""
        return frozenset(g.key for g in self.visible(session_id) if g.kind != "folder")

    def person_covers(
        self, session_id: str | None, name: str, *, inherit: bool = True
    ) -> bool:
        """Whether the person kept an answer that holds here and covers the question *name* under another name.

        Another host of its site (#577), another human check its name covers
        (#583): :func:`~localmelo.melo.contracts.approvals.kept_covers`.
        Matched at lookup, so the rows keep the names they were given. Never
        Auto's: its read is about the host it let the chat read, and nobody
        vouched for the site's others.
        """
        return any(
            g.decided_by == "person" and kept_covers(g.key, name)
            for g in self.visible(session_id, inherit=inherit)
        )

    def folders_for(self, session_id: str | None) -> tuple[str, ...]:
        """The folders outside the workspace a session may read in, oldest first."""
        return tuple(
            dict.fromkeys(g.key for g in self.visible(session_id) if g.kind == "folder")
        )

    def view(
        self,
        session: str | Callable[[], str | None],
        kind: SessionKind,
        *,
        inherit: bool = True,
    ) -> SessionGrants:
        """The grants of one session (see :class:`SessionGrants`)."""
        return SessionGrants(self, session, kind, inherit=inherit)

    # ── Writing ──

    def _changed(self) -> None:
        for listener in list(self.listeners):
            try:
                listener()
            except Exception:
                logger.warning("A grant listener failed", exc_info=True)

    def add_now(
        self,
        session_id: str | None,
        session_kind: SessionKind,
        kind: GrantKind,
        key: str,
        *,
        by: DecidedBy = "person",
        request: ApprovalRequest | None = None,
        decision: ApprovalDecision | None = None,
    ) -> tuple[Grant, asyncio.Task[None] | None]:
        """Keep a grant now and queue its row; the grant and the write (``None``: none).

        :class:`GrantRefusedError` for one that may never be kept. The same grant
        given again in the same session is the row it has.
        """
        if session_kind not in SESSION_KINDS:
            raise GrantRefusedError(f"not a session kind: {session_kind!r}")
        check_grant(kind, key, by, request=request, decision=decision)
        sid = session_id or ""
        for grant in self._rows.values():
            if (grant.session_id, grant.kind, grant.key, grant.decided_by) == (
                sid,
                kind,
                key,
                by,
            ):
                return grant, None
        label, target = describe(kind, key, by, request)
        grant = Grant(
            uuid.uuid4().hex[:12],
            sid,
            session_kind,
            scope_for(session_kind, by),
            kind,
            key,
            _plain(label, LABEL_CHARS),
            _plain(target, LABEL_CHARS),
            by,
            self._clock(),
            request.request_id if request is not None else "",
        )
        self._rows[grant.id] = grant
        self._changed()
        if not sid or self._store is None:
            return grant, None  # no session yet, or no file: memory only
        if not self._opened:
            self._unsaved.append(grant)
            return grant, None
        return grant, self._queue(self._insert(grant), "grant_add")

    async def add(
        self,
        session_id: str | None,
        session_kind: SessionKind,
        kind: GrantKind,
        key: str,
        *,
        by: DecidedBy = "person",
        request: ApprovalRequest | None = None,
        decision: ApprovalDecision | None = None,
    ) -> Grant:
        """:meth:`add_now`, waiting for its row; a failed write is logged, the grant kept."""
        grant, write = self.add_now(
            session_id,
            session_kind,
            kind,
            key,
            by=by,
            request=request,
            decision=decision,
        )
        if write is not None:
            try:
                await write
            except Exception:
                logger.warning("Could not save grant %s", grant.id, exc_info=True)
        return grant

    def _drop(self, grants: list[Grant]) -> asyncio.Task[None] | None:
        if not grants:
            return None
        at = self._clock()
        for grant in grants:
            self._rows.pop(grant.id, None)
        self._unsaved = [g for g in self._unsaved if g.id in self._rows]
        self._changed()
        saved = [g.id for g in grants if g.session_id]
        if self._store is None or not self._opened or not saved:
            return None
        return self._queue(self._mark_revoked(saved, at), "grant_revoke")

    async def revoke(self, grant_id: str) -> Grant | None:
        """Take back one grant; ``None`` when there is no live one. A failed write raises.

        The check stops using it at once, before the row is written.
        """
        grant = self._rows.get(grant_id)
        if grant is None:
            return None
        write = self._drop([grant])
        if write is not None:
            await write
        return replace(grant, revoked_at=self._clock())

    def forget_now(
        self,
        session_id: str | None,
        key: str | None = None,
        *,
        folders: bool = False,
    ) -> list[Grant]:
        """Take back what a session holds now; the grants taken back.

        With *key*: that name's grants that hold in the session, its own and a
        global one (``global`` grants count everywhere, so taking one back
        anywhere takes it back everywhere); with *folders* the key is a
        folder. Without *key*: every grant the session gave itself (in the
        main chat that is every global one); a global grant from the main
        chat stays in a side chat.
        """
        found = self._held(session_id, key, folders)
        self._drop(found)
        return found

    async def take_back_site(
        self,
        session_id: str | None,
        key: str,
        by: str | None = None,
        *,
        inherit: bool = True,
    ) -> bool:
        """Take back what covers the site *key*, ``ask:site:<site>``, in a session (#577).

        A refusal of the site: both schemes, opening and clicking, the person's
        and Auto's (with *by*, that decider's only), the session's own rows and,
        with *inherit*, the ``global`` ones that hold there: the person's newest
        word wins, and the main chat asks once more. :attr:`outside`'s rows are
        another file's and stay.

        Whether the file took it. The checks stop using the rows at once, and
        what the file refused stays owed to the next write.
        """
        sid = session_id or ""
        write = self._drop(
            [
                g
                for g in self._rows.values()
                if (g.session_id == sid or (inherit and g.scope == "global"))
                and g.kind == "site"
                and question_site_key(g.key) == key
                and by in (None, g.decided_by)
            ]
        )
        if write is None:
            return True
        await asyncio.wait([write])  # a run stopped now does not stop the write
        return not write.cancelled() and write.exception() is None

    async def forget(
        self,
        session_id: str | None,
        key: str | None = None,
        *,
        folders: bool = False,
    ) -> list[Grant]:
        """:meth:`forget_now`, waiting for the rows to be marked.

        The checks stop using the grants at once; a write that fails raises
        :class:`SessionStoreError`, so a caller can say the revoke was not
        saved (it would come back after a restart).
        """
        found = self._held(session_id, key, folders)
        write = self._drop(found)
        if write is not None:
            await write
        return found

    def _held(
        self, session_id: str | None, key: str | None, folders: bool
    ) -> list[Grant]:
        sid = session_id or ""
        if key is None:
            return [g for g in self._rows.values() if g.session_id == sid]
        return [
            g
            for g in self._rows.values()
            if g.key == key
            and (g.kind == "folder") == folders
            and (g.session_id == sid or g.scope == "global")
        ]

    def forget_session(self, session_id: str) -> None:
        """Forget a deleted session's rows here; its file rows went with it."""
        gone = [g for g in self._rows.values() if g.session_id == session_id]
        for grant in gone:
            self._rows.pop(grant.id, None)
        if gone:
            self._changed()

    # ── The write queue ──

    def _queue(self, write: Awaitable[None], label: str) -> asyncio.Task[None] | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            close = getattr(write, "close", None)
            if callable(close):
                close()
            logger.warning("Could not save a grant change: no event loop (%s)", label)
            return None
        task = loop.create_task(self._ordered(write, label))
        self._pending.add(task)
        task.add_done_callback(self._settled)
        return task

    def _settled(self, task: asyncio.Task[None]) -> None:
        if task not in self._pending:
            return  # a wait below dropped it already
        self._pending.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._failed += 1
            self._failure = task.exception()
            logger.warning("Could not save a grant change", exc_info=task.exception())

    def _unfinished(self) -> list[asyncio.Task[None]]:
        """The queued writes still running, after dropping the finished ones here.

        A write that finished in the loop pass in which a wait looks has its
        done-callback still queued, and ``gather`` of done tasks returns
        without yielding: a wait that left the drop to the callback spun on
        it for ever. So each wait drops what is done itself, and waits only
        on what still runs.
        """
        for task in [task for task in self._pending if task.done()]:
            self._settled(task)
        return list(self._pending)

    async def _ordered(self, write: Awaitable[None], label: str) -> None:
        async with self._write_lock:
            try:
                await write
                if self._owed:
                    await self._mark_revoked([], 0.0)
            except (sqlite3.Error, OSError) as exc:
                raise SessionStoreError(
                    f"Could not save a grant change: {exc}"
                ) from exc

    async def settle(self) -> None:
        """Wait for every queued write; one that failed meanwhile raises :class:`SessionStoreError`.

        A write that fails while this waits counts whoever drops it (its own
        done-callback, or :meth:`_unfinished`), so a failure is never lost
        to a write that finished between two looks.
        """
        before = self._failed
        while running := self._unfinished():
            await asyncio.gather(*running, return_exceptions=True)
        failed = self._failure
        if self._failed > before and failed is not None:
            raise SessionStoreError(str(failed)) from failed

    async def flush(self) -> None:
        """Wait for every queued write (a failed one is logged), and try what is owed."""
        if self._owed and not self._unfinished():
            self._queue(self._mark_revoked([], 0.0), "grant_revoke")
        while running := self._unfinished():
            await asyncio.gather(*running, return_exceptions=True)

    async def _insert(self, grant: Grant) -> None:
        assert self._store is not None
        async with self._store.component_transaction("grant_add") as conn:
            await conn.execute(
                f"INSERT INTO grants ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    grant.id,
                    grant.session_id,
                    grant.session_kind,
                    grant.scope,
                    grant.kind,
                    grant.key,
                    grant.label,
                    grant.target,
                    grant.decided_by,
                    grant.granted_at,
                    grant.request_id,
                    None,
                ),
            )

    async def _mark_revoked(self, ids: list[str], at: float) -> None:
        """Mark *ids*, and what the file refused before: owed until it lands (#577)."""
        assert self._store is not None
        self._owed.update(dict.fromkeys(ids, at))
        async with self._store.component_transaction("grant_revoke") as conn:
            await conn.executemany(
                "UPDATE grants SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                [(when, grant_id) for grant_id, when in self._owed.items()],
            )
        self._owed.clear()


class SessionGrants:
    """The grants that hold in one session, and the way to give or take one back.

    *session* is the session's id, or a function that gives it (the
    terminal's own chat has none before its first turn). *kind* decides the
    scope of what the person gives here (:func:`scope_for`). With
    *inherit*, the global grants hold here too; a task's own view leaves
    them out, since its owner chat's grants reach it after the review of
    what changed since that chat began (#369).
    """

    def __init__(
        self,
        book: GrantBook,
        session: str | Callable[[], str | None],
        kind: SessionKind,
        *,
        inherit: bool = True,
    ) -> None:
        self.book = book
        self._session = session
        self.kind: SessionKind = kind
        self.inherit = inherit

    @property
    def session_id(self) -> str | None:
        found = self._session() if callable(self._session) else self._session
        return found or None

    def _visible(self) -> list[Grant]:
        return self.book.visible(self.session_id or "", inherit=self.inherit)

    def names(self) -> frozenset[str]:
        """The tools, sites and windows the person allowed here (or everywhere)."""
        return frozenset(
            g.key
            for g in self._visible()
            if g.kind != "folder" and g.decided_by == "person"
        )

    def covers(self, name: str) -> bool:
        """Whether the person's kept answers here cover the question *name*: that name, another host of its site, or another human check one covers."""
        return name in self.names() or self.book.person_covers(
            self.session_id or "", name, inherit=self.inherit
        )

    def passed(self) -> frozenset[str]:
        """The sites Auto let this chat read."""
        return frozenset(g.key for g in self._visible() if g.decided_by == "auto")

    def folders(self) -> tuple[str, ...]:
        """The folders outside the workspace allowed for reading here, oldest first."""
        return tuple(
            dict.fromkeys(g.key for g in self._visible() if g.kind == "folder")
        )

    def everywhere(self) -> frozenset[str]:
        """The names that hold here because the person allowed them in the main chat."""
        return frozenset(
            g.key for g in self._visible() if g.scope == "global" and g.kind != "folder"
        )

    def granted_at(self, name: str) -> float:
        """When the newest grant of *name* that holds here began; 0 for none."""
        return max(
            (g.granted_at for g in self._visible() if g.key == name), default=0.0
        )

    def grant_now(
        self,
        key: str,
        *,
        by: DecidedBy = "person",
        kind: GrantKind | None = None,
        request: ApprovalRequest | None = None,
        decision: ApprovalDecision | None = None,
    ) -> Grant | None:
        """Keep a grant here; ``None`` (and a log line) when it may never be kept."""
        found = kind or grant_kind(key)
        if found is None:
            logger.warning("Refused to keep %r: it asks every time", key)
            return None
        try:
            grant, _ = self.book.add_now(
                self.session_id,
                self.kind,
                found,
                key,
                by=by,
                request=request,
                decision=decision,
            )
        except GrantRefusedError as exc:
            logger.warning("Refused to keep %r: %s", key, exc)
            return None
        return grant

    async def grant(
        self,
        key: str,
        *,
        by: DecidedBy = "person",
        kind: GrantKind | None = None,
        request: ApprovalRequest | None = None,
        decision: ApprovalDecision | None = None,
    ) -> Grant | None:
        """:meth:`grant_now`, waiting until its row is written."""
        found = kind or grant_kind(key)
        if found is None:
            logger.warning("Refused to keep %r: it asks every time", key)
            return None
        try:
            return await self.book.add(
                self.session_id,
                self.kind,
                found,
                key,
                by=by,
                request=request,
                decision=decision,
            )
        except GrantRefusedError as exc:
            logger.warning("Refused to keep %r: %s", key, exc)
            return None

    def forget_now(self, key: str | None = None) -> list[Grant]:
        """Take back *key* (here and, when global, everywhere), or all this session gave."""
        return self.book.forget_now(self.session_id, key)

    async def take_back_site(self, key: str, by: str | None = None) -> bool:
        """Take back what covers the site *key* here (:meth:`GrantBook.take_back_site`)."""
        return await self.book.take_back_site(
            self.session_id, key, by, inherit=self.inherit
        )

    def forget_folder_now(self, folder: str) -> bool:
        """Take back a folder's grant; whether one held here."""
        return bool(self.book.forget_now(self.session_id, folder, folders=True))


__all__ = [
    "COMPONENT",
    "GRANTS_VERSION",
    "STORED_KINDS",
    "TOOL_WORDS",
    "WINDOW_PREFIX",
    "Grant",
    "GrantBook",
    "GrantRefusedError",
    "SessionGrants",
    "check_grant",
    "describe",
    "grant_kind",
    "listed",
    "MainChatGrants",
    "scope_for",
    "trusted",
]
