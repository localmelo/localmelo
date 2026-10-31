"""The task boundary: what a worker may be, where it may run and what it may use.

Three pieces of policy live here, all in code and never in a model's JSON:

- :class:`TaskSpec` normalizes a task request into the object that
  ``tasks.spec`` stores, so an API route (S3a.2) and a front's tool pass the
  same checks (design 4.2).
- :func:`validate_task_workspace` resolves a worker's folder and refuses one
  outside the boundary the lead chose: for the user's own tasks any existing
  folder strictly under ``~`` with no dot component and nothing under
  ``~/Library``; for every other origin a folder strictly inside
  ``[gateway] workspace`` (``~/localmelo-workspace`` when that key is unset
  or empty). ``/`` and the home folder are never a worker's root.
- :func:`worker_tool_allowlist` gives the origin's built-ins and says which
  grants the worker may have. Tasks of a non-user origin get no unmarked MCP
  tools and no session grants, and their approvals are never remembered.

The module is deliberately free of a store, FastAPI and a clock: the store
is :mod:`localmelo.support.gateway.task_store` and the routes arrive in
S3a.2.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Final, cast, get_args

from localmelo.support.scheduler.model import Origin
from localmelo.support.sessions.display import short_title

# Where a task's folder starts when neither the request nor the config names
# one (the same path the console's sessions use).
DEFAULT_WORKSPACE: Final = "~/localmelo-workspace"

# The TaskSpec fields' bounds (design 4.2). They keep one task row small
# enough to read whole and stop a route from storing an unbounded prompt.
MAX_OBJECTIVE_CHARS: Final = 100_000
MAX_TITLE_CHARS: Final = 200
MAX_REASON_CHARS: Final = 500
MAX_WORKSPACE_CHARS: Final = 4_096
_PRINCIPAL_ORIGINS: Final = frozenset({"user", "user_rule"})
_DEFERRED_FIELDS: Final = (
    "done_when",
    "user_words",
    "context",
    "inputs",
    "workspace_mode",
    "budget_s",
    "deadline",
    "features",
    "report",
)

# The built-ins a worker may register. The gateway's set already omits
# `shell_exec` (P13a), and a worker never gains a tool by origin alone: the
# origin only decides the MCP and session-grant flags below. It is its own
# set, not the gateway chat's default, so a worker's tools do not change
# with a chat's. `list_files` and `git_status` only read and never ask;
# `file_edit` changes part of a file and asks as `file_write` does (#324).
_WORKER_BUILTINS: Final = frozenset(
    {"file_read", "file_write", "file_edit", "list_files", "git_status", "python_exec"}
)


class TaskPolicyError(ValueError):
    """A task request the boundary refuses; *code* is the API error code."""

    code = "invalid_task"


class TaskSpecError(TaskPolicyError):
    """A malformed or out-of-bounds TaskSpec."""

    code = "invalid_task"


class WorkspaceError(TaskPolicyError):
    """A workspace outside the boundary for the request's origin."""

    code = "invalid_workspace"


class UnknownOriginError(TaskPolicyError):
    """An origin that is not one of the scheduler's seven."""

    code = "invalid_origin"


# ── TaskSpec ──


@dataclass(frozen=True)
class TaskSpec:
    """One task's normalized request, as ``tasks.spec`` stores it (design 4.2).

    ``workspace`` is only checked for its shape here;
    :func:`validate_task_workspace` resolves it against the config and origin.
    ``user_words`` is what the person typed when another model wrote the
    objective (the main chat's ``task_create``); only trusted code sets it,
    and a step review reads it as the person's words.
    """

    title: str
    objective: str
    workspace: str = ""
    level: int = 2
    level_source: str = "computed"
    reason: str = ""
    user_words: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaskSpec:
        """Normalize an untrusted request at computed P2, ignoring raw level.

        The ingress must use :meth:`from_user_payload` only after authorizing
        the user. Neither constructor accepts a payload-provided level_source.
        """
        return cls._from_payload(payload, user_level=False)

    @classmethod
    def from_user_payload(cls, payload: Mapping[str, Any]) -> TaskSpec:
        """Normalize a request from an authorized user, pinning an explicit level.

        This is a trusted adapter, not an authorization check. Callers must
        authenticate and authorize the actor before invoking it.
        """
        return cls._from_payload(payload, user_level=True)

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any], *, user_level: bool) -> TaskSpec:
        """Normalize a request; ``prompt`` is an alias for ``objective``."""
        if not isinstance(payload, Mapping):
            raise TaskSpecError("The task spec must be a JSON object")
        if "level_source" in payload:
            raise TaskSpecError("level_source is reserved for trusted code")
        for name in _DEFERRED_FIELDS:
            if name in payload:
                raise TaskSpecError(f"{name} is not supported yet")
        objective = payload.get("objective")
        if objective is None:
            objective = payload.get("prompt")
        objective = _text(
            objective, "objective", max_chars=MAX_OBJECTIVE_CHARS, required=True
        )
        raw_title = payload.get("title")
        title = (
            _derive_title(objective)
            if raw_title is None or raw_title == ""
            else _text(raw_title, "title", max_chars=MAX_TITLE_CHARS, required=True)
        )
        source = "user" if user_level and "level" in payload else "computed"
        level = _level(payload["level"]) if source == "user" else 2
        return cls(
            title=title,
            objective=objective,
            workspace=_workspace_text(payload.get("workspace")),
            level=level,
            level_source=source,
            reason=_text(
                payload.get("reason", ""),
                "reason",
                max_chars=MAX_REASON_CHARS,
                required=False,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """The JSON object ``tasks.spec`` holds (``user_words`` only when set)."""
        out: dict[str, Any] = {
            "title": self.title,
            "objective": self.objective,
            "workspace": self.workspace,
            "level": self.level,
            "level_source": self.level_source,
            "reason": self.reason,
        }
        if self.user_words:
            out["user_words"] = self.user_words
        return out


def _text(value: object, name: str, *, max_chars: int, required: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise TaskSpecError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise TaskSpecError(f"{name} must not be empty")
    if len(text) > max_chars:
        raise TaskSpecError(f"{name} must be at most {max_chars} characters")
    return text


def _derive_title(objective: str) -> str:
    """The title of a task that came without one: its request's first clause."""
    return short_title(objective)


def _workspace_text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TaskSpecError("workspace must be a string")
    text = value.strip()
    if len(text) > MAX_WORKSPACE_CHARS:
        raise TaskSpecError(
            f"workspace must be at most {MAX_WORKSPACE_CHARS} characters"
        )
    return text


def _level(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskSpecError("level must be an integer from 0 (P0) to 5 (Later)")
    if not 0 <= value <= 5:
        raise TaskSpecError("level must be an integer from 0 (P0) to 5 (Later)")
    return value


# ── Workspaces ──


def check_origin(origin: object) -> Origin:
    """*origin* if it names one of the scheduler's origins, else an error."""
    if origin not in get_args(Origin):
        raise UnknownOriginError(
            f"origin must be one of {', '.join(get_args(Origin))}, got {origin!r}"
        )
    return cast(Origin, origin)


def _configured_workspace(cfg: object) -> str:
    gateway = getattr(cfg, "gateway", None)
    # PR 4 adds `[gateway] workspace`; this module works on main, where the
    # key is absent, by falling back to the same default.
    configured = getattr(gateway, "workspace", "") if gateway is not None else ""
    text = configured.strip() if isinstance(configured, str) else ""
    return text or DEFAULT_WORKSPACE


def _relative_parts(path: str, root: str) -> list[str] | None:
    """Return components below root, including on case-insensitive filesystems."""
    parts: list[str] = []
    current = path
    while True:
        if current == root:
            return parts
        try:
            if os.path.samefile(current, root):
                return parts
        except OSError:
            pass
        parent, name = os.path.split(current)
        if parent == current or not name:
            return None
        parts.insert(0, name)
        current = parent


def _inside(path: str, root: str) -> bool:
    parts = _relative_parts(path, root)
    return bool(parts)


def _default_task_workspace(root: str, task_id: str | None) -> str:
    """Create a single non-symlink child selected by a trusted task ID."""
    if not isinstance(task_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id
    ):
        raise WorkspaceError("A safe task_id is required for a default workspace")
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(root, flags)
        try:
            with suppress(FileExistsError):
                os.mkdir(task_id, mode=0o700, dir_fd=root_fd)
            child_fd = os.open(task_id, flags, dir_fd=root_fd)
            os.close(child_fd)
        finally:
            os.close(root_fd)
    except OSError as exc:
        raise WorkspaceError(f"Could not create the task workspace: {exc}") from exc
    path = os.path.realpath(os.path.join(root, task_id))
    if not _inside(path, root):
        raise WorkspaceError("The task workspace must be strictly inside the root")
    return path


def validate_task_workspace(
    workspace: str | None,
    cfg: object,
    *,
    origin: str = "user",
    task_id: str | None = None,
) -> str:
    """Resolve a worker folder or raise :class:`WorkspaceError`.

    A missing user workspace uses the configured root or the default, and is
    refused like any other folder when that root is not a user's (never
    replaced by another folder). A missing
    non-user workspace creates a child named by the trusted single-component
    task_id. Explicit paths must already exist. The configured root must be an
    absolute or ``~/`` path, and cannot resolve to / or the home directory.
    """
    checked_origin = check_origin(origin)
    home = os.path.realpath(os.path.expanduser("~"))
    configured = _configured_workspace(cfg)
    if not (os.path.isabs(configured) or configured.startswith("~/")):
        raise WorkspaceError(
            "The configured workspace must be an absolute path or start with ~/."
        )
    root = os.path.realpath(os.path.expanduser(configured))
    if root == "/" or _relative_parts(root, home) == []:
        raise WorkspaceError(
            "The configured workspace may not be / or your home folder"
        )
    defaulted = workspace is None or (
        isinstance(workspace, str) and not workspace.strip()
    )
    if defaulted:
        if checked_origin != "user":
            path = _default_task_workspace(root, task_id)
            if _relative_parts(path, home) == []:
                raise WorkspaceError("A worker may not be rooted at your home folder")
            return path
        # A configured root outside ~, under ~/Library or in a dot folder is
        # refused below: the task never runs in another folder instead.
        path = root
    else:
        if not isinstance(workspace, str):
            raise WorkspaceError("The workspace must be a string")
        raw = workspace.strip()
        if not (os.path.isabs(raw) or raw.startswith("~/")):
            raise WorkspaceError(
                "The workspace must be an absolute path or start with ~/."
            )
        if len(raw) > MAX_WORKSPACE_CHARS:
            raise WorkspaceError(
                f"The workspace must be at most {MAX_WORKSPACE_CHARS} characters"
            )
        candidate = os.path.expanduser(raw)
        if not os.path.isdir(candidate):
            raise WorkspaceError(f"Not a directory: {candidate}")
        path = os.path.realpath(candidate)
    if path == "/" or _relative_parts(path, home) == []:
        raise WorkspaceError("A worker may not be rooted at / or your home folder")
    if checked_origin == "user":
        if not _user_workspace(path, home):
            raise WorkspaceError(_user_refusal(path, home))
    elif not defaulted and not _inside(path, root):
        raise WorkspaceError(
            f"A {checked_origin} task's workspace must be strictly inside {root}"
        )
    if defaulted:
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"Could not create the default workspace {path}: {exc}"
            ) from exc
    return path


def _user_refusal(path: str, home: str) -> str:
    """Why *path*, which :func:`_user_workspace` refuses, is no user task's folder."""
    if not _inside(path, home):
        return "A user task's workspace must be strictly inside your home folder"
    parts = _relative_parts(path, home) or []
    if any(part.startswith(".") for part in parts):
        return "A user task's workspace may not be in a hidden folder"
    return "A user task's workspace may not be under ~/Library"


def configured_root(cfg: object) -> str:
    """The configured ``[gateway] workspace`` (or the default), as an absolute path."""
    return os.path.abspath(os.path.expanduser(_configured_workspace(cfg)))


def folder_problem(path: str | None, cfg: object) -> str:
    """Why a user's task cannot work in *path* (the default when ``None``), or ``""``.

    A folder outside the allowed roots says so before whether it exists.
    """
    try:
        if path is None:
            validate_task_workspace(None, cfg, origin="user", task_id="check")
            return ""
        home = os.path.realpath(os.path.expanduser("~"))
        real = os.path.realpath(os.path.expanduser(path))
        if os.path.isabs(real) and not _user_workspace(real, home):
            return _user_refusal(real, home)
        validate_task_workspace(path, cfg, origin="user")
    except WorkspaceError as exc:
        return str(exc)
    return ""


def _user_workspace(path: str, home: str) -> bool:
    """Whether *path* is a user task's allowed folder under *home*."""
    parts = _relative_parts(path, home)
    if not parts:
        return False
    if any(part.startswith(".") for part in parts):
        return False
    return not (parts and parts[0].casefold() == "library")


# ── Tool allowlists ──


@dataclass(frozen=True)
class WorkerToolPolicy:
    """The tools and grants a worker of one origin may have (design 4.2)."""

    builtins: frozenset[str]
    remember_approvals: bool
    allow_unmarked_mcp: bool
    allow_session_grants: bool


def worker_tool_allowlist(origin: str) -> WorkerToolPolicy:
    """The origin's built-ins and grants.

    The user (and the user's rules) may have approvals remembered, unmarked
    MCP tools and session grants. A task the agent, a channel, the harness or
    the sleep pipeline started gets none of those: every marked call needs a
    fresh approval, and an unmarked MCP tool is not offered at all.
    """
    checked = check_origin(origin)
    principal = checked in _PRINCIPAL_ORIGINS
    return WorkerToolPolicy(
        builtins=_WORKER_BUILTINS,
        remember_approvals=principal,
        allow_unmarked_mcp=principal,
        allow_session_grants=principal,
    )


__all__ = [
    "DEFAULT_WORKSPACE",
    "TaskPolicyError",
    "TaskSpec",
    "TaskSpecError",
    "UnknownOriginError",
    "WorkerToolPolicy",
    "WorkspaceError",
    "check_origin",
    "configured_root",
    "folder_problem",
    "validate_task_workspace",
    "worker_tool_allowlist",
]
