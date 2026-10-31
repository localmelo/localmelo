"""HTTP client for gateway scheduler tasks; never starts a gateway."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

import httpx

from localmelo.support.loopback import is_loopback
from localmelo.support.task_follow import FollowError, follow

MAX_BRIEF_BYTES = 1024 * 1024
MAX_OBJECTIVE_CHARS = 100_000
LEVELS = tuple(f"P{n}" for n in range(6))
# The words people may type for P0 to P5: accepted wherever a P-code is.
WORDS = ("now", "today", "normal", "whenever", "idle", "later")


def _level_arg(value: str) -> str:
    """``P1`` for ``today``, ``Today`` or ``p1``; ``auto`` as it is."""
    text = value.strip().lower().removeprefix("if ")
    return (
        f"P{WORDS.index(text)}"
        if text in WORDS
        else text.upper().replace("AUTO", "auto")
    )


class ClientError(Exception):
    """An operation failed without exposing the server response or credential."""


def address(url: str) -> str:
    """Validate an explicit local gateway URL without DNS or proxy discovery."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme not in ("http", "https")
            or not host
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        if not is_loopback(host):
            raise ValueError
        if host.lower() in ("localhost", "localhost."):
            host = "127.0.0.1"
        else:
            host = str(
                socket.getaddrinfo(
                    host, None, type=socket.SOCK_STREAM, flags=socket.AI_NUMERICHOST
                )[0][4][0]
            )
        rendered = f"[{host}]" if ":" in host else host
        return f"{parsed.scheme}://{rendered}:{port}"
    except (OSError, UnicodeError, ValueError):
        raise ValueError(
            "gateway must be a loopback HTTP(S) address with a port"
        ) from None


def read_token(path: Path) -> str:
    """Read an existing private gateway token without creating any files."""
    path = Path(os.path.abspath(path))
    boundary = (
        path.parent if path.name == "auth" and path.parent.name == "gateway" else path
    )
    prefix = boundary.parent.resolve(strict=True)
    fd = os.open(prefix, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.relative_to(boundary.parent).parts:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError("gateway auth directory must be private")
        token_fd = os.open(
            "token", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        try:
            token_info = os.fstat(token_fd)
            if (
                not stat.S_ISREG(token_info.st_mode)
                or token_info.st_uid != os.getuid()
                or stat.S_IMODE(token_info.st_mode) != 0o600
                or token_info.st_nlink != 1
            ):
                raise PermissionError("gateway token must be a private regular file")
            value = os.read(token_fd, 129)
            if len(value) != 64 or any(c not in b"0123456789abcdef" for c in value):
                raise ValueError("gateway token is invalid")
            return value.decode("ascii")
        finally:
            os.close(token_fd)
    finally:
        os.close(fd)


def parser(prog: str = "melo-task") -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--gateway", help="Loopback gateway URL")
    common.add_argument(
        "--json", action="store_true", help="Write machine-readable JSON"
    )
    p = argparse.ArgumentParser(prog=prog, allow_abbrev=False, parents=[common])
    subs = p.add_subparsers(dest="command", required=True)
    new = subs.add_parser("new", parents=[common], allow_abbrev=False)
    new.add_argument("text", nargs="?")
    source = new.add_mutually_exclusive_group()
    source.add_argument("--brief", metavar="FILE")
    source.add_argument("--stdin", action="store_true")
    new.add_argument("--title")
    new.add_argument("--workspace")
    new.add_argument("--level", choices=LEVELS, type=_level_arg)
    new.add_argument("--request-key")
    listing = subs.add_parser("list", parents=[common], allow_abbrev=False)
    listing.add_argument("--state")
    listing.add_argument("--level", choices=LEVELS, type=_level_arg)
    for name in ("show", "pause", "resume", "stop", "follow", "level"):
        sub = subs.add_parser(name, parents=[common], allow_abbrev=False)
        sub.add_argument("id")
        if name == "resume":
            sub.add_argument("--level", choices=LEVELS[:-1], type=_level_arg)
        if name == "level":
            sub.add_argument("level", choices=(*LEVELS, "auto"), type=_level_arg)
        if name == "follow":
            sub.add_argument("--wait", action="store_true")
            sub.add_argument("--after", type=int, default=0)
    return p


def _objective(args: argparse.Namespace, stdin: TextIO) -> str:
    text = args.text
    if (args.brief or args.stdin) and text is not None:
        raise ValueError("choose one task text source")
    if args.brief:
        try:
            fd = os.open(args.brief, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    raise ValueError("brief must be a regular file")
                raw = f.read(MAX_BRIEF_BYTES + 1)
            text = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("cannot read UTF-8 brief") from exc
    elif args.stdin:
        text = stdin.read(MAX_BRIEF_BYTES + 1)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task text must not be empty")
    if len(text) > MAX_OBJECTIVE_CHARS or len(text.encode("utf-8")) > MAX_BRIEF_BYTES:
        raise ValueError("task text exceeds the gateway limit")
    return text


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("invalid task id")
    return value


def _safe(value: object) -> str:
    return "".join(c if c.isprintable() else "?" for c in str(value))


def _level_label(task: dict[str, Any]) -> str:
    """``P1``; ``not recorded`` without a level."""
    level = task.get("level")
    if type(level) is not int or not 0 <= level <= 5:
        return "not recorded"
    return f"P{level}"


def execute(
    args: argparse.Namespace,
    client: httpx.Client,
    *,
    out: TextIO,
    stdin: TextIO,
) -> int:
    """Run a parsed operation using an already authenticated client."""
    cmd = args.command
    if cmd == "follow":
        if not 0 <= args.after <= 2**63 - 1:
            raise ValueError("invalid event cursor")
        return follow(
            client,
            _identifier(args.id),
            json_mode=getattr(args, "json", False),
            wait=args.wait,
            out=out,
            after=args.after,
        )
    request_key: str | None = None
    if cmd == "new":
        payload: dict[str, Any] = {"objective": _objective(args, stdin)}
        for name in ("title", "workspace", "level"):
            value = getattr(args, name)
            if value is not None:
                payload[name] = value
        request_key = args.request_key or uuid.uuid4().hex
        if args.request_key == "":
            raise ValueError("request key must not be empty")
        payload["request_key"] = request_key
        method, path, params = "POST", "/api/tasks", None
    elif cmd == "list":
        payload = {}
        params = {
            k: v
            for k, v in {
                "state": args.state,
                "level": int(args.level[1:]) if args.level else None,
            }.items()
            if v is not None
        }
        method, path = "GET", "/api/tasks"
    else:
        payload = {}
        params = None
        path = "/api/tasks/" + _identifier(args.id)
        method = "GET" if cmd == "show" else "PATCH"
        if cmd in ("pause", "resume", "stop"):
            payload["state"] = cmd
            if cmd == "resume" and args.level is not None:
                payload["level"] = args.level
        elif cmd == "level":
            payload["level"] = args.level
    try:
        response = client.request(
            method, path, params=params, json=payload if method != "GET" else None
        )
    except httpx.RequestError as exc:
        suffix = f"; retry with --request-key {request_key}" if request_key else ""
        raise ClientError(
            f"gateway connection failed at {client.base_url}; "
            "start melo serve or check --gateway" + suffix
        ) from exc
    if response.is_redirect or response.status_code >= 400:
        suffix = (
            f"; retry with --request-key {request_key}"
            if request_key and response.status_code >= 500
            else ""
        )
        try:
            code = response.json().get("error", {}).get("code", "")
        except (ValueError, AttributeError):
            code = ""
        known = {
            "invalid_task",
            "invalid_request",
            "invalid_workspace",
            "submission_conflict",
            "not_found",
            "not_paused",
            "invalid_transition",
            "unauthorized",
        }
        label = f" ({code})" if isinstance(code, str) and code in known else ""
        raise ClientError(
            f"gateway returned HTTP {response.status_code}{label}" + suffix
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ClientError("gateway returned invalid JSON") from exc
    if not isinstance(data, dict) or not isinstance(
        data.get("tasks" if cmd == "list" else "task"), list if cmd == "list" else dict
    ):
        raise ClientError("gateway returned an invalid task response")
    if getattr(args, "json", False):
        print(
            json.dumps({**data, "v": 1, "command": cmd}, ensure_ascii=False),
            file=out,
            flush=True,
        )
    elif cmd == "list":
        rows = [("ID", "LEVEL", "STATE", "TITLE", "")]
        for task in data["tasks"]:
            if not isinstance(task, dict):
                raise ClientError("gateway returned an invalid task response")
            spec = task.get("spec")
            title = task.get("title") or (
                spec.get("title", "") if isinstance(spec, dict) else ""
            )
            rows.append(
                (
                    _safe(task.get("task_id", "")),
                    _level_label(task),
                    _safe(task.get("state", "")),
                    _safe(title),
                    _safe(_waiting_approval(task) or ""),
                )
            )
        # Pad every column but the last, so the table lines up.
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        for row in rows:
            cells = [row[i].ljust(widths[i]) for i in range(3)]
            print("  ".join([*cells, row[3]]).rstrip(), file=out)
            if row[4]:
                print(f"  {row[4]}", file=out)
        out.flush()
    else:
        task = data["task"]
        print(
            f"{_safe(task.get('task_id', ''))}  {_safe(task.get('state', ''))}",
            file=out,
            flush=True,
        )
        if cmd == "stop" and task.get("state") != "cancelled":
            # The gateway accepts a stop for a task with a live worker before
            # that worker has ended; the record turns cancelled once it has.
            print(
                f"  stop requested; the task stops when its worker ends. "
                f"Check with: melo-task show {_safe(task.get('task_id', ''))}",
                file=out,
                flush=True,
            )
        if cmd == "show":
            print(f"  Title: {_safe(task.get('title') or 'not recorded')}", file=out)
            print(f"  Level: {_level_label(task)}", file=out)
            approval = _waiting_approval(task)
            wait = task.get("wait_reason")
            reason = task.get("reason") or ("approval" if approval else None)
            # Reason and Waiting appear only when the task has one. The task
            # record has no outcome field, so a finished task adds nothing.
            if reason:
                print(f"  Reason: {_safe(reason)}", file=out)
            if approval:
                print(f"  {_safe(approval)}", file=out)
            elif wait:
                print(f"  Waiting: {_safe(wait)}", file=out)
        out.flush()
    return 0


def _waiting_approval(task: dict[str, Any]) -> str | None:
    """The line for a task that waits for a person's approval, or None."""
    pending = task.get("pending_approvals")
    if not isinstance(pending, list) or not pending or not isinstance(pending[0], dict):
        return None
    request = pending[0]
    target = request.get("target")
    arguments = request.get("arguments")
    detail = ""
    if isinstance(target, dict):
        detail = str(target.get("path_label") or "")
    elif isinstance(arguments, dict):
        detail = str(arguments.get("command") or arguments.get("path") or "")
    detail = " ".join(detail.split())
    if len(detail) > 80:
        detail = detail[:79] + "…"
    what = " ".join(
        part for part in (str(request.get("tool_name") or ""), detail) if part
    )
    return f"Waiting for your approval: {what} (answer it in the web console, Tasks)"


def token_path() -> Path:
    """Where the gateway keeps its token."""
    return Path.home() / ".cache" / "localmelo" / "gateway" / "auth"


def connect(gateway: str | None = None, *, timeout: float = 10) -> httpx.Client:
    """An authenticated client of the local gateway: *gateway*, else the config's.

    Raises ValueError for an address that is not loopback and
    :class:`ClientError` when the token cannot be read safely.
    """
    if gateway is not None:
        url = address(gateway)
    else:
        from localmelo.support.config import load

        cfg = load().gateway
        host = cfg.host
        if ":" in host and not (host.startswith("[") and host.endswith("]")):
            host = f"[{host}]"
        url = address(f"http://{host}:{cfg.port}")
    return httpx.Client(
        base_url=url,
        headers=client_headers(url),
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    )


def client_headers(url: str) -> dict[str, str]:
    """What a client of the gateway at *url* sends: its token and the Host it checks.

    Raises :class:`ClientError` when the token cannot be read safely.
    """
    try:
        token = read_token(token_path())
    except (OSError, ValueError) as exc:
        raise ClientError(
            "gateway token unavailable or unsafe; start melo serve or check permissions"
        ) from exc
    host = "[::1]" if urlsplit(url).hostname == "::1" else "127.0.0.1"
    return {"Authorization": f"Bearer {token}", "Host": f"{host}:{urlsplit(url).port}"}


def main(argv: list[str] | None = None, prog: str = "melo-task") -> int:
    args = parser(prog).parse_args(argv)
    try:
        if args.command == "follow" and not 0 <= args.after <= 2**63 - 1:
            raise ValueError("invalid event cursor")
        with connect(getattr(args, "gateway", None)) as client:
            return execute(args, client, out=sys.stdout, stdin=sys.stdin)
    except KeyboardInterrupt:
        return 130
    except (ValueError, ClientError, FollowError, OSError, httpx.HTTPError) as exc:
        code = 2 if isinstance(exc, ValueError) else 1
        message = _safe(exc)
        print(f"{prog}: {message}", file=sys.stderr)
        if (
            getattr(args, "json", False)
            and args.command != "follow"
            and not isinstance(exc, BrokenPipeError)
        ):
            try:
                print(
                    json.dumps({"v": 1, "command": args.command, "error": message}),
                    flush=True,
                )
            except (OSError, ValueError):
                return 1
        return code


if __name__ == "__main__":
    raise SystemExit(main())
