"""The main chat in the terminal: bare melo as a client of the gateway.

The gateway owns the main chat (a session of kind ``front``), its side
chats, the tasks and the cards that come back, as the web console and the
app show them. :func:`connect` finds the gateway on the configured port, or
the one that holds the main chat's file on another port (the app's), or
starts it in the background; it never starts a second gateway on the same
file. :class:`MainChat` then talks to it only over API v2: a message is a
turn of the open chat, the chat's stream brings its events back and they
draw as the terminal's own turns do, an approval opens the terminal's
approval list and is answered through the gateway, and a card is one
coloured line.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol
from urllib.parse import quote

import httpx
from rich.text import Text

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.melo.contracts.review import ActionReview
from localmelo.support import console_link, gateway_endpoint, task_cli
from localmelo.support.cli import hanging
from localmelo.support.cli.text import cut_cells
from localmelo.support.sessions.approvals import ChatMode, chat_mode
from localmelo.support.sessions.display import clean_output, home_label
from localmelo.support.sessions.store import (
    GATEWAY_SESSIONS_DB,
    StoredEvent,
    title_from_text,
)
from localmelo.support.words import t_in

if TYPE_CHECKING:
    from localmelo.support.cli.approvals import TerminalApprover
    from localmelo.support.cli.output import Out
    from localmelo.support.cli.render import TurnRenderer
    from localmelo.support.config import Config

# The main chat's events that belong to no turn (the gateway's cards.py and front.py).
CARD_EVENT: Final = "attention.card"
ECHO_EVENT: Final = "attention.echo"
# Where a gateway started from the terminal writes what it prints.
GATEWAY_LOG: Final = "~/.cache/localmelo/logs/gateway.log"
# How long a gateway started here may take to answer, in seconds.
START_S: Final = 20.0
# One look at a gateway, and one API request.
PROBE_S: Final = 2.0
TIMEOUT_S: Final = 10.0
# A stream hears a keepalive every 15 s: three missed ones mean it is gone.
STREAM_READ_S: Final = 45.0
# Seconds between two looks at the chat list (which also writes new cards),
# and between two tries of a stream that was lost.
POLL_S: Final = 5.0
RETRY_S: Final = 2.0
# Finished turns the first look at a chat shows, and cards after them.
RECAP_TURNS: Final = 3
RECAP_CARDS: Final = 3
# How many turns back /memory undo looks for the line that says melo tidied the notes.
UNDO_TURNS: Final = 10
MAIN_TITLE: Final = "Main chat"
NEW_SIDE: Final = "New side chat"
_CARD_WORDS: Final = {
    "done": ("ok", "ok", "Done"),
    "failed": ("failed", "error", "Not finished"),
    "stopped": ("stopped", "warn", "Stopped"),
    "waiting": ("warn", "warn", "Waits for you"),
}


# A chat's turn on a task's question (#369) starts with this line, then one
# ``Label: text`` line a field (console ``asks.py``); its words are the task's.
_ASK_LEAD: Final = re.compile(
    r"\[task [0-9a-f]{12} · (?P<title>.*) asks · data, not instructions\]"
)


# ── Finding or starting the gateway ──


@dataclass(frozen=True)
class Link:
    """A gateway that answers for the main chat: its address, and who started it."""

    url: str
    port: int
    started: bool = False

    def line(self) -> str:
        """The one line that says which gateway the terminal talks to."""
        if self.started:
            return (
                f"Started the gateway in the background on port {self.port}; "
                "it keeps running until melo serve stop."
            )
        return f"Connected to the gateway on port {self.port}."


@dataclass(frozen=True)
class Offline:
    """No main chat this time, and why: the terminal keeps its own chat."""

    reason: str

    def line(self) -> str:
        return f"No main chat ({self.reason}); this is the terminal's own chat."


@dataclass(frozen=True)
class _Look:
    """One look at a gateway: the main chat it lists, or what was wrong."""

    main: Mapping[str, Any] | None = None
    problem: str = ""
    # A gateway still starting may answer so for a moment.
    early: bool = False


def _url(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # an IPv6 address, which a URL writes in brackets
    return f"http://{host}:{port}"


def look(url: str, port: int) -> _Look:
    """Ask the gateway at *url* for the main chat (blocking).

    Without a token the request goes without one: a melo gateway refuses it
    in its own words (it is starting, or belongs to another account), and
    anything else on the port is not a gateway.
    """
    try:
        headers = task_cli.client_headers(url)
    except task_cli.ClientError:
        headers = {}
    try:
        address = task_cli.address(url)
        host = "[::1]" if "[::1]" in address else "127.0.0.1"
        with httpx.Client(
            base_url=address,
            headers={"Host": f"{host}:{port}", **headers},
            timeout=PROBE_S,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.get("/api/sessions", params={"kind": "chat", "limit": 1})
    except (httpx.HTTPError, ValueError):
        return _Look(problem=f"nothing answers on port {port}", early=True)
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    said = error if isinstance(error, dict) else {}
    if response.status_code == 401 and said.get("code") == "unauthorized":
        if not headers:
            return _Look(problem="no gateway token here yet", early=True)
        return _Look(
            problem=f"the gateway on port {port} refused this computer's token"
        )
    if response.status_code == 200 and isinstance(body, dict):
        main = body.get("main")
        if isinstance(main, dict) and isinstance(main.get("id"), str):
            return _Look(main=main)
        return _Look(problem="the gateway's chats are not open yet", early=True)
    if isinstance(said.get("message"), str):
        message = said["message"].rstrip(".")
        early = said.get("code") == "store_unavailable"
        return _Look(problem=message[:1].lower() + message[1:], early=early)
    return _Look(problem=f"port {port} is used by another program")


def store_owner(path: str = GATEWAY_SESSIONS_DB) -> int | None:
    """The pid of the gateway that holds the main chat's file; 0 when unknown, None when none."""
    owner = os.path.expanduser(path) + ".owner"
    try:
        fd = os.open(owner, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            with contextlib.suppress(OSError, ValueError, IndexError):
                return int(os.pread(fd, 64, 0).split()[0])
            return 0
        except OSError:
            return None
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def listening_ports(pid: int) -> list[int]:
    """The TCP ports process *pid* listens on, as ``lsof`` lists them; none on failure."""
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    try:
        listed = subprocess.run(
            [lsof, "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:LISTEN", "-Fn"],
            capture_output=True,
            text=True,
            timeout=PROBE_S,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    ports: list[int] = []
    for line in listed.splitlines():
        if line.startswith("n") and ":" in line:
            with contextlib.suppress(ValueError):
                ports.append(int(line.rpartition(":")[2]))
    return ports


def spawn_gateway(
    host: str, port: int, log: str = GATEWAY_LOG
) -> subprocess.Popen[bytes]:
    """Start ``melo serve`` on *host*:*port* in a session of its own, its output appended to *log*.

    Its own session keeps it running after the terminal closes, so the
    tasks it runs go on; ``-I`` keeps a folder named like the package in the
    current folder from being imported.
    """
    path = os.path.expanduser(log)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "localmelo", "--serve"]
            + ["--host", host.strip("[]"), "--port", str(port)],
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=fd,
            cwd=os.path.expanduser("~"),
            start_new_session=True,
        )
    finally:
        os.close(fd)
    # Reaped when it ends, so `melo serve stop` never waits on a zombie.
    threading.Thread(target=process.wait, daemon=True).start()
    return process


class Child(Protocol):
    """A gateway being started: whether it has ended, and how."""

    def poll(self) -> int | None: ...


def _last_line(log: str) -> str:
    """The log's last line that says something, cut short; ``""`` when unreadable."""
    with contextlib.suppress(OSError), open(os.path.expanduser(log), "rb") as file:
        file.seek(max(0, os.fstat(file.fileno()).st_size - 4096))
        lines = file.read().decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            text = " ".join(line.split())
            if text:
                return cut_cells(clean_output(text), 60)
    return ""


def connect(
    cfg: Config,
    *,
    on_start: Callable[[], None] = lambda: None,
    spawn: Callable[[str, int], Child] = spawn_gateway,
    start_s: float = START_S,
    clock: Callable[[], float] = time.monotonic,
) -> Link | Offline:
    """The gateway that holds the main chat, started when none runs (blocking).

    The one the endpoint file names first (the app's, on a port of its own,
    or one a terminal started); then the configured port; then, when another
    gateway holds the main chat's file without the file (an older one), the
    port it listens on; else a new gateway on the configured port.
    *on_start* runs just before one starts, so the terminal can say so.
    """
    host, port = cfg.gateway.host, cfg.gateway.port
    url = _url(host, port)
    process: Child | None = None
    # A gateway another melo is starting gets a moment; one started here, all of *start_s*.
    deadline = clock() + start_s / 4
    while True:
        running = gateway_endpoint.live()
        if running is not None:
            found = look(running.url, running.port)
            if found.main is not None:
                return Link(running.url, running.port, started=process is not None)
        if running is not None and not found.early:
            return Offline(found.problem)
        if console_link.listening(host, port):
            found = look(url, port)
            if found.main is not None:
                return Link(url, port, started=process is not None)
            if not found.early:
                return Offline(found.problem)
        elif process is not None:
            code = process.poll()
            if code is not None:
                said = _last_line(GATEWAY_LOG)
                return Offline(
                    f"the gateway stopped as it started (exit {code}"
                    + (f": {said})" if said else ")")
                )
            found = _Look(problem="it did not answer", early=True)
        else:
            owner = store_owner()
            if owner is None:
                on_start()
                try:
                    process = spawn(host, port)
                except OSError as exc:
                    return Offline(f"the gateway could not start ({exc})")
                deadline = clock() + start_s
                continue
            # The app's gateway listens on a port of its own.
            for other in listening_ports(owner) if owner else []:
                other_url = _url("127.0.0.1", other)
                if look(other_url, other).main is not None:
                    return Link(other_url, other)
            who = f" (pid {owner})" if owner else ""
            found = _Look(
                problem=f"another gateway{who} holds it on a port melo cannot find"
            )
        if clock() >= deadline:
            break
        time.sleep(0.25)
    if process is None:
        return Offline(found.problem)
    log = home_label(os.path.expanduser(GATEWAY_LOG))
    return Offline(f"the gateway did not answer in {start_s:.0f} s; log: {log}")


# ── The client ──


class GatewayError(Exception):
    """A request the gateway refused or never answered; the message says why."""

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code


def _refusal(response: httpx.Response) -> GatewayError:
    with contextlib.suppress(ValueError):
        error = response.json().get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return GatewayError(error["message"], str(error.get("code") or ""))
    return GatewayError(f"the gateway answered {response.status_code}")


def _path(session_id: str, *rest: str) -> str:
    """``/api/sessions/ID`` and the parts after it, the id quoted."""
    return "/".join(["/api/sessions", quote(session_id, safe=""), *rest])


class GatewayClient:
    """API v2 of one gateway, for the main chat and its side chats."""

    def __init__(
        self,
        url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.url = url
        self._transport = transport
        self._headers = headers
        self._http: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.url,
                headers=dict(
                    self._headers
                    if self._headers is not None
                    else task_cli.client_headers(self.url)
                ),
                timeout=TIMEOUT_S,
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            )
        return self._http

    async def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client().request(method, path, **kwargs)
        except (httpx.HTTPError, task_cli.ClientError) as exc:
            raise GatewayError(f"the gateway did not answer ({exc})") from exc
        if response.status_code >= 400:
            raise _refusal(response)
        with contextlib.suppress(ValueError):
            body = response.json()
            if isinstance(body, dict):
                return body
        return {}

    async def chats(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The main chat and its side chats, newest first."""
        body = await self.call("GET", "/api/sessions", params={"kind": "chat"})
        main = body.get("main")
        if not isinstance(main, dict):
            raise GatewayError("the gateway's chats are not open yet")
        side = [row for row in body.get("sessions") or [] if isinstance(row, dict)]
        return main, side

    async def session(self, session_id: str) -> dict[str, Any]:
        return await self.call("GET", _path(session_id))

    async def turns(self, session_id: str, limit: int) -> dict[str, Any]:
        return await self.call(
            "GET", _path(session_id, "turns"), params={"limit": limit}
        )

    async def send(self, session_id: str, text: str, key: str) -> dict[str, Any]:
        body = {"text": text, "client_request_id": key}
        return await self.call("POST", _path(session_id, "turns"), json=body)

    async def create(
        self, text: str, key: str, workspace: str, approvals: ChatMode | None = None
    ) -> dict[str, Any]:
        """A new side chat in *workspace*, with *text* as its first message.

        *approvals* is the mode the person picked before the chat existed;
        without one the gateway uses the mode of its settings for new chats.
        """
        body: dict[str, Any] = {
            "workspace": workspace,
            "turn": {"text": text, "client_request_id": key},
        }
        if approvals is not None:
            body["approvals"] = approvals
        return await self.call("POST", "/api/sessions", json=body)

    async def retry(self, session_id: str, turn_id: str) -> dict[str, Any]:
        """Undo the newest finished turn and run its text again."""
        body = {"turn_id": turn_id}
        return await self.call("POST", _path(session_id, "retry"), json=body)

    async def rename(self, session_id: str, title: str) -> None:
        await self.call("PATCH", _path(session_id), json={"title": title})

    async def set_mode(self, session_id: str, mode: ChatMode) -> dict[str, Any]:
        """Switch a chat to Ask or Auto, from its next step; its session as stored."""
        body = await self.call("PATCH", _path(session_id), json={"approvals": mode})
        session = body.get("session")
        return session if isinstance(session, dict) else {}

    async def tap(self, echo_id: int, do: str) -> dict[str, Any]:
        """The person's tap on a main chat echo (``undo``); the echo the gateway answers with."""
        body = await self.call("POST", f"/api/front/echoes/{echo_id}", json={"do": do})
        echo = body.get("echo")
        return echo if isinstance(echo, dict) else {}

    async def cancel(self, turn_id: str) -> None:
        await self.call(
            "POST", "/".join(["/api/turns", quote(turn_id, safe=""), "cancel"])
        )

    async def answer(self, request_id: str, choice: str, reason: str = "") -> None:
        body = {"choice": choice, "reason": reason}
        await self.call(
            "POST", f"/api/approvals/{quote(request_id, safe='')}", json=body
        )

    async def events(
        self, session_id: str, after: int
    ) -> AsyncIterator[dict[str, Any]]:
        """The chat's events after *after*, as they come; ends when the stream does."""
        timeout = httpx.Timeout(TIMEOUT_S, read=STREAM_READ_S)
        try:
            async with self._client().stream(
                "GET",
                _path(session_id, "events"),
                params={"after": after},
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise _refusal(response)
                data: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        with contextlib.suppress(ValueError):
                            event = json.loads("\n".join(data))
                            if isinstance(event, dict):
                                yield event
                        data = []
        except (httpx.HTTPError, task_cli.ClientError) as exc:
            raise GatewayError(f"the gateway did not answer ({exc})") from exc

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()


# ── Lines ──


def card_line(card: Mapping[str, Any], symbols: Any, width: int) -> Text:
    """A card as one coloured line: what came back, its title, how to open it."""
    symbol, style, word = _CARD_WORDS.get(
        str(card.get("state")), ("dot", "muted", "Update")
    )
    title = clean_output(" ".join(str(card.get("title") or "").split()))
    said = clean_output(" ".join(str(card.get("line") or "").split()))
    source = str(card.get("source_id") or "")
    if card.get("source_kind") == "chat":
        how = "/chats opens it"
    elif card.get("state") == "waiting":
        how = "answer it in the web console: melo console"
    else:
        how = f"melo task show {source}"
    head = f"{getattr(symbols, symbol)} {word}: {title}"
    body = f" {symbols.dot} {said}" if said else ""
    room = max(8, width - len(how) - 3)
    line = Text(cut_cells(head + body, room, symbols.ellipsis), style=style)
    line.append(f"  {how}", style="muted")
    return line


def task_question(user_text: str) -> tuple[str, str] | None:
    """The title and problem of a task's question a chat's turn answers; None for another turn."""
    lines = user_text.splitlines()
    found = _ASK_LEAD.fullmatch(lines[0].strip()) if lines else None
    if found is None:
        return None
    problem = next((ln[9:] for ln in lines if ln.startswith("Problem: ")), "")
    return found["title"], problem


def asked_line(title: str, problem: str, symbols: Any, width: int) -> Text:
    """``Task “X” asks: <problem>``, muted: what stands for the chat's turn on it."""
    words = t_in("en", "ask.asksAbout", title=title, problem=problem)
    words = clean_output(" ".join(words.split()))
    return Text(cut_cells(words, max(8, width), symbols.ellipsis), style="muted")


def _ask_echo(echo: Mapping[str, Any], symbols: Any, width: int) -> Text | None:
    """The rows of a task's question, as the page shows them; None for another echo."""
    kind, title = echo.get("echo"), str(echo.get("title") or "")
    if kind == "task_ask":
        fields = echo.get("fields")
        problem = fields.get("problem") if isinstance(fields, Mapping) else ""
        return asked_line(title, str(problem or ""), symbols, width)
    if kind == "task_answered":
        said = clean_output(" ".join(str(echo.get("line") or "").split()))
        text = clean_output(" ".join(str(echo.get("text") or "").split()))
        head = f"→ {said}" + (f" {symbols.dot} {text}" if text else "")
        return Text(
            cut_cells(head, max(8, width - 2), symbols.ellipsis), style="accent"
        )
    if kind == "task_escalate":
        parts = echo.get("parts")
        question = parts.get("question") if isinstance(parts, Mapping) else ""
        words = t_in("en", "ask.needsAbout", title=title, question=question or "")
        words = clean_output(" ".join(words.split()))
        if int(echo.get("waiting") or 1) > 1:
            how = f"several wait: /answer {clean_output(title)} …, or melo console"
        else:
            how = "answer in a message here, or in melo console"
        room = max(8, width - len(how) - 4)
        line = Text(cut_cells(f"! {words}", room, symbols.ellipsis), style="warn")
        line.append(f"  {how}", style="muted")
        return line
    return None


def echo_line(echo: Mapping[str, Any], symbols: Any, width: int) -> Text:
    """What the main chat did with a task, as one line (Started, Paused, …).

    An echo with no words (a queued message that ran) is an empty line, not drawn.
    """
    asked = _ask_echo(echo, symbols, width)
    if asked is not None:
        return asked
    said = clean_output(" ".join(str(echo.get("line") or "").split()))
    if not said:
        return Text("")
    kind = echo.get("echo")
    style = "error" if kind == "failed" else "accent"
    # What melo tidied while the person was away says how to put it back.
    hint = f" {symbols.dot} /memory undo" if kind == "sleep" else ""
    line = Text(
        cut_cells(f"→ {said}", max(8, width - 2 - len(hint)), symbols.ellipsis),
        style=style,
    )
    if hint:
        line.append(hint, style="muted")
    if kind == "ask":
        line.append("  /task and your message starts it", style="muted")
    owner = "Main chat" if echo.get("from_main") else echo.get("from_chat")
    if owner:  # a task of another chat says whose it is
        line.append(f"  from {clean_output(str(owner))}", style="muted")
    return line


def _sleeps(echoes: Sequence[Any]) -> list[dict[str, Any]]:
    """The "melo tidied your notes" lines among *echoes* that no later line undid, oldest first."""
    rows = [e for e in echoes if isinstance(e, dict) and isinstance(e.get("id"), int)]
    undone = {e.get("of") for e in rows}
    return [e for e in rows if e.get("echo") == "sleep" and e["id"] not in undone]


def approval_request(pending: Mapping[str, Any]) -> ApprovalRequest:
    """A pending approval as the gateway lists it, for the terminal's approval list."""
    review = pending.get("review")
    found: ActionReview | None = None
    if isinstance(review, dict) and review.get("verdict") in (
        "routine",
        "high",
        "undecided",
    ):
        found = ActionReview(
            verdict=review["verdict"],
            categories=tuple(str(c) for c in review.get("categories") or ()),
            reasons=tuple(str(r) for r in review.get("reasons") or ()),
        )
    arguments = pending.get("arguments")
    root = pending.get("workspace_root")
    return ApprovalRequest(
        request_id=str(pending.get("request_id") or ""),
        run_id="",
        call_id=str(pending.get("call_id") or ""),
        tool_name=str(pending.get("tool_name") or ""),
        arguments=dict(arguments) if isinstance(arguments, dict) else {},
        preview=str(pending.get("preview") or ""),
        workspace_root=root if isinstance(root, str) else None,
        review=found,
    )


# ── The chat ──


class Host(Protocol):
    """What the main chat needs of the terminal session."""

    out: Out
    renderer: TurnRenderer
    approver: TerminalApprover
    title: str

    @property
    def state(self) -> str: ...

    @property
    def workspace(self) -> str: ...

    def on_event(self, stored: StoredEvent) -> None: ...

    def spawn(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]: ...

    def follow_turn(self, text: str, *, echo: bool = True) -> bool: ...

    async def choose_from(
        self, title: str, options: Sequence[str], *, default: int = 0
    ) -> int | None: ...


@dataclass
class Chat:
    """The open chat: the main chat, a side chat, or a side chat its first message makes."""

    id: str | None
    title: str
    main: bool = False
    named: bool = False  # /side TITLE: the title to give it once it exists
    # Ask or Auto, as the gateway stores it; for a chat not made yet, what it
    # will start in. ``picked`` is set when the person chose one for it.
    mode: ChatMode = "ask"
    picked: bool = False


@dataclass
class _Turn:
    key: str  # its client_request_id; "" for a turn another window started
    done: asyncio.Future[str]  # its final status
    number: int | None = None
    turn_id: str = ""
    stop: bool = False  # a stop that waits for the turn's id
    # A retry has no key: the next turn to start in the idle chat is it.
    retry: bool = False


class MainChat:
    """The terminal's side of the main chat (see the module docstring)."""

    def __init__(
        self,
        host: Host,
        client: GatewayClient,
        link: Link,
        *,
        new_mode: ChatMode = "ask",
    ) -> None:
        self.host, self.client, self.link = host, client, link
        # The mode a side chat starts in (``[approvals] mode``) until the person picks one.
        self.new_mode: ChatMode = new_mode
        self.chat = Chat(None, MAIN_TITLE, main=True, mode=new_mode)
        self.main_id = ""
        self._cursors: dict[str, int] = {}
        self._follow: asyncio.Task[None] | None = None
        self._turn: _Turn | None = None
        self._asks: dict[str, asyncio.Task[None]] = {}
        # The card and echo events already drawn, by id.
        self._seen: set[int] = set()
        self._lost = False
        # Set once the main chat is open (or could not open).
        self._ready = asyncio.Event()

    # ── Opening chats ──

    async def start(self) -> None:
        """Open the main chat, with a recap of where it was."""
        try:
            main, _ = await self.client.chats()
            self.main_id = str(main["id"])
            await self.open(Chat(self.main_id, MAIN_TITLE, main=True))
        finally:
            self._ready.set()

    async def watch(self) -> None:
        """Look at the chat list every :data:`POLL_S` seconds while the UI is open.

        The gateway writes the main chat's new cards when its list is read,
        so they reach the main chat's stream within a few seconds.
        """
        while True:
            await asyncio.sleep(POLL_S)
            with contextlib.suppress(GatewayError):
                main, side = await self.client.chats()
                self._read_mode([main, *side])

    def _read_mode(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Take the open chat's mode from its row: another window may have switched it."""
        chat = self.chat
        for row in rows:
            if chat.id is not None and str(row.get("id")) == chat.id:
                chat.mode = chat_mode(row, chat.mode)

    async def open(self, chat: Chat) -> None:
        """Make *chat* the open one: its stream follows from where it was left."""
        await self._stop_following()
        self.chat = chat
        self.host.title = chat.title
        if chat.id is None:
            return  # its first message makes it
        with contextlib.suppress(GatewayError):
            found = (await self.client.session(chat.id)).get("session")
            chat.mode = chat_mode(found, chat.mode)
        after = self._cursors.get(chat.id)
        if after is None:
            after = await self._first_look(chat.id)
        self._follow_from(chat.id, after)

    def _follow_from(self, chat_id: str, after: int) -> None:
        self._cursors[chat_id] = after
        self._follow = self.host.spawn(self._follow_chat(chat_id, after))

    async def _stop_following(self) -> None:
        follow, self._follow = self._follow, None
        if follow is not None and not follow.done():
            follow.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await follow
        for task in list(self._asks.values()):
            task.cancel()
        self._asks.clear()

    async def _first_look(self, chat_id: str) -> int:
        """Show the chat's last turns and newer cards; where its stream starts.

        A turn still running is not in the recap: the stream starts at its
        beginning, so it draws as it goes on.
        """
        after = int((await self.client.session(chat_id)).get("last_event_id") or 0)
        page = await self.client.turns(chat_id, RECAP_TURNS)
        turns = [t for t in page.get("turns") or [] if isinstance(t, dict)]
        if turns and turns[-1].get("status") == "running":
            running = await self.client.turns(chat_id, 1)
            after = int(running.get("replay_from") or 0)
            turns = turns[:-1]
        self._recap(turns, page.get("cards") or [], page.get("echoes") or [])
        return after

    def _recap(
        self, turns: list[dict[str, Any]], cards: list[Any], echoes: Sequence[Any] = ()
    ) -> None:
        out, symbols, width = (
            self.host.out,
            self.host.renderer.symbols,
            self.host.out.width,
        )
        finished = [t for t in turns if not t.get("undone")]
        if not finished:
            return
        # Under a command's echo the connector and the hang indent already
        # set the rows in; at the start they are indented by two.
        lead = "" if hanging.active() else "  "
        for turn in finished:
            typed = " ".join(str(turn.get("user_text") or "").split())
            answer = " ".join(str(turn.get("answer") or "").split())
            asked = task_question(str(turn.get("user_text") or ""))
            if asked is not None:
                line = asked_line(*asked, symbols, width - len(lead))
                line.pad_left(len(lead))
                out.print(line)
            else:
                out.print(
                    Text(
                        cut_cells(
                            f"{lead}{symbols.prompt} {clean_output(typed)}", width
                        ),
                        style="muted",
                    )
                )
            if answer:
                out.print(
                    Text(
                        cut_cells(f"{lead}  {clean_output(answer)}", width),
                        style="muted",
                    )
                )
        since = float(finished[-1].get("started_at") or 0)
        newer = [
            c for c in cards if isinstance(c, dict) and float(c.get("ts") or 0) > since
        ]
        for card in newer[-RECAP_CARDS:]:
            if isinstance(card.get("id"), int):
                self._seen.add(card["id"])
            out.print(card_line(card, symbols, width))
        # What melo tidied since the last turn and nobody put back.
        for echo in _sleeps(echoes):
            if float(echo.get("ts") or 0) > since:
                self._seen.add(int(echo["id"]))
                out.print(echo_line(echo, symbols, width))
        out.print(Text(""))

    # ── The stream ──

    async def _follow_chat(self, chat_id: str, after: int) -> None:
        while True:
            try:
                async for event in self.client.events(chat_id, after):
                    if self._lost:
                        self._lost = False
                        self._say("Connected to the gateway again.", "muted")
                    event_id = event.get("id")
                    if isinstance(event_id, int):
                        after = self._cursors[chat_id] = event_id
                    self._dispatch(chat_id, event)
                problem = "its stream ended"
            except GatewayError as exc:
                if exc.code == "not_found":
                    self._say(
                        "This chat was deleted in another window. /main opens the main chat.",
                        "error",
                    )
                    self._fail_turn("the chat was deleted")
                    return
                problem = str(exc)
            if not self._lost:
                self._lost = True
                self._say(
                    f"The gateway is not answering ({problem}); trying again.", "warn"
                )
                self._fail_turn("the gateway stopped answering during this turn")
            await asyncio.sleep(RETRY_S)

    def _dispatch(self, chat_id: str, event: Mapping[str, Any]) -> None:
        kind, data = event.get("type"), event.get("data")
        if not isinstance(data, Mapping):
            return
        event_id = event.get("id")
        if kind in (CARD_EVENT, ECHO_EVENT):
            if isinstance(event_id, int):
                if event_id in self._seen:
                    return
                self._seen.add(event_id)
            draw = card_line if kind == CARD_EVENT else echo_line
            line = draw(data, self.host.renderer.symbols, self.host.out.width)
            if not line.plain:
                return
            # A card stands apart from the turns around it.
            self.host.out.print(*([line, Text("")] if kind == CARD_EVENT else [line]))
            return
        turn, number = self._turn, event.get("turn")
        if kind == "turn.started":
            key = data.get("client_request_id")
            if (
                turn is not None
                and turn.number is None
                and (turn.retry or (turn.key and key == turn.key))
            ):
                turn.number = (
                    data.get("turn") if isinstance(data.get("turn"), int) else None
                )
                turn.turn_id = str(data.get("turn_id") or "")
                self._stop_if_wanted(turn)
            elif turn is None and self.host.state == "idle":
                # A message sent in another window (the console, the app).
                loop = asyncio.get_running_loop()
                number = data.get("turn")
                self._turn = _Turn("", loop.create_future(), number=number)
                self._turn.turn_id = str(data.get("turn_id") or "")
                text = str(data.get("user_text") or "")
                # A task's question: its row above says so; its frame is never shown.
                mine = not (data.get("origin") or task_question(text))
                if not self.host.follow_turn(text, echo=mine):
                    self._turn = None
            return
        if turn is None or turn.number is None or number != turn.number:
            return
        stored = {
            k: v for k, v in event.items() if k not in ("id", "session_id", "turn")
        }
        self.host.on_event(
            StoredEvent(
                event_id if isinstance(event_id, int) else None, chat_id, number, stored
            )
        )
        if kind == "approval.requested":
            request_id = str(data.get("request_id") or "")
            if request_id and request_id not in self._asks:
                self._asks[request_id] = self.host.spawn(self._ask(chat_id, request_id))
        elif kind == "approval.resolved":
            asked = self._asks.pop(str(data.get("request_id") or ""), None)
            if asked is not None and not asked.done():
                asked.cancel()  # answered in another window: the list goes away
        elif kind == "turn.finished" and not turn.done.done():
            turn.done.set_result(str(data.get("status") or ""))

    def _fail_turn(self, why: str) -> None:
        turn = self._turn
        if turn is not None and not turn.done.done():
            turn.done.set_exception(GatewayError(why))

    def _say(self, text: str, style: str) -> None:
        self.host.out.print(Text(text, style=style))

    # ── Approvals ──

    async def _ask(self, chat_id: str, request_id: str) -> None:
        """The terminal's approval list for one call, answered through the gateway.

        The arguments come from the session's details, not the event, which
        may cut them; a call too large to show whole is left to the console.
        """
        try:
            details = await self.client.session(chat_id)
            pending = next(
                (
                    p
                    for p in details.get("pending_approvals") or []
                    if isinstance(p, dict) and p.get("request_id") == request_id
                ),
                None,
            )
            if pending is None:
                return  # answered already
            if pending.get("arguments_complete") is not True:
                self._say(
                    "This call is too large to show here: answer it in the web console (melo console).",
                    "warn",
                )
                return
            decision = await self.host.approver.ask(approval_request(pending))
            # Answered here: the gateway's "resolved" for it cancels nothing now.
            if self._asks.get(request_id) is asyncio.current_task():
                del self._asks[request_id]
            choice = (
                "deny"
                if not decision.allow
                else "session" if decision.remember else "once"
            )
            await self.client.answer(request_id, choice, decision.reason)
        except GatewayError as exc:
            self._say(f"Could not answer the approval: {exc}", "error")
        finally:
            if self._asks.get(request_id) is asyncio.current_task():
                del self._asks[request_id]

    # ── Turns ──

    async def last_turn(self) -> tuple[str, str] | None:
        """The open chat's newest finished turn, its id and text; None without one."""
        if self.chat.id is None:
            return None
        page = await self.client.turns(self.chat.id, 1)
        turns = [t for t in page.get("turns") or [] if isinstance(t, dict)]
        if not turns or turns[-1].get("status") == "running":
            return None
        if turns[-1].get("origin"):
            return None  # a task's question is not the person's to run again
        return str(turns[-1].get("id") or ""), str(turns[-1].get("user_text") or "")

    async def run_turn(self, text: str, *, retry: str = "") -> str:
        """Send *text* in the open chat and wait for its turn; its final status.

        ``echo`` when the main chat answered with an echo line (it started or
        changed a task) and no turn ran. With *retry*, the gateway undoes
        that turn and runs its text again.
        """
        loop = asyncio.get_running_loop()
        key = "" if retry else uuid.uuid4().hex
        turn = _Turn(key, loop.create_future(), retry=bool(retry))
        self._turn = turn
        try:
            if self.chat.main and self.chat.id is None:
                await self._ready.wait()  # a message typed as melo opens
                if self.chat.id is None:
                    raise GatewayError("the main chat did not open")
            chat = self.chat
            if retry and chat.id is not None:
                body = await self.client.retry(chat.id, retry)
            elif chat.id is None:
                body = await self.client.create(
                    text,
                    turn.key,
                    self.host.workspace,
                    chat.mode if chat.picked else None,
                )
                session = body.get("session") or {}
                chat.id = str(session.get("id") or "")
                chat.mode = chat_mode(session, chat.mode)
                if not chat.named:
                    chat.title = str(session.get("title") or title_from_text(text))
                    self.host.title = chat.title
                self._follow_from(chat.id, 0)
                if chat.named:
                    with contextlib.suppress(GatewayError):
                        await self.client.rename(chat.id, chat.title)
            else:
                body = await self.client.send(chat.id, text, turn.key)
            echo = body.get("echo")
            if isinstance(echo, dict):
                event_id = echo.get("id")
                if not (isinstance(event_id, int) and event_id in self._seen):
                    if isinstance(event_id, int):
                        self._seen.add(event_id)
                    symbols, width = self.host.renderer.symbols, self.host.out.width
                    self.host.out.print(echo_line(echo, symbols, width))
                return "echo"
            started = body.get("turn")
            number = started.get("turn") if isinstance(started, dict) else None
            if turn.number is None and isinstance(number, int):
                turn.number = number
            turn.turn_id = turn.turn_id or str(body.get("turn_id") or "")
            self._stop_if_wanted(turn)
            return await turn.done
        finally:
            if self._turn is turn:
                self._turn = None

    async def wait(self) -> str:
        """Wait for the turn another window started (see :meth:`_dispatch`)."""
        turn = self._turn
        if turn is None:
            return ""
        try:
            return await turn.done
        finally:
            if self._turn is turn:
                self._turn = None

    def stop(self) -> None:
        """Esc: ask the gateway to stop the running turn, once its id is known."""
        turn = self._turn
        if turn is None or turn.stop:
            return
        turn.stop = True
        self._stop_if_wanted(turn)

    def _stop_if_wanted(self, turn: _Turn) -> None:
        if turn.stop and turn.turn_id:
            turn_id, turn.turn_id = turn.turn_id, ""
            self.host.spawn(self._cancel(turn_id))

    async def _cancel(self, turn_id: str) -> None:
        with contextlib.suppress(GatewayError):
            await self.client.cancel(turn_id)

    # ── Commands ──

    async def undo_sleep(self) -> None:
        """/memory undo: put back the newest tidy-up the main chat shows that is not undone.

        The gateway does it as it does for the console's Undo, and answers with
        the line that says what was put back; ``ValueError`` when there is none.
        """
        await self._ready.wait()
        if not self.main_id:
            raise GatewayError("the main chat did not open")
        page = await self.client.turns(self.main_id, UNDO_TURNS)
        shown = _sleeps(page.get("echoes") or [])
        if not shown:
            raise ValueError(t_in("en", "slash.memoryUndoNone"))
        echo = await self.client.tap(int(shown[-1]["id"]), "undo")
        event_id = echo.get("id")
        # The stream may have drawn it already; a side chat's stream is not the main chat's.
        if isinstance(event_id, int) and event_id in self._seen:
            return
        if isinstance(event_id, int):
            self._seen.add(event_id)
        symbols, width = self.host.renderer.symbols, self.host.out.width
        self.host.out.print(echo_line(echo, symbols, width))

    @property
    def mode(self) -> ChatMode:
        """The open chat's mode: Ask or Auto."""
        return self.chat.mode

    async def set_mode(self, mode: ChatMode) -> None:
        """Switch the open chat to *mode* on the gateway (``/approvals`` and Shift+Tab).

        The gateway stores it with the chat and applies it from the chat's
        next step. A side chat its first message has not made yet has
        nothing stored: it starts in the mode picked here.
        """
        chat = self.chat
        if chat.main and chat.id is None:
            await self._ready.wait()  # as melo opens
        if chat.id is None:
            if chat.main:
                raise GatewayError("the main chat did not open")
            chat.mode, chat.picked = mode, True
            return
        stored = await self.client.set_mode(chat.id, mode)
        chat.mode = chat_mode(stored, mode)

    async def pick(self) -> None:
        """/chats: the main chat and its side chats as a list; Enter opens one."""
        await self._ready.wait()
        main, side = await self.client.chats()
        chats = [Chat(str(main["id"]), MAIN_TITLE, main=True)]
        rows = [_row(MAIN_TITLE, main)]
        for data in side:
            title = str(data.get("title") or "Untitled")
            chats.append(Chat(str(data.get("id")), title))
            rows.append(_row(title, data))
        current = next(
            (n for n, c in enumerate(chats) if c.id == self.chat.id), len(chats)
        )
        rows = [f"{row} (open)" if n == current else row for n, row in enumerate(rows)]
        index = await self.host.choose_from("Chats", [*rows, NEW_SIDE], default=current)
        if index is None or index == current:
            return
        if index == len(chats):
            await self.side("")
        else:
            await self.open(chats[index])
            self._opened()

    async def side(self, title: str) -> None:
        """/side [TITLE]: a new side chat in this folder, made by its first message."""
        await self._ready.wait()
        await self.open(
            Chat(None, title or NEW_SIDE, named=bool(title), mode=self.new_mode)
        )
        where = home_label(self.host.workspace)
        self._say(
            f"◆ {self.chat.title} in {where}: your first message starts it. /main returns.",
            "accent",
        )

    async def main(self) -> None:
        """/main: back to the main chat."""
        await self._ready.wait()
        if self.chat.main:
            self._say("◆ This is the main chat.", "muted")
            return
        await self.open(Chat(self.main_id, MAIN_TITLE, main=True))
        self._opened()

    def _opened(self) -> None:
        if self.chat.main:
            self._say("◆ Main chat · /chats lists the side chats", "accent")
        else:
            self._say(f"◆ Side chat: {self.chat.title} · /main returns", "accent")

    async def close(self) -> None:
        """Stop following; a turn still running goes on in the gateway."""
        self._fail_turn("melo is closing")
        await self._stop_following()
        await self.client.aclose()


def _row(title: str, data: Mapping[str, Any]) -> str:
    """One row of /chats: the title, then what waits or runs, then the folder."""
    parts = [clean_output(title)]
    pending = data.get("pending_approvals")
    if isinstance(pending, int) and pending > 0:
        parts.append("waits for your approval")
    elif data.get("busy"):
        parts.append("working")
    label = data.get("workspace_label")
    if isinstance(label, str) and label and not data.get("main"):
        parts.append(clean_output(label))
    return " · ".join(parts)


__all__ = [
    "Chat",
    "GatewayClient",
    "GatewayError",
    "Link",
    "MainChat",
    "Offline",
    "approval_request",
    "card_line",
    "connect",
    "echo_line",
    "store_owner",
]
