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
import shutil
import subprocess
import sys
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
from localmelo.support import console_link, task_cli
from localmelo.support.cli.text import cut_cells
from localmelo.support.gateway.console.cards import CARD_EVENT
from localmelo.support.gateway.console.front import ECHO_EVENT
from localmelo.support.sessions.display import clean_output, home_label
from localmelo.support.sessions.store import (
    GATEWAY_SESSIONS_DB,
    StoredEvent,
    title_from_text,
)

if TYPE_CHECKING:
    from localmelo.support.cli.approvals import TerminalApprover
    from localmelo.support.cli.output import Out
    from localmelo.support.cli.render import TurnRenderer
    from localmelo.support.config import Config

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
MAIN_TITLE: Final = "Main chat"
NEW_SIDE: Final = "New side chat"
_CARD_WORDS: Final = {
    "done": ("ok", "ok", "Done"),
    "failed": ("failed", "error", "Not finished"),
    "waiting": ("warn", "warn", "Waits for you"),
}


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
                "it keeps running after melo quits."
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
        return subprocess.Popen(
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
    spawn: Callable[[str, int], subprocess.Popen[bytes]] = spawn_gateway,
    start_s: float = START_S,
    clock: Callable[[], float] = time.monotonic,
) -> Link | Offline:
    """The gateway that holds the main chat, started when none runs (blocking).

    The configured port first; then, when another gateway holds the main
    chat's file (the app's, on a port of its own), the port it listens on;
    else a new gateway on the configured port. *on_start* runs just before
    one starts, so the terminal can say so.
    """
    host, port = cfg.gateway.host, cfg.gateway.port
    url = _url(host, port)
    process: subprocess.Popen[bytes] | None = None
    # A gateway another melo is starting gets a moment; one started here, all of *start_s*.
    deadline = clock() + start_s / 4
    while True:
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


def _path(session_id: str) -> str:
    return "/api/sessions/" + quote(session_id, safe="")


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
            "GET", _path(session_id) + "/turns", params={"limit": limit}
        )

    async def send(self, session_id: str, text: str, key: str) -> dict[str, Any]:
        body = {"text": text, "client_request_id": key}
        return await self.call("POST", _path(session_id) + "/turns", json=body)

    async def create(self, text: str, key: str, workspace: str) -> dict[str, Any]:
        """A new side chat in *workspace*, with *text* as its first message."""
        body = {
            "workspace": workspace,
            "turn": {"text": text, "client_request_id": key},
        }
        return await self.call("POST", "/api/sessions", json=body)

    async def rename(self, session_id: str, title: str) -> None:
        await self.call("PATCH", _path(session_id), json={"title": title})

    async def cancel(self, turn_id: str) -> None:
        await self.call("POST", f"/api/turns/{quote(turn_id, safe='')}/cancel")

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
                _path(session_id) + "/events",
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


def echo_line(echo: Mapping[str, Any], symbols: Any, width: int) -> Text:
    """What the main chat did with a task, as one line (Started, Paused, …)."""
    said = clean_output(" ".join(str(echo.get("line") or "").split()))
    kind = echo.get("echo")
    style = "error" if kind == "failed" else "accent"
    line = Text(
        cut_cells(f"→ {said}", max(8, width - 2), symbols.ellipsis), style=style
    )
    if kind == "ask":
        line.append("  /task and your message starts it", style="muted")
    return line


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

    def follow_turn(self, text: str) -> bool: ...

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


@dataclass
class _Turn:
    key: str  # its client_request_id; "" for a turn another window started
    done: asyncio.Future[str]  # its final status
    number: int | None = None
    turn_id: str = ""
    stop: bool = False  # a stop that waits for the turn's id


class MainChat:
    """The terminal's side of the main chat (see the module docstring)."""

    def __init__(self, host: Host, client: GatewayClient, link: Link) -> None:
        self.host, self.client, self.link = host, client, link
        self.chat = Chat(None, MAIN_TITLE, main=True)
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
                await self.client.chats()

    async def open(self, chat: Chat) -> None:
        """Make *chat* the open one: its stream follows from where it was left."""
        await self._stop_following()
        self.chat = chat
        self.host.title = chat.title
        if chat.id is None:
            return  # its first message makes it
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
        self._recap(turns, page.get("cards") or [])
        return after

    def _recap(self, turns: list[dict[str, Any]], cards: list[Any]) -> None:
        out, symbols, width = (
            self.host.out,
            self.host.renderer.symbols,
            self.host.out.width,
        )
        finished = [t for t in turns if not t.get("undone")]
        if not finished:
            return
        for turn in finished:
            typed = " ".join(str(turn.get("user_text") or "").split())
            answer = " ".join(str(turn.get("answer") or "").split())
            out.print(
                Text(
                    cut_cells(f"  {symbols.prompt} {clean_output(typed)}", width),
                    style="muted",
                )
            )
            if answer:
                out.print(
                    Text(cut_cells(f"    {clean_output(answer)}", width), style="muted")
                )
        since = float(finished[-1].get("started_at") or 0)
        newer = [
            c for c in cards if isinstance(c, dict) and float(c.get("ts") or 0) > since
        ]
        for card in newer[-RECAP_CARDS:]:
            if isinstance(card.get("id"), int):
                self._seen.add(card["id"])
            out.print(card_line(card, symbols, width))
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
            self.host.out.print(
                draw(data, self.host.renderer.symbols, self.host.out.width)
            )
            return
        turn, number = self._turn, event.get("turn")
        if kind == "turn.started":
            if (
                turn is not None
                and turn.number is None
                and turn.key
                and data.get("client_request_id") == turn.key
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
                if not self.host.follow_turn(str(data.get("user_text") or "")):
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

    async def run_turn(self, text: str) -> str:
        """Send *text* in the open chat and wait for its turn; its final status.

        ``echo`` when the main chat answered with an echo line (it started or
        changed a task) and no turn ran.
        """
        loop = asyncio.get_running_loop()
        turn = _Turn(uuid.uuid4().hex, loop.create_future())
        self._turn = turn
        try:
            if self.chat.main and self.chat.id is None:
                await self._ready.wait()  # a message typed as melo opens
                if self.chat.id is None:
                    raise GatewayError("the main chat did not open")
            chat = self.chat
            if chat.id is None:
                body = await self.client.create(text, turn.key, self.host.workspace)
                session = body.get("session") or {}
                chat.id = str(session.get("id") or "")
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
        await self.open(Chat(None, title or NEW_SIDE, named=bool(title)))
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
