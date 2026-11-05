"""The ``chrome`` driver: Google Chrome in the background, with no window.

melo starts the Google Chrome installed on this Mac with ``--headless=new``:
no window, no Dock icon, never the front app and never a switch of Space.
It drives Chrome over ``--remote-debugging-pipe``, two pipes on Chrome's file
descriptors 3 and 4 that carry DevTools messages ended by a NUL byte. Chrome
opens no port and writes no ``DevToolsActivePort`` file, so only the process
that holds the pipes can drive it.

The profile is melo's own, ``~/.cache/localmelo/browser/chrome`` (mode 0700,
made with umask 077), never the person's Chrome profile. ``--use-mock-keychain``
keeps Chrome off the login keychain, so its cookies at rest are protected by
the file mode and FileVault only. Sign-ins whose cookies persist are kept from
one chat to the next and across restarts; a cookie that lasts only for the
session is gone once Chrome stops. Downloads are refused.

One Chrome serves every chat in the process (a profile can be open in one
Chrome only). Each driver sees only the tabs it opened and the tabs those
opened (Chrome's ``openerId``), so one chat never lists, reads or adopts
another's. Every tab is a page in a window of its own, so no tab is a hidden
background tab.

Pages are read with the page script Safari's driver uses
(:data:`~localmelo.support.browser.pagescript.PAGE_JS`), in an isolated world the
page's own scripts cannot reach. An act is checked, guarded and looked at by
the same act scripts (:class:`~localmelo.support.browser.pagescript.PageScriptActs`),
which run in the page itself as they do in Safari, and done by Chrome's own
input where it can: trusted mouse and key events at the middle of the element
melo checked, or at the point of a picture. The page script is the backup, and
a choice in a list is always the page script's (Chrome opens no list in the
background).

A page that asks Chrome to open an address of another app's kind
(``chat://...``) is not followed: Chrome in the background drops it. The
driver sees each such request (``Page.frameRequestedNavigation``) and puts it on
the act's result (``ActResult.handoff``), and melo asks the person. Chrome is
never told to allow a scheme.

A tab's page can be streamed as pictures for the live view
(:meth:`ChromeBackgroundDriver.start_screencast`, Chrome's ``Page.startScreencast``):
Chrome sends a JPEG only when the page changes, and sends the next one only
after the last was acknowledged, so the acknowledgements are held to a
rate (default 15 a second) and a busy page costs a fraction of its full rate.
A picture goes to the one function that asked for it and is kept nowhere else.

Chrome's Local Network Access check stays on: a page cannot reach this
computer or its network, and Chrome in the background cannot ask the person
to allow it. With a :class:`~localmelo.support.browser.local_apps.LocalRule`
(``local``, which melo's browser tools pass), Chrome pauses the requests to a
listed sign-in helper and to addresses that name this computer or its
network (``Fetch`` for the whole browser). melo answers a listed helper's
request from its own sign-in page itself, blocks the others before they are
sent and tells the step (:meth:`ChromeBackgroundDriver.local_attempts`), and
lets anything else go on. No permission is ever granted.

A box the page puts up blocks it until it is answered: an ``alert`` is closed
with OK; a ``confirm`` during an act is put to the person through
:attr:`ChromeBackgroundDriver.on_dialog` and is otherwise answered Cancel; a
``prompt`` is always answered Cancel (melo types only where it was asked to);
a page that asks before it is left is left, since the step that leaves it
was allowed. The act's result lists them (``ActResult.dialogs``).
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import binascii
import contextlib
import dataclasses
import fcntl
import json
import os
import plistlib
import shlex
import shutil
import signal
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast

from localmelo.support import config
from localmelo.support.browser import apps, local_apps
from localmelo.support.browser.driver import (
    MAX_TYPED_CHARS,
    Act,
    ActResult,
    BrowserError,
    BrowserSetupError,
    Dialog,
    Shot,
    TabInfo,
    View,
)
from localmelo.support.browser.live import DEFAULT_FPS, DEFAULT_QUALITY, Frame
from localmelo.support.browser.local_apps import LocalAttempt, LocalRule
from localmelo.support.browser.pagescript import (
    PASSED_OVER,
    SETTLE_S,
    STALE,
    PageScriptActs,
    act_script,
    act_spec,
    as_number,
    cover_stop,
    covered_why,
    decode,
    new_key,
    nothing_reached,
    observed,
    reached,
    stopped,
    wait_probe,
)

# A command line that replaces Chrome's, for tests (a fake peer) and a Chrome
# installed elsewhere.
CHROME_ENV = "LOCALMELO_CHROME"
CHROME_APPS = (
    "/Applications/Google Chrome.app",
    "~/Applications/Google Chrome.app",
)
APP_BINARY = "Contents/MacOS/Google Chrome"
MISSING_MESSAGE = (
    "Google Chrome is not installed, so melo cannot browse in the background. "
    "Install Google Chrome (google.com/chrome), or switch browsing to Safari "
    "windows (/browser safari), which come to the front."
)
STOPPED_MESSAGE = "Chrome stopped. Ask again: melo starts it again."
# The size of a tab's window, in page pixels; the page's view (and its picture)
# is as wide, and lower by the window's own bar.
VIEW_W = 1280
VIEW_H = 800
SHOT_QUALITY = 70
MAX_SHOT_BYTES = 10 * 1024 * 1024
# The most one message from Chrome may hold (a picture, a long page).
MAX_MESSAGE = 64 * 1024 * 1024
STARTUP_S = 20.0
CALL_S = 60.0
CLOSE_S = 5.0
DIALOG_S = 5.0
# The most addresses of other apps' kinds one tab keeps until an act takes them.
MAX_HANDOFFS = 5
# The most one streamed picture may hold, and how long starting or stopping a stream may take.
MAX_FRAME_BYTES = 4 * 1024 * 1024
STREAM_S = 10.0
WORLD = "melo"
# Chrome's flags, as the background browser needs them. The pipe is the only
# way in: never ``--remote-debugging-port``.
FLAGS: tuple[str, ...] = (
    "--headless=new",
    "--remote-debugging-pipe",
    "--no-first-run",
    "--no-default-browser-check",
    "--use-mock-keychain",
    "--disable-sync",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-features=Translate,MediaRouter,DialMediaRouteProvider,OptimizationHints",
    "--disable-blink-features=AutomationControlled",
    "--mute-audio",
    f"--window-size={VIEW_W},{VIEW_H}",
)
GPU_OFF = "--disable-gpu"
# The modifier bit of the Command key in a DevTools key event.
_META = 4
# The calls the person's input may make (take over), and the most that may wait for Chrome.
PERSON_METHODS = frozenset(
    {
        "Input.dispatchMouseEvent",
        "Input.dispatchKeyEvent",
        "Input.insertText",
        "Input.imeSetComposition",
    }
)
MAX_INFLIGHT = 64
# The most blocked requests kept for one page until a step tells them, and the most pages kept.
MAX_ATTEMPTS = 10
MAX_ATTEMPT_PAGES = 50
# How far up melo looks for the page a frame is in (a frame in a frame in a page).
MAX_FRAME_DEPTH = 8


def chrome_command() -> list[str] | None:
    """The command that starts Chrome: :data:`CHROME_ENV` when set, else the installed app; None when there is none."""
    said = os.environ.get(CHROME_ENV, "").strip()
    if said:
        return shlex.split(said)
    for app in CHROME_APPS:
        binary = os.path.join(os.path.expanduser(app), APP_BINARY)
        if os.access(binary, os.X_OK):
            return [binary]
    return None


def chrome_installed() -> bool:
    """Whether melo can start Chrome: the app (or the command in :data:`CHROME_ENV`) is there."""
    command = chrome_command()
    if not command:
        return False
    program = command[0]
    return bool(shutil.which(program)) or os.access(program, os.X_OK)


def chrome_version(binary: str) -> str:
    """The version of the Chrome app *binary* belongs to (``156.0.8078.80``), or ``""``."""
    try:
        plist = Path(binary).resolve().parents[1] / "Info.plist"
        with plist.open("rb") as handle:
            found = plistlib.load(handle).get("CFBundleShortVersionString")
    except (OSError, ValueError, IndexError, plistlib.InvalidFileException):
        return ""
    return found if isinstance(found, str) else ""


def user_agent(version: str) -> str:
    """The user agent the same Chrome sends with a window (no ``HeadlessChrome``), or ``""``."""
    major = version.split(".", 1)[0]
    if not major.isdigit():
        return ""
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    )


def default_profile() -> str:
    """melo's own Chrome profile, under the current HOME."""
    return os.path.join(config.config_dir(), "browser", "chrome")


def launch_args(profile: str, version: str, *, gpu: bool = False) -> list[str]:
    """Chrome's arguments after the program: flags, the profile, the user agent, a blank page."""
    args = [*FLAGS]
    if not gpu:
        args.append(
            GPU_OFF
        )  # no contention with a local model; a still page costs nothing
    args.append(f"--user-data-dir={profile}")
    agent = user_agent(version)
    if agent:
        args.append(f"--user-agent={agent}")
    args.append("about:blank")
    return args


def make_profile(path: str) -> None:
    """Make melo's profile folder, private to this user (0700, umask 077); never a link."""
    if os.path.islink(path):
        raise BrowserError(
            f"melo's Chrome profile {path} is a link, so melo did not use it."
        )
    old = os.umask(0o077)
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    finally:
        os.umask(old)
    os.chmod(path, 0o700)


class CdpError(BrowserError):
    """Chrome answered a call with an error; the message is Chrome's."""


# ── The Chrome process, shared by every driver of one profile ──


class _Cast:
    """The stream of one tab's pictures: who gets them, and the pace the acknowledgements keep."""

    def __init__(
        self,
        on_frame: Callable[[Frame], None],
        on_end: Callable[[], None],
        interval: float,
        on_dialog: Callable[[], None] | None = None,
    ) -> None:
        self.on_frame = on_frame
        self.on_end = on_end
        self.interval = interval
        # Told when the page puts up a box or one closes (the live view shows it to the person).
        self.on_dialog = on_dialog
        self.next_ack = 0.0
        self.timers: set[asyncio.TimerHandle] = set()

    def cancel(self) -> None:
        for timer in self.timers:
            timer.cancel()
        self.timers.clear()


class _Chrome:
    """One headless Chrome and its pipe: requests, answers and the events melo keeps.

    The events are kept by session: a box the page put up (``dialogs``, and
    ``woken`` to wake a call that waits), addresses of other apps' kinds a
    page asked for (``handoffs``), and how many times the main frame moved
    (``navs``), so a navigation can be waited for.
    """

    def __init__(self, pid: int, profile: str, loop: asyncio.AbstractEventLoop) -> None:
        self.pid = pid
        self.profile = profile
        self.loop = loop
        self.users = 0
        self.closed = False
        self._writer: asyncio.WriteTransport | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._n = 0
        self.dialogs: dict[str, dict[str, Any]] = {}
        self.woken: dict[str, asyncio.Event] = {}
        self.handoffs: dict[str, list[str]] = {}
        self.navs: dict[str, int] = {}
        self.main: dict[str, str] = {}  # session -> its main frame
        self.casts: dict[str, _Cast] = {}  # session -> its stream of pictures
        self.spare: set[str] = set()  # the pages Chrome started with: nobody's
        self.claimed: set[str] = set()  # the pages some driver has as its own
        # The rule for requests to this computer (None: Chrome's own check alone),
        # the requests it is answering, and what it blocked, by page, until told.
        self.local: LocalRule | None = None
        self.jobs: set[asyncio.Task[None]] = set()
        self.attempts: dict[str, list[LocalAttempt]] = {}
        self.relays = asyncio.Semaphore(local_apps.MAX_RELAYS)

    @classmethod
    async def launch(
        cls,
        argv: list[str],
        profile: str,
        timeout_s: float,
        local: LocalRule | None = None,
    ) -> _Chrome:
        loop = asyncio.get_running_loop()
        program = shutil.which(argv[0]) or argv[0]
        cmd_r, cmd_w = os.pipe()  # melo writes, Chrome reads on fd 3
        res_r, res_w = os.pipe()  # Chrome writes on fd 4, melo reads
        # Above 4, so that putting them on 3 and 4 never meets one of them.
        high_r = fcntl.fcntl(cmd_r, fcntl.F_DUPFD_CLOEXEC, 10)
        high_w = fcntl.fcntl(res_w, fcntl.F_DUPFD_CLOEXEC, 10)
        os.close(cmd_r)
        os.close(res_w)
        actions = [
            (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
            (os.POSIX_SPAWN_OPEN, 1, os.devnull, os.O_WRONLY, 0),
            (os.POSIX_SPAWN_OPEN, 2, os.devnull, os.O_WRONLY, 0),
            (os.POSIX_SPAWN_DUP2, high_r, 3),
            (os.POSIX_SPAWN_DUP2, high_w, 4),
        ]
        old = os.umask(0o077)  # what Chrome writes in the profile is private too
        try:
            pid = os.posix_spawn(
                program, argv, dict(os.environ), file_actions=actions, setpgroup=0
            )
        except OSError as exc:
            for fd in (cmd_w, res_r):
                os.close(fd)
            raise BrowserError(f"Chrome did not start: {exc}.") from None
        finally:
            os.umask(old)
            os.close(high_r)
            os.close(high_w)
        chrome = cls(pid, profile, loop)
        _ALIVE[pid] = chrome
        _kill_at_exit()
        try:
            reader = asyncio.StreamReader(limit=MAX_MESSAGE)
            await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader),
                os.fdopen(res_r, "rb", 0),
            )
            writer, _ = await loop.connect_write_pipe(
                asyncio.Protocol, os.fdopen(cmd_w, "wb", 0)
            )
            chrome._writer = cast(asyncio.WriteTransport, writer)
            chrome._reader = asyncio.ensure_future(chrome._read(reader))
            async with asyncio.timeout(timeout_s):
                await chrome.request("Browser.getVersion")
            await chrome.request("Browser.setDownloadBehavior", {"behavior": "deny"})
            if local is not None:
                # For the whole browser: every tab, frame and worker, before any loads.
                chrome.local = local
                await chrome.request("Fetch.enable", {"patterns": local.patterns()})
            listed = await chrome.request("Target.getTargets")
        except TimeoutError:
            chrome.kill()
            raise BrowserError(
                f"Chrome did not answer within {timeout_s:g} s of starting."
            ) from None
        except BaseException:
            chrome.kill()
            raise
        chrome.spare = {
            str(t.get("targetId"))
            for t in listed.get("targetInfos") or []
            if isinstance(t, dict) and t.get("type") == "page"
        }
        return chrome

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                chunk = await reader.readuntil(b"\0")
                try:
                    message = json.loads(chunk[:-1])
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._dispatch(message)
        except (
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ConnectionError,
            OSError,
        ):
            pass
        finally:
            self._stopped()

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message:
            future = self._pending.pop(message["id"], None)
            if future is None or future.done():
                return
            error = message.get("error")
            if isinstance(error, dict):
                said = str(error.get("message") or "error")[:300]
                future.set_exception(CdpError(f"Chrome reported a problem: {said}."))
            else:
                result = message.get("result")
                future.set_result(result if isinstance(result, dict) else {})
            return
        method = str(message.get("method") or "")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        session = str(message.get("sessionId") or "")
        if method == "Page.javascriptDialogOpening":
            self.dialogs[session] = params
            self.woken.setdefault(session, asyncio.Event()).set()
            self._dialog_changed(session)
        elif method == "Page.javascriptDialogClosed":
            self.dialogs.pop(session, None)
            self._dialog_changed(session)
        elif method == "Page.screencastFrame":
            self._frame(session, params)
        elif method == "Page.frameRequestedNavigation":
            url = str(params.get("url") or "")
            if apps.custom_scheme(url):
                kept = self.handoffs.setdefault(session, [])
                if len(kept) < MAX_HANDOFFS:
                    kept.append(url[:2000])
        elif method == "Page.frameNavigated":
            frame = params.get("frame")
            if isinstance(frame, dict) and not frame.get("parentId"):
                self.main[session] = str(frame.get("id") or "")
                self.navs[session] = self.navs.get(session, 0) + 1
        elif method == "Page.navigatedWithinDocument":
            if params.get("frameId") == self.main.get(session):
                self.navs[session] = self.navs.get(session, 0) + 1
        elif method == "Fetch.requestPaused" and not session and self.local is not None:
            job = self.loop.create_task(self._paused(self.local, params))
            self.jobs.add(job)
            job.add_done_callback(self.jobs.discard)

    async def _paused(self, rule: LocalRule, params: dict[str, Any]) -> None:
        """Answer one request Chrome paused for *rule*: relayed, blocked or let go on; exactly once."""
        ident = params.get("requestId")
        if not isinstance(ident, str) or not ident:
            return
        method = "Fetch.failRequest"
        answer: dict[str, Any] = {"requestId": ident, "errorReason": "Failed"}
        blocked: LocalAttempt | None = None
        try:
            request = params.get("request")
            request = request if isinstance(request, dict) else {}
            origin = local_apps.origin_of(request.get("headers"))
            verdict = rule.decide(str(request.get("url") or ""), origin)
            if verdict.kind == "pass":
                method, answer = "Fetch.continueRequest", {"requestId": ident}
            elif verdict.kind == "relay" and verdict.helper is not None:
                # A relay waits for a free place a bounded time, then fails;
                # requests that are not relayed never wait for one.
                async with asyncio.timeout(local_apps.RELAY_S):
                    await self.relays.acquire()
                try:
                    got = await local_apps.relay(verdict.helper, verdict.port, request)
                finally:
                    self.relays.release()
                method, answer = "Fetch.fulfillRequest", local_apps.fulfilled(
                    ident, got
                )
            else:
                answer["errorReason"] = "BlockedByClient"
                named = verdict.helper.name if verdict.helper is not None else None
                blocked = LocalAttempt(verdict.host, verdict.port, origin, named)
        except local_apps.RelayError as exc:
            answer["errorReason"] = exc.reason
        except Exception:  # never leave a request paused: it fails instead
            pass
        try:
            await self.request(method, answer)
        except CdpError:
            # Chrome refused the answer (a header it would not take, say): the request fails.
            if method != "Fetch.failRequest":
                with contextlib.suppress(BrowserError):
                    await self.request(
                        "Fetch.failRequest",
                        {"requestId": ident, "errorReason": "Failed"},
                    )
        except BrowserError:
            pass  # Chrome stopped
        if blocked is not None:
            with contextlib.suppress(BrowserError):
                page = await self._page_of(str(params.get("frameId") or ""))
                if page:
                    self._keep(page, blocked)

    async def _page_of(self, frame: str) -> str:
        """The page (target id) *frame* is in, or ``""`` when melo cannot tell.

        A page's main frame has the page's id; a frame from another site is a
        target of its own, whose parent leads to the page; a frame in the
        page's own process is found in the frames of the pages melo drives.
        """
        if not frame:
            return ""
        if frame in self.claimed or frame in self.spare:
            return frame
        # Chrome lists a frame from another site, but tells nothing of it by its id alone.
        try:
            listed = await self.request("Target.getTargets")
        except BrowserError:
            listed = {}
        infos = {
            str(t.get("targetId")): t
            for t in listed.get("targetInfos") or []
            if isinstance(t, dict)
        }
        current = frame
        for _ in range(MAX_FRAME_DEPTH):
            info = infos.get(current)
            if info is None:
                break
            if info.get("type") == "page":
                return current
            current = str(info.get("parentId") or "")
            if current in self.claimed or current in self.spare:
                return current
        for session, main in list(self.main.items()):
            try:
                tree = await self.request("Page.getFrameTree", {}, session)
            except BrowserError:
                continue
            if _holds(tree.get("frameTree"), frame):
                return main
        return ""

    def _keep(self, page: str, attempt: LocalAttempt) -> None:
        kept = self.attempts.setdefault(page, [])
        if attempt not in kept and len(kept) < MAX_ATTEMPTS:
            kept.append(attempt)
        while len(self.attempts) > MAX_ATTEMPT_PAGES:
            self.attempts.pop(next(iter(self.attempts)))

    def _dialog_changed(self, session: str) -> None:
        cast = self.casts.get(session)
        if cast is not None and cast.on_dialog is not None:
            with contextlib.suppress(
                Exception
            ):  # a viewer's failure never stops the browser
                cast.on_dialog()

    def _frame(self, session: str, params: dict[str, Any]) -> None:
        """A picture of a tab's page: to the stream's function, and its acknowledgement paced.

        Chrome sends the next picture only after this one is acknowledged, and about
        three are in flight, so acknowledging at most *rate* times a second holds the
        pictures (and the work of drawing them) to about that rate. A picture nobody
        asked for is acknowledged at once.
        """
        ident = params.get("sessionId")
        if not isinstance(ident, int):
            return
        cast = self.casts.get(session)
        if cast is None:
            self._ack(session, ident)
            return
        frame = _picture(params)
        if frame is not None:
            with contextlib.suppress(
                Exception
            ):  # a viewer's failure never stops the browser
                cast.on_frame(frame)
        now = self.loop.time()
        at = max(now, cast.next_ack)
        cast.next_ack = at + cast.interval
        if at - now < 0.001:
            self._ack(session, ident)
            return
        handle: asyncio.TimerHandle | None = None

        def later() -> None:
            if handle is not None:
                cast.timers.discard(handle)
            self._ack(session, ident)

        handle = self.loop.call_later(at - now, later)
        cast.timers.add(handle)

    def _ack(self, session: str, ident: int) -> None:
        if self.closed or self._writer is None or self._writer.is_closing():
            return
        try:
            future = self.request(
                "Page.screencastFrameAck", {"sessionId": ident}, session
            )
        except BrowserError:
            return
        # An answer is not waited for; an error in it (the tab is gone) is not a problem.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())

    def end_cast(self, session: str) -> None:
        cast = self.casts.pop(session, None)
        if cast is not None:
            cast.cancel()

    def _stopped(self) -> None:
        self.closed = True
        for job in list(self.jobs):
            job.cancel()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(BrowserError(STOPPED_MESSAGE))
        self._pending.clear()
        for event in self.woken.values():
            event.set()
        streams, self.casts = list(self.casts.values()), {}
        for stream in streams:
            stream.cancel()
            with contextlib.suppress(Exception):
                stream.on_end()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session: str | None = None,
    ) -> asyncio.Future[dict[str, Any]]:
        """Send one call; its answer comes in the future (a :class:`CdpError` for Chrome's error)."""
        if self.closed or self._writer is None or self._writer.is_closing():
            raise BrowserError(STOPPED_MESSAGE)
        self._n += 1
        message: dict[str, Any] = {"id": self._n, "method": method}
        if params:
            message["params"] = params
        if session:
            message["sessionId"] = session
        future: asyncio.Future[dict[str, Any]] = self.loop.create_future()
        self._pending[self._n] = future
        self._writer.write(json.dumps(message).encode("utf-8") + b"\0")
        return future

    async def close(self) -> None:
        """Ask Chrome to quit, then make sure it did (its whole process group)."""
        if not self.closed:
            with contextlib.suppress(BrowserError, TimeoutError):
                async with asyncio.timeout(CLOSE_S):
                    await self.request("Browser.close")
        if not await self._reaped(CLOSE_S):
            self._signal(signal.SIGTERM)
            if not await self._reaped(2.0):
                self._signal(signal.SIGKILL)
                await self._reaped(2.0)
        self._done()

    def kill(self) -> None:
        """Stop Chrome at once, without waiting."""
        self._signal(signal.SIGKILL)
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(self.pid, os.WNOHANG)
        self._done()

    def _signal(self, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(self.pid, sig)

    async def _reaped(self, seconds: float) -> bool:
        end = self.loop.time() + seconds
        while True:
            try:
                pid, _ = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return True
            if pid:
                return True
            if self.loop.time() >= end:
                return False
            await asyncio.sleep(0.05)

    def _done(self) -> None:
        self.closed = True
        _ALIVE.pop(self.pid, None)
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        for (profile, loop), chrome in list(_SHARED.items()):
            if chrome is self:
                _SHARED.pop((profile, loop), None)


# The Chrome of each profile, per event loop; and every Chrome alive, by pid.
_SHARED: dict[tuple[str, int], _Chrome] = {}
_LOCKS: dict[tuple[str, int], asyncio.Lock] = {}
_ALIVE: dict[int, _Chrome] = {}
_AT_EXIT = False


def _kill_at_exit() -> None:
    global _AT_EXIT
    if not _AT_EXIT:
        _AT_EXIT = True
        atexit.register(_kill_all)


def _kill_all() -> None:
    """No Chrome melo started outlives the process."""
    for chrome in list(_ALIVE.values()):
        chrome.kill()


async def _shared(
    argv: list[str], profile: str, timeout_s: float, local: LocalRule | None = None
) -> _Chrome:
    """The Chrome of *profile*, started now when it is not running; one more user of it.

    A Chrome that starts takes *local*; one that runs keeps the rule it started with.
    """
    loop = asyncio.get_running_loop()
    key = (profile, id(loop))
    lock = _LOCKS.get(key)
    if lock is None:
        lock = _LOCKS[key] = asyncio.Lock()
    async with lock:
        found = _SHARED.get(key)
        if found is not None and (found.closed or found.loop is not loop):
            found.kill()
            found = None
        if found is None:
            for (other, _), chrome in list(_SHARED.items()):
                if other == profile and chrome.loop.is_closed():
                    chrome.kill()  # a Chrome left by a loop that ended holds the profile
            found = await _Chrome.launch(argv, profile, timeout_s, local)
            _SHARED[key] = found
        found.users += 1
        return found


def _holds(tree: object, frame: str) -> bool:
    """Whether the frame tree *tree* (``Page.getFrameTree``) has *frame* in it."""
    if not isinstance(tree, dict):
        return False
    found = tree.get("frame")
    if isinstance(found, dict) and found.get("id") == frame:
        return True
    children = tree.get("childFrames")
    return isinstance(children, list) and any(_holds(c, frame) for c in children)


def _picture(params: dict[str, Any]) -> Frame | None:
    """The :class:`Frame` a ``Page.screencastFrame`` carries, or None when it is not a usable JPEG."""
    data = params.get("data")
    if not isinstance(data, str) or len(data) > MAX_FRAME_BYTES * 4 // 3 + 8:
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw.startswith(b"\xff\xd8") or len(raw) > MAX_FRAME_BYTES:
        return None
    meta = params.get("metadata")
    meta = meta if isinstance(meta, dict) else {}

    def number(key: str, default: float = 0.0) -> float:
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return default
        return float(value) if value == value else default

    return Frame(
        raw,
        width=number("deviceWidth"),
        height=number("deviceHeight"),
        offset_top=number("offsetTop"),
        scale=number("pageScaleFactor", 1.0) or 1.0,
        scroll_x=number("scrollOffsetX"),
        scroll_y=number("scrollOffsetY"),
    )


# ── The driver ──


DialogAsk = Callable[[str, str, str], Awaitable[bool]]


class ChromeBackgroundDriver(PageScriptActs):
    """:class:`BrowserDriver` over headless Chrome and its pipe. See the module docstring.

    *command* replaces the program (default: :func:`chrome_command`),
    *profile* the profile folder (default: :func:`default_profile`) and
    *version* the Chrome version the user agent names (default: the app's).
    *local* is the rule for requests to this computer (see the module
    docstring); without one, Chrome's own check alone applies.
    :attr:`on_dialog` is asked about a ``confirm`` a page puts up during an
    act (its kind, its words and the page's address): True answers OK.
    """

    name = "chrome"
    label = "Chrome"

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        profile: str | None = None,
        version: str | None = None,
        gpu: bool = False,
        startup_timeout_s: float = STARTUP_S,
        call_timeout_s: float = CALL_S,
        local: LocalRule | None = None,
    ) -> None:
        self._command = list(command) if command else None
        self.local = local
        self._profile = profile
        self._version = version
        self._gpu = gpu
        self._startup_s = startup_timeout_s
        self._call_s = call_timeout_s
        self._browser: _Chrome | None = None
        self._mine: set[str] = set()
        self._sessions: dict[str, str] = {}  # target -> session
        self._targets: dict[str, str] = {}  # session -> target
        self._attaching = asyncio.Lock()
        self._current: str | None = None
        self._acting: set[str] = set()
        self._said: dict[str, list[Dialog]] = {}
        self.on_dialog: DialogAsk | None = None
        # The sessions of tabs the person has (take over): their boxes are the person's.
        self._held: set[str] = set()
        self._inflight = 0  # the person's input calls Chrome has not answered yet
        self._ring = f"melo-ring-{uuid.uuid4().hex[:12]}"

    # ── Lifecycle ──

    @property
    def profile(self) -> str:
        return self._profile or default_profile()

    async def start(self) -> None:
        if self._browser is not None and not self._browser.closed:
            return
        self._release()
        argv = self._command or chrome_command()
        if not argv:
            raise BrowserSetupError(MISSING_MESSAGE)
        profile = self.profile
        make_profile(profile)
        version = (
            self._version if self._version is not None else chrome_version(argv[0])
        )
        full = [*argv, *launch_args(profile, version, gpu=self._gpu)]
        self._browser = await _shared(full, profile, self._startup_s, self.local)

    async def close(self) -> None:
        browser = self._release()
        if browser is not None and browser.users <= 0:
            await browser.close()

    def kill(self) -> None:
        browser = self._release()
        if browser is not None and browser.users <= 0:
            browser.kill()

    def _release(self) -> _Chrome | None:
        """Let go of Chrome: this driver's tabs and sessions are forgotten; the Chrome it used, if any."""
        browser, self._browser = self._browser, None
        if browser is not None:
            browser.users -= 1
            browser.claimed -= self._mine
            for session in self._sessions.values():
                browser.end_cast(session)
                for kept in (browser.dialogs, browser.handoffs, browser.navs):
                    kept.pop(session, None)
            for target in self._mine:
                browser.attempts.pop(target, None)
        self._mine.clear()
        self._sessions.clear()
        self._targets.clear()
        self._held.clear()
        self._current = None
        return browser

    def _live(self) -> _Chrome:
        browser = self._browser
        if browser is None:
            raise BrowserError("Chrome is not running. Ask again: melo starts it.")
        if browser.closed:
            self._release()
            raise BrowserError(STOPPED_MESSAGE)
        return browser

    # ── Calls ──

    async def _cmd(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """One call to Chrome; a box the page puts up meanwhile is answered first.

        The time a person takes to answer a box is not counted against the call.
        """
        browser = self._live()
        if session and session in browser.dialogs and session not in self._held:
            await self._answer(browser, session)
        future = browser.request(method, params, session)
        loop = asyncio.get_running_loop()
        end = loop.time() + self._call_s
        while True:
            woken: asyncio.Future[Any] | None = None
            waiting: set[asyncio.Future[Any]] = {future}
            if session:
                event = browser.woken.setdefault(session, asyncio.Event())
                woken = asyncio.ensure_future(event.wait())
                waiting.add(woken)
            try:
                done, _ = await asyncio.wait(
                    waiting,
                    timeout=max(end - loop.time(), 0.0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if woken is not None and not woken.done():
                    woken.cancel()
            if future in done:
                return future.result()
            if woken is not None and woken in done and session:
                browser.woken[session].clear()
                if session in browser.dialogs and session not in self._held:
                    await self._answer(browser, session)
                    end = loop.time() + self._call_s
                elif browser.closed:
                    return await future  # raises: Chrome stopped
                continue
            future.cancel()
            raise BrowserError(f"Chrome did not answer {method} in {self._call_s:g} s.")

    async def _answer(self, browser: _Chrome, session: str) -> None:
        """Answer the box the page in *session* put up (see the module docstring)."""
        params = browser.dialogs.pop(session, None)
        if params is None:
            return
        kind = str(params.get("type") or "alert")
        message = " ".join(str(params.get("message") or "").split())[:300]
        accept = kind in ("alert", "beforeunload")
        if kind == "confirm" and session in self._acting and self.on_dialog is not None:
            try:
                accept = bool(
                    await self.on_dialog(kind, message, str(params.get("url") or ""))
                )
            except BrowserError:
                accept = False
        with contextlib.suppress(BrowserError, TimeoutError):
            async with asyncio.timeout(DIALOG_S):
                await browser.request(
                    "Page.handleJavaScriptDialog", {"accept": accept}, session
                )
        target = self._targets.get(session, "")
        self._said.setdefault(target, []).append(Dialog(kind, message, accept))

    async def _attach(self, target: str) -> str:
        """The session melo drives *target* in, attached (and set up) on first use."""
        session = self._sessions.get(target)
        if session:
            return session
        async with self._attaching:  # the live view and a tool call may ask together
            session = self._sessions.get(target)
            if session:
                return session
            got = await self._cmd(
                "Target.attachToTarget", {"targetId": target, "flatten": True}
            )
            session = str(got.get("sessionId") or "")
            if not session:
                raise BrowserError("Chrome did not let melo drive that tab.")
            self._sessions[target] = session
            self._targets[session] = target
            await self._cmd("Page.enable", {}, session)
            # A page in a window of its own behaves as if its window had the focus.
            await self._cmd(
                "Emulation.setFocusEmulationEnabled", {"enabled": True}, session
            )
            return session

    def _resolve(self, tab: str | None) -> str:
        target = self._current if not tab or tab == "current" else tab
        if not target:
            raise BrowserError("No tab is open. Use browser_open first.")
        if target not in self._mine:
            raise BrowserError("Chrome no longer has that tab.")
        return target

    async def _focus(self, tab: str | None) -> str:
        """Make *tab* the one the scripts run in; its session."""
        target = self._resolve(tab)
        self._current = target
        return await self._attach(target)

    async def _session(self) -> str:
        return await self._focus(None)

    @staticmethod
    def _value(got: dict[str, Any]) -> Any:
        details = got.get("exceptionDetails")
        if isinstance(details, dict):
            exception = details.get("exception")
            said = (
                exception.get("description")
                if isinstance(exception, dict)
                else details.get("text")
            )
            raise BrowserError(f"A page script failed in Chrome: {str(said)[:200]}")
        result = got.get("result")
        value = result.get("value") if isinstance(result, dict) else None
        if isinstance(value, str) and value[:1] in "{[":
            return decode(value)
        return value

    async def _eval(self, source: str) -> Any:
        """Run *source* in the page itself (the act scripts and the look after an act)."""
        session = await self._session()
        got = await self._cmd(
            "Runtime.evaluate",
            {"expression": source, "returnByValue": True, "awaitPromise": False},
            session,
        )
        return self._value(got)

    async def _read(self, source: str) -> Any:
        """Run *source* in melo's isolated world of the page: what the page's scripts changed is not seen."""
        session = await self._session()
        context = await self._world(session)
        got = await self._cmd(
            "Runtime.evaluate",
            {
                "expression": source,
                "returnByValue": True,
                "awaitPromise": False,
                "contextId": context,
            },
            session,
        )
        return self._value(got)

    async def _world(self, session: str) -> int:
        browser = self._live()
        frame = browser.main.get(session, "")
        for attempt in range(2):
            if not frame or attempt:
                tree = await self._cmd("Page.getFrameTree", {}, session)
                top = tree.get("frameTree")
                top = top.get("frame") if isinstance(top, dict) else None
                frame = str(top.get("id") or "") if isinstance(top, dict) else ""
                browser.main[session] = frame
            try:
                got = await self._cmd(
                    "Page.createIsolatedWorld",
                    {
                        "frameId": frame,
                        "worldName": WORLD,
                        "grantUniveralAccess": False,
                    },
                    session,
                )
            except CdpError:
                if attempt:
                    raise
                continue
            context = got.get("executionContextId")
            if isinstance(context, int):
                return context
        raise BrowserError("melo could not set up its own view of the page in Chrome.")

    async def _info(self, target: str) -> dict[str, Any]:
        got = await self._cmd("Target.getTargetInfo", {"targetId": target})
        info = got.get("targetInfo")
        return info if isinstance(info, dict) else {}

    async def _address(self, tab: str | None) -> str:
        """The address Chrome itself reports for *tab*, or ``""`` when it cannot say."""
        try:
            target = self._resolve(tab)
            return str((await self._info(target)).get("url") or "")
        except BrowserError:
            return ""

    async def _here(self, target: str) -> TabInfo:
        info = await self._info(target)
        return TabInfo(target, str(info.get("url") or ""), str(info.get("title") or ""))

    async def _go(
        self, session: str, start: Callable[[], Awaitable[dict[str, Any]]]
    ) -> None:
        """Start a navigation, wait until the tab shows the new document, then until it has loaded."""
        browser = self._live()
        before = browser.navs.get(session, 0)
        got = await start()
        error = str(got.get("errorText") or "")
        if error:
            raise BrowserError(f"Chrome could not load that page ({error[:120]}).")
        loop = asyncio.get_running_loop()
        end = loop.time() + SETTLE_S
        while browser.navs.get(session, 0) == before and loop.time() < end:
            await asyncio.sleep(0.05)
        await self._ready()

    async def _create(self) -> str:
        """A new tab in a window of its own, in the background; melo's."""
        browser = self._live()
        got = await self._cmd(
            "Target.createTarget",
            {
                "url": "about:blank",
                "newWindow": True,
                "background": True,
                "width": VIEW_W,
                "height": VIEW_H,
            },
        )
        target = str(got.get("targetId") or "")
        if not target:
            raise BrowserError("Chrome did not open a tab.")
        self._mine.add(target)
        browser.claimed.add(target)
        # The page Chrome started with is nobody's: it goes once melo has one of its own.
        for spare in list(browser.spare):
            browser.spare.discard(spare)
            with contextlib.suppress(BrowserError):
                await self._cmd("Target.closeTarget", {"targetId": spare})
        return target

    def _drop(self, target: str) -> None:
        self._mine.discard(target)
        session = self._sessions.pop(target, None)
        if session is not None:
            self._targets.pop(session, None)
        if self._browser is not None:
            if session is not None:
                self._browser.end_cast(session)
            self._browser.claimed.discard(target)
            self._browser.attempts.pop(target, None)
        if self._current == target:
            self._current = None

    # ── The driver ──

    async def open(self, url: str, *, new_tab: bool, tab: str | None = None) -> TabInfo:
        target = await self._create() if new_tab else self._resolve(tab)
        session = await self._focus(target)
        await self._go(
            session, lambda: self._cmd("Page.navigate", {"url": url}, session)
        )
        return await self._here(target)

    async def back(self, tab: str | None) -> TabInfo:
        target = self._resolve(tab)
        session = await self._focus(target)
        history = await self._cmd("Page.getNavigationHistory", {}, session)
        index, entries = history.get("currentIndex"), history.get("entries")
        if not isinstance(index, int) or not isinstance(entries, list) or index <= 0:
            raise BrowserError("There is no earlier page in this tab.")
        entry = entries[index - 1]
        if index == 1 and isinstance(entry, dict) and entry.get("url") == "about:blank":
            # The empty page a new tab starts on is no page to go back to.
            raise BrowserError("There is no earlier page in this tab.")
        entry_id = entry.get("id") if isinstance(entry, dict) else None
        await self._go(
            session,
            lambda: self._cmd(
                "Page.navigateToHistoryEntry", {"entryId": entry_id}, session
            ),
        )
        return await self._here(target)

    async def tabs(self) -> list[TabInfo]:
        browser = self._live()
        got = await self._cmd("Target.getTargets")
        pages = [
            t
            for t in got.get("targetInfos") or []
            if isinstance(t, dict) and t.get("type") == "page"
        ]
        alive = {str(t.get("targetId")) for t in pages}
        # A tab one of melo's tabs opened is melo's too, however deep.
        grew = True
        while grew:
            grew = False
            for item in pages:
                target = str(item.get("targetId"))
                opener = str(item.get("openerId") or "")
                if (
                    target not in self._mine
                    and target not in browser.claimed
                    and opener in self._mine
                ):
                    self._mine.add(target)
                    browser.claimed.add(target)
                    grew = True
        for gone in self._mine - alive:
            self._drop(gone)
        return [
            TabInfo(
                str(t.get("targetId")),
                str(t.get("url") or ""),
                str(t.get("title") or ""),
            )
            for t in pages
            if str(t.get("targetId")) in self._mine
        ]

    async def switch(self, tab: str) -> TabInfo:
        target = self._resolve(tab)
        if target not in {t.id for t in await self.tabs()}:
            raise BrowserError("Chrome no longer has that tab.")
        self._current = target
        return await self._here(target)

    def local_attempts(self) -> list[LocalAttempt]:
        """The requests to this computer melo blocked on this driver's pages since the last call, once."""
        browser = self._browser
        if browser is None:
            return []
        found: list[LocalAttempt] = []
        for target in [t for t in browser.attempts if t in self._mine]:
            found.extend(browser.attempts.pop(target))
        return found

    async def close_tab(self, tab: str) -> None:
        target = self._resolve(tab)
        await self._cmd("Target.closeTarget", {"targetId": target})
        self._drop(target)

    async def location(self, tab: str | None) -> str:
        return await self._address(tab)

    # ── The live view ──

    async def start_screencast(
        self,
        tab: str | None,
        on_frame: Callable[[Frame], None],
        on_end: Callable[[], None],
        *,
        fps: float = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
        on_dialog: Callable[[], None] | None = None,
    ) -> None:
        """Stream *tab*'s page as pictures to *on_frame*, at most about *fps* a second.

        Chrome sends a picture when the page changes (none for a page that does
        not), as a JPEG of *quality* the size of the view. *on_end* is called if
        the stream ends because Chrome stops, *on_dialog* when the page puts up a
        box or one closes (:meth:`dialog_of`). A tab that already streams has its
        functions and pace replaced, and is not started again. This does not make
        the tab the one melo's tools use, and does not answer a box the page put up.
        """
        target = self._resolve(tab)
        session = await self._attach(target)
        browser = self._live()
        cast = _Cast(on_frame, on_end, 1.0 / max(float(fps), 1.0), on_dialog)
        running = session in browser.casts
        browser.casts[session] = cast
        if running:
            return
        try:
            async with asyncio.timeout(STREAM_S):
                await browser.request(
                    "Page.startScreencast",
                    {
                        "format": "jpeg",
                        "quality": int(quality),
                        "maxWidth": VIEW_W,
                        "maxHeight": VIEW_H,
                        "everyNthFrame": 1,
                    },
                    session,
                )
        except TimeoutError:
            browser.end_cast(session)
            raise BrowserError("Chrome did not start showing that tab.") from None
        except BaseException:
            browser.end_cast(session)
            raise

    async def stop_screencast(self, tab: str | None) -> None:
        """Stop streaming *tab*; nothing happens for a tab that does not stream."""
        browser = self._browser
        if browser is None or browser.closed:
            return
        try:
            target = self._resolve(tab)
        except BrowserError:
            return
        session = self._sessions.get(target)
        if not session or session not in browser.casts:
            return
        browser.end_cast(session)
        with contextlib.suppress(BrowserError, TimeoutError):
            async with asyncio.timeout(STREAM_S):
                await browser.request("Page.stopScreencast", {}, session)

    # ── The person at the panel (take over) ──

    def hold_dialogs(self, tab: str, held: bool) -> None:
        """While the person has *tab*, the boxes its page puts up are theirs: melo's own calls leave them open."""
        try:
            target = self._resolve(tab)
        except BrowserError:
            return
        session = self._sessions.get(target)
        if not session:
            return
        if held:
            self._held.add(session)
        else:
            self._held.discard(session)

    def dialog_of(self, tab: str) -> dict[str, str] | None:
        """The box *tab*'s page shows now, for the person: its kind, its words and a prompt's default text; None when none."""
        browser = self._browser
        if browser is None or browser.closed:
            return None
        try:
            session = self._sessions.get(self._resolve(tab))
        except BrowserError:
            return None
        params = browser.dialogs.get(session or "")
        if params is None:
            return None
        return {
            "kind": str(params.get("type") or "alert")[:20],
            "message": " ".join(str(params.get("message") or "").split())[:300],
            "default": str(params.get("defaultPrompt") or "")[:300],
        }

    async def answer_dialog(self, tab: str, accept: bool, text: str = "") -> bool:
        """The person's answer to the box *tab*'s page shows (*text* for a prompt); whether there was one."""
        browser = self._live()
        session = self._sessions.get(self._resolve(tab))
        if not session or session not in browser.dialogs:
            return False
        params: dict[str, Any] = {"accept": bool(accept)}
        if accept and browser.dialogs[session].get("type") == "prompt":
            params["promptText"] = text
        browser.dialogs.pop(session, None)
        with contextlib.suppress(BrowserError, TimeoutError):
            async with asyncio.timeout(DIALOG_S):
                await browser.request("Page.handleJavaScriptDialog", params, session)
        return True

    def person_input(self, tab: str, method: str, params: dict[str, Any]) -> bool:
        """Send one input call of the person's to *tab*, in order and without waiting; whether it was sent.

        Only the input methods go this way, only to a tab that streams to a panel,
        and at most :data:`MAX_INFLIGHT` wait for Chrome at a time (a page held by a
        box answers none until it closes): more are dropped. Chrome's answer, an
        error too, is never read, so nothing of what was typed is ever echoed back.
        """
        if method not in PERSON_METHODS or self._inflight >= MAX_INFLIGHT:
            return False
        browser = self._browser
        if browser is None or browser.closed:
            return False
        try:
            session = self._sessions.get(self._resolve(tab))
        except BrowserError:
            return False
        if not session or session not in browser.casts:
            return False
        try:
            future = browser.request(method, params, session)
        except BrowserError:
            return False
        self._inflight += 1

        def done(found: asyncio.Future[dict[str, Any]]) -> None:
            self._inflight = max(0, self._inflight - 1)
            if not found.cancelled():
                found.exception()  # read, so it is never logged as never retrieved

        future.add_done_callback(done)
        return True

    async def holds(
        self, tab: str | None, until: str, text: str, *, origin: str = ""
    ) -> bool:
        await self._focus(tab)
        return await self._probe(wait_probe(until, text, origin))

    async def _probe(self, source: str) -> bool:
        """Whether the wait probe *source* is true now; a page between two documents does not answer yet."""
        try:
            return await super()._probe(source)
        except BrowserError:
            if self._browser is None or self._browser.closed:
                raise
            return False

    async def _capture(self, view: View | None) -> Shot:
        got = await self._cmd(
            "Page.captureScreenshot",
            {"format": "jpeg", "quality": SHOT_QUALITY},
            await self._session(),
        )
        data = str(got.get("data") or "")
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raw = b""
        if not raw.startswith(b"\xff\xd8") or len(raw) > MAX_SHOT_BYTES:
            raise BrowserError("Chrome's picture is not one melo can use.")
        return Shot("image/jpeg", data, view)

    async def blank(self, tab: str | None) -> None:
        """Leave the page in *tab* for an empty one, in its place in the history.

        ``location.replace`` swaps the page's own history entry, so Back never
        returns to the page that was refused. Only when that does not take,
        Chrome's own navigation is used, which adds an entry after the page.
        """
        session = await self._focus(tab)
        try:
            await self._read("(location.replace('about:blank'), true)")
            if await self._settled_blank(tab):
                return
        except BrowserError:
            if self._browser is None or self._browser.closed:
                raise
        await self._go(
            session,
            lambda: self._cmd("Page.navigate", {"url": "about:blank"}, session),
        )

    async def act(self, tab: str | None, act: Act) -> ActResult:
        """Do *act*: by Chrome's own input first, else by the page script.

        Chrome's own input is a trusted click at the middle of the element
        the page script checked and brought into view (or at the point of the
        picture), with the page script's guards up; typing clicks the box,
        selects what it holds and puts the text in. When another element
        covers the element, it is in a frame, or Chrome's input did not reach
        it, the page script acts instead and the result says why. A choice in
        a list is the page script's. After either way melo looks at the page
        until it is quiet; the result also carries an address of another app's
        kind the page asked for meanwhile, and the boxes it put up.
        """
        if len(act.text) > MAX_TYPED_CHARS:
            raise BrowserError(
                f"melo types at most {MAX_TYPED_CHARS} characters at a time."
            )
        session = await self._focus(tab)
        browser = self._live()
        target = self._targets[session]
        browser.handoffs.pop(session, None)
        self._said.pop(target, None)
        if act.element.frame.startswith("x"):
            raise BrowserError(STALE["reach"])
        self._acting.add(session)
        try:
            spec = act_spec(act)
            passed, failed = "", False
            result: ActResult | None = None
            if act.point is not None or (act.native and act.kind != "select"):
                done = await self._native(session, spec, act)
                if isinstance(done, ActResult):
                    result = done
                else:
                    passed, failed = done
            if result is None:
                result = await self._scripted(spec, act, passed, failed)
        finally:
            self._acting.discard(session)
        asked = browser.handoffs.pop(session, [])
        return dataclasses.replace(
            result,
            handoff=asked[0] if asked else "",
            dialogs=tuple(self._said.pop(target, [])),
        )

    async def _native(
        self, session: str, spec: dict[str, Any], act: Act
    ) -> ActResult | tuple[str, bool]:
        """The act by Chrome's own input; else why not, and whether that holds for the site.

        The element is scrolled to the middle of the view and a point of it
        that nothing covers is found (:meth:`_aim`). A real click that still
        lands on another element is stopped by the guards, so nothing is done;
        the element is then scrolled to once more and the click tried again.
        When that lands elsewhere too the error says what got it.
        """
        for attempt in range(2):
            aimed = await self._aim(spec, act)
            if isinstance(aimed, str):
                return aimed, False
            x, y = aimed
            key = new_key()
            armed = await self._script(act_script({**spec, "mode": "arm", "key": key}))
            try:
                if act.kind == "type" and act.point is None:
                    await self._type_at(session, x, y, act.text, act.enter)
                else:
                    await self._click_at(session, x, y)
            except BrowserError:
                if self._browser is None or self._browser.closed:
                    raise
                # Whether it acted before it failed is read from the page: an act
                # that shows there is kept, one that does not falls to the page script.
            data, quiet = await self._look(key, spec["doc"], act)
            blocked = [str(k) for k in data.get("blocked") or []]
            if "moved" in blocked:
                if attempt == 0 and act.point is None and nothing_reached(data):
                    continue  # the page moved under the click: scroll to it again
                raise cover_stop(data.get("movedBy"), act.element.kind)
            break
        if not reached(act, data, frame=armed.get("frame") is True):
            return PASSED_OVER["missed"], True
        self._check_typed(act, data)
        return ActResult(
            stopped(blocked),
            "point" if act.point is not None else "browser",
            "",
            observed(data, quiet),
        )

    async def _aim(self, spec: dict[str, Any], act: Act) -> tuple[float, float] | str:
        """Where Chrome's input goes for *act*: a point of the view, else why not.

        An element is scrolled to the middle of the view at once. Its middle
        is the place when nothing covers it; else the part of it with most
        room that nothing covers; else, after a moment, the same once more.
        An element still covered is left to the page script, and the reason
        names what covers it.
        """
        if act.point is not None:
            return act.point
        if act.element.frame:
            return PASSED_OVER["frame"]
        ready = await self._scrolled(spec)
        if ready.get("covered") is True:
            return covered_why(ready.get("by"))
        if not isinstance(ready.get("cx"), int | float) or not isinstance(
            ready.get("cy"), int | float
        ):
            return PASSED_OVER["name"]
        return as_number(ready["cx"]), as_number(ready["cy"])

    async def _mouse(self, session: str, kind: str, x: float, y: float) -> None:
        params: dict[str, Any] = {"type": kind, "x": x, "y": y}
        if kind != "mouseMoved":
            params.update(
                button="left", buttons=1 if kind == "mousePressed" else 0, clickCount=1
            )
        await self._cmd("Input.dispatchMouseEvent", params, session)

    async def _click_at(self, session: str, x: float, y: float) -> None:
        """A trusted click at (*x*, *y*) of the view (page pixels)."""
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self._mouse(session, kind, x, y)

    async def _key(self, session: str, kind: str, **params: Any) -> None:
        await self._cmd("Input.dispatchKeyEvent", {"type": kind, **params}, session)

    async def _type_at(
        self, session: str, x: float, y: float, text: str, enter: bool
    ) -> None:
        """Click the box at (*x*, *y*), replace what it holds with *text*, then press Enter if asked."""
        await self._click_at(session, x, y)
        select = {
            "key": "a",
            "code": "KeyA",
            "windowsVirtualKeyCode": 65,
            "modifiers": _META,
        }
        await self._key(session, "rawKeyDown", commands=["selectAll"], **select)
        await self._key(session, "keyUp", **select)
        if text:
            await self._cmd("Input.insertText", {"text": text}, session)
        else:
            erase = {
                "key": "Backspace",
                "code": "Backspace",
                "windowsVirtualKeyCode": 8,
            }
            await self._key(session, "rawKeyDown", **erase)
            await self._key(session, "keyUp", **erase)
        if enter:
            press = {
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
                "nativeVirtualKeyCode": 13,
            }
            await self._key(session, "keyDown", text="\r", unmodifiedText="\r", **press)
            await self._key(session, "keyUp", **press)
