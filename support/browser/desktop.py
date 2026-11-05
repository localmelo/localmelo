"""Windows another app opens after melo acts in the browser.

A page melo acts on can make another app put a window on the screen: a chat
app's sign-in window that asks "authorize?", say. The browser never shows that
window, so melo looks at the screen around its own acts:

1. With the desktop control helper (an MCP server that holds the Mac's
   Accessibility and Screen Recording grants for its own app, so melo needs
   neither; Accessibility alone gives text, buttons and presses by name,
   without a picture or a point), melo lists the windows before and after each
   act and while it waits (``list_windows``, the helper's cheapest call). A window of another
   app that appeared meanwhile gets a short id (``w1``) and is named in the
   step's result with its buttons. melo may read it (its accessibility text,
   its buttons by name and a picture of only that window) and press one of its
   buttons: through accessibility first, and, when that does not go through,
   at a point of the window's picture that the model names, with the point
   marked on the picture melo shows next.
2. Without the helper, or without its Accessibility grant, the window cannot
   be read. melo still notices that another app came to the front (``lsappinfo``, which needs
   no grant) and says which app it is, so the person can answer it there.

Only windows that appeared soon after one of melo's own acts are melo's to read
or press (and then for :data:`KEEP_S` seconds, while they stay open); melo never
lists the person's other windows to the model, never roams the desktop, and never
types into another app. Pressing a button asks its own question
(``desktop-act:<app>``) every time. Only the person answers it, in Ask and in
Auto: no model reviews a press.

A window may come a moment after the act (a chat app starts, then asks), so the
look at the screen does not end with the act. What melo knows of the screen is
two things that are kept apart:

* the *baseline*, the windows and the app in front at the last look. Every look
  of any browser step renews it, and it is only what a later look is compared
  with (:meth:`DesktopWatch.look`);
* the *admission deadline*, :data:`ADMIT_S` seconds after an act that can hand off
  to another app (a click, a point click, a send, an address another app opens;
  :meth:`DesktopWatch.admit`). Only while it is open is a new window reported,
  read for its buttons and given an id, or an app that came to the front named.
  Nothing renews it: reads, waits, screenshots and finds look at the screen but
  never admit anything after it, so a window the person opens on their own
  later is never taken for the act.

Admission is by time only, not by app: a third-party sign-in often shows its
window from another app or helper process than the one the page asked for. The
app the page asked for, when known, is a hint: its windows come first and say so.
A click that may hand off also waits up to :data:`LATE_S` seconds for a window.

A window is new only when its id was not listed at all before: the baseline
holds every window of every Space, on the screen or not (hidden apps,
minimized windows, never-shown ones), so a Space switch or an app the person
un-hides shows only windows that were listed, and is never taken for the act.
The helper lists layer 0 of every app in one call and the other layers only by
process, so each look also lists every layer of a few apps (:meth:`_listing`):
one that made a main-layer window since (a sign-in app often makes a hidden one
beside its panel), the app in front, apps that came to the front after an earlier
act, and apps with a window on the screen. A panel above the main layer is a
candidate whether or not its app came to the front, on any layer a window to
answer sits on (:func:`answerable`).

A model can take longer than the deadline to plan its next step, so while the
deadline is open the provider also samples the screen about every
:data:`SAMPLE_S` seconds in the background. A window first seen then is admitted
at that moment, with the same rules as any look, and told by the next browser
step even when that step comes after the deadline; the time it was first seen,
never the time it is told, is what counts.

Some apps draw their own buttons, so the window's accessibility tree names none.
melo then presses a point of the window's picture. A click in the background
does not reach every such window: when the window is still there a second later
and looks as it did, melo clicks once more in the foreground, which moves the
pointer and brings the app to the front for a moment. The person is asked once,
before the first click, and the question says so.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import plistlib
import re
import shlex
import shutil
import struct
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import urlsplit

from localmelo.melo.contracts.approvals import (
    DESKTOP_ACT,
    TOOL_ASK,
    WINDOW_READ,
    ToolAsk,
    denial_text,
)
from localmelo.melo.schema import ImagePart
from localmelo.support.browser import render
from localmelo.support.browser.driver import BrowserError
from localmelo.support.words import t_in

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The helper's command line, for tests and for a helper installed elsewhere.
HELPER_ENV = "LOCALMELO_DESKTOP_HELPER"
HELPER_PROGRAM = "cua-driver"
HELPER_PATHS = (
    "~/.local/bin/cua-driver",
    "/Applications/CuaDriver.app/Contents/MacOS/cua-driver",
)
# The picture of a window, as the benchmark's tap sends one: JPEG at quality 85,
# the long side at most 1280 pixels and the file at most 256 KB.
SHOT_LONG_SIDE = 1280
SHOT_QUALITY = 85
SHOT_MAX_BYTES = 256 * 1024
# A window admitted after an act stays melo's to read and press this long (while it is
# open). This keeps a window that was validly admitted; it never admits a new one.
KEEP_S = 600.0
# How long after an act that can hand off to another app a new window is told, read for
# its buttons and given an id. The act's own look and 3 s wait come first; then the next
# step: the retest store shows 4-6 s from one step's end to the next step's start (about
# 10 s with an approval card), so 15 s reaches the next step of a slow model, and no more
# than that of the person's own doing is ever taken for the act.
ADMIT_S = 15.0
# The most windows one act adopts, buttons one window names, and characters of its text.
MAX_WINDOWS = 3
# The most windows waiting for the person's say at one look.
MAX_PENDING = 3
MAX_BUTTONS = 12
TEXT_CHARS = 4000
# macOS window levels (CoreGraphics' CGWindowLevel), the helper's ``layer``. A window
# to answer sits from the normal level up to below the menu bar: documents (0),
# floating panels (3), modal panels (8), utility windows (19) and the levels a few
# apps give a panel that stays above every other window (21-23). The Dock (20), the
# menu bar (24) and its status items (25), pop-up menus (101), overlays (102),
# tooltips (200), the screen saver and the cursor never are.
NORMAL_LEVEL = 0
DOCK_LEVEL = 20
MAIN_MENU_LEVEL = 24
# The most apps that came to the front after an act, watched on every layer.
MAX_WATCHED = 3
# The most apps listed on every layer at one look (one helper call each, on top of
# the one call that lists the main layer of every app on every Space).
MAX_LAYERED = 6
# After a click at a point of a window, how long melo waits for the window to close
# before it compares the window with what it was, and how often it looks meanwhile.
POINT_SETTLE_S = 1.0
POINT_POLL_S = 0.25
# How the helper starts the error of a click whose point it could not mark on a picture: it
# takes that picture after the click, so the click itself went out.
MARK_FAILED = "debug_image_out write failed"
# After an act that may hand off to another app, the screen is looked at this often for
# up to this long in all (the wait's own helper and lsappinfo calls count), and the wait
# ends when a window appears or the tab moves to another address or another tab opens.
LATE_S = 3.0
LATE_POLL_S = 0.5
# While the admission deadline is open the screen is also sampled in the background this
# often (see the provider), so a window that comes while the model is still planning its
# next step is admitted when it is first seen, not when the next step looks. Each sample is
# cut at LATE_S like the late wait.
SAMPLE_S = 1.0
# How long one call to the helper or to lsappinfo may take.
CALL_S = 10.0
FRONT_S = 2.0
SIPS = "/usr/bin/sips"
# The system's own surfaces: never "another app's window".
SYSTEM_APPS = frozenset(
    {
        "Control Center",
        "Dock",
        "Notification Center",
        "NotificationCenter",
        "Spotlight",
        "SystemUIServer",
        "Wallpaper",
        "Window Server",
        "WindowManager",
        "loginwindow",
        "TextInputMenuAgent",
        "Screenshot",
    }
)
# The app that opens addresses of a scheme (``chat://...``), from LaunchServices through
# a script that only asks (it shows nothing and needs no grant); the scheme goes in as an
# argument, never the address.
SCHEME_S = 2.0
OSASCRIPT = "/usr/bin/osascript"
_HANDLER_SCRIPT = (
    'function run(argv){ObjC.import("AppKit");'
    'var u=$.NSURL.URLWithString(argv[0]+"://x");'
    "var a=$.NSWorkspace.sharedWorkspace.URLForApplicationToOpenURL(u);"
    'return a.isNil()?"":ObjC.unwrap(a.path);}'
)
_SCHEME_WORD = re.compile(r"[a-z][a-z0-9+.]{0,31}")
# Roles a window's control can be pressed by (accessibility roles without "AX").
_PRESSABLE = frozenset(
    {"button", "link", "checkbox", "radiobutton", "menubutton", "popupbutton"}
)
_TEXT_ROLES = frozenset({"statictext", "text", "heading", "textfield", "textarea"})
_SHORT = re.compile(r"w[1-9][0-9]{0,5}")


class DesktopError(BrowserError):
    """The helper could not do what melo asked; the message says what to do."""


@dataclass(frozen=True)
class AppWindow:
    """A window of another app, as the helper lists it."""

    id: int
    pid: int
    app: str
    title: str
    z: int | None = None
    layer: int = 0
    on_screen: bool = True


def answerable(layer: int) -> bool:
    """Whether a window on *layer* can be another app's window to answer (see :data:`MAIN_MENU_LEVEL`)."""
    return NORMAL_LEVEL <= layer < MAIN_MENU_LEVEL and layer != DOCK_LEVEL


@dataclass(frozen=True)
class Listing:
    """One look at the windows: every row listed (any Space, on the screen or not), the
    processes listed on every layer, and the Space on the screen (None: not said)."""

    rows: tuple[AppWindow, ...]
    layered: frozenset[int]
    space: int | None


@dataclass(frozen=True)
class Control:
    """A control of a window that melo may press, by its name."""

    name: str
    role: str
    token: str


@dataclass(frozen=True)
class Mark:
    """What melo knows of the screen: a baseline and an admission deadline.

    The baseline is ``windows``, every window id listed (the main layer of every
    app on every Space, on the screen or not, and every layer of the processes in
    ``layered``; None: unknown), ``front``, the app in front, ``top``, the highest
    window id listed, and ``space``, the Space on the screen. Only against a
    listing of a process on every layer is its panel known to be new; a panel of
    a process not listed so is new only when it is numbered after every window
    listed (the window server numbers windows in the order they are made) and its
    app also made a main-layer window since. Every look renews the baseline.

    The deadline is ``until``, on the watch's clock, and ``target``, the names of
    the app the page asked for (lower case), or none. It is set only by
    :meth:`DesktopWatch.admit`, and a look carries it over unchanged.
    """

    windows: frozenset[int] | None
    front: Front | None
    layered: frozenset[int] = frozenset()
    top: int = 0
    space: int | None = None
    until: float = 0.0
    target: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Front:
    """The app in front: its name, bundle id and process id (0 when unknown)."""

    name: str
    bundle: str = ""
    pid: int = 0

    def same_app(self, other: Front | None) -> bool:
        return other is not None and (self.name, self.bundle) == (
            other.name,
            other.bundle,
        )


@dataclass(frozen=True)
class Seen:
    """What one act left on the screen from another app.

    ``short`` is the window's id for ``window_read`` and ``window_press``, or
    ``""`` when melo can only say which app came to the front. ``target`` says
    the window is the app the page asked for.
    """

    short: str
    app: str
    title: str = ""
    buttons: tuple[str, ...] = ()
    target: bool = False


@dataclass
class _Kept:
    window: AppWindow
    since: float
    view: tuple[int, int] | None = None  # the size of its last picture, in pixels
    look: str = ""  # a digest of its last reading: picture, text and controls


@dataclass
class _State:
    """A window's accessibility reading: its text, its controls, its picture."""

    text: str
    controls: list[Control]
    image: tuple[str, str] | None  # (mime type, base64)
    size: tuple[int, int] | None
    extra: dict[str, Any] = field(default_factory=dict)


def helper_command() -> list[str] | None:
    """The command line that runs the helper's MCP server, or None when it is not installed.

    :data:`HELPER_ENV` first (a whole command line; an empty value means none),
    then the program on ``PATH``, then where its installer puts it.
    """
    if HELPER_ENV in os.environ:
        words = shlex.split(os.environ[HELPER_ENV])
        return words or None
    found = shutil.which(HELPER_PROGRAM)
    if found:
        return [found, "mcp"]
    for path in HELPER_PATHS:
        full = os.path.expanduser(path)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return [full, "mcp"]
    return None


async def _lsappinfo(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "/usr/bin/lsappinfo",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), FRONT_S)
    except TimeoutError:
        proc.kill()
        raise
    return out.decode("utf-8", "replace")


async def front_app() -> Front | None:
    """The app in front now; None when the system cannot say.

    From ``lsappinfo``, which needs no grant and shows nothing of any window:
    ``lsappinfo front`` names the app, ``lsappinfo info`` its name, bundle id and pid.
    """
    try:
        asn = (await _lsappinfo("front")).strip()
        if not asn.startswith("ASN:"):
            return None
        text = await _lsappinfo(
            "info", "-only", "name", "-only", "bundleid", "-only", "pid", asn
        )
    except (OSError, TimeoutError):
        return None
    return parse_front(text)


def parse_front(text: str) -> Front | None:
    """The app ``lsappinfo info -only name -only bundleid -only pid ASN`` describes."""
    name = re.match(r'\s*"([^"]*)"', text)
    bundle = re.search(r'bundleID="([^"]*)"', text)
    pid = re.search(r"\bpid = (\d+)", text)
    if name is None:
        return None
    return Front(
        name.group(1),
        bundle.group(1) if bundle else "",
        int(pid.group(1)) if pid else 0,
    )


async def scheme_apps(url: str) -> frozenset[str]:
    """The names, in lower case, that the app opening addresses like *url* goes by; none when that cannot be told.

    Only the scheme is looked up. The names are the app's file name and the
    name, display name, executable and bundle id of its Info.plist: a window's
    owner is listed by one of them.
    """
    try:
        scheme = urlsplit(url.strip()).scheme.lower()
    except ValueError:
        return frozenset()
    if not _SCHEME_WORD.fullmatch(scheme):
        return frozenset()
    try:
        proc = await asyncio.create_subprocess_exec(
            OSASCRIPT,
            "-l",
            "JavaScript",
            "-e",
            _HANDLER_SCRIPT,
            scheme,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), SCHEME_S)
        except TimeoutError:
            proc.kill()
            raise
    except (OSError, TimeoutError):
        return frozenset()
    return app_names(out.decode("utf-8", "replace").strip())


def app_names(path: str) -> frozenset[str]:
    """The names, in lower case, the app at *path* (``/Applications/Chat.app``) goes by."""
    if not path.endswith(".app"):
        return frozenset()
    names = {os.path.basename(path)[: -len(".app")]}
    try:
        with open(os.path.join(path, "Contents", "Info.plist"), "rb") as handle:
            info = plistlib.load(handle)
    except Exception:  # noqa: BLE001 - a hint: an unreadable plist leaves the file name
        info = {}
    for key in (
        "CFBundleName",
        "CFBundleDisplayName",
        "CFBundleExecutable",
        "CFBundleIdentifier",
    ):
        value = info.get(key) if isinstance(info, dict) else None
        if isinstance(value, str) and value:
            names.add(value)
    return frozenset(name.lower() for name in names if name)


async def bundle_of(pid: int) -> str:
    """The bundle id of the app with process *pid* (``""`` when it cannot be told); from ``lsappinfo``, no grant."""
    if pid <= 0:
        return ""
    try:
        found = re.search(
            r"ASN:0x[0-9a-f]+-0x[0-9a-f]+", await _lsappinfo("find", f"pid={pid}")
        )
        if found is None:
            return ""
        text = await _lsappinfo("info", "-only", "bundleid", found.group(0))
    except (OSError, TimeoutError):
        return ""
    bundle = re.search(r'bundleID="([^"]*)"', text)
    return bundle.group(1) if bundle else ""


# The helper's own identity, as its grants are held: the daemon it starts through
# LaunchServices (``cua-driver mcp`` is a proxy to it), not the app that starts the proxy.
HELPER_APP = "Cua Driver"
HELPER_BUNDLE = "com.trycua.driver"


def _grant_holder(source: Any) -> str:
    """The app whose Accessibility and Screen Recording grants the helper's answer reflects.

    ``check_permissions`` names it in ``source``: the helper's own daemon
    (``driver-daemon``, how ``cua-driver mcp`` runs) or, when the helper was
    started to run in its caller's process, that host.
    """
    if isinstance(source, dict):
        bundle = str(source.get("bundle_id") or "")
        if source.get("attribution") != "driver-daemon" and bundle != HELPER_BUNDLE:
            name = os.path.basename(str(source.get("executable") or "")) or bundle
            if name:
                return render.field_text(name, 60)
    return HELPER_APP


@dataclass(frozen=True)
class HelperStatus:
    """What is known of the desktop control helper: ``missing`` (not installed), ``installed``
    (its grants not checked yet), ``granted``, ``no_grant`` (Accessibility is off) or ``failed``
    (it did not start). ``app`` is the app to give the grants to; ``pictures`` whether it also
    holds Screen Recording (None: not known)."""

    state: str
    app: str = HELPER_APP
    pictures: bool | None = None
    problem: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "app": self.app,
            "pictures": self.pictures,
            "problem": self.problem,
        }


# The newest status any check of the helper found, with the command it ran and when.
_last: tuple[HelperStatus, tuple[str, ...], float] | None = None
# A check of the helper for the settings page is not repeated within this many seconds.
STATUS_TTL_S = 30.0


class DesktopHelper:
    """The desktop control helper over MCP: lists windows, reads one, presses in it.

    It starts on first use and checks the helper's own grants once
    (``check_permissions`` without a prompt): with Accessibility off it is not
    used, and :attr:`problem` says why; with Screen Recording off it reads text
    and buttons and presses by name, with no picture and no point
    (:attr:`pictures`). :attr:`phase` is None until the check ran, then
    ``ready``, ``no_accessibility`` or ``failed``; :attr:`app` is the app that
    holds (or needs) the grants.
    """

    def __init__(self, command: Sequence[str]) -> None:
        from localmelo.support.mcp.config import McpServerConfig

        program, *args = command
        self._command = tuple(command)
        self._config = McpServerConfig(
            name="desktop", command=program, args=list(args), startup_timeout_s=CALL_S
        )
        self._client: Any = None
        self._ready: bool | None = None
        self.problem = ""
        self.why = ""
        self.phase: str | None = None
        self.app = HELPER_APP
        # Whether it holds Screen Recording: without it melo reads a window's
        # text and buttons and presses by name, but has no picture to show or
        # to point at.
        self.pictures = False

    async def ready(self) -> bool:
        """Whether the helper runs and holds Accessibility (checked once, with Screen Recording)."""
        if self._ready is not None:
            return self._ready
        from localmelo.support.mcp.client import McpError, StdioMcpClient

        client = StdioMcpClient(self._config)
        try:
            await client.start()
            self._client = client
            grants = await self._call("check_permissions", {"prompt": False})
        except asyncio.CancelledError:
            await self._drop(client)
            raise
        except (McpError, DesktopError, OSError) as exc:
            await self._drop(client)
            self.fail(str(exc))
            return False
        self.app = _grant_holder(grants.get("source"))
        if grants.get("accessibility") is not True:
            self.problem = "the desktop control helper has no Accessibility grant"
            self.phase = "no_accessibility"
        else:
            self.phase = "ready"
        self.pictures = grants.get("screen_recording") is True
        self._ready = not self.problem
        if not self._ready:
            await self._drop(client)
        self._remember()
        return self._ready

    def fail(self, why: str) -> None:
        """The helper did not start (or did not answer): *why*, and it is not used."""
        self.why = render.field_text(why, 100)
        self.problem = f"the desktop control helper did not start ({self.why})"
        self.phase = "failed"
        self._ready = False
        self._remember()

    def status(self) -> HelperStatus:
        """What the check found (``installed`` before it ran)."""
        if self.phase == "ready":
            return HelperStatus("granted", self.app, self.pictures)
        if self.phase == "no_accessibility":
            return HelperStatus("no_grant", self.app, None, self.problem)
        if self.phase == "failed":
            return HelperStatus("failed", self.app, None, self.problem)
        return HelperStatus("installed", self.app)

    def _remember(self) -> None:
        global _last
        _last = (self.status(), self._command, time.monotonic())

    def notice(self) -> str:
        """The one line that says why melo cannot read other apps' windows ("" when it can)."""
        if self.phase == "no_accessibility":
            return t_in("en", "desktop.noAccessibility", app=self.app)
        if self.phase == "failed":
            return t_in("en", "desktop.failed", why=self.why)
        return ""

    def recheck(self) -> None:
        """Look at the helper's grants again at the next use, when the last look found it unusable.

        A chat that starts after the person turned the grant on then finds it.
        """
        if self._ready is False:
            self._ready = None
            self.phase = None
            self.problem = ""
            self.why = ""

    async def _drop(self, client: Any) -> None:
        self._client = None
        with contextlib.suppress(Exception):
            await client.close()

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The structured answer of one tool call; :class:`DesktopError` when it failed."""
        from localmelo.support.mcp.client import McpError

        if self._client is None:
            raise DesktopError("The desktop control helper is not running.")
        try:
            result = await asyncio.wait_for(
                self._client.call_tool(name, arguments), CALL_S
            )
        except (McpError, TimeoutError) as exc:
            raise DesktopError(f"The desktop control helper failed: {exc}") from None
        data = _structured(result)
        if result.get("isError") is True or data.get("refusal"):
            why = data.get("refusal") or _first_text(result)
            raise DesktopError(
                f"The desktop control helper refused: {render.field_text(str(why), 200)}"
            )
        data["_images"] = [
            (str(part.get("mimeType")), str(part.get("data")))
            for part in result.get("content") or []
            if isinstance(part, dict)
            and part.get("type") == "image"
            and isinstance(part.get("data"), str)
        ]
        return data

    async def windows(self, *, app: str = "", pid: int = 0) -> list[AppWindow]:
        """The windows of every Space, on the screen or not: layer 0 (of the apps *app* names); every layer of *pid*."""
        return (await self._list(app=app, pid=pid))[0]

    async def everywhere(self) -> tuple[list[AppWindow], int | None]:
        """Layer 0 of every app on every Space, on the screen or not, and the Space on the screen (None: not said)."""
        return await self._list()

    async def _list(
        self, *, app: str = "", pid: int = 0
    ) -> tuple[list[AppWindow], int | None]:
        args: dict[str, Any] = {}
        if app:
            args["app"] = app
        if pid:
            args["pid"] = pid
        data = await self._call("list_windows", args)
        found = []
        for row in data.get("windows") or []:
            if not isinstance(row, dict) or not isinstance(row.get("window_id"), int):
                continue
            z, layer = row.get("z_index"), row.get("layer")
            found.append(
                AppWindow(
                    row["window_id"],
                    int(row.get("pid") or 0),
                    str(row.get("app_name") or ""),
                    str(row.get("title") or ""),
                    z if isinstance(z, int) else None,
                    layer if isinstance(layer, int) else 0,
                    row.get("is_on_screen") is not False,
                )
            )
        space = data.get("current_space_id")
        return found, space if isinstance(space, int) else None

    async def state(self, window: AppWindow, *, picture: bool) -> _State:
        """The window's accessibility tree, and with *picture* its picture."""
        data = await self._call(
            "get_window_state",
            {
                "pid": window.pid,
                "window_id": window.id,
                "tree_format": "elements",
                "include_screenshot": picture and self.pictures,
                "max_image_dimension": SHOT_LONG_SIDE,
            },
        )
        lines: list[str] = []
        controls: list[Control] = []
        for row in data.get("elements") or []:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").lower().removeprefix("ax")
            words = " ".join(
                str(row.get(key) or "").strip()
                for key in ("label", "value")
                if str(row.get(key) or "").strip()
            )
            token = row.get("element_token")
            if role in _PRESSABLE and words and row.get("enabled") is not False:
                if isinstance(token, str) and token:
                    controls.append(Control(words, role, token))
            elif role in _TEXT_ROLES and words:
                lines.append(words)
        size = None
        width, height = data.get("screenshot_width"), data.get("screenshot_height")
        if isinstance(width, int) and isinstance(height, int) and width and height:
            size = (width, height)
        images = data.get("_images") or []
        return _State("\n".join(lines), controls, images[0] if images else None, size)

    async def press(self, token: str) -> dict[str, Any]:
        """Press the control *token* names through accessibility."""
        return await self._call("click", {"element_token": token})

    async def click_at(
        self,
        window: AppWindow,
        x: float,
        y: float,
        marked: str,
        *,
        foreground: bool = False,
    ) -> dict[str, Any]:
        """Click (*x*, *y*) of the window's last picture; the picture with the point goes to *marked*.

        In the background by default; *foreground* brings the window to the
        front for the click and moves the pointer, as some windows need.
        """
        args: dict[str, Any] = {
            "pid": window.pid,
            "window_id": window.id,
            "x": x,
            "y": y,
            "debug_image_out": marked,
        }
        if foreground:
            args["delivery_mode"] = "foreground"
        return await self._call("click", args)

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
        self._ready = None

    def kill(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.kill()


def _structured(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("structuredContent")
    if isinstance(data, dict):
        return dict(data)
    with contextlib.suppress(ValueError, TypeError):
        parsed = json.loads(_first_text(result))
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_text(result: dict[str, Any]) -> str:
    for part in result.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text") or "")
    return ""


def _pid_of(front: Front, rows: Sequence[AppWindow]) -> int:
    """The process of the app in *front* among *rows* (0 when none of its windows is listed)."""
    name = front.name.lower()
    return min((w.pid for w in rows if w.app.lower() == name), default=0)


def _look_of(state: _State) -> str:
    """A digest of a window's reading (its picture, text and controls), to tell whether it changed."""
    image = state.image[1] if state.image is not None else ""
    names = "\n".join(f"{c.role}:{c.name}" for c in state.controls)
    return hashlib.sha256(f"{image}\0{state.text}\0{names}".encode()).hexdigest()


FrontApp = Callable[[], Awaitable[Front | None]]
SchemeApps = Callable[[str], Awaitable[frozenset[str]]]
BundleOf = Callable[[int], Awaitable[str]]


class DesktopWatch:
    """What melo sees of other apps' windows around its own browser acts (see the module docstring).

    *helper* is None when the helper is not installed. *skip* are app names and
    bundle ids that are never another app: the browser melo drives and melo itself.
    """

    def __init__(
        self,
        helper: DesktopHelper | None,
        *,
        skip: Sequence[str] = (
            "Safari",
            "Safari Technology Preview",
            "com.apple.Safari",
            "com.apple.SafariTechnologyPreview",
            "LocalMelo",
            "org.localmelo.app",
            "CuaDriver",
            "Cua Driver",
            "com.trycua.driver",
        ),
        front: FrontApp | None = None,
        handler: SchemeApps | None = None,
        bundle: BundleOf | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        pace: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._helper = helper
        self._skip = frozenset(name.lower() for name in skip)
        # The module's front_app and scheme_apps, looked up at each call, unless given.
        self._front = front or (lambda: front_app())
        self._handler = handler or (lambda url: scheme_apps(url))
        self._bundle = bundle or (lambda pid: bundle_of(pid))
        self._clock = clock
        self._sleep = sleep
        # The pause between two background samples (a test paces them by hand).
        self._pace = pace
        self._kept: dict[str, _Kept] = {}
        self._n = 0
        # Windows of other apps that appeared inside the deadline but are not the app the
        # page asked for: only the person decides whether melo may see one, so nothing
        # about them is told to a model until then. The provider takes them (``take_pending``).
        self._pending: list[AppWindow] = []
        # The apps of those windows until the person lets melo see one of them: such an
        # app is never named, not even by the front-app line, while it is held.
        self._held: set[str] = set()
        # The apps that came to the front after an act, newest first: their
        # windows on every layer are listed at each later look.
        self._pids: list[int] = []
        # Whether the one line about a helper that cannot run was given in this chat.
        self._noticed = False

    @property
    def can_read(self) -> bool:
        """Whether melo has a helper to read and press with (it may still lack its grants)."""
        return self._helper is not None

    def _other(self, app: str, bundle: str = "") -> bool:
        return bool(app) and not (
            app.lower() in self._skip
            or bundle.lower() in self._skip
            or app in SYSTEM_APPS
        )

    async def _helper_ready(self) -> DesktopHelper | None:
        helper = self._helper
        if helper is None or not await helper.ready():
            return None
        return helper

    def notice(self) -> str:
        """The one line per chat that says the helper cannot run, or "" (it can, it is not installed, or it was said)."""
        helper = self._helper
        if helper is None or self._noticed:
            return ""
        line = helper.notice()
        self._noticed = bool(line)
        return line

    async def mark(self) -> Mark:
        """The screen now."""
        helper = await self._helper_ready()
        front = await self._safe_front()
        listing = None
        if helper is not None:
            listing = await self._listing(helper, None, front)
        return self._marked(listing, front)

    async def _listing(
        self, helper: DesktopHelper, mark: Mark | None, front: Front | None
    ) -> Listing | None:
        """One look at the windows (None when the helper could not list them).

        The main layer of every app on every Space, on the screen or not, in one
        call; then every layer of at most :data:`MAX_LAYERED` apps, one call each,
        sent together:
        an app that made a main-layer window since *mark* (wherever it is: a
        sign-in app often makes a hidden one beside its panel), the app in
        *front*, the apps watched since they came to the front, and the apps with
        a window on the screen, the topmost first.
        """
        try:
            main, space = await helper.everywhere()
        except DesktopError:
            return None
        pids: list[int] = []

        def add(pid: int) -> None:
            if pid > 0 and pid not in pids and len(pids) < MAX_LAYERED:
                pids.append(pid)

        others = [w for w in main if self._other(w.app)]
        if mark is not None and mark.windows is not None:
            for window in others:
                if window.id not in mark.windows:
                    add(window.pid)
        if front is not None and self._other(front.name, front.bundle):
            add(front.pid or _pid_of(front, main))
        for pid in self._pids:
            add(pid)
        for window in sorted(
            (w for w in others if w.on_screen),
            key=lambda w: -(w.z if w.z is not None else -1),
        ):
            add(window.pid)

        async def layers(pid: int) -> list[AppWindow] | None:
            try:
                return await helper.windows(pid=pid)
            except DesktopError:
                return None

        # Asked all at once: the helper's session answers each by its id, so a slow
        # listing of one app does not hold up the others.
        found = await asyncio.gather(*(layers(pid) for pid in pids))
        rows = {w.id: w for w in main}
        layered: set[int] = set()
        for pid, windows in zip(pids, found, strict=True):
            if windows is None:
                continue
            for window in windows:
                rows.setdefault(window.id, window)
            layered.add(pid)
        return Listing(tuple(rows.values()), frozenset(layered), space)

    def _marked(
        self,
        listing: Listing | None,
        front: Front | None,
        carry: Mark | None = None,
        unseen: frozenset[int] = frozenset(),
    ) -> Mark:
        """A baseline of *listing* and *front*; without a listing, *carry*'s windows stay.

        *unseen* are windows made since the act but not on the screen yet: they
        stay out of the baseline, so they are new when they show (while the
        deadline is open). The admission deadline is *carry*'s, unchanged: a look
        renews the baseline, never the deadline.
        """
        if listing is None:
            base = carry if carry is not None else Mark(None, None)
            windows, layered, top, space = (
                base.windows,
                base.layered,
                base.top,
                base.space,
            )
        else:
            ids = frozenset(w.id for w in listing.rows)
            windows = ids - unseen
            layered = listing.layered
            top = max([*ids, carry.top if carry is not None else 0])
            space = listing.space
        if carry is None:
            return Mark(windows, front, layered, top, space)
        return Mark(windows, front, layered, top, space, carry.until, carry.target)

    def admit(
        self, mark: Mark | None, target: frozenset[str] = frozenset()
    ) -> Mark | None:
        """*mark* with the admission deadline set :data:`ADMIT_S` seconds from now, for an act that can hand off to another app.

        *target* are the names of the app the page asked for, when known (see
        :meth:`scheme_app`): a hint for the order and the words, never a filter.
        Only an act calls this; no look, wait or read does. Without a mark there
        is nothing to compare with yet, so nothing is admitted and the next look
        only takes the baseline.
        """
        if mark is None:
            return None
        return dataclasses.replace(
            mark, until=self._clock() + ADMIT_S, target=frozenset(target)
        )

    def admitting(self, mark: Mark | None) -> bool:
        """Whether the admission deadline of *mark* is still open."""
        return mark is not None and self._clock() <= mark.until

    async def pace(self, seconds: float) -> None:
        """Pause *seconds* between two background samples."""
        await self._pace(seconds)

    async def scheme_app(self, url: str) -> frozenset[str]:
        """The names of the app that opens addresses like *url*, or none (see :func:`scheme_apps`)."""
        try:
            return await self._handler(url)
        except Exception:  # noqa: BLE001 - a hint, never a failure
            return frozenset()

    async def _safe_front(self) -> Front | None:
        try:
            return await self._front()
        except Exception:  # noqa: BLE001 - the front app is a hint, never a failure
            return None

    async def look(
        self,
        mark: Mark | None,
        *,
        wait_s: float = 0.0,
        stop: Callable[[], Awaitable[bool]] | None = None,
        limit_s: float | None = None,
    ) -> tuple[list[Seen], Mark | None]:
        """What other apps put on the screen since *mark*, and the mark to keep from now on.

        Both come from one listing, so a window is told once and none is lost
        between two looks. The baseline is renewed whatever is told; something is
        told only while the mark's admission deadline is open (see :meth:`admit`).
        Without a mark the screen is only marked.

        With *wait_s* the screen is looked at every :data:`LATE_POLL_S` seconds
        until something appeared, *wait_s* seconds have passed or *stop* answers
        true (the tab moved to another address or another tab opened, so the act
        did not hand off). The seconds are counted from the start, and every
        call inside, the helper's and lsappinfo's too, is cut at their end: a
        look that did not finish tells nothing and leaves the mark as it was.
        A single look with no waiting is cut after *limit_s* seconds when that
        is given (a background sample).
        """
        cap = wait_s if wait_s > 0 else limit_s
        end = self._clock() + cap if cap and cap > 0 else None
        nothing: tuple[list[Seen], Mark | None] = ([], mark)
        seen, fresh = await self._timed(end, lambda: self._look_once(mark), nothing)
        if wait_s <= 0 or end is None or seen:
            return seen, fresh
        while self._clock() < end:
            if stop is not None and await self._timed(end, stop, False):
                break
            left = end - self._clock()
            if left <= 0:
                break
            await self._sleep(min(LATE_POLL_S, left))
            seen, fresh = await self._timed(
                end,
                functools.partial(self._look_once, fresh),
                (seen, fresh),
            )
            if seen:
                break
        return seen, fresh

    async def _timed(
        self, end: float | None, work: Callable[[], Awaitable[_T]], late: _T
    ) -> _T:
        """The result of *work()*, or *late* when it did not finish before the clock reached *end* (None: no limit)."""
        if end is None:
            return await work()
        left = end - self._clock()
        if left <= 0:
            return late
        try:
            return await asyncio.wait_for(work(), left)
        except TimeoutError:
            return late

    async def _look_once(self, mark: Mark | None) -> tuple[list[Seen], Mark]:
        if mark is None:
            return [], await self.mark()
        return await self._compare(mark)

    def _asked(self, mark: Mark, *names: str) -> bool:
        """Whether one of *names* is the app the page asked for (see :meth:`admit`)."""
        return any(name.lower() in mark.target for name in names if name)

    def _alone(self, mark: Mark, window: AppWindow) -> bool:
        """Whether *window* is read on its own (the app the page asked for); any other waits for the person."""
        return self._asked(mark, window.app)

    async def _compare(self, mark: Mark) -> tuple[list[Seen], Mark]:
        """The windows of other apps that appeared since *mark*, each adopted (at most :data:`MAX_WINDOWS`), and the mark of the listing that showed them.

        The listing renews the baseline whatever the deadline; windows are told,
        read for their buttons and adopted only while it is open. Nothing is
        adopted until every call is done, so a look cut short adopts nothing.
        """
        helper = await self._helper_ready()
        front = await self._safe_front()
        listing = None
        if helper is not None:
            listing = await self._listing(helper, mark, front)
        # A window is a candidate only inside the deadline, which only melo's own act
        # that can hand off sets: reads, waits and screenshots never open or extend it.
        # After it, a look renews the baseline and nothing else: no report, no state
        # read, no id, no name. Inside it the deadline is no proof that the act caused
        # the window, so only the app the page asked for (the scheme's handler, known
        # when the address named one) is read on its own. A third-party sign-in often
        # asks from another app or helper process (an SSO agent, a password manager),
        # so any other window is not dropped: it waits for the person, who decides
        # whether melo sees it (``take_pending``).
        if not self.admitting(mark):
            return [], self._marked(listing, front, mark)
        moved = (
            front is not None
            and not front.same_app(mark.front)
            and self._other(front.name, front.bundle)
        )
        new, unseen = self._new(listing, mark)
        if moved and front is not None and listing is not None:
            # Watched from now on: its panels on every layer are known at later looks.
            self._follow(front.pid or _pid_of(front, listing.rows))
        # The app the page asked for first, then the topmost window.
        new.sort(
            key=lambda w: (
                not self._asked(mark, w.app),
                -(w.z if w.z is not None else -1),
            )
        )
        # Admission is by time, but only the app the page asked for is read on its own:
        # a window of any other app waits for the person (``take_pending``), and
        # nothing about it, not even its app, reaches a model before they allow it.
        asked = [w for w in new if self._alone(mark, w)]
        waiting = [w for w in new if not self._alone(mark, w)]
        read: list[tuple[AppWindow, tuple[str, ...]]] = []
        for window in asked[:MAX_WINDOWS]:
            names: tuple[str, ...] = ()
            if helper is not None:
                with contextlib.suppress(DesktopError):
                    state = await helper.state(window, picture=False)
                    names = tuple(c.name for c in state.controls[:MAX_BUTTONS])
            read.append((window, names))
        fresh = self._marked(listing, front, mark, unseen)
        seen = [
            Seen(
                self._adopt(window),
                window.app,
                window.title,
                names,
                self._asked(mark, window.app),
            )
            for window, names in read
        ]
        self._pending.extend(waiting[:MAX_PENDING])
        self._held.update(w.app.lower() for w in waiting)
        # An app that came to the front with a window waiting for the person is not named
        # either, nor one whose window is held for the person or was not allowed; one with
        # no window melo can place is named, by name only. When the Space on the screen
        # changed meanwhile, the app in front is the one of that Space, not the act's doing,
        # unless it is the app the page asked for.
        switched = (
            listing is not None
            and None not in (listing.space, mark.space)
            and listing.space != mark.space
        )
        if (
            not seen
            and not waiting
            and moved
            and front is not None
            and front.name.lower() not in self._held
            and not (switched and not self._asked(mark, front.name, front.bundle))
        ):
            seen = [Seen("", front.name)]
        return seen, fresh

    def _new(
        self, listing: Listing | None, mark: Mark
    ) -> tuple[list[AppWindow], frozenset[int]]:
        """The windows of other apps in *listing* that are new since *mark* and on the screen, and those new but not shown yet.

        A window is new only when it was not listed at all before: on another
        Space, hidden or off the screen counts as listed, so a Space switch or an
        app un-hidden shows nothing new. A window above the main layer is new
        against a listing of its process on every layer; without one, only when
        its app also made a main-layer window since and the window server
        numbered it after every window listed before. Only a window on a layer a
        window to answer sits on (:func:`answerable`) is a candidate.
        """
        if listing is None or mark.windows is None:
            return [], frozenset()
        known, adopted = mark.windows, self._adopted()
        made = {
            w.pid
            for w in listing.rows
            if w.layer == 0 and w.id not in known and self._other(w.app)
        }
        new: list[AppWindow] = []
        unseen: set[int] = set()
        for window in listing.rows:
            if window.id in known or window.id in adopted:
                continue
            if not self._other(window.app):
                continue
            if not (
                window.layer == 0
                or window.pid in mark.layered
                or (window.pid in made and window.id > mark.top)
            ):
                continue  # not known to be new: from now on it is part of the baseline
            if not window.on_screen:
                unseen.add(window.id)
            elif answerable(window.layer):
                new.append(window)
        return new, frozenset(unseen)

    def _follow(self, pid: int) -> None:
        """Watch the app with process *pid* on every layer from now on (it came to the front)."""
        if pid <= 0:
            return
        if pid in self._pids:
            self._pids.remove(pid)
        self._pids = [pid, *self._pids][:MAX_WATCHED]

    def take_pending(self) -> list[AppWindow]:
        """The windows waiting for the person to allow melo to see them, once."""
        pending, self._pending = self._pending, []
        return pending

    async def adopt_seen(self, window: AppWindow) -> Seen | None:
        """*window*, which the person let melo see: adopted and read for its buttons; None when it is gone."""
        helper = await self._helper_ready()
        if helper is None or window.id in self._adopted():
            return None
        try:
            listed = await helper.windows(pid=window.pid)
            if window.id not in {w.id for w in listed}:
                return None
            state = await helper.state(window, picture=False)
        except DesktopError:
            return None
        names = tuple(c.name for c in state.controls[:MAX_BUTTONS])
        self._held.discard(window.app.lower())
        return Seen(self._adopt(window), window.app, window.title, names)

    async def window_allowed(self, ask: ToolAsk | None, window: AppWindow) -> bool:
        """Whether the person lets melo see *window* of another app: the one check for "is this app's window allowed for this chat".

        It asks the question ``window-read:<bundle id>`` (the app's name when
        its bundle id cannot be told) through the run's approval policy, which
        raises the card (the person only; the card shows the window's title) or,
        when the person answered "Allow <App> in this chat" before, allows it
        without one. Without a way to ask, never.
        """
        if ask is None:
            return False
        app = render.field_text(window.app, 60)
        title = render.field_text(window.title, 80)
        bundle = await self._bundle(window.pid)
        key = WINDOW_READ + (bundle or window.app)
        arguments = {
            "app": app,
            "window": title,
            "bundle": bundle,
            "access": "window-read",
        }
        shown = f' "{title}"' if title else ""
        decision = await ask(
            key,
            arguments,
            f"{app} opened a window{shown} right after melo's click: let melo see it?",
        )
        return decision.allow

    def _adopted(self) -> set[int]:
        return {kept.window.id for kept in self._kept.values()}

    def _adopt(self, window: AppWindow) -> str:
        self._n += 1
        short = f"w{self._n}"
        self._kept[short] = _Kept(window, self._clock())
        return short

    def forget(self) -> None:
        """Nothing that appeared before is melo's any more, and a helper that could not run is tried again."""
        self._kept.clear()
        self._pids.clear()
        self._held.clear()
        self._noticed = False
        if self._helper is not None:
            self._helper.recheck()

    def _window(self, short: str | None) -> tuple[str, _Kept]:
        """The kept window *short* names (default the newest); :class:`DesktopError` otherwise."""
        now = self._clock()
        for name in [n for n, k in self._kept.items() if now - k.since > KEEP_S]:
            del self._kept[name]
        word = str(short or "").strip().lower()
        if not word:
            if not self._kept:
                raise DesktopError(
                    "No window of another app appeared after melo's own acts. melo "
                    "reads only such windows; it does not look around the desktop."
                )
            word = max(self._kept, key=lambda n: int(n[1:]))
        if not _SHORT.fullmatch(word) or word not in self._kept:
            raise DesktopError(
                f'"{render.field_text(word, 20)}" is not a window that appeared after '
                "melo's own acts (w1, w2 ... from the step's result)."
            )
        return word, self._kept[word]

    async def _ready_for(self, short: str | None) -> tuple[DesktopHelper, str, _Kept]:
        helper = self._helper
        if helper is None:
            raise DesktopError(
                "melo cannot read other apps' windows: the desktop control helper is "
                "not installed. Ask the person to answer that window themselves."
            )
        if not await helper.ready():
            raise DesktopError(
                f"melo cannot read other apps' windows: {helper.problem}. Ask the "
                "person to answer that window themselves."
            )
        name, kept = self._window(short)
        listed = await helper.windows(pid=kept.window.pid)
        if kept.window.id not in {w.id for w in listed}:
            del self._kept[name]
            raise DesktopError(f"Window {name} of {kept.window.app} is closed now.")
        return helper, name, kept

    async def read(self, short: str | None) -> tuple[str, list[ImagePart]]:
        """The text, buttons and picture of a window that appeared (``window_read``)."""
        helper, name, kept = await self._ready_for(short)
        state = await helper.state(kept.window, picture=True)
        kept.view = state.size
        kept.look = _look_of(state)
        images: list[ImagePart] = []
        note = ""
        if state.image is not None:
            fitted = await fit_picture(*state.image)
            if fitted is None:
                note = "[No picture: it did not fit melo's picture limits.]"
            else:
                images.append(fitted)
        elif not helper.pictures:
            note = (
                "[No picture: the desktop control helper has no Screen Recording "
                "permission, so melo reads only the window's text and buttons, and "
                "presses by name only.]"
            )
        else:
            note = "[No picture: the helper sent none.]"
        return (
            _block(
                name,
                kept.window,
                state.text,
                state.controls,
                note,
                pointable=state.image is not None,
            ),
            images,
        )

    async def press(self, short: str | None, button: str) -> str:
        """Press the control named *button* in a window that appeared, through accessibility."""
        helper, name, kept = await self._ready_for(short)
        state = await helper.state(kept.window, picture=False)
        control = _named(state.controls, button)
        await ask_press(kept.window, f'"{control.name}"', control.name)
        try:
            done = await helper.press(control.token)
        except DesktopError as exc:
            raise DesktopError(
                f"{exc} The accessibility press did not go through. "
                + _point_way(helper, name, kept.window.app)
            ) from None
        effect = str(done.get("effect") or "")
        if effect in ("refused", "suspected_noop", "unverifiable"):
            words = {
                "refused": "the window refused it",
                "suspected_noop": "nothing seemed to happen",
                "unverifiable": "the helper cannot tell whether it worked",
            }[effect]
            line = (
                f'Pressed "{control.name}" in {kept.window.app} (via accessibility), '
                f"but {words}. " + _point_way(helper, name, kept.window.app)
            )
        else:
            line = f'Pressed "{control.name}" in {kept.window.app} (via accessibility).'
        return render.field_text(line, 400) + await self._after(helper, name, kept)

    async def press_point(
        self, short: str | None, x: float, y: float
    ) -> tuple[str, list[ImagePart]]:
        """Click (*x*, *y*), 0-1000 across and down the window's last picture; the point is marked.

        The person is asked once, before the first click, and the question says
        that a click that does not reach the window is repeated in the foreground.
        The click goes in the background. When the window is still there
        :data:`POINT_SETTLE_S` seconds later and reads as it did, the click did
        not reach it (a window that draws its own buttons may take clicks only in
        front): melo clicks once more in the foreground, the pointer moving and
        the app coming to the front, and gives that click the same time before it
        says what the window did. Each click uses up the picture it was read on,
        so another point needs ``window_read`` first.
        """
        helper, name, kept = await self._ready_for(short)
        app = kept.window.app
        if not helper.pictures:
            raise DesktopError(
                "melo cannot click a point of this window: the desktop control helper "
                "has no Screen Recording permission, so there is no picture to point "
                f"at. Ask the person to press it in {app}."
            )
        if kept.view is None:
            raise DesktopError(
                f"Use window_read {name} first: a point is read on the window's last picture."
            )
        what = f"a point ({x:g}, {y:g}) of its picture"
        await ask_press(kept.window, what, "", foreground_retry=True)
        images = await self._point(helper, kept, x, y, foreground=False)
        then = ""
        after = await self._settled(helper, kept)
        if after == "same":
            images = await self._point(helper, kept, x, y, foreground=True)
            then = (
                " The click in the background changed nothing, so melo clicked "
                "there again in the foreground: the pointer moved and "
                f"{app} came to the front for a moment."
            )
            after = await self._settled(helper, kept)
            if after == "same":
                then += (
                    " The window looks as it did a moment after that click too; "
                    f"it may still be closing: window_read {name} shows it."
                )
        if after == "changed":
            then += f" The window changed: window_read {name} shows it now."
        line = f"Clicked ({x:g}, {y:g}) on the picture of {app}'s window (via point)."
        if images:
            line += " The picture shows where."
        line += then
        kept.view = None  # the next point is read on a picture taken after this click
        return (
            render.field_text(line, 600) + await self._after(helper, name, kept),
            images,
        )

    async def _point(
        self,
        helper: DesktopHelper,
        kept: _Kept,
        x: float,
        y: float,
        *,
        foreground: bool,
    ) -> list[ImagePart]:
        """Click (*x*, *y*), 0-1000 on the window's last picture; the picture with the point marked, if any.

        A click that closed the window before its picture was taken was delivered:
        the helper's error is then only about the picture (:data:`MARK_FAILED`),
        and the result has no picture. Any other helper error is raised.
        """
        assert kept.view is not None
        width, height = kept.view
        px, py = round(x / 1000 * width, 1), round(y / 1000 * height, 1)
        folder = tempfile.mkdtemp(prefix="melo-desktop-")
        marked = os.path.join(folder, "point.png")
        picture: str | None = None
        try:
            await helper.click_at(kept.window, px, py, marked, foreground=foreground)
            picture = _read_file(marked)
        except DesktopError as exc:
            # Only the picture of the point failed, and the window it was taken of is
            # gone: the click closed it. No picture, and _settled says "closed".
            if not await self._clicked_then_gone(helper, kept, exc):
                raise
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        fitted = await fit_picture("image/png", picture) if picture else None
        return [fitted] if fitted is not None else []

    async def _settled(self, helper: DesktopHelper, kept: _Kept) -> str:
        """What a click at a point left: ``closed`` (the window went away or off the screen), ``changed``, ``same`` (still there and reading as before) or ``""`` (cannot tell).

        The window is looked for every :data:`POINT_POLL_S` seconds up to
        :data:`POINT_SETTLE_S`; still there, it is read again with its picture,
        which is also the reading the next click's point is placed on, in the
        same helper session.
        """
        waited = 0.0
        while True:
            try:
                found = await self._listed(helper, kept)
            except DesktopError:
                return ""
            if found is None or not found.on_screen:
                return "closed"
            if waited >= POINT_SETTLE_S:
                break
            step = min(POINT_POLL_S, POINT_SETTLE_S - waited)
            await self._sleep(step)
            waited += step
        try:
            state = await helper.state(kept.window, picture=True)
        except DesktopError:
            return ""
        if state.size is not None:
            kept.view = state.size
        return "same" if _look_of(state) == kept.look else "changed"

    async def _listed(self, helper: DesktopHelper, kept: _Kept) -> AppWindow | None:
        """The window as the helper lists it now, or None when it is not listed (gone)."""
        listed = await helper.windows(pid=kept.window.pid)
        return next((w for w in listed if w.id == kept.window.id), None)

    async def _clicked_then_gone(
        self, helper: DesktopHelper, kept: _Kept, error: DesktopError
    ) -> bool:
        """Whether *error* came after a click went out: only its picture failed, of a window that is gone now.

        An error before the click was delivered is none of this, and stays an error.
        """
        if MARK_FAILED not in str(error):
            return False
        try:
            found = await self._listed(helper, kept)
        except DesktopError:
            return False
        return found is None or not found.on_screen

    async def _after(self, helper: DesktopHelper, name: str, kept: _Kept) -> str:
        with contextlib.suppress(DesktopError):
            found = await self._listed(helper, kept)
            if found is None:
                del self._kept[name]
                return f" Window {name} closed."
            if not found.on_screen:
                return f" Window {name} is no longer on the screen."
            return f" Window {name} is still open."
        return ""

    async def close(self) -> None:
        if self._helper is not None:
            await self._helper.close()

    def kill(self) -> None:
        if self._helper is not None:
            self._helper.kill()


def _point_way(helper: DesktopHelper, name: str, app: str) -> str:
    """What the model can do when a press by name did not go through."""
    if helper.pictures:
        return (
            f"window_read {name} shows the window now; window_press with x and y "
            "(0-1000 on its picture) clicks the button's place."
        )
    return (
        "With no Screen Recording permission melo has no picture to click a point "
        f"of: ask the person to press it in {app}."
    )


def _named(controls: list[Control], button: str) -> Control:
    """The one control called *button* (its whole name, else a part of it)."""
    wanted = " ".join(str(button).split()).lower()
    if not wanted:
        raise DesktopError("window_press needs a button's name, or x and y.")
    exact = [c for c in controls if " ".join(c.name.split()).lower() == wanted]
    found = exact or [c for c in controls if wanted in c.name.lower()]
    if len(found) == 1:
        return found[0]
    names = ", ".join(
        f'"{render.field_text(c.name, 40)}"' for c in controls[:MAX_BUTTONS]
    )
    if not found:
        raise DesktopError(
            f'No button "{render.field_text(button, 40)}" in that window. Its buttons: '
            f"{names or 'none'}."
        )
    raise DesktopError(
        f'Several buttons are called "{render.field_text(button, 40)}": {names}. '
        "Use the whole name."
    )


def _block(
    name: str,
    window: AppWindow,
    text: str,
    controls: list[Control],
    note: str,
    *,
    pointable: bool = False,
) -> str:
    """The window between page markers: its words are another app's, data and never orders.

    A window that names no buttons draws its own: with a picture, the model is
    told to press a point of it.
    """
    lines = [render.clean(text)[:TEXT_CHARS] or "(no readable text)"]
    if controls:
        lines.append("buttons:")
        lines.extend(
            f'[{c.role} "{render.field_text(c.name, render.LABEL_CHARS)}"]'
            for c in controls[:MAX_BUTTONS]
        )
        lines.append(
            "[window_press presses a button by its name. melo does not type into "
            "other apps.]"
        )
    elif pointable:
        lines.append(
            "[This window names no buttons: it draws its own. window_press with x "
            "and y, 0-1000 across and down this picture, presses a point of it. "
            "melo does not type into other apps.]"
        )
    else:
        lines.append(
            "[This window names no buttons, and melo has no picture of it to press "
            f"a point of: ask the person to answer it in "
            f"{render.field_text(window.app, 60)}. melo does not type into other apps.]"
        )
    if note:
        lines.append(note)
    attrs = (
        f'app="{render.field_text(window.app, 60)}" '
        f'title="{render.field_text(window.title, render.TITLE_CHARS)}"'
    )
    return render.block(name, attrs, "\n".join(lines))


def seen_lines(seen: Sequence[Seen]) -> list[str]:
    """What an act left from other apps, as "what changed" lines (cleaned where they are shown)."""
    lines = []
    for item in seen:
        if not item.short:
            lines.append(
                f"{item.app} came to the front: it may show a window melo cannot "
                f"see. Tell the person to answer it in {item.app}."
            )
            continue
        title = f' "{item.title}"' if item.title else ""
        who = f"{item.app} (the app the page asked for)" if item.target else item.app
        if not item.buttons:
            lines.append(
                f"{who} opened a window{title} ({item.short}); it names no buttons. "
                f"window_read {item.short} shows its picture; window_press with x "
                "and y presses a point of it."
            )
            continue
        buttons = ", ".join(f'"{name}"' for name in item.buttons)
        lines.append(
            f"{who} opened a window{title} ({item.short}); its buttons: "
            f"{buttons}. window_read {item.short} shows it; window_press presses a button."
        )
    return lines


async def ask_press(
    window: AppWindow, what: str, button: str, *, foreground_retry: bool = False
) -> None:
    """Ask about pressing *what* in *window*: :class:`DesktopError` if refused.

    Asked every time (``desktop-act:`` answers are never kept for the chat).
    Only the person answers it, in Ask and in Auto alike (no model is shown the
    window or the press; with nobody to ask, Auto refuses). A click at a point
    (*foreground_retry*, ``delivery`` ``foreground_retry``) is asked once for
    both tries: the question says that when a click in the background does not
    reach the window, melo clicks once more in the foreground, where the pointer
    moves and the app comes to the front.
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    app = render.field_text(window.app, 60)
    title = render.field_text(window.title, 80)
    arguments: dict[str, Any] = {
        "app": app,
        "access": "desktop-act",
        "how": "press",
        "on": what,
        "window": title,
    }
    if button:
        arguments["button"] = render.field_text(button, 80)
    shown = f' "{title}"' if title else ""
    preview = f"press {what} in {app}'s window{shown}"
    if foreground_retry:
        arguments["delivery"] = "foreground_retry"
        preview += (
            "; if a click in the background does not reach it, melo clicks once "
            f"more in the foreground: the pointer moves there and {app} comes to "
            "the front for a moment"
        )
    decision = await ask(DESKTOP_ACT + app, arguments, preview)
    if not decision.allow:
        raise DesktopError(f"melo did not press it in {app}. " + denial_text(decision))


def _read_file(path: str) -> str | None:
    """The base64 of the file at *path* (never a link it points to), or None."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        data = handle.read(SHOT_MAX_BYTES * 64)
    return base64.b64encode(data).decode("ascii") if data else None


def _png_size(raw: bytes) -> tuple[int, int] | None:
    if raw[:8] != b"\x89PNG\r\n\x1a\n" or len(raw) < 24:
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


async def _sips(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        SIPS,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), CALL_S)
    except TimeoutError:
        proc.kill()
        return -1, ""
    return proc.returncode or 0, out.decode("utf-8", "replace")


async def _picture_size(raw: bytes, path: str) -> tuple[int, int] | None:
    size = _png_size(raw)
    if size is not None:
        return size
    code, out = await _sips("-g", "pixelWidth", "-g", "pixelHeight", path)
    found = re.findall(r"pixel(?:Width|Height): (\d+)", out)
    if code != 0 or len(found) != 2:
        return None
    return int(found[0]), int(found[1])


async def fit_picture(mime: str, data: str) -> ImagePart | None:
    """A window's picture within the tap's limits: JPEG, long side 1280, at most 256 KB.

    Made with the system's ``sips``; without it, a picture already within the
    size limit is kept as it is. None when no picture fits.
    """
    try:
        raw = base64.b64decode(data, validate=False)
    except ValueError:
        return None
    if not os.path.exists(SIPS):
        if len(raw) <= SHOT_MAX_BYTES and mime in ("image/png", "image/jpeg"):
            return ImagePart(mime_type=mime, data=data)
        return None
    folder = tempfile.mkdtemp(prefix="melo-desktop-")
    try:
        source = os.path.join(folder, "in.png" if _png_size(raw) else "in.jpg")
        with open(source, "wb") as handle:
            handle.write(raw)
        size = await _picture_size(raw, source)
        for side in (SHOT_LONG_SIDE, 960, 640):
            out = os.path.join(folder, f"out{side}.jpg")
            args = ["-s", "format", "jpeg", "-s", "formatOptions", str(SHOT_QUALITY)]
            if size is None or max(size) > side:
                args += ["-Z", str(side)]  # sips -Z also enlarges: only to shrink
            code, _ = await _sips(*args, source, "--out", out)
            if code != 0 or not os.path.isfile(out):
                return None
            if os.path.getsize(out) <= SHOT_MAX_BYTES:
                with open(out, "rb") as handle:
                    jpeg = handle.read()
                return ImagePart(
                    mime_type="image/jpeg", data=base64.b64encode(jpeg).decode("ascii")
                )
        return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


async def helper_status(*, check: bool = False) -> HelperStatus:
    """What is known of the helper on this Mac, for the settings page and the health route.

    ``missing`` when it is not installed. Otherwise the newest status a check
    found (a chat's first use of the helper is one), else ``installed``. With
    *check*, and when the newest status is older than :data:`STATUS_TTL_S`, the
    helper is started once to read its grants (without a prompt) and stopped
    again.
    """
    command = helper_command()
    if command is None:
        return HelperStatus("missing")

    def newest() -> tuple[HelperStatus, float] | None:
        found = _last
        return (found[0], found[2]) if found and found[1] == tuple(command) else None

    known = newest()
    if check and (known is None or time.monotonic() - known[1] > STATUS_TTL_S):
        helper = DesktopHelper(command)
        try:
            await asyncio.wait_for(helper.ready(), CALL_S * 2)
        except TimeoutError:
            helper.fail("it did not answer")
        finally:
            await helper.close()
        known = newest()
    return known[0] if known else HelperStatus("installed")


# melo itself and the helper: never another app, whichever browser melo drives.
MELO_APPS = ("LocalMelo", "org.localmelo.app", "CuaDriver", HELPER_APP, HELPER_BUNDLE)
SAFARI_APPS = (
    "Safari",
    "Safari Technology Preview",
    "com.apple.Safari",
    "com.apple.SafariTechnologyPreview",
)


def watch_for(enabled: bool, driver: str = "safari-automation") -> DesktopWatch | None:
    """The watch a browsing chat gets: with the helper when it is installed; None when *enabled* is off.

    With Safari's automation windows Safari itself comes to the front, so it
    is never "another app". Chrome in the background never comes to the front,
    so with the ``chrome`` driver every app that does, Safari and Chrome
    included, is another app.
    """
    if not enabled:
        return None
    command = helper_command()
    skip = MELO_APPS if driver == "chrome" else (*SAFARI_APPS, *MELO_APPS)
    return DesktopWatch(DesktopHelper(command) if command else None, skip=skip)
