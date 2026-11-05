"""The person takes over melo's browser from the Browser panel: who has each tab, and the person's input.

**Who has a tab.** Each tab of a chat is held by ``melo`` or by ``you`` (the
person), in :class:`Owners`, which melo's browser tools and the live view share.
The person takes a tab from the panel (or from the "melo needs you to sign in"
card) and gives it back from the panel. While they hold it:

* melo's browser steps on that tab wait (:meth:`Owners.wait_for_melo`), each for a
  bounded time, and the model is told plainly that the person is using the page;
* the page's boxes (``alert``, ``confirm``, ``prompt``) are the person's to
  answer, in the panel;
* when they give it back, melo reads the page again before it acts: its refs are
  of a page the person may have changed.

When nobody watches a held tab any more (the panel closed, the page hidden, the
gateway's token rotated), the tab stays the person's for :data:`GRACE_S` seconds
and then comes back to melo (:meth:`Owners.watching`). The panel says so.

**The person's input** comes over the live view's socket as small JSON
messages (:func:`cdp_events`): mouse moves, presses and releases, the wheel,
keys, text (an input method's commit and a paste, text only) and an input
method's composition. Each is checked for its shape, its coordinates are kept on
the page's view, and it becomes the browser's own trusted input
(``Input.dispatchMouseEvent``, ``Input.dispatchKeyEvent``, ``Input.insertText``,
``Input.imeSetComposition``). :class:`Bucket` holds a viewer to
:data:`RATE` messages a second.

**Nothing typed is kept.** The input goes from the panel to the browser and
nowhere else: no log line, event, store, memory or model ever holds a key or a
text the person typed or pasted. :class:`InputRefusedError` names only what is
wrong with a message's shape, never its content, and :class:`QuietFrames` keeps
the websocket library's frame-by-frame debug lines out of the logs.

**When melo needs the person.** :func:`needs_person` tells from a reading of a
page whether it is a sign-in (a password or code field on a sign-in page), a
code to enter, a QR code to scan or a check that the visitor is human; melo
then asks the person on a card to take over (see the browser tools).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

Owner = Literal["melo", "you"]
MELO: Final = "melo"
YOU: Final = "you"

# Seconds a held tab stays the person's once nobody watches it.
GRACE_S: Final = 120.0
# Seconds one browser step of melo's waits for the person to give a tab back.
# A step runs under the tool timeout of the run (60 s), so it waits less.
STEP_WAIT_S: Final = 45.0
# The person's input: messages a second a viewer may send, and the burst above that.
RATE: Final = 120.0
BURST: Final = 240.0
# The most characters one text, paste or composition message may carry.
MAX_TEXT: Final = 1000
MAX_COMPOSE: Final = 200
# How far one wheel message may scroll, in page pixels.
MAX_WHEEL: Final = 3000.0
# The modifier bits of a DevTools input event.
ALT, CTRL, META, SHIFT = 1, 2, 4, 8

_BUTTONS = {"left", "middle", "right", "none", "back", "forward"}
_MOUSE = {"move": "mouseMoved", "down": "mousePressed", "up": "mouseReleased"}
_KEY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]{0,23}")
_CODE = re.compile(r"[A-Za-z0-9]{0,32}")
# The keys a panel may send by name, with the key code the browser expects.
NAMED_KEYS: Final[Mapping[str, int]] = {
    "Backspace": 8,
    "Tab": 9,
    "Enter": 13,
    "Shift": 16,
    "Control": 17,
    "Alt": 18,
    "Escape": 27,
    "PageUp": 33,
    "PageDown": 34,
    "End": 35,
    "Home": 36,
    "ArrowLeft": 37,
    "ArrowUp": 38,
    "ArrowRight": 39,
    "ArrowDown": 40,
    "Delete": 46,
    "Meta": 91,
    **{f"F{n}": 111 + n for n in range(1, 13)},
}
# What a key does in a text field of Chrome on a Mac, which Chrome in the
# background needs said (its editing commands come from the Mac's key bindings,
# which a headless browser does not have). By key and modifiers.
_MAC_COMMANDS: Final[Mapping[tuple[str, int], tuple[str, ...]]] = {
    ("Backspace", 0): ("deleteBackward",),
    ("Backspace", SHIFT): ("deleteBackward",),
    ("Backspace", ALT): ("deleteWordBackward",),
    ("Backspace", META): ("deleteToBeginningOfLine",),
    ("Delete", 0): ("deleteForward",),
    ("Delete", ALT): ("deleteWordForward",),
    ("Escape", 0): ("cancelOperation",),
    ("ArrowUp", 0): ("moveUp",),
    ("ArrowDown", 0): ("moveDown",),
    ("ArrowLeft", 0): ("moveLeft",),
    ("ArrowRight", 0): ("moveRight",),
    ("ArrowUp", SHIFT): ("moveUpAndModifySelection",),
    ("ArrowDown", SHIFT): ("moveDownAndModifySelection",),
    ("ArrowLeft", SHIFT): ("moveLeftAndModifySelection",),
    ("ArrowRight", SHIFT): ("moveRightAndModifySelection",),
    ("ArrowLeft", ALT): ("moveWordLeft",),
    ("ArrowRight", ALT): ("moveWordRight",),
    ("ArrowLeft", ALT | SHIFT): ("moveWordLeftAndModifySelection",),
    ("ArrowRight", ALT | SHIFT): ("moveWordRightAndModifySelection",),
    ("ArrowUp", META): ("moveToBeginningOfDocument",),
    ("ArrowDown", META): ("moveToEndOfDocument",),
    ("ArrowLeft", META): ("moveToLeftEndOfLine",),
    ("ArrowRight", META): ("moveToRightEndOfLine",),
    ("ArrowLeft", META | SHIFT): ("moveToLeftEndOfLineAndModifySelection",),
    ("ArrowRight", META | SHIFT): ("moveToRightEndOfLineAndModifySelection",),
    ("Home", 0): ("scrollToBeginningOfDocument",),
    ("End", 0): ("scrollToEndOfDocument",),
    ("PageUp", 0): ("scrollPageUp",),
    ("PageDown", 0): ("scrollPageDown",),
    ("a", META): ("selectAll",),
    ("z", META): ("undo",),
    ("z", META | SHIFT): ("redo",),
    ("x", META): ("cut",),
    ("c", META): ("copy",),
}


class InputRefusedError(ValueError):
    """A message of the person's that is not input melo passes on.

    Its words name what is wrong with the message's shape, never anything it carries.
    """


# ── Who has a tab ──


@dataclass(frozen=True)
class Returned:
    """A tab the person gave back (or left): how long they had it, and how it came back.

    ``how`` is ``gave`` (Give back to melo), ``left`` (nobody watched it for the
    grace period) or ``card`` (they never took it after the card).
    """

    tab: str
    seconds: float
    how: Literal["gave", "left"]


@dataclass
class _Held:
    since: float
    away: float | None = None  # when nobody watched it any more, on the owners' clock
    timer: Any = None


class Owners:
    """Who has each tab of one chat: ``melo`` unless the person took it. See the module docstring.

    *on_change* is called after every change (the live view tells its viewers,
    and a waiting step looks again). *clock* and *later* (``later(seconds, fn)``
    returning something with ``cancel()``) are a test's fake time; by default the
    running loop's.
    """

    def __init__(
        self,
        *,
        on_change: Callable[[], None] | None = None,
        grace_s: float = GRACE_S,
        clock: Callable[[], float] = time.monotonic,
        later: Callable[[float, Callable[[], None]], Any] | None = None,
    ) -> None:
        self._on_change = on_change
        self.grace_s = grace_s
        self._clock = clock
        self._later = later
        self._held: dict[str, _Held] = {}
        self._returned: list[Returned] = []
        self._changed = asyncio.Event()

    def owner(self, tab: str | None) -> Owner:
        return YOU if tab is not None and tab in self._held else MELO

    @property
    def held(self) -> frozenset[str]:
        """The tabs the person has now."""
        return frozenset(self._held)

    def since(self, tab: str) -> float | None:
        """When the person took *tab*, on the owners' clock; None when melo has it."""
        held = self._held.get(tab)
        return held.since if held is not None else None

    def take(self, tab: str, *, watched: bool = True) -> bool:
        """The person takes *tab*; whether it changed hands (False when they had it).

        *watched* says whether a panel shows the tab now; one that does not
        starts the grace period at once (the card was answered with no panel open).
        """
        if tab in self._held:
            if watched:
                self.watching(tab, 1)
            return False
        self._held[tab] = _Held(self._clock())
        if not watched:
            self._go_away(tab)
        self._changed_now()
        return True

    def give(self, tab: str, how: Literal["gave", "left"] = "gave") -> Returned | None:
        """*tab* comes back to melo; what the next step of melo's is told (None when the person did not have it)."""
        held = self._held.pop(tab, None)
        if held is None:
            return None
        self._cancel(held)
        back = Returned(tab, max(0.0, self._clock() - held.since), how)
        self._returned.append(back)
        self._changed_now()
        return back

    def forget(self, tab: str) -> None:
        """*tab* is gone (closed, or no longer the chat's): nobody has it, and nothing is told about it."""
        held = self._held.pop(tab, None)
        if held is not None:
            self._cancel(held)
        self._returned = [r for r in self._returned if r.tab != tab]
        if held is not None:
            self._changed_now()

    def clear(self) -> None:
        """The chat is over: every tab is melo's again, and nothing waits."""
        for held in self._held.values():
            self._cancel(held)
        had = bool(self._held)
        self._held.clear()
        self._returned.clear()
        if had:
            self._changed_now()

    def watching(self, tab: str, viewers: int) -> None:
        """How many panels show *tab* now: none starts the grace period of a held tab, one or more ends it."""
        held = self._held.get(tab)
        if held is None:
            return
        if viewers > 0:
            if held.away is not None:
                self._cancel(held)
                held.away = None
                self._changed_now()
        elif held.away is None:
            self._go_away(tab)
            self._changed_now()

    def returned(self) -> list[Returned]:
        """The tabs given back since this was last asked, once each."""
        back, self._returned = self._returned, []
        return back

    async def wait_for_melo(self, tab: str, timeout_s: float) -> bool:
        """Wait until melo has *tab*, at most *timeout_s* seconds; whether it has."""
        loop = asyncio.get_running_loop()
        end = loop.time() + max(0.0, timeout_s)
        while tab in self._held:
            left = end - loop.time()
            if left <= 0:
                return False
            self._changed.clear()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(left):
                    await self._changed.wait()
        return True

    # ── Inside ──

    def _go_away(self, tab: str) -> None:
        held = self._held[tab]
        held.away = self._clock()
        held.timer = self._schedule(self.grace_s, lambda: self._expire(tab, held))

    def _expire(self, tab: str, held: _Held) -> None:
        if self._held.get(tab) is held and held.away is not None:
            held.timer = None
            self.give(tab, "left")

    def _schedule(self, seconds: float, fn: Callable[[], None]) -> Any:
        if self._later is not None:
            return self._later(seconds, fn)
        try:
            return asyncio.get_running_loop().call_later(seconds, fn)
        except RuntimeError:
            return None

    @staticmethod
    def _cancel(held: _Held) -> None:
        timer, held.timer = held.timer, None
        if timer is not None:
            with contextlib.suppress(Exception):
                timer.cancel()

    def _changed_now(self) -> None:
        self._changed.set()
        if self._on_change is not None:
            with contextlib.suppress(Exception):
                self._on_change()


def held_words(seconds: float) -> str:
    """``42 s`` or ``2 min 5 s``: how long the person had a tab."""
    whole = int(round(seconds))
    if whole < 60:
        return f"{max(whole, 1)} s"
    minutes, rest = divmod(whole, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


# ── The person's input ──


class Bucket:
    """At most *rate* messages a second with bursts of *burst* (a token bucket); the rest are dropped."""

    def __init__(
        self,
        rate: float = RATE,
        burst: float = BURST,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._tokens = burst
        self._at = clock()
        self.dropped = 0

    def allow(self) -> bool:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._at) * self.rate)
        self._at = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        self.dropped += 1
        return False


INPUT_TYPES: Final = frozenset({"mouse", "wheel", "key", "text", "compose", "paste"})


def _number(message: Mapping[str, Any], key: str) -> float:
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputRefusedError(f"{key} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise InputRefusedError(f"{key} is not a number")
    return number


def _whole(message: Mapping[str, Any], key: str, low: int, high: int) -> int:
    value = message.get(key, low)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputRefusedError(f"{key} is not a whole number")
    if not low <= value <= high:
        raise InputRefusedError(f"{key} is out of range")
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _point(
    message: Mapping[str, Any], view: tuple[float, float]
) -> tuple[float, float]:
    """The message's point, kept on the page's view of *view* (CSS pixels)."""
    width, height = view
    if not (width > 0 and height > 0):
        raise InputRefusedError("the page's size is not known yet")
    x = _clamp(_number(message, "x"), 0.0, width)
    y = _clamp(_number(message, "y"), 0.0, height)
    return round(x, 2), round(y, 2)


def _text(message: Mapping[str, Any], limit: int) -> str:
    value = message.get("text", "")
    if not isinstance(value, str):
        raise InputRefusedError("text is not text")
    if len(value) > limit:
        raise InputRefusedError(f"text is longer than {limit} characters")
    # Control characters other than new lines and tabs are not typed.
    return "".join(c for c in value if c in "\n\t" or (ord(c) >= 32 and ord(c) != 127))


def _key_name(message: Mapping[str, Any]) -> str:
    key = message.get("key")
    if not isinstance(key, str) or not key:
        raise InputRefusedError("key is missing")
    if key in NAMED_KEYS or (len(key) <= 2 and key.isprintable()):
        return key
    if _KEY_NAME.fullmatch(key):
        return key  # another named key (a media key, say): sent without text or command
    raise InputRefusedError("key is not a key")


def cdp_events(
    message: Mapping[str, Any], view: tuple[float, float]
) -> list[tuple[str, dict[str, Any]]]:
    """The browser input calls for one input *message* of the panel, on a page view of *view* CSS pixels.

    :class:`InputRefusedError` for a message that is not input of a known shape.
    The point of a mouse or wheel message is in page CSS pixels already (the
    panel maps the picture to the page by the frame's metadata) and is kept
    on the view.
    """
    kind = message.get("type")
    if kind == "mouse":
        action = message.get("action")
        if action not in _MOUSE:
            raise InputRefusedError("action is not move, down or up")
        x, y = _point(message, view)
        button = message.get("button", "none" if action == "move" else "left")
        if button not in _BUTTONS:
            raise InputRefusedError("button is not a button")
        params: dict[str, Any] = {
            "type": _MOUSE[str(action)],
            "x": x,
            "y": y,
            "modifiers": _whole(message, "mods", 0, 15),
            "button": button,
            "buttons": _whole(message, "buttons", 0, 31),
        }
        if action != "move":
            params["clickCount"] = _whole(message, "clicks", 0, 3) or 1
        return [("Input.dispatchMouseEvent", params)]
    if kind == "wheel":
        x, y = _point(message, view)
        return [
            (
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": x,
                    "y": y,
                    "deltaX": round(
                        _clamp(_number(message, "dx"), -MAX_WHEEL, MAX_WHEEL), 2
                    ),
                    "deltaY": round(
                        _clamp(_number(message, "dy"), -MAX_WHEEL, MAX_WHEEL), 2
                    ),
                    "modifiers": _whole(message, "mods", 0, 15),
                },
            )
        ]
    if kind == "key":
        return [("Input.dispatchKeyEvent", _key(message))]
    if kind in ("text", "paste"):
        text = _text(message, MAX_TEXT)
        return [("Input.insertText", {"text": text})] if text else []
    if kind == "compose":
        text = _text(message, MAX_COMPOSE)
        end = len(text)
        start = _whole(message, "start", 0, end) if "start" in message else end
        stop = _whole(message, "end", start, end) if "end" in message else end
        return [
            (
                "Input.imeSetComposition",
                {"text": text, "selectionStart": start, "selectionEnd": stop},
            )
        ]
    raise InputRefusedError("type is not input")


def _key(message: Mapping[str, Any]) -> dict[str, Any]:
    action = message.get("action")
    if action not in ("down", "up"):
        raise InputRefusedError("action is not down or up")
    key = _key_name(message)
    code = message.get("code", "")
    if not isinstance(code, str) or not _CODE.fullmatch(code):
        raise InputRefusedError("code is not a key code")
    mods = _whole(message, "mods", 0, 15)
    key_code = NAMED_KEYS.get(key)
    if key_code is None:
        key_code = _whole(message, "keyCode", 0, 255)
    params: dict[str, Any] = {
        "key": key,
        "code": code,
        "windowsVirtualKeyCode": key_code,
        "nativeVirtualKeyCode": key_code,
        "modifiers": mods,
    }
    if message.get("repeat") is True:
        params["autoRepeat"] = True
    if action == "up":
        params["type"] = "keyUp"
        return params
    # A printable key types its character unless a shortcut is held (Shift and
    # Option still type); Enter types a new line. Anything else is a raw key.
    text = ""
    if key == "Enter" and not mods & (META | CTRL):
        text = "\r"
    elif (
        len(key) <= 2
        and key.isprintable()
        and key not in NAMED_KEYS
        and not mods & (META | CTRL)
    ):
        text = key
    params["type"] = "keyDown" if text else "rawKeyDown"
    if text:
        params["text"] = params["unmodifiedText"] = text
    commands = _MAC_COMMANDS.get((key.lower() if len(key) == 1 else key, mods))
    if commands:
        params["commands"] = list(commands)
    return params


# ── Logging ──


# The websocket library's debug line for each frame sent or received, and for
# each header of the opening handshake (request and response).
_FRAME_LINES: Final = ("< %s", "> %s")
_HEADER_LINES: Final = ("< %s: %s", "> %s: %s")
# The headers that carry a credential (the gateway's bearer and its cookie).
CREDENTIAL_HEADERS: Final = frozenset(
    {"authorization", "proxy-authorization", "cookie", "set-cookie"}
)
REDACTED: Final = "[redacted]"


def _credential(name: object) -> bool:
    lowered = str(name).strip().lower()
    return lowered in CREDENTIAL_HEADERS or any(
        word in lowered for word in ("token", "secret", "api-key", "apikey")
    )


class QuietFrames(logging.Filter):
    """Keeps what the person typed and the gateway's credential out of the websocket library's logs.

    Its debug line for each frame sent or received, which would show what the
    person typed, is dropped. A handshake header that carries a credential
    (:data:`CREDENTIAL_HEADERS`: the bearer, the cookie) keeps its line with the
    value replaced by :data:`REDACTED`, whatever the record's level: a logger
    turned to debug after the gateway started logs no credential either.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.msg, str):
            return True
        if record.msg in _FRAME_LINES:
            return record.levelno > logging.DEBUG
        args = record.args
        if (
            record.msg in _HEADER_LINES
            and isinstance(args, tuple)
            and len(args) == 2
            and _credential(args[0])
        ):
            record.args = (args[0], REDACTED)
        return True


# The loggers the websocket server writes its frames to: uvicorn's own and the library's.
FRAME_LOGGERS: Final = (
    "uvicorn.error",
    "websockets",
    "websockets.server",
    "websockets.protocol",
    "websockets.asyncio.server",
    "websockets.legacy.server",
)


def quiet_frames(names: Iterable[str] = FRAME_LOGGERS) -> None:
    """Put :class:`QuietFrames` on each logger in *names* (once)."""
    for name in names:
        logger = logging.getLogger(name)
        if not any(isinstance(f, QuietFrames) for f in logger.filters):
            logger.addFilter(QuietFrames())


# ── When melo needs the person ──

Need = Literal["", "sign-in", "code", "qr", "captcha"]

_SIGN_IN_WORDS = re.compile(
    r"log ?in|log-in|sign ?in|sign-in|signin|login|password|passwort"
    r"|登录|登入|登陆|ログイン|로그인|anmelden|connexion|iniciar sesi|accedi|entrar",
    re.IGNORECASE,
)
_CODE_PAGE_WORDS = re.compile(
    r"verif|one[- ]?time|\botp\b|2fa|two[- ]factor|two[- ]step|security code|"
    r"sms|text message|authenticat|验证|驗證|認証|인증|校验|确认码|動態|动态",
    re.IGNORECASE,
)
_QR_WORDS = re.compile(
    r"\bqr\b|qr ?code|qr-code|二维码|二維碼|扫码|掃碼|扫一扫|掃一掃|QRコード|QR 코드",
    re.IGNORECASE,
)
_SCAN_WORDS = re.compile(r"scan|扫|掃|スキャン|스캔|読み取", re.IGNORECASE)
# A check the visitor must pass in person, in its own words (a page that only
# says it is protected by one is not asking anything yet).
_HUMAN_WORDS = re.compile(
    r"i'?m not a robot|verify (that )?you are (a )?human|are you a robot|"
    r"human verification|confirm you are human|press (and|&) hold|"
    r"人机验证|人機驗證|我不是机器人|我不是機器人|拖动滑块|向右滑动|安全验证|"
    r"ロボットではありません",
    re.IGNORECASE,
)
_HUMAN_TITLES = re.compile(
    r"^(just a moment|attention required|one more step|verify you are human|"
    r"security check|are you a robot|请稍候|安全验证)",
    re.IGNORECASE,
)
_CHECK_HOSTS = re.compile(
    r"(^|\.)(recaptcha\.net|hcaptcha\.com|challenges\.cloudflare\.com|"
    r"geetest\.com|arkoselabs\.com|funcaptcha\.com|captcha[a-z0-9-]*\.[a-z.]+)$"
)
# A page this small that holds a password field is a sign-in form, whatever its address says.
SMALL_PAGE: Final = 30


def needs_person(
    *,
    url: str,
    title: str,
    text: str,
    sensitive: Iterable[str],
    frames: Iterable[str] = (),
    elements: int = 0,
) -> Need:
    """What on a page needs the person, from one reading of it; ``""`` when nothing does.

    *sensitive* are the kinds of the secret fields on it (``password``,
    ``code``, ``card``), *frames* the origins of boxes from other sites melo
    cannot read, *text* the page's words and *elements* how many elements the
    reading numbered. In order:

    * ``captcha``: a box of a known human check, a page titled as one, or words
      that ask the visitor to prove they are human;
    * ``qr``: a QR code to scan, on a page that signs in;
    * ``sign-in``: a password field on a page whose address or title says it
      signs in, or on a page with few elements (a sign-in form); a password
      field at the side of a long page is not one;
    * ``code``: a one-time code field on a page that says it verifies.
    """
    kinds = set(sensitive)
    hosts = {_host(origin) for origin in frames}
    words = f"{title}\n{text}"
    if any(_CHECK_HOSTS.search(host) for host in hosts if host) or (
        _HUMAN_TITLES.match(title.strip()) or _HUMAN_WORDS.search(words)
    ):
        return "captcha"
    if (
        _QR_WORDS.search(words)
        and _SCAN_WORDS.search(words)
        and _SIGN_IN_WORDS.search(f"{url}\n{words}")
    ):
        return "qr"
    if "password" in kinds and (
        _SIGN_IN_WORDS.search(f"{_path(url)}\n{title}") or elements <= SMALL_PAGE
    ):
        return "sign-in"
    if "code" in kinds and _CODE_PAGE_WORDS.search(f"{_path(url)}\n{words}"):
        return "code"
    return ""


def _path(url: str) -> str:
    """The part of *url* after its host: what a sign-in page's address says about it."""
    match = re.match(r"^[a-z][a-z0-9+.-]*://[^/?#]*(.*)$", url.strip(), re.IGNORECASE)
    return match.group(1) if match else url


def _host(origin: str) -> str:
    match = re.match(r"^[a-z][a-z0-9+.-]*://([^/:?#]+)", origin.strip().lower())
    return match.group(1) if match else ""
