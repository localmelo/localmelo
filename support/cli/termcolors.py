"""The terminal's colours: how many it shows, which background it has, and the arithmetic.

:func:`color_depth` reads the environment and the ``[cli] colors`` setting.
:func:`query_colors` asks the terminal for its background and foreground
(OSC 11 and OSC 10, then DA1 as a sentinel that every VT100-compatible
terminal answers), for the interactive UI only and for at most
:data:`QUERY_TIMEOUT_S`. The rest is colour arithmetic: WCAG contrast, sRGB
blending, OKLab distance, a lightness fit that keeps the hue, and the nearest
xterm-256 index.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import select
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

Depth = Literal["none", "16", "256", "truecolor"]
DEPTHS: Final = ("auto", "16", "256", "truecolor")
BACKGROUNDS: Final = ("auto", "dark", "light")
QUERY_TIMEOUT_S: Final = 0.1
# Background, foreground, then the primary device attributes.
QUERY: Final = b"\x1b]11;?\x07\x1b]10;?\x07\x1b[c"
_OSC_REPLY: Final = re.compile(
    rb"\x1b\]1([01]);rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})"
    rb"(?:/[0-9a-fA-F]{1,4})?(?:\x07|\x1b\\)"
)
_DA1_REPLY: Final = re.compile(rb"\x1b\[\?[0-9;]*c")


def color_depth(environ: Mapping[str, str], setting: str = "auto") -> Depth:
    """The colours to draw with; the first rule that matches wins.

    ``NO_COLOR`` (not empty) or ``TERM=dumb`` gives none; then the
    ``[cli] colors`` setting; then ``COLORTERM`` of ``truecolor`` or
    ``24bit``; then a ``TERM`` that ends in ``256color``; else the 16 named
    colours.
    """
    if environ.get("NO_COLOR") or environ.get("TERM") == "dumb":
        return "none"
    if setting in ("16", "256", "truecolor"):
        return setting  # type: ignore[return-value]
    if environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if environ.get("TERM", "").endswith("256color"):
        return "256"
    return "16"


@dataclass(frozen=True)
class TerminalColors:
    """What the terminal said about its colours, each ``#rrggbb`` or ``None``."""

    background: str | None = None
    foreground: str | None = None


def _channel(digits: bytes) -> int:
    top = (1 << (4 * len(digits))) - 1
    return round(int(digits, 16) * 255 / top)


def parse_reply(data: bytes) -> tuple[TerminalColors, bool]:
    """The colours in the terminal's reply so far, and whether DA1's answer came."""
    found: dict[bytes, str] = {}
    for match in _OSC_REPLY.finditer(data):
        which, *channels = match.groups()
        found[which] = "#" + "".join(f"{_channel(c):02x}" for c in channels)
    colors = TerminalColors(found.get(b"1"), found.get(b"0"))
    return colors, _DA1_REPLY.search(data) is not None


def query_colors(
    fd_in: int,
    fd_out: int,
    environ: Mapping[str, str],
    *,
    timeout: float = QUERY_TIMEOUT_S,
    clock: Callable[[], float] = time.monotonic,
) -> TerminalColors | None:
    """Ask the terminal on *fd_in*/*fd_out* for its colours; ``None`` when it cannot.

    Only a terminal on both descriptors is asked (never a pipe, a file or
    ``TERM=dumb``). The terminal is read raw, without echo, until DA1's reply
    or *timeout* (three times as long once a reply has begun); what has
    arrived when the time is up is used and the rest of the input is
    discarded, so a reply that is late cannot reach the prompt.
    """
    if environ.get("TERM") == "dumb" or not (os.isatty(fd_in) and os.isatty(fd_out)):
        return None
    import termios

    try:
        saved = termios.tcgetattr(fd_in)
    except termios.error:
        return None
    raw = termios.tcgetattr(fd_in)
    raw[3] &= ~(termios.ICANON | termios.ECHO)
    raw[6][termios.VMIN], raw[6][termios.VTIME] = 0, 0
    data = b""
    try:
        termios.tcsetattr(fd_in, termios.TCSANOW, raw)
        os.write(fd_out, QUERY)
        start = clock()
        while True:
            colors, done = parse_reply(data)
            left = start + timeout * (3 if data else 1) - clock()
            if done or left <= 0:
                break
            ready, _, _ = select.select([fd_in], [], [], left)
            chunk = os.read(fd_in, 1024) if ready else b""
            if ready and not chunk:
                break
            data += chunk
        if not done:
            termios.tcflush(fd_in, termios.TCIFLUSH)
    except (OSError, termios.error):
        return None
    finally:
        with contextlib.suppress(OSError, termios.error):
            termios.tcsetattr(fd_in, termios.TCSANOW, saved)
    return colors if colors.background or colors.foreground else None


# ── Arithmetic ──

RGB = tuple[float, float, float]


def rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def to_hex(channels: Iterable[float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in channels)


def _linear(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return 255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)


def luminance(color: str) -> float:
    r, g, b = (_linear(c) for c in rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(first: str, second: str) -> float:
    """The WCAG contrast ratio of two colours, 1 to 21."""
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def is_dark(background: str) -> bool:
    """Whether light text reads better than dark text on *background*."""
    return contrast(background, "#ffffff") > contrast(background, "#000000")


def blend(top: str, bottom: str, alpha: float) -> str:
    """*top* over *bottom* at opacity *alpha*, in sRGB, as terminals blend."""
    pairs = zip(rgb(top), rgb(bottom), strict=True)
    return to_hex(t * alpha + b * (1 - alpha) for t, b in pairs)


def _oklab(color: str) -> RGB:
    r, g, b = (_linear(c) for c in rgb(color))
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    l_, m_, s_ = (math.copysign(abs(x) ** (1 / 3), x) for x in lms)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _from_oklch(lightness: float, chroma: float, hue: float) -> str:
    """An sRGB colour at this OKLCH point, its chroma lowered until it fits."""
    for _ in range(60):
        a, b = chroma * math.cos(hue), chroma * math.sin(hue)
        l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
        linear = (
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
        )
        if all(-0.0005 <= c <= 1.0005 for c in linear):
            return to_hex(_gamma(c) for c in linear)
        chroma *= 0.95
    return to_hex(_gamma(lightness**3) for _ in range(3))


def fit(color: str, backgrounds: Iterable[str], target: float) -> str:
    """*color* with its OKLCH lightness moved, hue kept, until it meets *target* on each background."""
    grounds = list(backgrounds)
    if min(contrast(color, bg) for bg in grounds) >= target:
        return color
    lightness, a, b = _oklab(color)
    chroma, hue = math.hypot(a, b), math.atan2(b, a)
    step = 0.005 if sum(map(is_dark, grounds)) * 2 > len(grounds) else -0.005
    candidate = color
    for _ in range(200):
        lightness = max(0.0, min(1.0, lightness + step))
        candidate = _from_oklch(lightness, chroma, hue)
        if min(contrast(candidate, bg) for bg in grounds) >= target:
            break
    return candidate


def _xterm() -> tuple[str, ...]:
    steps = (0, 95, 135, 175, 215, 255)
    cube = [
        to_hex((steps[i // 36], steps[i // 6 % 6], steps[i % 6])) for i in range(216)
    ]
    greys = [to_hex((8 + 10 * i,) * 3) for i in range(24)]
    return ("",) * 16 + tuple(cube) + tuple(greys)


# Indices 16-255; 0-15 are the user's theme and are never chosen.
XTERM: Final = _xterm()


def nearest256(color: str, backgrounds: Iterable[str] = (), target: float = 0.0) -> int:
    """The closest xterm index in 16-255 (OKLab) that meets *target* on each background."""
    grounds = list(backgrounds)
    here = _oklab(color)
    ranked = sorted(range(16, 256), key=lambda i: math.dist(here, _oklab(XTERM[i])))
    for index in ranked:
        if all(contrast(XTERM[index], bg) >= target for bg in grounds):
            return index
    return ranked[0]
