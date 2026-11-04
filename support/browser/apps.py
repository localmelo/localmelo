"""Addresses another app opens (``chat://...``): which app, and opening one for the person.

A page can ask the browser to open an address of another kind than a web page,
such as a chat app's sign-in link. Chrome in the background drops such a
request without a word, so its driver reports it, melo names the app and asks
the person, and on a yes melo opens the address itself with ``/usr/bin/open``:
the app comes to the front as it would after a click in the person's own
browser. melo never tells the browser to open such addresses by itself.
"""

from __future__ import annotations

import asyncio
import contextlib
from urllib.parse import urlsplit

from localmelo.support.browser.driver import BrowserError

# The browser's own kinds of address, and the web's: never another app's.
OWN_SCHEMES = frozenset(
    {
        "http",
        "https",
        "file",
        "about",
        "data",
        "javascript",
        "blob",
        "chrome",
        "devtools",
        "view-source",
        "ws",
        "wss",
        "filesystem",
    }
)
# The longest scheme melo names (a scheme is page data, so it is kept short).
SCHEME_CHARS = 30
OPEN = "/usr/bin/open"
OSASCRIPT = "/usr/bin/osascript"
CALL_S = 5.0
# Which app this Mac opens an address with: LaunchServices' answer, by name.
# The address comes in as an argument, never inside the script.
APP_JS = """function run(argv) {
  ObjC.import('AppKit');
  var url = $.NSURL.URLWithString(argv[0]);
  if (!url || url.isNil()) return '';
  var app = $.NSWorkspace.sharedWorkspace.URLForApplicationToOpenURL(url);
  if (!app || app.isNil()) return '';
  return ObjC.unwrap($.NSFileManager.defaultManager.displayNameAtPath(app.path));
}"""


def scheme_of(url: str) -> str:
    """The scheme of *url* in lower case (``chat``), or ``""`` when it has none."""
    try:
        return urlsplit(url.strip()).scheme.lower()
    except ValueError:
        return ""


def custom_scheme(url: str) -> bool:
    """Whether *url* is an address another app opens (``chat://...``), not a page or the browser's own."""
    scheme = scheme_of(url)
    return bool(scheme) and scheme not in OWN_SCHEMES and "-" not in scheme


def shown_scheme(url: str) -> str:
    """The kind of address in words for a step line: ``chat:…`` (never the rest of it)."""
    return scheme_of(url)[:SCHEME_CHARS] + ":…"


class AppOpener:
    """Names the app that opens an address and opens it, on this Mac.

    Tests pass their own; nothing here runs before melo has an address of
    another app's kind to ask about.
    """

    async def name_for(self, url: str) -> str:
        """The name of the app this Mac opens *url* with, or ``""`` when none (or melo cannot tell)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                OSASCRIPT,
                "-l",
                "JavaScript",
                "-e",
                APP_JS,
                url,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return ""
        try:
            async with asyncio.timeout(CALL_S):
                out, _ = await proc.communicate()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return ""
        if proc.returncode != 0:
            return ""
        name = out.decode("utf-8", errors="replace").strip()
        return name.removesuffix(".app")[:80]

    async def open(self, url: str) -> None:
        """Open *url* with the app this Mac opens it with; :class:`BrowserError` when that fails."""
        try:
            proc = await asyncio.create_subprocess_exec(
                OPEN,
                url,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(CALL_S):
                _, err = await proc.communicate()
        except (OSError, TimeoutError) as exc:
            raise BrowserError(f"melo could not open it: {exc}") from None
        if proc.returncode != 0:
            said = err.decode("utf-8", errors="replace").strip()[:200]
            raise BrowserError(f"melo could not open it: {said or 'open failed'}")
