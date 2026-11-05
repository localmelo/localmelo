"""The ``safari-automation`` driver: Safari's own MCP server, ``safaridriver --mcp``.

It runs in Safari's isolated automation windows: a clean slate like a private
window, no saved logins, one session at a time, and a visible orange address
bar. Nothing is installed; Safari needs two switches turned on once (see
:data:`SETUP_MESSAGE`). The server is started on first use, with the stdio MCP
client the rest of melo uses.

Pages are read with ``evaluate_javascript``: one script returns the page's text
and elements as JSON (:data:`~localmelo.support.browser.pagescript.PAGE_JS`), so
what the model reads does not depend on the format of the server's own content
tool. Tool and argument names are looked up in the server's ``tools/list``
(:data:`TOOL_NAMES`, :data:`ARG_NAMES`).

An act is done by the browser itself where it can be: the page script checks the
element and finds which of the browser's node ids it is (from
``get_page_content``), and ``page_interactions`` clicks, types or chooses on that
node, with real events a page cannot tell from a person's. The page script's own
events are the backup, and the result names the way used. Around either way an
observer set before the act sees what the page shows new, and melo waits until
the page is quiet (see :meth:`SafariAutomationDriver.act`). The page scripts and
what is done with their answers are the ones Chrome's driver uses too (see
:mod:`~localmelo.support.browser.pagescript`).

A tab is the handle the server gave it. A tab's address is what the server's own
tab list reports for that handle (a bare ``location.href`` is read only when the
list carries none), never a value a page serialises for itself. Every call
switches to the tab it means. A new tab is the one handle that appeared between
two tab lists; if not exactly one did, nothing is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import stat
import tempfile
import uuid
from base64 import b64encode
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser import fields, pagescript
from localmelo.support.browser.driver import (
    MAX_TYPED_CHARS,
    Act,
    ActResult,
    BrowserError,
    BrowserSetupError,
    Element,
    ElementKind,
    FieldAttrs,
    Sensitive,
    Shot,
    StaleElementError,
    TabInfo,
    View,
)
from localmelo.support.browser.pagescript import (
    PASSED_OVER,
    STALE,
    PageScriptActs,
    act_script,
    act_spec,
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
from localmelo.support.mcp import McpError, McpServerConfig, McpTool, StdioMcpClient
from localmelo.support.mcp.results import tool_result_to_output

SETUP_MESSAGE = (
    "Safari is not set up for melo yet. In Safari, open Settings > Developer and "
    'turn on "Allow remote automation" (its full label may read "Allow remote '
    'automation and external agents"). If Settings has no Developer tab, open the '
    'Advanced tab and tick "Show features for web developers" first. Then ask '
    "again: nothing else needs installing."
)
_SETUP_TEXT = re.compile(
    r"remote automation|external agents|enable .*automation|automation .*(disabled|not (enabled|allowed))"
    r"|session not created|web developers",
    re.IGNORECASE,
)

# The tools melo calls, by what they do, and the names Safari gives them.
TOOL_NAMES = {
    "open": "navigate_to_url",
    "script": "evaluate_javascript",
    "shot": "screenshot",
    "tabs": "list_tabs",
    "switch": "switch_tab",
    "new_tab": "create_tab",
    "close": "close_tab",
    "content": "get_page_content",
    "interact": "page_interactions",
}
REQUIRED = ("open", "script")
# The argument names a tool may use for what melo sends, first match wins; a
# tool with one parameter gets the value there whatever it is called.
ARG_NAMES = {
    "url": ("url", "uri", "address"),
    "script": ("expression", "script", "javascript", "code", "source"),
    "tab": ("handle", "tab_uuid", "tab_id", "tabId", "tab", "id", "window", "index"),
    "save_path": ("savePath", "save_path", "path"),
}
# A picture is a PNG file Safari saves where melo says, and no larger than this.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_SHOT_BYTES = 10 * 1024 * 1024
MAX_NODES = 800
MAX_CONTENT_BYTES = 4 * 1024 * 1024


def _translate(text: str, *, starting: bool = False) -> BrowserError:
    """The error to show for *text* from the server: a setup problem, or what it said."""
    if _SETUP_TEXT.search(text):
        return BrowserSetupError(SETUP_MESSAGE)
    where = "did not start" if starting else "reported a problem"
    hint = (
        " Check Safari > Settings > Developer > Allow remote automation."
        if starting
        else ""
    )
    return BrowserError(
        f"Safari's automation helper {where}: {text.strip()[:300]}.{hint}"
    )


def _parse_tabs(text: str) -> list[tuple[TabInfo, bool]]:
    """Tabs (and whether each is the active one) from a ``list_tabs`` reply."""
    value = decode(text)
    items = value.get("tabs", []) if isinstance(value, dict) else value
    if not isinstance(items, list):
        return []
    tabs = []
    for item in items:
        # A tab is the handle the browser gave it. A row with none is not a tab
        # melo can name: its place in the list is no identity, since it moves
        # when another tab closes.
        if not isinstance(item, dict):
            continue
        ident = next(
            (item[k] for k in (*ARG_NAMES["tab"], "windowHandle") if item.get(k)), ""
        )
        if not isinstance(ident, str) or not ident:
            continue
        active = any(item.get(k) for k in ("active", "current", "selected"))
        info = TabInfo(ident, str(item.get("url") or ""), str(item.get("title") or ""))
        tabs.append((info, active))
    return tabs


class SafariAutomationDriver(PageScriptActs):
    """:class:`BrowserDriver` over ``safaridriver --mcp``. See the module docstring."""

    name = "safari-automation"

    def __init__(
        self,
        *,
        command: str = "/usr/bin/safaridriver",
        args: tuple[str, ...] = ("--mcp",),
        client_factory: type[StdioMcpClient] = StdioMcpClient,
        startup_timeout_s: float = 30.0,
        call_timeout_s: float = 60.0,
    ) -> None:
        self._config = McpServerConfig(
            name="safari",
            command=command,
            args=list(args),
            timeout_s=call_timeout_s,
            startup_timeout_s=startup_timeout_s,
        )
        self._factory = client_factory
        self._call_timeout_s = call_timeout_s
        self._client: StdioMcpClient | None = None
        self._tools: dict[str, McpTool] = {}
        self._current: str | None = None
        self._returns: bool | None = None  # whether scripts need a leading "return"
        self._shots: str | None = (
            None  # melo's private folder for the files Safari saves for it
        )
        self._ring = f"melo-ring-{uuid.uuid4().hex[:12]}"

    @property
    def tool_names(self) -> list[str]:
        """The names the server listed (empty before it started)."""
        return sorted(self._tools)

    # ── Lifecycle ──

    async def start(self) -> None:
        if self._client is not None:
            return
        client = self._factory(self._config)
        try:
            await client.start()
            listed = await client.list_tools()
        except McpError as exc:
            await _quiet_close(client)
            raise _translate(str(exc), starting=True) from exc
        except BaseException:
            await _quiet_close(client)
            raise
        tools = {tool.name: tool for tool in listed}
        missing = [TOOL_NAMES[key] for key in REQUIRED if TOOL_NAMES[key] not in tools]
        if missing:
            await _quiet_close(client)
            raise BrowserError(
                "This Safari's automation server lacks the tools melo needs "
                f"({', '.join(missing)}). Update macOS and Safari, then try again."
            )
        self._client, self._tools = client, tools

    async def close(self) -> None:
        client, self._client = self._client, None
        self._current = None
        shots, self._shots = self._shots, None
        if shots is not None:
            shutil.rmtree(
                shots, ignore_errors=True
            )  # the folder melo made, and only it
        if client is not None:
            await client.close()

    def kill(self) -> None:
        if self._client is not None:
            self._client.kill()

    # ── Calls ──

    def _args(self, tool: McpTool, **values: Any) -> dict[str, Any]:
        props = tool.input_schema.get("properties")
        props = props if isinstance(props, dict) else {}
        out: dict[str, Any] = {}
        for key, value in values.items():
            named = next((n for n in ARG_NAMES[key] if n in props), None)
            if named is None:
                named = next(iter(props)) if len(props) == 1 else key
            out[named] = value
        return out

    async def _call(self, key: str, **values: Any) -> ToolOutput:
        tool = self._tools.get(TOOL_NAMES[key])
        if self._client is None or tool is None:
            raise BrowserError(
                f"Safari's automation server has no {TOOL_NAMES[key]} tool."
            )
        return await self._send(tool, self._args(tool, **values))

    async def _call_with(self, key: str, arguments: dict[str, Any]) -> ToolOutput:
        """Call a tool with the *arguments* its schema lists (any other is left out)."""
        tool = self._tools.get(TOOL_NAMES[key])
        if self._client is None or tool is None:
            raise BrowserError(
                f"Safari's automation server has no {TOOL_NAMES[key]} tool."
            )
        props = tool.input_schema.get("properties")
        props = props if isinstance(props, dict) else {}
        return await self._send(
            tool, {k: v for k, v in arguments.items() if k in props}
        )

    async def _send(self, tool: McpTool, arguments: dict[str, Any]) -> ToolOutput:
        assert self._client is not None
        try:
            async with asyncio.timeout(self._call_timeout_s):
                result = await self._client.call_tool(tool.name, arguments)
        except TimeoutError:
            raise BrowserError(
                f"Safari did not answer {tool.name} in {self._call_timeout_s:g} s."
            ) from None
        except McpError as exc:
            self._client, dead = None, self._client  # the next call starts it again
            if dead is not None:
                await _quiet_close(dead)
            raise _translate(str(exc)) from exc
        output = tool_result_to_output(result)
        if output.is_error:
            raise _translate(output.text)
        return output

    async def _eval(self, source: str) -> Any:
        if self._returns is None:  # which of the two ways this server runs a script
            self._returns = False
            for returns in (False, True):
                try:
                    probe = await self._call(
                        "script", script=("return 1+1" if returns else "1+1")
                    )
                except BrowserSetupError:
                    raise
                except BrowserError:
                    continue
                if decode(probe.text) == 2:
                    self._returns = returns
                    break
        out = await self._call(
            "script", script=("return " + source if self._returns else source)
        )
        return decode(out.text)

    async def _focus(self, tab: str | None) -> None:
        """Make *tab* the tab the scripts run in.

        Done on every call, not remembered: Safari's current tab can move
        without melo (a page that opens a tab, the person), and a script must
        never run in a tab melo did not choose.
        """
        if not tab or tab == "current":
            return
        if self._has("switch"):
            await self._call("switch", tab=tab)
        self._current = tab

    async def _listed(self) -> list[tuple[TabInfo, bool]]:
        if not self._has("tabs"):
            raise BrowserError(
                "Safari's automation server cannot list tabs, so melo cannot tell "
                "which tab is which."
            )
        return _parse_tabs((await self._call("tabs")).text)

    async def _info(self, tab: str) -> TabInfo:
        """*tab* as Safari's own tab list reports it: its address and title."""
        for info, _ in await self._listed():
            if info.id == tab:
                return info
        raise BrowserError("Safari no longer has that tab.")

    async def _address(self, tab: str | None) -> str:
        """The address Safari reports for *tab* (``""`` when it cannot say).

        It comes from the tab list. Only a server whose list has no address is
        asked in the page, with a bare ``location.href`` (an unforgeable
        property read by the server itself, not a value the page serialises).
        """
        handle = tab if tab and tab != "current" else self._current
        if not handle:
            return ""
        with contextlib.suppress(BrowserError):
            if url := (await self._info(handle)).url:
                return url
        await self._focus(handle)
        where = await self._eval("location.href")
        return where if isinstance(where, str) else ""

    async def _here(self, tab: str) -> TabInfo:
        """Where *tab* is now: its address from Safari, its title from the list or the page."""
        info = TabInfo(tab)
        with contextlib.suppress(BrowserError):
            info = await self._info(tab)
        url = info.url or await self._address(tab)
        title = info.title
        if not title:
            await self._focus(tab)
            said = await self._eval("document.title")
            title = said if isinstance(said, str) else ""
        return TabInfo(tab, url, title)

    # ── The driver ──

    async def open(self, url: str, *, new_tab: bool, tab: str | None = None) -> TabInfo:
        if new_tab:
            if not self._has("new_tab"):
                raise BrowserError("Safari's automation server cannot open a new tab.")
            before = {info.id for info, _ in await self._listed()}
            tool = self._tools[TOOL_NAMES["new_tab"]]
            props = tool.input_schema.get("properties")
            sent = isinstance(props, dict) and "url" in props
            await self._call("new_tab", **({"url": url} if sent else {}))
            # The tab melo opened is the one that appeared: never "the active one"
            # or "the last one", which another tab can be.
            fresh = [i.id for i, _ in await self._listed() if i.id not in before]
            if len(fresh) != 1:
                raise BrowserError(
                    "melo could not tell which tab Safari opened, so it did not use it."
                )
            target = fresh[0]
            await self._focus(target)
            if not sent:
                await self._call("open", url=url)
        else:
            target = tab if tab and tab != "current" else (self._current or "")
            if not target:
                raise BrowserError("No tab is open. Use browser_open first.")
            await self._focus(target)
            await self._call("open", url=url)
        await self._ready()
        return await self._here(target)

    async def _other_sites(
        self, boxes: tuple[str, ...]
    ) -> tuple[list[Element], tuple[str, ...]]:
        """The controls in frames from other sites, as the browser's own reading names them.

        A frame from another site cannot be read by the page script. Its
        controls are the browser's own reading of the page (acted on by its
        node ids); a frame with none stays among the boxes left unreached.
        """
        if not boxes or not self._has("content"):
            return [], boxes
        found = foreign_elements(await self._nodes("entire_page", frames=True), boxes)
        got = {e.site for e in found}
        return found, tuple(b for b in boxes if sites_origin(b) not in got)

    async def back(self, tab: str | None) -> TabInfo:
        handle = tab if tab and tab != "current" else (self._current or "")
        if not handle:
            raise BrowserError("No tab is open. Use browser_open first.")
        await self._focus(handle)
        await self._eval("(history.back(), true)")
        await asyncio.sleep(pagescript.POLL_S)
        await self._ready()
        return await self._here(handle)

    async def tabs(self) -> list[TabInfo]:
        return [tab for tab, _ in _parse_tabs((await self._call("tabs")).text)]

    async def switch(self, tab: str) -> TabInfo:
        await self._call("switch", tab=tab)
        self._current = tab
        return await self._here(tab)

    async def close_tab(self, tab: str) -> None:
        await self._call("close", tab=tab)
        if self._current == tab:
            self._current = None

    async def location(self, tab: str | None) -> str:
        return await self._address(tab)

    async def holds(
        self, tab: str | None, until: str, text: str, *, origin: str = ""
    ) -> bool:
        await self._focus(tab)
        try:
            return await self._probe(wait_probe(until, text, origin))
        except BrowserSetupError:
            raise
        except BrowserError:
            return False  # a page between two documents does not answer yet

    async def _capture(self, view: View | None) -> Shot:
        # Safari saves the picture to a file and answers with text only. melo
        # names the file, in a folder only it can enter, reads that path and
        # never one it finds in the reply, and deletes the file whatever happens.
        path = os.path.join(self._folder(), f"{uuid.uuid4().hex}.png")
        try:
            out = await self._call("shot", save_path=path)
            if out.images:
                return Shot(out.images[0].mime_type, out.images[0].data, view)
            picture = _read_picture(path)
            return Shot(picture.mime_type, picture.data, view)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)  # a link is removed, never followed

    def _folder(self) -> str:
        """melo's private folder for the files Safari saves for it (mode 0700)."""
        if self._shots is None:
            self._shots = tempfile.mkdtemp(prefix="melo-safari-")
        return self._shots

    async def blank(self, tab: str | None) -> None:
        """Leave the page in *tab* for an empty one, in its place in the history.

        ``location.replace`` swaps the page's own history entry, so Back never
        returns to the page that was refused (it is an unforgeable property, a
        page cannot swap it for its own). Only when that does not take, the
        browser's own navigation is used, which adds an entry after the page.
        """
        await self._focus(tab)
        try:
            await self._eval("(location.replace('about:blank'), true)")
            if await self._settled_blank(tab):
                return
        except BrowserSetupError:
            raise
        except BrowserError:
            pass
        await self._call("open", url="about:blank")

    async def act(self, tab: str | None, act: Act) -> ActResult:
        """Do *act*: by the browser's own input first, else by the page script.

        The browser's own input (``page_interactions``) needs the element to be
        one the browser names: the page script checks the element as it was
        read, brings it into view and finds its node id among those the
        browser's own reading of the view gives (``get_page_content``), keeps
        its guards up, and then the browser acts on that node, or at the point.
        When the browser cannot act (no such tool, no node id, another element
        covering it) or its input did not reach the element, the page script
        acts instead and the result says why. After either way melo looks at
        the page until it is quiet (see :meth:`_look`).
        """
        if len(act.text) > MAX_TYPED_CHARS:
            raise BrowserError(
                f"melo types at most {MAX_TYPED_CHARS} characters at a time."
            )
        await self._focus(tab)
        if act.element.frame.startswith("x"):
            return await self._foreign_act(act)
        spec = act_spec(act)
        passed, failed = "", False
        if act.native or act.point is not None:
            done = await self._native(spec, act)
            if isinstance(done, ActResult):
                return done
            passed, failed = done
        return await self._scripted(spec, act, passed, failed)

    async def _native(
        self, spec: dict[str, Any], act: Act
    ) -> ActResult | tuple[str, bool]:
        """The act by the browser's own input; else why not, and whether that holds for the site.

        The element is scrolled to the middle of the view first. This Safari
        clicks the node itself, not a point melo picks, so an element whose
        middle is covered is left to the page script (and the result names what
        covers it). A real click that still lands on another element is stopped
        by the guards; the element is then scrolled to once more and the
        click tried again, and when that lands elsewhere too the error says what
        got it.
        """
        if not self._has("interact"):
            return PASSED_OVER["tool"], False
        for attempt in range(2):
            uid = ""
            if act.point is None:
                if act.element.frame:
                    return PASSED_OVER["frame"], False
                if not self._has("content"):
                    return PASSED_OVER["tool"], False
                ready = await self._scrolled(spec)
                if ready.get("covered") is True or ready.get("shifted") is True:
                    return covered_why(ready.get("by")), False
                uid = bind(ready, await self._nodes())
                if not uid:
                    return PASSED_OVER["name"], False
            key = new_key()
            armed = await self._script(
                act_script({**spec, "mode": "arm", "key": key, "uid": uid})
            )
            try:
                await self._call_with(
                    "interact", {"interactions": [_interaction(act, uid, armed)]}
                )
            except BrowserSetupError:
                with contextlib.suppress(BrowserError):
                    await self._finish(key, spec["doc"])
                raise
            except BrowserError:
                # Whether it acted before it failed is read from the page: an act
                # that shows there is kept, one that does not falls to the page script.
                pass
            data, quiet = await self._look(key, spec["doc"], act)
            blocked = [str(k) for k in data.get("blocked") or []]
            if "moved" in blocked:
                if attempt == 0 and act.point is None and nothing_reached(data):
                    continue  # the page moved under the click: scroll to it again
                raise cover_stop(data.get("movedBy"), act.element.kind)
            break
        if not reached(act, data, frame=armed.get("frame") is True):
            if act.point is not None and armed.get("frame") is True:
                raise BrowserError(STALE["reach"])
            return PASSED_OVER["missed"], True
        self._check_typed(act, data)
        return ActResult(
            stopped(blocked),
            "point" if act.point is not None else "browser",
            "",
            observed(data, quiet),
        )

    async def _foreign_act(self, act: Act) -> ActResult:
        """An act on a control in a frame from another site, by the browser's own input.

        No page script reaches into that frame, so the checks are the
        browser's: the node must still be in its reading with the same tag and
        words, and the browser acts on it. melo asked the person first.
        """
        if not self._has("interact") or not self._has("content"):
            raise BrowserError(STALE["reach"])
        if await self._address(None) != act.url:
            raise StaleElementError(STALE["page"], "page")
        uid = act.element.frame[1:]
        now = foreign_elements(
            await self._nodes("entire_page", frames=True), (act.element.site,)
        )
        if not any(e.frame == act.element.frame and e == act.element for e in now):
            raise StaleElementError(STALE["gone"], "gone")
        try:
            await self._call_with(
                "interact", {"interactions": [_interaction(act, uid, {})]}
            )
        except BrowserSetupError:
            raise
        except BrowserError as exc:
            if "stale" in str(exc).lower():
                raise StaleElementError(STALE["gone"], "gone") from None
            raise
        await asyncio.sleep(pagescript.QUIET_S)
        with contextlib.suppress(BrowserError):
            await self._ready()
        return ActResult(via="browser")

    async def _nodes(
        self, region: str = "viewport", frames: bool = False
    ) -> list[Node]:
        """The nodes the browser itself names in *region* (its own reading of the page), in order.

        With *frames* the reading goes into frames from other sites too (their
        nodes have ids like ``2_1_90``). The reading goes to a file melo names in
        its private folder (a long one would otherwise land in a file the server
        picks); only the node lines are taken from it.
        """
        path = os.path.join(self._folder(), f"{uuid.uuid4().hex}.txt")
        try:
            out = await self._call_with(
                "content",
                {
                    "format": "textTree",
                    "nodeIds": "interactive",
                    "region": region,
                    "includeSubframes": frames,
                    "includeURLs": False,
                    "maxWordsPerParagraph": 40,
                    "savePath": path,
                },
            )
            text = _read_text(path)
            if text is None:
                text = out.text  # a server that answered inline after all
        except BrowserSetupError:
            raise
        except BrowserError:
            return []
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)
        return parse_nodes(text)

    def _has(self, key: str) -> bool:
        return TOOL_NAMES[key] in self._tools


@dataclass(frozen=True)
class Node:
    """One node of the browser's own reading of a page (``get_page_content``, ``textTree``).

    ``uid`` is the browser's id for it (``2_1_90`` for one inside a frame from
    another site), ``tag`` its tag or role, ``attrs`` the ``name=value`` pairs
    the line shows, ``text`` its quoted text (cut with ``…`` when long) and
    ``depth`` how far it is indented.
    """

    uid: str
    tag: str
    attrs: dict[str, str]
    text: str
    depth: int


_NODE_LINE = re.compile(r"^(\t*)([A-Za-z][\w:-]*)\b(.*)$")
_NODE_ATTR = re.compile(r"([A-Za-z][\w-]*)=('(?:[^'\\]|\\.)*'|\S+)")
_NODE_TEXT = re.compile(r"'((?:[^'\\]|\\.)*)'\s*$")


def parse_nodes(reply: str) -> list[Node]:
    """The nodes with an id in a ``textTree`` reading (the reply, or its ``content``)."""
    data = decode(reply)
    text = data.get("content") if isinstance(data, dict) else reply
    nodes: list[Node] = []
    for line in str(text or "").split("\n"):
        match = _NODE_LINE.match(line)
        if match is None or "uid=" not in match.group(3):
            continue
        rest = match.group(3)
        attrs = {
            k: v[1:-1] if v.startswith("'") and v.endswith("'") else v
            for k, v in _NODE_ATTR.findall(rest)
        }
        uid = attrs.pop("uid", "")
        said = _NODE_TEXT.search(_NODE_ATTR.sub("", rest))
        if uid:
            nodes.append(
                Node(
                    uid,
                    match.group(2).lower(),
                    attrs,
                    said.group(1) if said else "",
                    len(match.group(1)),
                )
            )
        if len(nodes) >= MAX_NODES:
            break
    return nodes


_TEXT_TYPES = frozenset({"text", "email", "search", "tel", "url"})


def sites_origin(url: str) -> str:
    """``scheme://host[:port]`` of *url* in lower case, or ``""``."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    return (
        f"{parts.scheme}://{parts.netloc}".lower()
        if parts.scheme and parts.netloc
        else ""
    )


def foreign_elements(nodes: list[Node], boxes: tuple[str, ...]) -> list[Element]:
    """The controls the browser's reading names inside frames from the sites of *boxes*.

    A frame line says ``origin=host:port``; its controls are the deeper lines
    after it, with ids like ``2_1_90``. A field is a text box only when the
    reading names its type as plain text and no rule takes it for a secret;
    any other field counts as a secret one. At most 20 a frame.
    """
    known = {
        urlsplit(b).netloc.lower(): sites_origin(b) for b in boxes if sites_origin(b)
    }
    out: list[Element] = []
    frame: tuple[int, str] | None = None
    count = 0
    for node in nodes:
        if frame is not None and node.depth <= frame[0]:
            frame = None
        if node.tag == "iframe":
            site = known.get(node.attrs.get("origin", "").lower(), "")
            frame, count = ((node.depth, site), 0) if site else (None, 0)
            continue
        if frame is None or "_" not in node.uid or count >= 20:
            continue
        item = _foreign(node, frame[1])
        if item is not None:
            out.append(item)
            count += 1
    return out


def _foreign(node: Node, site: str) -> Element | None:
    """One control of a frame from another site, as an element melo lists; None for anything else."""
    words = " ".join(
        node.attrs.get(k, "")
        for k in ("label", "name", "id", "placeholder", "aria-label")
    ).strip()
    label = (node.attrs.get("label") or node.attrs.get("aria-label") or "").strip()

    def made(
        kind: ElementKind,
        name: str,
        sensitive: Sensitive = "",
        attrs: FieldAttrs | None = None,
    ) -> Element:
        return Element(
            kind,
            name,
            sensitive=sensitive,
            attrs=attrs,
            frame="x" + node.uid,
            site=site,
            fp="uid:" + node.uid,
        )

    if node.tag in ("input", "textarea"):
        kind = node.attrs.get("type", "text" if node.tag == "textarea" else "").lower()
        if kind in ("checkbox", "radio"):
            return made("checkbox", label or node.text)
        if kind in ("submit", "button"):
            return made("button", node.text or label)
        attrs = FieldAttrs(type=kind or "unknown", words=words)
        plain_text = kind in _TEXT_TYPES or node.tag == "textarea"
        secret = fields.sensitive_kind(attrs) or ("" if plain_text else "password")
        if secret:
            return made("textbox", "", sensitive=secret)
        return made("textbox", label or node.attrs.get("placeholder", ""), attrs=attrs)
    if node.tag in ("button", "a", "link") or node.attrs.get("role") in (
        "button",
        "link",
    ):
        name = node.text or label
        return made("button", name) if name else None
    return None


def _plain(words: Any) -> str:
    return " ".join(str(words or "").split()).lower()


def bind(described: dict[str, Any], nodes: list[Node]) -> str:
    """The browser's id of the element the page script *described*, or ``""``.

    It is the one node of the page itself (not of a frame) in view with the same
    tag or role and the same words (its text, value, label, ``aria-label``,
    title or placeholder; a cut text matches the start) or the same ``name`` or
    ``id``. None, or more than one, is no id: melo then acts by the page
    script. Where the browser's input lands is checked again at the act, on
    the element itself, so a wrong id is stopped, never used.
    """
    tag, role = str(described.get("tag") or ""), str(described.get("role") or "")
    words = {
        _plain(described.get(k))
        for k in ("text", "value", "label", "aria", "title", "placeholder")
    } - {""}
    names = {str(described.get(k) or "") for k in ("name", "id")} - {""}
    found: list[str] = []
    for node in nodes:
        if "_" in node.uid:
            continue  # a node inside a frame
        kinds = {node.tag, node.attrs.get("role", "")}
        if not (
            tag in kinds or (role and role in kinds) or (tag == "a" and "link" in kinds)
        ):
            continue
        said = [
            node.text,
            *(
                node.attrs.get(k, "")
                for k in ("label", "aria-label", "title", "placeholder")
            ),
        ]
        same = any(
            _plain(w).rstrip("…")
            and (
                _plain(w) in words
                or (
                    _plain(w).endswith("…")
                    and any(x.startswith(_plain(w)[:-1]) for x in words)
                )
            )
            for w in said
        )
        if same or {node.attrs.get("name", ""), node.attrs.get("id", "")} & names:
            found.append(node.uid)
    return found[0] if len(found) == 1 else ""


def _interaction(act: Act, uid: str, armed: dict[str, Any]) -> dict[str, Any]:
    """The one ``page_interactions`` step that does *act* on node *uid* (or at its point)."""
    if act.point is not None:
        # Safari's helper reads a point as if the screen had two pixels to a page
        # pixel: it is scaled by the screen's own ratio, so the click lands there.
        ratio = (act.view.ratio if act.view is not None else 2.0) / 2.0
        x, y = act.point
        return {
            "type": "click",
            "point": {"x": round(x * ratio, 1), "y": round(y * ratio, 1)},
            "purpose": "Click the checked point",
        }
    if act.kind == "type":
        return {
            "type": "type",
            "node": uid,
            "value": act.text,
            "replaceAll": True,
            "pressReturn": act.enter,
            "purpose": "Type the approved text",
        }
    if act.kind == "select":
        return {
            "type": "selectMenuItem",
            "node": uid,
            "text": str(armed.get("option") or act.text),
            "purpose": "Choose the approved option",
        }
    return {"type": "click", "node": uid, "purpose": "Click the checked element"}


def _read_text(path: str) -> str | None:
    """The text file at *path* that melo asked Safari to write, or None when there is none."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return None
        data = handle.read(MAX_CONTENT_BYTES)
    return data.decode("utf-8", errors="replace")


def _read_picture(path: str) -> Shot:
    """The PNG at *path*, the file melo asked Safari to save: a regular file, never a link."""
    problem = "Safari's screenshot tool saved no picture melo can use."
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise BrowserError(problem) from None
    with os.fdopen(fd, "rb") as handle:
        regular = stat.S_ISREG(os.fstat(handle.fileno()).st_mode)
        data = handle.read(MAX_SHOT_BYTES + 1) if regular else b""
    if not data.startswith(PNG_MAGIC) or len(data) > MAX_SHOT_BYTES:
        raise BrowserError(problem)
    return Shot("image/png", b64encode(data).decode("ascii"))


async def _quiet_close(client: StdioMcpClient) -> None:
    # The error being reported matters more than one from closing.
    with contextlib.suppress(Exception):
        await client.close()
