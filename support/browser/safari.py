"""The ``safari-automation`` driver: Safari's own MCP server, ``safaridriver --mcp``.

It runs in Safari's isolated automation windows: a clean slate like a private
window, no saved logins, one session at a time, and a visible orange address
bar. Nothing is installed; Safari needs two switches turned on once (see
:data:`SETUP_MESSAGE`). The server is started on first use, with the stdio MCP
client the rest of melo uses.

Pages are read with ``evaluate_javascript``: one script returns the page's text
and elements as JSON (:data:`PAGE_JS`), so what the model reads does not depend
on the format of the server's own content tool. Tool and argument names are
looked up in the server's ``tools/list`` (:data:`TOOL_NAMES`, :data:`ARG_NAMES`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any, cast

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser.driver import (
    BrowserError,
    BrowserSetupError,
    Element,
    RawPage,
    Sensitive,
    Shot,
    TabInfo,
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
}
REQUIRED = ("open", "script")
# The argument names a tool may use for what melo sends, first match wins; a
# tool with one parameter gets the value there whatever it is called.
ARG_NAMES = {
    "url": ("url", "uri", "address"),
    "script": ("script", "javascript", "code", "expression", "source"),
    "tab": ("tab_id", "tabId", "tab", "id", "handle", "window", "index"),
}
KINDS = ("link", "button", "textbox", "checkbox", "select")
SECRETS = ("password", "code", "card")
SETTLE_S = 15.0
POLL_S = 0.4

# Returns JSON: the page's text (the main part when there is one) and its
# visible links, buttons and fields. Secret fields are marked, their values never read.
PAGE_JS = r"""(function () {
  var cap = function (s, n) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim().slice(0, n); };
  var out = {url: location.href, title: document.title || "", text: "", elements: []};
  var main = document.querySelector("main,[role=main],article");
  var root = main && (main.innerText || "").length > 300 ? main : document.body;
  out.text = root ? String(root.innerText || "").slice(0, 200000) : "";
  var nodes = document.querySelectorAll("a[href],button,input,textarea,select,[role=button],[role=link]");
  for (var i = 0; i < nodes.length && out.elements.length < 400; i++) {
    var n = nodes[i], r = n.getBoundingClientRect(), st = getComputedStyle(n);
    if (st.visibility === "hidden" || st.display === "none" || (r.width === 0 && r.height === 0)) continue;
    var tag = n.tagName.toLowerCase(), type = (n.getAttribute("type") || "").toLowerCase();
    var ac = (n.getAttribute("autocomplete") || "").toLowerCase();
    var name = (n.getAttribute("name") || "") + " " + (n.id || "") + " " + (n.getAttribute("aria-label") || "");
    var label = cap(n.getAttribute("aria-label") || n.innerText || n.getAttribute("placeholder") || n.getAttribute("title") || n.getAttribute("alt") || n.value || n.name, 120);
    var el = {label: label};
    if (tag === "a" || n.getAttribute("role") === "link") {
      if (!/^https?:/i.test(n.href || "")) continue;
      el.kind = "link"; el.href = n.href;
    } else if (tag === "button" || n.getAttribute("role") === "button" || (tag === "input" && /^(button|submit|reset|image)$/.test(type))) {
      el.kind = "button";
    } else if (tag === "select") {
      el.kind = "select"; el.value = cap(n.options && n.selectedIndex >= 0 ? n.options[n.selectedIndex].text : "", 120);
    } else if (tag === "input" && (type === "checkbox" || type === "radio")) {
      el.kind = "checkbox"; el.checked = !!n.checked;
    } else if (tag === "textarea" || (tag === "input" && type !== "hidden" && type !== "file")) {
      el.kind = "textbox"; el.value = cap(n.value, 120);
      var max = parseInt(n.getAttribute("maxlength") || "0", 10);
      if (type === "password" || ac === "current-password" || ac === "new-password") el.sensitive = "password";
      else if (ac === "one-time-code" || /otp|one.?time|verif|验证码/i.test(name) || (n.getAttribute("inputmode") === "numeric" && max >= 4 && max <= 8)) el.sensitive = "code";
      else if (ac.indexOf("cc-") === 0) el.sensitive = "card";
    } else { continue; }
    if (el.sensitive) { el.value = ""; el.label = ""; }
    out.elements.push(el);
  }
  return JSON.stringify(out);
})()"""

FOCUS_JS = r"""(function () {
  var a = document.activeElement;
  if (!a || a.tagName !== "INPUT") return false;
  var t = (a.getAttribute("type") || "").toLowerCase(), ac = (a.getAttribute("autocomplete") || "").toLowerCase();
  return t === "password" || ac === "one-time-code" || ac.indexOf("cc-") === 0 || ac.indexOf("-password") > 0;
})()"""


def _decode(text: str) -> Any:
    """The value a script returned: JSON, possibly quoted or wrapped in ``{"result": ...}``."""
    value: Any = text.strip()
    for _ in range(4):
        if isinstance(value, dict) and "result" in value:
            value = value["result"]
        elif isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                break
        else:
            break
    return value


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


def _element(item: Any) -> Element | None:
    """An :class:`Element` from the page script's JSON; None for a kind melo ignores."""
    if not isinstance(item, dict) or item.get("kind") not in KINDS:
        return None
    checked, secret = item.get("checked"), item.get("sensitive")
    return Element(
        item["kind"],
        str(item.get("label") or ""),
        href=str(item.get("href") or ""),
        value=str(item.get("value") or ""),
        checked=checked if isinstance(checked, bool) else None,
        sensitive=cast(Sensitive, secret) if secret in SECRETS else "",
    )


def _parse_tabs(text: str) -> list[tuple[TabInfo, bool]]:
    """Tabs (and whether each is the active one) from a ``list_tabs`` reply."""
    value = _decode(text)
    items = value.get("tabs", []) if isinstance(value, dict) else value
    if not isinstance(items, list):  # plain lines: one tab each, by position
        items = [line for line in text.splitlines() if line.strip()]
    tabs = []
    for index, item in enumerate(items):
        row = item if isinstance(item, dict) else {"title": str(item)}
        ident = next(
            (row[k] for k in (*ARG_NAMES["tab"], "windowHandle") if k in row), index
        )
        active = any(row.get(k) for k in ("active", "current", "selected"))
        info = TabInfo(
            str(ident), str(row.get("url") or ""), str(row.get("title") or "")
        )
        tabs.append((info, active))
    return tabs


class SafariAutomationDriver:
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
        try:
            async with asyncio.timeout(self._call_timeout_s):
                result = await self._client.call_tool(
                    tool.name, self._args(tool, **values)
                )
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
                if _decode(probe.text) == 2:
                    self._returns = returns
                    break
        out = await self._call(
            "script", script=("return " + source if self._returns else source)
        )
        return _decode(out.text)

    async def _focus(self, tab: str | None) -> None:
        if tab and tab != "current" and tab != self._current:
            await self._call("switch", tab=tab)
            self._current = tab

    async def _ready(self) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + SETTLE_S
        while loop.time() < end:
            if await self._eval("document.readyState") == "complete":
                return
            await asyncio.sleep(POLL_S)

    async def _here(self) -> TabInfo:
        where = await self._eval(
            "JSON.stringify({url: location.href, title: document.title})"
        )
        info = where if isinstance(where, dict) else {}
        tab = self._current
        if self._has("tabs"):
            try:
                tabs = _parse_tabs((await self._call("tabs")).text)
            except BrowserError:
                tabs = []
            tab = next(
                (t.id for t, active in tabs if active), tabs[-1][0].id if tabs else tab
            )
        self._current = tab or "current"
        return TabInfo(
            self._current, str(info.get("url") or ""), str(info.get("title") or "")
        )

    # ── The driver ──

    async def open(self, url: str, *, new_tab: bool) -> TabInfo:
        sent = False
        if new_tab and self._has("new_tab"):
            tool = self._tools[TOOL_NAMES["new_tab"]]
            props = tool.input_schema.get("properties")
            sent = isinstance(props, dict) and "url" in props
            await self._call("new_tab", **({"url": url} if sent else {}))
            self._current = None
        if not sent:
            await self._call("open", url=url)
        await self._ready()
        return await self._here()

    async def page(self, tab: str | None) -> RawPage:
        await self._focus(tab)
        data = await self._eval(PAGE_JS)
        if not isinstance(data, dict):
            raise BrowserError("Safari returned a page melo could not read.")
        elements = [e for e in map(_element, data.get("elements") or []) if e]
        return RawPage(
            str(data.get("url") or ""),
            str(data.get("title") or ""),
            str(data.get("text") or ""),
            tuple(elements),
        )

    async def back(self, tab: str | None) -> TabInfo:
        await self._focus(tab)
        await self._eval("(history.back(), true)")
        await asyncio.sleep(POLL_S)
        await self._ready()
        return await self._here()

    async def tabs(self) -> list[TabInfo]:
        return [tab for tab, _ in _parse_tabs((await self._call("tabs")).text)]

    async def switch(self, tab: str) -> TabInfo:
        await self._call("switch", tab=tab)
        self._current = tab
        return await self._here()

    async def close_tab(self, tab: str) -> None:
        await self._call("close", tab=tab)
        if self._current == tab:
            self._current = None

    async def wait(
        self, tab: str | None, until: str, text: str, seconds: float
    ) -> bool:
        await self._focus(tab)
        probe = {
            "load": "document.readyState === 'complete'",
            "text": f"(document.body.innerText || '').indexOf({json.dumps(text)}) >= 0",
            "gone": f"(document.body.innerText || '').indexOf({json.dumps(text)}) < 0",
        }[until]
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while True:
            if await self._eval(probe) is True:
                return True
            if loop.time() >= end:
                return False
            await asyncio.sleep(POLL_S)

    async def screenshot(self, tab: str | None) -> Shot:
        await self._focus(tab)
        if await self._eval(FOCUS_JS) is True:
            raise BrowserError(
                "A password, code or card field has focus, so melo took no picture."
            )
        out = await self._call("shot")
        if out.images:
            return Shot(out.images[0].mime_type, out.images[0].data)
        raise BrowserError("Safari's screenshot tool returned no picture melo can use.")

    async def blank(self, tab: str | None) -> None:
        await self._focus(tab)
        await self._eval("(location.replace('about:blank'), true)")

    def _has(self, key: str) -> bool:
        return TOOL_NAMES[key] in self._tools


async def _quiet_close(client: StdioMcpClient) -> None:
    # The error being reported matters more than one from closing.
    with contextlib.suppress(Exception):
        await client.close()
