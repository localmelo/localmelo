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
import os
import re
import shutil
import stat
import tempfile
import uuid
from base64 import b64encode
from typing import Any, cast

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser.driver import (
    Act,
    BrowserError,
    BrowserSetupError,
    Element,
    FieldAttrs,
    RawPage,
    Sensitive,
    Shot,
    StaleElementError,
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
    "script": ("expression", "script", "javascript", "code", "source"),
    "tab": ("handle", "tab_uuid", "tab_id", "tabId", "tab", "id", "window", "index"),
    "save_path": ("savePath", "save_path", "path"),
}
# A picture is a PNG file Safari saves where melo says, and no larger than this.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_SHOT_BYTES = 10 * 1024 * 1024
KINDS = ("link", "button", "textbox", "checkbox", "select")
SECRETS = ("password", "code", "card")
SETTLE_S = 15.0
_STALE = {
    "page": "The page is not the one that was read (its address changed). Use browser_read for current refs.",
    "gone": "That element is no longer on the page as it was read. Use browser_read for current refs.",
    "field": "That field changed since it was read, so melo did not use it. Use browser_read for current refs.",
}
POLL_S = 0.4

# The page script's pieces, shared by reading a page and by acting on it, so
# an element is found again by the same rules that listed it. ``SEL`` lists
# what may be a target; ``lookAt`` describes one (or says why it is not one:
# ``skip`` 1 is counted as skipped, 0 is not worth counting). Nothing here
# reads cookies or storage, and nothing changes the page.
COMMON_JS = r"""
var SEL = "a[href],button,input,textarea,select,summary,[role=button],[role=link],[role=menuitem],[role=menuitemcheckbox],[role=menuitemradio],[role=tab],[role=switch],[role=option],[role=checkbox],[role=radio],[onclick],[tabindex]:not([tabindex='-1']),[class*=btn],[class*=button]";
var NESTED = "a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=menuitem],[role=tab]";
var cap = function (s, n) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim().slice(0, n); };
var attr = function (n, a) { return n.getAttribute(a) || ""; };
var inside = function (n, q) { return Element.prototype.querySelector.call(n, q); };
var formAddress = function (form, button) {
  var raw = button ? button.getAttribute("formaction") : null;
  if (raw === null || raw === undefined) raw = form.getAttribute("action");
  try { return new URL(raw === null || raw === "" ? location.href : raw, document.baseURI).href; }
  catch (e) { return location.href; }
};
var defaultButton = function (form) {
  return inside(form, "button:not([type]),button[type=submit],input[type=submit],input[type=image]");
};
var lookAt = function (n) {
  var tag = n.tagName.toLowerCase(), type = attr(n, "type").toLowerCase(), role = attr(n, "role").toLowerCase();
  var r = n.getBoundingClientRect(), st = getComputedStyle(n);
  if (st.visibility === "hidden" || st.display === "none" || (r.width === 0 && r.height === 0)) return {skip: 1};
  if (n.disabled || attr(n, "aria-disabled") === "true") return {skip: 1};
  var native = /^(a|button|input|textarea|select|summary)$/.test(tag);
  var known = /^(button|link|menuitem|menuitemcheckbox|menuitemradio|tab|switch|option|checkbox|radio)$/.test(role);
  if (!native && !known) {
    // A plain element that only looks clickable: it must say so, hold no other target,
    // and not sit inside one (the target is then the outer element).
    if (inside(n, SEL) || (n.parentElement && n.parentElement.closest(NESTED))) return {skip: 0};
    if (!attr(n, "onclick") && st.cursor !== "pointer") return {skip: 0};
  }
  var labels = n.labels ? Array.prototype.map.call(n.labels, function (l) { return cap(l.innerText || l.textContent, 80); }).join(" ") : "";
  var aria = cap(attr(n, "aria-label"), 120), ph = cap(attr(n, "placeholder"), 120);
  var title = cap(attr(n, "title") || attr(n, "alt"), 120);
  var shown = cap(n.innerText || n.textContent, 120);
  var el = {};
  if (tag === "select") {
    el.kind = "select";
    el.label = labels || aria || title || cap(n.name, 80);
    el.value = cap(n.options && n.selectedIndex >= 0 ? n.options[n.selectedIndex].text : "", 120);
    el.options = Array.prototype.slice.call(n.options || [], 0, 40).map(function (o) { return cap(o.text, 60); });
  } else if (tag === "input" && (type === "checkbox" || type === "radio")) {
    el.kind = "checkbox"; el.checked = !!n.checked;
    el.label = labels || aria || title || cap(n.name, 80);
  } else if (tag === "textarea" || (tag === "input" && !/^(button|submit|reset|image|hidden|file|checkbox|radio)$/.test(type))) {
    el.kind = "textbox"; el.value = cap(n.value, 120);
    el.label = labels || aria || ph || title || cap(n.name, 80);
    var max = parseInt(attr(n, "maxlength") || "0", 10);
    var ac = attr(n, "autocomplete").toLowerCase();
    var words = [attr(n, "name"), n.id, aria, ph, labels].join(" ");
    var masked = String(st.webkitTextSecurity || st.getPropertyValue("-webkit-text-security") || "none") !== "none";
    el.attrs = {type: tag === "textarea" ? "textarea" : type || "text", autocomplete: ac, words: cap(words, 300), inputmode: attr(n, "inputmode").toLowerCase(), maxlength: isNaN(max) ? 0 : max, masked: masked};
    if (type === "password" || masked || /(^|\s)(current|new)-password(\s|$)/.test(ac)) el.sensitive = "password";
    else if (/(^|\s)one-time-code(\s|$)/.test(ac) || /otp|one.?time|verif|验证码/i.test(words) || (el.attrs.inputmode === "numeric" && max >= 4 && max <= 8)) el.sensitive = "code";
    else if (/(^|\s)cc-/.test(ac)) el.sensitive = "card";
    if (n.form && !el.sensitive) {
      var db = defaultButton(n.form);
      el.submit = formAddress(n.form, db);
    }
  } else if (tag === "input" || tag === "button") {
    el.kind = "button";
    el.label = tag === "input" ? cap(n.value, 120) || aria || title : shown || aria || title || cap(n.name, 80);
    if (n.form && (tag === "input" ? /^(submit|image)$/.test(type) : (type === "" || type === "submit"))) el.submit = formAddress(n.form, n);
  } else if (tag === "a" || role === "link") {
    var href = tag === "a" ? (typeof n.href === "string" ? n.href : attr(n, "href")) : "";
    if (/^https?:/i.test(href)) { el.kind = "link"; el.href = href; }
    else if (/^(mailto|tel|sms|file|blob|data|ftp):/i.test(href)) return {skip: 1};
    else el.kind = "button";  // javascript: and empty addresses are clicked, not opened
    el.label = shown || aria || title;
  } else if (/^(checkbox|switch|radio|menuitemcheckbox|menuitemradio)$/.test(role)) {
    el.kind = "checkbox"; el.checked = attr(n, "aria-checked") === "true";
    el.label = shown || aria || title;
  } else {
    el.kind = "button";
    el.label = shown || aria || title;
  }
  var other = aria && aria !== el.label ? aria : (title && title !== el.label ? title : "");
  if (other) el.also = other;
  if (el.sensitive) { el.value = ""; el.label = ""; el.also = ""; }
  return {el: el};
};
"""

# Returns JSON: the page's text (the main part when there is one) and its
# visible links, buttons and fields, each with its place (``at``) among the
# page's possible targets. Secret fields are marked, their values never read.
PAGE_JS = r"""(function () {""" + COMMON_JS + r"""
  var out = {url: location.href, title: document.title || "", text: "", elements: [], skipped: 0};
  var main = document.querySelector("main,[role=main],article");
  var root = main && (main.innerText || "").length > 300 ? main : document.body;
  out.text = root ? String(root.innerText || "").slice(0, 200000) : "";
  var nodes = document.querySelectorAll(SEL);
  for (var i = 0; i < nodes.length && i < 4000; i++) {
    var d = lookAt(nodes[i]);
    if (!d.el) { out.skipped += d.skip; continue; }
    if (out.elements.length >= 400) { out.skipped += 1; continue; }
    d.el.at = i;
    out.elements.push(d.el);
  }
  return JSON.stringify(out);
})()"""

MAX_TYPED_CHARS = 2000


def act_script(spec: dict[str, Any]) -> str:
    """The script that does one :class:`Act`, given its *spec* as plain data.

    It refuses, and does nothing, unless the page still has the address it was
    read at and the element at ``at`` is the one that was read (same kind,
    label and address, and for a text box the same field attributes).
    """
    data = json.dumps(spec).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return r"""(function (spec) {""" + COMMON_JS + r"""
  var say = function (o) { return JSON.stringify(o); };
  if (location.href !== spec.url) return say({ok: false, stale: "page"});
  var n = document.querySelectorAll(SEL)[spec.at];
  var d = n ? lookAt(n) : {};
  var el = d.el;
  if (!el || el.kind !== spec.kind || el.label !== spec.label || (el.href || "") !== spec.href) return say({ok: false, stale: "gone"});
  if (spec.act === "type") {
    var a = el.attrs, b = spec.attrs || {};
    if (el.kind !== "textbox" || !a || a.type !== b.type || a.autocomplete !== b.autocomplete || a.words !== b.words || a.inputmode !== b.inputmode || a.maxlength !== b.maxlength || a.masked !== b.masked) return say({ok: false, stale: "field"});
    if (el.sensitive) return say({ok: false, stale: "field"});
  }
  if (spec.act !== "type" && el.sensitive) return say({ok: false, stale: "field"});
  n.scrollIntoView({block: "center", inline: "center"});
  var fire = function (type) { n.dispatchEvent(new Event(type, {bubbles: true})); };
  if (spec.act === "click") {
    ["mousedown", "mouseup"].forEach(function (t) { n.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window})); });
    n.click();
  } else if (spec.act === "type") {
    n.focus();
    var proto = n.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(n, spec.text);
    fire("input"); fire("change");
    if (spec.enter) {
      var key = function (t) { return new KeyboardEvent(t, {key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true}); };
      var go = n.dispatchEvent(key("keydown"));
      n.dispatchEvent(key("keypress"));
      n.dispatchEvent(key("keyup"));
      if (go && n.form) HTMLFormElement.prototype.requestSubmit.call(n.form, defaultButton(n.form) || undefined);
    }
  } else if (spec.act === "select") {
    var want = String(spec.text).trim().toLowerCase(), opts = Array.prototype.slice.call(n.options);
    var hits = opts.filter(function (o) { return !o.disabled && (cap(o.text, 60).toLowerCase() === want || String(o.value).toLowerCase() === want); });
    if (!hits.length) hits = opts.filter(function (o) { return !o.disabled && want && cap(o.text, 60).toLowerCase().indexOf(want) >= 0; });
    if (hits.length !== 1) return say({ok: false, option: hits.length});
    n.value = hits[0].value; hits[0].selected = true;
    fire("input"); fire("change");
  }
  return say({ok: true});
})(""" + data + ")"


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


def _attrs(item: Any) -> FieldAttrs | None:
    """The :class:`FieldAttrs` the page script read for a text box, or None."""
    if not isinstance(item, dict):
        return None
    limit = item.get("maxlength")
    return FieldAttrs(
        type=str(item.get("type") or ""),
        autocomplete=str(item.get("autocomplete") or ""),
        words=str(item.get("words") or ""),
        inputmode=str(item.get("inputmode") or ""),
        maxlength=limit if isinstance(limit, int) and limit > 0 else 0,
        masked=item.get("masked") is True,
    )


def _element(item: Any) -> Element | None:
    """An :class:`Element` from the page script's JSON; None for a kind melo ignores."""
    if not isinstance(item, dict) or item.get("kind") not in KINDS:
        return None
    checked, secret = item.get("checked"), item.get("sensitive")
    options, place = item.get("options"), item.get("at")
    return Element(
        item["kind"],
        str(item.get("label") or ""),
        href=str(item.get("href") or ""),
        value=str(item.get("value") or ""),
        checked=checked if isinstance(checked, bool) else None,
        sensitive=cast(Sensitive, secret) if secret in SECRETS else "",
        also=str(item.get("also") or ""),
        submit=str(item.get("submit") or ""),
        options=tuple(str(o) for o in options) if isinstance(options, list) else (),
        attrs=_attrs(item.get("attrs")),
        at=place if isinstance(place, int) else -1,
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
        self._shots: str | None = (
            None  # melo's private folder for pictures Safari saves
        )

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

    async def open(self, url: str, *, new_tab: bool, tab: str | None = None) -> TabInfo:
        sent = False
        if new_tab and self._has("new_tab"):
            tool = self._tools[TOOL_NAMES["new_tab"]]
            props = tool.input_schema.get("properties")
            sent = isinstance(props, dict) and "url" in props
            await self._call("new_tab", **({"url": url} if sent else {}))
            self._current = None
        if not sent:
            if not new_tab:
                await self._focus(tab)
            await self._call("open", url=url)
        await self._ready()
        return await self._here()

    async def page(self, tab: str | None) -> RawPage:
        await self._focus(tab)
        data = await self._eval(PAGE_JS)
        if not isinstance(data, dict):
            raise BrowserError("Safari returned a page melo could not read.")
        elements = [e for e in map(_element, data.get("elements") or []) if e]
        skipped = data.get("skipped")
        return RawPage(
            str(data.get("url") or ""),
            str(data.get("title") or ""),
            str(data.get("text") or ""),
            tuple(elements),
            skipped=skipped if isinstance(skipped, int) and skipped > 0 else 0,
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

    async def location(self, tab: str | None) -> str:
        await self._focus(tab)
        where = await self._eval("location.href")
        return where if isinstance(where, str) else ""

    async def wait(
        self,
        tab: str | None,
        until: str,
        text: str,
        seconds: float,
        *,
        origin: str = "",
    ) -> bool:
        await self._focus(tab)
        probe = {
            "load": "document.readyState === 'complete'",
            "text": f"(document.body.innerText || '').indexOf({json.dumps(text)}) >= 0",
            "gone": f"(document.body.innerText || '').indexOf({json.dumps(text)}) < 0",
        }[until]
        if origin:
            probe = f"location.origin === {json.dumps(origin)} && ({probe})"
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
        # Safari saves the picture to a file and answers with text only. melo
        # names the file, in a folder only it can enter, reads that path and
        # never one it finds in the reply, and deletes the file whatever happens.
        if self._shots is None:
            self._shots = tempfile.mkdtemp(prefix="melo-safari-")  # mode 0700
        path = os.path.join(self._shots, f"{uuid.uuid4().hex}.png")
        try:
            out = await self._call("shot", save_path=path)
            if out.images:
                return Shot(out.images[0].mime_type, out.images[0].data)
            return _read_picture(path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)  # a link is removed, never followed

    async def blank(self, tab: str | None) -> None:
        await self._focus(tab)
        await self._eval("(location.replace('about:blank'), true)")

    async def act(self, tab: str | None, act: Act) -> None:
        await self._focus(tab)
        item = act.element
        spec: dict[str, Any] = {
            "act": act.kind,
            "url": act.url,
            "at": item.at,
            "kind": item.kind,
            "label": item.label,
            "href": item.href,
            "text": act.text[:MAX_TYPED_CHARS],
            "enter": act.enter,
        }
        if item.attrs is not None:
            spec["attrs"] = {
                "type": item.attrs.type,
                "autocomplete": item.attrs.autocomplete,
                "words": item.attrs.words,
                "inputmode": item.attrs.inputmode,
                "maxlength": item.attrs.maxlength,
                "masked": item.attrs.masked,
            }
        result = await self._eval(act_script(spec))
        if not isinstance(result, dict):
            raise BrowserError("Safari returned an answer melo could not read.")
        if result.get("ok") is not True:
            if result.get("stale"):
                raise StaleElementError(_STALE[str(result["stale"])])
            if "option" in result:
                raise BrowserError(
                    "That list has no single option with those words (none, or "
                    "more than one). Use browser_read for its options."
                )
            raise BrowserError("The page did not take that action.")
        await asyncio.sleep(
            POLL_S
        )  # a click may start a load that is not yet under way
        with contextlib.suppress(BrowserError):  # the page may have closed itself
            await self._ready()

    def _has(self, key: str) -> bool:
        return TOOL_NAMES[key] in self._tools


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
