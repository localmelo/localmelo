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

A tab is the handle the server gave it. A tab's address is what the server's own
tab list reports for that handle (a bare ``location.href`` is read only when the
list carries none), never a value a page serialises for itself. Every call
switches to the tab it means. A new tab is the one handle that appeared between
two tab lists; if not exactly one did, nothing is used.
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
from localmelo.support.browser import fields
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
    "doc": "The page was reloaded or replaced since it was read. Use browser_read for current refs.",
    "gone": "That element is no longer on the page as it was read. Use browser_read for current refs.",
    "dest": "Where that element leads or sends changed since it was read, so melo did not use it. Use browser_read for current refs.",
    "field": "That field changed since it was read, so melo did not use it. Use browser_read for current refs.",
    "grew": "That field turned into a protected one while melo typed, so melo cleared it. Ask the person to fill it in themselves.",
}
_BLOCKED = {
    "send": "The page tried to send a form that melo had not asked about, so melo stopped it. To send a form, click its button: melo asks first.",
    "link": "The page changed where that link goes while it was clicked, so melo stopped it.",
}
POLL_S = 0.4

# The page script's pieces, shared by reading a page and by acting on it, so
# an element is found again by the same rules that listed it. ``SEL`` lists
# what may be a target; ``lookAt`` describes one (or says why it is not one:
# ``skip`` 1 is counted as skipped, 0 is not worth counting). Nothing here
# reads cookies or storage, and nothing changes the page.
#
# ``P`` holds the page-reading built-ins taken from a fresh, hidden frame, so
# a page that replaced ``getAttribute`` or ``JSON.stringify`` for itself does
# not change what these checks see. When the frame cannot be made, or one of
# its functions does not work, the page's own are used.
COMMON_JS = (
    r"""
var P = (function () {
  var own = {
    qsa: Document.prototype.querySelectorAll, qs: Element.prototype.querySelector,
    ga: Element.prototype.getAttribute, ra: Element.prototype.removeAttribute,
    json: JSON.stringify, gcs: window.getComputedStyle, add: EventTarget.prototype.addEventListener,
    req: HTMLFormElement.prototype.requestSubmit, sub: HTMLFormElement.prototype.submit,
    click: HTMLElement.prototype.click, dispatch: EventTarget.prototype.dispatchEvent,
    rm: EventTarget.prototype.removeEventListener
  };
  var frame = null;
  try {
    frame = document.createElement("iframe");
    frame.setAttribute("style", "display:none");
    (document.body || document.documentElement).appendChild(frame);
    var w = frame.contentWindow;
    var fresh = {
      qsa: w.Document.prototype.querySelectorAll, qs: w.Element.prototype.querySelector,
      ga: w.Element.prototype.getAttribute, ra: w.Element.prototype.removeAttribute,
      json: w.JSON.stringify, gcs: w.getComputedStyle.bind(window), add: w.EventTarget.prototype.addEventListener,
      req: w.HTMLFormElement.prototype.requestSubmit, sub: w.HTMLFormElement.prototype.submit,
      click: w.HTMLElement.prototype.click, dispatch: w.EventTarget.prototype.dispatchEvent,
      rm: w.EventTarget.prototype.removeEventListener
    };
    // The fresh functions are tried on their own terms, never against the page's.
    var probe = w.document.createElement("div");
    w.Element.prototype.setAttribute.call(probe, "lang", "x");
    if (fresh.ga.call(probe, "lang") !== "x" || fresh.json({a: 1}) !== '{"a":1}' ||
        typeof fresh.qsa.call(document, "html").length !== "number" || typeof fresh.gcs(document.documentElement).display !== "string") throw new Error("frame");
    for (var k in fresh) if (typeof fresh[k] !== "function") throw new Error(k);
    return fresh;
  } catch (e) {
    return own;
  } finally {
    if (frame && frame.parentNode) frame.parentNode.removeChild(frame);
  }
})();
var SEL = "a[href],button,input,textarea,select,summary,[role=button],[role=link],[role=menuitem],[role=menuitemcheckbox],[role=menuitemradio],[role=tab],[role=switch],[role=option],[role=checkbox],[role=radio],[onclick],[tabindex]:not([tabindex='-1']),[class*=btn],[class*=button]";
var NESTED = "a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=menuitem],[role=tab]";
var cap = function (s, n) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim().slice(0, n); };
var attr = function (n, a) { return P.ga.call(n, a) || ""; };
var inside = function (n, q) { return P.qs.call(n, q); };
var hrefOf = function (a) { return typeof a.href === "string" ? a.href : attr(a, "href"); };  // an SVG link has no string href
"""
    + fields.JS_RULES
    + r"""
var docId = function () { return String(performance.timeOrigin); };
var formAddress = function (form, button) {
  var raw = button ? P.ga.call(button, "formaction") : null;
  if (raw === null || raw === undefined) raw = P.ga.call(form, "action");
  try { return new URL(raw === null || raw === "" ? location.href : raw, document.baseURI).href; }
  catch (e) { return location.href; }
};
var defaultButton = function (form) {
  return inside(form, "button:not([type]),button[type=submit],input[type=submit],input[type=image]");
};
var textOf = function (id) { var t = document.getElementById(id); return t ? cap(t.textContent, 80) : ""; };
var refsText = function (n, a) { return attr(n, a).split(/\s+/).filter(Boolean).slice(0, 4).map(textOf).join(" "); };
var legendOf = function (n) {
  var f = n.closest ? n.closest("fieldset") : null, l = f ? inside(f, "legend") : null;
  return l ? cap(l.textContent, 80) : "";
};
var textBefore = function (n) {
  var p = n.previousSibling, t = "";
  while (p && !t) {
    if (p.nodeType === 3 || (p.nodeType === 1 && !/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(p.tagName))) t = cap(p.textContent, 60);
    p = p.previousSibling;
  }
  return t;
};
var labelsOf = function (n) {
  return n.labels ? Array.prototype.map.call(n.labels, function (l) { return cap(l.innerText || l.textContent, 80); }).join(" ") : "";
};
var fingerprint = function (n) {
  var parts = [], cur = n, depth = 0;
  while (cur && cur.nodeType === 1 && depth < 12) {
    var i = 1, sib = cur;
    while ((sib = sib.previousElementSibling)) if (sib.tagName === cur.tagName) i++;
    parts.push(cur.tagName.toLowerCase() + (i > 1 ? ":" + i : ""));
    cur = cur.parentElement; depth++;
  }
  return [n.tagName.toLowerCase(), attr(n, "type").toLowerCase(), attr(n, "role").toLowerCase(), cap(n.id, 40), cap(attr(n, "name"), 40), parts.reverse().join(">")].join("|");
};
var attrsOf = function (n, st, labels, aria, ph) {
  var tag = n.tagName.toLowerCase(), type = attr(n, "type").toLowerCase();
  var max = parseInt(attr(n, "maxlength") || "0", 10);
  var words = [attr(n, "name"), n.id, aria, attr(n, "title"), ph, labels, refsText(n, "aria-labelledby"), refsText(n, "aria-describedby"), legendOf(n), textBefore(n)].join(" ");
  var masked = String(st.webkitTextSecurity || st.getPropertyValue("-webkit-text-security") || "none") !== "none";
  return {type: tag === "textarea" ? "textarea" : tag === "select" ? "select" : type || "text", autocomplete: attr(n, "autocomplete").toLowerCase(), words: cap(words, 400), inputmode: attr(n, "inputmode").toLowerCase(), maxlength: isNaN(max) ? 0 : max, masked: masked};
};
var formFields = function (form) {
  var out = [], list = form.elements;
  for (var i = 0; i < list.length && out.length < 12; i++) {
    var f = list[i], tag = f.tagName.toLowerCase(), type = attr(f, "type").toLowerCase();
    if (/^(button|submit|reset|image|file)$/.test(type) || /^(fieldset|button|object|output)$/.test(tag)) continue;
    var name = cap(attr(f, "name") || f.id || attr(f, "aria-label"), 40);
    if (!name) continue;
    if (type === "hidden") { out.push([name, "", "h"]); continue; }
    var secret = secretKind(attrsOf(f, P.gcs(f), labelsOf(f), cap(attr(f, "aria-label"), 120), cap(attr(f, "placeholder"), 120)));
    var v = tag === "select" ? cap(f.options && f.selectedIndex >= 0 ? f.options[f.selectedIndex].text : "", 80)
      : (type === "checkbox" || type === "radio") ? (f.checked ? "on" : "off") : cap(f.value, 80);
    out.push([name, secret ? "" : v, secret ? "s" : ""]);
  }
  return out;
};
var lookAt = function (n) {
  var tag = n.tagName.toLowerCase(), type = attr(n, "type").toLowerCase(), role = attr(n, "role").toLowerCase();
  var r = n.getBoundingClientRect(), st = P.gcs(n);
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
  var labels = labelsOf(n);
  var aria = cap(attr(n, "aria-label"), 120), ph = cap(attr(n, "placeholder"), 120);
  var title = cap(attr(n, "title") || attr(n, "alt"), 120);
  var shown = cap(n.innerText || n.textContent, 120);
  var el = {};
  if (tag === "select") {
    el.kind = "select";
    el.label = labels || aria || title || cap(n.name, 80);
    el.value = cap(n.options && n.selectedIndex >= 0 ? n.options[n.selectedIndex].text : "", 120);
    el.options = Array.prototype.slice.call(n.options || [], 0, 40).map(function (o) { return cap(o.text, 60); });
    el.attrs = attrsOf(n, st, labels, aria, ph);
    var picked = secretKind(el.attrs);
    if (picked) el.sensitive = picked;
  } else if (tag === "input" && (type === "checkbox" || type === "radio")) {
    el.kind = "checkbox"; el.checked = !!n.checked;
    el.label = labels || aria || title || cap(n.name, 80);
  } else if (tag === "textarea" || (tag === "input" && !/^(button|submit|reset|image|hidden|file|checkbox|radio)$/.test(type))) {
    el.kind = "textbox"; el.value = cap(n.value, 120);
    el.label = labels || aria || ph || title || cap(n.name, 80);
    el.attrs = attrsOf(n, st, labels, aria, ph);
    var kind = secretKind(el.attrs);
    if (kind) el.sensitive = kind;
    if (n.form && !el.sensitive) {
      el.submit = formAddress(n.form, defaultButton(n.form));
      el.form = formFields(n.form);
    }
  } else if (tag === "input" || tag === "button") {
    el.kind = "button";
    el.label = tag === "input" ? cap(n.value, 120) || aria || title : shown || aria || title || cap(n.name, 80);
    if (n.form && (tag === "input" ? /^(submit|image)$/.test(type) : (type === "" || type === "submit"))) {
      el.submit = formAddress(n.form, n);
      el.form = formFields(n.form);
    }
  } else if (tag === "a" || role === "link") {
    var href = tag === "a" ? hrefOf(n) : "";
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
  if (el.sensitive) { el.value = ""; el.label = ""; el.also = ""; el.options = []; el.form = []; }
  el.fp = fingerprint(n);
  return {el: el};
};
"""
)

# Returns JSON: the page's text (the main part when there is one) and its
# visible links, buttons and fields, each with its place (``at``) among the
# page's possible targets. Secret fields are marked, their values never read.
PAGE_JS = r"""(function () {""" + COMMON_JS + r"""
  var out = {url: location.href, doc: docId(), title: document.title || "", text: "", elements: [], skipped: 0, frames: []};
  var main = document.querySelector("main,[role=main],article");
  var root = main && (main.innerText || "").length > 300 ? main : document.body;
  out.text = root ? String(root.innerText || "").slice(0, 200000) : "";
  var nodes = P.qsa.call(document, SEL);
  for (var i = 0; i < nodes.length && i < 4000; i++) {
    var d = lookAt(nodes[i]);
    if (!d.el) { out.skipped += d.skip; continue; }
    if (out.elements.length >= 400) { out.skipped += 1; continue; }
    d.el.at = i;
    out.elements.push(d.el);
  }
  var frames = P.qsa.call(document, "iframe,frame");
  for (var k = 0; k < frames.length && k < 20; k++) {
    var u = "";
    try { u = String(frames[k].contentWindow.location.href); } catch (e) { u = ""; }
    if (!u || u === "about:blank") u = String(frames[k].src || "");
    if (u) out.frames.push(u.slice(0, 500));
  }
  return P.json(out);
})()"""

MAX_TYPED_CHARS = 2000


def act_script(spec: dict[str, Any]) -> str:
    """The script that does one :class:`Act`, given its *spec* as plain data.

    It does nothing, and says why, unless the page is the one that was read:
    the same address in the same document, and at ``at`` the same element (its
    fingerprint, kind, label, address and the address its form is sent to, and
    for a text box the same field attributes). It looks again after the field
    has focus, after the text is in and after the page's own handlers ran, and
    clears the field if it turned into a secret one. While it works, a form
    the page sends that melo did not ask about is stopped, and so is a link
    whose address is no longer the one that was read when it is pressed or
    clicked. It guards only while it runs.
    """
    data = json.dumps(spec).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return r"""(function (spec) {""" + COMMON_JS + r"""
  var say = function (o) { return P.json(o); };
  var blocked = [];
  if (location.href !== spec.url) return say({ok: false, stale: "page"});
  if (docId() !== spec.doc) return say({ok: false, stale: "doc"});
  var n = P.qsa.call(document, SEL)[spec.at];
  var same = function () {
    var d = n ? lookAt(n) : {}, el = d.el;
    if (!el || el.fp !== spec.fp || el.kind !== spec.kind || el.label !== spec.label || (el.href || "") !== spec.href) return "gone";
    if ((el.submit || "") !== spec.submit) return "dest";
    if (spec.attrs) {
      var a = el.attrs, b = spec.attrs, keys = ["type", "autocomplete", "words", "inputmode", "maxlength", "masked"];
      if (!a || keys.some(function (k) { return a[k] !== b[k]; })) return "field";
    }
    if (el.sensitive) return "field";
    return "";
  };
  var why = same();
  if (why) return say({ok: false, stale: why});
  var allowed = function (form, button) {
    return spec.send && (spec.submit === "" || formAddress(form, button) === spec.submit);
  };
  var onSubmit = function (e) {
    if (!allowed(e.target, e.submitter)) { e.preventDefault(); e.stopImmediatePropagation(); blocked.push("send"); }
  };
  var onClick = function (e) {
    var t = e.target, a = t && t.closest ? t.closest("a[href]") : null;
    if (spec.kind === "link" && a && hrefOf(a) !== spec.href) { e.preventDefault(); e.stopImmediatePropagation(); blocked.push("link"); }
  };
  var proto = HTMLFormElement.prototype;
  var saved = {submit: Object.getOwnPropertyDescriptor(proto, "submit"), requestSubmit: Object.getOwnPropertyDescriptor(proto, "requestSubmit")};
  var put = function (name, fn) { Object.defineProperty(proto, name, {value: fn, configurable: true, writable: true}); };
  var done = function (o) { blocked.length && (o.blocked = blocked.slice()); return say(o); };
  try {
    put("submit", function () { if (allowed(this, null)) return P.sub.call(this); blocked.push("send"); });
    put("requestSubmit", function (b) { if (allowed(this, b)) return P.req.call(this, b); blocked.push("send"); });
    P.add.call(window, "submit", onSubmit, true);
    P.add.call(window, "submit", onSubmit, false);
    P.add.call(window, "click", onClick, true);
    P.add.call(n, "click", onClick, false);  // after the page's own handlers on the element
    if (n.form) P.add.call(n.form, "submit", onSubmit, false);
    n.scrollIntoView({block: "center", inline: "center"});
    var fire = function (type) { P.dispatch.call(n, new Event(type, {bubbles: true})); };
    var cleared = function (stale) {
      if (spec.act === "type") { Object.getOwnPropertyDescriptor(n.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, "value").set.call(n, ""); }
      return done({ok: false, stale: stale});
    };
    if (spec.act === "click") {
      if (n.tagName === "A" || n.tagName === "AREA") P.ra.call(n, "ping");
      ["mousedown", "mouseup"].forEach(function (t) { P.dispatch.call(n, new MouseEvent(t, {bubbles: true, cancelable: true, view: window})); });
      var moved = same();  // a handler of the press may have changed where this goes
      if (moved) return done({ok: false, stale: moved === "gone" ? "dest" : moved});
      if (n instanceof HTMLElement) P.click.call(n);
      else P.dispatch.call(n, new MouseEvent("click", {bubbles: true, cancelable: true, view: window}));
    } else if (spec.act === "type") {
      n.focus();
      var after = same();  // the page's focus handlers have run
      if (after) return done({ok: false, stale: after});
      var setter = Object.getOwnPropertyDescriptor(n.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, "value").set;
      setter.call(n, spec.text);
      var typed = same();
      if (typed) return cleared("grew");
      fire("input"); fire("change");
      var handled = same();  // and the handlers of the typing
      if (handled) return cleared("grew");
      if (spec.enter) {
        var key = function (t) { return new KeyboardEvent(t, {key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true}); };
        var go = P.dispatch.call(n, key("keydown"));
        P.dispatch.call(n, key("keypress"));
        P.dispatch.call(n, key("keyup"));
        if (go && n.form) P.req.call(n.form, defaultButton(n.form) || undefined);
      }
    } else if (spec.act === "select") {
      var want = String(spec.text).trim().toLowerCase(), opts = Array.prototype.slice.call(n.options);
      var hits = opts.filter(function (o) { return !o.disabled && (cap(o.text, 60).toLowerCase() === want || String(o.value).toLowerCase() === want); });
      if (!hits.length) hits = opts.filter(function (o) { return !o.disabled && want && cap(o.text, 60).toLowerCase().indexOf(want) >= 0; });
      if (hits.length !== 1) return done({ok: false, option: hits.length});
      hits[0].selected = true;
      fire("input"); fire("change");
    }
    return done({ok: true});
  } finally {
    ["submit", "requestSubmit"].forEach(function (name) {
      if (saved[name]) Object.defineProperty(proto, name, saved[name]); else delete proto[name];
    });
    P.rm.call(window, "submit", onSubmit, true);
    P.rm.call(window, "submit", onSubmit, false);
    P.rm.call(window, "click", onClick, true);
    if (n) P.rm.call(n, "click", onClick, false);
    if (n && n.form) P.rm.call(n.form, "submit", onSubmit, false);
  }
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
    options, place, rows = item.get("options"), item.get("at"), item.get("form")
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
        fp=str(item.get("fp") or ""),
        form=fields.form_pairs(rows) if isinstance(rows, list) else (),
    )


def _parse_tabs(text: str) -> list[tuple[TabInfo, bool]]:
    """Tabs (and whether each is the active one) from a ``list_tabs`` reply."""
    value = _decode(text)
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

    async def _ready(self) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + SETTLE_S
        while loop.time() < end:
            if await self._eval("document.readyState") == "complete":
                return
            await asyncio.sleep(POLL_S)

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

    async def page(self, tab: str | None) -> RawPage:
        await self._focus(tab)
        url = await self._address(tab)
        data = await self._eval(PAGE_JS)
        if not isinstance(data, dict):
            raise BrowserError("Safari returned a page melo could not read.")
        elements = [e for e in map(_element, data.get("elements") or []) if e]
        skipped, frames = data.get("skipped"), data.get("frames")
        return RawPage(
            url,
            str(data.get("title") or ""),
            str(data.get("text") or ""),
            tuple(elements),
            skipped=skipped if isinstance(skipped, int) and skipped > 0 else 0,
            doc=str(data.get("doc") or ""),
            frames=tuple(str(f) for f in frames) if isinstance(frames, list) else (),
        )

    async def back(self, tab: str | None) -> TabInfo:
        handle = tab if tab and tab != "current" else (self._current or "")
        if not handle:
            raise BrowserError("No tab is open. Use browser_open first.")
        await self._focus(handle)
        await self._eval("(history.back(), true)")
        await asyncio.sleep(POLL_S)
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
        try:  # the browser's own navigation, not a script the page could have changed
            await self._call("open", url="about:blank")
        except BrowserSetupError:
            raise
        except BrowserError:
            await self._eval("(location.replace('about:blank'), true)")

    async def act(self, tab: str | None, act: Act) -> str:
        await self._focus(tab)
        item = act.element
        spec: dict[str, Any] = {
            "act": act.kind,
            "url": act.url,
            "doc": act.doc,
            "at": item.at,
            "fp": item.fp,
            "kind": item.kind,
            "label": item.label,
            "href": item.href,
            "submit": item.submit,
            "text": act.text[:MAX_TYPED_CHARS],
            "enter": act.enter,
            "send": act.send,
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
        stopped = [
            _BLOCKED[str(k)] for k in result.get("blocked") or [] if str(k) in _BLOCKED
        ]
        if result.get("ok") is not True:
            if result.get("stale"):
                raise StaleElementError(
                    _STALE.get(str(result["stale"]), _STALE["gone"])
                )
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
        return " ".join(dict.fromkeys(stopped))

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
