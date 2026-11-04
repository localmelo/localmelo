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

An act is done by the browser itself where it can be: the page script checks the
element and finds which of the browser's node ids it is (from
``get_page_content``), and ``page_interactions`` clicks, types or chooses on that
node, with real events a page cannot tell from a person's. The page script's own
events are the backup, and the result names the way used. Around either way an
observer set before the act sees what the page shows new, and melo waits until
the page is quiet (see :meth:`SafariAutomationDriver.act`).

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
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser import fields, render
from localmelo.support.browser.driver import (
    MAX_TYPED_CHARS,
    Act,
    ActResult,
    BrowserError,
    BrowserSetupError,
    Changes,
    CoveredError,
    Element,
    ElementKind,
    FieldAttrs,
    PointHit,
    RawPage,
    Sensitive,
    Shot,
    StaleElementError,
    TabInfo,
    View,
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
    "frame": "melo could not set up its safe page checks on this page, so it did nothing. Use browser_read to try again.",
    "point": "What is at that point is not what melo found there a moment ago, so it did not click. Take a new browser_screenshot.",
    "scrolled": "The page scrolled since the picture, so a point on it is no longer where it was. Take a new browser_screenshot.",
    "reach": "That point is inside a box from another site, and this Safari cannot click there for melo.",
}
# Why the browser's own input was passed over for the page script (the result says it).
_PASSED = {
    "tool": "this Safari cannot act on page elements itself",
    "covered": "another element covers it, so only the page script reaches it",
    "name": "the browser could not name that element",
    "missed": "the browser's own input did not reach it",
    "frame": "it is in a frame, where only the page script reaches",
}
_MOVED = (
    "A real click or key press landed on another element than the one melo checked, "
    "so melo stopped it. Nothing was clicked or typed, and the page and its refs "
    "are as they were: close or move what is in the way, or ask the person."
)
# The words for the kind of element something stands over.
_NOUN = {"textbox": "field", "select": "list", "checkbox": "box"}
# A page that is still settling (a bar sliding in, an ad loading) gets this long
# before an element found covered is looked at once more.
COVER_AGAIN_S = 0.25
# Looking at the page after an act: a poll this often, done once nothing changed for
# QUIET_S (a click waits up to LOOK_CAP_S for the page's first change), never longer.
LOOK_STEP_S = 0.1
QUIET_S = 0.3
LOOK_CAP_S = 2.0
MAX_NODES = 800
MAX_CONTENT_BYTES = 4 * 1024 * 1024
NO_CHECKS_NOTE = (
    "(melo could not set up its safe page checks on this page, so no links, buttons "
    "or fields are listed.)"
)
_BLOCKED = {
    "send": "The page tried to send a form that melo had not asked about, so melo stopped it. To send a form, click its button: melo asks first.",
    "link": "The page changed where that link goes while it was clicked, so melo stopped it.",
}
POLL_S = 0.4
BLANK_TRIES = 4

# The page script's pieces, shared by reading a page and by acting on it, so
# an element is found again by the same rules that listed it. ``SEL`` lists
# what may be a target; ``lookAt`` describes one (or says why it is not one:
# ``skip`` 1 is counted as skipped, 0 is not worth counting). Nothing here
# reads cookies or storage, and nothing changes the page.
#
# ``P`` holds the page-reading built-ins taken from a fresh, hidden frame, so
# a page that replaced ``getAttribute`` or ``JSON.stringify`` for itself does
# not change what these checks see. When the frame cannot be made, or one of
# its functions does not work, ``P`` is null: the page's own built-ins are never
# used instead, so a page is then read as text only and nothing is done on it.
COMMON_JS = (
    r"""
var P = (function () {
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
      rm: w.EventTarget.prototype.removeEventListener, closest: w.Element.prototype.closest,
      cdp: w.Node.prototype.compareDocumentPosition, matches: w.Element.prototype.matches,
      rect: w.Element.prototype.getBoundingClientRect,
      inner: Object.getOwnPropertyDescriptor(w.HTMLIFrameElement.prototype, "contentDocument").get
    };
    // The fresh functions are tried on their own terms, never against the page's.
    var probe = w.document.createElement("div");
    w.Element.prototype.setAttribute.call(probe, "lang", "x");
    if (fresh.ga.call(probe, "lang") !== "x" || fresh.json({a: 1}) !== '{"a":1}' ||
        typeof fresh.qsa.call(document, "html").length !== "number" || typeof fresh.gcs(document.documentElement).display !== "string") throw new Error("frame");
    for (var k in fresh) if (typeof fresh[k] !== "function") throw new Error(k);
    // Only what is at a point needs this one; without it no point is resolved.
    var efp = w.Document.prototype.elementFromPoint;
    fresh.efp = typeof efp === "function" ? efp : null;
    return fresh;
  } catch (e) {
    return null;  // never the page's own: nothing is listed or done without the fresh ones
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
  var f = P.closest.call(n, "fieldset"), l = f ? inside(f, "legend") : null;
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
var HEADINGS = null;
var headingBefore = function (n) {
  if (!HEADINGS) HEADINGS = Array.prototype.slice.call(P.qsa.call(document, "h1,h2,h3,h4,h5,h6,[role=heading]"));
  var lo = 0, hi = HEADINGS.length - 1, found = null;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    if (P.cdp.call(HEADINGS[mid], n) & 4) { found = HEADINGS[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
};
// The words around an element: its row's text, else the heading above it. A page that
// redraws a list in place leaves the same tags and labels at the same places; the
// items they belong to are what changed.
var contextOf = function (n) {
  var row = P.closest.call(n, "tr,li,[role=row],[role=listitem],article,[role=article]");
  var text = row && row !== n ? cap(row.innerText || row.textContent, 100) : "";
  if (text) return "r:" + text;
  var head = headingBefore(n);
  return head ? "h:" + cap(head.innerText || head.textContent, 80) : "";
};
// The text next to an element that is no button's or field's own: that of the nearer of
// its two closest containers holding some (a card's name, say, even when it is a link),
// cut short.
var LINKS = "a[href],[role=link]";
var nearText = function (n) {
  for (var c = n.parentElement, i = 0; c && i < 2 && c !== document.body && c !== document.documentElement; c = c.parentElement, i++) {
    var out = [], stack = [c], seen = 0, size = 0;
    while (stack.length && seen < 120 && size < 120) {
      var x = stack.pop(); seen++;
      if (x.nodeType === 3) { var t = cap(x.nodeValue, 80); if (t) { out.push(t); size += t.length + 1; } continue; }
      if (x.nodeType !== 1 || /^(SCRIPT|STYLE|TEMPLATE)$/.test(x.tagName) || (x !== c && P.matches.call(x, SEL) && !P.matches.call(x, LINKS))) continue;
      for (var k = x.childNodes.length - 1; k >= 0; k--) stack.push(x.childNodes[k]);
    }
    if (out.length) return cap(out.join(" "), 120);
  }
  return "";
};
// Where an element belongs, for finding a ref the page redrew again: what it is (tag,
// type, role, id, name), its row or the heading above it, and the text next to it.
// Without a row, a heading or such text nothing says where it belongs: "".
var belongsTo = function (n) {
  var around = contextOf(n), near = nearText(n);
  if (!around && !near) return "";
  return P.json([n.tagName.toLowerCase(), attr(n, "type").toLowerCase(), attr(n, "role").toLowerCase(), cap(n.id, 40), cap(attr(n, "name"), 40), around, near]) + frameMark(n);
};
var fingerprint = function (n, nth) {
  var parts = [], cur = n, depth = 0;
  while (cur && cur.nodeType === 1 && depth < 12) {
    var i = 1, sib = cur;
    while ((sib = sib.previousElementSibling)) if (sib.tagName === cur.tagName) i++;
    parts.push(cur.tagName.toLowerCase() + (i > 1 ? ":" + i : ""));
    cur = cur.parentElement; depth++;
  }
  return [n.tagName.toLowerCase(), attr(n, "type").toLowerCase(), attr(n, "role").toLowerCase(), cap(n.id, 40), cap(attr(n, "name"), 40), parts.reverse().join(">"), "#" + nth, contextOf(n)].join("|");
};
var attrsOf = function (n, st, labels, aria, ph) {
  var tag = n.tagName.toLowerCase(), type = attr(n, "type").toLowerCase();
  var max = parseInt(attr(n, "maxlength") || "0", 10);
  if (isNaN(max) || max < 0) max = 0;
  var mode = attr(n, "inputmode").toLowerCase();
  // The shape of a code without the attributes that usually say so: a digits-only
  // pattern ("[0-9]{6}", "\\d{4,6}") gives the length and makes the box numeric, and
  // so does an all-9s max on a number box.
  var shape = /^\^?(?:\[0-9\]|\\d)(?:[*+]|\{(\d+)(?:,(\d+))?\})\$?$/.exec(attr(n, "pattern").trim());
  if (shape) {
    if (!mode) mode = "numeric";
    if (!max && shape[1]) max = parseInt(shape[2] || shape[1], 10);
  }
  if (!max && type === "number" && /^9{4,8}$/.test(attr(n, "max").trim())) max = attr(n, "max").trim().length;
  var words = [attr(n, "name"), n.id, aria, attr(n, "title"), ph, labels, refsText(n, "aria-labelledby"), refsText(n, "aria-describedby"), legendOf(n), textBefore(n)].join(" ");
  var masked = String(st.webkitTextSecurity || st.getPropertyValue("-webkit-text-security") || "none") !== "none";
  return {type: tag === "textarea" ? "textarea" : tag === "select" ? "select" : type || "text", autocomplete: attr(n, "autocomplete").toLowerCase(), words: cap(words, 400), inputmode: mode, maxlength: max, masked: masked};
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
var lookAt = function (n, counts) {
  var tag = n.tagName.toLowerCase(), type = attr(n, "type").toLowerCase(), role = attr(n, "role").toLowerCase();
  var r = n.getBoundingClientRect(), st = P.gcs(n);
  if (st.visibility === "hidden" || st.display === "none" || (r.width === 0 && r.height === 0)) return {skip: 1};
  if (n.disabled || attr(n, "aria-disabled") === "true") return {skip: 1};
  var native = /^(a|button|input|textarea|select|summary)$/.test(tag);
  var known = /^(button|link|menuitem|menuitemcheckbox|menuitemradio|tab|switch|option|checkbox|radio)$/.test(role);
  if (!native && !known) {
    // A plain element that only looks clickable: it must say so, hold no other target,
    // and not sit inside one (the target is then the outer element).
    if (inside(n, SEL) || (n.parentElement && P.closest.call(n.parentElement, NESTED))) return {skip: 0};
    if (!attr(n, "onclick") && st.cursor !== "pointer") return {skip: 0};
  }
  var labels = labelsOf(n);
  var aria = cap(attr(n, "aria-label"), 120), ph = cap(attr(n, "placeholder"), 120);
  var title = cap(attr(n, "title") || attr(n, "alt"), 120);
  var shown = cap(n.innerText || n.textContent, 120);
  var el = {};
  if (!native && !known) el.plain = true;
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
  // Which of the same-labelled elements before it in the document this one is (counts
  // is how many came before; none given means the fingerprint is not wanted).
  if (counts) el.fp = fingerprint(n, (counts[el.kind + "|" + el.label] || 0) + 1) + frameMark(n);
  if (counts && !el.sensitive) el.context = belongsTo(n);
  return {el: el};
};
// A frame of the page from the same site is read and acted in as part of the page.
// Its place is the index of each frame on the way down ("2", "2.0"), and an element
// in it is bound to the frame's address too, so a frame that moved is another page.
var frameMark = function (n) {
  var d = n.ownerDocument;
  return d && d !== document ? "|f:" + cap(d.URL, 300) : "";
};
var subdoc = function (f) {
  try { return /^iframe$/i.test(f.tagName) ? P.inner.call(f) : f.contentDocument; } catch (e) { return null; }
};
var docAt = function (path) {
  var D = document, parts = path ? String(path).split(".") : [];
  for (var i = 0; i < parts.length && D; i++) {
    var f = P.qsa.call(D, "iframe,frame")[+parts[i]];
    D = f ? subdoc(f) : null;
  }
  return D || null;
};
var within = function (n, t) { return !!t && (t === n || !!(P.cdp.call(n, t) & 16)); };
// What is at a point of the view: the element hit, in the frames of the same site too,
// and the control it belongs to. A label moves to its field; a drawing, picture or
// video takes the click itself; a frame from another site is a box melo cannot see into.
var hitAt = function (x, y) {
  if (!P.efp) return {node: null, surface: "none", doc: document, path: "", x: x, y: y};
  var D = document, h = P.efp.call(D, x, y), path = "", depth = 0;
  while (h && /^(iframe|frame)$/i.test(h.tagName) && depth < 3) {
    var sub = subdoc(h);
    if (!sub) return {node: h, surface: "frame", doc: D, path: path, x: x, y: y};
    var r = P.rect.call(h), list = P.qsa.call(D, "iframe,frame");
    path = (path ? path + "." : "") + Array.prototype.indexOf.call(list, h);
    x -= r.left + h.clientLeft; y -= r.top + h.clientTop;
    D = sub; h = P.efp.call(D, x, y); depth++;
  }
  if (!h || h.nodeType !== 1) return {node: null, surface: "none", doc: D, path: path, x: x, y: y};
  var lab = P.closest.call(h, "label");
  if (lab && lab.control) return {node: lab.control, surface: "control", doc: D, path: path, x: x, y: y, raw: h, label: lab};
  for (var c = h, i = 0; c && c.nodeType === 1 && i < 8; c = c.parentElement, i++) {
    if (P.matches.call(c, SEL) && lookAt(c).el) return {node: c, surface: "control", doc: D, path: path, x: x, y: y, raw: h};
  }
  for (c = h, i = 0; c && c.nodeType === 1 && i < 3; c = c.parentElement, i++) {
    if (/^(canvas|img|svg|video|embed|object)$/i.test(c.tagName)) return {node: c, surface: "canvas", doc: D, path: path, x: x, y: y, raw: h};
  }
  if (P.gcs(h).cursor === "pointer") return {node: h, surface: "control", doc: D, path: path, x: x, y: y, raw: h};
  return {node: null, surface: "none", doc: D, path: path, x: x, y: y};
};
// What stands over a point: where it sits (a sticky or fixed bar, a dialog, a frame) and
// what it is, in words for the model: {what: "a sticky header", detail: "h1 'Daily'"}.
// A bar is told by its place in the view, a frame by the site it comes from; an ad is
// one that says so in its id, class or title.
var AD_WORDS = /(^|[^a-z])(ads?|adverts?|advertisements?|sponsored|banners?)([^a-z]|$)/i;
var coverOf = function (t) {
  if (!t || t.nodeType !== 1) return {what: "another element", detail: ""};
  var tag = t.tagName.toLowerCase(), bar = null, pos = "", c, i;
  for (c = t, i = 0; c && c.nodeType === 1 && i < 15; c = c.parentElement, i++) {
    var p = P.gcs(c).position;
    if (p === "sticky" || p === "fixed") { bar = c; pos = p; break; }
  }
  var named = function (e) { return [attr(e, "id"), attr(e, "class"), attr(e, "title"), attr(e, "name")].join(" "); };
  var adLike = AD_WORDS.test(named(t)) || (bar ? AD_WORDS.test(named(bar)) : false);
  if (tag === "iframe" || tag === "frame") {
    var host = "";
    try { host = new URL(attr(t, "src"), document.baseURI).hostname; } catch (e) { host = ""; }
    return {what: adLike ? "an ad" : "a frame", detail: "iframe" + (host ? " from " + host : "")};
  }
  var text = cap(tag === "input" ? attr(t, "aria-label") || attr(t, "placeholder") : t.innerText || t.textContent, 40);
  var detail = tag + (text ? " '" + text + "'" : "");
  if (P.closest.call(t, "dialog,[role=dialog],[role=alertdialog],[aria-modal=true]")) return {what: "a dialog", detail: detail};
  if (!bar) return {what: adLike ? "an ad" : "another element", detail: detail};
  var r = P.rect.call(bar), vh = window.innerHeight || 0;
  var place = r.bottom <= vh * 0.6 && r.top < vh * 0.25 ? "header" : r.top >= vh * 0.4 && r.bottom > vh * 0.75 ? "bar at the bottom" : "layer";
  var which = pos === "sticky" ? "sticky" : "fixed";
  if (adLike) return {what: "an ad" + (place === "bar at the bottom" ? " at the bottom" : place === "header" ? " at the top" : ""), detail: detail};
  return {what: "a " + which + " " + place, detail: detail};
};
// The state of an element that an action may change, never the value of a secret field.
var stateOf = function (n) {
  if (!n || !n.isConnected) return {gone: true};
  var tag = n.tagName.toLowerCase(), type = attr(n, "type").toLowerCase();
  var r = P.rect.call(n), st = P.gcs(n);
  var s = {shown: !(st.visibility === "hidden" || st.display === "none" || (r.width === 0 && r.height === 0)),
           disabled: !!n.disabled || attr(n, "aria-disabled") === "true"};
  var field = /^(input|textarea|select)$/.test(tag);
  var secret = field && (/^(password|hidden)$/.test(type) || !!secretKind(attrsOf(n, st, labelsOf(n), cap(attr(n, "aria-label"), 120), cap(attr(n, "placeholder"), 120))));
  if (tag === "input" && /^(checkbox|radio)$/.test(type)) s.checked = !!n.checked;
  else if (tag === "select") s.value = secret ? "" : cap(n.options && n.selectedIndex >= 0 ? n.options[n.selectedIndex].text : "", 120);
  else if (field && !/^(button|submit|reset|image)$/.test(type)) s.value = secret ? "" : cap(n.value, 120);
  else s.label = tag === "input" ? cap(n.value, 120) : cap(n.innerText || n.textContent, 120);
  ["aria-pressed", "aria-selected", "aria-expanded", "aria-checked", "aria-current"].forEach(function (a) { var v = attr(n, a); if (v) s[a.slice(5)] = v; });
  return s;
};
var linesOf = function (node, into, limit) {
  var text = String(node && (node.innerText || node.textContent) || "");
  var parts = text.split("\n");
  for (var i = 0; i < parts.length && i < limit; i++) { var t = cap(parts[i], 200); if (t.length > 1) into.push(t); }
  return into;
};
"""
)

# Returns JSON: the page's text (the main part when there is one) and its
# visible links, buttons and fields, each with its place (``at``) among the
# page's possible targets. Secret fields are marked, their values never read.
PAGE_JS = (
    r"""(function () {"""
    + COMMON_JS
    + r"""
  var out = {url: location.href, doc: docId(), title: document.title || "", text: "", elements: [], skipped: 0, frames: [], unreached: []};
  var main = document.querySelector("main,[role=main],article");
  var root = main && (main.innerText || "").length > 300 ? main : document.body;
  out.text = root ? String(root.innerText || "").slice(0, 200000) : "";
  if (!P) {
    // The safe checks could not be set up: the text is read, nothing is listed to act on.
    out.text = """
    + json.dumps(NO_CHECKS_NOTE)
    + r""" + "\n" + out.text;
    return JSON.stringify(out);
  }
  // The page and its frames from the same site (two levels down); a visible frame
  // from another site is named, since melo cannot read into it.
  var docs = [[document, ""]];
  for (var q = 0; q < docs.length && q < 12; q++) {
    var here = docs[q][0], path = docs[q][1], list = P.qsa.call(here, "iframe,frame");
    for (var k = 0; k < list.length && k < 20; k++) {
      var f = list[k], sub = subdoc(f), u = "";
      try { u = String(f.contentWindow.location.href); } catch (e) { u = ""; }
      if (!u || u === "about:blank") u = String(attr(f, "src") ? f.src : "");
      if (u && u !== "about:blank") out.frames.push(u.slice(0, 500));
      if (sub && sub.documentElement) {
        if (!path || path.split(".").length < 2) docs.push([sub, (path ? path + "." : "") + k]);
      } else if (/^https?:/i.test(u)) {
        var fr = P.rect.call(f), fs = P.gcs(f);
        if (fr.width >= 20 && fr.height >= 20 && fs.visibility !== "hidden" && fs.display !== "none") out.unreached.push(u.slice(0, 500));
      }
    }
  }
  for (var di = 0; di < docs.length; di++) {
    var D = docs[di][0], nodes = P.qsa.call(D, SEL), counts = {};
    for (var i = 0; i < nodes.length && i < 4000; i++) {
      var d = lookAt(nodes[i], counts);
      if (!d.el) { out.skipped += d.skip; continue; }
      var key = d.el.kind + "|" + d.el.label;
      counts[key] = (counts[key] || 0) + 1;
      if (out.elements.length >= 400) { out.skipped += 1; continue; }
      d.el.at = i;
      if (docs[di][1]) d.el.frame = docs[di][1];
      out.elements.push(d.el);
    }
    if (di > 0 && D.body) out.text += "\n\n[frame " + cap(D.URL, 200) + "]\n" + String(D.body.innerText || "").slice(0, 20000);
  }
  return P.json(out);
})()"""
)


def _data(value: Any) -> str:
    """*value* as a JavaScript literal that holds no ``$`` (the server's ``$uid()`` macro) and no line separators."""
    return (
        json.dumps(value)
        .replace("$", "\\u0024")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def act_script(spec: dict[str, Any]) -> str:
    """The script that does, or prepares, one :class:`Act`, given its *spec* as plain data.

    It does nothing, and says why, unless the page is the one that was read:
    the same address in the same document, and at ``at`` (in the frame
    ``frame`` of the same site) the same element (its fingerprint, kind, label,
    address and the address its form is sent to, and for a text box the same
    field attributes). With ``point`` the element must also be what is at that
    point of the view (``at`` -1: a drawing or a frame from another site,
    known by its place only), and the view must not have scrolled.

    ``mode`` says what it does once that holds:

    - ``script`` (the default) acts with page-script events. It looks again
      after the field has focus, after the text is in and after the page's own
      handlers ran, and clears the field if it turned into a secret one. While
      it works it tries to stop a form the page sends that melo did not ask
      about, and a link whose address is no longer the one that was read when
      it is pressed or clicked.
    - ``scroll`` brings the element to the middle of the view at once (not
      smoothly) and says where it can be clicked: ``cx``, ``cy`` is its middle,
      or, when another element covers the middle, the clear point of the part
      inside the view with most room around it (``shifted``); with no clear
      point ``covered`` is true and ``by`` says what stands over it (``what``
      and ``detail``). It also describes the element (tag, role, text, label,
      name, id), so melo can find it in the browser's own reading of the page.
    - ``arm`` keeps the same guards up while the browser itself acts on the
      node melo found (``uid``), or at the point: also, until
      the first look after the act, a real press, click or keystroke that lands
      on another element is stopped (``moved``; the report's ``movedBy`` says
      what it landed on). With a ``point`` the place is shown by a short-lived
      ring that takes no clicks.

    With ``key``, an observer set before the act keeps what the page shows new
    and the element's state for :data:`SETTLE_JS` and :data:`DISARM_JS`, which
    also take the guards down (by themselves after 20 s at the latest).

    The guards are best effort and last only that long: a page can send later
    (a timer, a request from a script, a function it saved earlier), and once
    a text is in a page the page may send it itself. When the fresh built-ins
    (``P``) cannot be had it does nothing.
    """
    return r"""(function (spec) {""" + COMMON_JS + r"""
  if (!P) return JSON.stringify({ok: false, stale: "frame"});  // nothing is done without the fresh built-ins
  var say = function (o) { return P.json(o); };
  var blocked = [], seen = {__proto__: null}, undo = [];
  if (location.href !== spec.url) return say({ok: false, stale: "page"});
  if (docId() !== spec.doc) return say({ok: false, stale: "doc"});
  var pt = spec.point || null;
  if (pt && (Math.abs(window.scrollX - pt.sx) > 2 || Math.abs(window.scrollY - pt.sy) > 2)) return say({ok: false, stale: "scrolled"});
  var D = docAt(spec.frame || ""), n = null, counts = {};
  if (spec.at >= 0) {
    if (!D) return say({ok: false, stale: "gone"});
    var nodes = P.qsa.call(D, SEL);
    n = nodes[spec.at];
    for (var j = 0; j < spec.at && j < nodes.length; j++) {  // which of its label this one is
      var before = lookAt(nodes[j]).el;
      if (before) { var key = before.kind + "|" + before.label; counts[key] = (counts[key] || 0) + 1; }
    }
  }
  var hit = null;
  if (pt) {
    hit = hitAt(pt.x, pt.y);
    if (spec.at >= 0 ? hit.node !== n : !hit.node || fingerprint(hit.node, 0) + frameMark(hit.node) !== spec.fp) return say({ok: false, stale: "point"});
    if (spec.at < 0) { n = hit.node; D = hit.doc; }
  }
  var W = (D && D.defaultView) || window;
  var same = function () {
    if (spec.at < 0) return n && n.isConnected && fingerprint(n, 0) + frameMark(n) === spec.fp ? "" : "gone";
    var d = n ? lookAt(n, counts) : {}, el = d.el;
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
  var mode = spec.mode || "script";
  var setValue = function (v) {
    Object.getOwnPropertyDescriptor(n.tagName === "TEXTAREA" ? W.HTMLTextAreaElement.prototype : W.HTMLInputElement.prototype, "value").set.call(n, v);
  };
  var pick = function () {  // the one option the words name, or the count of those they fit
    var want = String(spec.text).trim().toLowerCase(), opts = Array.prototype.slice.call(n.options || []);
    var hits = opts.filter(function (o) { return !o.disabled && (cap(o.text, 60).toLowerCase() === want || String(o.value).toLowerCase() === want); });
    if (!hits.length) hits = opts.filter(function (o) { return !o.disabled && want && cap(o.text, 60).toLowerCase().indexOf(want) >= 0; });
    return hits.length === 1 ? hits[0] : hits.length;
  };
  // To the middle of the view, at once: a page that scrolls smoothly would still be on its way
  // when the place is measured, and the click would land where the element was.
  var centre = function () {
    try { n.scrollIntoView({block: "center", inline: "center", behavior: "instant"}); }
    catch (e) { n.scrollIntoView({block: "center", inline: "center"}); }
  };
  if (mode === "scroll") {
    centre();
    var r = P.rect.call(n), tag = n.tagName.toLowerCase();
    var vw = W.innerWidth || 0, vh = W.innerHeight || 0;
    var topAt = function (x, y) { return D && P.efp ? P.efp.call(D, x, y) : n; };
    var clear = function (x, y) { return !P.efp || (x >= 0 && y >= 0 && x < vw && y < vh && within(n, topAt(x, y))); };
    var cx = r.left + r.width / 2, cy = r.top + r.height / 2, covered = !clear(cx, cy), shifted = false, by = null;
    if (covered) {
      by = coverOf(topAt(cx, cy));  // what stands over the middle
      // Another part of it may be clear: the part with most room around it, nearest the middle.
      var l = Math.max(r.left, 0), tp = Math.max(r.top, 0), rt = Math.min(r.right, vw), bt = Math.min(r.bottom, vh);
      if (rt - l >= 2 && bt - tp >= 2) {
        var G = 7, ok = [], best = null, gx, gy, dx, dy;
        for (gy = 0; gy < G; gy++) {
          ok[gy] = [];
          for (gx = 0; gx < G; gx++) ok[gy][gx] = clear(l + (rt - l) * (gx + 0.5) / G, tp + (bt - tp) * (gy + 0.5) / G);
        }
        for (gy = 0; gy < G; gy++) for (gx = 0; gx < G; gx++) {
          if (!ok[gy][gx]) continue;
          var room = 0, px = l + (rt - l) * (gx + 0.5) / G, py = tp + (bt - tp) * (gy + 0.5) / G;
          for (dy = -1; dy <= 1; dy++) for (dx = -1; dx <= 1; dx++) if (ok[gy + dy] && ok[gy + dy][gx + dx]) room++;
          var away = Math.abs(px - cx) + Math.abs(py - cy);
          if (!best || room > best.room || (room === best.room && away < best.away)) best = {x: px, y: py, room: room, away: away};
        }
        if (best) { cx = best.x; cy = best.y; covered = false; shifted = true; }
      }
    }
    return say({ok: true, covered: covered, shifted: shifted, by: by, cx: cx, cy: cy, tag: tag, role: attr(n, "role").toLowerCase(),
      text: cap(tag === "input" ? "" : n.innerText || n.textContent, 200), value: tag === "input" && /^(submit|button|reset)$/.test(attr(n, "type").toLowerCase()) ? cap(n.value, 120) : "",
      label: cap(labelsOf(n), 120), aria: cap(attr(n, "aria-label"), 120), name: cap(attr(n, "name"), 80), id: cap(n.id, 80),
      title: cap(attr(n, "title") || attr(n, "alt"), 120), placeholder: cap(attr(n, "placeholder"), 120)});
  }
  var option = null;
  if (spec.act === "select") {
    option = pick();
    if (typeof option === "number") return say({ok: false, option: option});
  }
  var wins = [window];
  if (W !== window) wins.push(W);
  var listen = function (target, type, fn, capture, list) {
    P.add.call(target, type, fn, capture);
    (list || undo).push(function () { P.rm.call(target, type, fn, capture); });
  };
  var allowed = function (form, button) {
    return spec.send && (spec.submit === "" || formAddress(form, button) === spec.submit);
  };
  var onSubmit = function (e) {
    seen.submit = (seen.submit || 0) + 1;
    if (!allowed(e.target, e.submitter)) { e.preventDefault(); e.stopImmediatePropagation(); blocked.push("send"); }
  };
  var onClick = function (e) {
    var t = e.target, a = t && t.nodeType === 1 ? P.closest.call(t, "a[href]") : null;
    if (spec.kind === "link" && a && hrefOf(a) !== spec.href) { e.preventDefault(); e.stopImmediatePropagation(); blocked.push("link"); }
  };
  var guard = function () {
    wins.forEach(function (w) {
      var proto = w.HTMLFormElement.prototype;
      var saved = {submit: Object.getOwnPropertyDescriptor(proto, "submit"), requestSubmit: Object.getOwnPropertyDescriptor(proto, "requestSubmit")};
      var put = function (name, fn) { Object.defineProperty(proto, name, {value: fn, configurable: true, writable: true}); };
      undo.push(function () {
        ["submit", "requestSubmit"].forEach(function (name) {
          if (saved[name]) Object.defineProperty(proto, name, saved[name]); else delete proto[name];
        });
      });
      put("submit", function () { if (allowed(this, null)) return P.sub.call(this); blocked.push("send"); });
      put("requestSubmit", function (b) { if (allowed(this, b)) return P.req.call(this, b); blocked.push("send"); });
      listen(w, "submit", onSubmit, true);
      listen(w, "submit", onSubmit, false);
      listen(w, "click", onClick, true);
    });
    listen(n, "click", onClick, false);  // after the page's own handlers on the element
    if (n.form) listen(n.form, "submit", onSubmit, false);
  };
  var unguard = function (list) { while (list.length) { try { list.pop()(); } catch (e) { /* the page took it already */ } } };
  // Until the first look after the browser's own act: a real press, click or keystroke
  // must land on the element that was checked, or it is stopped.
  var held = [];
  var zone = [n];  // where a real input may land: the element, and what is at a point
  if (hit && hit.raw) zone.push(hit.raw);
  if (hit && hit.label) zone.push(hit.label);
  // Enter in a field clicks its form's default button: the browser's own send, asked about already.
  if (spec.act === "type" && spec.enter && spec.send && n.form) { var first = defaultButton(n.form); if (first) zone.push(first); }
  var inZone = function (t) { return zone.some(function (z) { return within(z, t); }); };
  var hitBy = null;  // what a stopped real click or key press landed on
  var contain = function () {
    var on = function (e) {
      if (inZone(e.target)) { seen[e.type] = (seen[e.type] || 0) + 1; return; }
      if (e.isTrusted === false) return;  // the page's own events are its business
      e.preventDefault(); e.stopImmediatePropagation(); blocked.push("moved");
      if (!hitBy) { try { hitBy = coverOf(e.target); } catch (x) { hitBy = null; } }
    };
    ["pointerdown", "mousedown", "pointerup", "mouseup", "click", "keydown", "beforeinput"].forEach(function (t) {
      wins.forEach(function (w) { listen(w, t, on, true, held); });
    });
    var count = function (e) { if (inZone(e.target)) seen[e.type] = (seen[e.type] || 0) + 1; };
    wins.forEach(function (w) { listen(w, "input", count, true); listen(w, "change", count, true); });
  };
  var ring = null;
  var mark = function () {
    try {
      ring = document.createElement("div");
      ring.setAttribute("aria-hidden", "true");
      ring.setAttribute("style", "position:fixed;left:" + (pt.x - 14) + "px;top:" + (pt.y - 14) + "px;width:28px;height:28px;box-sizing:border-box;" +
        "border:3px solid #ff3b30;border-radius:50%;background:rgba(255,59,48,.2);box-shadow:0 0 0 2px #fff;pointer-events:none;z-index:2147483647");
      (document.body || document.documentElement).appendChild(ring);
      undo.push(function () { if (ring.parentNode) ring.parentNode.removeChild(ring); });
    } catch (e) { ring = null; }
  };
  // What the page shows new from now on, and the element's state before.
  var watch = function () {
    var s0 = P.json(stateOf(n)), before = {__proto__: null}, texts = [], count = 0, last = Date.now(), done = false;
    [document, D].forEach(function (d, i) {
      if (d && !(i && d === document)) linesOf(d.body, [], 5000).forEach(function (t) { before[t] = 1; });
    });
    var take = function (el) {
      if (!el || el.nodeType !== 1 || texts.length >= 8 || (ring && within(ring, el))) return;
      if (String(el.textContent || "").length > 20000 || !el.getClientRects().length) return;
      linesOf(el, [], 50).forEach(function (t) { if (!before[t] && texts.indexOf(t) < 0 && texts.length < 8) texts.push(t); });
    };
    var mine = function (r) {
      if (!ring) return false;
      var touches = function (list) { return Array.prototype.indexOf.call(list || [], ring) >= 0; };
      return within(ring, r.target) || touches(r.addedNodes) || touches(r.removedNodes);
    };
    var MO = window.MutationObserver;
    if (MO) {
      var mo = new MO(function (records) {
        var theirs = false;
        for (var i = 0; i < records.length && i < 300; i++) {
          var r = records[i];
          if (mine(r)) continue;
          theirs = true;
          if (r.type === "childList") {
            for (var a = 0; a < r.addedNodes.length && a < 50; a++) { var x = r.addedNodes[a]; take(x.nodeType === 1 ? x : x.parentElement); }
          } else take(r.type === "characterData" ? r.target.parentElement : r.target);
        }
        if (theirs) { count++; last = Date.now(); }
      });
      var opts = {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["class", "style", "hidden", "open", "aria-hidden"]};
      mo.observe(document, opts);
      if (D && D !== document) mo.observe(D, opts);
      undo.push(function () { mo.disconnect(); });
    }
    var finish = function () {
      if (done) return P.json({lost: true});
      done = true;
      var later = same();
      if (later && spec.act === "type" && n.isConnected) {
        // The field changed after the text went in (it may be a secret one now): it is emptied.
        try { setValue(""); } catch (e) { /* gone */ }
        later = "field";
      }
      var s1 = P.json(stateOf(n));
      unguard(held); unguard(undo);
      try { delete window[spec.key]; } catch (e) { /* kept */ }
      return P.json({blocked: blocked, texts: texts, before: s0, after: s1, seen: seen, stale: later === "gone" ? "" : later, count: count, movedBy: hitBy});
    };
    var poll = function () {
      unguard(held);  // the act is over: the person's own clicks are theirs again
      return P.json({doc: true, count: count, quiet: Date.now() - last, target: P.json(stateOf(n)) !== s0, moved: blocked.indexOf("moved") >= 0});
    };
    var timer = window.setTimeout(finish, 20000);
    undo.push(function () { window.clearTimeout(timer); });
    Object.defineProperty(window, spec.key, {value: Object.freeze({poll: poll, finish: finish}), configurable: true});
  };
  if (mode === "arm") {
    try {
      if (pt) mark();
      guard();
      contain();
      watch();
    } catch (e) {
      unguard(held); unguard(undo);
      return say({ok: false, stale: "frame"});
    }
    return say({ok: true, option: option ? cap(option.text, 60) : "", frame: !!(hit && hit.surface === "frame")});
  }
  // The page-script act: its guards hold while it runs; the observer stays for the look after.
  var done = function (o) { blocked.length && (o.blocked = blocked.slice()); return say(o); };
  var cleared = function (stale) {
    if (spec.act === "type") setValue("");
    return done({ok: false, stale: stale});
  };
  var fire = function (type) { P.dispatch.call(n, new Event(type, {bubbles: true})); };
  var mouse = function (type) {
    var o = {bubbles: true, cancelable: true, view: W};
    if (hit) { o.clientX = hit.x; o.clientY = hit.y; }
    P.dispatch.call(n, new MouseEvent(type, o));
  };
  var temp = [];
  try {
    if (spec.key) { if (pt) mark(); watch(); }
    temp = undo; undo = [];
    guard();
    if (!pt) centre();
    if (spec.act === "click") {
      if (hit && hit.surface === "frame") return done({ok: false, stale: "reach"});
      if (n.tagName === "A" || n.tagName === "AREA") P.ra.call(n, "ping");
      mouse("mousedown"); mouse("mouseup");
      var moved = same();  // a handler of the press may have changed where this goes
      if (moved) return done({ok: false, stale: moved === "gone" ? "dest" : moved});
      if (n instanceof W.HTMLElement && !hit) P.click.call(n);
      else mouse("click");
    } else if (spec.act === "type") {
      n.focus();
      var after = same();  // the page's focus handlers have run
      if (after) return done({ok: false, stale: after});
      setValue(spec.text);
      if (same()) return cleared("grew");
      fire("input"); fire("change");
      if (same()) return cleared("grew");  // and the handlers of the typing
      if (spec.enter) {
        var keyOf = function (t) { return new KeyboardEvent(t, {key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true}); };
        var go = P.dispatch.call(n, keyOf("keydown"));
        P.dispatch.call(n, keyOf("keypress"));
        P.dispatch.call(n, keyOf("keyup"));
        if (go && n.form) P.req.call(n.form, defaultButton(n.form) || undefined);
      }
    } else if (spec.act === "select") {
      option.selected = true;
      fire("input"); fire("change");
    }
    return done({ok: true});
  } finally {
    unguard(undo);
    undo = temp;
  }
})(""" + _data(spec) + ")"


# A look at the page after an act, through what ``act_script`` left under ``key``:
# whether it is still the same document, how many changes the page made and how
# long ago the last one was, and whether the element's state changed. No page
# built-in is set up for it, so the look itself changes nothing.
SETTLE_JS = r"""(function (settle) {
  if (String(performance.timeOrigin) !== settle.doc) return '{"doc":false}';
  var S = window[settle.key];
  return S ? S.poll() : '{"lost":true}';
})(%s)"""

# What the act changed (new text, the element's state before and after), the
# guards' reports, and the guards taken down.
DISARM_JS = r"""(function (disarm) {
  if (String(performance.timeOrigin) !== disarm.doc) return '{"lost":true}';
  var S = window[disarm.key];
  return S ? S.finish() : '{"lost":true}';
})(%s)"""


def hit_script(spec: dict[str, Any]) -> str:
    """The script that says what is at a point of the view (``x``, ``y`` in page pixels).

    It answers only on the page that was pictured (``url``, ``doc``) and when
    the view has not scrolled since (``sx``, ``sy``). The answer is the
    element the point resolves to, as :data:`PAGE_JS` lists elements (with its
    place ``at`` and ``frame``), or a surface known by its place only, and the
    words around it, which melo checks for consent and terms.
    """
    return r"""(function (at) {""" + COMMON_JS + r"""
  var spec = at;
  if (!P) return JSON.stringify({stale: "frame"});
  if (location.href !== spec.url) return P.json({stale: "page"});
  if (docId() !== spec.doc) return P.json({stale: "doc"});
  if (Math.abs(window.scrollX - spec.sx) > 2 || Math.abs(window.scrollY - spec.sy) > 2) return P.json({stale: "scrolled"});
  var h = hitAt(spec.x, spec.y), out = {surface: h.surface};
  if (!h.node) return P.json(out);
  if (h.surface === "frame") {
    var u = "";
    try { u = String(h.node.src || ""); } catch (e) { u = ""; }
    out.site = u.slice(0, 500);
    out.el = {kind: "button", label: cap(attr(h.node, "title") || attr(h.node, "aria-label") || attr(h.node, "name"), 120), fp: fingerprint(h.node, 0) + frameMark(h.node)};
    return P.json(out);
  }
  var nodes = P.qsa.call(h.doc, SEL), at = Array.prototype.indexOf.call(nodes, h.node), counts = {};
  for (var j = 0; at > 0 && j < at; j++) {
    var b = lookAt(nodes[j]).el;
    if (b) { var key = b.kind + "|" + b.label; counts[key] = (counts[key] || 0) + 1; }
  }
  var d = at >= 0 ? lookAt(h.node, counts) : {}, el = d.el;
  if (el && h.surface === "control") {
    el.at = at;
    if (h.path) el.frame = h.path;
    if (h.label && !h.path) {
      // A label hit moves to its field, when the field itself is there to click.
      var r = P.rect.call(h.node), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      if (r.width > 0 && r.height > 0 && within(h.node, P.efp.call(document, cx, cy))) { out.x = cx; out.y = cy; }
    }
  } else {
    var t = h.node;
    out.surface = h.surface === "control" ? "control" : h.surface;
    el = {kind: "button", label: cap(attr(t, "aria-label") || attr(t, "title") || attr(t, "alt") || t.innerText || t.textContent, 120), at: -1, fp: fingerprint(t, 0) + frameMark(t)};
    if (h.path) el.frame = h.path;
  }
  var row = P.closest.call(h.node, "label,tr,li,p,fieldset,[role=row],[role=listitem],[role=dialog]");
  out.words = cap([el.label, el.also, el.attrs ? el.attrs.words : "", row ? row.innerText || row.textContent : ""].join(" "), 400);
  out.el = el;
  return P.json(out);
})(""" + _data(spec) + ")"


# The view a picture shows: the page and how far it is scrolled, in page pixels.
VIEW_JS = r"""(function (view) {
  return JSON.stringify({url: location.href, doc: String(performance.timeOrigin), w: window.innerWidth, h: window.innerHeight,
    x: window.scrollX, y: window.scrollY, dpr: window.devicePixelRatio || 1});
})(0)"""


def mark_script(ring: str, view: View, x: float, y: float, on: bool) -> str:
    """Show (*on*) or take away the ring *ring* at (*x*, *y*) for a picture, only on the page and view *view*."""
    spec = {
        "id": ring,
        "doc": view.doc,
        "sx": view.x,
        "sy": view.y,
        "x": x,
        "y": y,
        "on": on,
    }
    return r"""(function (mark) {
  var old = document.getElementById(mark.id);
  if (old && old.parentNode) old.parentNode.removeChild(old);
  if (!mark.on || String(performance.timeOrigin) !== mark.doc) return false;
  if (Math.abs(window.scrollX - mark.sx) > 2 || Math.abs(window.scrollY - mark.sy) > 2) return false;
  var ring = document.createElement("div");
  ring.id = mark.id;
  ring.setAttribute("aria-hidden", "true");
  ring.setAttribute("style", "position:fixed;left:" + (mark.x - 14) + "px;top:" + (mark.y - 14) + "px;width:28px;height:28px;box-sizing:border-box;" +
    "border:3px solid #ff3b30;border-radius:50%;background:rgba(255,59,48,.2);box-shadow:0 0 0 2px #fff;pointer-events:none;z-index:2147483647");
  (document.body || document.documentElement).appendChild(ring);
  return true;
})(""" + _data(spec) + ")"


FOCUS_JS = r"""(function () {
  var a = document.activeElement;
  if (!a || a.tagName !== "INPUT") return false;
  var t = (a.getAttribute("type") || "").toLowerCase(), ac = (a.getAttribute("autocomplete") || "").toLowerCase();
  return t === "password" || ac === "one-time-code" || ac.indexOf("cc-") === 0 || ac.indexOf("-password") > 0;
})()"""


def same_page(left: str, right: Any) -> bool:
    """Whether two addresses are one page: scheme, host, port, path and query (not the fragment).

    The scheme and the host are compared in lower case, a missing path is ``/``
    and a port that is the scheme's own does not count. An address that cannot
    be read is a different page.
    """
    if not isinstance(right, str):
        return False
    try:
        a, b = urlsplit(left), urlsplit(right)
        keys = []
        for part in (a, b):
            port = part.port
            if port == {"http": 80, "https": 443}.get(part.scheme):
                port = None
            keys.append(
                (part.scheme, part.hostname, port, part.path or "/", part.query)
            )
    except ValueError:
        return False
    return keys[0] == keys[1]


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
        frame=str(item.get("frame") or ""),
        plain=item.get("plain") is True,
        context=str(item.get("context") or "")[:1000],
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


class PageScriptActs:
    """What a driver that runs melo's page scripts in the page does around an act.

    A driver gives :meth:`_eval` (a script run in the page of the tab it
    means now, its value decoded); this base runs an act script and reads its
    answer (:meth:`_script`), acts by the page script (:meth:`_scripted`) and
    looks at the page after an act until it is quiet (:meth:`_look`). Both
    the Safari and the Chrome drivers use it, so an act is checked, guarded and
    looked at by the same rules whichever browser does it. ``label`` is the
    browser's name in what melo says.
    """

    label = "Safari"

    async def _eval(self, source: str) -> Any:
        raise NotImplementedError

    async def _ready(self) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + SETTLE_S
        while loop.time() < end:
            if await self._eval("document.readyState") == "complete":
                return
            await asyncio.sleep(POLL_S)

    async def _script(self, source: str) -> dict[str, Any]:
        """Run an act script: its answer, or the error it means (stale, no such option)."""
        result = await self._eval(source)
        if not isinstance(result, dict):
            raise BrowserError(f"{self.label} returned an answer melo could not read.")
        if result.get("ok") is not True:
            if result.get("stale"):
                why = str(result["stale"])
                raise StaleElementError(_STALE.get(why, _STALE["gone"]), why)
            if "option" in result:
                raise BrowserError(
                    "That list has no single option with those words (none, or "
                    "more than one). Use browser_read for its options."
                )
            raise BrowserError("The page did not take that action.")
        return result

    async def _scrolled(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Scroll the element to the middle of the view and say where it can be clicked.

        An element found covered gets one more try after a moment: a bar
        sliding in or an ad loading may still be moving into place.
        """
        ready = await self._script(act_script({**spec, "mode": "scroll"}))
        if ready.get("covered") is True:
            await asyncio.sleep(COVER_AGAIN_S)
            ready = await self._script(act_script({**spec, "mode": "scroll"}))
        return ready

    async def _scripted(
        self, spec: dict[str, Any], act: Act, passed: str, failed: bool
    ) -> ActResult:
        """The act by the page script, with the look after it."""
        key = _key()
        try:
            result = await self._script(
                act_script({**spec, "mode": "script", "key": key})
            )
        except BrowserError:
            with contextlib.suppress(BrowserError):
                await self._finish(key, spec["doc"])
            raise
        data, quiet = await self._look(key, spec["doc"], act)
        self._check_typed(act, data)
        blocked = [
            str(k)
            for k in [*(result.get("blocked") or []), *(data.get("blocked") or [])]
        ]
        return ActResult(
            _stopped(blocked),
            "point" if act.point is not None else "page-script",
            passed,
            _changes(data, quiet),
            native_failed=failed,
        )

    @staticmethod
    def _check_typed(act: Act, data: dict[str, Any]) -> None:
        if act.kind == "type" and data.get("stale") in ("field", "grew"):
            raise StaleElementError(_STALE["grew"], "grew")

    async def _look(self, key: str, doc: str, act: Act) -> tuple[dict[str, Any], bool]:
        """Wait until the page is quiet after *act*, then take what the observer saw.

        The page is quiet once nothing changed for :data:`QUIET_S` (and that
        long after the act); a click, or Enter, first waits for the page's
        first change, since a reply from the site may take a moment. It never
        waits longer than :data:`LOOK_CAP_S`. A page that left for another
        document is waited for until it has loaded. Returns the observer's
        report (``{"lost": True}`` once the page is another document) and
        whether the page was quiet.
        """
        loop = asyncio.get_running_loop()
        start = loop.time()
        first = act.kind == "click" or act.enter
        probe = SETTLE_JS % _data({"key": key, "doc": doc})
        quiet = True
        while True:
            if LOOK_STEP_S:
                await asyncio.sleep(LOOK_STEP_S)
            try:
                now = await self._eval(probe)
            except BrowserSetupError:
                raise
            except BrowserError:
                now = {"doc": False}  # a page between two documents does not answer
            spent = loop.time() - start
            if not isinstance(now, dict) or now.get("doc") is False or now.get("lost"):
                with contextlib.suppress(BrowserError):
                    await self._ready()
                return {"lost": True}, True
            if now.get("moved") is True:
                break  # a real click or key press was stopped: nothing more will happen
            changed = bool(now.get("count")) or now.get("target") is True
            still = _number(now.get("quiet")) >= QUIET_S * 1000
            if spent >= QUIET_S and still and (changed or not first):
                break
            if spent >= LOOK_CAP_S:
                quiet = still or not changed
                break
        return await self._finish(key, doc), quiet

    async def _finish(self, key: str, doc: str) -> dict[str, Any]:
        """The observer's report under *key*, its guards taken down."""
        data = await self._eval(DISARM_JS % _data({"key": key, "doc": doc}))
        return data if isinstance(data, dict) else {"lost": True}


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
        # The address was checked before the script ran. If the page is no longer
        # there, what the script read belongs to a page nobody checked.
        if url and not same_page(url, data.get("url")):
            raise BrowserError(
                "The page moved while melo was reading it, so melo did not read it. "
                "Use browser_read to try again."
            )
        elements = [e for e in map(_element, data.get("elements") or []) if e]
        skipped, frames = data.get("skipped"), data.get("frames")
        unreached = data.get("unreached")
        boxes = (
            tuple(dict.fromkeys(str(f) for f in unreached))[:10]
            if isinstance(unreached, list)
            else ()
        )
        if boxes and self._has("content"):
            # A frame from another site: its controls as the browser's own reading
            # names them, acted on by that reading's node ids.
            reached = foreign_elements(
                await self._nodes("entire_page", frames=True), boxes
            )
            elements += reached
            got = {e.site for e in reached}
            boxes = tuple(b for b in boxes if sites_origin(b) not in got)
        return RawPage(
            url,
            str(data.get("title") or ""),
            str(data.get("text") or ""),
            tuple(elements),
            skipped=skipped if isinstance(skipped, int) and skipped > 0 else 0,
            doc=str(data.get("doc") or ""),
            frames=tuple(str(f) for f in frames) if isinstance(frames, list) else (),
            unreached=boxes,
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
        probe = wait_probe(until, text, origin)
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while True:
            if await self._eval(probe) is True:
                return True
            if loop.time() >= end:
                return False
            await asyncio.sleep(POLL_S)

    async def holds(
        self, tab: str | None, until: str, text: str, *, origin: str = ""
    ) -> bool:
        await self._focus(tab)
        try:
            return await self._eval(wait_probe(until, text, origin)) is True
        except BrowserSetupError:
            raise
        except BrowserError:
            return False  # a page between two documents does not answer yet

    async def screenshot(
        self, tab: str | None, mark: tuple[View, float, float] | None = None
    ) -> Shot:
        await self._focus(tab)
        if await self._eval(FOCUS_JS) is True:
            raise BrowserError(
                "A password, code or card field has focus, so melo took no picture."
            )
        view = _view(await self._eval(VIEW_JS))
        ring = False
        if mark is not None and view is not None:
            with contextlib.suppress(BrowserError):
                ring = await self._eval(mark_script(self._ring, *mark, on=True)) is True
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
            if ring and mark is not None:
                with contextlib.suppress(BrowserError):
                    await self._eval(mark_script(self._ring, *mark, on=False))

    def _folder(self) -> str:
        """melo's private folder for the files Safari saves for it (mode 0700)."""
        if self._shots is None:
            self._shots = tempfile.mkdtemp(prefix="melo-safari-")
        return self._shots

    async def hit(self, tab: str | None, view: View, x: float, y: float) -> PointHit:
        await self._focus(tab)
        spec = {"url": view.url, "doc": view.doc, "sx": view.x, "sy": view.y}
        data = await self._eval(hit_script({**spec, "x": x, "y": y}))
        if not isinstance(data, dict):
            raise BrowserError("Safari returned an answer melo could not read.")
        if data.get("stale"):
            why = str(data["stale"])
            raise StaleElementError(_STALE.get(why, _STALE["gone"]), why)
        surface = data.get("surface")
        if surface not in ("control", "canvas", "frame"):
            return PointHit("none")
        item = _element(data.get("el"))
        site = str(data.get("site") or "") if surface == "frame" else ""
        mx, my = data.get("x"), data.get("y")
        moved = isinstance(mx, int | float) and isinstance(my, int | float)
        return PointHit(
            cast(Any, surface),
            item,
            site=site,
            consent=fields.consent(str(data.get("words") or "")),
            at=(_number(mx), _number(my)) if moved else None,
        )

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

    async def _settled_blank(self, tab: str | None) -> bool:
        """Whether the browser says *tab* is empty now (it may take a moment to say so)."""
        for attempt in range(BLANK_TRIES):
            if attempt:
                await asyncio.sleep(POLL_S)
            with contextlib.suppress(BrowserError):
                if await self._address(tab) == "about:blank":
                    return True
        return False

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
        spec = _spec(act)
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
            return _PASSED["tool"], False
        for attempt in range(2):
            uid = ""
            if act.point is None:
                if act.element.frame:
                    return _PASSED["frame"], False
                if not self._has("content"):
                    return _PASSED["tool"], False
                ready = await self._scrolled(spec)
                if ready.get("covered") is True or ready.get("shifted") is True:
                    return _covered_why(ready.get("by")), False
                uid = bind(ready, await self._nodes())
                if not uid:
                    return _PASSED["name"], False
            key = _key()
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
                if attempt == 0 and act.point is None and _nothing_reached(data):
                    continue  # the page moved under the click: scroll to it again
                raise _cover_stop(data.get("movedBy"), act.element.kind)
            break
        if not _reached(act, data, frame=armed.get("frame") is True):
            if act.point is not None and armed.get("frame") is True:
                raise BrowserError(_STALE["reach"])
            return _PASSED["missed"], True
        self._check_typed(act, data)
        return ActResult(
            _stopped(blocked),
            "point" if act.point is not None else "browser",
            "",
            _changes(data, quiet),
        )

    async def _foreign_act(self, act: Act) -> ActResult:
        """An act on a control in a frame from another site, by the browser's own input.

        No page script reaches into that frame, so the checks are the
        browser's: the node must still be in its reading with the same tag and
        words, and the browser acts on it. melo asked the person first.
        """
        if not self._has("interact") or not self._has("content"):
            raise BrowserError(_STALE["reach"])
        if await self._address(None) != act.url:
            raise StaleElementError(_STALE["page"], "page")
        uid = act.element.frame[1:]
        now = foreign_elements(
            await self._nodes("entire_page", frames=True), (act.element.site,)
        )
        if not any(e.frame == act.element.frame and e == act.element for e in now):
            raise StaleElementError(_STALE["gone"], "gone")
        try:
            await self._call_with(
                "interact", {"interactions": [_interaction(act, uid, {})]}
            )
        except BrowserSetupError:
            raise
        except BrowserError as exc:
            if "stale" in str(exc).lower():
                raise StaleElementError(_STALE["gone"], "gone") from None
            raise
        await asyncio.sleep(QUIET_S)
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
    data = _decode(reply)
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


def wait_probe(until: str, text: str, origin: str = "") -> str:
    """The script that says whether what ``wait`` waits for (*until*: ``load``, ``text``, ``gone``) holds now.

    A page with no body yet (a bot check or an interstitial between two
    documents) is not loaded: nothing on it counts. With *origin* the
    condition counts only while the page is on that origin.
    """
    words = json.dumps(text)
    probe = {
        "load": "document.readyState === 'complete' && !!document.body",
        "text": f"!!document.body && (document.body.innerText || '').indexOf({words}) >= 0",
        "gone": f"!!document.body && (document.body.innerText || '').indexOf({words}) < 0",
    }[until]
    if origin:
        probe = f"location.origin === {json.dumps(origin)} && ({probe})"
    return probe


def _key() -> str:
    """A fresh name for what an act script leaves on the page for the look after it."""
    return f"__melo_{uuid.uuid4().hex}"


def _spec(act: Act) -> dict[str, Any]:
    """*act* as the plain data the act script checks against."""
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
        "text": act.text,
        "enter": act.enter,
        "send": act.send,
        "frame": item.frame,
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
    if act.point is not None and act.view is not None:
        spec["point"] = {
            "x": act.point[0],
            "y": act.point[1],
            "sx": act.view.x,
            "sy": act.view.y,
        }
    return spec


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


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _reached(act: Act, data: dict[str, Any], *, frame: bool = False) -> bool:
    """Whether the browser's own input shows on the page: events on the element, a change, a new document.

    A click inside a frame from another site leaves nothing melo can see, so
    it counts as done.
    """
    if data.get("lost") or frame:
        return True
    seen = data.get("seen")
    seen = seen if isinstance(seen, dict) else {}
    kinds = {
        "click": ("pointerdown", "mousedown", "click"),
        "type": ("keydown", "beforeinput", "input"),
        "select": ("input", "change"),
    }[act.kind]
    if any(_number(seen.get(k)) > 0 for k in kinds):
        return True
    before, after = _decode(str(data.get("before") or "")), _decode(
        str(data.get("after") or "")
    )
    return before != after or _number(data.get("count")) > 0


def _cover_words(info: Any) -> str:
    """What stands over an element, as the page script told it: ``a sticky header (h1 'Daily')``; ``""`` when it did not say."""
    if not isinstance(info, dict):
        return ""
    what = render.field_text(str(info.get("what") or ""), 40)
    detail = render.field_text(str(info.get("detail") or ""), 80)
    if not what:
        return ""
    return f"{what} ({detail})" if detail else what


def _covered_why(by: Any) -> str:
    """Why the browser's own input was passed over for an element something covers."""
    words = _cover_words(by)
    return (
        f"{words} covers it, so only the page script reaches it"
        if words
        else _PASSED["covered"]
    )


def _cover_stop(by: Any, kind: str = "") -> CoveredError:
    """The error for a real click or key press that landed on what covers the element, saying what that is."""
    words = _cover_words(by)
    if not words:
        return CoveredError(_MOVED)
    noun = _NOUN.get(kind, kind or "element")
    return CoveredError(
        f"{words[:1].upper()}{words[1:]} covers that {noun}, so a real click or key "
        "press would land on it, and melo stopped it. Nothing was clicked or typed, "
        "and the page and its refs are as they were: close or move what covers it "
        "(its close button, say), or ask the person. Do not open the page again."
    )


def _nothing_reached(data: dict[str, Any]) -> bool:
    """Whether no event reached the element (the stop came before anything happened)."""
    seen = data.get("seen")
    seen = seen if isinstance(seen, dict) else {}
    return not any(_number(v) > 0 for v in seen.values())


def _stopped(blocked: list[str]) -> str:
    return " ".join(dict.fromkeys(_BLOCKED[k] for k in blocked if k in _BLOCKED))


_STATE_WORDS = {
    "pressed": ("pressed", "not pressed"),
    "selected": ("selected", "not selected"),
    "expanded": ("expanded", "collapsed"),
    "current": ("current", "not current"),
}


def _target_words(before: Any, after: Any) -> str:
    """How the acted-on element changed, in words (``""`` when it did not, or cannot be told)."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return ""
    if before.get("gone"):
        return ""
    if after.get("gone"):
        return "gone from the page"
    words: list[str] = []
    if before.get("shown") and after.get("shown") is False:
        words.append("hidden")
    if bool(before.get("disabled")) != bool(after.get("disabled")):
        words.append("disabled" if after.get("disabled") else "enabled")
    if before.get("checked") != after.get("checked") and "checked" in after:
        state = after["checked"]
        words.append(
            "partly checked"
            if state == "mixed"
            else "checked" if state in (True, "true") else "unchecked"
        )
    for key, (yes, no) in _STATE_WORDS.items():
        if before.get(key) != after.get(key):
            words.append(yes if after.get(key) == "true" else no)
    if "value" in after and before.get("value") != after.get("value"):
        words.append(f'value "{str(after["value"])[:60]}"')
    if "label" in after and before.get("label") != after.get("label"):
        words.append(f'reads "{str(after["label"])[:60]}"')
    return "now " + ", ".join(words) if words else ""


def _changes(data: dict[str, Any], quiet: bool) -> Changes | None:
    """What the observer saw, or None when the page became another document."""
    if data.get("lost"):
        return None
    texts = data.get("texts")
    shown = (
        tuple(str(t)[:200] for t in texts if isinstance(t, str))[:8]
        if isinstance(texts, list)
        else ()
    )
    before = _decode(str(data.get("before") or ""))
    after = _decode(str(data.get("after") or ""))
    changed = _number(data.get("count")) > 0 or before != after
    return Changes(shown, _target_words(before, after), quiet, changed)


def _view(data: Any) -> View | None:
    """The view :data:`VIEW_JS` read, or None when it read nothing usable."""
    if not isinstance(data, dict) or not isinstance(data.get("doc"), str):
        return None
    width, height = _number(data.get("w")), _number(data.get("h"))
    if width <= 0 or height <= 0:
        return None
    ratio = _number(data.get("dpr"))
    return View(
        str(data.get("url") or ""),
        data["doc"],
        width,
        height,
        _number(data.get("x")),
        _number(data.get("y")),
        ratio if ratio > 0 else 2.0,
    )


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
