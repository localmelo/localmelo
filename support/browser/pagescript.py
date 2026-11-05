"""The page scripts both browser drivers run, and what a driver does with their answers.

The Safari and the Chrome drivers read a page, find an element again, act on it
and look at the page afterwards with the same JavaScript, so an element is found,
checked, guarded and looked at by one set of rules whichever browser does it:

- :data:`PAGE_JS` reads a page (its text, links, buttons and fields),
  :func:`hit_script` says what is at a point of a picture, :func:`act_script`
  checks and does, or prepares, one act, :data:`VIEW_JS`, :func:`mark_script` and
  :data:`FOCUS_JS` serve the pictures, and :func:`wait_probe` is what a wait polls for.
- The functions after them turn a script's answer into melo's own types
  (:func:`observed` and the element and the view of a page) or into the words and
  errors an act ends with (:func:`cover_stop`, :func:`stopped`).
- :class:`PageScriptActs` is the base both drivers share: it reads, waits for,
  pictures and acts on a page with these scripts, and a driver gives it the few
  things only its browser knows.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from typing import Any, cast
from urllib.parse import urlsplit

from localmelo.support.browser import fields
from localmelo.support.browser.driver import (
    Act,
    ActResult,
    BrowserError,
    BrowserSetupError,
    Changes,
    CoveredError,
    Element,
    FieldAttrs,
    PointHit,
    RawPage,
    Sensitive,
    Shot,
    StaleElementError,
    View,
)

KINDS = ("link", "button", "textbox", "checkbox", "select")
SECRETS = ("password", "code", "card")
# How long a page may take to load, and how often it is looked at while melo waits
# for it (also the number of looks when a page is being emptied).
SETTLE_S = 15.0
POLL_S = 0.4
BLANK_TRIES = 4
# A page that is still settling (a bar sliding in, an ad loading) gets this long
# before an element found covered is looked at once more.
COVER_AGAIN_S = 0.25
# Looking at the page after an act: a poll this often, done once nothing changed for
# QUIET_S (a click waits up to LOOK_CAP_S for the page's first change), never longer.
LOOK_STEP_S = 0.1
QUIET_S = 0.3
LOOK_CAP_S = 2.0
# What an act says when the page script refuses it (``stale`` in its answer).
STALE = {
    "page": "The page is not the one that was read (its address changed). Use browser_read for current refs.",
    "doc": "The page was reloaded or replaced since it was read. Use browser_read for current refs.",
    "gone": "That element is no longer on the page as it was read. Use browser_read for current refs.",
    "dest": "Where that element leads or sends changed since it was read, so melo did not use it. Use browser_read for current refs.",
    "field": "That field changed since it was read, so melo did not use it. Use browser_read for current refs.",
    "grew": "That field turned into a protected one while melo typed, so melo cleared it. Ask the person to fill it in themselves.",
    "frame": "melo could not set up its safe page checks on this page, so it did nothing. Use browser_read to try again.",
    "point": "What is at that point is not what melo found there a moment ago, so it did not click. Take a new browser_screenshot.",
    "scrolled": "The page scrolled since the picture, so a point on it is no longer where it was. Take a new browser_screenshot.",
    "reach": "That point is inside a box from another site, and melo's browser cannot click there.",
}
# Why the browser's own input was passed over for the page script (the result says it).
PASSED_OVER = {
    "tool": "melo's browser cannot act on page elements itself",
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
NO_CHECKS_NOTE = (
    "(melo could not set up its safe page checks on this page, so no links, buttons "
    "or fields are listed.)"
)
_BLOCKED = {
    "send": "The page tried to send a form that melo had not asked about, so melo stopped it. To send a form, click its button: melo asks first.",
    "link": "The page changed where that link goes while it was clicked, so melo stopped it.",
}

# What may be said of an element that stands over another: its tag when it is one of these
# (any other is "element"), and one of these kinds. The page script builds its answer from
# the same list, and the words are checked against it again here, so nothing the page wrote
# (its text, a custom tag, a frame's title) reaches the model through them.
COVER_TAGS = [
    "html",
    "body",
    "header",
    "footer",
    "nav",
    "main",
    "aside",
    "section",
    "article",
    "div",
    "span",
    "p",
    "a",
    "b",
    "i",
    "em",
    "strong",
    "small",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "table",
    "tr",
    "td",
    "form",
    "label",
    "button",
    "input",
    "textarea",
    "select",
    "img",
    "svg",
    "canvas",
    "video",
    "picture",
    "figure",
    "dialog",
    "center",
    "iframe",
]


COVER_KINDS = frozenset(
    [
        "an ad",
        "an ad at the top",
        "an ad at the bottom",
        "a dialog",
        "a frame",
        "another element",
    ]
    + [
        f"a {how} {place}"
        for how in ("sticky", "fixed")
        for place in ("header", "bar at the bottom", "layer")
    ]
)


_COVER_HOST = re.compile(r"^iframe(?: from [a-z0-9.-]{1,60})?$", re.IGNORECASE)


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
// what it is, in words for the model: {what: "a sticky header", detail: "h1"}. Never a word
// the page wrote: a bar is told by its place in the view, an element by its tag when that is
// a standard one, a frame by the host of its address. An ad is one that says so in its id,
// class or title (only that yes or no is used).
var AD_WORDS = /(^|[^a-z])(ads?|adverts?|advertisements?|sponsored|banners?)([^a-z]|$)/i;
var COVER_TAGS = new RegExp(__COVER_TAGS__);
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
    if (!/^[a-z0-9.-]{1,60}$/i.test(host)) host = "";
    return {what: adLike ? "an ad" : "a frame", detail: "iframe" + (host ? " from " + host : "")};
  }
  var detail = COVER_TAGS.test(tag) ? tag : "element";
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
).replace("__COVER_TAGS__", json.dumps("^(?:" + "|".join(COVER_TAGS) + ")$"))


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


def decode(text: str) -> Any:
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


def _view(data: Any) -> View | None:
    """The view :data:`VIEW_JS` read, or None when it read nothing usable."""
    if not isinstance(data, dict) or not isinstance(data.get("doc"), str):
        return None
    width, height = as_number(data.get("w")), as_number(data.get("h"))
    if width <= 0 or height <= 0:
        return None
    ratio = as_number(data.get("dpr"))
    return View(
        str(data.get("url") or ""),
        data["doc"],
        width,
        height,
        as_number(data.get("x")),
        as_number(data.get("y")),
        ratio if ratio > 0 else 2.0,
    )


def as_number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def new_key() -> str:
    """A fresh name for what an act script leaves on the page for the look after it."""
    return f"__melo_{uuid.uuid4().hex}"


def act_spec(act: Act) -> dict[str, Any]:
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


def reached(act: Act, data: dict[str, Any], *, frame: bool = False) -> bool:
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
    if any(as_number(seen.get(k)) > 0 for k in kinds):
        return True
    before, after = decode(str(data.get("before") or "")), decode(
        str(data.get("after") or "")
    )
    return before != after or as_number(data.get("count")) > 0


def nothing_reached(data: dict[str, Any]) -> bool:
    """Whether no event reached the element (the stop came before anything happened)."""
    seen = data.get("seen")
    seen = seen if isinstance(seen, dict) else {}
    return not any(as_number(v) > 0 for v in seen.values())


def _cover_words(info: Any) -> str:
    """What stands over an element, as the page script told it: ``a sticky header (h1)``; ``""`` when it did not say.

    Only words from a fixed list pass (see :data:`COVER_KINDS`): the kind, and
    the tag of a standard element or ``iframe from <host>``. What else the page
    sent is left out, so a page cannot put its own words into the answer.
    """
    if not isinstance(info, dict):
        return ""
    what = info.get("what")
    if not isinstance(what, str) or what not in COVER_KINDS:
        return ""
    detail = info.get("detail")
    if not isinstance(detail, str) or not (
        detail in COVER_TAGS or detail == "element" or _COVER_HOST.match(detail)
    ):
        return what
    return f"{what} ({detail})"


def covered_why(by: Any) -> str:
    """Why the browser's own input was passed over for an element something covers."""
    words = _cover_words(by)
    return (
        f"{words} covers it, so only the page script reaches it"
        if words
        else PASSED_OVER["covered"]
    )


def cover_stop(by: Any, kind: str = "") -> CoveredError:
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


def stopped(blocked: list[str]) -> str:
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


def observed(data: dict[str, Any], quiet: bool) -> Changes | None:
    """What the observer saw, or None when the page became another document."""
    if data.get("lost"):
        return None
    texts = data.get("texts")
    shown = (
        tuple(str(t)[:200] for t in texts if isinstance(t, str))[:8]
        if isinstance(texts, list)
        else ()
    )
    before = decode(str(data.get("before") or ""))
    after = decode(str(data.get("after") or ""))
    changed = as_number(data.get("count")) > 0 or before != after
    return Changes(shown, _target_words(before, after), quiet, changed)


def _stale(why: str) -> StaleElementError:
    """The error for a page script that found the page or the element not as they were read."""
    return StaleElementError(STALE.get(why, STALE["gone"]), why)


class PageScriptActs:
    """What a driver that runs melo's page scripts in the page does with them.

    A driver gives the few things only its browser knows: :meth:`_eval` (a
    script run in the page of the tab it means now, its value decoded),
    :meth:`_focus` and :meth:`_address` (the tab the scripts run in, and the
    address the browser itself reports for it), :meth:`_capture` (one picture
    of the page) and :attr:`label` (the browser's name in what melo says).
    Where reading must not trust the page's own scripts, it also gives
    :meth:`_read` (a script run where the page's scripts cannot reach); where
    a wait should go on through a page that does not answer (between two
    documents), :meth:`_probe`; and a browser that lists frames from other
    sites in its own words gives :meth:`_other_sites`.

    This base then reads a page (:meth:`page`), says what is at a point
    (:meth:`hit`), waits for a page (:meth:`wait`), takes a picture with the
    ring and the checks around it (:meth:`screenshot`), runs an act script and
    reads its answer (:meth:`_script`), acts by the page script
    (:meth:`_scripted`) and looks at the page after an act until it is quiet
    (:meth:`_look`). An act is checked, guarded and looked at by the same
    rules whichever browser does it.
    """

    label = "Safari"
    _ring: str

    async def _eval(self, source: str) -> Any:
        raise NotImplementedError

    async def _read(self, source: str) -> Any:
        """Run a script that only reads the page. By default it is run like any other."""
        return await self._eval(source)

    async def _focus(self, tab: str | None) -> object:
        """Make *tab* the tab the scripts run in."""
        raise NotImplementedError

    async def _address(self, tab: str | None) -> str:
        """The address the browser reports for *tab* (``""`` when it cannot say)."""
        raise NotImplementedError

    async def _capture(self, view: View | None) -> Shot:
        """One picture of the page in the tab that was focused, taken by the browser."""
        raise NotImplementedError

    async def _other_sites(
        self, boxes: tuple[str, ...]
    ) -> tuple[list[Element], tuple[str, ...]]:
        """The controls in the frames from other sites of *boxes* that the browser can name, and the frames left."""
        return [], boxes

    async def page(self, tab: str | None) -> RawPage:
        await self._focus(tab)
        url = await self._address(tab)
        data = await self._read(PAGE_JS)
        if not isinstance(data, dict):
            raise BrowserError(f"{self.label} returned a page melo could not read.")
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
        framed, boxes = await self._other_sites(boxes)
        return RawPage(
            url,
            str(data.get("title") or ""),
            str(data.get("text") or ""),
            tuple(elements + framed),
            skipped=skipped if isinstance(skipped, int) and skipped > 0 else 0,
            doc=str(data.get("doc") or ""),
            frames=tuple(str(f) for f in frames) if isinstance(frames, list) else (),
            unreached=boxes,
        )

    async def hit(self, tab: str | None, view: View, x: float, y: float) -> PointHit:
        await self._focus(tab)
        spec = {"url": view.url, "doc": view.doc, "sx": view.x, "sy": view.y}
        data = await self._read(hit_script({**spec, "x": x, "y": y}))
        if not isinstance(data, dict):
            raise BrowserError(f"{self.label} returned an answer melo could not read.")
        if data.get("stale"):
            raise _stale(str(data["stale"]))
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
            at=(as_number(mx), as_number(my)) if moved else None,
        )

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
            if await self._probe(probe):
                return True
            if loop.time() >= end:
                return False
            await asyncio.sleep(POLL_S)

    async def _probe(self, source: str) -> bool:
        """Whether the wait probe *source* is true on the page now. An error ends the wait."""
        return await self._read(source) is True

    async def screenshot(
        self, tab: str | None, mark: tuple[View, float, float] | None = None
    ) -> Shot:
        await self._focus(tab)
        if await self._read(FOCUS_JS) is True:
            raise BrowserError(
                "A password, code or card field has focus, so melo took no picture."
            )
        seen = _view(await self._read(VIEW_JS))
        ring = False
        if mark is not None and seen is not None:
            with contextlib.suppress(BrowserError):
                ring = await self._read(mark_script(self._ring, *mark, on=True)) is True
        try:
            return await self._capture(seen)
        finally:
            if ring and mark is not None:
                with contextlib.suppress(BrowserError):
                    await self._read(mark_script(self._ring, *mark, on=False))

    async def _settled_blank(self, tab: str | None) -> bool:
        """Whether the browser says *tab* is empty now (it may take a moment to say so)."""
        for attempt in range(BLANK_TRIES):
            if attempt:
                await asyncio.sleep(POLL_S)
            with contextlib.suppress(BrowserError):
                if await self._address(tab) == "about:blank":
                    return True
        return False

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
                raise _stale(str(result["stale"]))
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
        key = new_key()
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
            stopped(blocked),
            "point" if act.point is not None else "page-script",
            passed,
            observed(data, quiet),
            native_failed=failed,
        )

    @staticmethod
    def _check_typed(act: Act, data: dict[str, Any]) -> None:
        if act.kind == "type" and data.get("stale") in ("field", "grew"):
            raise StaleElementError(STALE["grew"], "grew")

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
            still = as_number(now.get("quiet")) >= QUIET_S * 1000
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
