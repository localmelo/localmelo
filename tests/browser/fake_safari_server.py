"""A fake ``safaridriver --mcp`` for the localmelo tests (stdlib only).

It speaks newline-delimited JSON-RPC like the real server, lists its 17 tools
as ``safaridriver --mcp`` listed them on Safari 27.2 (recorded in
``fixtures/safari_tools_list.json``) and understands just enough of the scripts
melo sends to answer from a tiny set of pages. ``screenshot`` answers as the real one does: text only,
``Saved screenshot to '<path>' (N B).``, with the PNG in the file named by
``savePath``. Modes (the first argument, or
``FAKE_SAFARI_MODE`` in the environment):

- ``automation_off``: every tool call fails as Safari does when remote
  automation is off
- ``exit_off``: write that message to stderr and exit before ``initialize``
- ``no_script``: no ``evaluate_javascript`` tool
- ``return_style``: a script runs only with a leading ``return``
- ``alt_names``: other argument names (``address``, ``javascript``, ``handle``)
- ``slow_load``: ``document.readyState`` says ``loading`` twice first
- ``no_body``: the page has no ``document.body`` for its first two looks (a bot
  check between two documents): a script that reads ``document.body.innerText``
  then fails as Safari's does, unless it checks for the body first
- ``focus_secret``: a password field has focus
- ``inline_image``: ``screenshot`` answers with an image part instead of a file
- ``decoy``: ``screenshot`` ignores ``savePath`` and names the file in
  ``FAKE_SAFARI_DECOY``, which exists, in its reply
- ``symlink``: ``screenshot`` puts a link to ``FAKE_SAFARI_DECOY`` at ``savePath``
- ``shot_error``: ``screenshot`` saves the file, then fails
- ``long_ids``: tab ids as long as the real server's (``page-<UUID>``, 41 characters)
- ``act_stale``: a script that acts on an element answers that the page changed
- ``act_option``: a script that chooses an option answers that none matched
- ``act_reloaded``: the page was reloaded since it was read (its document differs)
- ``act_moved``: the element is no longer the one that was read (its fingerprint differs)
- ``no_active``: ``list_tabs`` rows say nothing of which tab is active
- ``no_url``: ``list_tabs`` rows carry no address or title
- ``no_ids``: ``list_tabs`` rows carry no handle (only their place in the list)
- ``focus_stays``: ``create_tab`` leaves the old tab current, as a browser may
- ``page_moves``: the first page-reading script finds the tab on another address
  than ``list_tabs`` just said (a page that navigates between the two calls)
- ``replace_fails``: ``location.replace('about:blank')`` does nothing, so only the
  browser's own navigation can leave a page
- ``no_native``: no ``page_interactions`` tool
- ``no_uids``: ``get_page_content`` names no nodes
- ``twins``: the browser's own reading names two nodes like each element
- ``native_error``: ``page_interactions`` fails and does nothing
- ``native_missed``: ``page_interactions`` answers but its input reaches nothing
- ``native_redirect``: a real click lands on another element, which melo's guard stops
  (every time; the guard says it was a sticky header)
- ``native_redirect_once``: only the first real click lands on another element

A script that acts (melo's ``act_script``) is answered from the tiny page
model too: an element may say ``goes`` (the address a click lands on), and a
typed text is kept in ``typed``. Every page read has the document ``DOC`` and
an element may have a fingerprint ``fp``; an act that names another is stale.
The browser's own reading names the node of element ``at`` ``100 + at``, in the
real server's ``textTree`` lines; its own input acts on that node and is recorded in
``native`` (and in ``FAKE_SAFARI_LOG``, a file a test reads); an element that
says ``trusted_only`` ignores the page script's click, ``notice`` is text the
page shows after a click, ``redraw`` makes every click redraw the page (new
fingerprints), ``covered`` puts another element over it (``by`` says which, as
the page script tells it; ``shifted`` says only a part of it is clear). A page's ``points``
say what is at a point of the view (``"x,y"``).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODE = os.environ.get("FAKE_SAFARI_MODE") or (
    sys.argv[1] if len(sys.argv) > 2 else ""  # `fake_safari_server.py MODE --mcp`
)
OFF = (
    "Could not create a session: You must enable the 'Allow Remote Automation' "
    "option in Safari's Develop menu to control Safari via WebDriver."
)
PNG = base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
        "AAAABJRU5ErkJggg=="
    )
).decode()

PAGES: dict[str, dict[str, Any]] = {
    "https://example.com/": {
        "title": "Example Home",
        "text": "Welcome to the example.\nSecond line, with 中文。",
        "elements": [
            {"kind": "link", "label": "News", "href": "https://example.com/news"},
            {"kind": "textbox", "label": "Search", "value": "cats"},
            {"kind": "textbox", "label": "", "value": "", "sensitive": "password"},
            {"kind": "button", "label": "Go"},
            {"kind": "bogus", "label": "ignored"},
        ],
    },
    "https://example.com/news": {
        "title": "News",
        "text": "Today's news.",
        "elements": [],
    },
    "https://example.com/form": {
        "title": "Form",
        "text": "A form.",
        "skipped": 2,
        "elements": [
            {
                "kind": "button",
                "label": "Check in",
                "also": "Daily check-in",
                "fp": "button|check in|0",
                "goes": "https://example.com/done",
            },
            {
                "kind": "textbox",
                "label": "Nickname",
                "value": "",
                "submit": "https://example.com/save",
                "attrs": {
                    "type": "text",
                    "autocomplete": "",
                    "words": "nick Nickname",
                    "inputmode": "",
                    "maxlength": 40,
                    "masked": False,
                },
            },
            {
                "kind": "select",
                "label": "Country",
                "value": "France",
                "options": ["France", "Spain"],
            },
        ],
    },
    "https://example.com/done": {
        "title": "Done",
        "text": "Checked in.",
        "elements": [],
    },
    # A page whose check-in ignores page-script clicks and shows a notice after a
    # real one; its mood button redraws the page on every click.
    "https://example.com/daily": {
        "title": "Daily",
        "text": "Daily check-in.",
        "elements": [
            {
                "kind": "button",
                "label": "Happy",
                "fp": "button|happy|1",
                "redraw": True,
            },
            {
                "kind": "button",
                "label": "Check in",
                "fp": "button|check in|1",
                "trusted_only": True,
                "notice": "Checked in: day 12",
            },
            {
                "kind": "button",
                "label": "Hidden away",
                "fp": "button|hidden|1",
                "covered": True,
                "by": {"what": "an ad", "detail": "iframe from ads.example.net"},
            },
            {
                "kind": "button",
                "label": "Half hidden",
                "fp": "button|half|1",
                "shifted": True,
                "by": {"what": "a sticky header", "detail": "h1"},
            },
        ],
        "points": {
            "300,100": {
                "surface": "canvas",
                "el": {"kind": "button", "label": "", "at": -1, "fp": "canvas|1"},
                "words": "",
            },
            "500,250": {
                "surface": "frame",
                "site": "https://accounts.example.org/box",
                "el": {"kind": "button", "label": "", "fp": "iframe|1"},
            },
        },
    },
    # A sign-in box from another site: the page script cannot read into it, the
    # browser's own reading lists its controls.
    "https://example.com/box": {
        "title": "Box",
        "text": "Sign in to post.",
        "elements": [{"kind": "button", "label": "Post", "fp": "button|post|1"}],
        "unreached": ["https://accounts.example.org/login"],
        "box": [
            "\tiframe uid=88 title='Sign-in box' origin=accounts.example.org",
            "\t\tinput uid=2_1_89 type=email label=Email name=email ''",
            "\t\tinput uid=2_1_90 type=password label=Password name=pw ''",
            "\t\tinput uid=2_1_91 label=Answer name=a ''",
            "\t\tbutton uid=2_1_92 'Continue'",
        ],
    },
}
TOOLS = json.loads(
    (Path(__file__).parent / "fixtures" / "safari_tools_list.json").read_text("utf-8")
)


def tab_id(n: int) -> str:
    if MODE == "long_ids":
        return f"page-635DB8F8-27ED-4CA5-81A2-7EAEE1DF{n:04X}"
    return f"page-{n}"


tabs: list[dict[str, Any]] = [{"id": tab_id(1), "history": ["about:blank"]}]
typed: dict[str, str] = {}
current = 0
polls = 0
moved = False
body_looks = 0
STYLE_RETURN = MODE == "return_style"
redraws: dict[str, int] = {}
looks: dict[str, dict[str, Any]] = {}
redirected: list[int] = []  # the real clicks the page has already moved elsewhere
armed = ""
native: list[str] = []
LOG = os.environ.get("FAKE_SAFARI_LOG", "")


def log(line: str) -> None:
    native.append(line)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


RENAMED = {"url": "address", "expression": "javascript", "handle": "tab_id"}


def tool_list() -> list[dict[str, Any]]:
    """The real server's tools, as recorded; some modes rename or drop a few."""
    tools = json.loads(json.dumps(TOOLS["tools"]))
    if MODE == "alt_names":
        for tool in tools:
            if tool["name"] in (
                "navigate_to_url",
                "evaluate_javascript",
                "switch_tab",
                "close_tab",
            ):
                props = tool["inputSchema"]["properties"]
                for old, new in RENAMED.items():
                    if old in props:
                        props[new] = props.pop(old)
    dropped = {"no_script": "evaluate_javascript", "no_native": "page_interactions"}
    return [t for t in tools if t["name"] != dropped.get(MODE)]


def text(value: str, error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": value}]}
    if error:
        result["isError"] = True
    return result


DOC = "doc-1"


def fp_of(url: str, element: dict[str, Any]) -> str:
    """An element's fingerprint now: a page that redrew gives every element a new one."""
    fp = str(element.get("fp", ""))
    return fp + (f"~{redraws[url]}" if redraws.get(url) and fp else "")


def page_now() -> dict[str, Any]:
    url = tabs[current]["history"][-1]
    found = PAGES.get(url, {"title": "", "text": "", "elements": []})
    keep = (
        "kind",
        "label",
        "href",
        "value",
        "sensitive",
        "also",
        "submit",
        "options",
        "attrs",
        "checked",
    )
    elements = [
        {**{k: e[k] for k in keep if k in e}, "at": i, "fp": fp_of(url, e)}
        for i, e in enumerate(found["elements"])
    ]
    info = {k: v for k, v in found.items() if k not in ("elements", "points", "box")}
    return {"url": url, "doc": DOC, **info, "elements": elements}


def spec_of(source: str) -> tuple[dict[str, Any], str]:
    """The spec an act or point script carries, and what follows it (the node lookups)."""
    start = source.rindex('})({"')
    spec, end = json.JSONDecoder().raw_decode(source, start + 3)
    return spec, source[end:]


def element_of(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The element *spec* names on the current page, or why it is stale."""
    here = tabs[current]["history"]
    if MODE == "act_stale":
        return None, "gone"
    if spec["url"] != here[-1]:
        return None, "page"
    if MODE == "act_reloaded" or spec.get("doc") != DOC:
        return None, "doc"
    elements = PAGES.get(here[-1], {"elements": []})["elements"]
    if not 0 <= spec["at"] < len(elements):
        return None, "gone"
    element = elements[spec["at"]]
    if (element["kind"], element.get("label", ""), element.get("href", "")) != (
        spec["kind"],
        spec["label"],
        spec["href"],
    ):
        return None, "gone"
    if MODE == "act_moved" or spec.get("fp", "") != fp_of(here[-1], element):
        return None, "gone"
    return element, ""


def acted(element: dict[str, Any], key: str, *, real: bool) -> None:
    """What a click does on the page model: lands, shows a notice, redraws."""
    here = tabs[current]["history"]
    if element.get("trusted_only") and not real:
        return  # the page ignores a click a script made
    look = looks.setdefault(key, {})
    look.setdefault("seen", {})["click"] = 1
    if element.get("redraw"):
        redraws[here[-1]] = redraws.get(here[-1], 0) + 1
        look["count"] = 1
        look["after"] = json.dumps({"gone": True})
    if element.get("notice"):
        look["texts"] = [element["notice"]]
        look["count"] = 1
    if element.get("goes"):
        here.append(element["goes"])


TAGS = {
    "button": "button",
    "link": "a",
    "textbox": "input",
    "select": "select",
    "checkbox": "input",
}


def do_act(spec: dict[str, Any]) -> dict[str, Any]:
    """What melo's act script answers, on the page model above."""
    global armed
    if spec.get("point") and spec["at"] < 0:
        # A drawing or a box from another site, known by its place on the view.
        here = tabs[current]["history"]
        spot = (
            PAGES.get(here[-1], {})
            .get("points", {})
            .get(f"{spec['point']['x']:g},{spec['point']['y']:g}")
        )
        if spec["url"] != here[-1] or not spot or spot["el"]["fp"] != spec["fp"]:
            return {"ok": False, "stale": "point"}
        key = spec.get("key", "")
        armed = key
        looks[key] = {"doc": DOC, "spec": spec, "url": here[-1], "before": "{}"}
        if spec.get("mode") == "arm":
            return {
                "ok": True,
                "uid": "",
                "bound": 0,
                "frame": spot["surface"] == "frame",
            }
        if spot["surface"] == "frame":
            return {"ok": False, "stale": "reach"}
        looks[key]["seen"] = {"click": 1}
        return {"ok": True}
    element, stale = element_of(spec)
    if element is None:
        return {"ok": False, "stale": stale}
    here = tabs[current]["history"]
    mode, key = spec.get("mode", "script"), spec.get("key", "")
    if mode == "scroll":
        said = {"textbox": "label", "select": "label"}.get(element["kind"], "text")
        return {
            "ok": True,
            "covered": bool(element.get("covered")),
            "shifted": bool(element.get("shifted")),
            "by": element.get("by"),
            "tag": TAGS[element["kind"]],
            said: element.get("label", ""),
        }
    option = ""
    if spec["act"] == "select":
        if MODE == "act_option" or spec["text"] not in element["options"]:
            return {"ok": False, "option": 0}
        option = spec["text"]
    if mode == "arm":
        armed = key
        looks[key] = {
            "doc": DOC,
            "spec": spec,
            "url": here[-1],
            "before": json.dumps({"label": element.get("label", "")}),
        }
        return {"ok": True, "option": option}
    if key:
        looks[key] = {
            "doc": DOC,
            "spec": spec,
            "url": here[-1],
            "before": json.dumps({"label": element.get("label", "")}),
        }
    if spec["act"] == "click":
        acted(element, key, real=False)
    elif spec["act"] == "type":
        typed[element["label"]] = spec["text"] + ("\n" if spec["enter"] else "")
        if spec["enter"] and element.get("submit"):
            here.append(element["submit"])
    elif spec["act"] == "select":
        element["value"] = spec["text"]
    return {"ok": True}


def interact(args: dict[str, Any]) -> dict[str, Any]:
    """The browser's own input, on node ``100 + at`` of the current page, or at a point."""
    item = (args.get("interactions") or [{}])[0]
    if MODE == "native_error":
        return text("Could not perform the interaction.", error=True)
    log(json.dumps(item, sort_keys=True))
    if MODE == "native_missed":
        return text("No changes.")
    here = tabs[current]["history"]
    page = PAGES.get(here[-1], {"elements": []})
    look = looks.setdefault(armed, {})
    if MODE == "native_redirect" or (MODE == "native_redirect_once" and not redirected):
        redirected.append(1)
        look["blocked"] = ["moved"]
        look["moved_by"] = {"what": "a sticky header", "detail": "h1"}
        return text("Clicked.")
    if "point" in item:
        spot = page.get("points", {}).get(
            f"{item['point']['x']:g},{item['point']['y']:g}"
        )
        look.setdefault("seen", {})["click"] = 1
        if spot and spot.get("goes"):
            here.append(spot["goes"])
        return text("Clicked.")
    if "_" in str(item.get("node", "")):
        return text("Clicked in the frame.")  # a node in a frame from another site
    at = int(item.get("node", "0")) - 100
    elements = page["elements"]
    if not 0 <= at < len(elements):
        return text("No node with that uid.", error=True)
    element = elements[at]
    if item["type"] == "click":
        acted(element, armed, real=True)
    elif item["type"] == "type":
        typed[element["label"]] = item["value"] + (
            "\n" if item.get("pressReturn") else ""
        )
        look.setdefault("seen", {})["keydown"] = 1
        if item.get("pressReturn") and element.get("submit"):
            here.append(element["submit"])
    elif item["type"] == "selectMenuItem":
        element["value"] = item["text"]
        look.setdefault("seen", {})["change"] = 1
    return text("Done.")


def content(args: dict[str, Any]) -> dict[str, Any]:
    """The browser's own reading of the view: a line per element, with its node id."""
    page = page_now()
    lines = ["root", f"\t'{page.get('title', '')}'"]
    for e in page["elements"]:
        uid = "" if MODE == "no_uids" else f" uid={100 + e['at']}"
        label = e.get("label", "")
        field = e["kind"] in ("textbox", "select")
        named = f" label={label.replace(' ', '_')}" if field else ""
        said = "" if field else f" '{label}'"
        tag = TAGS.get(e["kind"], "div")
        lines.append(f"\t{tag}{uid}{named}{said}")
        if MODE == "twins":
            lines.append(f"\t{tag} uid={500 + e['at']}{named}{said}")
    box = PAGES.get(page["url"], {}).get("box", [])
    # As the real server: a frame from another site is a line; what is inside it is
    # read only with includeSubframes.
    lines.extend(box if args.get("includeSubframes") is True else box[:1])
    body = json.dumps({"content": "\n".join(lines), "url": page["url"]})
    target = args.get("savePath")
    if target:
        Path(str(target)).write_text(body, encoding="utf-8")
        return text(f"Saved page content to '{target}'.")
    return text(body)


def run_script(source: str) -> Any:
    global polls, moved, body_looks
    if STYLE_RETURN and not source.startswith("return "):
        raise SyntaxError("Unexpected token")
    source = source.removeprefix("return ")
    here = tabs[current]["history"]
    if source.startswith("(function (spec)"):
        spec, _ = spec_of(source)
        return json.dumps(do_act(spec))
    if source.startswith(("(function (settle)", "(function (disarm)")):
        asked = json.loads(source[source.rindex("})(") + 3 : -1])
        look = looks.get(asked["key"])
        if asked["doc"] != DOC or look is None or look.get("url", here[-1]) != here[-1]:
            return {"doc": False} if "settle" in source[:20] else {"lost": True}
        look.setdefault("url", here[-1])
        if "settle" in source[:20]:
            return {
                "doc": True,
                "count": look.get("count", 0),
                "quiet": 5000,
                "target": False,
                "moved": "moved" in look.get("blocked", []),
            }
        looks.pop(asked["key"])
        return {
            "movedBy": look.get("moved_by"),
            "blocked": look.get("blocked", []),
            "texts": look.get("texts", []),
            "before": look.get("before", ""),
            "after": look.get("after", look.get("before", "")),
            "seen": look.get("seen", {}),
            "count": look.get("count", 0),
        }
    if source.startswith("(function (at)"):
        spec, _ = spec_of(source)
        if spec["url"] != here[-1]:
            return {"stale": "page"}
        spot = (
            PAGES.get(here[-1], {})
            .get("points", {})
            .get(f"{spec['x']:g},{spec['y']:g}")
        )
        return spot or {"surface": "none"}
    if source.startswith("(function (view)"):
        return {
            "url": here[-1],
            "doc": DOC,
            "w": 1000,
            "h": 500,
            "x": 0,
            "y": 0,
            "dpr": 2,
        }
    if source.startswith("(function (mark)"):
        log("mark " + source[source.rindex("})(") + 3 : -1])
        return True
    if source == "1+1":
        return 2
    if source == "location.href":
        return here[-1]
    if source == "document.title":
        return page_now().get("title", "")
    if source.startswith("location.origin === "):
        # The wait's guard: the condition holds only on the page's own site.
        wanted, _, source = source.removeprefix("location.origin === ").partition(
            " && "
        )
        parts = urlsplit(here[-1])
        if json.loads(wanted) != f"{parts.scheme}://{parts.netloc}":
            return False
        source = source.removeprefix("(").removesuffix(")")
    if "document.readyState" in source:
        polls += 1
        state = "loading" if MODE == "slow_load" and polls <= 2 else "complete"
        return state if source == "document.readyState" else state == "complete"
    if source.startswith("JSON.stringify({url"):
        return json.dumps({"url": here[-1], "title": page_now().get("title", "")})
    if "getBoundingClientRect" in source:
        if MODE == "page_moves" and not moved:
            moved = True
            here.append("https://example.com/news")  # moved before the script ran
        return json.dumps(page_now())
    if "history.back" in source:
        if len(here) > 1:
            here.pop()
        return True
    if "about:blank" in source:
        if MODE != "replace_fails":
            here[-1] = "about:blank"  # location.replace: the entry is swapped
        return True
    if "activeElement" in source:
        return MODE == "focus_secret"
    if MODE == "no_body" and "document.body.innerText" in source:
        body_looks += 1
        if body_looks <= 2:
            if "!!document.body &&" not in source:
                raise TypeError(
                    "A JavaScript exception occured: null is not an object "
                    "(evaluating 'document.body.innerText')"
                )
            return False
    if "indexOf" in source:
        needle = json.loads(source.split("indexOf(", 1)[1].split(")", 1)[0])
        found = needle in page_now().get("text", "")
        return found if ">= 0" in source else not found
    raise ValueError(f"unexpected script: {source[:60]}")


def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    global current
    if MODE == "automation_off":
        return text(OFF, error=True)
    value = next(iter(args.values()), None)
    if name == "navigate_to_url":
        tabs[current]["history"].append(str(value))
        return text("navigated")
    if name == "create_tab":
        tabs.append(
            {"id": tab_id(len(tabs) + 1), "history": [str(value or "about:blank")]}
        )
        if MODE != "focus_stays":
            current = len(tabs) - 1
        return text("created")
    if name == "switch_tab":
        current = next(i for i, t in enumerate(tabs) if t["id"] == value)
        return text("switched")
    if name == "close_tab":
        tabs[:] = [t for t in tabs if t["id"] != value]
        current = max(min(current, len(tabs) - 1), 0)
        return text("closed")
    if name == "list_tabs":
        rows = []
        for i, t in enumerate(tabs):
            row: dict[str, Any] = {
                "id": t["id"],
                "url": t["history"][-1],
                "title": PAGES.get(t["history"][-1], {}).get("title", ""),
                "active": i == current,
            }
            if MODE == "no_active":
                del row["active"]
            if MODE == "no_url":
                del row["url"], row["title"]
            if MODE == "no_ids":
                del row["id"]
            rows.append(row)
        return text(json.dumps({"tabs": rows}))
    if name == "screenshot":
        if MODE == "inline_image":
            return {
                "content": [{"type": "image", "data": PNG, "mimeType": "image/png"}]
            }
        decoy = os.environ.get("FAKE_SAFARI_DECOY", "")
        target = str(args.get("savePath") or "/nonexistent/default.png")
        if MODE == "decoy":
            return text(f"Saved screenshot to '{decoy}' (9 B).")
        if MODE == "symlink":
            os.symlink(decoy, target)
            return text(f"Saved screenshot to '{target}' (9 B).")
        with open(target, "wb") as handle:
            handle.write(base64.b64decode(PNG))
        if MODE == "shot_error":
            return text("Could not finish the screenshot.", error=True)
        return text(f"Saved screenshot to '{target}' (72 B).")
    if name == "evaluate_javascript":
        script = args.get("expression", args.get("javascript", value))
        try:
            result = run_script(str(script))
        except SyntaxError as exc:
            return text(f"SyntaxError: {exc}", error=True)
        except TypeError as exc:
            return text(f"evaluateJavaScriptFunction: {exc}.", error=True)
        return text(result if isinstance(result, str) else json.dumps(result))
    if name == "page_interactions":
        return interact(args)
    if name == "get_page_content":
        return content(args)
    return text(f"{name}: ok")


def main() -> None:
    if MODE == "exit_off":
        sys.stderr.write(OFF + "\n")
        sys.exit(1)
    for line in sys.stdin:
        message = json.loads(line)
        method, ident = message.get("method"), message.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "result": {
                        "protocolVersion": TOOLS["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-safari", "version": "1"},
                    },
                }
            )
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": ident, "result": {"tools": tool_list()}})
        elif method == "tools/call":
            params = message["params"]
            send(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "result": call(params["name"], params.get("arguments") or {}),
                }
            )


main()
