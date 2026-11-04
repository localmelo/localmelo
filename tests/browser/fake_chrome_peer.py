"""A fake headless Chrome for the localmelo tests: DevTools over the pipe, no browser (stdlib only).

It is started the way melo starts Chrome (``LOCALMELO_CHROME`` names it): it
reads NUL-delimited DevTools messages on fd 3 and answers on fd 4, as Chrome
does with ``--remote-debugging-pipe``. It understands just enough of what melo
sends to answer from a tiny set of pages, like the fake Safari server: the
page script, the act scripts and the look after an act are told apart by how
they start. Every call (and the command line, first) is logged as one JSON
line to ``FAKE_CHROME_LOG`` when it is set.

An element sits at ``(100, 50 + 40 * at)`` of the view; a press and a release
there are a real click on it. An element may say ``goes`` (where a click lands),
``scheme`` (an address of another app's kind the page asks for on a click:
the request is told as Chrome tells it, and nothing opens), ``dialog`` (a box
the click puts up: the page waits until it is answered; a confirm answered OK
then goes to ``goes``), ``popup`` (a tab the click opens), ``notice`` (text the
page shows after a real click), ``trusted_only`` (a script's click does
nothing) or ``covered``. A covered element may say ``by`` (what covers it, as the
page script tells it), ``alt`` (a point of it that is clear: the script then aims
there), ``clears_on_scroll`` (the scroll to the middle that finds it clear, the
first being 1) and ``moves`` (how many real clicks on it land on another element
first, which the guards stop; ``moved_by`` says which).

A screencast (``Page.startScreencast``) sends ``FAKE_CHROME_FRAMES`` pictures
(default 60; ``1`` is a page that does not change), at most three unacknowledged
at a time, as Chrome does: the next one comes only after an acknowledgement
(``Page.screencastFrameAck``). Every call is logged with its time (``t``).

Modes (``FAKE_CHROME_MODE``): ``silent`` (never answers), ``exit`` (quits at
once), ``native_missed`` (a real click reaches nothing), ``no_world`` (no
isolated world can be made), ``nav_error`` (every navigation fails), ``cast_error`` (a screencast cannot start).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any, cast

MODE = os.environ.get("FAKE_CHROME_MODE", "")
LOG = os.environ.get("FAKE_CHROME_LOG", "")
JPEG = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9").decode()
DOC = "doc-1"

PAGES: dict[str, dict[str, Any]] = {
    "https://example.com/": {
        "title": "Example Home",
        "text": "Welcome to the example.",
        "elements": [
            {"kind": "link", "label": "News", "href": "https://example.com/news"},
            {"kind": "textbox", "label": "Search", "value": "cats"},
            {"kind": "textbox", "label": "", "value": "", "sensitive": "password"},
            {"kind": "button", "label": "Go"},
        ],
    },
    "https://example.com/news": {"title": "News", "text": "Today's news."},
    "https://example.com/done": {"title": "Done", "text": "Checked in."},
    "https://example.com/deleted": {"title": "Deleted", "text": "It is gone."},
    "https://example.com/form": {
        "title": "Form",
        "text": "A form.",
        "elements": [
            {
                "kind": "button",
                "label": "Check in",
                "fp": "button|check in",
                "goes": "https://example.com/done",
            },
            {
                "kind": "textbox",
                "label": "Nickname",
                "value": "",
                "submit": "https://example.com/save",
                "fp": "input|nick",
            },
            {
                "kind": "select",
                "label": "Country",
                "value": "France",
                "options": ["France", "Spain"],
                "fp": "select|country",
            },
            {
                "kind": "button",
                "label": "Hidden away",
                "fp": "button|hidden",
                "covered": True,
            },
        ],
    },
    "https://example.com/cover": {
        "title": "Cover",
        "text": "A page with a bar over its buttons.",
        "elements": [
            {
                "kind": "button",
                "label": "Half covered",
                "fp": "button|half",
                "covered": True,
                "alt": [130, 400],
                "goes": "https://example.com/done",
            },
            {
                "kind": "button",
                "label": "Clears later",
                "fp": "button|later",
                "covered": True,
                "clears_on_scroll": 2,
                "goes": "https://example.com/done",
            },
            {
                "kind": "button",
                "label": "Under the header",
                "fp": "button|under",
                "covered": True,
                "by": {"what": "a sticky header", "detail": "h1"},
            },
            {
                "kind": "button",
                "label": "Slips away",
                "fp": "button|slips",
                "moves": 1,
                "moved_by": {"what": "a sticky header", "detail": "h1"},
                "goes": "https://example.com/done",
            },
            {
                "kind": "button",
                "label": "Always slips",
                "fp": "button|always",
                "moves": 9,
                "moved_by": {"what": "an ad", "detail": "iframe from ads.example.net"},
                "goes": "https://example.com/done",
            },
        ],
    },
    "https://example.com/app": {
        "title": "App",
        "text": "Sign in with the chat app.",
        "elements": [
            {
                "kind": "button",
                "label": "Open in Chat",
                "fp": "button|chat",
                "scheme": "chat://login?token=secret-token",
            },
            {
                "kind": "button",
                "label": "Delete",
                "fp": "button|delete",
                "dialog": {"type": "confirm", "message": "Delete the draft?"},
                "goes": "https://example.com/deleted",
            },
            {
                "kind": "button",
                "label": "Hello",
                "fp": "button|hello",
                "dialog": {"type": "alert", "message": "Hi there"},
            },
            {
                "kind": "button",
                "label": "More",
                "fp": "button|more",
                "popup": "https://example.com/news",
            },
            {
                "kind": "button",
                "label": "Daily",
                "fp": "button|daily",
                "trusted_only": True,
                "notice": "Checked in: day 12",
            },
        ],
    },
}

reader = os.fdopen(3, "rb", 0)
writer = os.fdopen(4, "wb", 0)
targets: dict[str, dict[str, Any]] = {}
sessions: dict[str, str] = {}
looks: dict[str, dict[str, Any]] = {}
armed: dict[str, str] = {}  # target -> the key of the act armed there
focused: dict[str, dict[str, Any]] = {}  # target -> the text box a click focused
typed: dict[str, str] = {}
waiting: dict[str, list[dict[str, Any]]] = {}  # target -> calls held while a box is up
dialog_after: dict[str, tuple[int, dict[str, Any]]] = {}  # target -> (call id, element)
n_targets = 0
scrolls: dict[int, int] = (
    {}
)  # element place -> how many times it was scrolled to the middle
FRAMES = int(os.environ.get("FAKE_CHROME_FRAMES", "60"))
casts: dict[str, dict[str, int]] = (
    {}
)  # session -> how many pictures are left, in flight, sent


def log(entry: dict[str, Any]) -> None:
    if LOG:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def send(message: dict[str, Any]) -> None:
    writer.write(json.dumps(message).encode() + b"\0")


def event(method: str, params: dict[str, Any], session: str = "") -> None:
    message: dict[str, Any] = {"method": method, "params": params}
    if session:
        message["sessionId"] = session
    send(message)


def session_of(target: str) -> str:
    return next((s for s, t in sessions.items() if t == target), "")


def new_target(url: str, opener: str = "") -> str:
    global n_targets
    n_targets += 1
    target = f"T{n_targets}"
    targets[target] = {"history": [url], "index": 0, "opener": opener}
    return target


def url_of(target: str) -> str:
    t = targets[target]
    return str(t["history"][t["index"]])


def elements_of(target: str) -> list[dict[str, Any]]:
    return cast(
        "list[dict[str, Any]]", PAGES.get(url_of(target), {}).get("elements", [])
    )


def navigate(target: str, url: str) -> None:
    t = targets[target]
    t["history"] = t["history"][: t["index"] + 1] + [url]
    t["index"] += 1
    event(
        "Page.frameNavigated",
        {"frame": {"id": target, "url": url, "loaderId": "L"}},
        session_of(target),
    )


def info(target: str) -> dict[str, Any]:
    out = {
        "targetId": target,
        "type": "page",
        "url": url_of(target),
        "title": PAGES.get(url_of(target), {}).get("title", ""),
        "attached": bool(session_of(target)),
    }
    if targets[target]["opener"]:
        out["openerId"] = targets[target]["opener"]
    return out


def page_now(target: str) -> dict[str, Any]:
    url = url_of(target)
    found = PAGES.get(url, {"title": "", "text": ""})
    keep = ("kind", "label", "href", "value", "sensitive", "submit", "options", "fp")
    elements = [
        {**{k: e[k] for k in keep if k in e}, "at": i}
        for i, e in enumerate(found.get("elements", []))
    ]
    return {
        "url": url,
        "doc": DOC,
        "title": found.get("title", ""),
        "text": found.get("text", ""),
        "elements": elements,
    }


def spec_of(source: str) -> dict[str, Any]:
    start = source.rindex('})({"')
    spec, _ = json.JSONDecoder().raw_decode(source, start + 3)
    return cast("dict[str, Any]", spec)


def clicked(target: str, element: dict[str, Any], key: str, *, real: bool) -> int:
    """What a click does; a box it puts up returns 1 (the click's answer waits)."""
    if element.get("trusted_only") and not real:
        return 0
    look = looks.setdefault(key, {"doc": DOC, "url": url_of(target), "before": "{}"})
    look.setdefault("seen", {})["click"] = 1
    if element.get("notice"):
        look["texts"] = [element["notice"]]
        look["count"] = 1
    if element.get("scheme"):
        event(
            "Page.frameRequestedNavigation",
            {
                "frameId": target,
                "reason": "anchorClick",
                "url": element["scheme"],
                "disposition": "currentTab",
            },
            session_of(target),
        )
    if element.get("popup"):
        new_target(element["popup"], opener=target)
    if element.get("dialog"):
        box = element["dialog"]
        event(
            "Page.javascriptDialogOpening",
            {
                "url": url_of(target),
                "message": box["message"],
                "type": box["type"],
                "hasBrowserHandler": False,
                "defaultPrompt": "",
            },
            session_of(target),
        )
        return 1
    if element.get("goes"):
        navigate(target, element["goes"])
    return 0


def do_act(target: str, spec: dict[str, Any]) -> dict[str, Any]:
    url = url_of(target)
    if spec["url"] != url:
        return {"ok": False, "stale": "page"}
    if spec.get("doc") != DOC:
        return {"ok": False, "stale": "doc"}
    elements = elements_of(target)
    at = spec["at"]
    if not 0 <= at < len(elements):
        return {"ok": False, "stale": "gone"}
    element = elements[at]
    if (element["kind"], element.get("label", ""), element.get("fp", "")) != (
        spec["kind"],
        spec["label"],
        spec.get("fp", ""),
    ):
        return {"ok": False, "stale": "gone"}
    mode, key = spec.get("mode", "script"), spec.get("key", "")
    if mode in ("scroll", "arm"):
        log({"act_mode": mode, "at": at})
    if mode == "scroll":
        scrolls[at] = scrolls.get(at, 0) + 1
        covered = bool(element.get("covered"))
        if scrolls[at] >= element.get("clears_on_scroll", 99):
            covered = False
        said: dict[str, Any] = {
            "ok": True,
            "covered": covered,
            "cx": 100,
            "cy": 50 + 40 * at,
            "tag": "button",
            "text": element.get("label", ""),
        }
        if covered and element.get("alt"):
            said.update(covered=False, shifted=True)
            said["cx"], said["cy"] = element["alt"]
        if covered and element.get("by"):
            said["by"] = element["by"]
        return said
    option = ""
    if spec["act"] == "select":
        if spec["text"] not in element.get("options", []):
            return {"ok": False, "option": 0}
        option = spec["text"]
    if key:
        looks[key] = {
            "doc": DOC,
            "url": url,
            "before": json.dumps({"label": element.get("label", "")}),
        }
    if mode == "arm":
        armed[target] = key
        return {"ok": True, "option": option}
    if spec["act"] == "click":
        clicked(target, element, key, real=False)
    elif spec["act"] == "type":
        typed[element["label"]] = spec["text"] + ("\n" if spec["enter"] else "")
        looks.setdefault(key, {}).setdefault("seen", {})["input"] = 1
    elif spec["act"] == "select":
        element["value"] = spec["text"]
        looks.setdefault(key, {}).setdefault("seen", {})["change"] = 1
    return {"ok": True}


def run_script(target: str, source: str) -> Any:
    here = url_of(target)
    if source.startswith("(function (spec)"):
        return json.dumps(do_act(target, spec_of(source)))
    if source.startswith(("(function (settle)", "(function (disarm)")):
        asked = json.loads(source[source.rindex("})(") + 3 : -1])
        look = looks.get(asked["key"])
        if asked["doc"] != DOC or look is None or look.get("url", here) != here:
            return {"doc": False} if "settle" in source[:20] else {"lost": True}
        if "settle" in source[:20]:
            return {
                "doc": True,
                "count": look.get("count", 0),
                "quiet": 5000,
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
        return json.dumps({"surface": "none"})
    if source.startswith("(function (view)"):
        return json.dumps(
            {"url": here, "doc": DOC, "w": 1280, "h": 800, "x": 0, "y": 0, "dpr": 1}
        )
    if source.startswith("(function (mark)"):
        return True
    if "getBoundingClientRect" in source:
        return json.dumps(page_now(target))
    if "activeElement" in source:
        return False
    if source == "document.readyState":
        return "complete"
    if "location.replace('about:blank')" in source:
        t = targets[target]
        t["history"][t["index"]] = "about:blank"
        return True
    if "document.readyState === 'complete'" in source:
        return True
    if "indexOf(" in source:
        needle = json.loads(source.split("indexOf(", 1)[1].split(")", 1)[0])
        found = needle in page_now(target).get("text", "")
        return found if ">= 0" in source else not found
    raise ValueError(f"unexpected script: {source[:60]}")


def at_point(target: str, x: float, y: float) -> dict[str, Any] | None:
    for i, element in enumerate(elements_of(target)):
        if (round(x), round(y)) in (
            (100, 50 + 40 * i),
            tuple(element.get("alt") or ()),
        ):
            return element
    return None


def feed(session: str) -> None:
    """Send pictures of a screencast until three wait for their acknowledgement."""
    cast = casts.get(session)
    while cast is not None and cast["left"] > 0 and cast["flight"] < 3:
        cast["left"] -= 1
        cast["flight"] += 1
        cast["sent"] += 1
        data = base64.b64encode(
            b"\xff\xd8\xff\xe0frame-%d\xff\xd9" % cast["sent"]
        ).decode()
        event(
            "Page.screencastFrame",
            {
                "data": data,
                "metadata": {
                    "offsetTop": 0,
                    "pageScaleFactor": 1,
                    "deviceWidth": 1280,
                    "deviceHeight": 713,
                    "scrollOffsetX": 0,
                    "scrollOffsetY": 0,
                    "timestamp": time.time(),
                },
                "sessionId": cast["sent"],
            },
            session,
        )


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """The answer to one call; None when the answer waits (a box is up)."""
    method = message["method"]
    params = message.get("params") or {}
    session = message.get("sessionId", "")
    target = sessions.get(session, "")
    if target and target in dialog_after and method != "Page.handleJavaScriptDialog":
        waiting.setdefault(target, []).append(message)
        return None
    if method == "Browser.getVersion":
        return {"product": "HeadlessChrome/156.0.0.0", "protocolVersion": "1.3"}
    if method == "Browser.close":
        send({"id": message["id"], "result": {}})
        sys.exit(0)
    if method == "Target.getTargets":
        return {"targetInfos": [info(t) for t in targets]}
    if method == "Target.createTarget":
        return {"targetId": new_target(params.get("url", "about:blank"))}
    if method == "Target.attachToTarget":
        sid = f"S-{params['targetId']}"
        sessions[sid] = params["targetId"]
        return {"sessionId": sid}
    if method == "Target.closeTarget":
        if params["targetId"] not in targets:
            return {"__error": "No target with given id found"}
        targets.pop(params["targetId"])
        return {"success": True}
    if method == "Target.getTargetInfo":
        if params["targetId"] not in targets:
            return {"__error": "No target with given id found"}
        return {"targetInfo": info(params["targetId"])}
    if method == "Page.getFrameTree":
        return {"frameTree": {"frame": {"id": target, "url": url_of(target)}}}
    if method == "Page.createIsolatedWorld":
        if MODE == "no_world":
            return {"__error": "Cannot create isolated world"}
        return {"executionContextId": 42}
    if method == "Page.navigate":
        if MODE == "nav_error":
            return {"frameId": target, "errorText": "net::ERR_NAME_NOT_RESOLVED"}
        navigate(target, params["url"])
        return {"frameId": target, "loaderId": "L"}
    if method == "Page.getNavigationHistory":
        t = targets[target]
        entries = [{"id": i, "url": u} for i, u in enumerate(t["history"])]
        return {"currentIndex": t["index"], "entries": entries}
    if method == "Page.navigateToHistoryEntry":
        targets[target]["index"] = int(params["entryId"])
        event(
            "Page.frameNavigated",
            {"frame": {"id": target, "url": url_of(target)}},
            session,
        )
        return {}
    if method == "Page.captureScreenshot":
        return {"data": JPEG}
    if method == "Page.startScreencast":
        if MODE == "cast_error":
            return {"__error": "Screencast is not available"}
        casts[session] = {"left": FRAMES, "flight": 0, "sent": 0}
        feed(session)
        return {}
    if method == "Page.stopScreencast":
        casts.pop(session, None)
        return {}
    if method == "Page.screencastFrameAck":
        cast = casts.get(session)
        if cast is not None:
            cast["flight"] = max(cast["flight"] - 1, 0)
            feed(session)
        return {}
    if method == "Page.handleJavaScriptDialog":
        held = dialog_after.pop(target, None)
        event("Page.javascriptDialogClosed", {"result": params["accept"]}, session)
        if held is not None:
            ident, element = held
            if params["accept"] and element.get("goes"):
                navigate(target, element["goes"])
            send({"id": ident, "result": {}})
        for later in waiting.pop(target, []):
            answer(later)
        return {}
    if method == "Runtime.evaluate":
        value = run_script(target, params["expression"])
        return {"result": {"type": type(value).__name__, "value": value}}
    if method == "Input.dispatchMouseEvent":
        if MODE == "native_missed":
            return {}
        hit = at_point(target, params["x"], params["y"])
        if params["type"] == "mouseReleased" and hit is not None and hit.get("moves"):
            # The page slid another element under the click: the guards stop it.
            hit["moves"] -= 1
            look = looks.setdefault(armed.get(target, ""), {})
            look["blocked"] = ["moved"]
            look["moved_by"] = hit.get("moved_by")
        elif params["type"] == "mouseReleased" and hit is not None:
            if hit["kind"] == "textbox":
                focused[target] = hit
                looks.setdefault(armed.get(target, ""), {}).setdefault("seen", {})[
                    "click"
                ] = 1
            elif clicked(target, hit, armed.get(target, ""), real=True):
                dialog_after[target] = (message["id"], hit)
                return None
        return {}
    if method == "Input.dispatchKeyEvent":
        box = focused.get(target)
        if box is not None and params.get("type") in ("rawKeyDown", "keyDown"):
            look = looks.setdefault(armed.get(target, ""), {})
            look.setdefault("seen", {})["keydown"] = 1
            if params.get("key") == "Enter":
                typed[box["label"]] = typed.get(box["label"], "") + "\n"
                if box.get("submit"):
                    navigate(target, box["submit"])
        return {}
    if method == "Input.insertText":
        box = focused.get(target)
        if box is not None:
            typed[box["label"]] = params["text"]
            looks.setdefault(armed.get(target, ""), {}).setdefault("seen", {})[
                "input"
            ] = 1
        return {}
    return {}


def answer(message: dict[str, Any]) -> None:
    log(
        {
            "method": message["method"],
            "t": time.time(),
            "session": message.get("sessionId", ""),
            "params": {
                k: v
                for k, v in (message.get("params") or {}).items()
                if k != "expression"
            },
            "script": str((message.get("params") or {}).get("expression", ""))[:40],
            "typed": dict(typed),
        }
    )
    try:
        result = handle(message)
    except (KeyError, ValueError) as exc:
        send({"id": message["id"], "error": {"code": -32000, "message": str(exc)}})
        return
    if result is None:
        return
    if "__error" in result:
        send(
            {
                "id": message["id"],
                "error": {"code": -32000, "message": result["__error"]},
            }
        )
        return
    send({"id": message["id"], "result": result})


def main() -> None:
    log({"argv": sys.argv[1:], "fds": [3, 4]})
    if MODE == "exit":
        sys.exit(1)
    new_target("about:blank")  # the page Chrome starts with
    buffer = b""
    while True:
        chunk = reader.read(65536)
        if not chunk:
            return
        buffer += chunk
        while b"\0" in buffer:
            raw, buffer = buffer.split(b"\0", 1)
            if MODE == "silent":
                continue
            answer(json.loads(raw))


main()
