"""A fake ``safaridriver --mcp`` for the localmelo tests (stdlib only).

It speaks newline-delimited JSON-RPC like the real server, lists its 17 tools
with the real argument names (as ``safaridriver --mcp`` listed them on Safari
27.2) and understands just enough of the scripts melo sends to answer from a
tiny set of pages. ``screenshot`` answers as the real one does: text only,
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

A script that acts (melo's ``act_script``) is answered from the tiny page
model too: an element may say ``goes`` (the address a click lands on), and a
typed text is kept in ``typed``. Every page read has the document ``DOC`` and
an element may have a fingerprint ``fp``; an act that names another is stale.
"""

from __future__ import annotations

import base64
import json
import os
import sys
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
}


def tab_id(n: int) -> str:
    if MODE == "long_ids":
        return f"page-635DB8F8-27ED-4CA5-81A2-7EAEE1DF{n:04X}"
    return f"page-{n}"


tabs: list[dict[str, Any]] = [{"id": tab_id(1), "history": ["about:blank"]}]
typed: dict[str, str] = {}
current = 0
polls = 0
STYLE_RETURN = MODE == "return_style"


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def schema(**props: str) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}


def tool_list() -> list[dict[str, Any]]:
    alt = MODE == "alt_names"
    page = {
        "format": "string",
        "includeURLs": "boolean",
        "savePath": "string",
        "nodeIds": "array",
        "region": "object",
        "maxWordsPerParagraph": "integer",
    }
    names: dict[str, dict[str, Any]] = {
        "browser_console_messages": schema(
            clear="boolean", limit="integer", tab_handle="string"
        ),
        "browser_dialogs": schema(action="string", inputText="string"),
        "close_tab": schema(**{"tab_id" if alt else "handle": "string"}),
        "create_tab": schema(url="string"),
        "evaluate_javascript": schema(
            **{"javascript" if alt else "expression": "string", "frameId": "string"}
        ),
        "get_network_request": schema(request_id="string"),
        "get_page_content": schema(**page),
        "list_network_requests": schema(
            clear="boolean", filter="string", since="integer"
        ),
        "list_tabs": schema(),
        "navigate_to_url": schema(
            **{"tab_uuid": "string", "address" if alt else "url": "string"}
        ),
        "page_info": schema(),
        "page_interactions": schema(fullText="boolean", interactions="array"),
        "screenshot": schema(full_page="boolean", node="string", savePath="string"),
        "set_emulated_media": schema(media="string"),
        "set_viewport_size": schema(height="integer", width="integer"),
        "switch_tab": schema(**{"tab_id" if alt else "handle": "string"}),
        "wait_for_navigation": schema(timeout_seconds="number"),
    }
    if MODE == "no_script":
        del names["evaluate_javascript"]
    return [
        {"name": n, "description": f"fake {n}", "inputSchema": s}
        for n, s in names.items()
    ]


def text(value: str, error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": value}]}
    if error:
        result["isError"] = True
    return result


DOC = "doc-1"


def page_now() -> dict[str, Any]:
    url = tabs[current]["history"][-1]
    found = PAGES.get(url, {"title": "", "text": "", "elements": []})
    elements = [{**e, "at": i} for i, e in enumerate(found["elements"])]
    return {"url": url, "doc": DOC, **found, "elements": elements}


def do_act(spec: dict[str, Any]) -> dict[str, Any]:
    """What melo's act script answers, on the page model above."""
    here = tabs[current]["history"]
    if MODE == "act_stale":
        return {"ok": False, "stale": "gone"}
    if spec["url"] != here[-1]:
        return {"ok": False, "stale": "page"}
    if MODE == "act_reloaded" or spec.get("doc") != DOC:
        return {"ok": False, "stale": "doc"}
    elements = PAGES.get(here[-1], {"elements": []})["elements"]
    if not 0 <= spec["at"] < len(elements):
        return {"ok": False, "stale": "gone"}
    element = elements[spec["at"]]
    if (element["kind"], element.get("label", ""), element.get("href", "")) != (
        spec["kind"],
        spec["label"],
        spec["href"],
    ):
        return {"ok": False, "stale": "gone"}
    if MODE == "act_moved" or spec.get("fp", "") != element.get("fp", ""):
        return {"ok": False, "stale": "gone"}
    if spec["act"] == "click":
        if element.get("goes"):
            here.append(element["goes"])
    elif spec["act"] == "type":
        typed[element["label"]] = spec["text"] + ("\n" if spec["enter"] else "")
        if spec["enter"] and element.get("submit"):
            here.append(element["submit"])
    elif spec["act"] == "select":
        if MODE == "act_option" or spec["text"] not in element["options"]:
            return {"ok": False, "option": 0}
        element["value"] = spec["text"]
    return {"ok": True}


def run_script(source: str) -> Any:
    global polls
    if STYLE_RETURN and not source.startswith("return "):
        raise SyntaxError("Unexpected token")
    source = source.removeprefix("return ")
    if source.startswith("(function (spec)"):
        spec = json.loads(source[source.rindex("})(") + 3 : -1])
        return json.dumps(do_act(spec))
    here = tabs[current]["history"]
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
        return json.dumps(page_now())
    if "history.back" in source:
        if len(here) > 1:
            here.pop()
        return True
    if "about:blank" in source:
        here.append("about:blank")
        return True
    if "activeElement" in source:
        return MODE == "focus_secret"
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
        try:
            result = run_script(str(value))
        except SyntaxError as exc:
            return text(f"SyntaxError: {exc}", error=True)
        return text(result if isinstance(result, str) else json.dumps(result))
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
                        "protocolVersion": "2025-06-18",
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
