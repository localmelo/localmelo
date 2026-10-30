"""A fake ``safaridriver --mcp`` for the localmelo tests (stdlib only).

It speaks newline-delimited JSON-RPC like the real server, lists the tools the
design names (plus filler) and understands just enough of the scripts melo
sends to answer from a tiny set of pages. Modes (the first argument, or
``FAKE_SAFARI_MODE`` in the environment):

- ``automation_off``: every tool call fails as Safari does when remote
  automation is off
- ``exit_off``: write that message to stderr and exit before ``initialize``
- ``no_script``: no ``evaluate_javascript`` tool
- ``return_style``: a script runs only with a leading ``return``
- ``alt_names``: other argument names (``address``, ``javascript``, ``handle``)
- ``slow_load``: ``document.readyState`` says ``loading`` twice first
- ``focus_secret``: a password field has focus
- ``file_image``: ``screenshot`` answers with text only
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any

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
}
tabs: list[dict[str, Any]] = [{"id": "w1", "history": ["about:blank"]}]
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
    names: dict[str, dict[str, Any]] = {
        "navigate_to_url": schema(**{"address" if alt else "url": "string"}),
        "get_page_content": schema(),
        "page_interactions": schema(actions="array"),
        "screenshot": schema(),
        "list_tabs": schema(),
        "switch_tab": schema(**{"handle" if alt else "tab_id": "string"}),
        "create_tab": schema(),
        "close_tab": schema(**{"handle" if alt else "tab_id": "string"}),
        "wait_for_navigation": schema(),
        "evaluate_javascript": schema(**{"javascript" if alt else "script": "string"}),
    }
    names.update({f"filler_{n}": schema() for n in range(7)})
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


def page_now() -> dict[str, Any]:
    url = tabs[current]["history"][-1]
    return {"url": url, **PAGES.get(url, {"title": "", "text": "", "elements": []})}


def run_script(source: str) -> Any:
    global polls
    if STYLE_RETURN and not source.startswith("return "):
        raise SyntaxError("Unexpected token")
    source = source.removeprefix("return ")
    here = tabs[current]["history"]
    if source == "1+1":
        return 2
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
        tabs.append({"id": f"w{len(tabs) + 1}", "history": ["about:blank"]})
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
        rows = [
            {"id": t["id"], "url": t["history"][-1], "active": i == current}
            for i, t in enumerate(tabs)
        ]
        return text(json.dumps({"tabs": rows}))
    if name == "screenshot":
        if MODE == "file_image":
            return text("saved to /tmp/nothing.png")
        return {"content": [{"type": "image", "data": PNG, "mimeType": "image/png"}]}
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
