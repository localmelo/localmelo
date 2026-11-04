"""A fake desktop control helper for the localmelo tests (stdlib only, no real screen).

It speaks newline-delimited JSON-RPC like the real helper's ``mcp`` command and
offers the four tools melo uses, with the argument names the real one listed
(``list_windows``, ``get_window_state``, ``click``, ``check_permissions``). The
screen is a JSON file, the first argument, read on every call so a test can
change it between calls::

    {"grants": {"accessibility": true, "screen_recording": true},
     "space": 1,
     "windows": [{"window_id": 7, "pid": 70, "app_name": "Chat", "title": "Sign in",
                  "z_index": 9, "layer": 0, "elements": [...], "press": {...}}],
     "press_fails": false}

A window's ``elements`` are the rows ``get_window_state`` answers (role, label,
element_token); ``press`` maps a token to what pressing it does: ``{"close":
true}`` closes the window, ``{"effect": "..."}`` is the effect the answer
names.

Like the real helper, ``list_windows`` lists every Space and off-screen windows
too: layer 0 only without a ``pid``, every layer of that ``pid`` with one, and
only what is on the screen with ``on_screen_only``. A window is on the screen
when it is on the current ``space`` (its ``spaces``, a list of Space ids; none
given: every Space), not ``hidden`` (its app hidden or it is minimized) and not
``on_screen: false`` (made but never shown).

A click at a point (``x``, ``y``) needs a read of that window
(``get_window_state``) earlier in the same session, as the real helper does,
and uses it up. What it does is the window's ``clicks``, by ``delivery_mode``
(``background``, the default, or ``foreground``): ``"close"`` closes the
window, ``"change"`` changes its picture and text, ``"nothing"`` (the default)
does nothing. ``"close_then_fail"`` closes the window too, but before the picture that
marks the point is taken, so that picture fails (the helper takes it after the click).
``"gone"`` is a window that went away just before the click, which the helper refuses. Every call is appended to the file named by the second argument
(one JSON line each).
"""

from __future__ import annotations

import base64
import json
import sys
from typing import Any

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
    "AAAABJRU5ErkJggg=="
)
# The picture of a window that changed after a click: another 1x1 PNG.
CHANGED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgaGAAAAEEAIFw9selAAAAAElFTkSuQmCC"
)
STATE, LOG = sys.argv[1], sys.argv[2]
# The windows read in this session (this process is one helper session): a click at a
# point needs one, and uses it up.
READ: set[int] = set()
TOOLS = [
    {"name": "list_windows", "inputSchema": {"type": "object", "properties": {
        "app": {"type": "string"}, "pid": {"type": "integer"},
        "on_screen_only": {"type": "boolean"}, "session": {"type": "string"}}}},
    {"name": "get_window_state", "inputSchema": {"type": "object", "properties": {
        "pid": {"type": "integer"}, "window_id": {"type": "integer"},
        "include_screenshot": {"type": "boolean"}, "tree_format": {"type": "string"},
        "max_image_dimension": {"type": "integer"}}, "required": ["pid", "window_id"]}},
    {"name": "click", "inputSchema": {"type": "object", "properties": {
        "element_token": {"type": "string"}, "pid": {"type": "integer"},
        "window_id": {"type": "integer"}, "x": {"type": "number"}, "y": {"type": "number"},
        "debug_image_out": {"type": "string"}, "delivery_mode": {"type": "string"}}}},
    {"name": "check_permissions", "inputSchema": {"type": "object", "properties": {
        "prompt": {"type": "boolean"}}}},
]  # fmt: skip


def load() -> dict[str, Any]:
    with open(STATE, encoding="utf-8") as handle:
        state: dict[str, Any] = json.load(handle)
    return state


def save(state: dict[str, Any]) -> None:
    with open(STATE, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def answer(
    data: dict[str, Any], *, error: bool = False, image: bytes | None = None
) -> dict:
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(data)}]
    if image is not None:
        content.append(
            {
                "type": "image",
                "data": base64.b64encode(image).decode(),
                "mimeType": "image/png",
            }
        )
    return {"content": content, "structuredContent": data, "isError": error}


def on_screen(window: dict[str, Any], space: int) -> bool:
    spaces = window.get("spaces")
    return (
        window.get("on_screen", True) is not False
        and not window.get("hidden")
        and (spaces is None or space in spaces)
    )


def row(window: dict[str, Any], space: int) -> dict[str, Any]:
    keys = ("window_id", "pid", "app_name", "title", "z_index")
    spaces = window.get("spaces")
    return {**{k: window.get(k) for k in keys}, "layer": window.get("layer", 0),
            "is_on_screen": on_screen(window, space),
            "space_ids": [1, 2, 3] if spaces is None else spaces,
            "bounds": {"x": 0, "y": 0, "width": 400, "height": 300}}  # fmt: skip


def clicked(state: dict[str, Any], window: dict[str, Any], how: str) -> None:
    """What a click at a point does to *window*, as its ``clicks`` say for *how*."""
    done = (window.get("clicks") or {}).get(how, "nothing")
    if done in ("close", "close_then_fail"):
        state["windows"] = [w for w in state["windows"] if w is not window]
    elif done == "change":
        window["picture"] = "changed"
        window["elements"] = [{"role": "AXStaticText", "label": "Done"}]
    if done != "nothing":
        save(state)


def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    state = load()
    windows = state.get("windows", [])
    if name == "check_permissions":
        return answer(
            state.get("grants", {"accessibility": True, "screen_recording": True})
        )
    if name == "list_windows":
        space = int(state.get("space", 1))
        found = []
        for window in windows:
            if "pid" in args:
                if window["pid"] != args["pid"]:
                    continue
            elif window.get("layer", 0) != 0:
                continue
            if args.get("on_screen_only") and not on_screen(window, space):
                continue
            app = str(args.get("app") or "")
            if (
                app
                and app.lower() not in window["app_name"].lower()
                and app != window.get("bundle")
            ):
                continue
            found.append(row(window, space))
        return answer({"current_space_id": space, "windows": found})
    if name == "get_window_state":
        window = next(
            (w for w in windows if w["window_id"] == args.get("window_id")), None
        )
        if window is None:
            return answer({"refusal": "window_id_not_found"}, error=True)
        READ.add(window["window_id"])
        picture = args.get("include_screenshot", True) is not False
        data = {
            "pid": window["pid"],
            "window_id": window["window_id"],
            "elements": window.get("elements", []),
            "window_title": window["title"],
        }
        if picture:
            data.update(screenshot_width=800, screenshot_height=600)
        shown = CHANGED_PNG if window.get("picture") == "changed" else PNG
        return answer(data, image=shown if picture else None)
    if name == "click":
        token = args.get("element_token")
        if token:
            if state.get("press_fails"):
                return answer({"refusal": "ax_action_failed"}, error=True)
            for window in windows:
                done = window.get("press", {}).get(token)
                if done is None:
                    continue
                if done.get("close"):
                    state["windows"] = [w for w in windows if w is not window]
                    save(state)
                return answer({"effect": done.get("effect", "confirmed")})
            return answer({"refusal": "stale element_token"}, error=True)
        window = next(
            (w for w in windows if w["window_id"] == args.get("window_id")), None
        )
        if window is None:
            return answer({"refusal": "window_id_not_found"}, error=True)
        if window["window_id"] not in READ:
            return answer({"refusal": "window_never_read"}, error=True)
        READ.discard(window["window_id"])
        how = str(args.get("delivery_mode") or "background")
        if (window.get("clicks") or {}).get(how) == "gone":
            state["windows"] = [w for w in windows if w is not window]
            save(state)
            return answer({"refusal": "window_id_not_found"}, error=True)
        closes = (window.get("clicks") or {}).get(how) == "close_then_fail"
        if args.get("debug_image_out") and not closes:
            with open(args["debug_image_out"], "wb") as handle:
                handle.write(PNG)
        clicked(state, window, how)
        if args.get("debug_image_out") and closes:
            wid = window["window_id"]
            return answer(
                {"refusal": f"debug_image_out write failed: window {wid} capture "
                 f"failed: native: ScreenCaptureKit capture failed for window {wid}: "
                 f"WindowServer no longer lists window id {wid}; using shell fallback"},
                error=True,
            )  # fmt: skip
        return answer({"effect": "unverifiable"})
    return answer({"refusal": f"unknown tool {name}"}, error=True)


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        message = json.loads(line)
        method, ident = message.get("method"), message.get("id")
        if ident is None:
            continue
        params = message.get("params") or {}
        if method == "initialize":
            result: Any = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-desktop", "version": "0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            with open(LOG, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps([params.get("name"), params.get("arguments")]) + "\n"
                )
            result = call(str(params.get("name")), params.get("arguments") or {})
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "error": {"code": -32601, "message": method},
                }
            )
            continue
        send({"jsonrpc": "2.0", "id": ident, "result": result})


if __name__ == "__main__":
    main()
