"""Taking over melo's browser against the real Google Chrome, headless (opt in).

Skipped unless ``LOCALMELO_CHROME_SMOKE=1`` and Google Chrome is installed. Two
headless Chromes run, neither with a window, each on a profile under ``tmp_path``:
melo's, which a chat's browser tools drive on a form served by the test on
``127.0.0.1`` (the system picks the port), and the person's, which opens the
console's Browser panel alone (``#/live/<chat>``) under the real loopback guard.
The person's Chrome is driven with its own trusted input at the panel, as a
person's mouse and keyboard would: Take over, a click into the form, typing, an
input method composing Chinese, a paste, and Give back to melo. melo then reads
the page and finds what was typed in the name field, and nothing of the password.

A pinch zoom on melo's page checks that a point on the picture lands on the
same point of the page when the page is zoomed. Every test watches the app in
front (``lsappinfo front``, compared by process id): one of the test's Chromes
taking the front fails it. Run it by hand:
``LOCALMELO_CHROME_SMOKE=1 uv run pytest tests/browser/test_real_takeover.py -s``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.gateway.console import Console
from localmelo.tests.browser import test_real_live as smoke
from localmelo.tests.browser.test_provider import run
from localmelo.tests.browser.test_real_live import (
    RealAgent,
    RealBuilder,
    _loaded,
    chat,
    kept,
    settled,
    sign_in,
)
from localmelo.tests.fixtures.http_stub import HttpStub
from localmelo.tests.gateway.console.test_browser_live import Served

# The live view smoke test's own: skipped unless asked for, its fixtures (the app in
# front watched throughout, melo's pages allowed on 127.0.0.1, the console under the
# guard, the person's headless Chrome).
pytestmark = smoke.pytestmark
never_the_front = smoke.never_the_front
local_pages = smoke.local_pages
builder = smoke.builder
console = smoke.console
served = smoke.served
workspace = smoke.workspace
person = smoke.person

FORM = """<!doctype html><meta charset="utf-8"><title>Test form</title>
<body style="margin:0;font:16px -apple-system,sans-serif;background:#fff">
<form onsubmit="return false" style="padding:24px">
<p><label>Name <input id="name" name="name" autocomplete="off" style="width:320px;height:34px;font-size:18px"></label></p>
<p><label>Password <input id="pw" type="password" name="password" style="width:320px;height:34px"></label></p>
<button id="hit" type="button" style="width:220px;height:70px;font-size:18px"
  onclick="this.dataset.n=String((+this.dataset.n||0)+1);this.textContent='Pressed '+this.dataset.n">Press</button>
</form>
<div style="height:1600px"></div></body>"""
NAME = "Alex 你好"
SECRET = "hunter2-Secret"


class Pages(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        data = FORM.encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def base() -> Iterator[str]:
    with HttpStub(Pages) as stub:
        yield stub.base_url


def as_dict(raw: Any) -> dict[str, Any]:
    """What a page script gave back as JSON (the driver reads a JSON text into a dict itself)."""
    return dict(raw if isinstance(raw, dict) else json.loads(raw))


# ── The person's own input, at the panel ──


async def session_of(driver: ChromeBackgroundDriver) -> str:
    return await driver._session()


async def mouse(
    person: ChromeBackgroundDriver, kind: str, x: float, y: float, buttons: int
) -> None:
    params: dict[str, Any] = {"type": kind, "x": x, "y": y}
    if kind != "mouseMoved":
        params.update(button="left", buttons=buttons, clickCount=1)
    else:
        params.update(buttons=buttons)
    await person._cmd("Input.dispatchMouseEvent", params, await session_of(person))


async def click(person: ChromeBackgroundDriver, x: float, y: float) -> None:
    await mouse(person, "mouseMoved", x, y, 0)
    await mouse(person, "mousePressed", x, y, 1)
    await mouse(person, "mouseReleased", x, y, 0)


async def press(
    person: ChromeBackgroundDriver, key: str, code: str, key_code: int, text: str = ""
) -> None:
    session = await session_of(person)
    common = {"key": key, "code": code, "windowsVirtualKeyCode": key_code}
    if text:
        await person._cmd(
            "Input.dispatchKeyEvent",
            {"type": "keyDown", "text": text, **common},
            session,
        )
    else:
        await person._cmd(
            "Input.dispatchKeyEvent", {"type": "rawKeyDown", **common}, session
        )
    await person._cmd("Input.dispatchKeyEvent", {"type": "keyUp", **common}, session)


async def type_text(person: ChromeBackgroundDriver, text: str) -> None:
    for char in text:
        code = (
            f"Key{char.upper()}"
            if char.isalpha()
            else ("Digit" + char if char.isdigit() else "")
        )
        await press(
            person, char, code, ord(char.upper()) if char.isalnum() else 0, char
        )


async def compose(
    person: ChromeBackgroundDriver, steps: list[str], commit: str
) -> None:
    """The person's input method: the text it composes, step by step, then what it commits."""
    session = await session_of(person)
    for step in steps:
        await person._cmd(
            "Input.imeSetComposition",
            {"text": step, "selectionStart": len(step), "selectionEnd": len(step)},
            session,
        )
    await person._cmd("Input.insertText", {"text": commit}, session)


async def panel(person: ChromeBackgroundDriver) -> dict[str, Any]:
    raw = await person._eval("""JSON.stringify((() => {
      const p = document.getElementById('browser-panel'), c = document.getElementById('bp-canvas');
      const r = c.getBoundingClientRect(), k = document.getElementById('bp-keys');
      return {state: p.dataset.state, owner: p.dataset.owner, keys: p.dataset.keys, picture: p.dataset.picture,
        status: document.getElementById('bp-status').textContent, hint: document.getElementById('bp-hint').textContent,
        take: !document.getElementById('bp-take').hidden, give: !document.getElementById('bp-give').hidden,
        focused: document.activeElement === k, left: r.left, top: r.top, width: r.width, height: r.height,
        box: k.value};
    })())""")
    return as_dict(raw)


async def element(driver: ChromeBackgroundDriver, selector: str) -> dict[str, float]:
    """Where an element of melo's page is, in page CSS pixels."""
    raw = await driver._eval(
        f"JSON.stringify((() => {{ const r = document.querySelector({json.dumps(selector)}).getBoundingClientRect();"
        f" return {{x: r.left + r.width / 2, y: r.top + r.height / 2}}; }})())"
    )
    return as_dict(raw)


def on_panel(
    found: dict[str, Any], page_w: float, scale: float, point: dict[str, float]
) -> tuple[float, float]:
    """Where a page point shows on the panel's picture (the inverse of the panel's own mapping)."""
    k = found["width"] / page_w
    return found["left"] + point["x"] * scale * k, found["top"] + point["y"] * scale * k


async def melo_value(driver: ChromeBackgroundDriver, selector: str) -> str:
    return str(
        await driver._eval(f"document.querySelector({json.dumps(selector)}).value")
    )


async def opened_panel(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    person: ChromeBackgroundDriver,
) -> tuple[RealAgent, dict[str, Any], str]:
    session_id = await chat(console, workspace)
    agent = builder.agents[session_id]
    tools = {t.definition.name: t for t in await agent.provider.start()}
    await run(tools, "browser_open", url=base + "/form")
    root = await sign_in(person, served)
    await person.open(f"{root}?pop=1#/live/{session_id}", new_tab=False)
    await settled(lambda: _loaded(person))

    async def live() -> bool:
        found = await panel(person)
        return bool(
            found["state"] == "live" and found["picture"] == "true" and found["take"]
        )

    await settled(live)
    return agent, tools, session_id


async def test_the_person_takes_over_types_and_gives_back_and_melo_reads_it(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    person: ChromeBackgroundDriver,
    tmp_path: Path,
) -> None:
    agent, tools, _ = await opened_panel(
        console, builder, workspace, served, base, person
    )
    melo = agent.driver
    shot = await person.screenshot(None)
    (tmp_path / "panel-before.jpg").write_bytes(base64.b64decode(shot.data))
    # Take over, with the person's own click on the button.
    found = await panel(person)
    take = as_dict(
        await person._eval(
            "JSON.stringify((() => { const r = document.getElementById('bp-take').getBoundingClientRect();"
            " return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })())"
        )
    )
    await click(person, take["x"], take["y"])

    async def yours() -> bool:
        return bool((await panel(person))["owner"] == "you")

    await settled(yours)
    assert agent.provider.owners.owner("t1") == "you"
    found = await panel(person)
    assert (
        found["status"] == "You are in control" and found["give"] and not found["take"]
    )

    # A click on the name field in the picture puts melo's caret there; the keys follow.
    started = time.monotonic()
    x, y = on_panel(found, 1280.0, 1.0, await element(melo, "#name"))
    await click(person, x, y)
    await settled(lambda: _focused_name(melo))
    clicked_s = time.monotonic() - started
    assert (await panel(person))["focused"], "the keys went to the page"
    started = time.monotonic()
    await type_text(person, "Alex ")
    await settled(lambda: _value_is(melo, "#name", "Alex "))
    typed_s = time.monotonic() - started
    # Chinese through an input method: composed, then committed.
    await compose(person, ["n", "ni", "nih", "niha", "nihao"], "你好")
    await settled(lambda: _value_is(melo, "#name", NAME))
    assert (await panel(person))["box"] == " ", "nothing stays in the panel's text box"
    # A paste, as text only (the panel's paste event; a headless Chrome has no clipboard).
    x, y = on_panel(found, 1280.0, 1.0, await element(melo, "#pw"))
    await click(person, x, y)
    await settled(lambda: _focused(melo, "#pw"))
    await person._eval(
        f"(() => {{ const d = new DataTransfer(); d.setData('text/plain', {json.dumps(SECRET)});"
        " d.setData('text/html', '<b>x</b>'); document.getElementById('bp-keys').dispatchEvent("
        "new ClipboardEvent('paste', {clipboardData: d, bubbles: true, cancelable: true})); })()"
    )
    await settled(lambda: _value_is(melo, "#pw", SECRET))
    shot = await person.screenshot(None)
    (tmp_path / "panel-in-control.jpg").write_bytes(base64.b64decode(shot.data))
    await person._cmd(
        "Emulation.setEmulatedMedia",
        {"features": [{"name": "prefers-color-scheme", "value": "light"}]},
        await session_of(person),
    )
    await asyncio.sleep(0.3)
    shot = await person.screenshot(None)
    (tmp_path / "panel-in-control-light.jpg").write_bytes(base64.b64decode(shot.data))

    # Give back to melo, with the person's click.
    give = as_dict(
        await person._eval(
            "JSON.stringify((() => { const r = document.getElementById('bp-give').getBoundingClientRect();"
            " return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })())"
        )
    )
    await click(person, give["x"], give["y"])

    async def melos() -> bool:
        return bool((await panel(person))["owner"] == "melo")

    await settled(melos)
    out = await run(tools, "browser_read")
    assert out.text.startswith("The user had tab t1 for "), out.text[:200]
    assert f'value="{NAME}"' in out.text, "melo reads what was typed in the name field"
    assert (
        SECRET not in out.text and "password field: melo never types here" in out.text
    )
    kept(
        tmp_path,
        "takeover",
        {"click_to_caret_s": round(clicked_s, 3), "type_5_keys_s": round(typed_s, 3)},
    )


async def test_a_point_on_a_pinch_zoomed_page_lands_where_it_shows(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    person: ChromeBackgroundDriver,
    tmp_path: Path,
) -> None:
    agent, _tools, _ = await opened_panel(
        console, builder, workspace, served, base, person
    )
    melo = agent.driver
    session = await melo._session()
    await melo._cmd("Emulation.setPageScaleFactor", {"pageScaleFactor": 2}, session)
    live = agent.provider.live
    assert live is not None

    def zoomed() -> bool:
        return any(
            c.geometry is not None and c.geometry[3] == 2 for c in live._casts.values()
        )

    try:
        async with asyncio.timeout(5):
            while not zoomed():
                await asyncio.sleep(0.05)
    except TimeoutError:
        pytest.skip("this Chrome does not report a pinch zoom in its pictures")
    await person._eval("document.getElementById('bp-take').click()")

    async def yours() -> bool:
        return bool((await panel(person))["owner"] == "you")

    await settled(yours)
    found = await panel(person)
    target = await element(
        melo, "#hit"
    )  # CSS pixels of the page, as the page lays them out
    x, y = on_panel(found, 1280.0, 2.0, target)
    await click(person, x, y)
    await settled(lambda: _pressed(melo))
    shot = await person.screenshot(None)
    (tmp_path / "panel-zoomed.jpg").write_bytes(base64.b64decode(shot.data))
    kept(
        tmp_path,
        "pinch",
        {"scale": 2, "target_css": target, "panel_point": [round(x, 1), round(y, 1)]},
    )


async def _focused_name(driver: ChromeBackgroundDriver) -> bool:
    return await _focused(driver, "#name")


async def _focused(driver: ChromeBackgroundDriver, selector: str) -> bool:
    return bool(
        await driver._eval(
            f"document.activeElement === document.querySelector({json.dumps(selector)})"
        )
    )


async def _value_is(driver: ChromeBackgroundDriver, selector: str, value: str) -> bool:
    return await melo_value(driver, selector) == value


async def _pressed(driver: ChromeBackgroundDriver) -> bool:
    return bool(await driver._eval("document.getElementById('hit').dataset.n === '1'"))
