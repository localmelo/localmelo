"""A check of Safari's automation server with no model: ``python -m localmelo.support.browser.probe``.

It starts the real driver (so Safari opens an automation window), lists the
server's tools and their argument names, reads one page, lists tabs, takes a
picture and closes. Each line says what the server answered, so a mismatch
with what melo expects shows here and not in a chat. Give another address as
the first argument.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys

from localmelo.support.browser.driver import BrowserError
from localmelo.support.browser.safari import SafariAutomationDriver


async def probe(url: str) -> int:
    command, *args = shlex.split(os.environ.get("LOCALMELO_SAFARIDRIVER", "")) or [
        "/usr/bin/safaridriver"
    ]
    driver = SafariAutomationDriver(command=command, args=(*args, "--mcp"))
    try:
        await driver.start()
        print(f"tools ({len(driver.tool_names)}): {' '.join(driver.tool_names)}")
        for name, tool in sorted(driver._tools.items()):
            props = tool.input_schema.get("properties") or {}
            print(f"  {name}({', '.join(props)})")
        info = await driver.open(url, new_tab=True)
        print(f"opened: tab={info.id!r} url={info.url!r} title={info.title!r}")
        raw = await driver.page(None)
        print(f"page: {len(raw.text)} characters, {len(raw.elements)} elements")
        print("  first text:", " ".join(raw.text.split())[:120])
        for element in raw.elements[:5]:
            print(f"  {element.kind} {element.label[:40]!r} {element.href[:60]}")
        print("tabs:", [(t.id, t.url) for t in await driver.tabs()])
        shot = await driver.screenshot(None)
        print(f"screenshot: {shot.mime_type}, {len(shot.data) * 3 // 4} bytes")
        print("ok")
        return 0
    except BrowserError as exc:
        print(f"FAILED: {exc}")
        return 1
    finally:
        await driver.close()


if __name__ == "__main__":
    sys.exit(
        asyncio.run(probe(sys.argv[1] if len(sys.argv) > 1 else "https://example.com/"))
    )
