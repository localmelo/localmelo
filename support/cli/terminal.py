"""Clipboard writes requested explicitly by the person at the terminal."""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from collections.abc import Callable
from typing import Final, Literal

PBCOPY: Final = "/usr/bin/pbcopy"
CLIPBOARD_TIMEOUT_S: Final = 2.0


async def copy_text(
    text: str, control: Callable[[str], bool]
) -> Literal["pbcopy", "osc52"]:
    """Prefer the local macOS clipboard; OSC 52 is a request, not an acknowledgement."""
    if sys.platform == "darwin" and not (
        os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION")
    ):
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                PBCOPY,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async with asyncio.timeout(CLIPBOARD_TIMEOUT_S):
                await process.communicate(text.encode("utf-8"))
            if process.returncode == 0:
                return "pbcopy"
        except (OSError, TimeoutError):
            pass
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    if not control(f"\x1b]52;c;{payload}\x07"):
        raise ValueError("Could not send the answer to this terminal's clipboard.")
    return "osc52"
