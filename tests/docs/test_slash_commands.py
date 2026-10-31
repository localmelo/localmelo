"""The terminal and console guides document the same commands.

The terminal's table is ``support/cli/commands.py``; the console and the app run every
command of it (``GET /api/commands``). Each guide, in English and Chinese, names every one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.support.cli.commands import COMMANDS

ROOT = Path(__file__).resolve().parents[2]
GUIDES = [
    "docs/guides/terminal.md",
    "docs/zh-CN/guides/terminal.md",
    "docs/guides/web-console.md",
    "docs/zh-CN/guides/web-console.md",
]


@pytest.mark.parametrize("name", GUIDES)
def test_a_guide_names_every_command_of_the_terminal(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    missing = [c.name for c in COMMANDS if f"`/{c.name}" not in text]
    assert missing == [], f"{name} does not document /{', /'.join(missing)}"
