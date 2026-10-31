"""``GET /api/commands``: the terminal's slash commands for the console's "/" menu.

The list is the terminal's own table. The fixture ``tests/fixtures/slash_commands.json`` is the
route's answer; ``tests/gateway/test_slash.mjs`` drives every command in it through the page, so a
command added to the terminal fails here until the page has an action for it, and fails there
until that action exists. After a change to the table or its words, write the fixture again with
``uv run python tests/gateway/console/test_commands.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from localmelo.support.cli import commands as terminal
from localmelo.support.gateway.console.routes_commands import (
    PAGE_ACTIONS,
    commands_json,
)
from localmelo.support.words import catalog

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "slash_commands.json"
CJK = re.compile(r"[一-鿿]")


def test_the_route_lists_the_terminals_commands_in_the_terminals_order(
    client: TestClient,
) -> None:
    response = client.get("/api/commands")

    assert response.status_code == 200
    listed = response.json()["commands"]
    assert [item["name"] for item in listed] == [c.name for c in terminal.COMMANDS]
    for item, command in zip(listed, terminal.COMMANDS, strict=True):
        assert item["aliases"] == list(command.aliases)
        assert item["args"] == command.args
        assert item["during_turn"] == command.during_turn
        assert item["keys"] == command.keys
        assert item["help"]["en"] == command.summary


def test_every_command_has_its_help_in_both_languages() -> None:
    english, chinese = catalog("en"), catalog("zh-Hans")
    for command in terminal.COMMANDS:
        key = f"commands.{command.name}.help"
        # The catalog's English is the terminal's summary, written once.
        assert english[key] == command.summary
        assert CJK.search(str(chinese[key])), key
    listed = commands_json()["commands"]
    assert all(CJK.search(item["help"]["zh-Hans"]) for item in listed)


def test_a_command_without_a_page_action_fails_here() -> None:
    names = {command.name for command in terminal.COMMANDS}
    assert set(PAGE_ACTIONS) == names, "name every terminal command's page action"
    assert all(PAGE_ACTIONS.values())
    assert [item["action"] for item in commands_json()["commands"]] == [
        PAGE_ACTIONS[command.name] for command in terminal.COMMANDS
    ]


def test_the_everyday_commands_are_the_terminals_help_groups() -> None:
    everyday = commands_json()["everyday"]
    names = {command.name for command in terminal.COMMANDS}
    assert everyday and set(everyday) <= names
    assert len(everyday) == len(set(everyday))
    for _, group in (*terminal.HELP_GROUPS, *terminal.MAIN_HELP_GROUPS):
        assert set(group) <= set(everyday)


def test_the_page_test_reads_the_routes_answer() -> None:
    hint = "uv run python tests/gateway/console/test_commands.py"
    assert json.loads(FIXTURE.read_text("utf-8")) == commands_json(), hint


if __name__ == "__main__":
    FIXTURE.write_text(
        json.dumps(commands_json(), ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(f"wrote {FIXTURE}")
