"""The agent page is a contract with a size limit, and the changelog stays a list.

The tables the code owns (``melo -h all``, the gateway's routes) and all of
``llms.txt`` are written by ``scripts/agent_docs.py``; these tests fail while they
are stale, while a page outgrows its limit, or while a list the code owns names
something a page does not.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import localmelo.__main__ as cli
from localmelo.scripts import agent_docs

ROOT = Path(__file__).resolve().parents[2]
AGENT_PAGE = ROOT / "docs" / "for-agents.md"


def _codes(text: str, heading: str) -> set[int]:
    """The codes in the first column of the table under *heading*."""
    section = text.split(heading, 1)[1].split("\n#", 1)[0]
    found: set[int] = set()
    for cell in re.findall(r"^\| ([\d, ]+) \|", section, re.M):
        found.update(int(code) for code in cell.split(","))
    return found


def test_the_generated_parts_are_up_to_date() -> None:
    stale = agent_docs.sync(check=True)
    assert stale == [], "Run: uv run python scripts/agent_docs.py sync"


def test_the_agent_files_stay_small() -> None:
    page = AGENT_PAGE.read_bytes()
    assert len(page) <= 32 * 1024, f"for-agents.md is {len(page)} bytes; link instead"
    assert max(map(len, page.decode().splitlines())) <= 400
    assert len((ROOT / "docs" / "llms.txt").read_bytes()) <= 10 * 1024


def test_every_route_is_listed_and_documented() -> None:
    routes = agent_docs.routes()
    assert ("POST", "/v1/agent/run") in routes and len(routes) == len(set(routes))
    block = agent_docs.routes_block()
    undocumented = [line for line in block.splitlines() if line.endswith("| — |")]
    assert undocumented == [], "document these routes in docs/reference/"


def test_the_exit_codes_match_the_code_and_the_reference() -> None:
    page = _codes(AGENT_PAGE.read_text(), "### Exit codes")
    reference = _codes((ROOT / "docs/reference/cli.md").read_text(), "## Exit codes")
    constants = {
        value
        for name, value in vars(cli).items()
        if name.startswith("EXIT_") and isinstance(value, int)
    }
    assert constants <= page == reference


def test_every_environment_variable_is_in_the_configuration_reference() -> None:
    product = ("*.py", "*.swift", ":!tests", ":!scripts")
    found = subprocess.run(
        ["git", "grep", "-ohE", r"LOCALMELO_[A-Z0-9_]+", "--", *product],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    reference = (ROOT / "docs/reference/configuration.md").read_text()
    section = reference.split("## Environment variables", 1)[1].split("\n## ", 1)[0]
    assert sorted({n for n in found if f"`{n}`" not in section}) == []


def test_a_changelog_entry_is_one_short_line() -> None:
    lines = (ROOT / "CHANGELOG.md").read_text().splitlines()
    long = [line[:60] for line in lines if len(line) > 240]
    assert long == [], "one line per change a person notices, at most 240 characters"
    continued = [line[:60] for line in lines if line.startswith("  ")]
    assert continued == [], "an entry is one line"
    numbered = [line[:60] for line in lines if re.search(r"#\d+", line)]
    assert numbered == [], "no pull request numbers"
