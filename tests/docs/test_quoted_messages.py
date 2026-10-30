"""The messages the docs quote are the messages melo prints.

The troubleshooting page and the CLI reference show some messages word for word, so a
reader can search for what the terminal said. A message that changes in the code and not
in the page leaves them looking for text that no longer appears. These tests take each
message from the code that prints it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import localmelo.__main__ as cli
from localmelo.support import onboard
from localmelo.support.cli import launch

ROOT = Path(__file__).resolve().parents[2]
TROUBLESHOOTING = (
    "docs/reference/troubleshooting.md",
    "docs/zh-CN/reference/troubleshooting.md",
)
CLI_REFERENCE = ("docs/reference/cli.md", "docs/zh-CN/reference/cli.md")


def _squash(text: str) -> str:
    """*text* on one line, so a message wrapped in a page still matches."""
    return " ".join(text.split())


def _page(name: str) -> str:
    return _squash((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [*TROUBLESHOOTING, *CLI_REFERENCE])
def test_the_pages_quote_the_message_of_a_gateway_with_no_backend(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli._exit_without_setup()
    message = capsys.readouterr().err
    assert message.startswith("melo: no backend is configured; run melo ")
    assert _squash(message) in _page(name)


@pytest.mark.parametrize("name", TROUBLESHOOTING)
@pytest.mark.parametrize(
    "message",
    [launch.NO_TERMINAL, onboard._SAVING_EMBEDDING_OFF],
    ids=["no-terminal", "embedding-off"],
)
def test_the_troubleshooting_page_quotes_the_other_messages_it_shows(
    name: str, message: str
) -> None:
    assert _squash(message) in _page(name)
