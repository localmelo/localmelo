"""``[memory] keep_statements``: a developer setting, off by default.

With it on, code keeps after a chat turn the plain, lasting facts the user
stated that the model did not keep with ``remember``; by default only the
model's judgement and the user's own requests write notes. It is set for one
run with ``-X`` and listed by the developer help only.
"""

from __future__ import annotations

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli

KEY = "memory.keep_statements"


def test_x_turns_it_on_for_one_run_and_only_the_developer_help_lists_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{KEY}=on", "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert config.overrides() == {KEY: True}
    assert config.load().memory.keep_statements is True
    assert config_cli.main(["keys", "keep_statements", "-X", f"{KEY}=on"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert [KEY, "true", "false", "-X"] in rows
    developer = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith(f"  {KEY} ") and line.endswith("(#299)")
        for line in developer.splitlines()
    )
    assert KEY not in cli._build_parser(developer=False).format_help()
