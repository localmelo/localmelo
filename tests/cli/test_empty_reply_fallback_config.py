"""``[agent] empty_reply_fallback``: the switch for what a thinking model's empty turn becomes.

``"on"`` (the default) asks the turn again with thinking off and, failing that, answers
with the end of the thinking; ``"off"`` leaves the run as it ended before. The key has no
option: ``-X agent.empty_reply_fallback=off`` sets it for one run, ``melo config set`` saves
it, and a direct run, the interactive UI, the console and the gateway all read it. What the
two values do to a run is pinned in ``tests/agent/test_thinking_answer.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_session_options,
)

BASE = ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m"]
KEY = "agent.empty_reply_fallback"


@pytest.mark.parametrize("bad", ["maybe", ""])
def test_any_other_value_is_refused_by_the_validation(bad: object) -> None:
    cfg = config.Config()
    cfg.agent.empty_reply_fallback = bad  # type: ignore[assignment]
    errors = cfg.validate()
    assert any(
        f"[agent] empty_reply_fallback {bad!r} is not recognised" in e for e in errors
    ), errors
    assert not any("empty_reply_fallback" in e for e in config.Config().validate())


def test_x_sets_it_for_one_run_and_the_developer_listings_show_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{KEY}=off", *BASE, "fix it"])
    assert config.overrides() == {KEY: "off"}
    assert config.load().agent.empty_reply_fallback == "off"
    assert config_cli.main(["keys", "empty_reply", "-X", f"{KEY}=off"]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row == [KEY, '"off"', '"on"', "-X"]
    text = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith("  agent.empty_reply_fallback ") and line.endswith("(#469)")
        for line in text.splitlines()
    ), text


@pytest.mark.parametrize("value", ["maybe", "false", "0"])
def test_x_refuses_any_other_value(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        cli._parse(["-X", f"{KEY}={value}", *BASE, "fix it"])
    assert stop.value.code == 2
    assert "empty_reply_fallback" in capsys.readouterr().err


@pytest.mark.parametrize("via", ["file", "x", "default"])
@pytest.mark.parametrize("task", ["fix it", ""])
async def test_a_direct_run_reads_it(via: str, task: str) -> None:
    argv = [*BASE, task]
    if via == "file":
        config.save(config.Config(agent=config.AgentConfig(empty_reply_fallback="off")))
    elif via == "x":
        argv = ["-X", f"{KEY}=off", *argv]
    parser, args = cli._parse(argv)
    options = cli._direct_options(parser, args)
    assert (
        options.empty_reply_fallback == "on"
    )  # no option sets it: the builder reads it
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.empty_reply_fallback == (
            "on" if via == "default" else "off"
        )
    finally:
        await agent.close()


async def test_the_interactive_ui_the_console_and_the_gateway_read_it(
    tmp_path: Path,
) -> None:
    cfg = config.Config(agent=config.AgentConfig(empty_reply_fallback="off"))
    assert configured_session_options(cfg).empty_reply_fallback == "off"
    assert configured_session_options(config.Config()).empty_reply_fallback == "on"
    agent = await build_session_agent(
        cfg=cfg, base_url=BASE[1], model="m", workspace=str(tmp_path)
    )
    try:
        assert agent.options.empty_reply_fallback == "off"
    finally:
        await agent.close()
