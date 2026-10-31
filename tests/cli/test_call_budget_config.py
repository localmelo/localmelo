"""``[agent] max_calls`` (a key with no option): a direct run reads it.

A developer setting, unset by default. The interactive UI, the console and the
gateway do not read it.
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


def test_the_key_is_written_only_when_set_and_must_be_an_int_of_two_or_more() -> None:
    config.save(config.Config())
    path = Path(config.config_path())
    assert "max_calls" not in path.read_text()
    assert config.AgentConfig().max_calls is None
    config.save(config.Config(agent=config.AgentConfig(max_calls=30)))
    assert "max_calls = 30" in path.read_text()
    assert config.load().agent.max_calls == 30
    for bad in (1, 0, True, "30", 2.5):
        cfg = config.Config()
        cfg.agent.max_calls = bad  # type: ignore[assignment]
        assert any("max_calls must be an int >= 2" in e for e in cfg.validate()), bad


def test_x_sets_it_for_one_run_and_config_keys_lists_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", "agent.max_calls=30", *BASE, "fix it"])
    assert config.overrides() == {"agent.max_calls": 30}
    assert config.load().agent.max_calls == 30
    assert config_cli.main(["keys", "max_calls", "-X", "agent.max_calls=30"]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row == ["agent.max_calls", "30", "unset", "-X"]
    text = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith("  agent.max_calls ") and line.endswith("(#185)")
        for line in text.splitlines()
    )


@pytest.mark.parametrize("bad", ["1", "0", "many", "true"])
def test_x_refuses_a_value_that_is_not_an_int_of_two_or_more(
    bad: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        cli._parse(["-X", f"agent.max_calls={bad}", *BASE, "fix it"])
    assert stop.value.code == 2
    assert "agent.max_calls" in capsys.readouterr().err


@pytest.mark.parametrize("via", ["file", "x", "unset"])
@pytest.mark.parametrize("task", ["fix it", ""])
async def test_a_direct_run_reads_it(via: str, task: str) -> None:
    argv = [*BASE, task]
    if via == "file":
        config.save(config.Config(agent=config.AgentConfig(max_calls=7)))
    elif via == "x":
        argv = ["-X", "agent.max_calls=7", *argv]
    parser, args = cli._parse(argv)
    options = cli._direct_options(parser, args)
    assert options.max_calls is None  # no option sets it: the builder reads the key
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.max_calls == (None if via == "unset" else 7)
        assert agent.chat.max_calls == agent.options.max_calls
    finally:
        await agent.close()


async def test_the_chat_keeps_a_call_for_the_answer_only_with_answer_on_stop() -> None:
    parser, args = cli._parse(
        ["-X", "agent.max_calls=7", "-X", "agent.answer_on_stop=on", *BASE, "fix it"]
    )
    options = cli._direct_options(parser, args)
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert (agent.chat.max_calls, agent.chat.reserve_answer) == (7, True)
    finally:
        await agent.close()


async def test_the_interactive_ui_and_the_gateway_ignore_it(tmp_path: Path) -> None:
    cfg = config.Config(agent=config.AgentConfig(max_calls=7))
    parser, args = cli._parse(["-X", "agent.max_calls=7", *BASE])
    assert cli._direct_options(parser, args).max_calls is None  # bare melo
    assert configured_session_options(cfg).max_calls is None  # console, gateway
    agent = await build_session_agent(
        cfg=cfg, base_url=BASE[1], model="m", workspace=str(tmp_path)
    )
    try:
        assert agent.options.max_calls is None
    finally:
        await agent.close()
