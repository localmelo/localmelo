"""``[agent] lean_output``: a key with no option; a task reads it, the UI and gateway do not."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent._chat import LEAN_SYSTEM_PROMPT, SYSTEM_PROMPT
from localmelo.support import config, config_cli
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_session_options,
)

BASE = ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m"]


def test_the_key_is_written_only_when_on_and_must_be_a_boolean() -> None:
    config.save(config.Config())
    path = Path(config.config_path())
    assert "lean_output" not in path.read_text()
    config.save(config.Config(agent=config.AgentConfig(lean_output=True)))
    assert "lean_output = true" in path.read_text()
    assert config.load().agent.lean_output is True
    bad = config.Config()
    bad.agent.lean_output = "yes"  # type: ignore[assignment]
    assert any("lean_output must be a boolean" in e for e in bad.validate())


def test_x_sets_it_for_one_run_and_config_keys_lists_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", "agent.lean_output=true", *BASE, "fix it"])
    assert config.overrides() == {"agent.lean_output": True}
    assert config.load().agent.lean_output is True
    assert config_cli.main(["keys", "lean", "-X", "agent.lean_output=on"]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row == ["agent.lean_output", "true", "false", "-X"]
    text = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith("  agent.lean_output ") and line.endswith("(#183)")
        for line in text.splitlines()
    )


@pytest.mark.parametrize("via", ["file", "x"])
@pytest.mark.parametrize(("task", "lean"), [("fix it", True), ("", False)])
async def test_a_task_reads_it_and_the_loop_does_not(
    via: str, task: str, lean: bool
) -> None:
    argv = [*BASE, task]
    if via == "file":
        config.save(config.Config(agent=config.AgentConfig(lean_output=True)))
    else:
        argv = ["-X", "agent.lean_output=true", *argv]
    parser, args = cli._parse(argv)
    options = cli._direct_options(parser, args)
    assert options.lean_output is False  # no option sets it: the builder reads the key
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.lean_output is lean
        assert agent.chat.system_prompt == (
            LEAN_SYSTEM_PROMPT if lean else SYSTEM_PROMPT
        )
    finally:
        await agent.close()


async def test_the_interactive_ui_and_the_gateway_ignore_it(tmp_path: Path) -> None:
    cfg = config.Config(agent=config.AgentConfig(lean_output=True))
    parser, args = cli._parse(["-X", "agent.lean_output=true", *BASE])
    assert cli._direct_options(parser, args).lean_output is False  # bare melo
    assert configured_session_options(cfg).lean_output is False  # console, gateway
    agent = await build_session_agent(
        cfg=cfg, base_url=BASE[1], model="m", workspace=str(tmp_path)
    )
    try:
        assert agent.options.lean_output is False
        assert agent.chat.system_prompt == SYSTEM_PROMPT
    finally:
        await agent.close()
