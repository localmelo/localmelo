"""``[agent] answer_on_stop`` (a key with no option) and ``--reflection-gate``.

A direct run reads the key; the interactive UI, the console and the gateway do not.
Both are on by default: ``answer_on_stop`` is ``true`` and the gate is ``patient``;
``-X agent.answer_on_stop=off`` and ``--reflection-gate flags`` give the earlier
behavior.
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


def test_the_key_is_written_only_when_off_and_must_be_a_boolean() -> None:
    config.save(config.Config())
    path = Path(config.config_path())
    assert "answer_on_stop" not in path.read_text()
    assert config.AgentConfig().answer_on_stop is True
    config.save(config.Config(agent=config.AgentConfig(answer_on_stop=False)))
    assert "answer_on_stop = false" in path.read_text()
    assert config.load().agent.answer_on_stop is False
    config.save(config.Config(agent=config.AgentConfig(answer_on_stop=True)))
    assert "answer_on_stop" not in path.read_text()
    assert config.load().agent.answer_on_stop is True
    bad = config.Config()
    bad.agent.answer_on_stop = "yes"  # type: ignore[assignment]
    assert any("answer_on_stop must be a boolean" in e for e in bad.validate())


def test_x_sets_it_for_one_run_and_config_keys_lists_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", "agent.answer_on_stop=off", *BASE, "fix it"])
    assert config.overrides() == {"agent.answer_on_stop": False}
    assert config.load().agent.answer_on_stop is False
    assert config_cli.main(["keys", "answer_on", "-X", "agent.answer_on_stop=off"]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row == ["agent.answer_on_stop", "false", "true", "-X"]
    text = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith("  agent.answer_on_stop ") and line.endswith("(#185)")
        for line in text.splitlines()
    )


@pytest.mark.parametrize("via", ["file", "x", "default"])
@pytest.mark.parametrize("task", ["fix it", ""])
async def test_a_direct_run_reads_it(via: str, task: str) -> None:
    argv = [*BASE, task]
    if via == "file":
        config.save(config.Config(agent=config.AgentConfig(answer_on_stop=False)))
    elif via == "x":
        argv = ["-X", "agent.answer_on_stop=off", *argv]
    parser, args = cli._parse(argv)
    options = cli._direct_options(parser, args)
    assert (
        options.answer_on_stop is True
    )  # no option sets it: the builder reads the key
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.answer_on_stop is (via == "default")
    finally:
        await agent.close()


async def test_the_interactive_ui_and_the_gateway_ignore_it(tmp_path: Path) -> None:
    cfg = config.Config(agent=config.AgentConfig(answer_on_stop=False))
    parser, args = cli._parse(["-X", "agent.answer_on_stop=off", *BASE])
    assert cli._direct_options(parser, args).answer_on_stop is True  # bare melo
    assert configured_session_options(cfg).answer_on_stop is True  # console, gateway
    agent = await build_session_agent(
        cfg=cfg, base_url=BASE[1], model="m", workspace=str(tmp_path)
    )
    try:
        assert agent.options.answer_on_stop is True
    finally:
        await agent.close()


@pytest.mark.parametrize("spelling", ["flag", "x"])
def test_the_flags_gate_reaches_the_options(spelling: str) -> None:
    given = (
        ["--reflection-gate", "flags"]
        if spelling == "flag"
        else ["-X", "agent.reflection_gate=flags"]
    )
    parser, args = cli._parse([*given, *BASE, "fix it"])
    assert cli._direct_options(parser, args).reflection_gate == "flags"


def test_the_gate_defaults_are_patient_and_the_answer_request() -> None:
    parser, args = cli._parse([*BASE, "fix it"])
    options = cli._direct_options(parser, args)
    assert (options.reflection_gate, options.answer_on_stop) == ("patient", True)
