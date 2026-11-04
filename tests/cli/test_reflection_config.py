"""``[agent] reflection`` and ``steps_per_attempt``: reflect when stuck after many steps.

``reflection`` (true by default) lets a run reflect once its attempt of 50 steps is used;
``false`` never reflects and runs straight to the total (``attempt_policy = "none"``). The
developer key ``steps_per_attempt`` sets the attempt's length. ``-X agent.reflection=off``
sets the switch for one run, ``melo config set`` saves it, and a direct run, the
interactive UI, the console, a task and the gateway all read it. The option
``--attempt-policy`` keeps its say. What the two policies do to a run is pinned in
``tests/agent/test_long_attempts.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config, config_cli
from localmelo.support.gateway.session import SessionManager
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_attempt_options,
    configured_session_options,
)

BASE = ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m"]
KEY = "agent.reflection"
LENGTH = "agent.steps_per_attempt"


def test_an_old_config_without_the_keys_gets_the_defaults() -> None:
    path = Path(config.config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('chat_backend = "vllm"\n\n[agent]\nreasoning = "legacy"\n')
    loaded = config.load()
    assert loaded.agent.reflection is True
    assert loaded.agent.steps_per_attempt is None
    assert configured_session_options(loaded).attempt_policy == "reflect"
    assert configured_session_options(loaded).steps_per_attempt == 50


@pytest.mark.parametrize("bad", [0, -1])
def test_an_attempt_length_that_is_not_a_positive_int_is_refused(bad: object) -> None:
    cfg = config.Config()
    cfg.agent.steps_per_attempt = bad  # type: ignore[assignment]
    assert any(
        "[agent] steps_per_attempt must be an int >= 1" in e for e in cfg.validate()
    )


def test_the_helper_turns_the_switch_into_the_attempt_policy() -> None:
    off = config.Config(agent=config.AgentConfig(reflection=False))
    on = config.Config()
    assert configured_attempt_options(off, AgentOptions()).attempt_policy == "none"
    assert configured_attempt_options(on, AgentOptions()).attempt_policy == "reflect"
    # Reflection on leaves what the caller set; the switch never turns reflection on.
    none = AgentOptions(attempt_policy="none")
    assert configured_attempt_options(on, none).attempt_policy == "none"
    # An option the caller gave keeps its say over the file.
    reflect = AgentOptions(attempt_policy="reflect")
    kept = configured_attempt_options(off, reflect, {"attempt_policy"})
    assert kept.attempt_policy == "reflect"
    # No config, no change.
    assert configured_attempt_options(None, reflect) is reflect


def test_the_helper_sets_the_attempt_length_and_refuses_a_bad_one() -> None:
    cfg = config.Config(agent=config.AgentConfig(steps_per_attempt=20))
    assert configured_attempt_options(cfg, AgentOptions()).steps_per_attempt == 20
    kept = configured_attempt_options(cfg, AgentOptions(), {"steps_per_attempt"})
    assert kept.steps_per_attempt == 50
    cfg.agent.steps_per_attempt = 0
    with pytest.raises(ValueError, match=r"\[agent\] steps_per_attempt"):
        configured_attempt_options(cfg, AgentOptions())


def test_x_sets_the_switch_for_one_run_and_the_developer_listings_show_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{KEY}=off", *BASE, "fix it"])
    assert config.overrides() == {KEY: False}
    assert config.load().agent.reflection is False
    assert config_cli.main(["keys", "agent.reflection", "-X", f"{KEY}=off"]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row == [KEY, "false", "true", "-X"]
    text = cli._build_parser(developer=True).format_help()
    assert any(
        line.startswith("  agent.steps_per_attempt ") and line.endswith("(#185)")
        for line in text.splitlines()
    ), text


@pytest.mark.parametrize("value", ["maybe", "2"])
def test_x_refuses_any_other_value(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        cli._parse(["-X", f"{KEY}={value}", *BASE, "fix it"])
    assert stop.value.code == 2
    assert "reflection" in capsys.readouterr().err


def test_x_sets_the_attempt_length_and_agent_max_steps_still_points_to_the_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{LENGTH}=10", *BASE, "fix it"])
    assert config.load().agent.steps_per_attempt == 10
    with pytest.raises(SystemExit):
        cli._parse(["-X", "agent.max_steps=5", *BASE, "fix it"])
    assert "-s N (--steps)" in capsys.readouterr().err


@pytest.mark.parametrize("via", ["file", "x", "default"])
@pytest.mark.parametrize("task", ["fix it", ""])
async def test_a_direct_run_reads_the_switch(via: str, task: str) -> None:
    argv = [*BASE, task]
    if via == "file":
        config.save(config.Config(agent=config.AgentConfig(reflection=False)))
    elif via == "x":
        argv = ["-X", f"{KEY}=off", *argv]
    parser, args = cli._parse(argv)
    options = cli._direct_options(parser, args)
    assert (
        options.attempt_policy == "reflect"
    )  # no option sets it: the builder reads it
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.attempt_policy == (
            "reflect" if via == "default" else "none"
        )
        assert agent.options.max_steps == 100
        assert agent.options.steps_per_attempt == 50
    finally:
        await agent.close()


async def test_an_option_keeps_its_say_over_the_file() -> None:
    config.save(config.Config(agent=config.AgentConfig(reflection=False)))
    parser, args = cli._parse(["--attempt-policy", "reflect", *BASE, "fix it"])
    options = cli._direct_options(parser, args)
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.attempt_policy == "reflect"
    finally:
        await agent.close()


async def test_a_direct_run_reads_the_attempt_length() -> None:
    config.save(config.Config(agent=config.AgentConfig(steps_per_attempt=20)))
    parser, args = cli._parse([*BASE, "fix it"])
    options = cli._direct_options(parser, args)
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.steps_per_attempt == 20
        assert agent.options.max_steps == 100
    finally:
        await agent.close()


def test_the_steps_option_sets_the_run_and_leaves_the_attempt() -> None:
    parser, args = cli._parse(["-s", "40", *BASE, "fix it"])
    options = cli._direct_options(parser, args)
    assert (options.max_steps, options.steps_per_attempt) == (40, 50)
    parser, args = cli._parse(["--attempt-policy", "none", "-s", "30", *BASE, "x"])
    options = cli._direct_options(parser, args)
    assert (options.attempt_policy, options.max_steps) == ("none", 30)


async def test_the_interactive_ui_the_console_and_the_gateway_read_the_switch(
    tmp_path: Path,
) -> None:
    cfg = config.Config(agent=config.AgentConfig(reflection=False))
    assert configured_session_options(cfg).attempt_policy == "none"
    assert configured_session_options(config.Config()).attempt_policy == "reflect"
    # An option the terminal chat was given is kept.
    explicit = configured_session_options(
        cfg, AgentOptions(attempt_policy="reflect"), explicit={"attempt_policy"}
    )
    assert explicit.attempt_policy == "reflect"
    agent = await build_session_agent(
        cfg=cfg, base_url=BASE[1], model="m", workspace=str(tmp_path)
    )
    try:
        assert agent.options.attempt_policy == "none"
    finally:
        await agent.close()


def _gateway_options(cfg: config.Config) -> AgentOptions:
    with mock.patch("localmelo.support.gateway.session.Agent") as agent_cls:
        SessionManager(config=cfg)._create_agent()
    options: AgentOptions = agent_cls.call_args.kwargs["options"]
    return options


def test_a_gateway_session_reads_the_switch_and_the_attempt_length() -> None:
    cfg = config.Config(chat_backend="mlc")
    options = _gateway_options(cfg)
    assert (options.attempt_policy, options.max_steps, options.steps_per_attempt) == (
        "reflect",
        100,
        50,
    )
    cfg.agent.reflection = False
    cfg.agent.steps_per_attempt = 25
    options = _gateway_options(cfg)
    assert (options.attempt_policy, options.steps_per_attempt) == ("none", 25)
