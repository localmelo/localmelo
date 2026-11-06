"""``[agent] thinking_steps``: on which planning steps the model thinks (#551).

``"all"`` (the default) leaves every request as it was; ``"first_and_failed"`` gives the
provider the fields that turn thinking off for the steps that need none. The key has no
option: ``-X agent.thinking_steps=first_and_failed`` sets it for one run, ``melo config set``
saves it, and a direct run (with ``--base-url`` or a configured backend) and the sessions
all read it. What the two values do to a run is pinned in ``tests/agent/test_thinking_steps.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config, config_cli
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_session_options,
    with_quiet_steps,
)

BASE = ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m"]
KEY = "agent.thinking_steps"
QUIET = AgentOptions(thinking_steps="first_and_failed")
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("bad", ["never", ""])
def test_any_other_value_is_refused_by_the_validation(bad: object) -> None:
    cfg = config.Config()
    cfg.agent.thinking_steps = bad  # type: ignore[assignment]
    assert any(
        f"[agent] thinking_steps {bad!r} is not recognised" in e for e in cfg.validate()
    )
    assert not any("thinking_steps" in e for e in config.Config().validate())


def test_x_sets_it_for_one_run_and_the_developer_listings_show_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", f"{KEY}=first_and_failed", *BASE, "fix it"])
    assert config.overrides() == {KEY: "first_and_failed"}
    assert config.load().agent.thinking_steps == "first_and_failed"
    assert config_cli.main(["keys", "thinking_steps", "-X", f"{KEY}=all"]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row == [KEY, '"all"', '"all"', "-X"]
    text = cli._build_parser(developer=True).format_help()
    listed = next(line for line in text.splitlines() if line.startswith(f"  {KEY} "))
    assert "(#551)" in listed, "a comparison: the issue that decides it"


def test_x_refuses_any_other_value(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli._parse(["-X", f"{KEY}=never", *BASE, "fix it"])
    assert stop.value.code == 2
    assert "thinking_steps" in capsys.readouterr().err


@pytest.mark.parametrize("via", ["file", "x", "default"])
async def test_a_direct_run_reads_it_and_only_then_gets_the_thinking_switch(
    via: str,
) -> None:
    argv = [*BASE, "fix it"]
    if via == "file":
        config.save(
            config.Config(agent=config.AgentConfig(thinking_steps="first_and_failed"))
        )
    elif via == "x":
        argv = ["-X", f"{KEY}=first_and_failed", *argv]
    parser, args = cli._parse(argv)
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        on = via != "default"
        assert agent.options.thinking_steps == ("first_and_failed" if on else "all")
        assert dict(agent._llm.thinking_off_body) == (OFF if on else {})  # type: ignore[attr-defined]
    finally:
        await agent.close()


async def test_the_interactive_ui_the_console_and_the_gateway_read_it(
    tmp_path: Path,
) -> None:
    cfg = config.Config(agent=config.AgentConfig(thinking_steps="first_and_failed"))
    assert configured_session_options(cfg).thinking_steps == "first_and_failed"
    assert configured_session_options(config.Config()).thinking_steps == "all"
    agent = await build_session_agent(
        cfg=cfg, base_url=BASE[1], model="m", workspace=str(tmp_path)
    )
    try:
        assert agent.options.thinking_steps == "first_and_failed"
        assert dict(agent._llm.thinking_off_body) == OFF  # type: ignore[attr-defined]
    finally:
        await agent.close()


def test_the_switch_is_the_backends_own_and_a_hosted_one_is_kept() -> None:
    local = OpenAICompatLLM(base_url="http://127.0.0.1:1/v1", model="m")
    with_quiet_steps(local, None, QUIET)
    assert dict(local.thinking_off_body) == OFF
    ollama = OllamaNativeChat(base_url="http://127.0.0.1:1", model="m")
    with_quiet_steps(ollama, "ollama", QUIET)
    assert dict(ollama.thinking_off_body) == {"think": False}
    hosted = OpenAICompatLLM(
        base_url="http://127.0.0.1:1/v1",
        model="m",
        thinking_off_body={"thinking": {"type": "disabled"}},
    )
    with_quiet_steps(hosted, "deepseek", QUIET)
    assert dict(hosted.thinking_off_body) == {"thinking": {"type": "disabled"}}
    # A cloud service with no switch of its own is sent nothing it might refuse.
    cloud = OpenAICompatLLM(base_url="http://127.0.0.1:1/v1", model="m")
    with_quiet_steps(cloud, "openai", QUIET)
    assert dict(cloud.thinking_off_body) == {}
    # Under the default no provider is touched.
    plain = OpenAICompatLLM(base_url="http://127.0.0.1:1/v1", model="m")
    with_quiet_steps(plain, None, AgentOptions())
    assert dict(plain.thinking_off_body) == {}
