"""``cache_layout`` and ``window_messages`` at the configuration and CLI boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.melo.schema import (
    SHORT_TERM_MAX,
    WINDOW_MESSAGES_MAX,
    WINDOW_MESSAGES_MIN,
)
from localmelo.support.config import AgentConfig, Config, load, save
from localmelo.support.option_tiers import EXPERIMENTAL, OPTION_TIERS
from localmelo.support.sessions.agents import (
    configured_cache_options,
    configured_session_options,
)

URL = "http://127.0.0.1:1/v1"
# name -> (default, a value that is not, the flag, its text)
SETTINGS: dict[str, tuple[Any, Any, str, str]] = {
    "cache_layout": ("v2", "v3", "--cache-layout", "v3"),
    "window_messages": (SHORT_TERM_MAX, 64, "--window-messages", "64"),
}


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    return tmp_path


def test_the_defaults_are_today_s_requests() -> None:
    assert AgentOptions().cache_layout == "v2"
    assert AgentOptions().window_messages == SHORT_TERM_MAX == 20
    assert AgentConfig().cache_layout == "v2"
    assert AgentConfig().window_messages == SHORT_TERM_MAX
    assert (WINDOW_MESSAGES_MIN, WINDOW_MESSAGES_MAX) == (8, 256)


@pytest.mark.parametrize("name", SETTINGS)
def test_the_config_roundtrip_writes_only_a_non_default_value(
    name: str, config_home: Path
) -> None:
    default, other, _, _ = SETTINGS[name]
    save(Config())
    assert name not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(**{name: default})))
    assert name not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(**{name: other})))
    text = (config_home / "config.toml").read_text()
    assert (
        f'{name} = "{other}"' if isinstance(other, str) else f"{name} = {other}"
    ) in text
    assert getattr(load().agent, name) == other
    assert getattr(configured_session_options(load()), name) == other
    explicit = configured_session_options(
        load(), AgentOptions(**{name: default}), explicit={name}
    )
    assert getattr(explicit, name) == default
    kept = configured_cache_options(load(), AgentOptions(), keep={name})
    assert getattr(kept, name) == default


def test_unknown_values_in_the_config_are_refused() -> None:
    cfg = Config()
    cfg.agent.cache_layout = "v9"
    assert any("cache_layout 'v9'" in e for e in cfg.validate())
    for bad in (7, 257, True, "20", 20.0):
        cfg = Config()
        cfg.agent.window_messages = bad  # type: ignore[assignment]
        assert any(
            "window_messages must be an int from 8 to 256" in e for e in cfg.validate()
        )
    cfg = Config()
    cfg.agent.window_messages = 256
    assert not any("window_messages" in e for e in cfg.validate())


def test_the_flags_are_validated_by_the_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    for argv, text in (
        (["--cache-layout", "v4", "t"], "--cache-layout"),
        (["--window-messages", "7", "t"], "from 8 to 256, got 7"),
        (["--window-messages", "257", "t"], "from 8 to 256, got 257"),
        (["--window-messages", "many", "t"], "not an integer"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
        assert text in capsys.readouterr().err


@pytest.mark.parametrize("name", SETTINGS)
def test_a_flag_reaches_the_options_and_none_keeps_the_default(name: str) -> None:
    default, other, flag, text = SETTINGS[name]
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, flag, text, "t"])
    assert getattr(cli._direct_options(parser, args), name) == other
    args = parser.parse_args(["--base-url", URL, "t"])
    assert getattr(cli._direct_options(parser, args), name) == default


def test_both_flags_are_experimental_and_decided_in_183() -> None:
    for flag in ("--cache-layout", "--window-messages"):
        assert OPTION_TIERS[flag] == EXPERIMENTAL
    assert not cli._build_parser(developer=False).format_help().count("--cache-layout")
    assert "--cache-layout" in cli._build_parser(developer=True).format_help()


@pytest.mark.parametrize("name", SETTINGS)
async def test_direct_cli_takes_the_config_and_a_flag_beats_it(
    name: str, config_home: Path
) -> None:
    _, other, flag, text = SETTINGS[name]
    save(Config(agent=AgentConfig(**{name: other})))
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert getattr(agent.options, name) == other
        assert agent.hippo.working._max_len == (
            other if name == "window_messages" else SHORT_TERM_MAX
        )
    finally:
        await agent.close()
    default = SETTINGS[name][0]
    flagged_text = str(default)
    args = parser.parse_args(["--base-url", URL, flag, flagged_text, "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert getattr(agent.options, name) == default
    finally:
        await agent.close()


async def test_direct_cli_refuses_bad_config_values(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli._build_parser()
    for body, text in (
        ('cache_layout = "v9"', "cache_layout 'v9'"),
        ("window_messages = 3", "window_messages must be an int from 8 to 256"),
    ):
        (config_home / "config.toml").write_text(f"[agent]\n{body}\n")
        args = parser.parse_args(["--base-url", URL, "task"])
        with pytest.raises(SystemExit):
            cli._build_direct_mode_agent(
                args, parser=parser, options=cli._direct_options(parser, args)
            )
        assert text in capsys.readouterr().err


async def test_the_result_config_names_only_a_non_default_value(
    config_home: Path,
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        config = agent._run_config()
    finally:
        await agent.close()
    assert "cache_layout" not in config and "window_messages" not in config
    args = parser.parse_args(
        ["--base-url", URL, "--cache-layout", "v3", "--window-messages", "64", "task"]
    )
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        config = agent._run_config()
    finally:
        await agent.close()
    assert (config["cache_layout"], config["window_messages"]) == ("v3", 64)
