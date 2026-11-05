"""``window_messages`` at the configuration and CLI boundaries.

It defaults to 64 on a local server and on a URL, and to 20 on a cloud vendor's
backend; the config leaves it unset (``None``) until a file sets it, and a value that
is set always wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.melo.schema import (
    LOCAL_WINDOW_MESSAGES,
    SHORT_TERM_MAX,
    WINDOW_MESSAGES_MAX,
    WINDOW_MESSAGES_MIN,
)
from localmelo.support.config import (
    AgentConfig,
    CloudVendorConfig,
    Config,
    default_window_messages,
    load,
    save,
)
from localmelo.support.option_tiers import (
    ADVANCED,
    EXPERIMENTAL_ISSUES,
    OPTION_TIERS,
)
from localmelo.support.sessions.agents import (
    configured_cache_options,
    configured_session_options,
)

URL = "http://127.0.0.1:1/v1"
# The AgentOptions default, the AgentConfig default (unset: the backend decides) and a
# value that is not the default.
OPTION_DEFAULT: Final = LOCAL_WINDOW_MESSAGES
SET_VALUE: Final = SHORT_TERM_MAX


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    return tmp_path


def test_the_defaults_are_the_measured_window() -> None:
    assert AgentOptions().window_messages == LOCAL_WINDOW_MESSAGES == 64
    assert AgentConfig().window_messages is None  # the backend decides
    assert (WINDOW_MESSAGES_MIN, WINDOW_MESSAGES_MAX) == (8, 256)


@pytest.mark.parametrize(
    ("backend", "cap"),
    [
        ("ollama", 64),
        ("mlc", 64),
        ("vllm", 64),
        ("sglang", 64),
        ("openai-compatible", 64),  # a --base-url server
        ("", 64),
        ("openai", 20),
        ("gemini", 20),
        ("anthropic", 20),
        ("nvidia", 20),
    ],
)
def test_the_window_default_is_64_except_on_a_cloud_vendor(
    backend: str, cap: int
) -> None:
    assert default_window_messages(backend) == cap
    assert AgentConfig().window_for(backend) == cap
    # A value the config sets wins on every backend.
    assert AgentConfig(window_messages=32).window_for(backend) == 32
    assert AgentConfig(window_messages=64).window_for(backend) == 64
    assert AgentConfig(window_messages=20).window_for(backend) == 20


def test_the_config_roundtrip_writes_only_a_value_that_is_set(
    config_home: Path,
) -> None:
    save(Config())
    assert "window_messages" not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(window_messages=SET_VALUE)))
    assert load().agent.window_messages == SET_VALUE
    assert configured_session_options(load()).window_messages == SET_VALUE
    explicit = configured_session_options(
        load(),
        AgentOptions(window_messages=OPTION_DEFAULT),
        explicit={"window_messages"},
    )
    assert explicit.window_messages == OPTION_DEFAULT
    kept = configured_cache_options(load(), AgentOptions(), keep={"window_messages"})
    assert kept.window_messages == OPTION_DEFAULT


def test_a_window_set_to_the_local_default_stays_on_a_cloud_backend(
    config_home: Path,
) -> None:
    # The config's own default is unset, so 64 written by hand is a choice.
    save(Config(chat_backend="openai", agent=AgentConfig(window_messages=64)))
    assert "window_messages = 64" in (config_home / "config.toml").read_text()
    assert configured_session_options(load()).window_messages == 64
    save(Config(chat_backend="openai"))
    assert configured_session_options(load()).window_messages == 20


def test_the_window_follows_the_backend_unless_the_config_sets_it() -> None:
    options = AgentOptions()
    for backend, cap in (("vllm", 64), ("ollama", 64), ("openai", 20), ("nvidia", 20)):
        cfg = Config(chat_backend=backend)
        assert configured_session_options(cfg).window_messages == cap
        assert configured_cache_options(cfg, options).window_messages == cap
    cloud = Config(chat_backend="openai")
    # A URL run is local whatever backend the config names.
    assert (
        configured_cache_options(cloud, options, backend="openai-compatible") == options
    )
    assert configured_cache_options(cloud, options, keep={"window_messages"}) is options
    assert configured_cache_options(Config(), options) is options


def test_unknown_values_in_the_config_are_refused() -> None:
    for bad in (7, 257):
        cfg = Config()
        cfg.agent.window_messages = bad  # type: ignore[assignment]
        assert any(
            "window_messages must be an int from 8 to 256" in e for e in cfg.validate()
        )
    for good in (None, 8, 256):
        cfg = Config()
        cfg.agent.window_messages = good
        assert not any("window_messages" in e for e in cfg.validate())


def test_the_flags_are_validated_by_the_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    for argv, text in (
        (["--window-messages", "7", "t"], "from 8 to 256, got 7"),
        (["--window-messages", "257", "t"], "from 8 to 256, got 257"),
        (["--window-messages", "many", "t"], "not an integer"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
        assert text in capsys.readouterr().err


def test_the_flag_reaches_the_options_and_none_keeps_the_default() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, "--window-messages", "20", "t"])
    assert cli._direct_options(parser, args).window_messages == SET_VALUE
    args = parser.parse_args(["--base-url", URL, "t"])
    assert cli._direct_options(parser, args).window_messages == OPTION_DEFAULT


def test_the_flag_is_an_advanced_option() -> None:
    flag = "--window-messages"
    assert OPTION_TIERS[flag] == ADVANCED
    assert flag not in EXPERIMENTAL_ISSUES
    assert flag not in cli._build_parser(developer=False).format_help()
    assert flag in cli._build_parser(developer=True).format_help()


def test_the_help_names_the_default() -> None:
    # `-h all` lists each setting with its default; the option keeps its own help,
    # which says what an unset window means.
    text = " ".join(cli._build_parser(developer=True).format_help().split())
    assert "agent.window_messages unset --window-messages" in text
    actions = cli._build_parser()._option_string_actions
    window = " ".join(str(actions["--window-messages"].help).split())
    assert "(default 64 on a local server or a URL, 20 on a cloud vendor)" in window


async def test_direct_cli_takes_the_config_and_a_flag_beats_it(
    config_home: Path,
) -> None:
    save(Config(agent=AgentConfig(window_messages=SET_VALUE)))
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert agent.options.window_messages == SET_VALUE
        assert agent.hippo.working._max_len == SET_VALUE
    finally:
        await agent.close()
    args = parser.parse_args(
        ["--base-url", URL, "--window-messages", str(OPTION_DEFAULT), "task"]
    )
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert agent.options.window_messages == OPTION_DEFAULT
    finally:
        await agent.close()


async def test_a_url_run_has_the_local_window_whatever_backend_the_config_names(
    config_home: Path,
) -> None:
    save(Config(chat_backend="openai"))
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert agent.options.window_messages == LOCAL_WINDOW_MESSAGES
    finally:
        await agent.close()
    # A window the config sets still wins for a URL.
    save(Config(chat_backend="openai", agent=AgentConfig(window_messages=32)))
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert agent.options.window_messages == 32
    finally:
        await agent.close()


async def test_a_configured_cloud_backend_keeps_the_earlier_window(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # nothing is sent to it
    cfg = Config(
        chat_backend="openai",
        embedding_backend="none",
        openai=CloudVendorConfig(api_key_env="OPENAI_API_KEY", chat_model="stub"),
    )
    parser = cli._build_parser()

    async def window(*argv: str) -> int:
        args = parser.parse_args([*argv, "task"])
        agent = cli._build_direct_mode_agent(
            args, parser=parser, options=cli._direct_options(parser, args)
        )
        try:
            return agent.options.window_messages
        finally:
            await agent.close()

    save(cfg)
    assert await window() == 20
    assert await window("--window-messages", "32") == 32  # a flag wins
    cfg.agent.window_messages = 64  # so does a value in the file
    save(cfg)
    assert await window() == 64


async def test_the_result_config_names_the_window_unless_earlier(
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
    assert config["window_messages"] == 64
    args = parser.parse_args(["--base-url", URL, "--window-messages", "20", "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        config = agent._run_config()
    finally:
        await agent.close()
    assert "window_messages" not in config
