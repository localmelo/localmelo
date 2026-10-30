"""``cache_layout`` and ``window_messages`` at the configuration and CLI boundaries.

``cache_layout`` defaults to ``"v3"`` everywhere. ``window_messages`` defaults to 64
on a local server and on a URL, and to 20 on a cloud vendor's backend; the config
leaves it unset (``None``) until a file sets it, and a value that is set always wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
# name -> (the AgentOptions default, the AgentConfig default, an opt-out value, the
# flag, its text). The config default of the window is unset: the backend decides.
SETTINGS: dict[str, tuple[Any, Any, Any, str, str]] = {
    "cache_layout": ("v3", "v3", "v2", "--cache-layout", "v2"),
    "window_messages": (
        LOCAL_WINDOW_MESSAGES,
        None,
        SHORT_TERM_MAX,
        "--window-messages",
        "20",
    ),
}


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    return tmp_path


def test_the_defaults_are_the_measured_layout_and_window() -> None:
    assert AgentOptions().cache_layout == "v3"
    assert AgentOptions().window_messages == LOCAL_WINDOW_MESSAGES == 64
    assert AgentConfig().cache_layout == "v3"
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


@pytest.mark.parametrize("name", SETTINGS)
def test_the_config_roundtrip_writes_only_a_value_that_is_set(
    name: str, config_home: Path
) -> None:
    option_default, config_default, opt_out, _, _ = SETTINGS[name]
    save(Config())
    assert name not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(**{name: config_default})))
    assert name not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(**{name: opt_out})))
    text = (config_home / "config.toml").read_text()
    assert (
        f'{name} = "{opt_out}"' if isinstance(opt_out, str) else f"{name} = {opt_out}"
    ) in text
    assert getattr(load().agent, name) == opt_out
    assert getattr(configured_session_options(load()), name) == opt_out
    explicit = configured_session_options(
        load(), AgentOptions(**{name: option_default}), explicit={name}
    )
    assert getattr(explicit, name) == option_default
    kept = configured_cache_options(load(), AgentOptions(), keep={name})
    assert getattr(kept, name) == option_default


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
    cfg = Config()
    cfg.agent.cache_layout = "v9"
    assert any("cache_layout 'v9'" in e for e in cfg.validate())
    for bad in (7, 257, True, "20", 20.0):
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
    default, _, opt_out, flag, text = SETTINGS[name]
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, flag, text, "t"])
    assert getattr(cli._direct_options(parser, args), name) == opt_out
    args = parser.parse_args(["--base-url", URL, "t"])
    assert getattr(cli._direct_options(parser, args), name) == default


def test_both_flags_are_advanced_options() -> None:
    for flag in ("--cache-layout", "--window-messages"):
        assert OPTION_TIERS[flag] == ADVANCED
        assert flag not in EXPERIMENTAL_ISSUES
        assert flag not in cli._build_parser(developer=False).format_help()
        assert flag in cli._build_parser(developer=True).format_help()


def test_the_help_names_the_defaults() -> None:
    text = " ".join(cli._build_parser(developer=True).format_help().split())
    assert "v3 (default) appends a reflection" in text
    assert "v2 sends the earlier requests" in text
    assert "(default 64 on a local server or a URL, 20 on a cloud vendor)" in text


@pytest.mark.parametrize("name", SETTINGS)
async def test_direct_cli_takes_the_config_and_a_flag_beats_it(
    name: str, config_home: Path
) -> None:
    default, _, opt_out, flag, _ = SETTINGS[name]
    save(Config(agent=AgentConfig(**{name: opt_out})))
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert getattr(agent.options, name) == opt_out
        assert agent.hippo.working._max_len == (
            opt_out if name == "window_messages" else LOCAL_WINDOW_MESSAGES
        )
    finally:
        await agent.close()
    args = parser.parse_args(["--base-url", URL, flag, str(default), "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert getattr(agent.options, name) == default
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

    async def window(*argv: str) -> tuple[str, int]:
        args = parser.parse_args([*argv, "task"])
        agent = cli._build_direct_mode_agent(
            args, parser=parser, options=cli._direct_options(parser, args)
        )
        try:
            return agent.options.cache_layout, agent.options.window_messages
        finally:
            await agent.close()

    save(cfg)
    assert await window() == ("v3", 20)
    assert await window("--window-messages", "32") == ("v3", 32)  # a flag wins
    cfg.agent.window_messages = 64  # so does a value in the file
    save(cfg)
    assert await window() == ("v3", 64)


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


async def test_the_result_config_names_the_layout_and_window_unless_earlier(
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
    assert (config["cache_layout"], config["window_messages"]) == ("v3", 64)
    args = parser.parse_args(
        ["--base-url", URL, "--cache-layout", "v2", "--window-messages", "20", "task"]
    )
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        config = agent._run_config()
    finally:
        await agent.close()
    assert "cache_layout" not in config and "window_messages" not in config
