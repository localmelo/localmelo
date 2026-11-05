"""The window settings' defaults in chat and gateway sessions, and the help of the flag.

``[agent] window_eviction`` defaults to ``"chunk"`` and the window to 64 messages on a
local server (20 on a cloud vendor). A direct run's config and flags are in
``test_window_eviction_config.py`` and ``test_window_messages_config.py``; here are the
sessions that read the same settings and the flag's help.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config
from localmelo.support.cli import launch
from localmelo.support.cli.agents import config_backend, url_backend
from localmelo.support.config import Config, save
from localmelo.support.gateway.session import SessionManager
from localmelo.support.option_tiers import EXPERIMENTAL, OPTION_TIERS
from localmelo.support.sessions.agents import (
    configured_cache_options,
    configured_session_options,
)

DEFAULTS = {"window_eviction": "chunk", "window_messages": 64}
URL = "http://127.0.0.1:1/v1"


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    return tmp_path


def _configured() -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=config.LocalBackendConfig(chat_url=URL, chat_model="stub"),
    )


# ── Help: the flag is a developer option and its help names the default ──


def test_default_help_hides_the_flag_and_developer_help_lists_it_as_experimental() -> (
    None
):
    shown = cli._build_parser(developer=False).format_help()
    developer = cli._build_parser(developer=True).format_help()
    assert "--window-eviction" not in shown
    assert "--window-eviction" in developer
    assert OPTION_TIERS["--window-eviction"] == EXPERIMENTAL


def test_the_help_of_the_flag_names_its_default() -> None:
    # The options' own help text (the full help lists settings, not this prose).
    text = " ".join(
        " ".join(str(a.help).split())
        for a in cli._build_parser(developer=True)._actions
    )
    assert "chunk eight messages at a time (default)" in text


# ── Chat sessions ──


def _chat_options(
    cfg: Config | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str
) -> AgentOptions:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    parser = cli._build_parser()
    args = parser.parse_args(list(argv))
    backend = (
        config_backend(cfg, None)
        if cfg is not None
        else url_backend(URL, "stub", source="--base-url")
    )
    consumer = mock.AsyncMock(return_value=(0, None))
    # A --base-url session reads the config file; a configured one has its Config.
    reader = mock.patch.object(config, "load", return_value=cfg) if cfg else None
    with (
        mock.patch.object(launch, "ui_terminal", return_value=True),
        mock.patch.object(
            launch, "choose_backend", return_value=launch.Choice(backend, cfg)
        ),
        reader or contextlib.nullcontext(),
        mock.patch.object(launch, "configure_logging", return_value="test.log"),
        mock.patch.object(launch, "main_async", consumer),
    ):
        code = launch.run_interactive(
            args, parser=parser, options=cli._direct_options(parser, args)
        )
    assert code == 0
    options: AgentOptions = consumer.call_args.kwargs["options"]
    return options


@pytest.mark.parametrize("from_config", [True, False])
def test_a_chat_session_starts_with_the_defaults(
    config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    from_config: bool,
) -> None:
    cfg = _configured() if from_config else None
    options = _chat_options(cfg, monkeypatch, tmp_path)
    assert {name: getattr(options, name) for name in DEFAULTS} == DEFAULTS


@pytest.mark.parametrize("from_config", [True, False])
def test_a_chat_session_keeps_the_settings_of_the_config_file(
    config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    from_config: bool,
) -> None:
    cfg = _configured()
    cfg.agent.window_eviction = "boundary"
    cfg.agent.window_messages = 20
    save(cfg)
    options = _chat_options(cfg if from_config else None, monkeypatch, tmp_path)
    assert (options.window_eviction, options.window_messages) == ("boundary", 20)


# ── Gateway sessions ──


def _gateway_options(cfg: Config) -> AgentOptions:
    with mock.patch("localmelo.support.gateway.session.Agent") as agent_cls:
        SessionManager(config=cfg)._create_agent()
    options: AgentOptions = agent_cls.call_args.kwargs["options"]
    return options


def test_a_gateway_session_starts_with_the_defaults() -> None:
    options = _gateway_options(_configured())
    assert {name: getattr(options, name) for name in DEFAULTS} == DEFAULTS
    assert options.mode == "gateway" and options.record_answers


def test_a_gateway_session_on_a_cloud_backend_keeps_the_earlier_window() -> None:
    cfg = Config(chat_backend="openai")
    assert _gateway_options(cfg).window_messages == 20
    cfg.agent.window_messages = 64  # a value the file sets wins
    assert _gateway_options(cfg).window_messages == 64


def test_the_console_and_the_workers_read_the_same_settings() -> None:
    cfg = _configured()
    cfg.agent.window_eviction = "boundary"
    gateway = AgentOptions(mode="gateway")
    assert configured_session_options(cfg, gateway).window_eviction == "boundary"
    assert configured_cache_options(cfg, gateway).window_eviction == "boundary"
    assert configured_cache_options(None, gateway) is gateway
    assert configured_cache_options(Config(), gateway) is gateway
    assert configured_cache_options(cfg, gateway, keep={"window_eviction"}) is gateway
