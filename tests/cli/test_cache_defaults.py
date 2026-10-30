"""The cache settings' defaults at the config, flag, chat and gateway boundaries.

``[agent] fixed_prefix``, ``window_eviction`` and ``reflect_tool_guard``
default to ``true``, ``"chunk"`` and ``true``; ``cache_layout`` to ``"v3"`` and the
window to 64 messages on a local server (20 on a cloud vendor). A config file that
sets a key keeps it, a file without the key has the default, and a flag beats the
file. The layout and the window have their own file, ``test_cache_layout_config.py``;
here they join the opt-outs that give the earlier requests and the default sessions.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config
from localmelo.support.cli import launch
from localmelo.support.cli.agents import config_backend, url_backend
from localmelo.support.config import AgentConfig, Config, load, save
from localmelo.support.gateway.session import SessionManager
from localmelo.support.option_tiers import EXPERIMENTAL, OPTION_TIERS
from localmelo.support.sessions.agents import (
    configured_cache_options,
    configured_session_options,
)
from localmelo.tests.legacy_options import LEGACY_CACHE_OPTIONS

# name -> (default, opt-out value, the flag that sets the opt-out value)
SETTINGS: dict[str, tuple[Any, Any, tuple[str, str]]] = {
    "fixed_prefix": (True, False, ("--fixed-prefix", "off")),
    "window_eviction": ("chunk", "slide", ("--window-eviction", "slide")),
    "reflect_tool_guard": (True, False, ("--reflect-tool-guard", "off")),
}
OPT_OUT_FLAGS = [part for _, _, flag in SETTINGS.values() for part in flag]
# The layout and the window: the defaults of a local server, and their opt-outs.
LAYOUT_DEFAULTS = {"cache_layout": "v3", "window_messages": 64}
LAYOUT_OPT_OUTS = {"cache_layout": "v2", "window_messages": 20}
LAYOUT_OPT_OUT_FLAGS = ["--cache-layout", "v2", "--window-messages", "20"]
URL = "http://127.0.0.1:1/v1"


def _toml(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value).lower()


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


def _direct_options(*argv: str) -> tuple[AgentOptions, Any]:
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", URL, *argv, "task"])
    return cli._direct_options(parser, args), (parser, args)


async def _direct_agent_options(*argv: str) -> AgentOptions:
    options, (parser, args) = _direct_options(*argv)
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        return agent.options
    finally:
        await agent.close()


# ── The config file ──


@pytest.mark.parametrize("name", SETTINGS)
def test_an_absent_key_is_the_default_and_is_not_written(
    config_home: Path, name: str
) -> None:
    default, _, _ = SETTINGS[name]
    (config_home / "config.toml").write_text('[agent]\nreasoning = "legacy"\n')
    assert getattr(load().agent, name) == default
    save(load())
    assert name not in (config_home / "config.toml").read_text()


@pytest.mark.parametrize("name", SETTINGS)
def test_an_explicit_opt_out_is_kept_and_written_back(
    config_home: Path, name: str
) -> None:
    _, opt_out, _ = SETTINGS[name]
    (config_home / "config.toml").write_text(f"[agent]\n{name} = {_toml(opt_out)}\n")
    assert getattr(load().agent, name) == opt_out
    save(load())
    assert f"{name} = {_toml(opt_out)}" in (config_home / "config.toml").read_text()
    assert getattr(load().agent, name) == opt_out
    assert getattr(configured_session_options(load()), name) == opt_out


@pytest.mark.parametrize("name", SETTINGS)
def test_an_explicit_default_is_kept_but_not_written_back(
    config_home: Path, name: str
) -> None:
    default, _, _ = SETTINGS[name]
    (config_home / "config.toml").write_text(f"[agent]\n{name} = {_toml(default)}\n")
    assert getattr(load().agent, name) == default
    save(load())
    assert name not in (config_home / "config.toml").read_text()


def test_a_file_that_sets_one_key_has_the_defaults_for_the_others(
    config_home: Path,
) -> None:
    (config_home / "config.toml").write_text("[agent]\nfixed_prefix = false\n")
    agent = load().agent
    assert (agent.fixed_prefix, agent.window_eviction, agent.reflect_tool_guard) == (
        False,
        "chunk",
        True,
    )


def test_reflect_tool_guard_must_be_a_boolean() -> None:
    cfg = Config()
    cfg.agent.reflect_tool_guard = "on"  # type: ignore[assignment]
    assert any("reflect_tool_guard must be a boolean" in e for e in cfg.validate())


async def test_direct_cli_refuses_a_non_boolean_reflect_tool_guard(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (config_home / "config.toml").write_text('[agent]\nreflect_tool_guard = "on"\n')
    options, (parser, args) = _direct_options()
    with pytest.raises(SystemExit):
        cli._build_direct_mode_agent(args, parser=parser, options=options)
    assert "reflect_tool_guard must be a boolean" in capsys.readouterr().err


# ── Direct runs: config and flags ──


async def test_a_direct_run_without_a_config_file_has_the_defaults(
    config_home: Path,
) -> None:
    options = await _direct_agent_options()
    assert {name: getattr(options, name) for name in (*SETTINGS, *LAYOUT_DEFAULTS)} == {
        **{name: default for name, (default, _, _) in SETTINGS.items()},
        **LAYOUT_DEFAULTS,
    }


@pytest.mark.parametrize("name", SETTINGS)
async def test_a_direct_run_keeps_an_opt_out_of_the_config_file(
    config_home: Path, name: str
) -> None:
    default, opt_out, _ = SETTINGS[name]
    (config_home / "config.toml").write_text(f"[agent]\n{name} = {_toml(opt_out)}\n")
    options = await _direct_agent_options()
    assert getattr(options, name) == opt_out
    others = [other for other in SETTINGS if other != name]
    assert all(getattr(options, o) == SETTINGS[o][0] for o in others)


@pytest.mark.parametrize("name", SETTINGS)
async def test_a_flag_beats_the_config_file_in_both_directions(
    config_home: Path, name: str
) -> None:
    default, opt_out, (flag, off_value) = SETTINGS[name]
    on_value = {"off": "on", "slide": "chunk"}[off_value]
    (config_home / "config.toml").write_text(f"[agent]\n{name} = {_toml(opt_out)}\n")
    assert getattr(await _direct_agent_options(flag, on_value), name) == default
    (config_home / "config.toml").write_text(f"[agent]\n{name} = {_toml(default)}\n")
    assert getattr(await _direct_agent_options(flag, off_value), name) == opt_out


async def test_the_five_opt_out_flags_give_the_options_of_the_earlier_defaults(
    config_home: Path,
) -> None:
    options = await _direct_agent_options(*OPT_OUT_FLAGS, *LAYOUT_OPT_OUT_FLAGS)
    assert options == AgentOptions(**LEGACY_CACHE_OPTIONS)


async def test_the_five_opt_out_flags_beat_a_config_file_that_says_the_opposite(
    config_home: Path,
) -> None:
    save(
        Config(
            agent=AgentConfig(
                window_eviction="boundary", cache_layout="v3", window_messages=48
            )
        )
    )
    options = await _direct_agent_options(*OPT_OUT_FLAGS, *LAYOUT_OPT_OUT_FLAGS)
    assert options == AgentOptions(**LEGACY_CACHE_OPTIONS)


# ── Help: the flags are developer options and their help names the defaults ──


def test_default_help_hides_the_flags_and_developer_help_lists_them_as_experimental() -> (
    None
):
    shown = cli._build_parser(developer=False).format_help()
    developer = cli._build_parser(developer=True).format_help()
    for _, _, (flag, _) in SETTINGS.values():
        assert flag not in shown
        assert flag in developer
        assert OPTION_TIERS[flag] == EXPERIMENTAL


def test_the_help_of_each_flag_names_its_new_default() -> None:
    # The options' own help text (the full help lists settings, not this prose).
    text = " ".join(
        " ".join(str(a.help).split())
        for a in cli._build_parser(developer=True)._actions
    )
    for fragment in (
        "Use a versioned run-frozen system prefix and local date (default on)",
        "chunk eight messages at a time (default)",
        "that asks for the JSON object, not a tool call (default on)",
    ):
        assert fragment in text


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
    assert {name: getattr(options, name) for name in (*SETTINGS, *LAYOUT_DEFAULTS)} == {
        **{name: default for name, (default, _, _) in SETTINGS.items()},
        **LAYOUT_DEFAULTS,
    }


@pytest.mark.parametrize("from_config", [True, False])
def test_a_chat_session_keeps_the_opt_outs_of_the_config_file(
    config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    from_config: bool,
) -> None:
    cfg = _configured()
    cfg.agent.fixed_prefix = False
    cfg.agent.window_eviction = "slide"
    cfg.agent.reflect_tool_guard = False
    cfg.agent.cache_layout = "v2"
    cfg.agent.window_messages = 20
    save(cfg)
    options = _chat_options(cfg if from_config else None, monkeypatch, tmp_path)
    assert options == AgentOptions(**LEGACY_CACHE_OPTIONS)


@pytest.mark.parametrize("from_config", [True, False])
def test_the_opt_out_flags_reach_a_chat_session(
    config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    from_config: bool,
) -> None:
    cfg = _configured() if from_config else None
    options = _chat_options(
        cfg, monkeypatch, tmp_path, *OPT_OUT_FLAGS, *LAYOUT_OPT_OUT_FLAGS
    )
    assert options == AgentOptions(**LEGACY_CACHE_OPTIONS)


# ── Gateway sessions ──


def _gateway_options(cfg: Config) -> AgentOptions:
    with mock.patch("localmelo.support.gateway.session.Agent") as agent_cls:
        SessionManager(config=cfg)._create_agent()
    options: AgentOptions = agent_cls.call_args.kwargs["options"]
    return options


def test_a_gateway_session_starts_with_the_defaults() -> None:
    options = _gateway_options(_configured())
    assert {name: getattr(options, name) for name in (*SETTINGS, *LAYOUT_DEFAULTS)} == {
        **{name: default for name, (default, _, _) in SETTINGS.items()},
        **LAYOUT_DEFAULTS,
    }
    assert options.mode == "gateway" and options.record_answers


def test_a_gateway_session_on_a_cloud_backend_keeps_the_earlier_window() -> None:
    cfg = Config(chat_backend="openai")
    options = _gateway_options(cfg)
    assert (options.cache_layout, options.window_messages) == ("v3", 20)
    cfg.agent.window_messages = 64  # a value the file sets wins
    assert _gateway_options(cfg).window_messages == 64


def test_a_gateway_session_keeps_the_opt_outs_of_the_config_file() -> None:
    cfg = _configured()
    cfg.agent.fixed_prefix = False
    cfg.agent.window_eviction = "slide"
    cfg.agent.reflect_tool_guard = False
    cfg.agent.cache_layout = "v2"
    cfg.agent.window_messages = 20
    options = _gateway_options(cfg)
    assert options == AgentOptions(
        **LEGACY_CACHE_OPTIONS, mode="gateway", record_answers=True
    )


def test_the_console_and_the_workers_read_the_same_settings() -> None:
    cfg = _configured()
    cfg.agent.window_eviction = "boundary"
    gateway = AgentOptions(mode="gateway")
    assert configured_session_options(cfg, gateway).window_eviction == "boundary"
    assert configured_cache_options(cfg, gateway).window_eviction == "boundary"
    assert configured_cache_options(None, gateway) is gateway
    assert configured_cache_options(Config(), gateway) is gateway
    assert configured_cache_options(cfg, gateway, keep={"window_eviction"}) is gateway
