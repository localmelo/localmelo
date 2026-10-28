"""Persistence and validation of the opt-in context budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support.config import AgentConfig, Config, ConfigError, load, save
from localmelo.support.sessions.agents import configured_session_options


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    return path


def test_defaults_and_existing_agent_serialization(config_path: Path) -> None:
    cfg = Config()
    assert cfg.agent == AgentConfig()
    assert (
        cfg.agent.context_window,
        cfg.agent.output_reserve,
        cfg.agent.safety_margin,
    ) == (
        None,
        None,
        None,
    )
    save(cfg)
    assert "[agent]" not in config_path.read_text(encoding="utf-8")
    cfg.agent.reasoning = "drop"
    save(cfg)
    text = config_path.read_text(encoding="utf-8")
    assert '[agent]\nreasoning = "drop"\n' in text
    assert not any(
        name in text for name in ("context_window", "output_reserve", "safety_margin")
    )
    assert load().agent == cfg.agent


@pytest.mark.parametrize(
    "values",
    [
        {"context_window": 16384, "output_reserve": 2048, "safety_margin": 1024},
        {"context_window": 1},
        {"output_reserve": 300, "safety_margin": 200},
        {"context_window": 1, "output_reserve": 2},
    ],
)
def test_full_and_partial_budgets_round_trip(
    config_path: Path, values: dict[str, int]
) -> None:
    cfg = Config(agent=AgentConfig(**values))
    assert cfg._validate_agent() == []
    save(cfg)
    text = config_path.read_text(encoding="utf-8")
    for name in ("context_window", "output_reserve", "safety_margin"):
        assert (f"{name} = " in text) == (name in values)
    loaded = load()
    assert loaded.agent == cfg.agent
    save(loaded)
    assert config_path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("name", ["context_window", "output_reserve", "safety_margin"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "1", [], {}])
def test_invalid_individual_values(name: str, value: Any) -> None:
    cfg = Config()
    setattr(cfg.agent, name, value)
    errors = cfg.validate()
    assert any(f"[agent] {name} must be a positive integer" in e for e in errors)
    with pytest.raises(ConfigError, match=name):
        cfg.validate_or_raise()


@pytest.mark.parametrize("window", [2, 3, 4])
def test_combined_input_limit(window: int) -> None:
    cfg = Config(
        agent=AgentConfig(context_window=window, output_reserve=2, safety_margin=1)
    )
    errors = cfg._validate_agent()
    assert bool(errors) == (window <= 3)
    if errors:
        assert "context_window - output_reserve - safety_margin" in errors[0]


@pytest.mark.parametrize("legacy", [False, True])
def test_invalid_toml_value_is_loaded_for_validation(
    config_path: Path, legacy: bool
) -> None:
    prefix = 'backend = "ollama"\n' if legacy else 'chat_backend = "ollama"\n'
    config_path.write_text(
        prefix + '[agent]\ncontext_window = "123"\n', encoding="utf-8"
    )
    cfg = load()
    assert cfg.agent.context_window == "123"
    assert any("[agent] context_window" in e for e in cfg.validate())


async def test_base_url_uses_config_and_per_flag_override(config_path: Path) -> None:
    save(
        Config(
            agent=AgentConfig(context_window=100, output_reserve=80, safety_margin=30)
        )
    )
    parser = cli._build_parser()
    args = parser.parse_args(
        ["--base-url", "http://127.0.0.1:1/v1", "--output-reserve", "20", "task"]
    )
    agent = cli._build_direct_mode_agent(args, parser=parser, options=AgentOptions())
    try:
        assert (
            agent.options.context_window,
            agent.options.output_reserve,
            agent.options.safety_margin,
        ) == (100, 20, 30)
    finally:
        await agent.close()


def test_session_budget_overrides_invalid_config() -> None:
    cfg = Config(
        agent=AgentConfig(context_window=100, output_reserve=80, safety_margin=30)
    )
    opts = configured_session_options(cfg, budget_overrides={"output_reserve": 20})
    assert (opts.context_window, opts.output_reserve, opts.safety_margin) == (
        100,
        20,
        30,
    )
    assert cfg.agent.output_reserve == 80


def test_partial_budget_is_refused_before_session() -> None:
    cfg = Config(agent=AgentConfig(context_window=100))
    with pytest.raises(ValueError, match="positive integers"):
        configured_session_options(cfg)
