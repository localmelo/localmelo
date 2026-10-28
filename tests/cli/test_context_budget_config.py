"""Persistence and validation of the opt-in context budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import AgentConfig, Config, ConfigError, load, save


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
