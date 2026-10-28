"""Fixed-prefix option at configuration and CLI boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support.config import AgentConfig, Config, load, save
from localmelo.support.sessions.agents import configured_session_options


def test_fixed_prefix_config_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    save(Config())
    assert "fixed_prefix" not in path.read_text()
    save(Config(agent=AgentConfig(fixed_prefix=True)))
    assert "fixed_prefix = true" in path.read_text()
    assert load().agent.fixed_prefix is True
    assert configured_session_options(load()).fixed_prefix is True
    assert not configured_session_options(
        load(), AgentOptions(fixed_prefix=False), explicit={"fixed_prefix"}
    ).fixed_prefix


def test_invalid_fixed_prefix_config() -> None:
    cfg = Config()
    cfg.agent.fixed_prefix = "on"  # type: ignore[assignment]
    assert any("fixed_prefix must be a boolean" in e for e in cfg.validate())
    with pytest.raises(ValueError, match="fixed_prefix"):
        AgentOptions(fixed_prefix="on")  # type: ignore[arg-type]


@pytest.mark.parametrize("choice", ["on", "off"])
async def test_direct_cli_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    save(Config(agent=AgentConfig(fixed_prefix=True)))
    parser = cli._build_parser()
    args = parser.parse_args(
        ["--base-url", "http://127.0.0.1:1/v1", "--fixed-prefix", choice, "task"]
    )
    options = cli._direct_options(parser, args)
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.fixed_prefix is (choice == "on")
    finally:
        await agent.close()
