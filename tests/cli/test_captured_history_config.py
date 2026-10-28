"""Captured history configuration stays opt-in."""

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support.config import AgentConfig, Config, load, save
from localmelo.support.sessions.agents import configured_session_options


def test_config_roundtrip_and_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    save(Config())
    assert "captured_history" not in path.read_text()
    save(Config(agent=AgentConfig(captured_history=True)))
    assert load().agent.captured_history is True
    assert configured_session_options(load()).captured_history
    assert not configured_session_options(
        load(), AgentOptions(), explicit={"captured_history"}
    ).captured_history
    with pytest.raises(ValueError, match="captured_history"):
        AgentOptions(captured_history="on")  # type: ignore[arg-type]


@pytest.mark.parametrize("choice", ["on", "off"])
async def test_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    save(Config(agent=AgentConfig(captured_history=True)))
    parser = cli._build_parser()
    args = parser.parse_args(
        ["--base-url", "http://127.0.0.1:1/v1", "--captured-history", choice, "task"]
    )
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert agent.options.captured_history is (choice == "on")
    finally:
        await agent.close()
