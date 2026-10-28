"""Config names opt in to tool environment inheritance, never persist values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.support import config


def configured() -> config.Config:
    return config.Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=config.LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1", chat_model="stub"
        ),
    )


async def test_tool_env_config_round_trip_and_builder_inheritance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.sessions.agents import build_session_agent

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MELO_TEST_TOOL_VAR", "synthetic-visible-value")
    monkeypatch.delenv("MELO_TEST_MISSING_VAR", raising=False)
    cfg = configured()
    cfg.agent.tool_env = ["MELO_TEST_TOOL_VAR", "MELO_TEST_MISSING_VAR"]
    cfg.validate_or_raise()
    config.save(cfg)
    saved = Path(config.config_path()).read_text()
    assert 'tool_env = ["MELO_TEST_TOOL_VAR", "MELO_TEST_MISSING_VAR"]' in saved
    assert "synthetic-visible-value" not in saved
    loaded = config.load()
    agent = await build_session_agent(
        cfg=loaded, base_url=None, model=None, workspace=str(tmp_path)
    )
    try:
        assert agent.options.tool_env == {
            "MELO_TEST_TOOL_VAR": "synthetic-visible-value"
        }
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "names",
    [
        "SSH_AUTH_SOCK",
        [1],
        [""],
        ["A=B"],
        ["A\x00B"],
        ["has space"],
        ["1BAD"],
        ["bad-name"],
        ["变量"],
    ],
)
def test_tool_env_names_are_validated_before_environment_access(names: Any) -> None:
    cfg = configured()
    cfg.agent.tool_env = names
    with pytest.raises(config.ConfigError, match="tool_env"):
        cfg.validate_or_raise()


def test_default_config_does_not_add_tool_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MELO_TEST_TOOL_VAR", "must-not-be-inherited")
    assert configured().tool_environment() == {}
