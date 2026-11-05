"""``window_eviction`` at the configuration and CLI boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.support.config import AgentConfig, Config, load, save
from localmelo.support.sessions.agents import configured_session_options


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "localmelo.support.config.CONFIG_PATH", str(tmp_path / "config.toml")
    )
    return tmp_path


def test_the_config_roundtrip_writes_only_a_non_default_mode(config_home: Path) -> None:
    save(Config())
    assert "window_eviction" not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(window_eviction="chunk")))
    assert "window_eviction" not in (config_home / "config.toml").read_text()
    save(Config(agent=AgentConfig(window_eviction="boundary")))
    assert 'window_eviction = "boundary"' in (config_home / "config.toml").read_text()
    assert load().agent.window_eviction == "boundary"
    assert configured_session_options(load()).window_eviction == "boundary"
    assert (
        configured_session_options(
            load(),
            cli.AgentOptions(window_eviction="chunk"),
            explicit={"window_eviction"},
        ).window_eviction
        == "chunk"
    )


def test_the_removed_slide_in_a_file_reads_as_chunk(config_home: Path) -> None:
    (config_home / "config.toml").write_text('[agent]\nwindow_eviction = "slide"\n')
    assert load().agent.window_eviction == "chunk"
    assert not any("window_eviction" in e for e in load().validate())


def test_an_unknown_mode_in_the_config_is_refused() -> None:
    cfg = Config()
    cfg.agent.window_eviction = "sliding"
    assert any("window_eviction 'sliding'" in e for e in cfg.validate())


@pytest.mark.parametrize("mode", ["sliding", "slide"])
def test_the_flag_is_validated_by_the_parser(
    mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--window-eviction", mode, "t"])
    assert "choose from 'chunk', 'boundary'" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["chunk", "boundary"])
def test_the_flag_reaches_the_options(mode: str) -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        ["--base-url", "http://127.0.0.1:1/v1", "--window-eviction", mode, "t"]
    )
    assert cli._direct_options(parser, args).window_eviction == mode


def test_the_default_is_chunk() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", "http://127.0.0.1:1/v1", "t"])
    assert cli._direct_options(parser, args).window_eviction == "chunk"


async def test_direct_cli_uses_the_config_when_the_flag_is_absent(
    config_home: Path,
) -> None:
    save(Config(agent=AgentConfig(window_eviction="boundary")))
    parser = cli._build_parser()
    args = parser.parse_args(["--base-url", "http://127.0.0.1:1/v1", "task"])
    agent = cli._build_direct_mode_agent(
        args, parser=parser, options=cli._direct_options(parser, args)
    )
    try:
        assert agent.options.window_eviction == "boundary"
    finally:
        await agent.close()


async def test_direct_cli_flag_overrides_the_config(config_home: Path) -> None:
    save(Config(agent=AgentConfig(window_eviction="boundary")))
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--window-eviction",
            "chunk",
            "task",
        ]
    )
    options = cli._direct_options(parser, args)
    agent = cli._build_direct_mode_agent(args, parser=parser, options=options)
    try:
        assert agent.options.window_eviction == "chunk"
    finally:
        await agent.close()
