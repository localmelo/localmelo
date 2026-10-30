"""``[browser] driver``: the default, the file, ``-X``, and which sessions get the tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config, settings
from localmelo.support.cli.agents import AgentFactory, url_backend
from localmelo.support.gateway.console import registry
from localmelo.support.sessions import agents as session_agents


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def test_browsing_is_off_by_default_and_a_default_config_says_nothing_of_it() -> None:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    assert cfg.browser.driver == "off" and cfg.setting_errors() == []
    config.save(cfg)
    assert "[browser]" not in Path(config.config_path()).read_text()
    key = settings.catalog()["browser.driver"]
    assert (key.default, key.file, key.section) == ("off", True, "browser")


def test_the_driver_is_saved_loaded_and_checked() -> None:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.browser.driver = "safari-automation"
    config.save(cfg)
    assert 'driver = "safari-automation"' in Path(config.config_path()).read_text()
    assert config.load().browser.driver == "safari-automation"
    cfg.browser.driver = "chrome"
    assert any("[browser] driver='chrome'" in e for e in cfg.setting_errors())
    assert any("[browser] driver='chrome'" in e for e in cfg.validate())


def test_x_sets_the_driver_for_one_run_and_refuses_another(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", "browser.driver=safari-automation", "q"])
    assert config.overrides() == {"browser.driver": "safari-automation"}
    assert config.load().browser.driver == "safari-automation"
    config.set_overrides({})
    with pytest.raises(SystemExit) as exc:
        cli._parse(["-X", "browser.driver=chrome", "q"])
    assert exc.value.code == 2
    assert "is not one of ('off', 'safari-automation')" in capsys.readouterr().err


async def test_a_session_gets_the_browser_tools_only_when_a_driver_is_named() -> None:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    names = {}
    for driver in ("off", "safari-automation"):
        agent = await session_agents.build_session_agent(
            cfg=cfg,
            base_url=None,
            model=None,
            workspace=str(Path.cwd()),
            options=AgentOptions(mode="interactive"),
            browser_driver=driver,
        )
        try:
            names[driver] = [
                t.name for t in agent.list_tools() if t.name.startswith("browser_")
            ]
        finally:
            await agent.close()
    assert names["off"] == [] and len(names["safari-automation"]) == 8
    with pytest.raises(ValueError, match="unknown browser driver"):
        await session_agents.build_session_agent(
            cfg=cfg,
            base_url=None,
            model=None,
            workspace=str(Path.cwd()),
            options=AgentOptions(),
            browser_driver="chrome",
        )


async def test_the_terminal_and_the_console_pass_the_configured_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    async def fake_build(**kwargs: Any) -> Any:
        seen.append(kwargs["browser_driver"])
        return object()

    monkeypatch.setattr("localmelo.support.cli.agents.build_session_agent", fake_build)
    monkeypatch.setattr(registry, "build_session_agent", fake_build)
    factory = AgentFactory(
        backend=url_backend("http://127.0.0.1:9/v1", "m", source="saved"),
        cfg=None,
        options=AgentOptions(),
        workspace="/w",
        browser_driver="safari-automation",
    )
    factory._live.clear()
    await factory.build()
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.browser.driver = "safari-automation"
    await registry.default_builder(cfg, "/w", None, "s1")
    assert seen == ["safari-automation", "safari-automation"]


def test_direct_runs_and_workers_never_name_a_driver() -> None:
    import argparse
    import inspect

    from localmelo.support.gateway import workers

    default = inspect.signature(session_agents.build_session_agent).parameters[
        "browser_driver"
    ]
    assert default.default == "off"
    assert "browser_driver" not in inspect.getsource(workers)
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.browser.driver = "safari-automation"
    config.save(cfg)
    args = argparse.Namespace(mcp_config=None)
    assert cli._direct_tool_providers(argparse.ArgumentParser(), args) == []


def test_the_safari_command_line_can_be_replaced_for_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.browser import provider_for
    from localmelo.support.browser.safari import SafariAutomationDriver

    def command(provider: Any) -> tuple[str, list[str]]:
        assert isinstance(provider.driver, SafariAutomationDriver)
        return provider.driver._config.command, provider.driver._config.args

    monkeypatch.setenv("LOCALMELO_SAFARIDRIVER", "python3 -I /x/fake.py")
    assert command(provider_for("safari-automation")) == (
        "python3",
        ["-I", "/x/fake.py", "--mcp"],
    )
    monkeypatch.delenv("LOCALMELO_SAFARIDRIVER")
    assert command(provider_for("safari-automation")) == (
        "/usr/bin/safaridriver",
        ["--mcp"],
    )
    assert provider_for("off") is None
