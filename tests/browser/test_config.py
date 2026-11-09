"""``[browser] driver``: the default, the file, ``-X``, and which sessions get the tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts import work_folder
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
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
    config.save(cfg)
    assert config.load().browser.driver == "chrome" and cfg.setting_errors() == []
    cfg.browser.driver = "firefox"
    assert any("[browser] driver='firefox'" in e for e in cfg.setting_errors())
    assert any("[browser] driver='firefox'" in e for e in cfg.validate())


def test_x_sets_the_driver_for_one_run_and_refuses_another(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._parse(["-X", "browser.driver=safari-automation", "q"])
    assert config.overrides() == {"browser.driver": "safari-automation"}
    assert config.load().browser.driver == "safari-automation"
    config.set_overrides({})
    cli._parse(["-X", "browser.driver=chrome", "q"])
    assert config.load().browser.driver == "chrome"
    config.set_overrides({})
    with pytest.raises(SystemExit) as exc:
        cli._parse(["-X", "browser.driver=firefox", "q"])
    assert exc.value.code == 2
    assert (
        "is not one of ('off', 'safari-automation', 'chrome')"
        in capsys.readouterr().err
    )


async def test_a_session_gets_the_browser_tools_only_when_a_driver_is_named() -> None:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    names = {}
    for driver in ("off", "safari-automation", "chrome"):
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
    assert names["off"] == [] and len(names["safari-automation"]) == 13
    assert names["chrome"] == names["safari-automation"]
    with pytest.raises(ValueError, match="unknown browser driver"):
        await session_agents.build_session_agent(
            cfg=cfg,
            base_url=None,
            model=None,
            workspace=str(Path.cwd()),
            options=AgentOptions(),
            browser_driver="firefox",
        )


async def test_a_session_with_a_browser_reads_its_chats_work_folder_and_no_other() -> (
    None
):
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    roots = {}
    for driver, chat_id in (("off", "abc123"), ("chrome", "abc123"), ("chrome", None)):
        agent = await session_agents.build_session_agent(
            cfg=cfg,
            base_url=None,
            model=None,
            workspace=str(Path.cwd()),
            options=AgentOptions(mode="interactive"),
            browser_driver=driver,
            chat_id=chat_id,
        )
        try:
            roots[driver, chat_id] = agent.executor._read_roots
        finally:
            await agent.close()
    assert roots["off", "abc123"] == ()
    assert roots["chrome", "abc123"] == (work_folder.path_for("abc123"),)
    [own] = roots["chrome", None]
    assert os.path.basename(own).startswith("run-"), "a run with no chat has its own"
    assert not os.path.exists(own), "and leaves nothing when it ends"


async def test_the_terminal_passes_the_configured_driver_and_no_console_chat_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    async def fake_build(**kwargs: Any) -> Any:
        seen.append(kwargs.get("browser_driver", "off"))
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


class _NoTools(BaseToolProvider):
    async def start(self) -> list[ProvidedTool]:
        return []

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("driver", "count"),
    [("safari-automation", 13), ("chrome", 13), ("off", 0), ("firefox", 0)],
)
async def test_a_chat_and_its_tasks_have_the_browser_tools_the_setting_names(
    driver: str, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a chat and its task are built with; a driver that is not one is off."""
    from localmelo.support import browser
    from localmelo.support.browser import configured_driver

    asked_for: dict[str, Any] = {}
    real = browser.provider_for

    def provider_for(name: str, **options: Any) -> Any:
        asked_for.update(options)
        return real(name, **options)

    monkeypatch.setattr(browser, "provider_for", provider_for)
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    cfg.browser.driver = driver
    chat = await registry.default_builder(
        cfg, str(Path.cwd()), None, "main", tools=[_NoTools()]
    )
    work = await session_agents.build_session_agent(
        cfg=cfg,
        base_url=None,
        model=None,
        workspace=str(Path.cwd()),
        options=AgentOptions(),
        browser_driver=configured_driver(cfg),
        desktop=False,
    )
    try:
        names = [t.name for t in work.list_tools() if t.name.startswith("browser_")]
        assert len(names) == count
        assert names == [
            t.name for t in chat.list_tools() if t.name.startswith("browser_")
        ]
        # Other apps stay out of work nobody sits beside.
        assert not [t for t in work.list_tools() if t.name.startswith("window_")]
        if count:
            assert asked_for["app_windows"] is False
            assert asked_for["opens_apps"] is False
        tiers = {t.name: t.tier for t in work.list_tools() if t.name in names}
        acting = {
            "browser_click",
            "browser_type",
            "browser_select",
            "browser_click_point",
            "browser_attach",
        }
        # Reading is read tier; what acts is not, so a surface that trusts the
        # read tier does not get it.
        assert all(
            tier == ("write" if name in acting else "read")
            for name, tier in tiers.items()
        )
    finally:
        await chat.close()
        await work.close()


def test_direct_runs_never_name_a_driver() -> None:
    import argparse
    import inspect

    default = inspect.signature(session_agents.build_session_agent).parameters[
        "browser_driver"
    ]
    assert default.default == "off"
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
