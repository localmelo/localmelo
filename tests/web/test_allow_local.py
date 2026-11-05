"""``[tools] web_fetch_allow_local``: web_fetch may read listed test servers on this computer.

A developer setting, off by default, for a test that needs web_fetch to read
a fixture page on ``127.0.0.1``. It loosens a guard, so most of these tests
are about its limits:

- the list holds exact ``127.0.0.1:PORT`` and ``[::1]:PORT`` entries; every
  other entry is a config error, and is never used;
- melo's own ports (the gateway and console, a running gateway, the model
  server) are refused even when they are listed;
- a name is never allowed, even one that resolves to a listed address, and a
  redirect meets the same guard (an unlisted port, a public page sending melo
  to a local one);
- only web_fetch honours it, and each fetch it allows is marked.

Each refusal test also reads the listed server, so a setting that does
nothing fails it too. No test reaches the network: the servers are
``pages.serve()`` on loopback.
"""

from __future__ import annotations

import inspect
import urllib.parse
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import TOOL_ASK, ApprovalDecision
from localmelo.melo.contracts.events import EVENT_FIELDS, OPTIONAL_EVENT_FIELDS
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support import (
    config,
    config_cli,
    gateway_endpoint,
    settings,
    web_local,
)
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.support.web import guard
from localmelo.support.web.fetch import Response, WebFetcher
from localmelo.support.web.guard import AddressError, LocalAllow
from localmelo.support.web.tool import WebFetchProvider
from localmelo.tests.web import pages
from localmelo.tests.web.test_gate import Person

KEY = "tools.web_fetch_allow_local"
# The line a result carries when a listed test server was read.
NOTE = "[This address is a test server on this computer, which [tools] web_fetch_allow_local lists.]"


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "LOCALMELO_SAFARIDRIVER", "/nonexistent/not-the-real-safaridriver"
    )


@pytest.fixture
def server() -> Iterator[dict[str, Any]]:
    with pages.serve() as state:
        yield state


@pytest.fixture
def other() -> Iterator[dict[str, Any]]:
    with pages.serve() as state:
        yield state


def _listed(*servers: dict[str, Any]) -> list[str]:
    return [f"127.0.0.1:{state['port']}" for state in servers]


def _cfg(entries: list[str] | None = None) -> config.Config:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    cfg.vllm.chat_model = "m"
    if entries is not None:
        cfg.tools.web_fetch_allow_local = entries
    return cfg


def _provider(entries: list[str], cfg: config.Config | None = None) -> WebFetchProvider:
    cfg = cfg or _cfg()
    cfg.tools.web_fetch_allow_local = entries
    return WebFetchProvider(fetcher=web_local.fetcher(cfg))


class Asks:
    """The run's approval policy as a tool sees it: answers each site question, keeps them.

    *names* are the hosts the person's own words name (what ``names_host``
    answers); a listed test server asks even then.
    """

    def __init__(self, *, allow: bool = True, names: tuple[str, ...] = ()) -> None:
        self.allow = allow
        self.names = names
        self.asked: list[tuple[str, dict[str, Any], str]] = []

    async def __call__(
        self, name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        self.asked.append((name, dict(arguments), preview))
        return ApprovalDecision(self.allow, "" if self.allow else "not that one")

    def names_host(self, host: str) -> bool:
        return host in self.names


_APPROVES: Any = object()


async def _call(
    provider: WebFetchProvider, url: str, *, ask: Any = _APPROVES, **more: Any
) -> str | ToolOutput:
    """One web_fetch call under a policy that says yes (``ask=None``: no policy at all)."""
    [tool] = await provider.start()
    token = TOOL_ASK.set(Asks() if ask is _APPROVES else ask)
    try:
        return await tool.function(url=url, **more)
    finally:
        TOOL_ASK.reset(token)


def _text(result: str | ToolOutput) -> str:
    assert isinstance(result, str), result
    return result


def _error(result: str | ToolOutput) -> str:
    assert isinstance(result, ToolOutput) and result.is_error, result
    return result.text


def _url(state: dict[str, Any], path: str = "/page", host: str = "127.0.0.1") -> str:
    return f"http://{host}:{state['port']}{path}"


def _paths(state: dict[str, Any]) -> list[str]:
    return [r["path"] for r in state["requests"]]  # type: ignore[index]


async def _control(provider: WebFetchProvider, state: dict[str, Any]) -> None:
    """The listed server reads: the setting does something."""
    assert "# Tomatoes" in _text(await _call(provider, _url(state)))


# ── Off by default, and a listed port ──


async def test_off_by_default_a_loopback_fetch_is_refused(
    server: dict[str, Any],
) -> None:
    assert config.Config().tools.web_fetch_allow_local == []
    for provider in (WebFetchProvider(), _provider([])):
        refusal = _error(await _call(provider, _url(server)))
        assert "this computer (loopback)" in refusal
    assert server["requests"] == []


async def test_a_listed_port_is_read_and_marked(server: dict[str, Any]) -> None:
    provider = _provider(_listed(server))
    text = _text(await _call(provider, _url(server)))
    assert text.startswith(f"Read {_url(server)}: a web page.")
    assert "# Tomatoes" in text
    assert NOTE in text
    assert _paths(server) == ["/page"]


async def test_the_default_provider_reads_the_config_file(
    server: dict[str, Any],
) -> None:
    config.save(_cfg(_listed(server)))
    assert "# Tomatoes" in _text(await _call(WebFetchProvider(), _url(server)))
    config.save(_cfg())
    assert "loopback" in _error(await _call(WebFetchProvider(), _url(server)))


async def test_x_gives_the_default_provider_its_list(server: dict[str, Any]) -> None:
    config.set_overrides({KEY: _listed(server)})
    assert "# Tomatoes" in _text(await _call(WebFetchProvider(), _url(server)))


async def test_a_later_part_of_a_kept_page_is_marked_too(
    server: dict[str, Any],
) -> None:
    provider = _provider(_listed(server))
    first = _text(await _call(provider, _url(server, "/long")))
    again = _text(await _call(provider, _url(server, "/long"), part=2))
    assert "part 2 of" in again
    assert [NOTE in text for text in (first, again)] == [True, True]
    assert _paths(server) == ["/long"], "the second part came from the kept page"


async def test_an_ipv6_entry_is_allowed_by_the_guard() -> None:
    allow = LocalAllow({("::1", 8123)})
    assert guard.check_url("http://[::1]:8123/x", allow=allow).port == 8123
    with pytest.raises(AddressError):
        guard.check_url("http://[::1]:8124/x", allow=allow)
    with pytest.raises(AddressError):
        guard.check_url("http://127.0.0.1:8123/x", allow=allow)


# ── Refused even if listed ──


async def test_another_port_is_refused(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    provider = _provider(_listed(server))
    await _control(provider, server)
    assert "loopback" in _error(await _call(provider, _url(other)))
    assert other["requests"] == []


async def test_localhost_is_refused(server: dict[str, Any]) -> None:
    provider = _provider(_listed(server))
    await _control(provider, server)
    refusal = _error(await _call(provider, _url(server, host="localhost")))
    assert "name for this computer" in refusal
    assert _paths(server) == ["/page"], "only the control reached the server"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.2:{port}/page",
        "http://127.1:{port}/page",
        "http://2130706433:{port}/page",
        "http://0x7f.0.0.1:{port}/page",
        "http://[::ffff:127.0.0.1]:{port}/page",
        "http://[::1]:{port}/page",
        "http://0.0.0.0:{port}/page",
    ],
)
async def test_another_way_to_write_the_address_is_refused(
    server: dict[str, Any], url: str
) -> None:
    provider = _provider(_listed(server))
    await _control(provider, server)
    _error(await _call(provider, url.format(port=server["port"])))
    assert _paths(server) == ["/page"]


async def test_a_private_address_is_refused_even_when_it_is_listed() -> None:
    provider = _provider(["10.0.0.5:80", "192.168.1.9:80"])
    assert "private network" in _error(await _call(provider, "http://10.0.0.5:80/"))
    assert "private network" in _error(await _call(provider, "http://192.168.1.9/"))


async def test_a_name_is_refused_even_when_it_resolves_to_a_listed_address(
    server: dict[str, Any],
) -> None:
    port = int(server["port"])
    fetcher = WebFetcher(
        resolver=pages.resolver_for(port), allow=LocalAllow({("127.0.0.1", port)})
    )
    with pytest.raises(AddressError, match="127.0.0.1"):
        await fetcher.fetch(f"http://{pages.HOST}:{port}/page")
    assert server["requests"] == []


async def test_the_gateway_port_is_refused_even_when_listed(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    cfg = _cfg()
    cfg.gateway.port = int(server["port"])
    provider = _provider(_listed(server, other), cfg)
    assert "# Tomatoes" in _text(await _call(provider, _url(other)))
    assert "loopback" in _error(await _call(provider, _url(server)))
    assert server["requests"] == []


async def test_the_model_servers_port_is_refused_even_when_listed(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    cfg = _cfg()
    cfg.vllm.chat_url = f"http://127.0.0.1:{server['port']}/v1"
    provider = _provider(_listed(server, other), cfg)
    await _control(provider, other)
    assert "loopback" in _error(await _call(provider, _url(server)))
    assert server["requests"] == []


@pytest.mark.parametrize(
    "field", ["embedding_url", "chat_url"], ids=["embedding", "other backend"]
)
async def test_every_configured_model_url_counts(
    server: dict[str, Any], other: dict[str, Any], field: str
) -> None:
    cfg = _cfg()
    # A section that is not the chat backend still names a model server.
    setattr(cfg.ollama, field, f"http://localhost:{server['port']}/v1")
    provider = _provider(_listed(server, other), cfg)
    await _control(provider, other)
    assert "loopback" in _error(await _call(provider, _url(server)))
    assert server["requests"] == []


async def test_the_url_of_this_run_is_refused_even_when_listed(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    """``melo -u URL``: the run's own model server is not in the config."""
    cfg = _cfg(_listed(server, other))
    run_url = f"http://127.0.0.1:{server['port']}/v1"
    provider = WebFetchProvider(fetcher=web_local.fetcher(cfg, protect=[run_url, None]))
    await _control(provider, other)
    assert "loopback" in _error(await _call(provider, _url(server)))
    assert server["requests"] == []


async def test_a_session_agent_hands_its_model_urls_to_web_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``melo -u URL``, the config's servers: web_fetch is told of each, for its refusals."""
    from localmelo.support import web as web_package
    from localmelo.support.sessions.agents import build_session_agent

    made: list[dict[str, Any]] = []

    class Spy(web_package.WebFetchProvider):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(web_package, "WebFetchProvider", Spy)
    for cfg, base_url in ((None, "http://127.0.0.1:9100/v1"), (_cfg(), None)):
        agent = await build_session_agent(
            cfg=cfg,
            base_url=base_url,
            model=None,
            workspace=str(Path.cwd()),
            options=AgentOptions(mode="interactive"),
            embedding_base_url="http://127.0.0.1:9101/v1" if cfg is None else None,
            web_fetch=True,
        )
        await agent.close()
    [by_url, by_config] = [list(kwargs["protect"]) for kwargs in made]
    assert by_url == ["http://127.0.0.1:9100/v1", "http://127.0.0.1:9101/v1"]
    assert by_config[:2] == [None, None] and "http://127.0.0.1:9/v1" in by_config


async def test_a_running_gateways_port_is_refused_even_when_listed(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    provider = _provider(_listed(server, other))
    await _control(provider, other)
    assert "# Tomatoes" in _text(await _call(provider, _url(server)))
    # A gateway that listens on a port of its own (--port 0) writes it here.
    gateway_endpoint.write(
        gateway_endpoint.Endpoint(
            pid=1,
            host="127.0.0.1",
            port=int(server["port"]),
            instance="",
            id="x",
            started=0.0,
        )
    )
    assert "loopback" in _error(await _call(provider, _url(server)))


# ── Redirects meet the guard ──


async def test_a_listed_origin_redirecting_to_an_unlisted_local_port_is_refused(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    provider = _provider(_listed(server))
    await _control(provider, server)
    target = urllib.parse.quote(_url(other), safe="")
    refusal = _error(await _call(provider, _url(server, f"/to?u={target}")))
    assert "loopback" in refusal
    assert other["requests"] == []


@pytest.mark.parametrize("where", ["/to-localhost", "/to-private", "/to-metadata"])
async def test_a_listed_origin_redirecting_to_another_local_name_is_refused(
    server: dict[str, Any], where: str
) -> None:
    provider = _provider(_listed(server))
    await _control(provider, server)
    _error(await _call(provider, _url(server, where)))


async def test_a_listed_origin_may_redirect_within_itself(
    server: dict[str, Any],
) -> None:
    provider = _provider(_listed(server))
    text = _text(await _call(provider, _url(server, "/to-loopback")))
    assert "# Tomatoes" in text and "redirected from" in text
    assert _paths(server) == ["/to-loopback", "/page"]


async def test_a_public_page_cannot_send_melo_to_a_listed_local_server(
    server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = web_local.fetcher(_cfg(_listed(server)))
    real = fetcher._get

    async def get(target: Any, addresses: Any, accept: Any) -> Response:
        if target.host == "93.184.216.34":
            return Response(
                target.url, 302, "Found", "", "", b"", location=_url(server)
            )
        return await real(target, addresses, accept)  # type: ignore[no-any-return]

    monkeypatch.setattr(fetcher, "_get", get)
    with pytest.raises(AddressError, match="public"):
        await fetcher.fetch("http://93.184.216.34/start")
    assert server["requests"] == []
    # The same page read straight from the listed address is allowed.
    assert (await fetcher.fetch(_url(server))).status == 200


# ── Config errors, and nothing bad is ever used ──

# One entry for each way an entry can be wrong: a name, another address, a
# network, a range, a wildcard, no port, a port out of range, stray spaces.
BAD = [
    ("localhost:8000", "name"),
    ("10.0.0.5:80", "127.0.0.1"),
    ("127.0.0.2:80", "127.0.0.1"),
    ("127.0.0.0/8", "port"),
    ("127.0.0.1:8000-8100", "range"),
    ("127.0.0.1:*", "wildcard"),
    ("127.0.0.1", "port"),
    ("127.0.0.1:65536", "1 to 65535"),
    ("127.0.0.1:80 ", "spaces"),
]


@pytest.mark.parametrize(("entry", "why"), BAD)
def test_a_bad_entry_is_a_config_error_and_never_used(entry: str, why: str) -> None:
    cfg = _cfg(["127.0.0.1:8000", entry])
    [error] = [e for e in cfg.setting_errors() if "web_fetch_allow_local" in e]
    assert error.startswith("[tools] web_fetch_allow_local: ")
    assert why in error
    assert any("web_fetch_allow_local" in e for e in cfg.validate())
    # It is never used, but the good entry beside it is.
    assert web_local.sites_of(cfg.tools.web_fetch_allow_local) == {("127.0.0.1", 8000)}


@pytest.mark.parametrize("value", ["127.0.0.1:8000", 8000, [None]])
def test_a_list_of_text_is_required(value: Any) -> None:
    cfg = _cfg()
    cfg.tools.web_fetch_allow_local = value
    [error] = [e for e in cfg.setting_errors() if "web_fetch_allow_local" in e]
    assert "must be a list" in error or "must be text" in error
    assert web_local.sites_of(value) == frozenset()


def test_good_entries_are_accepted() -> None:
    entries = ["127.0.0.1:8000", "[::1]:9000", "127.0.0.1:1", "127.0.0.1:65535"]
    cfg = _cfg(entries)
    assert [e for e in cfg.setting_errors() if "web_fetch_allow_local" in e] == []
    assert web_local.sites_of(entries) == {
        ("127.0.0.1", 8000),
        ("::1", 9000),
        ("127.0.0.1", 1),
        ("127.0.0.1", 65535),
    }


def test_a_bad_file_is_reported_when_it_is_loaded(tmp_path: Path) -> None:
    path = Path(config.config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'chat_backend = "vllm"\nembedding_backend = "none"\n'
        '[vllm]\nchat_url = "http://127.0.0.1:9/v1"\nchat_model = "m"\n'
        '[tools]\nweb_fetch_allow_local = ["localhost:8000", "127.0.0.1:8001"]\n'
    )
    cfg = config.load()
    assert cfg.tools.web_fetch_allow_local == ["localhost:8000", "127.0.0.1:8001"]
    with pytest.raises(config.ConfigError, match=r"web_fetch_allow_local.*localhost"):
        cfg.validate_or_raise()
    path.write_text('[tools]\nweb_fetch_allow_local = "127.0.0.1:8001"\n')
    assert any("must be a list" in e for e in config.load().setting_errors())


def test_the_gateway_port_in_the_config_is_an_error_when_listed() -> None:
    cfg = _cfg(["127.0.0.1:8401", "127.0.0.1:9100"])
    [error] = [e for e in cfg.setting_errors() if "web_fetch_allow_local" in e]
    assert "127.0.0.1:8401" in error and "gateway" in error and "9100" not in error
    cfg.gateway.port = 9100
    [error] = [e for e in cfg.validate() if "web_fetch_allow_local" in e]
    assert "127.0.0.1:9100" in error and "gateway" in error and "8401" not in error


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("vllm", "chat_url"),
        ("ollama", "embedding_url"),
        ("openai", "base_url"),
    ],
)
def test_a_model_servers_port_in_the_config_is_an_error_when_listed(
    section: str, field: str
) -> None:
    cfg = _cfg(["127.0.0.1:8500", "127.0.0.1:9100"])
    setattr(getattr(cfg, section), field, "http://localhost:8500/v1")
    [error] = [e for e in cfg.setting_errors() if "web_fetch_allow_local" in e]
    assert "127.0.0.1:8500" in error and "model server" in error
    assert "9100" not in error


def test_the_default_ports_of_a_url_without_one_count() -> None:
    cfg = _cfg(["127.0.0.1:443", "127.0.0.1:80"])
    cfg.openai.base_url = "https://proxy.example/v1"
    errors = [e for e in cfg.setting_errors() if "web_fetch_allow_local" in e]
    assert len(errors) == 1 and "127.0.0.1:443" in errors[0]


# ── The key: file, -X, melo config ──


def test_the_key_is_a_developer_setting_that_is_off_by_default() -> None:
    key = settings.catalog()[KEY]
    assert (key.default, key.file, key.section, key.flag) == ([], True, "tools", "")
    config.save(_cfg())
    assert "web_fetch_allow_local" not in Path(config.config_path()).read_text()


def test_x_parses_a_comma_list_a_toml_list_and_nothing() -> None:
    cli._parse(["-X", f"{KEY}=127.0.0.1:8000,[::1]:9000", "q"])
    assert config.overrides() == {KEY: ["127.0.0.1:8000", "[::1]:9000"]}
    assert config.load().tools.web_fetch_allow_local == ["127.0.0.1:8000", "[::1]:9000"]
    cli._parse(["-X", f'{KEY}=["127.0.0.1:8001"]', "q"])
    assert config.overrides() == {KEY: ["127.0.0.1:8001"]}
    cli._parse(["-X", f"{KEY}=", "q"])
    assert config.overrides() == {KEY: []}
    cli._parse(["-X", f"{KEY}=127.0.0.1:8000", "q"])
    assert config.overrides() == {KEY: ["127.0.0.1:8000"]}


@pytest.mark.parametrize("value", ["localhost:8000", "127.0.0.1:8000,localhost:9"])
def test_x_refuses_a_bad_entry(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse(["-X", f"{KEY}={value}", "q"])
    assert exc.value.code == 2
    assert KEY in capsys.readouterr().err
    assert config.overrides() == {}


def test_x_refuses_the_default_gateway_port(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse(["-X", f"{KEY}=127.0.0.1:8401", "q"])
    assert exc.value.code == 2 and "gateway" in capsys.readouterr().err


def test_melo_config_set_checks_and_saves(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert config_cli.main(["set", KEY, "127.0.0.1:8000"]) == 0
    assert config.load().tools.web_fetch_allow_local == ["127.0.0.1:8000"]
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        config_cli.main(["set", KEY, "localhost:8000"])
    assert exc.value.code == 2
    assert config.load().tools.web_fetch_allow_local == ["127.0.0.1:8000"]


def test_melo_config_keys_lists_it_with_a_plain_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert config_cli.main(["keys", "web_fetch", "-X", f"{KEY}=127.0.0.1:8000"]) == 0
    out, err = capsys.readouterr()
    rows = {line.split()[0]: line.split() for line in out.splitlines()[1:]}
    assert rows[KEY][1:] == ['["127.0.0.1:8000"]', "[]", "-X"]
    assert "tools.web_fetch" in rows
    assert err.startswith("melo: warning: tools.web_fetch_allow_local")
    assert "developer" in err and "this computer" in err
    # A listing without the key says nothing about it.
    assert config_cli.main(["keys", "retrieval.k"]) == 0
    assert capsys.readouterr().err == ""


def test_the_full_listing_and_the_help_carry_the_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert config_cli.main(["keys"]) == 0
    out, err = capsys.readouterr()
    names = [line.split()[0] for line in out.splitlines()[1:]]
    assert names == list(settings.catalog()) and KEY in names
    assert err.count("melo: warning: ") == 1
    text = cli._build_parser(developer=True).format_help()
    assert any(line.startswith(f"  {KEY} ") for line in text.splitlines())


def test_the_console_lists_it_for_developers_only_and_cannot_set_it() -> None:
    from localmelo.support.gateway.console import routes_settings

    config.save(_cfg(["127.0.0.1:8000"]))
    body = routes_settings.settings_json()
    assert "web_fetch_allow_local" not in body
    rows = {
        row["key"]: row for group in body["developer_settings"] for row in group["keys"]
    }
    assert rows[KEY]["value"] == '["127.0.0.1:8000"]' and rows[KEY]["default"] == "[]"
    assert "web_fetch_allow_local" not in routes_settings.SettingsPutBody.model_fields
    with pytest.raises(ValidationError):
        routes_settings.SettingsPutBody(  # type: ignore[call-arg]
            chat_backend="vllm",
            embedding_backend="none",
            workspace="~/x",
            revision="r",
            web_fetch_allow_local=["127.0.0.1:1"],
        )


def test_the_model_is_told_of_the_exception_only_when_the_list_has_an_entry() -> None:
    from localmelo.support.web.tool import NOTE as PUBLIC_NOTE
    from localmelo.support.web.tool import NOTE_BROWSER

    for entries in ([], ["localhost:1"]):
        assert _provider(entries).system_notes() == (PUBLIC_NOTE,)
    [told] = _provider(["127.0.0.1:8123"]).system_notes()
    assert told == PUBLIC_NOTE + web_local.MODEL_NOTE
    assert "8123" not in told and "refused" in told
    with_browser = WebFetchProvider(
        browser=True, fetcher=web_local.fetcher(_cfg(["127.0.0.1:8123"]))
    )
    assert with_browser.system_notes() == (
        PUBLIC_NOTE + NOTE_BROWSER + web_local.MODEL_NOTE,
    )


def test_the_model_cannot_set_it() -> None:
    """The list is config only: web_fetch takes an address and a part, nothing else."""
    from localmelo.support.web.tool import TOOL

    assert set(TOOL.parameters["properties"]) == {"url", "part"}
    assert set(inspect.signature(WebFetchProvider).parameters) == {
        "browser",
        "task",
        "fetcher",
        "protect",
        "cfg",
    }


# ── Only web_fetch honours it ──


async def test_the_guard_without_a_hook_never_reads_the_setting(
    server: dict[str, Any],
) -> None:
    """The browser tools use the guard's own checks with no hook (#439)."""
    config.save(_cfg(_listed(server)))
    url = _url(server)
    with pytest.raises(AddressError, match="loopback"):
        guard.check_url(url)
    target = guard.check_url(url, allow=LocalAllow({("127.0.0.1", server["port"])}))
    with pytest.raises(AddressError, match="loopback"):
        await guard.checked_addresses(target, guard.system_resolver)
    assert server["requests"] == []


def test_the_guard_has_no_hook_unless_it_is_given_one() -> None:
    """No default, no module state: the browser's calls carry nothing local."""
    for function in (guard.check_url, guard.checked_addresses):
        assert inspect.signature(function).parameters["allow"].default is None
    assert WebFetcher()._allow is None
    assert WebFetchProvider(fetcher=WebFetcher()).fetcher._allow is None


def test_a_plain_hook_never_lets_a_written_address_through() -> None:
    """The hook of tests and screenshots answers for host names only, as before."""
    for url in ("http://127.0.0.1:8000/", "http://[::1]:8000/", "http://10.0.0.5/"):
        with pytest.raises(AddressError):
            guard.check_url(url, allow=lambda host, ip: True)


def test_nothing_under_the_browser_reads_the_setting() -> None:
    root = Path(inspect.getfile(config)).parent / "browser"
    sources = sorted(root.glob("*.py"))
    assert sources
    for path in sources:
        text = path.read_text("utf-8")
        for word in ("web_fetch_allow_local", "LocalAllow", "web_local"):
            assert word not in text, f"{path.name} mentions {word}"


def test_only_a_web_fetch_fetcher_gets_a_hook() -> None:
    cfg = _cfg(["127.0.0.1:8000"])
    fetcher = web_local.fetcher(cfg)
    assert isinstance(fetcher._allow, LocalAllow)
    assert web_local.fetcher(_cfg())._allow is None
    # A list that holds no valid entry is no hook at all.
    assert web_local.fetcher(_cfg(["localhost:1"]))._allow is None


# ── The mark: tool.finished and the step's details ──


class ScriptedLLM(BaseLLMProvider):
    def __init__(self, replies: list[Message]) -> None:
        self.replies = deque(replies)

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.replies:
            return self.replies.popleft()
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


def _fetch_message(url: str) -> Message:
    return Message(
        role="assistant",
        tool_call=ToolCall(tool_name="web_fetch", arguments={"url": url}),
    )


async def _finished(
    provider: WebFetchProvider,
    *urls: str,
    person: Person | None = None,
    query: str = "read the page",
) -> list[Any]:
    llm = ScriptedLLM([_fetch_message(url) for url in urls])
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), tool_exposure="all"),
        tool_providers=[provider],
    )
    found: list[Any] = []
    try:
        await agent.run_task(
            query,
            on_event=found.append,
            approval=RememberingPolicy((person or Person()).ask),
        )
    finally:
        await agent.close()
    return [event for event in found if event.type == "tool.finished"]


def test_the_event_may_carry_allowed_local() -> None:
    assert "allowed_local" in EVENT_FIELDS["tool.finished"]
    assert "allowed_local" in OPTIONAL_EVENT_FIELDS["tool.finished"]


async def test_a_fetch_it_allows_is_marked_in_tool_finished(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    provider = _provider(_listed(server))
    allowed, refused = await _finished(provider, _url(server), _url(other))
    assert allowed.data["status"] == "success" and allowed.data["allowed_local"] is True
    assert NOTE in allowed.data["output"]
    assert refused.data["status"] == "error" and "allowed_local" not in refused.data


async def test_an_allowed_fetch_that_fails_is_marked_too(
    server: dict[str, Any],
) -> None:
    provider = _provider(_listed(server))
    [finished] = await _finished(provider, _url(server, "/missing"))
    assert finished.data["status"] == "error"
    assert finished.data["allowed_local"] is True
    assert "answered HTTP 404" in finished.data["error"]
    assert NOTE in finished.data["error"]


async def test_a_public_or_unlisted_fetch_has_no_mark(
    server: dict[str, Any],
) -> None:
    provider = WebFetchProvider(
        fetcher=WebFetcher(
            resolver=pages.resolver_for(int(server["port"])), allow=pages.allow
        )
    )
    [finished] = await _finished(provider, f"http://{pages.HOST}/page")
    assert finished.data["status"] == "success"
    assert "allowed_local" not in finished.data
    assert "test server" not in finished.data["output"]


async def test_a_redirect_inside_a_listed_origin_marks_the_fetch(
    server: dict[str, Any],
) -> None:
    provider = _provider(_listed(server))
    [finished] = await _finished(provider, _url(server, "/redirect"))
    assert finished.data["allowed_local"] is True


# ── The live config: the port the gateway really listens on ──


async def test_the_live_gateways_port_is_refused_even_when_the_file_says_another(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    """``melo serve --port N`` (or ``--port 0``) changes the running config only.

    The file keeps its own gateway port, and the endpoint file is missing
    when another gateway holds the store or its write failed, so the session's
    config is the only thing that knows the port the console is on.
    """
    config.save(_cfg(_listed(server, other)))
    live = config.load()
    assert live.gateway.port not in (server["port"], other["port"])
    live.gateway.port = int(server["port"])  # what --port does, in memory
    assert gateway_endpoint.read() is None
    provider = WebFetchProvider(cfg=live)
    await _control(provider, other)
    assert "loopback" in _error(await _call(provider, _url(server)))
    assert _paths(server) == []
    # The file alone does not know it: a provider built from the file reads it.
    assert "# Tomatoes" in _text(await _call(WebFetchProvider(), _url(server)))


async def test_a_session_agent_hands_its_live_config_to_web_fetch(
    server: dict[str, Any], other: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support import web as web_package
    from localmelo.support.sessions.agents import build_session_agent

    config.save(_cfg(_listed(server, other)))
    live = config.load()
    live.gateway.port = int(server["port"])
    made: list[WebFetchProvider] = []

    class Spy(web_package.WebFetchProvider):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            made.append(self)

    monkeypatch.setattr(web_package, "WebFetchProvider", Spy)
    agent = await build_session_agent(
        cfg=live,
        base_url=None,
        model=None,
        workspace=str(Path.cwd()),
        options=AgentOptions(mode="interactive"),
        web_fetch=True,
    )
    await agent.close()
    [provider] = made
    await _control(provider, other)
    assert "loopback" in _error(await _call(provider, _url(server)))
    assert _paths(server) == []


async def test_a_provider_made_from_a_config_lists_what_that_config_lists(
    server: dict[str, Any],
) -> None:
    """The list is the session's own: not the file, which may have changed since."""
    live = _cfg(_listed(server))  # never saved
    assert "# Tomatoes" in _text(await _call(WebFetchProvider(cfg=live), _url(server)))
    config.save(_cfg(_listed(server)))
    assert "loopback" in _error(
        await _call(WebFetchProvider(cfg=_cfg([])), _url(server))
    )


# ── A listed server asks the person, whatever else is known ──


async def test_a_listed_server_asks_even_when_the_persons_words_name_it(
    server: dict[str, Any],
) -> None:
    """A page may steer the model to a listed port; naming 127.0.0.1 once is no answer."""
    person = Person()
    provider = _provider(_listed(server))
    url = _url(server)
    [finished] = await _finished(
        provider, url, person=person, query="read http://127.0.0.1:1/ for me"
    )
    assert finished.data["status"] == "success"
    [asked] = person.asked
    assert asked.tool_name == f"browser:http://127.0.0.1:{server['port']}"
    assert asked.arguments["via"] == "web_fetch" and asked.arguments["url"] == url
    assert _paths(server) == ["/page"]


async def test_the_shortcut_is_still_taken_for_a_public_site_the_person_named(
    server: dict[str, Any],
) -> None:
    """The control: a named public site reads without a question, a listed one does not."""
    person = Person()
    provider = WebFetchProvider(
        fetcher=WebFetcher(
            resolver=pages.resolver_for(int(server["port"])), allow=pages.allow
        )
    )
    [finished] = await _finished(
        provider,
        f"http://{pages.HOST}/page",
        person=person,
        query=f"read {pages.HOST}",
    )
    assert finished.data["status"] == "success" and person.asked == []


async def test_a_no_leaves_a_listed_server_unread(server: dict[str, Any]) -> None:
    ask = Asks(allow=False, names=("127.0.0.1",))
    provider = _provider(_listed(server))
    error = _error(await _call(provider, _url(server), ask=ask))
    assert "127.0.0.1" in error and "melo did not read" in error
    assert [name for name, _, _ in ask.asked] == [
        f"browser:http://127.0.0.1:{server['port']}"
    ]
    assert _paths(server) == []


async def test_a_yes_covers_only_that_server(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    ask = Asks(names=("127.0.0.1",))
    provider = _provider(_listed(server, other))
    await _call(provider, _url(server), ask=ask)
    await _call(provider, _url(other), ask=ask)
    assert [name for name, _, _ in ask.asked] == [
        f"browser:http://127.0.0.1:{server['port']}",
        f"browser:http://127.0.0.1:{other['port']}",
    ]


async def test_a_redirect_to_another_listed_server_asks_about_it(
    server: dict[str, Any], other: dict[str, Any]
) -> None:
    target = urllib.parse.quote(_url(other), safe="")
    ask = Asks()
    provider = _provider(_listed(server, other))
    text = _text(await _call(provider, _url(server, f"/to?u={target}"), ask=ask))
    assert "# Tomatoes" in text
    assert [name for name, _, _ in ask.asked] == [
        f"browser:http://127.0.0.1:{server['port']}",
        f"browser:http://127.0.0.1:{other['port']}",
    ]


@pytest.mark.parametrize("task", [False, True])
async def test_a_listed_server_is_read_when_nobody_can_be_asked(
    server: dict[str, Any], other: dict[str, Any], task: bool
) -> None:
    """No approval policy (--yolo, an unattended run): the developer listed it, so it is read.

    Only that: an unlisted port is still refused, and a task with no policy
    still reads no public site.
    """
    provider = WebFetchProvider(
        task=task, fetcher=web_local.fetcher(_cfg(_listed(server)))
    )
    assert "# Tomatoes" in _text(await _call(provider, _url(server), ask=None))
    assert _paths(server) == ["/page"]
    assert "loopback" in _error(await _call(provider, _url(other), ask=None))
    assert _paths(other) == []
    if task:
        public = f"http://{pages.HOST}/page"
        assert "nobody to ask" in _error(await _call(provider, public, ask=None))


def test_a_list_can_name_only_these_two_addresses() -> None:
    hosts = {host for host, _ in web_local.sites_of(["127.0.0.1:1", "[::1]:1"])}
    assert hosts == {"127.0.0.1", "::1"}
