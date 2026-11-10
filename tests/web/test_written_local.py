"""A server on this computer whose address the person wrote is read; no other (#547).

The person's own words say it, through the run's question policy
(``written_sites``); a page, a tool result or the model's own address never
does. The address must be exactly the host and port they wrote, melo's own
ports stay refused, and a public page cannot send melo there. Each refusal
also reads the written server in the same test, so a feature that does
nothing fails it too. Both tools: web_fetch (a real loopback server) and the
browser (a scripted one).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import pytest

from localmelo.support import config
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.web.fetch import Response, WebFetcher
from localmelo.support.web.tool import WRITTEN_NOTE, WebFetchProvider
from localmelo.tests.browser.fakes import FakeDriver, page
from localmelo.tests.browser.test_acting import Asks as BrowserAsks
from localmelo.tests.browser.test_acting import asking
from localmelo.tests.browser.test_local_addresses import Anywhere
from localmelo.tests.browser.test_provider import run, tools_of
from localmelo.tests.web.test_allow_local import (  # noqa: F401
    Asks,
    _call,
    _error,
    _home,
    _text,
    _url,
    other,
    server,
)


class Wrote(Asks):
    """A run whose person wrote these ``(address, port)`` servers in their own words."""

    def __init__(self, *sites: tuple[str, int]) -> None:
        super().__init__()
        self.sites = set(sites)

    def written_sites(self) -> frozenset[tuple[str, int]]:
        return frozenset(self.sites)


class WroteForBrowser(BrowserAsks):
    def __init__(self, *sites: tuple[str, int]) -> None:
        super().__init__()
        self.sites = set(sites)

    def written_sites(self) -> frozenset[tuple[str, int]]:
        return frozenset(self.sites)


# ── web_fetch ──


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
async def test_web_fetch_reads_the_server_the_person_wrote_with_no_question(
    server: dict[str, Any], host: str  # noqa: F811
) -> None:
    ask = Wrote(("127.0.0.1", int(server["port"])))
    text = _text(await _call(WebFetchProvider(), _url(server, host=host), ask=ask))
    assert "# Tomatoes" in text and WRITTEN_NOTE in text
    assert ask.asked == [], "the person wrote it: no card"


async def test_web_fetch_reads_nothing_on_this_computer_that_the_person_did_not_write(
    server: dict[str, Any], other: dict[str, Any]  # noqa: F811
) -> None:
    wrote = Wrote(("127.0.0.1", int(server["port"])))
    control = _text(await _call(WebFetchProvider(), _url(server), ask=wrote))
    assert "# Tomatoes" in control
    refused = {
        "another port the person did not write": (_url(other), wrote),
        "a policy that cannot say": (_url(server), Asks()),
        "no policy at all": (_url(server), None),
        "a policy that says no": (_url(server), Wrote()),
        "a neighbour of the same port": (
            f"http://10.0.0.5:{server['port']}/page",
            Wrote(("10.0.0.5", int(server["port"]))),
        ),
    }
    for case, (url, ask) in refused.items():
        before = len(server["requests"]) + len(other["requests"])
        error = _error(await _call(WebFetchProvider(), url, ask=ask))
        assert "this computer" in error or "private" in error or "public" in error, case
        assert len(server["requests"]) + len(other["requests"]) == before, case


async def test_melos_own_ports_stay_refused_even_when_the_person_wrote_them(
    server: dict[str, Any],  # noqa: F811
) -> None:
    cfg = config.load()
    cfg.gateway.port = int(server["port"])
    provider = WebFetchProvider(cfg=cfg)
    ask = Wrote(("127.0.0.1", int(server["port"])))
    assert "loopback" in _error(await _call(provider, _url(server), ask=ask))
    assert server["requests"] == []


async def test_a_public_page_cannot_send_web_fetch_to_the_written_server(
    server: dict[str, Any], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    real = WebFetcher._get

    async def get(
        self: WebFetcher, target: Any, addresses: Any, accept: Any
    ) -> Response:
        if target.host == "93.184.216.34":
            return Response(
                target.url, 302, "Found", "", "", b"", location=_url(server)
            )
        return await real(self, target, addresses, accept)  # type: ignore[no-any-return]

    monkeypatch.setattr(WebFetcher, "_get", get)
    ask = Wrote(("127.0.0.1", int(server["port"])))
    out = _error(await _call(WebFetchProvider(), "http://93.184.216.34/x", ask=ask))
    assert "public" in out and server["requests"] == []


# ── the browser ──

DEV = "http://localhost:3000/"


def local_world() -> FakeDriver:
    return Anywhere(
        page(DEV, "Dev server.", title="Dev"),
        page("http://127.0.0.1:3000/", "Dev server.", title="Dev"),
    )


@pytest.mark.parametrize("url", [DEV, "http://127.0.0.1:3000/", DEV + "a?b=1"])
async def test_the_browser_opens_the_server_the_person_wrote(url: str) -> None:
    driver = local_world()
    tools = await tools_of(driver)
    with asking(WroteForBrowser(("127.0.0.1", 3000))):
        out = await run(tools, "browser_open", url=url)
    assert not out.is_error, out.text
    assert any(call == "open" for call, _ in driver.calls)


@pytest.mark.parametrize(
    "case",
    [
        "another port",
        "no port",
        "a neighbour",
        "a policy that cannot say",
        "a policy that says no",
        "melo's own port",
    ],
)
async def test_the_browser_still_refuses_every_other_local_address(case: str) -> None:
    gateway = config.load().gateway.port
    url, ask = {
        "another port": (
            "http://localhost:3001/",
            WroteForBrowser(("127.0.0.1", 3000)),
        ),
        "no port": ("http://localhost/", WroteForBrowser(("127.0.0.1", 3000))),
        "a neighbour": (
            "http://10.0.0.8:3000/",
            WroteForBrowser(("127.0.0.1", 3000), ("10.0.0.8", 3000)),
        ),
        "a policy that cannot say": (DEV, BrowserAsks()),
        "a policy that says no": (DEV, WroteForBrowser()),
        "melo's own port": (
            f"http://localhost:{gateway}/",
            WroteForBrowser(("127.0.0.1", gateway)),
        ),
    }[case]
    driver = local_world()
    tools = await tools_of(driver)
    with asking(ask):
        out = await run(tools, "browser_open", url=url)
    assert out.is_error and "does not open local or private addresses" in out.text
    assert not any(call == "open" for call, _ in driver.calls), urlsplit(url)


class Recording(Anywhere):
    """A browser that lets a local address through only when told: it keeps what it was told."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allowed: list[frozenset[tuple[str, int]]] = []

    def allow_local(self, grant: Any, listed: Any = None) -> None:
        """Keeps what the reader says when told (``None``: nothing is allowed)."""
        self.allowed.append(frozenset() if grant is None else frozenset(grant()))


async def test_each_browser_step_tells_the_driver_which_servers_the_person_wrote() -> (
    None
):
    driver = Recording(page(DEV, "Dev server.", title="Dev"))
    tools = await tools_of(driver)
    with asking(WroteForBrowser(("127.0.0.1", 3000))):
        await run(tools, "browser_open", url=DEV)
    with asking(WroteForBrowser()):  # the next message wrote none
        await run(tools, "browser_read")
    with asking(None):  # and a run with no policy never writes one
        await run(tools, "browser_read")
    with asking(BrowserAsks()):  # nor a policy that cannot say
        await run(tools, "browser_read")
    assert driver.allowed == [frozenset({("127.0.0.1", 3000)})] + [frozenset()] * 3


async def test_forgetting_the_chat_takes_back_what_the_open_tabs_may_reach() -> None:
    """A kept tab outlives its chat's approvals; it keeps no server on this computer (#547)."""
    driver = Recording(page(DEV, "Dev server.", title="Dev"))
    provider = BrowserToolProvider(driver)
    tools = {tool.definition.name: tool for tool in await provider.start()}
    with asking(WroteForBrowser(("127.0.0.1", 3000))):
        await run(tools, "browser_open", url=DEV)
    assert driver.allowed == [frozenset({("127.0.0.1", 3000)})]
    provider.forget()
    assert driver.allowed[1:] == [frozenset()]


async def test_a_web_search_reads_with_no_local_address_even_when_the_person_wrote_one() -> (
    None
):
    search = "https://93.184.216.34/search?q=tomatoes"
    driver = Recording(page(search, "Results.", title="Results"))
    provider = BrowserToolProvider(driver)
    await provider.start()
    with asking(WroteForBrowser(("127.0.0.1", 3000))):
        await provider.read_for_search(search)
    assert driver.allowed == [frozenset()]
