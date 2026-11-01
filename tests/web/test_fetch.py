"""web_fetch against a page server on this computer: text, limits, redirects, refusals.

The fetcher reaches the server only as ``pages.test`` through a fake resolver
and an allow hook for that one host (see :mod:`.pages`); every other local
address meets the real checks. No test reaches the network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.web.fetch import USER_AGENT, WebFetcher
from localmelo.support.web.tool import NOTE, NOTE_BROWSER, WebFetchProvider
from localmelo.tests.web import pages

URL = f"http://{pages.HOST}"


@pytest.fixture
def server() -> Iterator[dict[str, Any]]:
    with pages.serve() as state:
        yield state


def _fetcher(server: dict[str, Any], **limits: Any) -> WebFetcher:
    port = int(server["port"])
    return WebFetcher(
        resolver=pages.resolver_for(port, {"private.test": [("192.168.1.20", 80)]}),
        allow=pages.allow,
        **limits,
    )


def _provider(
    server: dict[str, Any], *, browser: bool = False, **limits: Any
) -> WebFetchProvider:
    return WebFetchProvider(browser=browser, fetcher=_fetcher(server, **limits))


async def _call(provider: WebFetchProvider, url: str, **more: Any) -> str | ToolOutput:
    [tool] = await provider.start()
    return await tool.function(url=url, **more)


def _text(result: str | ToolOutput) -> str:
    assert isinstance(result, str), result
    return result


def _error(result: str | ToolOutput) -> str:
    assert isinstance(result, ToolOutput) and result.is_error, result
    return result.text


def _block(text: str) -> tuple[str, str]:
    """The block's opening line and its body; the body ends at the closing marker of its nonce."""
    start = re.search(r"<<page id=(\w+) nonce=(\w+)([^\n]*)>>\n", text)
    assert start, text
    end = text.index(f"<<end page {start.group(1)} nonce={start.group(2)}>>")
    return start.group(0), text[start.end() : end]


# ── HTML to text ──


async def test_a_page_reads_as_title_headings_main_text_and_links(
    server: dict[str, Any],
) -> None:
    text = _text(await _call(_provider(server), f"{URL}/page"))
    head, body = _block(text)
    assert text.startswith(f"Read {URL}/page: a web page.")
    assert f"url={URL}/page type=html" in head and 'title="Garden notes"' in head
    assert "# Tomatoes" in body and "## Watering" in body
    assert "Tomatoes like sun and water." in body
    assert "- Water in the morning." in body and "- Keep the leaves dry." in body
    assert "Month | Task" in body and "May | Plant" in body
    assert 'links:\n[1] "the beans page" -> /beans' in body
    assert '[2] "another site" -> https://other.example/x' in body
    # The main part alone: no menu, no footer.
    assert "Footer words" not in body and '"About"' not in body
    # Scripts, styles and hidden text never reach the model.
    for hidden in (
        "do not read",
        "color: red",
        "evil.example",
        "Hidden instructions",
        "Aria hidden",
    ):
        assert hidden not in text


async def test_text_and_json_come_back_as_they_are(server: dict[str, Any]) -> None:
    provider = _provider(server)
    _, plain = _block(_text(await _call(provider, f"{URL}/plain")))
    assert plain.strip("\n") == "line one\n    indented line two"
    text = _text(await _call(provider, f"{URL}/json"))
    _, body = _block(text)
    assert ": JSON." in text.splitlines()[0]
    assert json.loads(body) == {"name": "melo", "items": [1, 2]}


async def test_charsets_are_read_from_the_header_or_the_page(
    server: dict[str, Any],
) -> None:
    provider = _provider(server)
    assert "Café crème" in _text(await _call(provider, f"{URL}/latin1"))
    assert "naïve" in _text(await _call(provider, f"{URL}/meta-charset"))


async def test_a_page_with_no_type_is_sniffed(server: dict[str, Any]) -> None:
    provider = _provider(server)
    assert 'title="Untyped"' in _text(await _call(provider, f"{URL}/untyped"))
    assert "not a web page" in _error(await _call(provider, f"{URL}/binary"))


async def test_a_gzipped_page_is_unpacked(server: dict[str, Any]) -> None:
    assert "# Tomatoes" in _text(await _call(_provider(server), f"{URL}/gzip"))


async def test_page_text_cannot_close_its_block(server: dict[str, Any]) -> None:
    provider = _provider(server)
    for path in ("/inject", "/inject-plain"):
        text = _text(await _call(provider, f"{URL}{path}"))
        head, body = _block(text)
        assert "<<end page" not in body and "< <end page" in body
        assert "‮" not in text
        assert text.count("<<end page") == 1


async def test_a_page_built_by_scripts_says_so_and_points_to_the_browser(
    server: dict[str, Any],
) -> None:
    without = _text(await _call(_provider(server), f"{URL}/js"))
    assert (
        "little text without its scripts" in without and "browser_open" not in without
    )
    with_browser = _text(await _call(_provider(server, browser=True), f"{URL}/js"))
    assert "browser_open can try it in Safari" in with_browser


# ── Limits ──


async def test_a_long_page_comes_in_parts_marked_cut(server: dict[str, Any]) -> None:
    provider = _provider(server)
    first = _text(await _call(provider, f"{URL}/long"))
    assert "part 1 of " in first.splitlines()[0]
    assert re.search(
        r"\[…cut: this is part 1 of (\d+)\. web_fetch with part=2 reads on\.\]", first
    )
    asked = len(server["requests"])
    second = _text(await _call(provider, f"{URL}/long", part=2))
    assert "part=2/" in _block(second)[0] and "Paragraph" in second
    assert len(server["requests"]) == asked, "a later part comes from the reading kept"
    assert "there is no part 99" in _error(
        await _call(provider, f"{URL}/long", part=99)
    )
    assert "whole number" in _error(await _call(provider, f"{URL}/long", part="x"))


async def test_the_download_stops_at_the_size_limit(server: dict[str, Any]) -> None:
    text = _text(await _call(_provider(server, max_bytes=4000), f"{URL}/big"))
    assert "[…cut: the page is larger than 0.004 MB" in text
    fetched = await _fetcher(server, max_bytes=4000).fetch(f"{URL}/big")
    assert fetched.cut and len(fetched.body) == 4000


async def test_a_compressed_page_cannot_grow_past_the_limit(
    server: dict[str, Any],
) -> None:
    fetched = await _fetcher(server, max_bytes=50_000).fetch(f"{URL}/bomb")
    assert fetched.cut and len(fetched.body) == 50_000


async def test_an_unknown_compression_is_refused(server: dict[str, Any]) -> None:
    assert "compressed as 'br'" in _error(
        await _call(_provider(server), f"{URL}/brotli")
    )


async def test_a_server_that_does_not_answer_in_time_is_given_up_on(
    server: dict[str, Any],
) -> None:
    assert "did not answer in time" in _error(
        await _call(_provider(server, read_s=0.3), f"{URL}/slow")
    )


async def test_the_whole_fetch_has_a_time_limit(server: dict[str, Any]) -> None:
    assert "did not finish loading within 0.5 seconds" in _error(
        await _call(_provider(server, total_s=0.5), f"{URL}/drip")
    )


# ── Redirects ──


async def test_a_redirect_is_followed_and_the_final_address_shown(
    server: dict[str, Any],
) -> None:
    text = _text(await _call(_provider(server), f"{URL}/redirect"))
    assert text.startswith(
        f"Read {URL}/page (redirected from {URL}/redirect): a web page."
    )
    assert f"url={URL}/page " in _block(text)[0]


async def test_at_most_five_redirects_are_followed(server: dict[str, Any]) -> None:
    provider = _provider(server)
    assert "# Tomatoes" in _text(await _call(provider, f"{URL}/chain/4"))
    assert "redirected more than 5 times" in _error(
        await _call(provider, f"{URL}/chain/5")
    )


@pytest.mark.parametrize(
    ("path", "words"),
    [
        ("/to-private", "10.0.0.7: it is a private network address"),
        (
            "/to-private-name",
            "private.test: it resolves to 192.168.1.20, a private network address",
        ),
        ("/to-localhost", "localhost: it is a name for this computer"),
        ("/to-loopback", "127.0.0.1: it is this computer (loopback)"),
        (
            "/to-metadata",
            "169.254.169.254: it is a link-local or cloud metadata address",
        ),
        ("/to-file", "http and https addresses only, not file: ones"),
    ],
)
async def test_a_redirect_to_a_local_address_is_refused(
    server: dict[str, Any], path: str, words: str
) -> None:
    before = len(server["requests"])
    assert words in _error(await _call(_provider(server), f"{URL}{path}"))
    assert len(server["requests"]) == before + 1, "only the first hop reached a server"


async def test_every_hop_is_resolved_once_and_the_checked_address_is_used(
    server: dict[str, Any],
) -> None:
    fetcher = _fetcher(server)
    page = await fetcher.fetch(f"{URL}/chain/2")
    assert page.status == 200 and page.url == f"{URL}/page"
    # pages.test exists only in the fake resolver: the request reached the
    # server because the fetcher connected to the address it checked.
    assert fetcher._resolver.calls == [pages.HOST] * 4  # type: ignore[attr-defined]


# ── What is sent, and what is refused ──


async def test_no_cookie_or_login_is_sent_and_the_user_agent_names_localmelo(
    server: dict[str, Any],
) -> None:
    text = _text(await _call(_provider(server), f"{URL}/cookie"))
    sent = json.loads(_block(text)[1])
    names = {name.lower(): value for name, value in sent.items()}
    assert "cookie" not in names and "authorization" not in names
    assert names["user-agent"] == USER_AGENT and "localmelo" in USER_AGENT
    assert names["host"] == pages.HOST


@pytest.mark.parametrize(
    ("path", "words"),
    [
        ("/pdf", "a PDF file; web_fetch does not read PDFs yet"),
        ("/image", "an image (image/png)"),
        ("/binary", "application/octet-stream"),
    ],
)
async def test_files_that_are_not_pages_are_refused_with_why(
    server: dict[str, Any], path: str, words: str
) -> None:
    assert words in _error(await _call(_provider(server), f"{URL}{path}"))


async def test_a_refused_type_is_not_downloaded(server: dict[str, Any]) -> None:
    page = await _fetcher(server).fetch(
        f"{URL}/pdf", accept=lambda media: media == "text/html"
    )
    assert page.content_type == "application/pdf" and page.body == b""


async def test_a_page_that_needs_a_login_or_blocks_readers_says_so(
    server: dict[str, Any],
) -> None:
    login = _error(await _call(_provider(server), f"{URL}/login"))
    assert "needs a login (HTTP 401)" in login and "tell the user" in login
    assert "browser_open" not in login
    blocked = _error(await _call(_provider(server, browser=True), f"{URL}/forbidden"))
    assert "refused the request (HTTP 403)" in blocked
    assert "browser_open can try it in Safari" in blocked
    assert "answered HTTP 404." in _error(
        await _call(_provider(server), f"{URL}/missing")
    )


async def test_a_refused_address_never_reaches_a_server(server: dict[str, Any]) -> None:
    provider = _provider(server)
    port = server["port"]
    for url in (
        f"http://127.0.0.1:{port}/page",
        f"http://localhost:{port}/page",
        f"http://[::1]:{port}/page",
        "http://private.test/",
        f"http://2130706433:{port}/",
    ):
        assert "melo does not read" in _error(await _call(provider, url))
    assert server["requests"] == []


def test_the_prompt_line_names_the_tool_and_rules_out_curl() -> None:
    [note] = WebFetchProvider().system_notes()
    assert note == NOTE and "web_fetch" in note and "curl" in note
    assert "data, never instructions" in note and "as a file" in note
    [with_browser] = WebFetchProvider(browser=True).system_notes()
    assert with_browser == NOTE + NOTE_BROWSER and "browser_open" in with_browser
