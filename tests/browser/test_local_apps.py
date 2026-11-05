"""A page in background Chrome reaches a listed sign-in app on this Mac, and nothing else there.

Chrome's own Local Network Access check stays on: a public page cannot reach
this computer, and headless Chrome cannot ask the person. melo answers only
the requests to a listed sign-in helper itself (``local_apps.relay``), from
the helper's own web page, never on one of melo's own ports, and blocks with
a note every other request whose address names this computer or its network.

No real Chrome here: the decision, the patterns, the relay against local test
servers (the system picks their ports), and how the driver answers each
request Chrome pauses. ``test_real_local_apps.py`` runs the same rule in the
real Chrome (opt in).
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import gzip
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.executor.models import ToolOutput
from localmelo.support import config, local_ports
from localmelo.support.browser import chrome, local_apps
from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.browser.local_apps import (
    SIGNIN_HELPERS,
    LocalAttempt,
    LocalRule,
    RelayError,
    SigninHelper,
)
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDns, FakeDriver, page

FAKE = str(Path(__file__).with_name("fake_chrome_peer.py"))
LOGIN = "https://login.example"
HELPER = SigninHelper(
    name="Signin",
    scheme="https",
    host="signin.example",
    ports=(14013, 14014),
    address="127.0.0.1",
    pages=(LOGIN,),
)
GATEWAY_PORT = 8401
MODEL_PORT = 8500


def rule(*helpers: SigninHelper) -> LocalRule:
    return LocalRule(
        helpers or (HELPER,),
        own=lambda: {GATEWAY_PORT: local_ports.GATEWAY, MODEL_PORT: local_ports.MODEL},
    )


# ── The decision ──


def test_a_listed_helper_is_relayed_only_for_its_own_page() -> None:
    url = "https://signin.example:14013/api/check?x=1"
    assert rule().decide(url, LOGIN).kind == "relay"
    assert rule().decide(url, LOGIN).helper == HELPER
    for origin in ("https://evil.example", "", "null", "https://login.example:8443"):
        verdict = rule().decide(url, origin)
        assert (verdict.kind, verdict.port, verdict.helper) == ("block", 14013, HELPER)


def test_only_the_helpers_scheme_host_and_ports_are_relayed() -> None:
    # Another scheme or port of the same name is no helper: Chrome's own check
    # stops it, since the name is not this computer's as written.
    assert rule().decide("http://signin.example:14013/", LOGIN).kind == "pass"
    assert rule().decide("https://signin.example:14099/", LOGIN).kind == "pass"
    assert rule().decide("https://signin.example/", LOGIN).kind == "pass"
    assert rule().decide("https://x.signin.example:14013/", LOGIN).kind == "pass"
    assert rule().decide("https://SIGNIN.example:14014/a", LOGIN).kind == "relay"


def test_melos_own_ports_are_never_relayed_even_when_listed() -> None:
    for port, what in ((GATEWAY_PORT, "gateway"), (MODEL_PORT, "model")):
        listed = SigninHelper(
            "Signin", "https", "signin.example", (port,), "127.0.0.1", (LOGIN,)
        )
        verdict = rule(listed).decide(f"https://signin.example:{port}/", LOGIN)
        assert (verdict.kind, verdict.port) == ("block", port), what


def test_a_helper_is_not_relayed_when_melos_own_ports_cannot_be_read() -> None:
    def broken() -> dict[int, str]:
        raise OSError("unreadable config")

    found = LocalRule((HELPER,), own=broken)
    assert found.decide("https://signin.example:14013/", LOGIN).kind == "block"


def test_the_default_own_ports_are_melos_gateway_and_model_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = config.Config()
    cfg.gateway.port = 14013
    config.save(cfg)
    listed = SigninHelper(
        "Signin", "https", "signin.example", (14013,), "127.0.0.1", (LOGIN,)
    )
    verdict = LocalRule((listed,)).decide("https://signin.example:14013/", LOGIN)
    assert (verdict.kind, verdict.port) == ("block", 14013)


@pytest.mark.parametrize(
    ("url", "port"),
    [
        ("http://127.0.0.1:8401/v1/agent/run", 8401),
        ("http://127.0.0.1:3000/", 3000),
        ("http://localhost:5173/", 5173),
        ("https://localhost/", 443),
        ("http://localhost.:8080/", 8080),
        ("http://app.localhost:9000/", 9000),
        ("http://[::1]:5000/", 5000),
        ("http://[::ffff:7f00:1]:9/", 9),
        ("http://0.0.0.0:8000/", 8000),
        ("http://10.0.0.5:8080/", 8080),
        ("http://172.16.0.1/", 80),
        ("http://192.168.1.1/admin", 80),
        ("http://169.254.169.254/latest/meta-data", 80),
        ("http://100.64.0.1/", 80),
        ("https://printer.local/", 443),
        ("http://nas.local:5000/", 5000),
        ("http://router.lan/", 80),
        ("http://box.home.arpa:8123/", 8123),
        ("http://127.0.0.2:1234/", 1234),
    ],
)
def test_an_address_of_this_computer_or_its_network_is_blocked(
    url: str, port: int
) -> None:
    verdict = rule().decide(url, "https://example.com")
    assert (verdict.kind, verdict.port) == ("block", port)
    assert verdict.helper is None


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "https://example.com/?next=http://127.0.0.1:8401/",
        "http://172.217.0.1/",
        "http://100.1.2.3/",
        "https://cdn.example.com/v10.2/app.js",
        "ws://127.0.0.1:8401/",  # Fetch never sees a WebSocket; Chrome's check does
        "data:text/plain,hi",
    ],
)
def test_a_public_address_goes_on(url: str) -> None:
    assert rule().decide(url, "https://example.com").kind == "pass"


def _chrome_glob(pattern: str) -> re.Pattern[str]:
    """Chrome's Fetch ``urlPattern``: ``*`` is any run, ``?`` one character, ``\\`` escapes."""
    out, escaped = [], False
    for char in pattern:
        if escaped:
            out.append(re.escape(char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "*":
            out.append(".*")
        elif char == "?":
            out.append(".")
        else:
            out.append(re.escape(char))
    return re.compile("".join(out), re.DOTALL)


def _paused(found: LocalRule, url: str, kind: str = "XHR") -> bool:
    return any(
        _chrome_glob(p["urlPattern"]).fullmatch(url)
        and p.get("resourceType", kind) == kind
        for p in found.patterns()
    )


def test_chrome_pauses_every_helper_address_and_every_local_one_written_as_such() -> (
    None
):
    found = rule()
    for port in HELPER.ports:
        assert _paused(found, f"https://signin.example:{port}/api/x?y=1")
    blocked = [
        "http://127.0.0.1:8401/v1/agent/run",
        "http://localhost:5173/",
        "https://localhost/",
        "http://localhost.:8080/",
        "http://app.localhost:9000/",
        "http://[::1]:5000/",
        "http://0.0.0.0:8000/",
        "http://10.0.0.5:8080/",
        "http://172.16.0.1/",
        "http://192.168.1.1/admin",
        "http://169.254.169.254/latest",
        "http://100.64.0.1/",
        "https://printer.local/",
        "http://nas.local:5000/",
        "http://router.lan/",
        "http://box.home.arpa:8123/",
    ]
    for url in blocked:
        assert _paused(found, url), url
    assert all(p.get("requestStage", "Request") == "Request" for p in found.patterns())
    # An ordinary public request is not paused at all.
    assert not _paused(found, "https://example.com/app.js")


def test_a_navigation_with_a_user_part_or_a_one_word_name_is_paused_and_blocked() -> (
    None
):
    # Chrome does not check where a page sends its tab; no glob sees these hosts.
    found = rule()
    for url, port in (
        ("http://x@127.0.0.1:8401/v1/agent/run", 8401),
        ("http://x@localhost:8401/", 8401),
        ("http://x@[::1]:8401/", 8401),
        ("http://router/admin", 80),
    ):
        assert _paused(found, url, "Document"), url
        verdict = found.decide(url, "https://example.com")
        assert (verdict.kind, verdict.port) == ("block", port), url


def test_the_listed_helpers_are_wechats_local_service_for_its_quick_login_page() -> (
    None
):
    (wechat,) = SIGNIN_HELPERS
    assert wechat.scheme == "https"
    assert wechat.host == "localhost.weixin.qq.com"
    assert set(wechat.ports) == {14013, 14014, 14015, 13013, 13014, 13015}
    assert wechat.address == "127.0.0.1"
    assert wechat.pages == ("https://open.weixin.qq.com",)
    assert wechat.waits == ("/api/authorize",)  # held open until the person answers
    for helper in SIGNIN_HELPERS:
        assert not set(helper.ports) & set(range(8400, 8800)), "melo's own range"


def test_the_sentence_says_which_port_melo_blocked() -> None:
    lines = local_apps.told(
        [
            LocalAttempt("127.0.0.1", 3000, "https://example.com", None),
            LocalAttempt("localhost", 3000, "https://example.com", None),
            LocalAttempt("192.168.1.1", 80, "https://example.com", None),
        ]
    )
    assert lines == [
        "The page tried to reach an app on this Mac (port 3000); melo blocked it.",
        "The page tried to reach an app on this Mac (port 80); melo blocked it.",
    ]


# ── The relay ──


class Helper(BaseHTTPRequestHandler):
    """A local sign-in helper: answers CORS like one, and records what reached it."""

    seen: list[dict[str, Any]] = []
    other_port = 0
    delay = (
        0.0  # how long a request to /wait takes, as an app that waits for the person
    )

    def log_message(self, *args: Any) -> None:
        pass

    def _answer(self) -> None:
        size = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(size) if size else b""
        Helper.seen.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
        )
        if self.path.startswith("/wait"):
            time.sleep(Helper.delay)
        if self.path.startswith("/redir"):
            self.send_response(302)
            self.send_header("location", f"http://127.0.0.1:{Helper.other_port}/x")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        if self.path.startswith("/big"):
            data = b"x" * (local_apps.MAX_ANSWER_BYTES + 1)
        elif self.path.startswith("/gzip"):
            data = gzip.compress(b"unzipped")
        else:
            data = json.dumps({"app": "signed-in user"}).encode()
        self.send_response(200)
        self.send_header(
            "access-control-allow-origin", self.headers.get("origin") or "*"
        )
        self.send_header("set-cookie", "a=1")
        self.send_header("set-cookie", "b=2")
        if self.path.startswith("/gzip"):
            self.send_header("content-encoding", "gzip")
        self.send_header("content-length", str(len(data)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._answer()

    def do_POST(self) -> None:  # noqa: N802
        self._answer()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._answer()


@pytest.fixture
def numbers(dns: FakeDns) -> FakeDns:
    """The fake lookup answers an address with itself, as the real one does."""
    dns.answers["127.0.0.1"] = ["127.0.0.1"]
    return dns


def _serve(context: ssl.SSLContext | None = None) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Helper)
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@pytest.fixture
def helper_server() -> Iterator[int]:
    Helper.seen = []
    server, port = _serve()
    other, Helper.other_port = _serve()
    try:
        yield port
    finally:
        for found in (server, other):
            found.shutdown()
            found.server_close()


def _plain(port: int) -> SigninHelper:
    return SigninHelper("Signin", "http", "signin.test", (port,), "127.0.0.1", (LOGIN,))


def _paused_request(url: str, method: str = "GET", body: bytes = b"") -> dict[str, Any]:
    request: dict[str, Any] = {
        "url": url,
        "method": method,
        "headers": {
            "Origin": LOGIN,
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Host": "evil.example",
            "User-Agent": "Mozilla/5.0",
            "Cookie": "session=secret",
        },
    }
    if body:
        request["hasPostData"] = True
        request["postDataEntries"] = [{"bytes": base64.b64encode(body).decode()}]
    return request


async def test_the_relay_sends_the_request_to_the_pinned_address_and_returns_the_answer(
    helper_server: int, numbers: FakeDns
) -> None:
    url = f"http://signin.test:{helper_server}/api/authorize?x=1"
    answer = await local_apps.relay(
        _plain(helper_server), helper_server, _paused_request(url, "POST", b'{"a":1}')
    )
    assert answer.status == 200
    assert json.loads(answer.body) == {"app": "signed-in user"}
    headers = [(k.lower(), v) for k, v in answer.headers]
    assert ("access-control-allow-origin", LOGIN) in headers
    assert "set-cookie" not in dict(headers)  # the app sets no cookie in melo's Chrome
    assert "connection" not in dict(headers) and "content-length" not in dict(headers)
    (seen,) = Helper.seen
    assert (seen["method"], seen["path"], seen["body"]) == (
        "POST",
        "/api/authorize?x=1",
        b'{"a":1}',
    )
    assert seen["headers"]["host"] == f"signin.test:{helper_server}"
    assert seen["headers"]["origin"] == LOGIN
    assert seen["headers"]["accept-encoding"] == "identity"
    assert seen["headers"]["user-agent"] == "Mozilla/5.0"
    assert "cookie" not in seen["headers"]
    # The name is never looked up: melo connects to the address the list pins.
    assert "signin.test" not in numbers.asked


async def test_a_request_that_waits_for_the_person_is_not_cut_at_the_usual_limit(
    helper_server: int, numbers: FakeDns, monkeypatch: pytest.MonkeyPatch
) -> None:
    # WeChat holds its authorize request open until the person answers its panel.
    monkeypatch.setattr(local_apps, "RELAY_S", 0.2)  # the usual limit, made short
    Helper.delay = 0.8
    helper = dataclasses.replace(_plain(helper_server), waits=("/wait",), wait_s=5.0)
    try:
        answer = await local_apps.relay(
            helper,
            helper_server,
            _paused_request(f"http://signin.test:{helper_server}/wait"),
        )
        assert answer.status == 200
        with pytest.raises(RelayError) as caught:  # any other request keeps the limit
            await local_apps.relay(
                dataclasses.replace(helper, waits=()),
                helper_server,
                _paused_request(f"http://signin.test:{helper_server}/wait"),
            )
        assert caught.value.reason == "TimedOut"
    finally:
        Helper.delay = 0.0


async def test_the_relay_hands_a_redirect_back_without_following_it(
    helper_server: int, numbers: FakeDns
) -> None:
    url = f"http://signin.test:{helper_server}/redir"
    answer = await local_apps.relay(
        _plain(helper_server), helper_server, _paused_request(url)
    )
    assert answer.status == 302
    assert dict((k.lower(), v) for k, v in answer.headers)["location"] == (
        f"http://127.0.0.1:{Helper.other_port}/x"
    )
    assert [s["path"] for s in Helper.seen] == ["/redir"]


async def test_the_relay_undoes_compression_and_refuses_an_answer_too_big(
    helper_server: int, numbers: FakeDns
) -> None:
    helper = _plain(helper_server)
    zipped = await local_apps.relay(
        helper,
        helper_server,
        _paused_request(f"http://signin.test:{helper_server}/gzip"),
    )
    assert zipped.body == b"unzipped"
    assert "content-encoding" not in {k.lower() for k, _ in zipped.headers}
    with pytest.raises(RelayError) as caught:
        await local_apps.relay(
            helper,
            helper_server,
            _paused_request(f"http://signin.test:{helper_server}/big"),
        )
    assert caught.value.reason == "Failed"


async def test_the_relay_fails_like_a_closed_app_when_nothing_listens(
    numbers: FakeDns,
) -> None:
    server, port = _serve()
    server.shutdown()
    server.server_close()
    with pytest.raises(RelayError) as caught:
        await local_apps.relay(
            _plain(port), port, _paused_request(f"http://signin.test:{port}/")
        )
    assert caught.value.reason == "ConnectionRefused"


async def test_the_relay_refuses_a_body_too_big_or_missing(numbers: FakeDns) -> None:
    big = _paused_request(
        "http://signin.test:1/", "POST", b"x" * (local_apps.MAX_BODY_BYTES + 1)
    )
    with pytest.raises(RelayError):
        await local_apps.relay(_plain(1), 1, big)
    missing = _paused_request("http://signin.test:1/", "POST")
    missing["hasPostData"] = True
    with pytest.raises(RelayError):
        await local_apps.relay(_plain(1), 1, missing)


OPENSSL = "/usr/bin/openssl"


@pytest.fixture
def tls(tmp_path: Path) -> dict[str, Path]:
    """A test CA and a certificate it signed for ``signin.test`` (made with the system's openssl)."""
    if not os.access(OPENSSL, os.X_OK):
        pytest.skip("no openssl")

    def run(*args: str) -> None:
        subprocess.run([OPENSSL, *args], check=True, capture_output=True, cwd=tmp_path)

    # Python checks strictly: the authority says it is one, the leaf what it is for.
    (tmp_path / "ca.cnf").write_text(
        "[req]\ndistinguished_name = dn\nx509_extensions = ca\nprompt = no\n"
        "[dn]\nCN = melo test CA\n"
        "[ca]\nbasicConstraints = critical, CA:TRUE\n"
        "keyUsage = critical, keyCertSign, cRLSign\nsubjectKeyIdentifier = hash\n"
    )
    (tmp_path / "leaf.cnf").write_text(
        "basicConstraints = CA:FALSE\n"
        "keyUsage = critical, digitalSignature, keyEncipherment\n"
        "extendedKeyUsage = serverAuth\nsubjectAltName = DNS:signin.test\n"
        "subjectKeyIdentifier = hash\nauthorityKeyIdentifier = keyid\n"
    )
    run(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "ca.key",
        "-out",
        "ca.pem",
        "-days",
        "2",
        "-config",
        "ca.cnf",
    )
    run(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "leaf.key",
        "-out",
        "leaf.csr",
        "-subj",
        "/CN=signin.test",
    )
    run(
        "x509",
        "-req",
        "-in",
        "leaf.csr",
        "-CA",
        "ca.pem",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "leaf.pem",
        "-days",
        "2",
        "-extfile",
        "leaf.cnf",
    )
    return {name: tmp_path / name for name in ("ca.pem", "leaf.pem", "leaf.key")}


async def test_the_relay_checks_the_certificate_for_the_helpers_name_not_its_address(
    tls: dict[str, Path], numbers: FakeDns
) -> None:
    Helper.seen = []
    served = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    served.load_cert_chain(tls["leaf.pem"], tls["leaf.key"])
    server, port = _serve(served)
    trusted = ssl.create_default_context(cafile=str(tls["ca.pem"]))
    try:
        good = SigninHelper(
            "Signin", "https", "signin.test", (port,), "127.0.0.1", (LOGIN,)
        )
        answer = await local_apps.relay(
            good,
            port,
            _paused_request(f"https://signin.test:{port}/api"),
            verify=trusted,
        )
        assert answer.status == 200
        assert Helper.seen[-1]["headers"]["host"] == f"signin.test:{port}"
        # The same server named otherwise: its certificate is not for that name.
        wrong = SigninHelper(
            "Signin", "https", "other.test", (port,), "127.0.0.1", (LOGIN,)
        )
        with pytest.raises(RelayError):
            await local_apps.relay(
                wrong,
                port,
                _paused_request(f"https://other.test:{port}/api"),
                verify=trusted,
            )
        # And a certificate no trusted authority signed is refused.
        with pytest.raises(RelayError):
            await local_apps.relay(
                good, port, _paused_request(f"https://signin.test:{port}/api")
            )
    finally:
        server.shutdown()
        server.server_close()


# ── How the driver answers what Chrome pauses ──


class Calls:
    """Stands in for Chrome's pipe: records each call and answers at once."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.made: list[tuple[str, dict[str, Any], str | None]] = []
        self.answers = answers or {}

    def __call__(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session: str | None = None,
    ) -> asyncio.Future[dict[str, Any]]:
        self.made.append((method, params or {}, session))
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        answer = self.answers.get(method, {})
        if isinstance(answer, Exception):
            future.set_exception(answer)
        else:
            future.set_result(answer)
        return future

    def answered(self) -> list[tuple[str, dict[str, Any]]]:
        return [(m, p) for m, p, _ in self.made if m.startswith("Fetch.")]


def _browser(found: LocalRule, calls: Calls) -> chrome._Chrome:
    browser = chrome._Chrome(-1, "p", asyncio.get_running_loop())
    browser.local = found
    browser.request = calls  # type: ignore[method-assign]
    browser.claimed = {"PAGE1", "PAGE2"}
    return browser


async def _pause(
    browser: chrome._Chrome,
    url: str,
    origin: str = "",
    frame: str = "PAGE1",
    ident: str = "r1",
) -> None:
    headers = {"Origin": origin} if origin else {}
    browser._dispatch(
        {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": ident,
                "frameId": frame,
                "resourceType": "XHR",
                "request": {"url": url, "method": "GET", "headers": headers},
            },
        }
    )
    while browser.jobs:
        await asyncio.gather(*list(browser.jobs), return_exceptions=True)


async def test_a_local_address_is_failed_before_it_is_sent_and_told_to_its_page() -> (
    None
):
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "http://127.0.0.1:3000/api", "https://example.com")
    assert calls.answered() == [
        ("Fetch.failRequest", {"requestId": "r1", "errorReason": "BlockedByClient"})
    ]
    assert browser.attempts == {
        "PAGE1": [LocalAttempt("127.0.0.1", 3000, "https://example.com", None)]
    }


async def test_a_public_address_goes_on_untouched() -> None:
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "https://example.com/?next=http://127.0.0.1/")
    assert calls.answered() == [("Fetch.continueRequest", {"requestId": "r1"})]
    assert browser.attempts == {}


async def test_a_helper_request_from_its_page_is_answered_with_the_helpers_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def relayed(
        helper: SigninHelper, port: int, request: dict[str, Any], **_: Any
    ) -> local_apps.Relayed:
        assert (helper, port, request["url"]) == (
            HELPER,
            14013,
            "https://signin.example:14013/x",
        )
        return local_apps.Relayed(
            200, "OK", [("access-control-allow-origin", LOGIN)], b"hi"
        )

    monkeypatch.setattr(local_apps, "relay", relayed)
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "https://signin.example:14013/x", LOGIN)
    assert calls.answered() == [
        (
            "Fetch.fulfillRequest",
            {
                "requestId": "r1",
                "responseCode": 200,
                "responsePhrase": "OK",
                "responseHeaders": [
                    {"name": "access-control-allow-origin", "value": LOGIN}
                ],
                "body": base64.b64encode(b"hi").decode(),
            },
        )
    ]


async def test_a_helper_that_does_not_answer_fails_the_request_as_chrome_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def closed(*_: Any, **__: Any) -> local_apps.Relayed:
        raise RelayError("ConnectionRefused")

    monkeypatch.setattr(local_apps, "relay", closed)
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "https://signin.example:14013/x", LOGIN)
    assert calls.answered() == [
        ("Fetch.failRequest", {"requestId": "r1", "errorReason": "ConnectionRefused"})
    ]


async def test_every_paused_request_gets_exactly_one_answer_even_when_melo_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*_: Any, **__: Any) -> local_apps.Relayed:
        raise RuntimeError("a bug")

    monkeypatch.setattr(local_apps, "relay", broken)
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "https://signin.example:14013/x", LOGIN)
    assert calls.answered() == [
        ("Fetch.failRequest", {"requestId": "r1", "errorReason": "Failed"})
    ]


async def test_an_answer_chrome_refuses_is_followed_by_one_failure() -> None:
    calls = Calls({"Fetch.continueRequest": chrome.CdpError("Invalid state")})
    browser = _browser(rule(), calls)
    await _pause(browser, "https://example.com/?u=http://127.0.0.1/")
    assert calls.answered() == [
        ("Fetch.continueRequest", {"requestId": "r1"}),
        ("Fetch.failRequest", {"requestId": "r1", "errorReason": "Failed"}),
    ]


async def test_the_helper_used_from_another_page_is_blocked_and_told() -> None:
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "https://signin.example:14013/x", "https://evil.example")
    assert calls.answered() == [
        ("Fetch.failRequest", {"requestId": "r1", "errorReason": "BlockedByClient"})
    ]
    assert browser.attempts["PAGE1"] == [
        LocalAttempt("signin.example", 14013, "https://evil.example", "Signin")
    ]


async def test_a_frame_inside_a_page_is_told_to_that_page() -> None:
    # An iframe from another site is a target of its own whose parent is the page;
    # Chrome lists it (Target.getTargets) but does not answer for its id alone.
    calls = Calls(
        {
            "Target.getTargets": {
                "targetInfos": [
                    {"targetId": "PAGE2", "type": "page"},
                    {"targetId": "FRAME9", "type": "iframe", "parentId": "PAGE2"},
                ]
            }
        }
    )
    browser = _browser(rule(), calls)
    await _pause(
        browser, "http://localhost:5173/", "https://ads.example", frame="FRAME9"
    )
    assert list(browser.attempts) == ["PAGE2"]


async def test_a_frame_in_the_pages_own_process_is_found_in_its_frames() -> None:
    calls = Calls(
        {
            "Page.getFrameTree": {
                "frameTree": {
                    "frame": {"id": "PAGE2"},
                    "childFrames": [{"frame": {"id": "SAME7"}}],
                }
            }
        }
    )
    browser = _browser(rule(), calls)
    browser.main = {"S2": "PAGE2"}
    await _pause(browser, "http://192.168.1.1/", "https://example.com", frame="SAME7")
    assert list(browser.attempts) == ["PAGE2"]


async def test_a_request_melo_cannot_place_is_still_blocked_and_told_to_no_one() -> (
    None
):
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "http://127.0.0.1:3000/", frame="WORKER")
    assert calls.answered() == [
        ("Fetch.failRequest", {"requestId": "r1", "errorReason": "BlockedByClient"})
    ]
    assert browser.attempts == {}


async def test_the_driver_takes_only_its_own_pages_attempts() -> None:
    calls = Calls()
    browser = _browser(rule(), calls)
    await _pause(browser, "http://127.0.0.1:3000/", frame="PAGE1", ident="r1")
    await _pause(browser, "http://127.0.0.1:4000/", frame="PAGE2", ident="r2")
    mine = ChromeBackgroundDriver(command=["unused"], local=rule())
    mine._browser = browser
    mine._mine = {"PAGE1"}
    assert [a.port for a in mine.local_attempts()] == [3000]
    assert mine.local_attempts() == []  # told once
    assert [a.port for a in browser.attempts["PAGE2"]] == [4000]


# ── Launch: Chrome's check stays on, and the rule is set up ──


def test_chromes_local_network_check_is_never_turned_off() -> None:
    args = " ".join(chrome.launch_args("/p", "156.0.8078.80"))
    assert "LocalNetworkAccess" not in args
    assert "PrivateNetworkAccess" not in args
    assert "--disable-web-security" not in args


class Peer:
    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "chrome.jsonl"
        self.profile = str(tmp_path / "profile")

    def lines(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


async def test_the_driver_asks_chrome_to_pause_the_rules_addresses_and_grants_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = Peer(tmp_path)
    monkeypatch.setenv("FAKE_CHROME_LOG", str(peer.path))
    monkeypatch.delenv("FAKE_CHROME_MODE", raising=False)
    found = rule()
    driver = ChromeBackgroundDriver(
        command=[sys.executable, FAKE],
        profile=peer.profile,
        version="156.0.8078.80",
        local=found,
    )
    await driver.start()
    try:
        await driver.open("https://example.com/", new_tab=True)
    finally:
        await driver.close()
    lines = peer.lines()
    enabled = [line for line in lines if line.get("method") == "Fetch.enable"]
    assert len(enabled) == 1
    assert enabled[0]["session"] == ""  # the whole browser: every tab, frame and worker
    assert enabled[0]["params"]["patterns"] == found.patterns()
    methods = {line.get("method") for line in lines}
    assert not methods & {"Browser.setPermission", "Browser.grantPermissions"}


async def test_a_driver_without_a_rule_pauses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = Peer(tmp_path)
    monkeypatch.setenv("FAKE_CHROME_LOG", str(peer.path))
    monkeypatch.delenv("FAKE_CHROME_MODE", raising=False)
    driver = ChromeBackgroundDriver(
        command=[sys.executable, FAKE], profile=peer.profile, version="156.0.8078.80"
    )
    await driver.start()
    await driver.close()
    assert "Fetch.enable" not in {line.get("method") for line in peer.lines()}


def test_melos_chrome_runs_with_the_sign_in_helpers_rule() -> None:
    from localmelo.support.browser import provider_for

    made = provider_for("chrome")
    assert made is not None
    driver = made.driver
    assert isinstance(driver, ChromeBackgroundDriver)
    assert driver.local is not None
    assert driver.local.helpers == SIGNIN_HELPERS


# ── The step's result ──


class Blocked(FakeDriver):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.waiting: list[LocalAttempt] = []

    def local_attempts(self) -> list[LocalAttempt]:
        found, self.waiting = self.waiting, []
        return found


async def test_the_step_says_the_page_tried_to_reach_an_app_on_this_mac() -> None:
    driver = Blocked(page("https://example.com/", "Welcome.", title="Home"))
    tools = {t.definition.name: t for t in await BrowserToolProvider(driver).start()}
    driver.waiting = [LocalAttempt("127.0.0.1", 3000, "https://example.com", None)]
    result = await tools["browser_open"].function(url="https://example.com/")
    text = result.text if isinstance(result, ToolOutput) else result
    assert (
        "The page tried to reach an app on this Mac (port 3000); melo blocked it."
        in text
    )
    again = await tools["browser_read"].function()
    text = again.text if isinstance(again, ToolOutput) else again
    assert "tried to reach an app" not in text  # told once
