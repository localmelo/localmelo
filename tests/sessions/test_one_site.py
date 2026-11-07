"""The hosts of one site count as one site for a kept answer (#577).

A kept answer about opening a site, or clicking and choosing on it, covers the
site's other hosts where it holds: same scheme, same registrable domain, read
from the bundled public suffix list. Anything that cannot be read that way
stays its exact origin, and nothing else gets wider.
"""

from __future__ import annotations

from collections.abc import Callable

import publicsuffixlist
import pytest

from localmelo.melo.contracts import review
from localmelo.melo.contracts.approvals import (
    ALLOWED_ONCE,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    EarlierBlocks,
    revoke_site,
    site_covers,
    whole_site,
)
from localmelo.melo.contracts.review import ActionReview, host_site
from localmelo.support.browser import sites
from localmelo.support.cli.approvals import Approvals
from localmelo.support.sessions.approvals import (
    ApprovalLedger,
    ChatPolicy,
    OwnedTaskPolicy,
    RememberingPolicy,
    request_step,
)
from localmelo.support.sessions.grants import GrantBook

OPEN, ACT = "browser:", "browser-act:"
KEPT = "https://www.example.com"
OTHER_HOST = "https://login.example.com"
STOPPED = ActionReview("high", ("beyond_request",), ("You did not ask for this.",))


class Asker:
    """Says *answer* to every request and keeps what it was asked."""

    def __init__(self, answer: ApprovalDecision | None = None) -> None:
        self.answer = answer or ApprovalDecision(False, origin="user")
        self.requests: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.answer


def _question(
    name: str, review: ActionReview | None = None, **arguments: object
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r1",
        run_id="",
        call_id="c1",
        tool_name=name,
        arguments={"site": sites.host_of(name.split(":", 1)[1]), **arguments},
        preview="a site",
        workspace_root="/work",
        review=review,
    )


def _policy(asker: Asker, *kept: str) -> RememberingPolicy:
    policy = RememberingPolicy(asker)
    for name in kept:
        assert policy.grants.grant_now(name) is not None
    return policy


async def _ran_unasked(kept: str, asked: str) -> bool:
    asker = Asker()
    decision = await _policy(asker, kept).decide(_question(asked))
    assert decision.allow == (not asker.requests)
    return decision.allow


def _origin(url: str) -> str:
    origin = sites.origin_of(url)
    assert origin is not None
    return origin


@pytest.mark.parametrize(
    ("host", "site"),
    [
        ("www.example.com", "example.com"),
        ("example.com", "example.com"),
        ("a.b.c.example.com", "example.com"),
        ("WWW.Example.COM.", "example.com"),  # one spelling
        ("www.bücher.de", "xn--bcher-kva.de"),
        ("login.example.co.uk", "example.co.uk"),  # a multi-part country suffix
        ("a.b.example.co.jp", "example.co.jp"),
        ("alice.github.io", "alice.github.io"),  # shared hosting, private section
        ("blog.alice.github.io", "alice.github.io"),
        ("a.b.ck", "a.b.ck"),  # a wildcard rule
        ("foo.www.ck", "www.ck"),  # and its exception
        ("10.0.0.1", None),  # addresses by number
        ("0x7f.0.0.1", None),
        ("2130706433", None),
        ("::1", None),
        ("[2001:db8::1]", None),
        ("localhost", None),  # local and single-label names
        ("app.localhost", None),
        ("printer.local", None),
        ("nas.home.arpa", None),
        ("1.0.0.10.in-addr.arpa", None),
        ("intranet", None),
        ("com", None),  # a host that is itself a public suffix
        ("co.uk", None),
        ("github.io", None),
        ("wiki.corp.internal", None),  # an unlisted suffix
        ("shop.example", None),
        ("example.com:8443", None),  # not a plain name
        ("user@example.com", None),
        ("a..example.com", None),
        (".example.com", None),
        ("example.com..", None),
        ("", None),
    ],
)
def test_one_function_says_what_site_a_host_belongs_to(
    host: str, site: str | None
) -> None:
    assert host_site(host) == site


def test_the_list_is_parsed_once() -> None:
    review._suffix_list.cache_clear()
    host_site("www.example.com")
    host_site("login.example.org")
    assert review._suffix_list.cache_info().misses == 1


@pytest.mark.parametrize("error", [OSError("no list"), ImportError("no package")])
def test_a_list_nobody_can_read_is_tried_once_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    tried: list[int] = []

    def unreadable(**_: object) -> None:
        tried.append(1)
        raise error

    monkeypatch.setattr(publicsuffixlist, "PublicSuffixList", unreadable)
    review._suffix_list.cache_clear()
    try:
        for _ in range(3):
            assert host_site("www.example.com") is None
            assert whole_site(OPEN + KEPT) is None
            assert not site_covers(OPEN + KEPT, OPEN + OTHER_HOST)
        assert len(tried) == 1
    finally:
        review._suffix_list.cache_clear()


ONE_SITE = [
    (KEPT, OTHER_HOST),
    ("https://example.com", OTHER_HOST),  # the bare domain
    (OTHER_HOST, "https://example.com"),
    ("https://a.b.example.com", "https://c.example.com"),  # any depth
    ("http://www.example.com", "http://login.example.com"),  # http with http
    ("https://www.example.co.uk", "https://login.example.co.uk"),  # a country suffix
    ("https://blog.alice.github.io", "https://alice.github.io"),  # one shared account
    ("https://www.bücher.de", "https://login.xn--bcher-kva.de"),  # IDNA
    ("https://WWW.EXAMPLE.COM.", "https://login.example.com:443"),  # a spelling
]
ANOTHER_SITE = [
    (KEPT, "https://www.example.org"),
    (KEPT, "https://example.com.evil.org"),  # the name inside another's
    ("https://alice.github.io", "https://bob.github.io"),  # a shared hosting suffix
    ("https://example.co.uk", "https://other.co.uk"),  # a country suffix
    (KEPT, "http://www.example.com"),  # https never covers http
    (KEPT, "http://login.example.com"),
    ("http://www.example.com", OTHER_HOST),  # nor the reverse
    (KEPT, "https://login.example.com:8443"),  # a port not the scheme's own
    ("https://www.example.com:8443", OTHER_HOST),
    ("https://www.example.com:8443", "https://login.example.com:8443"),
    ("http://www.example.com:443", "http://login.example.com:443"),
    ("https://10.0.0.1", "https://10.0.0.2"),  # addresses by number
    ("https://1.2.3.4", "https://5.2.3.4"),
    ("https://[2001:db8::1]", "https://[2001:db8::2]"),
    ("http://localhost", "http://app.localhost"),  # local names
    ("http://a.printer.local", "http://b.printer.local"),
    ("http://a.nas.home.arpa", "http://b.nas.home.arpa"),
    ("http://1.0.0.10.in-addr.arpa", "http://2.0.0.10.in-addr.arpa"),
    ("http://intranet", "http://wiki"),  # single-label names
    ("https://github.io", "https://alice.github.io"),  # a host that is a suffix
    ("https://alice.github.io", "https://github.io"),
    ("https://co.uk", "https://example.co.uk"),
    ("https://a.corp.internal", "https://b.corp.internal"),  # an unlisted suffix
    ("https://a.shop.example", "https://b.shop.example"),
]


@pytest.mark.parametrize(
    ("kept", "asked"),
    [
        ("https://www.example.com..", OTHER_HOST),
        (KEPT, "https://login.example.com.."),
        ("https://www.example.com..", "https://login.example.com.."),
        ("https://www.example.com...", "https://login.example.com..."),
    ],
)
async def test_more_than_one_final_dot_is_read_as_no_site(
    kept: str, asked: str
) -> None:
    name, wanted = OPEN + _origin(kept), OPEN + _origin(asked)
    assert not site_covers(name, wanted)
    assert not await _ran_unasked(name, wanted)


@pytest.mark.parametrize(
    "key",
    [
        OPEN + "https://www.example.com.",  # one dot: an origin never keeps it
        OPEN + "https://www.example.com..",
        OPEN + "https://WWW.EXAMPLE.COM",  # nor capitals
    ],
)
async def test_a_name_no_origin_gives_is_read_as_no_site(key: str) -> None:
    assert whole_site(key) is None
    assert not await _ran_unasked(key, OPEN + OTHER_HOST)
    assert not await _ran_unasked(OPEN + OTHER_HOST, key)


@pytest.mark.parametrize(("kept", "asked"), ONE_SITE)
@pytest.mark.parametrize("prefix", [OPEN, ACT])
async def test_a_kept_answer_covers_the_other_hosts_of_its_site(
    kept: str, asked: str, prefix: str
) -> None:
    assert await _ran_unasked(prefix + _origin(kept), prefix + _origin(asked))


@pytest.mark.parametrize(("kept", "asked"), ANOTHER_SITE)
@pytest.mark.parametrize("prefix", [OPEN, ACT])
async def test_anything_not_read_as_one_site_stays_its_exact_origin(
    kept: str, asked: str, prefix: str
) -> None:
    assert not await _ran_unasked(prefix + _origin(kept), prefix + _origin(asked))
    assert await _ran_unasked(prefix + _origin(kept), prefix + _origin(kept))


async def test_the_kept_name_stays_as_it_was_given() -> None:
    policy = _policy(Asker(), OPEN + KEPT)
    await policy.decide(_question(OPEN + OTHER_HOST))
    assert policy.allowed_tools == {OPEN + KEPT}
    assert policy.silent_for(_question(OPEN + OTHER_HOST))
    # Taking the kept name back takes the whole site back.
    policy.forget(OPEN + KEPT)
    assert not policy.silent_for(_question(OPEN + OTHER_HOST))


def test_the_list_of_grants_names_what_each_one_covers() -> None:
    book = GrantBook()
    view = book.view("chat", "side")
    opened = view.grant_now(OPEN + OTHER_HOST)
    clicks = view.grant_now(ACT + OTHER_HOST)
    read = view.grant_now(OPEN + "https://www.example.org", by="auto")
    exact = view.grant_now(OPEN + "https://github.io")
    assert opened is not None and opened.label == "Open example.com"
    assert opened.target == "example.com" and opened.key == OPEN + OTHER_HOST
    assert clicks is not None and clicks.label == "Click and choose on example.com"
    assert read is not None and read.label == "Read www.example.org"
    assert exact is not None and exact.label == "Open github.io"


async def test_a_task_uses_its_chat_s_kept_answer_for_the_site() -> None:
    book = GrantBook()
    chat = book.view("chat", "side")
    assert chat.grant_now(OPEN + KEPT) is not None
    assert chat.grant_now(OPEN + "https://www.example.org", by="auto") is not None
    ledger = ApprovalLedger()
    ledger.grants_of = book.allowed_for
    ledger.site_kept = book.site_kept

    async def owner() -> str:
        return "chat"

    async def started(chat_id: str) -> float:
        return 1.0

    asker = Asker()
    task = OwnedTaskPolicy(
        RememberingPolicy(asker), ledger, "t1", owner, started=started
    )
    await task.review_window()

    assert (await task.decide(_question(OPEN + OTHER_HOST))).origin == "session"
    assert asker.requests == []
    # Auto's own read of the chat stays its exact origin in the task too.
    await task.decide(_question(OPEN + "https://login.example.org"))
    assert [r.tool_name for r in asker.requests] == [OPEN + "https://login.example.org"]


@pytest.mark.parametrize(
    "layers",
    [
        lambda policy: policy,
        lambda policy: EarlierBlocks(policy, {}),
        lambda policy: AutoPolicy(asking=policy),
        lambda policy: EarlierBlocks(AutoPolicy(asking=policy), {}),
    ],
)
def test_a_refusal_reaches_the_kept_answers_through_the_layers_of_a_policy(
    layers: Callable[[RememberingPolicy], object],
) -> None:
    policy = _policy(Asker(), OPEN + KEPT, ACT + "http://cdn.example.com")
    other = OPEN + "https://www.example.org"
    assert policy.grants.grant_now(other) is not None
    assert policy.grants.grant_now(OPEN + OTHER_HOST, by="auto") is not None
    revoke_site(layers(policy), "ask:site:example.com", "auto")
    assert OPEN + OTHER_HOST not in policy.allowed_tools, "only Auto's went"
    assert OPEN + KEPT in policy.allowed_tools
    revoke_site(layers(policy), "ask:site:example.com")
    assert policy.allowed_tools == {other}


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_a_refusal_reaches_a_chat_s_and_the_terminal_s_answers(mode: str) -> None:
    chat = ChatPolicy(Asker(), mode)  # type: ignore[arg-type]
    assert chat.grants.grant_now(OPEN + KEPT) is not None
    revoke_site(chat, "ask:site:example.com")
    assert chat.allowed_tools == frozenset()
    terminal = Approvals(Asker(), mode)  # type: ignore[arg-type]
    assert terminal._policy.grants.grant_now(OPEN + KEPT) is not None
    revoke_site(terminal.policy(), "ask:site:example.com")
    assert terminal.allowed_tools == frozenset()


async def test_a_kept_answer_stays_in_the_chat_it_holds_in() -> None:
    book = GrantBook()
    assert book.view("side", "side").grant_now(OPEN + KEPT) is not None
    asker = Asker()
    other = RememberingPolicy(asker, grants=book.view("other", "side"))
    assert not (await other.decide(_question(OPEN + OTHER_HOST))).allow
    assert len(asker.requests) == 1


@pytest.mark.parametrize(
    "prefix",
    [
        "browser-type:",
        "browser-submit:",
        "browser-attach:",
        "browser-point:",
        "browser-dialog:",
        "takeover:",
    ],
)
async def test_what_asks_every_time_is_never_covered(prefix: str) -> None:
    asker = Asker()
    policy = _policy(asker, OPEN + KEPT, ACT + KEPT)
    for origin in (KEPT, OTHER_HOST):
        assert not (await policy.decide(_question(prefix + origin))).allow
    assert len(asker.requests) == 2


@pytest.mark.parametrize(("kept", "asked"), [(OPEN, ACT), (ACT, OPEN)])
async def test_opening_and_clicking_stay_two_answers(kept: str, asked: str) -> None:
    assert not await _ran_unasked(kept + KEPT, asked + OTHER_HOST)


async def test_allow_once_keeps_nothing_for_the_site() -> None:
    asker = Asker(ApprovalDecision(True, origin="user"))
    policy = RememberingPolicy(asker)
    assert (await policy.decide(_question(OPEN + KEPT))).allow
    assert (await policy.decide(_question(OPEN + OTHER_HOST))).allow
    assert len(asker.requests) == 2 and policy.allowed_tools == frozenset()


@pytest.mark.parametrize(
    "review",
    [
        ActionReview("high", ("payment",), ("Pays.",)),
        ActionReview("high", ("beyond_request", "secret"), ("Sends a key.",)),
        ActionReview("high", ("beyond_request",), ("Stopped.",), doubts=("unsure",)),
        ActionReview("undecided", (), ("Could not tell.",)),
    ],
)
@pytest.mark.parametrize("name", [OPEN + KEPT, OPEN + OTHER_HOST, ACT + OTHER_HOST])
async def test_a_high_impact_step_is_never_covered(
    review: ActionReview, name: str
) -> None:
    asker = Asker()
    policy = _policy(asker, OPEN + KEPT, ACT + KEPT)
    assert not (await policy.decide(_question(name, review))).allow
    assert len(asker.requests) == 1


async def test_auto_mode_card_is_covered_on_the_other_hosts_too() -> None:
    asker = Asker()
    policy = ChatPolicy(asker, "auto")
    assert policy.ask.grants.grant_now(ACT + KEPT) is not None
    decided = await policy.decide(_question(ACT + OTHER_HOST, STOPPED))
    assert decided.allow and decided.origin == "session" and asker.requests == []


async def test_what_auto_keeps_stays_its_exact_origin_and_never_a_click() -> None:
    asker = Asker()
    policy = ChatPolicy(asker, "auto")
    routine = ActionReview("routine")
    assert (await policy.decide(_question(OPEN + KEPT, routine))).allow
    assert (await policy.decide(_question(ACT + KEPT, routine))).allow
    assert policy.allowed_tools == {OPEN + KEPT}, "a click is never kept by Auto"

    # Ask reads the same host on it, as before; nobody vouched for another host.
    policy.mode = "ask"
    assert (await policy.decide(_question(OPEN + KEPT))).allow
    assert not (await policy.decide(_question(OPEN + OTHER_HOST))).allow
    assert not (await policy.decide(_question(ACT + KEPT))).allow
    assert [r.tool_name for r in asker.requests] == [
        OPEN + OTHER_HOST,
        ACT + KEPT,
    ]


async def test_an_allowance_in_auto_stays_one_step() -> None:
    auto = AutoPolicy()
    step = _question(OPEN + KEPT, STOPPED)
    other = _question(OPEN + OTHER_HOST, STOPPED)
    auto.allow_once(request_step(step), ("beyond_request",))
    decided = await auto.decide(step)
    assert decided.allow and decided.reason == ALLOWED_ONCE
    assert not (await auto.decide(other)).allow
