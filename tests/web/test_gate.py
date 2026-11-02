"""web_fetch's site gate: which sites read without a question, and which ask first.

A host the person's recent messages name (itself, ``www.`` aside) reads
without a question in every mode; a subdomain of it does not. Any other site
asks once per chat through the browser's site question
(``browser:<origin>``, carrying the address), so one "allow for this chat"
covers both tools. The gate runs before the name is looked up, on the first
address and on a redirect to another site. A task with nobody to ask is
refused.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._step_review import StepReviewer
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    EarlierBlocks,
)
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.browser.sites import URL_SHOWN
from localmelo.support.sessions.approvals import ChatPolicy, RememberingPolicy
from localmelo.support.web.fetch import WebFetcher
from localmelo.support.web.tool import WebFetchProvider
from localmelo.tests.web import pages

PAGE = f"http://{pages.HOST}/page"
OTHER = "other.test"


class ScriptedLLM(BaseLLMProvider):
    def __init__(self, replies: list[Message]) -> None:
        self.replies = deque(replies)
        self.calls: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append(list(messages))
        return self.replies.popleft() if self.replies else _say("done")

    async def close(self) -> None:
        pass


def _fetch(url: str) -> Message:
    return Message(
        role="assistant",
        tool_call=ToolCall(tool_name="web_fetch", arguments={"url": url}),
    )


def _say(text: str) -> Message:
    return Message(role="assistant", content=text)


class Person:
    """Answers each site question; allows for the chat unless told to refuse."""

    def __init__(self, *, refuse: bool = False, remember: bool = True) -> None:
        self.refuse = refuse
        self.remember = remember
        self.asked: list[ApprovalRequest] = []

    async def ask(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        if self.refuse:
            return ApprovalDecision(False, "not that one")
        return ApprovalDecision(True, remember=self.remember)


@pytest.fixture
def server() -> Iterator[dict[str, Any]]:
    with pages.serve() as state:
        yield state


def _resolver(server: dict[str, Any]) -> Any:
    return pages.resolver_for(
        int(server["port"]), {OTHER: [("127.0.0.1", int(server["port"]))]}
    )


def _allow(host: str, ip: Any) -> bool:
    return host in (pages.HOST, OTHER) and str(ip) == "127.0.0.1"


def _agent(llm: ScriptedLLM, resolver: Any, **provider: Any) -> Agent:
    fetcher = WebFetcher(resolver=resolver, allow=_allow)
    return Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), tool_exposure="all"),
        tool_providers=[WebFetchProvider(fetcher=fetcher, **provider)],
    )


def _results(llm: ScriptedLLM) -> list[str]:
    return [m.content for m in llm.calls[-1] if m.role == "tool"]


async def test_a_site_the_person_named_reads_without_a_question(
    server: dict[str, Any],
) -> None:
    person = Person()
    llm = ScriptedLLM([_fetch(PAGE), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task(
            "What does pages.test say about tomatoes?",
            approval=RememberingPolicy(person.ask),
        )
    finally:
        await agent.close()
    assert person.asked == []
    assert "# Tomatoes" in _results(llm)[0]


async def test_a_subdomain_of_a_named_site_asks_once(server: dict[str, Any]) -> None:
    # Naming a shared domain (vercel.app, workers.dev) must not name every
    # site under it, nor a subdomain that carries data in its name.
    person = Person()
    port = int(server["port"])
    sub = "docs.pages.test"
    resolver = pages.resolver_for(port, {sub: [("127.0.0.1", port)]})

    def allow(host: str, ip: Any) -> bool:
        return host == sub and str(ip) == "127.0.0.1"

    llm = ScriptedLLM(
        [_fetch(f"http://{sub}/page"), _fetch(f"http://{sub}/page"), _say("ok")]
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), tool_exposure="all"),
        tool_providers=[
            WebFetchProvider(fetcher=WebFetcher(resolver=resolver, allow=allow))
        ],
    )
    try:
        await agent.run_task(
            "read the docs on pages.test", approval=RememberingPolicy(person.ask)
        )
    finally:
        await agent.close()
    assert [r.tool_name for r in person.asked] == [f"browser:http://{sub}"]
    assert all("# Tomatoes" in text for text in _results(llm))


async def test_a_redirect_to_a_subdomain_of_a_named_site_asks_first(
    server: dict[str, Any],
) -> None:
    person = Person(refuse=True)
    port = int(server["port"])
    leak = "c2VjcmV0.pages.test"
    resolver = pages.resolver_for(port, {leak.lower(): [("127.0.0.1", port)]})
    llm = ScriptedLLM(
        [_fetch(f"http://{pages.HOST}/to?u=http://{leak}/page"), _say("ok")]
    )
    agent = _agent(llm, resolver)
    try:
        await agent.run_task(
            "what does pages.test say?", approval=RememberingPolicy(person.ask)
        )
    finally:
        await agent.close()
    assert [r.tool_name for r in person.asked] == [f"browser:http://{leak.lower()}"]
    assert resolver.calls == [pages.HOST]  # the subdomain was never looked up


@pytest.mark.parametrize(
    ("said", "host", "named"),
    [
        ("what's on pages.test?", "pages.test", True),
        ("what's on pages.test?", "www.pages.test", True),
        ("open www.pages.test", "pages.test", True),
        ("what's on pages.test?", "docs.pages.test", False),
        ("read vercel.app and workers.dev", "attacker.vercel.app", False),
        ("read docs.pages.test", "pages.test", False),
        # A file name is not a site; a full address always names its host.
        ("please fix README.md", "readme.md", False),
        ("run build.sh, then check config.yaml", "build.sh", False),
        ("see https://readme.md/intro", "readme.md", True),
        # A word inside a longer one is not a host.
        ("bücher.de", "cher.de", False),
        ("https://bücher.de/", "xn--bcher-kva.de", True),
        ("打开pages.test看看", "pages.test", True),
        ("pages.test.evil.example", "pages.test", False),
    ],
)
def test_which_sites_a_message_names(said: str, host: str, named: bool) -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.user_message(said)
    assert reviewer.names_host(host) is named


def test_only_the_last_three_messages_name_a_site() -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.user_message("read pages.test")
    for text in ("one", "two", "three"):
        reviewer.user_message(text)
    assert not reviewer.names_host("pages.test")
    reviewer.user_message("x" * 800 + " other.test")
    assert not reviewer.names_host("other.test")  # past the first 800 characters


async def test_an_unnamed_site_asks_once_per_chat_as_the_browser_does(
    server: dict[str, Any],
) -> None:
    person = Person()
    policy = RememberingPolicy(person.ask)
    llm = ScriptedLLM([_fetch(PAGE), _fetch(PAGE), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task("what do the garden notes say?", approval=policy)
        llm.replies.extend([_fetch(PAGE), _say("again")])
        await agent.run_task("and now?", approval=policy)
    finally:
        await agent.close()
    assert [r.tool_name for r in person.asked] == [f"browser:http://{pages.HOST}"]
    request = person.asked[0]
    assert request.preview == f"read {PAGE}"
    assert dict(request.arguments) == {
        "site": pages.HOST,
        "origin": f"http://{pages.HOST}",
        "access": "read",
        "via": "web_fetch",
        "url": PAGE,
    }
    assert "# Tomatoes" in _results(llm)[0]


async def test_the_question_shows_a_long_address_cut_short(
    server: dict[str, Any],
) -> None:
    person = Person(refuse=True)
    address = f"{PAGE}?d=" + "a" * 400
    llm = ScriptedLLM([_fetch(address), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task("garden notes?", approval=RememberingPolicy(person.ask))
    finally:
        await agent.close()
    [request] = person.asked
    assert request.tool_name == f"browser:http://{pages.HOST}"
    shown = request.arguments["url"]
    assert len(shown) == URL_SHOWN and shown == address[: URL_SHOWN - 1] + "…"
    assert request.preview == f"read {shown}"


async def test_a_browser_grant_covers_web_fetch(server: dict[str, Any]) -> None:
    person = Person()
    policy = RememberingPolicy(person.ask)
    policy._granted.add(f"browser:http://{pages.HOST}")  # the browser's "for this chat"
    llm = ScriptedLLM([_fetch(PAGE), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task("what do the garden notes say?", approval=policy)
    finally:
        await agent.close()
    assert person.asked == []
    assert "# Tomatoes" in _results(llm)[0]


async def test_allow_once_asks_again_next_time(server: dict[str, Any]) -> None:
    person = Person(remember=False)
    llm = ScriptedLLM([_fetch(PAGE), _fetch(PAGE), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task("garden notes?", approval=RememberingPolicy(person.ask))
    finally:
        await agent.close()
    assert len(person.asked) == 2


async def test_a_refused_site_is_never_looked_up_or_reached(
    server: dict[str, Any],
) -> None:
    person = Person(refuse=True)
    resolver = _resolver(server)
    llm = ScriptedLLM([_fetch(f"http://{pages.HOST}/page?d=secret"), _say("ok")])
    agent = _agent(llm, resolver)
    try:
        await agent.run_task("garden notes?", approval=RememberingPolicy(person.ask))
    finally:
        await agent.close()
    [result] = _results(llm)
    assert f"melo did not read {pages.HOST}" in result and "not that one" in result
    assert resolver.calls == [] and server["requests"] == []


async def test_a_redirect_to_another_site_asks_before_its_name_is_looked_up(
    server: dict[str, Any],
) -> None:
    person = Person(refuse=True)
    resolver = _resolver(server)
    leak = "http://other.test/page?d=secret"
    llm = ScriptedLLM([_fetch(f"http://{pages.HOST}/to?u={leak}"), _say("ok")])
    agent = _agent(llm, resolver)
    try:
        await agent.run_task(
            "read pages.test/to for me", approval=RememberingPolicy(person.ask)
        )
    finally:
        await agent.close()
    assert [r.tool_name for r in person.asked] == ["browser:http://other.test"]
    assert "melo did not read other.test" in _results(llm)[0]
    assert resolver.calls == [pages.HOST]  # other.test was never looked up


async def test_a_redirect_within_the_site_asks_nothing_more(
    server: dict[str, Any],
) -> None:
    person = Person()
    llm = ScriptedLLM([_fetch(f"http://{pages.HOST}/redirect"), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task("garden notes?", approval=RememberingPolicy(person.ask))
    finally:
        await agent.close()
    assert len(person.asked) == 1 and "# Tomatoes" in _results(llm)[0]


async def test_auto_mode_reads_a_named_site_without_a_question(
    server: dict[str, Any],
) -> None:
    person = Person()
    policy = ChatPolicy(person.ask, mode="auto")
    llm = ScriptedLLM([_fetch(PAGE), _say("ok")])
    agent = _agent(llm, _resolver(server))
    try:
        await agent.run_task("summarise pages.test/page", approval=policy)
    finally:
        await agent.close()
    assert person.asked == [] and "# Tomatoes" in _results(llm)[0]


async def test_a_task_with_nobody_to_ask_is_refused(server: dict[str, Any]) -> None:
    provider = WebFetchProvider(
        task=True, fetcher=WebFetcher(resolver=_resolver(server), allow=_allow)
    )
    [tool] = await provider.start()
    result = await tool.function(url=PAGE)
    assert isinstance(result, ToolOutput) and result.is_error
    assert f"melo did not read {pages.HOST}" in result.text
    assert server["requests"] == []


async def test_a_chat_with_no_approval_policy_reads_as_before(
    server: dict[str, Any],
) -> None:
    provider = WebFetchProvider(
        fetcher=WebFetcher(resolver=_resolver(server), allow=_allow)
    )
    [tool] = await provider.start()
    result = await tool.function(url=PAGE)
    assert isinstance(result, str) and "# Tomatoes" in result


async def test_a_site_refused_earlier_stays_refused_though_the_request_names_it(
    server: dict[str, Any],
) -> None:
    """A run its chat continued never reads a site refused before (#431).

    Not when the person's request names the site (the named-site shortcut
    comes after the earlier refusal), and not through another scheme.
    """
    resolver = _resolver(server)
    refusing = Person(refuse=True)
    first = ScriptedLLM([_fetch(f"http://{pages.HOST}/page?d=secret"), _say("ok")])
    agent = _agent(first, resolver)
    found: list[Any] = []
    try:
        await agent.run_task(
            "garden notes?",
            on_event=found.append,
            approval=RememberingPolicy(refusing.ask),
        )
    finally:
        await agent.close()
    [finished] = [e for e in found if e.type == "tool.finished"]
    [asked] = finished.data["tool_asks"]
    # The continued run: the request names the site; the model tries it again.
    person = Person()
    blocks = EarlierBlocks(
        RememberingPolicy(person.ask),
        {asked["call_key"]: ("the user", "")},
        {asked["call_key"]: asked["loose_keys"]},
    )
    again = ScriptedLLM(
        [_fetch(f"http://{pages.HOST}/page"), _fetch(f"https://{pages.HOST}/page")]
    )
    retried = _agent(again, resolver)
    try:
        await retried.run_task(f"What does {pages.HOST} say?", approval=blocks)
    finally:
        await retried.close()
    assert person.asked == [], "nobody was asked"
    assert resolver.calls == [] and server["requests"] == [], "never reached"
    named, other_scheme = _results(again)
    assert f"melo did not read {pages.HOST}" in named
    assert "the user blocked a step like this one earlier" in named
    assert "the user blocked a step like this one earlier" in other_scheme
