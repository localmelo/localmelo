"""web_fetch's site gate: which sites read without a question, and which ask first.

A site the person's own message names (or a parent domain of it) reads
without a question in every mode. Any other site asks once per chat through
the browser's site question (``browser:<origin>``), so one "allow for this
chat" covers both tools. The gate runs before the name is looked up, on the
first address and on a redirect to another site. A task with nobody to ask
is refused.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import Message, ToolCall, ToolDef
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


async def test_a_parent_domain_names_its_subdomains(server: dict[str, Any]) -> None:
    person = Person()
    resolver = pages.resolver_for(
        int(server["port"]), {"docs.pages.test": [("127.0.0.1", int(server["port"]))]}
    )

    def allow(host: str, ip: Any) -> bool:
        return host == "docs.pages.test" and str(ip) == "127.0.0.1"

    llm = ScriptedLLM([_fetch("http://docs.pages.test/page"), _say("ok")])
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
    assert person.asked == []


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
    assert request.preview == f"read {pages.HOST}"
    assert dict(request.arguments) == {
        "site": pages.HOST,
        "origin": f"http://{pages.HOST}",
        "access": "read",
        "via": "web_fetch",
    }
    assert "# Tomatoes" in _results(llm)[0]


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
