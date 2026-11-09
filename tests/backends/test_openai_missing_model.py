"""A chat model the server no longer has: the provider asks the listing (fake transport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.schema import Message
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

SWITCHED = (
    "“old” is not on the server, so melo uses “new”, the only chat model it offers."
)
OK_REPLY = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


class Server:
    """A fake server that has *serving* and answers any other model with 404."""

    def __init__(self, *serving: str, listing: int = 200) -> None:
        self.serving = serving
        self.listing = listing
        self.asked: list[tuple[str, str]] = []  # (path, model) of each request

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.asked.append((request.url.path, body.get("model", "")))
        if request.url.path.endswith("/models"):
            if self.listing != 200:
                return httpx.Response(self.listing, json={})
            return httpx.Response(
                200, json={"data": [{"id": name} for name in self.serving]}
            )
        if body["model"] not in self.serving:
            return httpx.Response(404, json={"error": {"message": "no such model"}})
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"choices": [{"delta": {"content": "hi"}, '
                    b'"finish_reason": "stop"}]}\n\ndata: [DONE]\n\n'
                ),
            )
        return httpx.Response(200, json=OK_REPLY)

    def chats(self) -> list[str]:
        """The model each chat request asked for."""
        return [model for path, model in self.asked if path.endswith("/completions")]


def _llm(
    server: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        base_url="http://llm.test/v1",
        model="old",
        transport=httpx.MockTransport(server),
        **kwargs,
    )


ASK = [Message(role="user", content="q")]


async def test_the_only_chat_model_is_used_from_the_next_request_on() -> None:
    server = Server("new", "nomic-embed")
    llm = _llm(server)
    with pytest.raises(ProviderError) as refused:
        await llm.chat(ASK)
    assert str(refused.value) == refused.value.notice == SWITCHED
    assert refused.value.status_code == 404
    assert refused.value.retryable  # the agent asks again, and says why
    assert llm.model == "new"
    assert (await llm.chat(ASK)).content == "hi"
    assert server.chats() == ["old", "new"]
    await llm.close()


async def test_a_stream_follows_the_same_way() -> None:
    server = Server("new")
    llm = _llm(server)
    with pytest.raises(ProviderError) as refused:
        async for _ in llm.stream_chat(ASK):
            pass
    assert str(refused.value) == SWITCHED
    assert refused.value.retryable
    fragments = [fragment async for fragment in llm.stream_chat(ASK)]
    assert fragments[-1].reply is not None
    assert fragments[-1].reply.content == "hi"
    await llm.close()


async def test_several_chat_models_leave_the_refusal_as_it_is() -> None:
    server = Server("a", "b", "nomic-embed")
    llm = _llm(server)
    with pytest.raises(ProviderError) as refused:
        await llm.chat(ASK)
    assert "failed with HTTP 404" in str(refused.value)
    assert not refused.value.retryable
    assert llm.model == "old"
    await llm.close()


@pytest.mark.parametrize("listing", [404, 500])
async def test_a_listing_that_fails_leaves_the_refusal_as_it_is(listing: int) -> None:
    llm = _llm(Server("new", listing=listing))
    with pytest.raises(ProviderError, match="failed with HTTP 404"):
        await llm.chat(ASK)
    assert llm.model == "old"
    await llm.close()


async def test_a_hosted_service_is_never_asked_for_its_models() -> None:
    server = Server("new")
    llm = _llm(server, api_key="secret")
    with pytest.raises(ProviderError, match="failed with HTTP 404"):
        await llm.chat(ASK)
    assert [path for path, _ in server.asked] == ["/v1/chat/completions"]
    await llm.close()


async def test_the_new_model_is_not_followed_again_in_one_request() -> None:
    # The only model it offers is refused too: the refusal ends the request.
    def server(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "new"}]})
        return httpx.Response(404, json={})

    llm = _llm(server)
    with pytest.raises(ProviderError):
        await llm.chat(ASK)
    with pytest.raises(ProviderError, match="failed with HTTP 404"):
        await llm.chat(ASK)
    await llm.close()


async def test_a_run_goes_on_with_the_offered_model_and_says_why() -> None:
    server = Server("new")
    agent = Agent(llm=_llm(server), options=AgentOptions(builtins=frozenset()))
    agent.chat.retry_delays = (0.0,)
    events: list[AgentEvent] = []
    result = await agent.run_task("say hi", on_event=events.append)
    assert result.status == "completed"
    assert server.chats()[-1] == "new"
    notices = [e.data["message"] for e in events if e.type == "notice"]
    assert notices == [SWITCHED], 'one line, with no "request failed" before it'
    await agent.close()
