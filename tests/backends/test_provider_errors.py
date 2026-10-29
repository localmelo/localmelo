"""Provider HTTP failures become ProviderError (fake transport, no network)."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import (
    LLM_REQUEST_ATTEMPT,
    LLM_REQUEST_KIND,
    LLM_REQUEST_KINDS,
    ProviderError,
)
from localmelo.melo.schema import Message
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

Handler = Callable[[httpx.Request], httpx.Response]

# The path each provider posts to, relative to its base URL.
PATHS = {
    "openai": "/chat/completions",
    "ollama": "/api/chat",
    "embedding": "/embeddings",
}

OK_BODIES: dict[str, Any] = {
    "openai": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    "ollama": {"message": {"role": "assistant", "content": "ok"}},
    "embedding": {"data": [{"embedding": [0.5, 0.25]}]},
}

KINDS = list(PATHS)


async def _request(kind: str, handler: Handler) -> Any:
    """Make one request with provider *kind* over a fake transport."""
    transport = httpx.MockTransport(handler)
    provider: OpenAICompatLLM | OllamaNativeChat | OpenAICompatEmbedding
    if kind == "embedding":
        provider = OpenAICompatEmbedding(
            base_url="http://emb.test/v1", model="e", transport=transport
        )
        try:
            return await provider.embed(["q"])
        finally:
            await provider.close()
    if kind == "openai":
        provider = OpenAICompatLLM(
            base_url="http://llm.test/v1", model="m", transport=transport
        )
    else:
        provider = OllamaNativeChat(
            base_url="http://ollama.test", model="m", transport=transport
        )
    try:
        return await provider.chat([Message(role="user", content="q")])
    finally:
        await provider.close()


async def _error(kind: str, handler: Handler) -> ProviderError:
    with pytest.raises(ProviderError) as excinfo:
        await _request(kind, handler)
    return excinfo.value


def _reply(status: int, body: str | bytes = "", **kwargs: Any) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, **kwargs)

    return handler


def _raise(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


# ── Successful requests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a_good_response_still_parses(kind: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OK_BODIES[kind])

    result = await _request(kind, handler)
    if kind == "embedding":
        assert result == [[0.5, 0.25]]
    else:
        assert result.content == "ok"


@pytest.mark.asyncio
async def test_an_ollama_reply_without_a_message_is_still_an_empty_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "done": True})

    reply = await _request("ollama", handler)
    assert (reply.content, reply.tool_calls) == ("", [])


# ── HTTP error statuses ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (404, False),
        (500, False),
        (501, False),
        (429, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
async def test_an_error_status_maps_by_code(
    kind: str, status: int, retryable: bool
) -> None:
    body = '{"error": {"message": "bench_budget_exhausted"}}'
    exc = await _error(kind, _reply(status, body))
    assert (exc.status_code, exc.retryable) == (status, retryable)
    assert str(exc) == f"POST {PATHS[kind]} failed with HTTP {status}: {body}"
    assert isinstance(exc.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_the_message_quotes_the_first_500_characters_of_the_body(
    kind: str,
) -> None:
    exc = await _error(kind, _reply(500, "x" * 499 + "yz" + "x" * 1000))
    assert str(exc).endswith(": " + "x" * 499 + "y")


@pytest.mark.asyncio
async def test_an_empty_body_is_named() -> None:
    exc = await _error("openai", _reply(503))
    assert str(exc) == "POST /chat/completions failed with HTTP 503: (empty body)"


# ── Requests without a response ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("error", "text"),
    [
        (httpx.ConnectTimeout(""), "ConnectTimeout"),
        (httpx.ReadTimeout("timed out"), "ReadTimeout: timed out"),
        (httpx.RemoteProtocolError("peer closed"), "RemoteProtocolError: peer"),
    ],
)
async def test_timeouts_and_transport_errors_are_retryable(
    kind: str, error: httpx.TransportError, text: str
) -> None:
    exc = await _error(kind, _raise(error))
    assert (exc.status_code, exc.retryable) == (None, True)
    assert str(exc).startswith(f"POST {PATHS[kind]} failed: {text}")
    assert exc.__cause__ is error


# The base URL each provider is built with in _request.
BASE_URLS = {
    "openai": "http://llm.test/v1",
    "ollama": "http://ollama.test",
    "embedding": "http://emb.test/v1",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a_refused_connection_names_the_server_and_the_fix(kind: str) -> None:
    error = httpx.ConnectError("All connection attempts failed")
    exc = await _error(kind, _raise(error))
    assert (exc.status_code, exc.retryable) == (None, True)
    assert str(exc) == (
        f"cannot reach the model server at {BASE_URLS[kind]}. "
        "Start it, or run melo to set one up. "
        "(ConnectError: All connection attempts failed)"
    )
    assert exc.__cause__ is error


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a_body_that_cannot_be_decoded_is_not_retryable(kind: str) -> None:
    handler = _reply(200, b"not gzip", headers={"Content-Encoding": "gzip"})
    exc = await _error(kind, handler)
    assert (exc.status_code, exc.retryable) == (None, False)
    assert str(exc).startswith(f"POST {PATHS[kind]} failed: DecodingError")


# ── Malformed bodies ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a_body_that_is_not_json_is_not_retryable(kind: str) -> None:
    exc = await _error(kind, _reply(200, "<html>proxy page</html>"))
    assert (exc.status_code, exc.retryable) == (200, False)
    assert str(exc) == (
        f"POST {PATHS[kind]} returned HTTP 200 with a malformed body "
        "(not JSON): <html>proxy page</html>"
    )
    assert isinstance(exc.__cause__, ValueError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "body"),
    [
        ("openai", "[]"),
        ("openai", "{}"),
        ("openai", '{"choices": []}'),
        ("openai", '{"choices": ["x"]}'),
        ("openai", '{"choices": [{"message": "x"}]}'),
        ("ollama", "[]"),
        ("ollama", '{"message": null}'),
        ("ollama", '{"message": "x"}'),
        ("embedding", "[]"),
        ("embedding", "{}"),
        ("embedding", '{"data": "x"}'),
        ("embedding", '{"data": [{"index": 0}]}'),
    ],
)
async def test_a_body_without_the_expected_fields_is_not_retryable(
    kind: str, body: str
) -> None:
    exc = await _error(kind, _reply(200, body))
    assert (exc.status_code, exc.retryable) == (200, False)
    assert str(exc).startswith(
        f"POST {PATHS[kind]} returned HTTP 200 with a malformed body ("
    )
    assert str(exc).endswith(f"): {body}")


# ── The attempt header ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_chat_requests_send_the_attempt_number(kind: str) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("X-Localmelo-Attempt"))
        return httpx.Response(200, json=OK_BODIES[kind])

    await _request(kind, handler)
    token = LLM_REQUEST_ATTEMPT.set(3)
    try:
        await _request(kind, handler)
    finally:
        LLM_REQUEST_ATTEMPT.reset(token)
    assert seen == ["1", "3"]


@pytest.mark.asyncio
async def test_embedding_requests_send_no_attempt_header() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=OK_BODIES["embedding"])

    await _request("embedding", handler)
    assert "X-Localmelo-Attempt" not in seen[0]


# ── The call-kind header ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_chat_requests_send_the_call_kind_when_it_is_set(kind: str) -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=OK_BODIES[kind])

    await _request(kind, handler)
    token = LLM_REQUEST_KIND.set("reflect")
    attempt = LLM_REQUEST_ATTEMPT.set(2)
    try:
        await _request(kind, handler)
    finally:
        LLM_REQUEST_ATTEMPT.reset(attempt)
        LLM_REQUEST_KIND.reset(token)
    assert "X-Localmelo-Call-Kind" not in seen[0]
    assert seen[0]["X-Localmelo-Attempt"] == "1"
    assert seen[1]["X-Localmelo-Call-Kind"] == "reflect"
    assert seen[1]["X-Localmelo-Attempt"] == "2"


@pytest.mark.asyncio
async def test_embedding_requests_send_no_call_kind() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=OK_BODIES["embedding"])

    token = LLM_REQUEST_KIND.set("plan")
    try:
        await _request("embedding", handler)
    finally:
        LLM_REQUEST_KIND.reset(token)
    assert "X-Localmelo-Call-Kind" not in seen[0]
    assert "X-Localmelo-Attempt" not in seen[0]


def test_the_call_kinds() -> None:
    assert LLM_REQUEST_KINDS == ("plan", "reflect", "retrieve", "estimate")
    assert LLM_REQUEST_KIND.get() == ""


def test_provider_error_defaults() -> None:
    exc = ProviderError("down")
    assert (str(exc), exc.status_code, exc.retryable) == ("down", None, False)


def _plan_reply(kind: str, content: str = "", call: bool = False) -> dict[str, Any]:
    """A reply body of provider *kind*; with *call*, one python_exec call."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if call and kind == "openai":
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "arguments": json.dumps({"code": "print(1)"}),
                },
            }
        ]
    elif call:
        message["tool_calls"] = [
            {"function": {"name": "python_exec", "arguments": {"code": "print(1)"}}}
        ]
    if kind == "openai":
        return {"choices": [{"message": message}]}
    return {"message": message}


REFLECTION = {
    "summary": "Printed 1.",
    "recommended_action": "continue",
    "task_still_feasible": True,
    "next_step_is_concrete": True,
    "next_step_is_novel": True,
    "rationale": "Answer now.",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_a_run_names_the_kind_of_each_request(kind: str) -> None:
    replies = deque(
        [
            _plan_reply(kind, call=True),
            # The first reflection reply is a tool call: asked again (#31).
            _plan_reply(kind, call=True),
            _plan_reply(kind, json.dumps(REFLECTION)),
            _plan_reply(kind, "1"),
        ]
    )
    seen: list[tuple[str | None, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.headers.get("X-Localmelo-Call-Kind"),
                request.headers.get("X-Localmelo-Attempt"),
            )
        )
        return httpx.Response(200, json=replies.popleft())

    transport = httpx.MockTransport(handler)
    llm: OpenAICompatLLM | OllamaNativeChat
    if kind == "openai":
        llm = OpenAICompatLLM(
            base_url="http://llm.test/v1", model="m", transport=transport
        )
    else:
        llm = OllamaNativeChat(
            base_url="http://ollama.test", model="m", transport=transport
        )
    agent = Agent(llm=llm, options=AgentOptions(steps_per_attempt=1))
    try:
        result = await agent.run_task("Print 1 with Python.")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert seen == [("plan", "1"), ("reflect", "1"), ("reflect", "1"), ("plan", "1")]
