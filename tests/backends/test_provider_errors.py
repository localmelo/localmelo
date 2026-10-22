"""Provider HTTP failures become ProviderError (fake transport, no network)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.providers import LLM_REQUEST_ATTEMPT, ProviderError
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
        (httpx.ConnectError("All connection attempts failed"), "ConnectError: All"),
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


def test_provider_error_defaults() -> None:
    exc = ProviderError("down")
    assert (str(exc), exc.status_code, exc.retryable) == ("down", None, False)
