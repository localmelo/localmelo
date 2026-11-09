"""``context_length``: the context a model server runs a model with.

The model's own entry in ``GET /v1/models`` is asked first (``max_model_len``,
else ``context_length``, ``context_window`` or ``max_context_length``), then
oMLX's ``GET /v1/models/status`` (``max_context_window``), then Ollama's
``GET /api/ps`` (the running model's ``context_length``). Ollama's ``/api/show``
gives the model's maximum and is never read for it. The image question
(``test_takes_images.py``) shares the routes, and each is fetched once.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import httpx
import pytest

from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

MODEL = "qwen3"


def server(routes: dict[str, httpx.Response], seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler)


def listing(**fields: Any) -> httpx.Response:
    """A ``GET /v1/models`` reply: another model's entry, then *MODEL*'s."""
    other = {"id": "other", "max_model_len": 999}
    return httpx.Response(200, json={"data": [other, {"id": MODEL, **fields}]})


def status(**fields: Any) -> httpx.Response:
    item = {"id": "Qwen3-4bit", "model_alias": MODEL, **fields}
    return httpx.Response(200, json={"models": [{"id": "other"}, item]})


def running(name: str, length: Any) -> httpx.Response:
    item = {"name": name, "model": name, "context_length": length}
    return httpx.Response(200, json={"models": [item]})


@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        ({"/v1/models": listing(max_model_len=32768)}, 32768),
        ({"/v1/models": listing(context_length=200000)}, 200000),
        ({"/v1/models": listing(context_window=131072)}, 131072),
        ({"/v1/models": listing(max_context_length=32000)}, 32000),
        (
            {"/v1/models": listing(max_model_len=8192, context_length=65536)},
            8192,
        ),
        (
            {
                "/v1/models": listing(),
                "/v1/models/status": status(max_context_window=262144),
            },
            262144,
        ),
        (
            {
                "/v1/models/status": httpx.Response(401),
                "/api/ps": running(f"{MODEL}:latest", 40960),
            },
            40960,
        ),
        ({"/api/ps": running("qwen3:8b", 40960)}, None),  # another model runs
        ({"/api/ps": running(MODEL, 0)}, None),  # a model with no text context
        ({"/v1/models": listing(max_model_len=True)}, None),
        ({"/v1/models": httpx.Response(200, text="not json")}, None),
        (
            {
                "/api/show": httpx.Response(
                    200, json={"model_info": {"qwen3.context_length": 40960}}
                )
            },
            None,
        ),
        ({}, None),
    ],
)
async def test_an_openai_compatible_server(
    routes: dict[str, httpx.Response], expected: int | None
) -> None:
    llm = OpenAICompatLLM("http://llm.test/v1", MODEL, transport=server(routes, []))
    assert await llm.context_length() == expected
    await llm.close()


async def test_each_route_is_fetched_once_for_both_questions() -> None:
    seen: list[str] = []
    routes = {"/v1/models/status": status(model_type="llm", max_context_window=8192)}
    llm = OpenAICompatLLM("http://llm.test/v1", MODEL, transport=server(routes, seen))
    assert await llm.context_length() == 8192
    assert await llm.takes_images() is False
    assert await llm.context_length() == 8192
    assert seen == ["GET /v1/models", "GET /v1/models/status"]
    await llm.close()


async def test_a_server_that_cannot_be_reached_is_unknown_and_asked_again() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return listing(max_model_len=4096)

    llm = OpenAICompatLLM(
        "http://llm.test/v1", MODEL, transport=httpx.MockTransport(handler)
    )
    assert await llm.context_length() is None
    assert await llm.context_length() == 4096
    await llm.close()


async def test_a_hosted_service_is_asked_its_listing_only() -> None:
    seen: list[str] = []
    routes = {"/v1/models/status": status(max_context_window=8192)}
    llm = OpenAICompatLLM(
        "http://hosted.test/v1",
        MODEL,
        probe_images=False,
        transport=server(routes, seen),
    )
    assert await llm.context_length() is None
    assert seen == ["GET /v1/models"]
    await llm.close()


@pytest.mark.parametrize(
    ("extra_body", "routes", "expected", "asked"),
    [
        ({"options": {"num_ctx": 16384}}, {"/api/ps": running(MODEL, 4096)}, 16384, []),
        ({}, {"/api/ps": running(f"{MODEL}:latest", 4096)}, 4096, ["GET /api/ps"]),
        (
            {},
            {
                "/api/show": httpx.Response(
                    200, json={"model_info": {"qwen3.context_length": 40960}}
                )
            },
            None,
            ["GET /api/ps"],
        ),
    ],
)
async def test_ollama_native(
    extra_body: dict[str, Any],
    routes: dict[str, httpx.Response],
    expected: int | None,
    asked: list[str],
) -> None:
    """What the requests send (num_ctx) wins; else the running model; never /api/show."""
    seen: list[str] = []
    llm = OllamaNativeChat(
        "http://ollama.test",
        MODEL,
        extra_body=extra_body,
        transport=server(routes, seen),
    )
    assert await llm.context_length() == expected
    assert seen == asked
    await llm.close()


async def test_the_output_limit_a_planning_request_sends() -> None:
    plain = OpenAICompatLLM("http://llm.test/v1", MODEL)
    capped = OpenAICompatLLM(
        "http://llm.test/v1", MODEL, extra_body={"max_tokens": 2048}
    )
    native = OllamaNativeChat(
        "http://ollama.test", MODEL, extra_body={"options": {"num_predict": 1024}}
    )
    assert (plain.output_limit(), capped.output_limit(), native.output_limit()) == (
        None,
        2048,
        1024,
    )
    for llm in (plain, capped, native):
        await llm.close()


async def test_the_gateway_wrapper_passes_the_answers_on() -> None:
    routes = {"/v1/models": listing(max_model_len=32768)}
    llm = OpenAICompatLLM(
        "http://llm.test/v1",
        MODEL,
        extra_body={"max_tokens": 512},
        transport=server(routes, []),
    )
    gated = GatedLLM(llm, mock.Mock())
    assert (await gated.context_length(), gated.output_limit()) == (32768, 512)
    await llm.close()
