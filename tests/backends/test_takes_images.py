"""``takes_images``: what a model server says about a model reading images.

oMLX answers ``GET /v1/models/status`` with each model's ``model_type``;
Ollama answers ``POST /api/show`` with ``capabilities``. Anything else is
unknown (``None``), and ``tool_images = "auto"`` then sends no pixels.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def server(routes: dict[str, httpx.Response], seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler)


def omlx(model_type: str) -> httpx.Response:
    item = {"id": "Qwen-VL-4bit", "model_alias": "qwen-vl", "model_type": model_type}
    return httpx.Response(200, json={"models": [{"id": "other"}, item]})


@pytest.mark.parametrize(
    ("routes", "model", "expected"),
    [
        ({"/v1/models/status": omlx("vlm")}, "qwen-vl", True),
        ({"/v1/models/status": omlx("vlm")}, "Qwen-VL-4bit", True),
        ({"/v1/models/status": omlx("llm")}, "qwen-vl", False),
        ({"/v1/models/status": omlx("embedding")}, "qwen-vl", None),
        (
            {"/api/show": httpx.Response(200, json={"capabilities": ["vision"]})},
            "qwen-vl",
            True,
        ),
        (
            {
                "/v1/models/status": httpx.Response(401),
                "/api/show": httpx.Response(200, json={"capabilities": ["tools"]}),
            },
            "qwen-vl",
            False,
        ),
        ({"/api/show": httpx.Response(200, text="not json")}, "qwen-vl", None),
        ({}, "qwen-vl", None),
    ],
)
async def test_openai_compatible_server(
    routes: dict[str, httpx.Response], model: str, expected: bool | None
) -> None:
    seen: list[str] = []
    llm = OpenAICompatLLM("http://omlx.test/v1", model, transport=server(routes, seen))
    assert await llm.takes_images() is expected
    assert seen[0] == "GET /v1/models/status"
    assert set(seen) <= {"GET /v1/models/status", "POST /api/show"}
    await llm.close()


async def test_an_unreachable_server_does_not_say() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm = OpenAICompatLLM(
        "http://omlx.test/v1", "m", transport=httpx.MockTransport(refuse)
    )
    assert await llm.takes_images() is None
    await llm.close()


async def test_ollama_native_asks_show_only() -> None:
    seen: list[str] = []
    routes = {"/api/show": httpx.Response(200, json={"capabilities": ["vision"]})}
    llm = OllamaNativeChat(
        "http://ollama.test", "qwen-vl", transport=server(routes, seen)
    )
    assert await llm.takes_images() is True
    assert seen == ["POST /api/show"]
    await llm.close()


async def test_the_base_provider_does_not_know() -> None:
    class Plain(BaseLLMProvider):
        async def chat(self, messages: Any, tools: Any = None) -> Any:
            raise AssertionError

        async def close(self) -> None:
            pass

    assert await Plain().takes_images() is None


async def test_the_console_gate_forwards_the_question() -> None:
    seen: list[str] = []
    routes = {"/v1/models/status": omlx("vlm")}
    inner = OpenAICompatLLM(
        "http://omlx.test/v1", "qwen-vl", transport=server(routes, seen)
    )
    gated = GatedLLM(inner, client=None)  # type: ignore[arg-type]
    assert await gated.takes_images() is True
    await inner.close()
