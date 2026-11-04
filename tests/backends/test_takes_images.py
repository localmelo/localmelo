"""``takes_images``: what a model server says about a model reading images.

An OpenAI-compatible server may list the model's own entry in ``GET /v1/models``
with ``vision``, ``input_modalities`` or ``capabilities``; that is asked first.
oMLX answers ``GET /v1/models/status`` with each model's ``model_type``;
Ollama answers ``POST /api/show`` with ``capabilities``. Anything else is
unknown (``None``), and ``tool_images = "auto"`` then sends no pixels.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.gateway.console.vision import Vision
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.fixtures.stub_llm import StubServer


def server(routes: dict[str, httpx.Response], seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler)


def omlx(model_type: str) -> httpx.Response:
    item = {"id": "Qwen-VL-4bit", "model_alias": "qwen-vl", "model_type": model_type}
    return httpx.Response(200, json={"models": [{"id": "other"}, item]})


def listing(*entries: dict[str, Any]) -> httpx.Response:
    """A ``GET /v1/models`` reply: another model's entry, then *entries*."""
    other = {"id": "other", "object": "model", "vision": False}
    return httpx.Response(200, json={"object": "list", "data": [other, *entries]})


def entry(**fields: Any) -> dict[str, Any]:
    return {"id": "qwen-vl", "object": "model", **fields}


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
    assert seen[0] == "GET /v1/models"
    assert set(seen) <= {
        "GET /v1/models",
        "GET /v1/models/status",
        "POST /api/show",
    }
    await llm.close()


# What the model's own entry in GET /v1/models says: True for vision, an image
# among input_modalities or vision among capabilities; False only when it says
# so; otherwise the next route is asked.
LISTED = [
    (entry(vision=True), True),
    (entry(vision=True, input_modalities=["text", "image", "pdf"]), True),
    (entry(input_modalities=["text", "image"]), True),
    (entry(capabilities=["completion", "vision"]), True),
    (entry(vision=False, input_modalities=["text", "image"]), True),
    (entry(vision=False), False),
    (entry(input_modalities=["text"]), False),
    (entry(vision=False, capabilities=["completion"]), False),
    (entry(vision="yes"), None),
    (entry(input_modalities=[]), None),
    (entry(capabilities=["completion"]), None),
    (entry(), None),
    ({"id": "elsewhere", "vision": True}, None),
]


@pytest.mark.parametrize(("item", "verdict"), LISTED)
async def test_the_models_listing_answers_first(
    item: dict[str, Any], verdict: bool | None
) -> None:
    seen: list[str] = []
    routes = {"/v1/models": listing(item)}
    llm = OpenAICompatLLM(
        "http://srv.test/v1", "qwen-vl", transport=server(routes, seen)
    )
    assert await llm.takes_images() is verdict
    # An answer ends the questions; none leaves the other routes to ask.
    assert seen == (
        ["GET /v1/models"]
        if verdict is not None
        else ["GET /v1/models", "GET /v1/models/status", "POST /api/show"]
    )
    await llm.close()


async def test_the_listing_beats_the_other_routes() -> None:
    seen: list[str] = []
    routes = {
        "/v1/models": listing(entry(vision=False)),
        "/v1/models/status": omlx("vlm"),
        "/api/show": httpx.Response(200, json={"capabilities": ["vision"]}),
    }
    llm = OpenAICompatLLM(
        "http://srv.test/v1", "qwen-vl", transport=server(routes, seen)
    )
    assert await llm.takes_images() is False
    assert seen == ["GET /v1/models"]
    await llm.close()


async def test_the_listing_names_a_model_by_its_alias_too() -> None:
    routes = {"/v1/models": listing({"id": "x", "model_alias": "qwen-vl", "vision": 1})}
    llm = OpenAICompatLLM("http://srv.test/v1", "qwen-vl", transport=server(routes, []))
    assert await llm.takes_images() is None  # 1 is not a boolean
    routes = {
        "/v1/models": listing({"id": "x", "model_alias": "qwen-vl", "vision": True})
    }
    llm = OpenAICompatLLM("http://srv.test/v1", "qwen-vl", transport=server(routes, []))
    assert await llm.takes_images() is True
    await llm.close()


@pytest.mark.parametrize(
    "status", [httpx.Response(401), httpx.Response(404), httpx.Response(500)]
)
async def test_a_refused_listing_leaves_the_other_routes(
    status: httpx.Response,
) -> None:
    """The listing at 401/404 is unknown, never an error; oMLX's status route answers."""
    seen: list[str] = []
    routes = {"/v1/models": status, "/v1/models/status": omlx("vlm")}
    llm = OpenAICompatLLM(
        "http://srv.test/v1", "qwen-vl", transport=server(routes, seen)
    )
    assert await llm.takes_images() is True
    assert seen == ["GET /v1/models", "GET /v1/models/status"]
    await llm.close()


async def test_a_listing_that_is_not_json_leaves_the_other_routes() -> None:
    routes = {
        "/v1/models": httpx.Response(200, text="<html>hi</html>"),
        "/api/show": httpx.Response(200, json={"capabilities": ["vision"]}),
    }
    llm = OpenAICompatLLM("http://srv.test/v1", "qwen-vl", transport=server(routes, []))
    assert await llm.takes_images() is True
    await llm.close()


async def test_a_listing_answers_where_the_status_and_show_routes_are_closed() -> None:
    """A server that lists vision in /v1/models and answers 401 on status and 404 on show."""
    seen: list[str] = []
    routes = {
        "/v1/models": listing(
            entry(
                id="chat-model-27b",
                vision=True,
                input_modalities=["text", "image", "pdf"],
            )
        ),
        "/v1/models/status": httpx.Response(401),
        "/api/show": httpx.Response(404),
    }
    llm = OpenAICompatLLM(
        "http://srv.test/v1",
        "chat-model-27b",
        transport=server(routes, seen),
    )
    assert await llm.takes_images() is True
    assert seen == ["GET /v1/models"]
    await llm.close()


async def test_every_question_keeps_its_two_second_timeout() -> None:
    timeouts: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(404)

    llm = OpenAICompatLLM(
        "http://srv.test/v1", "m", transport=httpx.MockTransport(handler)
    )
    assert await llm.takes_images() is None
    assert len(timeouts) == 3
    assert all(
        t == {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0} for t in timeouts
    )
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


SCENARIOS: dict[str, dict[str, Any]] = {
    "reads": {"models": [{"id": "stub-vl", "vision": True}], "replies": []},
    "text": {
        "models": [{"id": "stub-vl", "input_modalities": ["text"]}],
        "replies": [],
    },
    "silent": {"models": ["stub-vl"], "replies": []},
}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[StubServer]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = StubServer(SCENARIOS, default="silent").start()
    try:
        yield server
    finally:
        server.close()


@pytest.mark.parametrize(
    ("scenario", "verdict"), [("reads", True), ("text", False), ("silent", None)]
)
async def test_the_composer_and_the_agent_ask_the_same_question(
    stub: StubServer, scenario: str, verdict: bool | None
) -> None:
    """The console's warning follows the provider's answer, here from a real listing."""
    url = stub.base_url.removesuffix("/v1") + f"/s/{scenario}/v1"
    llm = OpenAICompatLLM(url, "stub-vl")
    assert await llm.takes_images() is verdict
    await llm.close()

    cfg = Config(chat_backend="vllm")
    cfg.vllm.chat_url, cfg.vllm.chat_model = url, "stub-vl"
    console: Any = SimpleNamespace(cfg=cfg)
    assert await Vision(console).reads() is verdict
