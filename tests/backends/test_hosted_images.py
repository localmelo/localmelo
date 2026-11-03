"""``tool_images = "auto"`` for hosted services: the preset says which models read images.

A hosted service's model listing does not say whether a model reads images, and its
host has no oMLX or Ollama route to ask, so each preset declares the name prefixes of
the models that do (``image_models``) and of the ones that do not (``text_models``).
A model the preset does not name is unknown, and ``auto`` then sends no pixels.
"""

from __future__ import annotations

from unittest import mock

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support import hosted
from localmelo.support.backends.registry import get_backend
from localmelo.support.config import CloudVendorConfig, Config
from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.providers.llm import openai_compat

DEEPSEEK = "https://api.deepseek.com/v1"
PROBE = "localmelo.support.providers.llm.openai_compat.server_takes_images"


def section(model: str, base_url: str = "") -> CloudVendorConfig:
    return CloudVendorConfig(chat_model=model, base_url=base_url, api_key_env="X")


@pytest.mark.parametrize(
    ("backend", "model", "base_url", "expected"),
    [
        ("openai", "gpt-4o", "", True),
        ("openai", "gpt-4o-mini", "", True),
        ("openai", "GPT-4.1", "", True),
        ("openai", "gpt-3.5-turbo", "", None),
        ("gemini", "gemini-2.0-flash", "", True),
        ("gemini", "models/gemini-2.0-flash", "", True),
        ("anthropic", "claude-sonnet-4-20250514", "", True),
        ("anthropic", "claude-3-5-haiku-latest", "", True),
        ("nvidia", "nvidia/llama-3.1-nemotron-70b-instruct", "", False),
        ("nvidia", "meta/llama-3.2-90b-vision-instruct", "", None),
        ("openai", "deepseek-flash", DEEPSEEK, False),
        ("openai", "deepseek-v4-pro", DEEPSEEK, None),
        ("openai", "gpt-4o", "https://llm.example.com/v1", None),
        ("openai", "", "", None),
    ],
)
def test_the_preset_says_whether_a_model_reads_images(
    backend: str, model: str, base_url: str, expected: bool | None
) -> None:
    assert hosted.reads_images(backend, section(model, base_url)) is expected


def test_every_declared_prefix_is_lowercase_and_a_suggested_model_is_not_in_both() -> (
    None
):
    for preset in hosted.PRESETS:
        for prefix in (*preset.image_models, *preset.text_models):
            assert prefix == prefix.lower()
        for model in preset.models:
            both = any(model.startswith(p) for p in preset.image_models) and any(
                model.startswith(p) for p in preset.text_models
            )
            assert not both, model


def test_a_suggested_model_that_reads_images_is_declared_for_each_service() -> None:
    reading = {
        preset.id: [
            model
            for model in preset.models
            if hosted.reads_images(
                preset.backend,
                section(model, preset.stored_url),
            )
        ]
        for preset in hosted.PRESETS
    }
    assert reading["openai"] == ["gpt-4o", "gpt-4o-mini"]
    assert reading["gemini"] == ["gemini-2.0-flash"]
    assert reading["anthropic"] == ["claude-sonnet-4-20250514"]
    assert reading["nvidia"] == []
    assert reading["deepseek"] == []
    assert reading["other"] == []


def config_for(backend: str, model: str, base_url: str = "") -> Config:
    cfg = Config(chat_backend=backend, embedding_backend="none")
    sec = getattr(cfg, backend)
    sec.chat_model = model
    sec.base_url = base_url
    sec.api_key_env = "T353_TEST_KEY"
    return cfg


@pytest.mark.parametrize(
    ("backend", "model", "base_url", "expected"),
    [
        ("openai", "gpt-4o", "", True),
        ("gemini", "gemini-2.0-flash", "", True),
        ("anthropic", "claude-sonnet-4-20250514", "", True),
        ("nvidia", "nvidia/llama-3.1-nemotron-70b-instruct", "", False),
        ("openai", "deepseek-flash", DEEPSEEK, False),
        ("openai", "deepseek-v4-pro", DEEPSEEK, None),
    ],
)
async def test_a_hosted_provider_answers_from_the_preset_and_asks_no_server(
    backend: str,
    model: str,
    base_url: str,
    expected: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("T353_TEST_KEY", "sk-test-0123456789")
    llm = get_backend(backend).build_chat_provider(config_for(backend, model, base_url))
    with mock.patch(PROBE, side_effect=AssertionError("a hosted host was asked")):
        assert await llm.takes_images() is expected
    await llm.close()


async def test_the_gateway_wrapper_passes_the_answer_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("T353_TEST_KEY", "sk-test-0123456789")
    llm = get_backend("openai").build_chat_provider(config_for("openai", "gpt-4o"))
    gated = GatedLLM(llm, mock.Mock())
    assert await gated.takes_images() is True
    await llm.close()


async def test_a_free_address_is_still_asked_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The free-form address may be an oMLX or Ollama behind a proxy, so it is asked."""
    monkeypatch.setenv("T353_TEST_KEY", "sk-test-0123456789")
    cfg = config_for("openai", "qwen-vl", "https://llm.example.com/v1")
    llm = get_backend("openai").build_chat_provider(cfg)
    asked: list[str] = []

    async def probe(http: object, model: str, **routes: object) -> bool:
        asked.append(model)
        return True

    with mock.patch(PROBE, probe):
        assert await llm.takes_images() is True
    assert asked == ["qwen-vl"]
    await llm.close()


async def test_a_provider_built_without_a_declaration_asks_as_before() -> None:
    llm = openai_compat.OpenAICompatLLM("http://omlx.test/v1", "qwen-vl")
    asked: list[str] = []

    async def probe(http: object, model: str, **routes: object) -> bool | None:
        asked.append(model)
        return None

    with mock.patch(PROBE, probe):
        assert await llm.takes_images() is None
    assert asked == ["qwen-vl"]
    await llm.close()


@pytest.mark.parametrize(
    ("backend", "model", "base_url", "sent"),
    [
        ("openai", "gpt-4o", "", True),
        ("anthropic", "claude-sonnet-4-20250514", "", True),
        ("openai", "deepseek-flash", DEEPSEEK, False),
        ("openai", "deepseek-v4-pro", DEEPSEEK, False),
    ],
)
async def test_auto_sends_pictures_to_a_hosted_model_that_reads_them(
    backend: str,
    model: str,
    base_url: str,
    sent: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("T353_TEST_KEY", "sk-test-0123456789")
    llm = get_backend(backend).build_chat_provider(config_for(backend, model, base_url))
    agent = Agent(llm=llm, options=AgentOptions(tool_images="auto"))
    try:
        assert await agent._sends_images() is sent
    finally:
        await agent.close()
