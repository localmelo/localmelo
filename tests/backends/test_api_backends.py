"""The shared OpenAI-compatible helpers, and what the ``vllm`` backend calls its server.

The backends themselves are rows of data, tested in ``test_openai_family.py``.
"""

from __future__ import annotations

import httpx

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.support.backends import discovery
from localmelo.support.backends.openai_compat import (
    build_openai_chat,
    build_openai_embedding,
    normalize_url,
)
from localmelo.support.backends.registry import get_backend
from localmelo.support.config import Config, LocalBackendConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_cfg(
    backend_key: str = "ollama",
    chat_url: str = "http://localhost:11434",
    chat_model: str = "qwen3:8b",
    embedding_url: str = "",
    embedding_model: str = "",
) -> Config:
    cfg = Config(chat_backend=backend_key)
    setattr(
        cfg,
        backend_key,
        LocalBackendConfig(
            chat_url=chat_url,
            chat_model=chat_model,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
        ),
    )
    return cfg


async def _list_an_omlx_server(url: str) -> None:
    """List the models of *url*, which answers as oMLX does (no network)."""
    reply = {"data": [{"id": "Qwen3-8B-4bit", "owned_by": "omlx"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
    await discovery.list_models("vllm", url, transport=transport)


# ===========================================================================
# normalize_url
# ===========================================================================


class TestNormalizeUrl:
    def test_appends_v1(self) -> None:
        assert normalize_url("http://localhost:11434") == "http://localhost:11434/v1"

    def test_strips_trailing_slash_then_appends(self) -> None:
        assert normalize_url("http://localhost:11434/") == "http://localhost:11434/v1"

    def test_keeps_existing_v1(self) -> None:
        assert normalize_url("http://localhost:11434/v1") == "http://localhost:11434/v1"

    def test_strips_trailing_slash_after_v1(self) -> None:
        assert (
            normalize_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"
        )


# ===========================================================================
# build_openai_chat / build_openai_embedding
# ===========================================================================


class TestBuildOpenaiChat:
    def test_returns_llm_provider(self) -> None:
        provider = build_openai_chat(
            base_url="http://localhost:8400/v1", model="test-model"
        )
        assert isinstance(provider, BaseLLMProvider)

    def test_passes_api_key(self) -> None:
        provider = build_openai_chat(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        assert isinstance(provider, BaseLLMProvider)


class TestBuildOpenaiEmbedding:
    def test_returns_embedding_provider(self) -> None:
        provider = build_openai_embedding(
            base_url="http://localhost:11434/v1", model="nomic-embed"
        )
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_passes_api_key(self) -> None:
        provider = build_openai_embedding(
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
            api_key="sk-test",
        )
        assert isinstance(provider, BaseEmbeddingProvider)


# ===========================================================================
# What the vllm backend calls its server
# ===========================================================================


class TestVllmLabel:
    def test_an_unknown_server_is_an_openai_compatible_server(self) -> None:
        cfg = _local_cfg("vllm", chat_url="http://localhost:8000")
        assert get_backend("vllm").label(cfg) == "OpenAI-compatible server"
        assert (
            get_backend("vllm").label(cfg, embedding=True) == "OpenAI-compatible server"
        )

    def test_a_config_that_says_omlx_is_omlx(self) -> None:
        cfg = _local_cfg("vllm", chat_url="http://localhost:8000")
        cfg.chat_alias = "omlx"
        assert get_backend("vllm").label(cfg) == "oMLX"
        # The embedding backend is named on its own.
        assert (
            get_backend("vllm").label(cfg, embedding=True) == "OpenAI-compatible server"
        )
        cfg.embedding_alias = "omlx"
        assert get_backend("vllm").label(cfg, embedding=True) == "oMLX"

    async def test_a_listing_that_named_omlx_makes_it_omlx(self) -> None:
        cfg = _local_cfg("vllm", chat_url="http://localhost:8000")
        cfg.vllm.embedding_url = "http://localhost:8001/v1"
        await _list_an_omlx_server(cfg.vllm.chat_url)
        assert get_backend("vllm").label(cfg) == "oMLX"
        # The embedding label follows the embedding URL, else the chat URL.
        assert (
            get_backend("vllm").label(cfg, embedding=True) == "OpenAI-compatible server"
        )
        cfg.vllm.embedding_url = ""
        assert get_backend("vllm").label(cfg, embedding=True) == "oMLX"

    def test_other_backends_keep_their_display_name(self) -> None:
        cfg = _local_cfg("ollama")
        assert get_backend("ollama").label(cfg) == "Ollama"
        assert get_backend("ollama").label(cfg, embedding=True) == "Ollama"


class TestVllmErrorsNameTheServer:
    def test_the_errors_name_the_server_as_the_labels_do(self) -> None:
        cfg = _local_cfg("vllm", chat_url="", chat_model="")
        assert get_backend("vllm").validate(cfg) == [
            "[vllm] chat_url is empty. "
            "Set it to your OpenAI-compatible server URL (e.g. http://localhost:8000/v1).",
            "[vllm] chat_model is empty. "
            "Specify the OpenAI-compatible server model name.",
        ]
        cfg.chat_alias = "omlx"
        assert get_backend("vllm").validate(cfg) == [
            "[vllm] chat_url is empty. "
            "Set it to your oMLX URL (e.g. http://localhost:8000/v1).",
            "[vllm] chat_model is empty. Specify the oMLX model name.",
        ]

    def test_the_embedding_errors_name_the_server_too(self) -> None:
        cfg = _local_cfg("vllm")
        assert get_backend("vllm").validate_embedding(cfg) == [
            "[vllm] embedding_url is empty. "
            "Set it to your OpenAI-compatible server URL for embedding.",
            "[vllm] embedding_model is empty. "
            "Set the OpenAI-compatible server embedding model name.",
        ]
        cfg.embedding_alias = "omlx"
        assert get_backend("vllm").validate_embedding(cfg) == [
            "[vllm] embedding_url is empty. Set it to your oMLX URL for embedding.",
            "[vllm] embedding_model is empty. Set the oMLX embedding model name.",
        ]
