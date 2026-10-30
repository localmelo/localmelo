"""What the terminal UI calls the model server: oMLX when it is one.

The label of a ``vllm`` backend is ``oMLX`` when the config file says
``omlx`` or a model listing of the server did (``owned_by``), else
``OpenAI-compatible server``. Every request goes to an ``httpx.MockTransport``,
so nothing reaches a model server.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.backends import discovery
from localmelo.support.cli.agents import (
    AgentFactory,
    _embedding_name,
    config_backend,
    url_backend,
    url_embedding,
)
from localmelo.support.config import Config, LocalBackendConfig

# Names that never resolve: no test here can reach a model server.
URL = "http://omlx.test/v1"
OTHER = "http://other.test/v1"
GENERIC = "OpenAI-compatible server"

_OMLX_LISTING = {"data": [{"id": "Qwen3-8B-4bit", "owned_by": "omlx"}]}
_VLLM_LISTING = {"data": [{"id": "Qwen3-8B-4bit", "owned_by": "vllm"}]}


def _cfg(**fields: Any) -> Config:
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=URL, chat_model="Qwen3-8B-4bit"),
    )
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def _factory(backend: Any, cfg: Config | None) -> AgentFactory:
    return AgentFactory(
        backend=backend,
        cfg=cfg,
        options=AgentOptions(),
        workspace="/tmp",
        saves=False,
    )


@pytest.fixture(autouse=True)
def _no_ollama_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moving a saved session to another server asks whether it is Ollama."""

    async def is_ollama(url: str, **kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(discovery, "is_ollama", is_ollama)


async def _list(url: str, payload: object = _OMLX_LISTING) -> None:
    """List the models of *url*, which answers *payload*; nothing is sent out."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    await discovery.list_models("vllm", url, transport=transport)


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The reply to every model listing; the real listing runs over a mock."""
    reply: dict[str, Any] = {"payload": _OMLX_LISTING}
    real = discovery.list_models

    async def list_models(key: str, url: str, **kwargs: Any) -> list[str]:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=reply["payload"])
        )
        return await real(key, url, transport=transport)

    monkeypatch.setattr(discovery, "list_models", list_models)
    return reply


# ── The banner's label ──


def test_a_saved_openai_compatible_server_is_not_called_vllm() -> None:
    backend = config_backend(_cfg(), None)
    assert (backend.key, backend.label) == ("vllm", GENERIC)


def test_a_config_that_says_omlx_is_labelled_omlx() -> None:
    assert config_backend(_cfg(chat_alias="omlx"), None).label == "oMLX"


async def test_a_listing_that_named_omlx_labels_the_saved_server_omlx() -> None:
    cfg = _cfg()
    assert config_backend(cfg, None).label == GENERIC
    await _list(URL)
    assert config_backend(cfg, None).label == "oMLX"


def test_another_backend_keeps_its_name() -> None:
    cfg = Config(
        chat_backend="ollama",
        embedding_backend="none",
        ollama=LocalBackendConfig(chat_url="http://127.0.0.1:11434", chat_model="m"),
    )
    assert config_backend(cfg, None).label == "Ollama"
    assert config_backend(Config(chat_backend="nonesuch"), None).label == "nonesuch"


async def test_a_url_session_is_generic_until_a_listing_says_more() -> None:
    assert url_backend(URL, "m", source="--base-url").label == GENERIC
    await _list(URL)
    assert url_backend(URL, "m", source="--base-url").label == "oMLX"
    assert url_backend(OTHER, "m", source="--base-url").label == GENERIC


async def test_the_embedding_server_is_named_the_same_way() -> None:
    assert url_embedding(None, "e") == ""
    assert url_embedding(URL, "e") == f"{GENERIC} · e"
    assert url_embedding(URL, None) == f"{GENERIC} · default"
    await _list(URL)
    assert url_embedding(URL, "e") == "oMLX · e"
    cfg = _cfg(embedding_backend="vllm", embedding_alias="omlx")
    cfg.vllm.embedding_url, cfg.vllm.embedding_model = OTHER, "bge-m3"
    assert _embedding_name(cfg) == "oMLX · bge-m3"
    cfg.embedding_alias = ""
    assert _embedding_name(cfg) == f"{GENERIC} · bge-m3"


# ── A listing during the session renames the server ──


async def test_the_sessions_listing_renames_a_saved_server(
    listing: dict[str, Any],
) -> None:
    factory = _factory(config_backend(_cfg(), None), _cfg())
    assert factory.backend.label == GENERIC
    assert await factory.list_models() == ["Qwen3-8B-4bit"]
    assert factory.backend.label == "oMLX"
    assert (factory.backend.key, factory.backend.model) == ("vllm", "Qwen3-8B-4bit")


async def test_the_sessions_listing_renames_a_url_server(
    listing: dict[str, Any],
) -> None:
    factory = _factory(url_backend(URL, "m", source="--base-url"), None)
    await factory.list_models()
    assert factory.backend.label == "oMLX"


async def test_a_listing_of_another_kind_of_server_renames_nothing(
    listing: dict[str, Any],
) -> None:
    listing["payload"] = _VLLM_LISTING
    factory = _factory(config_backend(_cfg(), None), _cfg())
    await factory.list_models()
    assert factory.backend.label == GENERIC


async def test_an_ollama_server_is_never_renamed(listing: dict[str, Any]) -> None:
    cfg = Config(
        chat_backend="ollama",
        embedding_backend="none",
        ollama=LocalBackendConfig(chat_url=URL, chat_model="m"),
    )
    factory = _factory(config_backend(cfg, None), cfg)
    await factory.list_models()
    assert factory.backend.label == "Ollama"


async def test_a_listing_of_another_url_does_not_rename_the_session(
    listing: dict[str, Any],
) -> None:
    factory = _factory(url_backend(OTHER, "m", source="--base-url"), None)
    await factory.list_models(URL)  # /backend asks about a server it may move to
    assert factory.backend.label == GENERIC
    await factory.use_url(URL, "Qwen3-8B-4bit")
    assert factory.backend.label == "oMLX"


async def test_a_session_moved_to_another_server_drops_the_files_omlx() -> None:
    cfg = _cfg(chat_alias="omlx")
    factory = _factory(config_backend(cfg, None), cfg)
    assert factory.backend.label == "oMLX"
    await factory.use_url(OTHER, "m")
    assert factory.backend.label == GENERIC
    assert cfg.chat_alias == "omlx"  # the session's own copy changed, not the file's


async def test_a_session_moved_to_the_same_server_keeps_the_files_omlx() -> None:
    cfg = _cfg(chat_alias="omlx")
    factory = _factory(config_backend(cfg, None), cfg)
    await factory.use_url(URL, "other-model")
    assert factory.backend.label == "oMLX"
