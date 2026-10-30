"""What the web console calls the model server: oMLX when it is one.

``model.backend_label`` of ``GET /api/health`` and the settings dropdown's
entry for the ``vllm`` backend. Every request goes to an
``httpx.MockTransport``, so nothing reaches a model server.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from localmelo.support.backends import discovery
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway.console import probe as probe_module
from localmelo.support.gateway.console.routes_settings import model_json, settings_json

# A name that never resolves: no test here can reach a model server.
URL = "http://omlx.test/v1"
GENERIC = "OpenAI-compatible server"


def _cfg(**fields: Any) -> Config:
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=URL, chat_model="Qwen3-8B-4bit"),
    )
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def test_a_saved_vllm_backend_is_an_openai_compatible_server() -> None:
    assert model_json(_cfg()) == {
        "backend": "vllm",
        "backend_label": GENERIC,
        "name": "Qwen3-8B-4bit",
        "embedding": "none",
    }


def test_a_config_that_says_omlx_is_labelled_omlx() -> None:
    body = model_json(_cfg(chat_alias="omlx"))
    # The backend key is still vllm: omlx is another spelling of it.
    assert (body["backend"], body["backend_label"]) == ("vllm", "oMLX")


def test_other_backends_and_unknown_ones_keep_their_label() -> None:
    cfg = Config(
        chat_backend="ollama",
        embedding_backend="none",
        ollama=LocalBackendConfig(chat_url="http://omlx.test", chat_model="m"),
    )
    assert model_json(cfg)["backend_label"] == "Ollama"
    assert model_json(Config(chat_backend="nonesuch"))["backend_label"] == "nonesuch"


async def test_the_health_probes_listing_makes_the_label_omlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe lists the models anyway; that listing says the server is oMLX."""
    real = discovery.list_models

    async def list_models(backend: str, url: str, **kwargs: Any) -> list[str]:
        reply = {"data": [{"id": "Qwen3-8B-4bit", "owned_by": "omlx"}]}
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
        return await real(backend, url, transport=transport)

    monkeypatch.setattr(probe_module, "discovery_list_models", list_models)
    cfg = _cfg()
    assert model_json(cfg)["backend_label"] == GENERIC
    assert await probe_module.default_probe("vllm", URL) == (True, "")
    assert model_json(cfg)["backend_label"] == "oMLX"


def test_the_settings_dropdown_names_omlx_beside_vllm() -> None:
    choices = settings_json()["choices"]
    entry = "oMLX or vLLM (OpenAI-compatible, local)"
    assert {"key": "vllm", "label": entry, "kind": "local"} in choices["chat_backends"]
    assert {"key": "vllm", "label": entry} in choices["embedding_backends"]
    # The other entries are unchanged.
    assert {"key": "ollama", "label": "Ollama (local endpoint)", "kind": "local"} in (
        choices["chat_backends"]
    )
