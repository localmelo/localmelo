"""What the web console calls the model server: oMLX when it is one.

``model.backend_label`` of ``GET /api/health`` and the settings dropdown's
oMLX and vLLM rows of the ``vllm`` backend. Every request goes to an
``httpx.MockTransport``, so nothing reaches a model server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support import config, onboard
from localmelo.support.backends import discovery
from localmelo.support.backends.openai_family import CLOUD_BACKENDS, LOCAL_BACKENDS
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway.console import probe as probe_module
from localmelo.support.gateway.console.routes_settings import model_json, settings_json
from localmelo.support.gateway.i18n import using

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
        "local": True,
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


def test_the_model_says_whether_it_runs_on_a_server_of_the_users_own() -> None:
    assert [model_json(Config(chat_backend=key))["local"] for key in LOCAL_BACKENDS]
    assert not any(
        model_json(Config(chat_backend=key))["local"] for key in (*CLOUD_BACKENDS, "")
    )


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


@pytest.mark.parametrize(
    ("spelling", "listed", "row"),
    [("omlx", False, "omlx"), ("vllm", False, "vllm"), ("vllm", True, "omlx")],
    ids=["omlx-file", "vllm-file", "vllm-file-of-a-server-listed-as-omlx"],
)
def test_omlx_and_vllm_are_two_rows_of_one_section_and_one_is_in_use(
    monkeypatch: pytest.MonkeyPatch, spelling: str, listed: bool, row: str
) -> None:
    path = Path(config.config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'chat_backend = "{spelling}"\nembedding_backend = "{spelling}"\n\n'
        f'[vllm]\nchat_url = "{URL}"\nchat_model = "m"\n'
        f'embedding_url = "{URL}"\nembedding_model = "e"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(discovery, "is_omlx", lambda url: listed)
    body = settings_json()
    # The backend stays vllm; the row in use agrees with the label.
    assert (body["chat_backend"], body["chat_choice"]) == ("vllm", row)
    assert (body["embedding_backend"], body["embedding_choice"]) == ("vllm", row)
    omlx = {"key": "omlx", "label": "oMLX (local endpoint)", "section": "vllm"}
    vllm = {"key": "vllm", "label": "vLLM (local endpoint)", "section": "vllm"}
    chat = body["choices"]["chat_backends"]
    assert chat[2:4] == [{**omlx, "kind": "local"}, {**vllm, "kind": "local"}]
    assert body["choices"]["embedding_backends"][2:4] == [omlx, vllm]
    assert {"key": "ollama", "label": "Ollama (local endpoint)", "kind": "local"} in (
        [{k: v for k, v in c.items() if k != "section"} for c in chat]
    )


def test_the_generic_kind_reads_in_the_page_s_language_and_names_stay_names() -> None:
    with using("zh-Hans"):
        assert model_json(_cfg())["backend_label"] == "兼容 OpenAI 的服务器"
        assert model_json(_cfg(chat_alias="omlx"))["backend_label"] == "oMLX"
        choices = settings_json()["choices"]
        labels = {c["key"]: c["label"] for c in choices["chat_backends"]}
        assert (labels["omlx"], labels["vllm"]) == (
            "oMLX（本地服务器）",
            "vLLM（本地服务器）",
        )
        assert labels["openai"] == "OpenAI"
        assert {"key": "none", "label": "无（不按含义搜索）", "section": "none"} in (
            choices["embedding_backends"]
        )
    # A client that asks for no language, and the terminal's setup, read English.
    assert model_json(_cfg())["backend_label"] == GENERIC
    choices = settings_json()["choices"]
    assert [(c["key"], c["label"]) for c in choices["chat_backends"]] == (
        onboard.CHAT_BACKENDS
    )
    assert [(c["key"], c["label"]) for c in choices["embedding_backends"]] == (
        onboard.EMBEDDING_BACKENDS
    )
