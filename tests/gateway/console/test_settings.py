"""API v2 settings, key-status, test and models routes (``routes_settings.py``)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from localmelo.support import config, onboard
from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    GatewayConfig,
    LocalBackendConfig,
    SchedulerConfig,
)
from localmelo.support.gateway.console import Console, routes_settings
from localmelo.support.gateway.console.probe import ModelServerProbe
from localmelo.support.onboard_probe import ProbeError

from .conftest import open_session, seed_turn


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    """Point ``config.CONFIG_DIR`` and ``config.CONFIG_PATH`` at *tmp_path*."""
    cfg_dir = tmp_path / "cfg"
    cfg_path = cfg_dir / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_DIR", str(cfg_dir)),
        mock.patch.object(config, "CONFIG_PATH", str(cfg_path)),
    ):
        yield cfg_path


def test_get_settings_shape_when_file_missing(client: TestClient) -> None:
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"
    body = resp.json()
    assert body["chat_backend"] == ""
    assert body["embedding_backend"] == ""
    assert body["workspace"] == "~/localmelo-workspace"
    assert body["revision"] == ""
    assert body["defaults"] == onboard.backend_defaults()
    assert set(body["sections"]) == set(config._LOCAL_SECTIONS) | set(
        config._CLOUD_SECTIONS
    )
    for key in config._LOCAL_SECTIONS:
        assert set(body["sections"][key]) == {
            "chat_url",
            "chat_model",
            "embedding_url",
            "embedding_model",
        }
    for key in config._CLOUD_SECTIONS:
        assert set(body["sections"][key]) == {
            "api_key_env",
            "api_key_set",
            "chat_model",
            "base_url",
        }
    assert body["choices"]["chat_backends"] == [
        {
            "key": key,
            "label": label,
            "kind": "local" if key in config._LOCAL_SECTIONS else "cloud",
        }
        for key, label in onboard.CHAT_BACKENDS
    ]
    assert body["choices"]["embedding_backends"] == [
        {"key": key, "label": label} for key, label in onboard.EMBEDDING_BACKENDS
    ]
    assert body["errors"]


def test_get_settings_with_saved_file(
    client: TestClient, isolated_config: Path
) -> None:
    cfg = Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(
            chat_url="http://127.0.0.1:8765/v1", chat_model="stub-model"
        ),
        gateway=GatewayConfig(port=8401, host="127.0.0.1", workspace="~/my-ws"),
    )
    config.save(cfg)
    expected_rev = hashlib.sha256(isolated_config.read_bytes()).hexdigest()[:16]

    body = client.get("/api/settings").json()
    assert body["chat_backend"] == "mlc"
    assert body["embedding_backend"] == "none"
    assert body["workspace"] == "~/my-ws"
    assert body["sections"]["mlc"]["chat_url"] == "http://127.0.0.1:8765/v1"
    assert body["sections"]["mlc"]["chat_model"] == "stub-model"
    assert body["errors"] == []
    assert body["revision"] == expected_rev


def test_api_key_value_never_leaves(
    client: TestClient,
    console: Console,
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-secret-value-9876543210abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    cfg = Config(
        chat_backend="openai",
        embedding_backend="none",
        openai=CloudVendorConfig(api_key_env="OPENAI_API_KEY", chat_model="gpt-4o"),
    )
    config.save(cfg)
    console.cfg = cfg

    with caplog.at_level(logging.DEBUG):
        r_settings = client.get("/api/settings")
        r_key = client.get("/api/settings/key-status?name=OPENAI_API_KEY")
        r_health = client.get("/api/health")
        r_put_fail = client.put(
            "/api/settings",
            json={
                "chat_backend": "openai",
                "embedding_backend": "none",
                "workspace": "~/ws",
                "sections": {
                    "openai": {"api_key_env": "OPENAI_API_KEY", "chat_model": ""}
                },
                "revision": r_settings.json()["revision"],
            },
        )
        with mock.patch.object(
            routes_settings,
            "probe_all",
            side_effect=ProbeError(f"rejected key {secret}"),
        ):
            r_test_fail = client.post(
                "/api/settings/test",
                json={
                    "chat_backend": "openai",
                    "embedding_backend": "none",
                    "workspace": "~/ws",
                    "sections": {
                        "openai": {
                            "api_key_env": "OPENAI_API_KEY",
                            "chat_model": "gpt-4o",
                        }
                    },
                },
            )

    assert r_settings.status_code == 200
    assert r_settings.json()["sections"]["openai"]["api_key_set"] is True
    assert r_key.status_code == 200
    assert r_key.json() == {"set": True}
    assert r_health.status_code == 200
    assert r_put_fail.status_code == 400
    assert r_test_fail.status_code == 200
    assert r_test_fail.json()["ok"] is False

    for resp in (r_settings, r_key, r_health, r_put_fail, r_test_fail):
        assert secret not in resp.text
    for record in caplog.records:
        assert secret not in record.getMessage()


@pytest.mark.parametrize(
    "bad_name",
    ["", "123ABC", "OPENAI-KEY", "KEY WITH SPACE", "FOO\n", "A" * 129],
)
def test_key_status_bad_name_is_400(client: TestClient, bad_name: str) -> None:
    resp = client.get("/api/settings/key-status", params={"name": bad_name})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_key_status_returns_only_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    resp = client.get("/api/settings/key-status?name=SOME_UNSET_KEY")
    assert resp.status_code == 200
    assert resp.json() == {"set": False}


async def test_put_settings_preserves_other_fields_and_closes_idle_agents(
    client: TestClient,
    console: Console,
    isolated_config: Path,
    workspace: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    initial = Config(
        chat_backend="vllm",
        embedding_backend="none",
        mcp_config="~/mcp.json",
        request_timeout_s=600,
        vllm=LocalBackendConfig(
            chat_url="http://127.0.0.1:8000/v1",
            chat_model="old-model",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        ollama=LocalBackendConfig(
            chat_url="http://127.0.0.1:11434", chat_model="keep-me"
        ),
        gateway=GatewayConfig(port=8401, host="127.0.0.1"),
        scheduler=SchedulerConfig(policy="class"),
    )
    config.save(initial)
    # Simulate a CLI --port override in the running config that must NOT be saved.
    console.cfg.gateway.port = 9999

    turn = await seed_turn(console.store, workspace)
    entry = await open_session(console.sessions, turn.session_id)
    assert entry.agent is not None

    rev = client.get("/api/settings").json()["revision"]
    with caplog.at_level(
        logging.INFO, logger="localmelo.support.gateway.console.routes_settings"
    ):
        resp = client.put(
            "/api/settings",
            json={
                "chat_backend": "vllm",
                "embedding_backend": "none",
                "workspace": "~/new-workspace",
                "sections": {
                    "vllm": {
                        "chat_url": "http://127.0.0.1:8000/v1",
                        "chat_model": 'model-"quoted"\\🚀',
                    }
                },
                "revision": rev,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["workspace"] == "~/new-workspace"
    assert body["sections"]["vllm"]["chat_model"] == 'model-"quoted"\\🚀'
    assert body["revision"] != rev

    # The disk file kept its port, extra_body, mcp_config, request_timeout_s, ollama, scheduler.
    saved = config.load()
    assert saved.gateway.port == 8401
    assert saved.gateway.workspace == "~/new-workspace"
    assert saved.mcp_config == "~/mcp.json"
    assert saved.request_timeout_s == 600
    assert saved.vllm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert saved.ollama.chat_model == "keep-me"
    assert saved.scheduler.policy == "class"

    # The running console.cfg updated its model while keeping its running port override.
    assert console.cfg.vllm.chat_model == 'model-"quoted"\\🚀'
    assert console.cfg.gateway.port == 9999

    # Idle Agent was closed and its conversation kept.
    assert entry.agent is None
    assert entry.kept is not None

    # Log line records field names only, never values.
    assert any(
        "settings saved: chat_backend=vllm embedding_backend=none changed=workspace,vllm.chat_model"
        in r.getMessage()
        for r in caplog.records
    )


def test_the_settings_carry_the_memory_switch_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    config.save(
        Config(
            chat_backend="mlc",
            embedding_backend="none",
            mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
        )
    )
    loaded = client.get("/api/settings").json()
    assert loaded["across_chats"] is True
    body = {
        "chat_backend": "mlc",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "sections": {},
        "revision": loaded["revision"],
    }
    saved = client.put("/api/settings", json=body)
    assert saved.status_code == 200 and saved.json()["across_chats"] is True
    off = client.put(
        "/api/settings",
        json={**body, "revision": saved.json()["revision"], "across_chats": False},
    )
    assert off.status_code == 200 and off.json()["across_chats"] is False
    assert config.load().memory.across_chats is False
    again = client.put(
        "/api/settings", json={**body, "revision": off.json()["revision"]}
    )
    assert again.json()["across_chats"] is False, "a body without it leaves it as it is"


def test_put_settings_conflict_on_stale_revision(
    client: TestClient, isolated_config: Path
) -> None:
    cfg = Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
    )
    config.save(cfg)
    resp = client.put(
        "/api/settings",
        json={
            "chat_backend": "mlc",
            "embedding_backend": "none",
            "workspace": "~/ws",
            "sections": {},
            "revision": "0000000000000000",
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "settings_changed"


@pytest.mark.parametrize(
    ("patch", "expected_code"),
    [
        ({"workspace": "relative/path"}, "invalid_request"),
        ({"workspace": "/"}, "invalid_request"),
        ({"workspace": "~/"}, "invalid_request"),
        ({"workspace": ""}, "invalid_request"),
        ({"workspace": "~/bad\nline"}, "invalid_request"),
        ({"chat_backend": "mlc\x00"}, "invalid_request"),
        (
            {"sections": {"mlc": {"chat_model": "bad\x07model"}}},
            "invalid_request",
        ),
        (
            {"sections": {"mlc": {"chat_model": "bad\x85model"}}},
            "invalid_request",
        ),
        (
            {"sections": {"openai": {"api_key_env": "FOO\n", "chat_model": "m"}}},
            "invalid_request",
        ),
        (
            {"sections": {"openai": {"api_key_set": True}}},
            "invalid_request",
        ),
        (
            {"sections": {"unknown": {"chat_model": "m"}}},
            "invalid_request",
        ),
        (
            {
                "sections": {
                    "openai": {"api_key_env": "UNSET_CLOUD_KEY", "chat_model": "gpt-4o"}
                },
                "chat_backend": "openai",
            },
            "invalid_settings",
        ),
    ],
)
def test_put_settings_validation_and_field_checks(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    patch: dict[str, Any],
    expected_code: str,
) -> None:
    monkeypatch.delenv("UNSET_CLOUD_KEY", raising=False)
    payload: dict[str, Any] = {
        "chat_backend": "mlc",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "sections": {"mlc": {"chat_url": "http://127.0.0.1:9/v1", "chat_model": "m"}},
        "revision": "",
    }
    payload.update(patch)
    resp = client.put("/api/settings", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == expected_code
    if expected_code == "invalid_settings":
        assert resp.json()["error"]["details"]["errors"]


def test_put_settings_requires_workspace_field(
    client: TestClient, isolated_config: Path
) -> None:
    cfg = Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
        gateway=GatewayConfig(port=8401, host="127.0.0.1", workspace="~/keep-me"),
    )
    config.save(cfg)
    rev = client.get("/api/settings").json()["revision"]

    resp = client.put(
        "/api/settings",
        json={
            "chat_backend": "mlc",
            "embedding_backend": "none",
            "sections": {},
            "revision": rev,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "workspace" in resp.json()["error"]["message"]
    assert config.load().gateway.workspace == "~/keep-me"


def test_put_settings_rejects_lone_surrogate_in_json(client: TestClient) -> None:
    raw = (
        b'{"chat_backend":"mlc","embedding_backend":"none",'
        b'"workspace":"~/\\ud800","sections":{},"revision":""}'
    )
    resp = client.put(
        "/api/settings",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_settings_test_endpoint_success_failure_and_timeout(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {
        "chat_backend": "mlc",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "sections": {"mlc": {"chat_url": "http://127.0.0.1:9/v1", "chat_model": "m"}},
    }

    async def ok_probe(cfg: Config) -> None:
        assert cfg.mlc.chat_model == "m"

    with mock.patch.object(routes_settings, "probe_all", ok_probe):
        resp = client.post("/api/settings/test", json=body)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    async def fail_probe(cfg: Config) -> None:
        raise ProbeError("chat probe failed: connection refused")

    with mock.patch.object(routes_settings, "probe_all", fail_probe):
        resp = client.post("/api/settings/test", json=body)
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": False,
        "error": "chat probe failed: connection refused",
        "failed": "chat",
    }

    async def slow_probe(cfg: Config) -> None:
        await asyncio.sleep(1.0)

    monkeypatch.setattr(routes_settings, "SETTINGS_TEST_TIMEOUT_S", 0.01)
    with mock.patch.object(routes_settings, "probe_all", slow_probe):
        resp = client.post("/api/settings/test", json=body)
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": False,
        "error": "The test did not finish within 30 s.",
    }


def test_models_endpoint_with_mock_transport(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "empty.test":
            return httpx.Response(200, json={"data": []})
        if request.url.host == "down.test":
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200, json={"data": [{"id": "qwen3-8b"}, {"id": "nomic-embed"}]}
        )

    transport = httpx.MockTransport(handler)
    real_list_models = routes_settings.discovery.list_models

    async def mocked_list(
        backend_key: str, url: str, *, timeout_s: float = 5.0
    ) -> list[str]:
        return await real_list_models(
            backend_key, url, timeout_s=timeout_s, transport=transport
        )

    monkeypatch.setattr(routes_settings, "list_models", mocked_list)

    ok = client.get("/api/models?backend=mlc&url=http://ok.test:8000/v1")
    assert ok.status_code == 200
    # A listed embedding model is the offer "Remember across chats with <model>?".
    assert ok.json() == {
        "models": ["qwen3-8b", "nomic-embed"],
        "embedding_models": ["nomic-embed"],
        "suggested_embedding": "nomic-embed",
    }

    # Without parameters, GET /api/models uses the saved chat backend and URL,
    # and patching discovery.list_models directly is honored too.
    config.save(
        Config(
            chat_backend="mlc",
            embedding_backend="none",
            mlc=LocalBackendConfig(
                chat_url="http://ok.test:8000/v1", chat_model="qwen3-8b"
            ),
        )
    )
    monkeypatch.setattr(
        routes_settings, "list_models", routes_settings._ORIG_LIST_MODELS
    )
    monkeypatch.setattr(routes_settings.discovery, "list_models", mocked_list)
    default_ok = client.get("/api/models")
    assert default_ok.status_code == 200
    assert default_ok.json()["models"] == ["qwen3-8b", "nomic-embed"]

    empty = client.get("/api/models?backend=mlc&url=http://empty.test:8000/v1")
    assert empty.status_code == 502
    assert empty.json()["error"] == {
        "code": "backend_unreachable",
        "message": "No models found at http://empty.test:8000/v1.",
        "details": {},
    }

    down = client.get("/api/models?backend=mlc&url=http://down.test:8000/v1")
    assert down.status_code == 502
    assert down.json()["error"]["code"] == "backend_unreachable"
    assert down.json()["error"]["message"].startswith(
        "Could not list models at http://down.test:8000/v1:"
    )

    cloud = client.get("/api/models?backend=openai")
    assert cloud.status_code == 400
    assert cloud.json()["error"] == {
        "code": "invalid_request",
        "message": "Listing models works for local backends only; type the model name.",
        "details": {},
    }


@pytest.mark.parametrize(
    "url", ["ftp://host.test/v1", "//host.test/v1", "host.test", "http:///missing"]
)
def test_probe_routes_reject_non_http_urls(client: TestClient, url: str) -> None:
    with (
        mock.patch.object(routes_settings, "list_models") as listing,
        mock.patch.object(routes_settings, "probe_all") as probe,
    ):
        models = client.get("/api/models", params={"backend": "mlc", "url": url})
        settings = client.post(
            "/api/settings/test",
            json={
                "chat_backend": "mlc",
                "embedding_backend": "none",
                "sections": {"mlc": {"chat_url": url, "chat_model": "m"}},
            },
        )
    assert models.status_code == settings.status_code == 400
    listing.assert_not_called()
    probe.assert_not_called()


@pytest.mark.parametrize("field", ["chat_url", "embedding_url", "base_url"])
def test_settings_rejects_each_active_probe_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    local = {
        "chat_url": "http://chat.test",
        "chat_model": "m",
        "embedding_url": "http://embed.test",
        "embedding_model": "e",
    }
    cloud = {
        "base_url": "https://cloud.test",
        "api_key_env": "TEST_KEY",
        "chat_model": "m",
    }
    (cloud if field == "base_url" else local)[field] = "file:///tmp/bad"
    body = {
        "chat_backend": "openai" if field == "base_url" else "mlc",
        "embedding_backend": "mlc" if field == "embedding_url" else "none",
        "sections": {"mlc": local, "openai": cloud},
    }
    with mock.patch.object(routes_settings, "probe_all") as probe:
        response = client.post("/api/settings/test", json=body)
    assert response.status_code == 400
    probe.assert_not_called()


@pytest.mark.parametrize(
    "backend,embedding",
    [("mlc", False), ("ollama", False), ("mlc", True), ("openai", False)],
)
def test_settings_probe_never_follows_redirects(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, backend: str, embedding: bool
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if embedding and request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )
        return httpx.Response(307, headers={"location": "http://other.test/target"})

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("TEST_KEY", "secret")
    local = {
        "chat_url": "http://chat.test/v1",
        "chat_model": "m",
        "embedding_url": "https://embed.test/v1",
        "embedding_model": "e",
    }
    cloud = {
        "base_url": "https://cloud.test/v1",
        "api_key_env": "TEST_KEY",
        "chat_model": "m",
    }
    response = client.post(
        "/api/settings/test",
        json={
            "chat_backend": backend,
            "embedding_backend": "mlc" if embedding else "none",
            "sections": {"mlc": local, "ollama": local, "openai": cloud},
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert len(seen) == (2 if embedding else 1)
    assert all("other.test" not in url for url in seen)


def test_models_probe_never_follows_redirect(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://other.test/models"})

    async def listing(backend: str, url: str, *, timeout_s: float) -> list[str]:
        return await routes_settings.discovery.list_models(
            backend, url, timeout_s=timeout_s, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(routes_settings, "list_models", listing)
    response = client.get(
        "/api/models", params={"backend": "mlc", "url": "https://typed.test"}
    )
    assert response.status_code == 502
    assert len(seen) == 1 and "typed.test" in seen[0]


async def test_put_settings_resets_model_server_probe(
    tmp_path: Path, isolated_config: Path
) -> None:
    from fastapi import FastAPI

    from localmelo.support.gateway.console import ConsoleOptions

    from .conftest import FakeBuilder

    calls: list[tuple[str, str]] = []

    async def fake_probe_fn(backend: str, url: str) -> tuple[bool, str]:
        calls.append((backend, url))
        return True, ""

    cfg = Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m1"),
    )
    config.save(cfg)
    probe = ModelServerProbe(cfg, probe=fake_probe_fn)
    probe._result = (True, "", 1000.0)
    c = Console(
        cfg,
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "s.db"),
            builder=FakeBuilder(),
            probe=probe,
        ),
    )
    await c.start()
    try:
        app = FastAPI()
        c.mount(app)
        tc = TestClient(app, base_url="http://127.0.0.1:8401")
        rev = tc.get("/api/settings").json()["revision"]
        resp = tc.put(
            "/api/settings",
            json={
                "chat_backend": "vllm",
                "embedding_backend": "none",
                "workspace": "~/ws",
                "sections": {
                    "vllm": {
                        "chat_url": "http://127.0.0.1:10/v1",
                        "chat_model": "m2",
                    }
                },
                "revision": rev,
            },
        )
        assert resp.status_code == 200
        assert probe._result is None
    finally:
        await c.aclose()


def test_unconfigured_gateway_server_boots_and_onboards_over_api(
    tmp_path: Path, isolated_config: Path
) -> None:
    from localmelo.support.gateway import GatewayServer, start_gateway
    from localmelo.support.gateway.console import ConsoleOptions

    from .conftest import FakeBuilder, FakeProbe

    empty_cfg = Config(
        chat_backend="",
        embedding_backend="",
        scheduler=SchedulerConfig(call_log=False),
    )

    # 1. start_gateway(empty_cfg, allow_unconfigured=True) runs GatewayServer.run()
    # without raising ConfigError, whereas allow_unconfigured=False raises.
    with pytest.raises(config.ConfigError):
        start_gateway(empty_cfg)
    with mock.patch.object(GatewayServer, "run") as mock_run:
        start_gateway(empty_cfg, allow_unconfigured=True)
    mock_run.assert_called_once()

    # 2. A real GatewayServer booted with an empty config serves /api/health
    # (configured: false) and /api/settings, accepts an onboarding PUT, and
    # then reports configured: true.
    server = GatewayServer(
        empty_cfg,
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "gw.db"),
            builder=FakeBuilder(),
            probe=FakeProbe(),
        ),
    )
    with TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    ) as tc:
        h1 = tc.get("/api/health").json()
        assert h1["configured"] is False
        assert h1["config_errors"]

        s1 = tc.get("/api/settings").json()
        assert s1["chat_backend"] == ""
        assert s1["revision"] == ""

        put = tc.put(
            "/api/settings",
            json={
                "chat_backend": "mlc",
                "embedding_backend": "none",
                "workspace": str(tmp_path / "onboard-ws"),
                "sections": {
                    "mlc": {
                        "chat_url": "http://127.0.0.1:9/v1",
                        "chat_model": "stub-model",
                    }
                },
                "revision": "",
            },
        )
        assert put.status_code == 200
        assert put.json()["errors"] == []

        h2 = tc.get("/api/health").json()
        assert h2["configured"] is True
        assert h2["config_errors"] == []
        assert h2["model"]["name"] == "stub-model"


# ── Developer mode (#186) ──


def _put_body(revision: str, **extra: Any) -> dict[str, Any]:
    return {
        "chat_backend": "mlc",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "sections": {},
        "revision": revision,
        **extra,
    }


def _saved_cfg() -> None:
    config.save(
        Config(
            chat_backend="mlc",
            embedding_backend="none",
            mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
        )
    )


def test_fresh_config_reads_default_mode_with_the_tiered_options(
    client: TestClient,
) -> None:
    body = client.get("/api/settings").json()
    assert body["mode"] == "default"
    tiers = {row["option"]: row for row in body["advanced_options"]}
    assert tiers["--host"] == {"option": "--host", "tier": "advanced", "issue": ""}
    assert tiers["--retriever"]["tier"] == "experimental"
    assert tiers["--retriever"]["issue"] == "#4"
    assert "--base-url" not in tiers  # the user tier is not an advanced option


def test_mode_round_trips_through_put_and_the_cli_config(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    first = client.get("/api/settings").json()
    resp = client.put(
        "/api/settings", json=_put_body(first["revision"], mode="developer")
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "developer"
    assert resp.json()["revision"] != first["revision"]
    assert config.load().ui.mode == "developer"  # what the CLI reads
    assert 'mode = "developer"' in isolated_config.read_text()
    # An absent mode keeps the saved one; "default" removes the table again.
    kept = client.put("/api/settings", json=_put_body(resp.json()["revision"]))
    assert kept.json()["mode"] == "developer"
    off = client.put(
        "/api/settings", json=_put_body(kept.json()["revision"], mode="default")
    )
    assert off.json()["mode"] == "default"
    assert "[ui]" not in isolated_config.read_text()


def test_saving_settings_never_writes_an_x_setting(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    # Streaming is on by default: the override must be off to differ from it.
    config.set_overrides({"cli.stream": False, "ui.mode": "developer"})
    first = client.get("/api/settings").json()
    assert first["mode"] == "developer"  # this process runs with the setting
    resp = client.put("/api/settings", json=_put_body(first["revision"]))
    assert resp.status_code == 200
    text = isolated_config.read_text()
    assert "stream" not in text and "[cli]" not in text and "[ui]" not in text
    # A value the page changes is the page's.
    resp = client.put(
        "/api/settings", json=_put_body(resp.json()["revision"], mode="default")
    )
    assert resp.status_code == 200 and "[ui]" not in isolated_config.read_text()
    config.set_overrides({})
    assert config.load().ui.mode == "default" and config.load().cli.stream is True


def test_saving_without_a_mode_does_not_write_a_ui_table(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    revision = client.get("/api/settings").json()["revision"]
    assert client.put("/api/settings", json=_put_body(revision)).status_code == 200
    assert "[ui]" not in isolated_config.read_text()


def test_mode_put_on_a_stale_revision_conflicts_and_writes_nothing(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    before = isolated_config.read_text()
    resp = client.put(
        "/api/settings", json=_put_body("0000000000000000", mode="developer")
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "settings_changed"
    assert isolated_config.read_text() == before


def test_an_unknown_mode_is_rejected(client: TestClient) -> None:
    _saved_cfg()
    revision = client.get("/api/settings").json()["revision"]
    resp = client.put("/api/settings", json=_put_body(revision, mode="expert"))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_a_hand_written_mode_wins_and_a_bad_one_reads_default(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    isolated_config.write_text(
        isolated_config.read_text() + '\n[ui]\nmode = "developer"\n'
    )
    body = client.get("/api/settings").json()
    assert body["mode"] == "developer"
    isolated_config.write_text(
        isolated_config.read_text().replace("developer", "expert")
    )
    stale = client.get("/api/settings").json()
    assert stale["mode"] == "default"
    # Saving from the console repairs it instead of failing validation.
    resp = client.put("/api/settings", json=_put_body(stale["revision"]))
    assert resp.status_code == 200
    assert "expert" not in isolated_config.read_text()


# ── The spelling of an oMLX server ──


def test_a_server_a_listing_named_omlx_is_saved_as_omlx_until_the_url_changes(
    client: TestClient, isolated_config: Path
) -> None:
    import tomllib

    omlx = "http://127.0.0.1:18001/v1"
    other = "http://127.0.0.1:18002/v1"
    reply = httpx.Response(200, json={"data": [{"id": "m", "owned_by": "omlx"}]})
    transport = httpx.MockTransport(lambda request: reply)
    asyncio.run(
        routes_settings.discovery.list_models("vllm", omlx, transport=transport)
    )

    def put(url: str, revision: str) -> dict[str, Any]:
        section = {"vllm": {"chat_url": url, "chat_model": "m"}}
        body = _put_body(revision, chat_backend="vllm", sections=section)
        resp = client.put("/api/settings", json=body)
        assert resp.status_code == 200
        saved: dict[str, Any] = resp.json()
        return saved

    saved = put(omlx, "")
    assert tomllib.loads(isolated_config.read_text())["chat_backend"] == "omlx"
    assert saved["chat_backend"] == "vllm", "the API keeps one key for the backend"
    put(other, saved["revision"])
    assert tomllib.loads(isolated_config.read_text())["chat_backend"] == "vllm"


def test_a_failed_test_names_the_model_that_did_not_answer(
    client: TestClient,
) -> None:
    body = {
        "chat_backend": "mlc",
        "embedding_backend": "mlc",
        "workspace": "~/ws",
        "sections": {
            "mlc": {
                "chat_url": "http://127.0.0.1:9/v1",
                "chat_model": "m",
                "embedding_url": "http://127.0.0.1:9/v1",
                "embedding_model": "e",
            }
        },
    }
    for message, part in (
        ("embedding probe failed: [mlc] refused", "embedding"),
        ("embedding probe returned an empty vector", "embedding"),
        ("chat probe failed: refused", "chat"),
    ):

        async def fail(cfg: Config, message: str = message) -> None:
            raise ProbeError(message)

        with mock.patch.object(routes_settings, "probe_all", fail):
            answer = client.post("/api/settings/test", json=body).json()
        assert answer == {"ok": False, "error": message, "failed": part}

    async def other(cfg: Config) -> None:
        raise ValueError("embedding probe in a message that is not a probe's")

    with mock.patch.object(routes_settings, "probe_all", other):
        answer = client.post("/api/settings/test", json=body).json()
    assert answer["failed"] == "chat"


def test_the_settings_carry_how_memory_finds_notes_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    loaded = client.get("/api/settings").json()
    assert loaded["memory_retriever"] == "auto"
    words = client.put(
        "/api/settings", json=_put_body(loaded["revision"], memory_retriever="bm25")
    )
    assert words.status_code == 200 and words.json()["memory_retriever"] == "bm25"
    assert config.load().memory.retriever == "bm25"
    assert 'retriever = "bm25"' in isolated_config.read_text()
    kept = client.put("/api/settings", json=_put_body(words.json()["revision"]))
    assert kept.json()["memory_retriever"] == "bm25", "absent leaves it as it is"
    unknown = client.put(
        "/api/settings",
        json=_put_body(kept.json()["revision"], memory_retriever="vectors"),
    )
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "invalid_request"
    # Meaning needs an embedding model: the file's own check refuses it, and writes nothing.
    meaning = client.put(
        "/api/settings",
        json=_put_body(kept.json()["revision"], memory_retriever="embedding"),
    )
    assert meaning.status_code == 400
    assert meaning.json()["error"]["code"] == "invalid_settings"
    assert config.load().memory.retriever == "bm25"


def test_the_settings_carry_the_console_language_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    loaded = client.get("/api/settings").json()
    assert loaded["language"] == "auto"
    chinese = client.put(
        "/api/settings", json=_put_body(loaded["revision"], language="zh-Hans")
    )
    assert chinese.status_code == 200 and chinese.json()["language"] == "zh-Hans"
    assert config.load().console.language == "zh-Hans"
    assert 'language = "zh-Hans"' in isolated_config.read_text()
    kept = client.put("/api/settings", json=_put_body(chinese.json()["revision"]))
    assert kept.json()["language"] == "zh-Hans", "absent leaves it as it is"
    traditional = client.put(
        "/api/settings", json=_put_body(kept.json()["revision"], language="zh-Hant")
    )
    assert traditional.status_code == 400
    assert traditional.json()["error"]["code"] == "invalid_request"
    assert config.load().console.language == "zh-Hans"


def test_the_settings_carry_how_new_chats_start_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    _saved_cfg()
    loaded = client.get("/api/settings").json()
    assert loaded["approvals"] == "ask"
    auto = client.put(
        "/api/settings", json=_put_body(loaded["revision"], approvals="auto")
    )
    assert auto.status_code == 200 and auto.json()["approvals"] == "auto"
    assert config.load().approvals.mode == "auto"
    assert '[approvals]\nmode = "auto"' in isolated_config.read_text()
    kept = client.put("/api/settings", json=_put_body(auto.json()["revision"]))
    assert kept.json()["approvals"] == "auto", "absent leaves it as it is"
    yolo = client.put(
        "/api/settings", json=_put_body(kept.json()["revision"], approvals="yolo")
    )
    assert yolo.status_code == 400
    assert yolo.json()["error"]["code"] == "invalid_request"
    assert config.load().approvals.mode == "auto"


def test_the_settings_carry_whether_melo_may_browse_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    """``/browser on`` and ``/browser off`` in the console save ``[browser] driver``."""
    _saved_cfg()
    loaded = client.get("/api/settings").json()
    assert loaded["browser"] is False
    on = client.put("/api/settings", json=_put_body(loaded["revision"], browser=True))
    assert on.status_code == 200 and on.json()["browser"] is True
    assert config.load().browser.driver == "safari-automation"
    assert 'driver = "safari-automation"' in isolated_config.read_text()
    kept = client.put("/api/settings", json=_put_body(on.json()["revision"]))
    assert kept.json()["browser"] is True, "absent leaves it as it is"
    off = client.put(
        "/api/settings", json=_put_body(kept.json()["revision"], browser=False)
    )
    assert off.status_code == 200 and off.json()["browser"] is False
    assert config.load().browser.driver == "off"
    word = client.put(
        "/api/settings", json=_put_body(off.json()["revision"], browser="on")
    )
    assert word.status_code == 400, "only true or false"


def test_the_settings_carry_whether_melo_reads_web_pages_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    """The Settings switch saves ``[tools] web_fetch``; on by default, so off is written."""
    _saved_cfg()
    loaded = client.get("/api/settings").json()
    assert loaded["web_fetch"] is True
    off = client.put(
        "/api/settings", json=_put_body(loaded["revision"], web_fetch=False)
    )
    assert off.status_code == 200 and off.json()["web_fetch"] is False
    assert config.load().tools.web_fetch is False
    assert "[tools]\nweb_fetch = false" in isolated_config.read_text()
    kept = client.put("/api/settings", json=_put_body(off.json()["revision"]))
    assert kept.json()["web_fetch"] is False, "absent leaves it as it is"
    on = client.put(
        "/api/settings", json=_put_body(kept.json()["revision"], web_fetch=True)
    )
    assert on.status_code == 200 and on.json()["web_fetch"] is True
    assert "[tools]" not in isolated_config.read_text(), "the default is not written"
    word = client.put(
        "/api/settings", json=_put_body(on.json()["revision"], web_fetch="off")
    )
    assert word.status_code == 400


def test_the_settings_carry_the_follow_up_switch_and_save_it_only_when_sent(
    client: TestClient, isolated_config: Path
) -> None:
    """``[tasks] follow_up`` (#411): on by default, so off is written; a developer key."""
    _saved_cfg()
    loaded = client.get("/api/settings").json()
    assert loaded["follow_up"] is True
    keys = {
        key["key"]: group["section"]
        for group in loaded["developer_settings"]
        for key in group["keys"]
    }
    assert keys["tasks.follow_up"] == "tasks"
    off = client.put(
        "/api/settings", json=_put_body(loaded["revision"], follow_up=False)
    )
    assert off.status_code == 200 and off.json()["follow_up"] is False
    assert config.load().tasks.follow_up is False
    assert "[tasks]\nfollow_up = false" in isolated_config.read_text()
    kept = client.put("/api/settings", json=_put_body(off.json()["revision"]))
    assert kept.json()["follow_up"] is False, "absent leaves it as it is"
    on = client.put(
        "/api/settings", json=_put_body(kept.json()["revision"], follow_up=True)
    )
    assert on.status_code == 200 and on.json()["follow_up"] is True
    assert "[tasks]" not in isolated_config.read_text(), "the default is not written"
    word = client.put(
        "/api/settings", json=_put_body(on.json()["revision"], follow_up="off")
    )
    assert word.status_code == 400
    bad = config.Config()
    bad.tasks.follow_up = "no"  # type: ignore[assignment]
    assert any("[tasks] follow_up must be a bool" in e for e in bad.setting_errors())


def test_the_developer_section_lists_every_key_by_section_with_its_value(
    client: TestClient,
) -> None:
    from localmelo.support import settings as setting_keys
    from localmelo.support.option_tiers import OPTION_ROWS, SHORT_ROWS

    config.save(
        Config(
            chat_backend="mlc",
            embedding_backend="none",
            mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
            scheduler=SchedulerConfig(n_srv=4),
        )
    )
    body = client.get("/api/settings").json()
    groups = body["developer_settings"]
    listed = [row["key"] for group in groups for row in group["keys"]]
    assert listed == list(setting_keys.catalog()), "every key, in the help's order"
    sections = [group["section"] for group in groups]
    assert sections == [s for s in setting_keys.SECTIONS if s in sections]
    rows = {row["key"]: row for group in groups for row in group["keys"]}
    assert set(rows["agent.reasoning"]) == {
        "key",
        "value",
        "default",
        "flag",
        "issue",
        "file",
    }
    assert rows["scheduler.n_srv"]["value"] == "4"
    assert rows["scheduler.n_srv"]["default"] == "8"
    assert rows["agent.reasoning"]["flag"] == "--reasoning"
    assert rows["agent.reasoning"]["issue"] == "#183"
    assert rows["memory.retriever"]["issue"] == "#299"
    assert rows["agent.attempt_policy"]["file"] is False
    assert rows["agent.attempt_policy"]["value"] == "reflect"
    assert body["run_options"] == [
        {"option": f"{names} {meta}".strip(), "text": text}
        for names, meta, text in OPTION_ROWS[SHORT_ROWS:]
    ]
