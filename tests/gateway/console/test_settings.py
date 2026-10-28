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
    ["", "123ABC", "OPENAI-KEY", "KEY WITH SPACE", "A" * 129],
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
        ({"workspace": ""}, "invalid_request"),
        ({"workspace": "~/bad\nline"}, "invalid_request"),
        ({"chat_backend": "mlc\x00"}, "invalid_request"),
        (
            {"sections": {"mlc": {"chat_model": "bad\x07model"}}},
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
    assert ok.json() == {"models": ["qwen3-8b", "nomic-embed"]}

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


async def test_put_settings_resets_model_server_probe(
    tmp_path: Path, isolated_config: Path
) -> None:
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
    probe.reset(
        Config(
            chat_backend="vllm",
            embedding_backend="none",
            vllm=LocalBackendConfig(chat_url="http://127.0.0.1:10/v1", chat_model="m2"),
        )
    )
    assert probe._result is None
    await probe.aclose()
