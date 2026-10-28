"""The real cached model-server probe behind ``GET /api/health``."""

from __future__ import annotations

import asyncio

from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    GatewayConfig,
    LocalBackendConfig,
)
from localmelo.support.gateway.console.probe import ModelServerProbe


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def local_cfg(url: str = "http://user:pass@127.0.0.1:9/v1") -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=url, chat_model="m"),
        gateway=GatewayConfig(port=8401, host="127.0.0.1"),
    )


async def _settle(probe: ModelServerProbe) -> None:
    for _ in range(100):
        if probe._task is None:  # type: ignore[attr-defined]
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the probe did not finish")


async def test_a_cloud_backend_has_no_model_server() -> None:
    cfg = Config(
        chat_backend="openai",
        embedding_backend="none",
        openai=CloudVendorConfig(chat_model="gpt"),
    )
    probe = ModelServerProbe(cfg)
    assert probe.model_server() is None
    await probe.aclose()


def test_a_malformed_or_relative_url_has_no_label() -> None:
    from localmelo.support.gateway.console.probe import url_label

    assert url_label("http://user:pass@127.0.0.1:9/v1") == "http://127.0.0.1:9/v1"
    assert url_label("http://host:bad/v1") == ""
    assert url_label("user:secret@host:9/v1") == ""
    assert url_label("not a url") == ""


async def test_a_local_backend_without_a_url_is_unreachable() -> None:
    probe = ModelServerProbe(local_cfg(""))
    assert probe.model_server() == {
        "url_label": "",
        "reachable": False,
        "error": "No chat URL is configured.",
        "checked_s_ago": 0,
    }
    await probe.aclose()


async def test_the_first_read_is_null_and_starts_one_probe() -> None:
    calls: list[tuple[str, str]] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake(backend: str, url: str) -> tuple[bool, str]:
        calls.append((backend, url))
        started.set()
        await release.wait()
        return True, ""

    clock = Clock()
    probe = ModelServerProbe(local_cfg(), probe=fake, clock=clock)
    state = probe.model_server()
    assert state is not None
    assert state["reachable"] is None
    assert state["url_label"] == "http://127.0.0.1:9/v1"
    await started.wait()
    assert calls == [("vllm", "http://user:pass@127.0.0.1:9/v1")]
    assert probe.model_server()["reachable"] is None  # type: ignore[index]
    assert len(calls) == 1
    release.set()
    await _settle(probe)
    state = probe.model_server()
    assert state is not None
    assert state["reachable"] is True
    assert state["error"] == ""
    assert state["checked_s_ago"] == 0
    await probe.aclose()


async def test_an_unreachable_result_is_reread_after_five_seconds() -> None:
    calls = 0

    async def fake(backend: str, url: str) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return False, "connection refused"

    clock = Clock()
    probe = ModelServerProbe(local_cfg(), probe=fake, clock=clock)
    probe.model_server()
    await _settle(probe)
    assert calls == 1
    clock.now += 4
    assert probe.model_server()["reachable"] is False  # type: ignore[index]
    assert calls == 1
    clock.now += 2
    probe.model_server()
    await _settle(probe)
    assert calls == 2
    await probe.aclose()


async def test_a_reachable_result_is_reread_after_thirty_seconds() -> None:
    calls = 0

    async def fake(backend: str, url: str) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return True, ""

    clock = Clock()
    probe = ModelServerProbe(local_cfg(), probe=fake, clock=clock)
    probe.model_server()
    await _settle(probe)
    clock.now += 29
    probe.model_server()
    assert calls == 1
    clock.now += 2
    probe.model_server()
    await _settle(probe)
    assert calls == 2
    await probe.aclose()


async def test_a_close_cancels_a_probe_in_flight() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def fake(backend: str, url: str) -> tuple[bool, str]:
        started.set()
        await never.wait()
        return True, ""

    probe = ModelServerProbe(local_cfg(), probe=fake)
    assert probe.model_server()["reachable"] is None  # type: ignore[index]
    await started.wait()
    await probe.aclose()
    assert probe._task is None  # type: ignore[attr-defined]
    assert probe.model_server()["reachable"] is None  # type: ignore[index]
