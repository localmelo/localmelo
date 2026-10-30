"""The CLI tests never probe a model server on this computer.

The fixtures in ``conftest.py`` turn off the setup wizard's server detection
and its model listings (a developer's machine may have Ollama on 11434 or a
vLLM or oMLX server on 8000). A test that wants either one fakes it.
"""

from __future__ import annotations

import httpx
import pytest

from localmelo.support.backends import discovery

from .conftest import REAL_LIST_MODELS


def test_detection_probes_nothing_by_default() -> None:
    assert discovery.detect_urls() == ()


def test_a_test_can_still_name_its_own_detection_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(discovery.DETECT_URLS_ENV, "http://models.test/v1")
    assert discovery.detect_urls() == ("http://models.test/v1",)


async def test_a_listing_without_a_fake_transport_finds_no_server() -> None:
    assert discovery.list_models is not REAL_LIST_MODELS  # the fixture's wrapper
    with pytest.raises(discovery.DiscoveryError, match="connection refused"):
        await discovery.list_models("ollama", "http://models.test:11434")


async def test_a_listing_with_a_fake_transport_still_works() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "m"}]})
    )
    found = await discovery.list_models(
        "vllm", "http://models.test/v1", transport=transport
    )
    assert found == ["m"]
