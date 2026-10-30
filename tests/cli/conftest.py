"""Fixtures for the CLI tests."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from localmelo.support.backends import discovery

# Taken at import, before any fixture replaces it.
REAL_LIST_MODELS = discovery.list_models


@pytest.fixture(autouse=True)
def _no_status_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never ask a real server which models are loaded.

    A choice between models asks ``GET <url>/models/status``; a test that
    cares fakes ``discovery.loaded_models`` itself, or passes a transport.
    """

    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels()

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)


@pytest.fixture(autouse=True)
def _no_server_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never probe this computer's default model ports.

    The setup wizard looks for a running model server on ports 8000, 8400,
    11434 and 30000, which are real servers on a developer's machine. An empty
    ``LOCALMELO_DETECT_URLS`` turns that off, so a test that runs the wizard
    probes nothing. A test of detection sets its own value (a fake server's
    URL) or fakes ``discovery.detect_local_server``.
    """
    monkeypatch.setenv(discovery.DETECT_URLS_ENV, "")


@pytest.fixture(autouse=True)
def _no_real_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never list a real server's models.

    The wizard lists the chat server it was given (``GET <url>/models``), and
    a test may name a real default port such as 11434. A listing without a
    fake ``transport`` finds nothing there, as when the server is down; a test
    that fakes ``discovery.list_models`` itself replaces this.
    """

    async def list_models(backend_key: str, url: str, **kwargs: Any) -> list[str]:
        if kwargs.get("transport") is None:
            raise discovery.DiscoveryError("connection refused", reachable=False)
        return await REAL_LIST_MODELS(backend_key, url, **kwargs)

    monkeypatch.setattr(discovery, "list_models", list_models)


@pytest.fixture(autouse=True)
def _direct_model_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct run without ``--chat-model`` asks for the model ``default``.

    The real run lists ``GET <url>/models`` first and chooses from the answer,
    so a test that builds one with only ``--base-url`` gets ``default`` here,
    as it did before the listing. A module that tests the choice itself sets
    ``LISTS_MODELS = True`` and serves a fake (``test_direct_model_choice.py``).
    """
    if getattr(request.module, "LISTS_MODELS", False):
        return

    def default_model(args: argparse.Namespace) -> str:
        return "default"

    monkeypatch.setattr("localmelo.__main__._direct_model", default_model)
