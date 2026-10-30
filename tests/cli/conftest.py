"""Fixtures for the CLI tests."""

from __future__ import annotations

import pytest

from localmelo.support.backends import discovery


@pytest.fixture(autouse=True)
def _no_status_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never ask a real server which models are loaded.

    A choice between models asks ``GET <url>/models/status``; a test that
    cares fakes ``discovery.loaded_models`` itself, or passes a transport.
    """

    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels()

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
