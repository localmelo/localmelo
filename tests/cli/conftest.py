"""Fixtures for the CLI tests."""

from __future__ import annotations

import argparse

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


@pytest.fixture(autouse=True)
def _no_model_listing(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never list a real server's models for a direct run without ``--chat-model``.

    That run asks ``GET <url>/models`` before its first request, so a test
    that builds one with only ``--base-url`` gets the model ``default``, as it
    did before the listing. A module that tests the choice itself sets
    ``LISTS_MODELS = True`` and serves a fake (``test_direct_model_choice.py``).
    """
    if getattr(request.module, "LISTS_MODELS", False):
        return

    def default_model(args: argparse.Namespace) -> str:
        return "default"

    monkeypatch.setattr("localmelo.__main__._direct_model", default_model)
