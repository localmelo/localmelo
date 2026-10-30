"""``GET /api/setup/detect``: the model server on this Mac, for the welcome's card.

Every request goes through an injected ``httpx.MockTransport`` and
``LOCALMELO_DETECT_URLS`` names URLs of its own, so no test reaches a real port.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from localmelo.support.backends import discovery
from localmelo.support.gateway.console import Console

URL = "http://127.0.0.1:18001/v1"
OLLAMA = {
    "http://127.0.0.1:18001/api/version": httpx.Response(200, json={"version": "1"})
}


def listing(*ids: str, owned_by: str = "") -> httpx.Response:
    data = [{"id": i, **({"owned_by": owned_by} if owned_by else {})} for i in ids]
    return httpx.Response(200, json={"data": data})


def detect(
    client: TestClient,
    console: Console,
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, httpx.Response],
    urls: tuple[str, ...] = (URL,),
) -> tuple[dict[str, Any], list[str]]:
    """The route's body, and the URLs it asked; what is not in *answers* refuses."""
    monkeypatch.setenv(discovery.DETECT_URLS_ENV, ",".join(urls))
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        if str(request.url) not in answers:
            raise httpx.ConnectError("connection refused")
        return answers[str(request.url)]

    transport = httpx.MockTransport(handler)
    console.options = replace(console.options, detect_transport=transport)
    response = client.get("/api/setup/detect")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return response.json(), asked


# answers -> (state, name, backend, url, models, loaded, suggested)
STATUS = {"models": [{"id": "a-32b", "loaded": False}, {"id": "b-8b", "loaded": True}]}
CASES = {
    "ollama names itself, without the embedding model": (
        {f"{URL}/models": listing("qwen3:8b", "nomic-embed-text"), **OLLAMA},
        ("found", "Ollama", "ollama", URL, ["qwen3:8b"], [], "qwen3:8b"),
    ),
    "omlx is named and its loaded model suggested": (
        {
            f"{URL}/models": listing("a-32b", "b-8b", "c-4b", owned_by="omlx"),
            f"{URL}/models/status": httpx.Response(200, json=STATUS),
        },
        ("found", "oMLX", "vllm", URL, ["a-32b", "b-8b", "c-4b"], ["b-8b"], "b-8b"),
    ),
    "several models and none known loaded leave the choice to the user": (
        {f"{URL}/models": listing("a-7b", "b-8b")},
        ("found", "", "vllm", URL, ["a-7b", "b-8b"], [], None),
    ),
    "a server that asks for a key": (
        {f"{URL}/models": httpx.Response(401)},
        ("needs_key", "", "", URL, [], [], None),
    ),
    "only embedding models": (
        {f"{URL}/models": listing("nomic-embed-text")},
        ("none", "", "", URL, [], [], None),
    ),
    "nothing answering": ({}, ("none", "", "", "", [], [], None)),
}


@pytest.mark.parametrize("case", CASES)
def test_detect(
    case: str, client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers, expected = CASES[case]
    body, _ = detect(client, console, monkeypatch, answers)
    keys = ("state", "name", "backend", "url", "models", "loaded", "suggested")
    assert tuple(body[key] for key in keys) == expected


def test_a_server_that_cannot_be_used_says_why(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = detect(
        client, console, monkeypatch, {f"{URL}/models": httpx.Response(403)}
    )
    assert discovery.NO_API_KEY_HINT in key["problem"]
    embed, _ = detect(
        client, console, monkeypatch, {f"{URL}/models": listing("nomic-embed-text")}
    )
    assert "embedding" in embed["problem"]


def test_a_url_with_credentials_is_never_echoed(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "http://user:hunter2@127.0.0.1:18001/v1"
    answers = {f"{secret}/models": listing("a-7b")}
    body, _ = detect(client, console, monkeypatch, answers, (secret,))
    assert (body["state"], body["url"]) == ("found", URL)


def test_detection_turned_off_asks_nobody(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    body, asked = detect(
        client, console, monkeypatch, {f"{URL}/models": listing("a-7b")}, ()
    )
    assert (body["state"], asked) == ("none", [])
