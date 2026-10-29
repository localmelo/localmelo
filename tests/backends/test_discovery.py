"""Listing a server's models and finding a local server (support/backends/discovery.py).

Every request goes to an httpx.MockTransport, so nothing reaches this
machine's model servers.
"""

from __future__ import annotations

import json

import httpx
import pytest

from localmelo.support.backends import discovery
from localmelo.support.backends.discovery import (
    DiscoveryError,
    detect_local_server,
    list_models,
    parse_model_ids,
    pick_model,
)

URL = "http://127.0.0.1:8500/v1"


def _listing(*ids: str) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200, json={"object": "list", "data": [{"id": i} for i in ids]}
        )

    return httpx.MockTransport(handle)


def _status(code: int, body: bytes = b"{}") -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(code, content=body))


def _raising(exc: Exception) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    ("payload", "ids"),
    [
        ({"data": [{"id": "a"}, {"id": "b"}]}, ["a", "b"]),
        ({"models": [{"name": "a"}, {"model": "b"}]}, ["a", "b"]),
        ({"data": ["a", "a", {"id": " b "}, {"id": " "}, {}]}, ["a", "b"]),
        ({"data": "a"}, []),
        ([{"id": "a"}], []),
        (None, []),
    ],
)
def test_parse_model_ids_reads_both_list_shapes(
    payload: object, ids: list[str]
) -> None:
    assert parse_model_ids(payload) == ids


async def test_list_models_sends_one_get_to_the_models_path() -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url}")
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    models = await list_models("vllm", URL + "/", transport=httpx.MockTransport(handle))
    assert models == ["m"]
    assert seen == [f"GET {URL}/models"]


async def test_an_ollama_url_without_v1_lists_its_openai_compatible_models() -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": []})

    assert (
        await list_models(
            "ollama", "http://localhost:11434", transport=httpx.MockTransport(handle)
        )
        == []
    )
    assert seen == ["http://localhost:11434/v1/models"]


@pytest.mark.parametrize("key", ["openai", "anthropic", "gemini", "nvidia", "x"])
async def test_cloud_vendors_are_not_listed(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        await list_models(key, URL, transport=_listing("m"))


@pytest.mark.parametrize("code", [401, 403])
async def test_a_refusal_for_want_of_a_key_says_so(code: int) -> None:
    with pytest.raises(DiscoveryError) as exc:
        await list_models("openai-compatible", URL, transport=_status(code))
    assert str(exc.value) == (
        f"HTTP {code}: the server requires an API key; localmelo sends none to "
        'local servers (oMLX: "Allow inference without API key")'
    )
    assert exc.value.reachable and exc.value.status_code == code


async def test_another_error_status_is_named() -> None:
    with pytest.raises(DiscoveryError, match=r"^HTTP 404$") as exc:
        await list_models("vllm", URL, transport=_status(404))
    assert exc.value.reachable and exc.value.status_code == 404


async def test_a_reply_that_is_not_json_is_not_a_model_list() -> None:
    with pytest.raises(DiscoveryError, match="the reply is not a model list") as exc:
        await list_models("vllm", URL, transport=_status(200, b"<html>"))
    assert exc.value.reachable


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (httpx.ConnectError("[Errno 61] Connection refused"), "connection refused"),
        (httpx.ConnectError("All connection attempts failed"), "connection refused"),
        (httpx.ReadTimeout("timed out"), "no answer in 5 s"),
        (httpx.ConnectTimeout("timed out"), "no answer in 5 s"),
        (
            httpx.ConnectError("nodename nor servname"),
            "ConnectError: nodename nor servname",
        ),
    ],
)
async def test_no_answer_says_why(error: Exception, reason: str) -> None:
    with pytest.raises(DiscoveryError) as exc:
        await list_models("vllm", URL, transport=_raising(error))
    assert str(exc.value) == reason
    assert not exc.value.reachable and exc.value.status_code is None


async def test_a_value_that_is_not_a_url_is_named() -> None:
    with pytest.raises(DiscoveryError, match="not a URL: localhost:8000") as exc:
        await list_models("vllm", "localhost:8000")
    assert not exc.value.reachable


# ── Telling Ollama apart ──


def _version(version: object = "0.12.3") -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(200, json={"version": version})
    )


async def test_an_ollama_server_is_asked_for_its_version_at_the_root() -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"version": "0.12.3"})

    assert await discovery.is_ollama(URL, transport=httpx.MockTransport(handle))
    # The chat URL ends in /v1; the version endpoint does not.
    assert seen == ["http://127.0.0.1:8500/api/version"]


@pytest.mark.parametrize(
    ("url", "asked"),
    [
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/api/version"),
        ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/api/version"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/api/version"),
    ],
)
async def test_a_url_with_or_without_v1_is_asked_at_the_root(
    url: str, asked: str
) -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"version": "0.12.3"})

    assert await discovery.is_ollama(url, transport=httpx.MockTransport(handle))
    assert seen == [asked]


@pytest.mark.parametrize(
    "transport",
    [
        _status(404),
        _status(302),
        _status(200, b"<html>"),
        _listing(),
        _version(12),
        _version(""),
        _version(None),
        _raising(httpx.ReadTimeout("timed out")),
        _raising(httpx.ConnectError("Connection refused")),
    ],
    ids=[
        "404",
        "redirect",
        "not-json",
        "a-model-list",
        "a-number",
        "empty",
        "null",
        "timeout",
        "refused",
    ],
)
async def test_a_server_that_does_not_answer_like_ollama_is_not_one(
    transport: httpx.MockTransport,
) -> None:
    assert not await discovery.is_ollama(URL, transport=transport)


@pytest.mark.parametrize(
    ("models", "picked"),
    [
        (["a", "b"], "a"),
        (["text-embedding-3", "Qwen3-Reranker", "chat"], "chat"),
        (["bge-m3-EMBED", "x-rerank"], "bge-m3-EMBED"),
        ([], None),
    ],
)
def test_pick_model_skips_embedding_and_reranking_models(
    models: list[str], picked: str | None
) -> None:
    assert pick_model(models) == picked


@pytest.mark.parametrize(
    ("models", "found"),
    [
        (
            [
                "qwen3:8b",
                "qwen3-embedding:0.6b",
                "bge-m3",
                "bge-base-en-v1.5",
                "nomic-embed-text",
                "snowflake-arctic-embed-m",
                "mxbai-embed-large",
                "bge-reranker-v2-m3",
                "Qwen3-8B",
            ],
            [
                "qwen3-embedding:0.6b",
                "bge-m3",
                "bge-base-en-v1.5",
                "nomic-embed-text",
                "snowflake-arctic-embed-m",
                "mxbai-embed-large",
            ],
        ),
        ([], []),
        (
            ["chat", "e5-large", "gte-base", "chat", "e5-large"],
            ["e5-large", "gte-base"],
        ),
        (["bge-reranker-v2-m3", "Qwen3-Reranker"], []),
    ],
)
def test_embedding_models_names_the_embedding_ids_in_order(
    models: list[str], found: list[str]
) -> None:
    assert discovery.embedding_models(models) == found


def test_embedding_detection_is_additive_to_the_chat_rule() -> None:
    assert discovery.is_embedding_model("bge-m3")
    # The older rule calls this a chat model; unchanged, so --base-url's
    # model choice stays byte-identical.
    assert discovery.is_chat_model("bge-m3")
    assert not discovery.is_embedding_model("bge-reranker-v2-m3")
    assert discovery.is_embedding_model("Qwen3-Embedding-0.6B")


@pytest.mark.parametrize(
    ("models", "recommended"),
    [
        ([], None),
        (
            ["bge-m3", "nomic-embed-text", "Qwen3-Embedding-0.6B"],
            "Qwen3-Embedding-0.6B",
        ),
        (["bge-m3", "nomic-embed-text"], "nomic-embed-text"),
        (["bge-m3", "bge-base-en-v1.5"], "bge-base-en-v1.5"),
        (["e5-large", "bge-m3"], "bge-m3"),
        (["unknown-embed", "e5-large", "bge-m3", "gte-base"], "bge-m3"),
        (["unknown-embed", "another-embed"], "unknown-embed"),
    ],
)
def test_recommend_embedding_puts_the_named_families_first(
    models: list[str], recommended: str | None
) -> None:
    assert discovery.recommend_embedding(models) == recommended


async def test_detection_returns_the_first_url_in_order_that_lists_a_model() -> None:
    lists = {
        "http://127.0.0.1:1/v1/models": None,  # nothing listening
        "http://127.0.0.1:2/v1/models": [],  # answers, lists nothing
        "http://127.0.0.1:3/v1/models": ["late"],
        "http://127.0.0.1:4/v1/models": ["later"],
    }
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url}")
        ids = lists[str(request.url)]
        if ids is None:
            raise httpx.ConnectError("Connection refused")
        return httpx.Response(
            200, content=json.dumps({"data": [{"id": i} for i in ids]})
        )

    urls = [u.removesuffix("/models") for u in lists]
    found = await detect_local_server(urls, transport=httpx.MockTransport(handle))
    assert found == discovery.FoundServer(url="http://127.0.0.1:3/v1", models=["late"])
    assert sorted(seen) == sorted(f"GET {u}" for u in lists)


async def test_detection_finds_nothing_when_nothing_answers() -> None:
    refused = _raising(httpx.ConnectError("Connection refused"))
    assert (
        await detect_local_server(["http://127.0.0.1:1/v1"], transport=refused) is None
    )
    assert await detect_local_server([], transport=_listing("m")) is None


def test_detect_urls_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(discovery.DETECT_URLS_ENV, raising=False)
    assert discovery.detect_urls() == discovery.DETECT_URLS
    monkeypatch.setenv(discovery.DETECT_URLS_ENV, " http://a/v1 , ,http://b/v1")
    assert discovery.detect_urls() == ("http://a/v1", "http://b/v1")
    monkeypatch.setenv(discovery.DETECT_URLS_ENV, "")
    assert discovery.detect_urls() == ()


def test_the_default_detection_urls_are_the_local_servers_default_ports() -> None:
    assert discovery.DETECT_URLS == (
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8400/v1",
        "http://127.0.0.1:11434/v1",
        "http://127.0.0.1:30000/v1",
    )


@pytest.mark.parametrize("port", ["80000", "65536"])
async def test_a_port_out_of_range_is_not_a_url(port: str) -> None:
    url = f"http://127.0.0.1:{port}/v1"
    with pytest.raises(DiscoveryError) as exc:
        await list_models("vllm", url, transport=_listing("m"))
    assert str(exc.value) == f"not a URL: {url} (port {port})"
    assert not exc.value.reachable


async def test_an_error_group_from_the_connection_names_its_error() -> None:
    group = ExceptionGroup("unhandled errors", [OverflowError("port must be 0-65535")])
    with pytest.raises(DiscoveryError) as exc:
        await list_models("vllm", URL, transport=_raising(group))
    assert str(exc.value) == "OverflowError: port must be 0-65535"
    assert not exc.value.reachable


async def test_a_reply_that_cannot_be_decoded_is_named() -> None:
    with pytest.raises(DiscoveryError) as exc:
        await list_models(
            "vllm", URL, transport=_raising(httpx.DecodingError("bad gzip"))
        )
    assert str(exc.value) == "the reply could not be read (DecodingError: bad gzip)"
    assert exc.value.reachable


async def test_a_redirect_is_not_followed_and_names_its_target() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "https://host/v1/models"})

    with pytest.raises(DiscoveryError) as exc:
        await list_models("vllm", URL, transport=httpx.MockTransport(handle))
    assert str(exc.value) == "HTTP 307: redirected to https://host/v1/models"
    assert exc.value.status_code == 307
    with pytest.raises(DiscoveryError, match=r"^HTTP 300$"):
        await list_models("vllm", URL, transport=_status(300))


def _servers(answers: dict[str, object]) -> httpx.MockTransport:
    """A transport answering each URL's /models: a list of ids, a status, or refused."""

    def handle(request: httpx.Request) -> httpx.Response:
        answer = answers[str(request.url).removesuffix("/models")]
        if answer is None:
            raise httpx.ConnectError("Connection refused")
        if isinstance(answer, int):
            return httpx.Response(answer, content=b"{}")
        assert isinstance(answer, list)
        return httpx.Response(200, json={"data": [{"id": i} for i in answer]})

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    ("answers", "found"),
    [
        (  # a chat server wins over an earlier embedding-only one
            {"http://a/v1": ["bge-m3-embed"], "http://b/v1": ["qwen3"]},
            discovery.FoundServer(url="http://b/v1", models=["qwen3"]),
        ),
        (  # with no chat model anywhere, the first server that lists one
            {"http://a/v1": None, "http://b/v1": ["bge-m3-embed"]},
            discovery.FoundServer(url="http://b/v1", models=["bge-m3-embed"]),
        ),
        (  # a list beats an answer without one, whatever the order
            {"http://a/v1": 401, "http://b/v1": ["qwen3"]},
            discovery.FoundServer(url="http://b/v1", models=["qwen3"]),
        ),
        (  # no list anywhere: the first server that answered, and why
            {"http://a/v1": None, "http://b/v1": 401, "http://c/v1": []},
            discovery.FoundServer(
                url="http://b/v1",
                models=[],
                problem=f"HTTP 401: {discovery.NO_API_KEY_HINT}",
            ),
        ),
        (
            {"http://a/v1": None, "http://b/v1": []},
            discovery.FoundServer(
                url="http://b/v1", models=[], problem="it lists no models"
            ),
        ),
    ],
)
async def test_detection_prefers_a_chat_server_then_any_answer(
    answers: dict[str, object], found: discovery.FoundServer
) -> None:
    assert (
        await detect_local_server(list(answers), transport=_servers(answers)) == found
    )


# ── Which models are in memory ──


def _server(routes: dict[str, httpx.Response | Exception]) -> httpx.MockTransport:
    """Answers each path in *routes*; a path not in it is a 404."""
    asked: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert "authorization" not in request.headers
        asked.append(str(request.url))
        answer = routes.get(request.url.path, httpx.Response(404))
        if isinstance(answer, Exception):
            raise answer
        return answer

    transport = httpx.MockTransport(handle)
    transport.asked = asked  # type: ignore[attr-defined]
    return transport


_OMLX_STATUS = httpx.Response(
    200,
    json={
        "models": [
            {"id": "big", "loaded": False},
            {"id": "small", "loaded": True},
            {"id": "tiny", "loaded": False},
        ]
    },
)


async def test_omlx_status_names_the_loaded_models() -> None:
    transport = _server({"/v1/models/status": _OMLX_STATUS})
    assert await discovery.loaded_models(URL, transport=transport) == {"small"}


async def test_ollama_ps_names_the_running_models() -> None:
    ps = httpx.Response(
        200, json={"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]}
    )
    transport = _server({"/api/ps": ps})
    assert await discovery.loaded_models(URL, transport=transport) == {"qwen3:8b"}
    assert transport.asked == [  # type: ignore[attr-defined]
        "http://127.0.0.1:8500/v1/models/status",
        "http://127.0.0.1:8500/api/ps",
    ]


async def test_a_server_without_a_status_route_has_nothing_loaded() -> None:
    assert await discovery.loaded_models(URL, transport=_server({})) == set()


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(401, json={"detail": "Not authenticated"}),
        httpx.Response(500),
        httpx.Response(302, headers={"location": "http://elsewhere/"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["small"]),
        httpx.Response(200, json={"models": "small"}),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
        RuntimeError("anything"),
    ],
)
async def test_a_failed_status_request_is_ignored(
    answer: httpx.Response | Exception,
) -> None:
    transport = _server({"/v1/models/status": answer, "/api/ps": answer})
    assert await discovery.loaded_models(URL, transport=transport) == set()


async def test_one_failed_source_does_not_hide_the_other() -> None:
    transport = _server(
        {
            "/v1/models/status": httpx.ReadTimeout("slow"),
            "/api/ps": httpx.Response(200, json={"models": [{"name": "small"}]}),
        }
    )
    assert await discovery.loaded_models(URL, transport=transport) == {"small"}


async def test_a_url_without_v1_is_asked_at_both_places() -> None:
    transport = _server({"/v1/models/status": _OMLX_STATUS})
    found = await discovery.loaded_models("http://127.0.0.1:8500/", transport=transport)
    assert found == {"small"}


def test_pick_model_prefers_a_loaded_chat_model_in_server_order() -> None:
    models = ["big", "embed-x", "small", "tiny"]
    assert pick_model(models, {"tiny", "small"}) == "small"
    assert pick_model(models, {"embed-x"}) == "big"  # not a chat model
    assert pick_model(models, set()) == "big"
    assert pick_model(models, {"gone"}) == "big"
    assert pick_model(["embed-x"], {"embed-x"}) == "embed-x"
