"""Ask a local model server which models it serves, and find one.

The requests here are ``GET <url>/models``, and ``GET <root>/api/version``
to tell an Ollama server apart. A first run also asks which models are in
memory: ``GET <url>/models/status`` (oMLX) and ``GET <root>/api/ps``
(Ollama). Nothing is generated, so listing neither waits for a server to
load a model nor competes with runs on it. Cloud vendors are not listed.
The same listing tells an oMLX server (see :func:`is_omlx`), so no request is
added to name one.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from localmelo.support.backends.openai_compat import normalize_url

DISCOVERY_TIMEOUT_S: Final = 5.0

# Where a local server listens by default: vLLM and oMLX, MLC, Ollama, SGLang.
DETECT_URLS: Final = (
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8400/v1",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.1:30000/v1",
)
DETECT_TIMEOUT_S: Final = 1.0

# Replaces DETECT_URLS when set: comma-separated URLs; empty turns detection off.
DETECT_URLS_ENV: Final = "LOCALMELO_DETECT_URLS"

# The key for a URL given on the command line rather than a configured backend.
OPENAI_COMPATIBLE: Final = "openai-compatible"

_LISTED_BACKENDS: Final = frozenset(
    {"mlc", "ollama", "vllm", "sglang", OPENAI_COMPATIBLE}
)

NO_API_KEY_HINT: Final = (
    "the server requires an API key; localmelo sends none to local servers "
    '(oMLX: "Allow inference without API key")'
)

# What a label or an error calls a server of the OpenAI-compatible kind.
OMLX_NAME: Final = "oMLX"
OPENAI_COMPATIBLE_NAME: Final = "OpenAI-compatible server"

# The servers whose last model listing named oMLX as its owner, by their
# normalized URL. A listing is asked for anyway, so this costs no request.
_OMLX_URLS: Final[set[str]] = set()


class DiscoveryError(Exception):
    """A model list could not be read; the message says why.

    ``reachable`` is True when the server answered, but not with a model
    list. ``status_code`` is the HTTP status of such an answer, if any.
    """

    def __init__(
        self, message: str, *, reachable: bool, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.reachable = reachable
        self.status_code = status_code


@dataclass(frozen=True)
class FoundServer:
    """A server that answered detection.

    ``models`` is its list as listed. It is empty when the server answered
    without a usable list, and ``problem`` then says why (an HTTP 401, no
    models).
    """

    url: str
    models: list[str]
    problem: str = ""


def parse_model_ids(payload: Any) -> list[str]:
    """The model ids of a ``/models`` reply, in order and without repeats.

    Reads OpenAI's ``{"data": [...]}`` and the ``{"models": [...]}`` shape,
    with items that are strings or objects with ``id``, ``name`` or
    ``model``.
    """
    items: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload["data"]
        elif isinstance(payload.get("models"), list):
            items = payload["models"]

    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        model_id = ""
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            for key in ("id", "name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    model_id = value.strip()
                    break
        if model_id and model_id not in seen:
            seen.add(model_id)
            names.append(model_id)
    return names


def says_omlx(payload: Any) -> bool:
    """Whether a ``/models`` reply names oMLX as the owner of a model.

    oMLX sets ``owned_by`` to ``omlx`` on every model it lists, also while
    its status route refuses a request without a key.
    """
    items = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(items, list) and any(
        isinstance(item, dict) and str(item.get("owned_by", "")).lower() == "omlx"
        for item in items
    )


def is_omlx(url: str) -> bool:
    """Whether the last model listing of the server at *url* said it is oMLX."""
    return normalize_url(url) in _OMLX_URLS if url else False


def server_name(url: str = "", *, declared: bool = False) -> str:
    """What to call the OpenAI-compatible server at *url*: ``oMLX`` or not.

    *declared* is a config file saying so (``chat_backend = "omlx"``);
    otherwise a model listing of *url* must have said so. Anything else is
    an ``OpenAI-compatible server``.
    """
    return OMLX_NAME if declared or is_omlx(url) else OPENAI_COMPATIBLE_NAME


def forget_servers() -> None:
    """Drop what the model listings said about servers (for tests)."""
    _OMLX_URLS.clear()


def _exception_text(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        return _exception_text(exc.exceptions[0])
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _connect_reason(exc: httpx.TransportError, timeout_s: float) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return f"no answer in {timeout_s:g} s"
    text = str(exc)
    if isinstance(exc, httpx.ConnectError) and (
        "refused" in text.lower() or "all connection attempts failed" in text.lower()
    ):
        return "connection refused"
    return _exception_text(exc)


async def list_models(
    backend_key: str,
    url: str,
    *,
    timeout_s: float = DISCOVERY_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[str]:
    """The models the server at *url* lists; ``[]`` when it lists none.

    *backend_key* is a local backend's key or ``"openai-compatible"``;
    a cloud vendor's key raises ``ValueError``. Any failure raises
    :class:`DiscoveryError`. *transport* replaces the network in tests.
    """
    if backend_key not in _LISTED_BACKENDS:
        raise ValueError(f"cannot list the models of backend '{backend_key}'")
    try:
        base_url = normalize_url(url)
        port = httpx.URL(base_url).port
        if port is not None and not 0 < port < 65536:
            raise DiscoveryError(f"not a URL: {url} (port {port})", reachable=False)
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport,
            follow_redirects=False,
        ) as client:
            response = await client.get("/models")
    except DiscoveryError:
        raise
    except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
        raise DiscoveryError(f"not a URL: {url}", reachable=False) from exc
    except httpx.TransportError as exc:
        raise DiscoveryError(_connect_reason(exc, timeout_s), reachable=False) from exc
    except httpx.HTTPError as exc:  # e.g. a body that cannot be decoded
        raise DiscoveryError(
            f"the reply could not be read ({_exception_text(exc)})", reachable=True
        ) from exc
    except Exception as exc:  # anything else the connection raised
        raise DiscoveryError(_exception_text(exc), reachable=False) from exc
    status = response.status_code
    if status in (401, 403):
        raise DiscoveryError(
            f"HTTP {status}: {NO_API_KEY_HINT}", reachable=True, status_code=status
        )
    if response.is_redirect:  # not followed, as chat requests are not
        location = response.headers.get("location", "")
        raise DiscoveryError(
            (
                f"HTTP {status}: redirected to {location}"
                if location
                else f"HTTP {status}"
            ),
            reachable=True,
            status_code=status,
        )
    if status >= 300:
        raise DiscoveryError(f"HTTP {status}", reachable=True, status_code=status)
    try:
        payload = response.json()
    except ValueError as exc:
        raise DiscoveryError(
            "the reply is not a model list", reachable=True, status_code=status
        ) from exc
    if says_omlx(payload):
        _OMLX_URLS.add(base_url)
    else:
        _OMLX_URLS.discard(base_url)
    return parse_model_ids(payload)


def _ollama_root(url: str) -> str:
    """The server root of *url*: the URL without a trailing slash or ``/v1``."""
    root = url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


async def is_ollama(
    url: str,
    *,
    timeout_s: float = DISCOVERY_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bool:
    """Whether the server at *url* answers like Ollama.

    Ollama answers ``GET <root>/api/version`` with ``{"version": "…"}``;
    *url* may end in ``/v1`` or not. Any other answer -- an HTTP error, a
    redirect, a body without a version, no answer at all -- means no.
    *transport* replaces the network in tests.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            response = await client.get(f"{_ollama_root(url)}/api/version")
        if response.status_code >= 300:  # not followed, as chat requests are not
            return False
        payload = response.json()
    except Exception:  # no answer, a malformed URL, a body that is not JSON
        return False
    version = payload.get("version") if isinstance(payload, dict) else None
    return isinstance(version, str) and bool(version.strip())


LOADED_TIMEOUT_S: Final = 1.0


@dataclass(frozen=True)
class LoadedModels:
    """Which models a server says are in memory, and whether it said anything.

    ``known`` is False when no source answered (an HTTP error such as a 401,
    no route, a timeout); ``ids`` is then empty. An answer that lists none
    in memory is known, with ``ids`` empty.
    """

    ids: frozenset[str] = frozenset()
    known: bool = False
    needs_key: bool = False


def parse_loaded_ids(payload: Any) -> set[str] | None:
    """The ids a status reply says are in memory; None if it is no status reply.

    Reads oMLX's ``{"models": [{"id": …, "loaded": true}]}`` and Ollama's
    ``/api/ps`` list, whose items are all running models (``name`` or
    ``model``). An item with ``loaded`` false is not counted. oMLX reports a
    model under its folder name (``id``) but lists it under its alias
    (``model_alias``), so a loaded model counts under both.
    """
    items = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    ids: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("loaded") is False:
            continue
        ids.update(parse_model_ids({"models": [item]}))
        alias = item.get("model_alias") if isinstance(item, dict) else None
        if isinstance(alias, str) and alias.strip():
            ids.add(alias.strip())
    return ids


async def loaded_models(
    url: str,
    *,
    timeout_s: float = LOADED_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LoadedModels:
    """The models the server at *url* says are in memory, if it says.

    Asks oMLX (``GET <url>/models/status``) and Ollama (``GET <root>/api/ps``)
    at once. Each source is optional: a server without the route, an HTTP
    error (a server with an API key answers 401, as it does to the listing,
    since no key is sent), a timeout or a body that is not a status list
    adds nothing, and when every source is like that the result is not
    ``known``. *transport* replaces the network in tests.
    """

    async def ask(
        client: httpx.AsyncClient, target: str
    ) -> tuple[set[str] | None, bool]:
        try:
            response = await client.get(target)
            if response.status_code in (401, 403):
                return None, True
            if response.status_code >= 300:  # not followed, as chat requests are not
                return None, False
            return parse_loaded_ids(response.json()), False
        except Exception:  # no answer, a malformed URL, a body that is not JSON
            return None, False

    try:
        status_url = f"{normalize_url(url).rstrip('/')}/models/status"
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            found = await asyncio.gather(
                ask(client, status_url), ask(client, f"{_ollama_root(url)}/api/ps")
            )
    except Exception:
        return LoadedModels()
    answers = [ids for ids, _ in found if ids is not None]
    return LoadedModels(
        ids=frozenset().union(*answers),
        known=bool(answers),
        needs_key=not answers and any(refused for _, refused in found),
    )


NO_DEFAULT_HINT: Final = (
    "Type the number of the model to use. A model that is not loaded yet can "
    "take minutes and a lot of memory to load."
)
NO_DEFAULT_MODEL: Final = (
    "several models are served and none is known to be loaded, so there is "
    "no default; answer with a number or a model id"
)


def is_chat_model(model: str) -> bool:
    """Whether *model*'s id does not name an embedding or reranking model."""
    lowered = model.lower()
    return "embed" not in lowered and "rerank" not in lowered


def pick_model(
    models: Sequence[str], loaded: LoadedModels | None = None, *, unique: bool = False
) -> str | None:
    """The model to default to; None when there is no safe default.

    The only chat model, if there is one. Among several, the first one in
    *loaded* (with *unique*, only when exactly one is). Loading a model
    that is not in memory can take minutes and tens of GB, so without a
    known loaded model there is no default. With no chat model, the first
    id.
    """
    chat = [model for model in models if is_chat_model(model)]
    if len(chat) <= 1:
        return chat[0] if chat else (models[0] if models else None)
    running = [model for model in chat if loaded and model in loaded.ids]
    if running and (len(running) == 1 or not unique):
        return running[0]
    return None


# Families that serve embeddings but whose ids do not say "embed".
_EMBEDDING_FAMILIES: Final = ("bge-", "e5-", "gte-")

# The order is a recommendation, not a quality claim: the first family is the
# one the recall measurements used, and each later one is smaller or older.
_EMBEDDING_PREFERENCE: Final = (
    "qwen3-embedding",
    "embeddinggemma",
    "nomic-embed",
    "bge-base",
    "snowflake-arctic-embed",
    "bge-m3",
    "e5-",
    "gte-",
)


def is_embedding_model(model: str) -> bool:
    """Whether *model*'s id names an embedding model.

    An id that says ``embed`` is one, and so is a known embedding family
    whose id does not (``bge-m3``). A reranking id never is, even when it
    also says ``embed``. :func:`is_chat_model` keeps its own rule; it is
    not this function's inverse (``bge-m3`` is both by those rules).
    """
    lowered = model.lower()
    if "rerank" in lowered:
        return False
    return "embed" in lowered or any(
        family in lowered for family in _EMBEDDING_FAMILIES
    )


def embedding_models(models: Sequence[str]) -> list[str]:
    """The ids of *models* that name embedding models, in listing order.

    Reranking and chat ids are left out. Duplicates keep their first
    place. A listing is only a candidate source: the proof that a model
    serves embeddings is an embeddings probe.
    """
    found: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        if is_embedding_model(model):
            found.append(model)
    return found


def _embedding_rank(model: str) -> int:
    lowered = model.lower()
    for rank, family in enumerate(_EMBEDDING_PREFERENCE):
        if family in lowered:
            return rank
    return len(_EMBEDDING_PREFERENCE)


def recommend_embedding(models: Sequence[str]) -> str | None:
    """The embedding candidate to suggest first; None for no candidates.

    Qwen3-Embedding comes first, then embeddinggemma, nomic-embed,
    bge-base, snowflake-arctic-embed, bge-m3, e5 and gte; an unknown
    family comes after those, in listing order. The order is a
    recommendation, not a quality claim.
    """
    if not models:
        return None
    return min(enumerate(models), key=lambda item: (_embedding_rank(item[1]), item[0]))[
        1
    ]


def detect_urls() -> tuple[str, ...]:
    """DETECT_URLS, or the URLs that ``LOCALMELO_DETECT_URLS`` names."""
    value = os.environ.get(DETECT_URLS_ENV)
    if value is None:
        return DETECT_URLS
    return tuple(part.strip() for part in value.split(",") if part.strip())


async def detect_local_server(
    urls: Sequence[str] = DETECT_URLS,
    *,
    timeout_s: float = DETECT_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FoundServer | None:
    """The server of *urls* to offer first; None if none answered.

    In order of preference: the first of *urls*, in order, that lists a
    chat model; then the first that lists any model; then the first that
    answered without a usable list (with ``models`` empty and the
    ``problem``). All URLs are asked at once, each with ``GET /models``
    only.
    """
    results = await asyncio.gather(
        *(
            list_models(
                OPENAI_COMPATIBLE, url, timeout_s=timeout_s, transport=transport
            )
            for url in urls
        ),
        return_exceptions=True,
    )
    listed = [
        (url, result)
        for url, result in zip(urls, results, strict=True)
        if isinstance(result, list) and result
    ]
    for url, models in listed:
        if any(is_chat_model(model) for model in models):
            return FoundServer(url=url, models=models)
    if listed:
        return FoundServer(url=listed[0][0], models=listed[0][1])
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, list):
            return FoundServer(url=url, models=[], problem="it lists no models")
        if isinstance(result, DiscoveryError) and result.reachable:
            return FoundServer(url=url, models=[], problem=str(result))
    return None
