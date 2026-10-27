"""Ask a local model server which models it serves, and find one.

Every request here is ``GET <url>/models``. Nothing is generated, so
listing neither waits for a server to load a model nor competes with runs
on it. Cloud vendors are not listed.
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
            base_url=base_url, timeout=timeout_s, transport=transport
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
    return parse_model_ids(payload)


def is_chat_model(model: str) -> bool:
    """Whether *model*'s id does not name an embedding or reranking model."""
    lowered = model.lower()
    return "embed" not in lowered and "rerank" not in lowered


def pick_model(models: Sequence[str]) -> str | None:
    """The first id that is not an embedding or reranking model, else the first."""
    for model in models:
        if is_chat_model(model):
            return model
    return models[0] if models else None


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
