"""HTTP plumbing shared by the model providers.

Every failed request becomes a
:class:`~localmelo.melo.contracts.providers.ProviderError`, so the agent can
retry it or report it without knowing about httpx.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from localmelo.melo.contracts.providers import LLM_REQUEST_ATTEMPT, ProviderError

# Statuses for which the same request may succeed when sent again: rate
# limiting, and a gateway or proxy in front of the model server failing.
# Every other error status is final.
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})

# Characters of a response body that an error message quotes.
BODY_PREVIEW_CHARS = 500

# Header of a chat request that carries LLM_REQUEST_ATTEMPT.
ATTEMPT_HEADER = "X-Localmelo-Attempt"


def attempt_headers() -> dict[str, str]:
    """Headers naming the attempt of the LLM request being sent."""
    return {ATTEMPT_HEADER: str(LLM_REQUEST_ATTEMPT.get())}


def _body_preview(response: httpx.Response) -> str:
    return response.text[:BODY_PREVIEW_CHARS] or "(empty body)"


def _exception_text(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def malformed_body(response: httpx.Response, path: str, problem: str) -> ProviderError:
    """The error for a response to ``POST <path>`` that lacks what it needs.

    It is not retryable: the server answered, and would answer the same
    request the same way.
    """
    return ProviderError(
        f"POST {path} returned HTTP {response.status_code} with a malformed "
        f"body ({problem}): {_body_preview(response)}",
        status_code=response.status_code,
        retryable=False,
    )


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
) -> tuple[httpx.Response, Any]:
    """POST *payload* to *path* and return the response and its JSON body.

    Every failure raises :class:`ProviderError`. A timeout or another
    transport error is retryable, and so is an HTTP error status in
    :data:`RETRYABLE_STATUS_CODES`; any other error status, any other
    request error and a body that is not JSON are not. The message names
    the request and, when a response came back, gives its status code and
    the first :data:`BODY_PREVIEW_CHARS` characters of its body.
    """
    try:
        response = await client.post(path, json=payload, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise ProviderError(
            f"POST {path} failed with HTTP {status}: {_body_preview(exc.response)}",
            status_code=status,
            retryable=status in RETRYABLE_STATUS_CODES,
        ) from exc
    except httpx.TransportError as exc:  # timeouts included
        raise ProviderError(
            f"POST {path} failed: {_exception_text(exc)}", retryable=True
        ) from exc
    except httpx.RequestError as exc:  # e.g. a body that could not be decoded
        raise ProviderError(f"POST {path} failed: {_exception_text(exc)}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise malformed_body(response, path, "not JSON") from exc
    return response, data
