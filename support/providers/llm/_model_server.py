"""What a model server says about a model: whether it reads images, and its context length.

Four routes, each asked only by the questions that read it, in this order:

1. The model's own entry in ``GET /v1/models``, the listing every
   OpenAI-compatible server has.

   - Images: yes with ``"vision": true``, an ``image`` among
     ``input_modalities`` or ``vision`` among ``capabilities``; no only when it
     says so: ``"vision": false`` or a non-empty ``input_modalities`` without
     ``image``.
   - Context: ``max_model_len`` (the length the server runs the model with on
     vLLM, SGLang and oMLX), else ``context_length``, ``context_window`` or
     ``max_context_length`` (the names hosted services use).

2. oMLX's ``GET /v1/models/status`` lists each model under its folder name
   (``id``) and its alias (``model_alias``): images from its ``model_type``
   (``vlm`` or ``llm``), context from its ``max_context_window``.
3. Ollama's ``POST /api/show`` lists ``vision`` among a model's
   ``capabilities`` (images only: the context length it gives is the model's
   maximum, not what the server runs it with).
4. Ollama's ``GET /api/ps`` gives the ``context_length`` a running model was
   loaded with (context only). A model that is not loaded is not listed there,
   and its length is unknown.

A route that answers 401 or 404, with a body that is not JSON, or with a
shape this module does not know, is not an error: it passes. A server that
cannot be reached ends the question with no answer (``None``). The first route
that answers decides.

A provider keeps one :data:`Replies` and passes it to both questions, so each
route is fetched once for both: a reply the server gave is remembered, and an
unreachable server is asked again by the next question.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any, Final, TypeVar

import httpx

PROBE_TIMEOUT_S: Final = 2.0

# The fields of a model's ``GET /v1/models`` entry that give its context length,
# in the order they are read.
LISTING_LENGTH_FIELDS: Final = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)

# The replies of one server, by route and (for a POST) model: the JSON body, or
# _PASS for a reply that says nothing.
Replies = dict[tuple[str, str], Any]
_PASS: Final = object()

T = TypeVar("T")
Question = tuple[str, str | None, Callable[[Any], T | None]]


def _names(item: Any, model: str) -> bool:
    return isinstance(item, dict) and model in (item.get("id"), item.get("model_alias"))


def _entry_for(payload: Any, key: str, model: str) -> dict[str, Any] | None:
    items = payload.get(key) if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if _names(item, model):
            assert isinstance(item, dict)
            return item
    return None


def as_length(value: Any) -> int | None:
    """*value* when it is a positive int (not a bool), else None."""
    return value if type(value) is int and value > 0 else None


def entry_verdict(entry: dict[str, Any]) -> bool | None:
    """What one model entry of ``GET /v1/models`` says about images, else None."""
    vision = entry.get("vision")
    modalities = entry.get("input_modalities")
    capabilities = entry.get("capabilities")
    if (
        vision is True
        or (isinstance(modalities, list) and "image" in modalities)
        or (isinstance(capabilities, list) and "vision" in capabilities)
    ):
        return True
    if vision is False or (isinstance(modalities, list) and modalities):
        return False
    return None


def listing_verdict(payload: Any, model: str) -> bool | None:
    """What the ``GET /v1/models`` reply says about *model*'s own entry."""
    entry = _entry_for(payload, "data", model)
    return None if entry is None else entry_verdict(entry)


def listing_length(payload: Any, model: str) -> int | None:
    """The context length *model*'s own ``GET /v1/models`` entry gives, else None."""
    entry = _entry_for(payload, "data", model) or {}
    for name in LISTING_LENGTH_FIELDS:
        if (length := as_length(entry.get(name))) is not None:
            return length
    return None


def omlx_verdict(payload: Any, model: str) -> bool | None:
    """What an oMLX status reply says about *model*: True for ``vlm``, False for ``llm``."""
    entry = _entry_for(payload, "models", model)
    kind = entry.get("model_type") if entry is not None else None
    return kind == "vlm" if kind in ("vlm", "llm") else None


def omlx_length(payload: Any, model: str) -> int | None:
    """The ``max_context_window`` an oMLX status reply gives *model*, else None."""
    entry = _entry_for(payload, "models", model) or {}
    return as_length(entry.get("max_context_window"))


def ollama_verdict(payload: Any) -> bool | None:
    """Whether an Ollama ``/api/show`` reply lists ``vision`` among its capabilities."""
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
    return "vision" in capabilities if isinstance(capabilities, list) else None


def ollama_running_length(payload: Any, model: str) -> int | None:
    """The ``context_length`` Ollama's ``/api/ps`` gives the running *model*, else None.

    A name without a tag is Ollama's ``:latest``.
    """
    names = {model} if ":" in model else {model, f"{model}:latest"}
    items = payload.get("models") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and names & {item.get("name"), item.get("model")}:
            return as_length(item.get("context_length"))
    return None


async def _ask(
    http: httpx.AsyncClient,
    model: str,
    questions: Sequence[Question[T]],
    replies: Replies | None,
) -> T | None:
    """The first answer of *questions* (method, route, reader); None if none answers."""
    if not model:
        return None
    for method, route, read in questions:
        if route is None:
            continue
        key = (route, model if method == "POST" else "")
        if replies is not None and key in replies:
            body = replies[key]
        else:
            try:
                if method == "GET":
                    reply = await http.get(route, timeout=PROBE_TIMEOUT_S)
                else:
                    reply = await http.post(
                        route, json={"model": model}, timeout=PROBE_TIMEOUT_S
                    )
            except (
                httpx.HTTPError
            ):  # unreachable or too slow: the next route would be too
                return None
            body = _PASS
            if reply.status_code == 200:
                with contextlib.suppress(ValueError):  # not JSON: it passes
                    body = reply.json()
            if replies is not None:
                replies[key] = body
        if body is _PASS:
            continue
        answer = read(body)
        if answer is not None:
            return answer
    return None


async def server_takes_images(
    http: httpx.AsyncClient,
    model: str,
    *,
    status: str | None,
    show: str | None,
    listing: str | None = None,
    replies: Replies | None = None,
) -> bool | None:
    """Ask the *listing* route (if given), then oMLX's *status*, then Ollama's *show*.

    The first route that says yes or no answers; None if none does or the server
    cannot be reached. Each question waits :data:`PROBE_TIMEOUT_S` at most.
    """
    questions: tuple[Question[bool], ...] = (
        ("GET", listing, lambda body: listing_verdict(body, model)),
        ("GET", status, lambda body: omlx_verdict(body, model)),
        ("POST", show, ollama_verdict),
    )
    return await _ask(http, model, questions, replies)


async def server_context_length(
    http: httpx.AsyncClient,
    model: str,
    *,
    listing: str | None,
    status: str | None = None,
    running: str | None = None,
    replies: Replies | None = None,
) -> int | None:
    """Ask the *listing* route, then oMLX's *status*, then Ollama's *running* (``/api/ps``).

    The first route that gives a length answers, in tokens; None if none does or
    the server cannot be reached. Each question waits :data:`PROBE_TIMEOUT_S` at most.
    """
    questions: tuple[Question[int], ...] = (
        ("GET", listing, lambda body: listing_length(body, model)),
        ("GET", status, lambda body: omlx_length(body, model)),
        ("GET", running, lambda body: ollama_running_length(body, model)),
    )
    return await _ask(http, model, questions, replies)
