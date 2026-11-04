"""Ask a model server whether a model reads images (``tool_images = "auto"``).

Three routes are asked in this order, and the first that says yes or no wins:

1. The model's own entry in ``GET /v1/models``, the listing every
   OpenAI-compatible server has. It says yes with ``"vision": true``, an
   ``image`` among ``input_modalities`` or ``vision`` among ``capabilities``,
   and says no only when it says so: ``"vision": false`` or a non-empty
   ``input_modalities`` without ``image``. An entry that says nothing, or none
   for the model, passes.
2. oMLX's ``GET /v1/models/status`` gives each model a ``model_type`` (``vlm``
   or ``llm``) under its folder name (``id``) and its alias (``model_alias``).
3. Ollama's ``POST /api/show`` lists ``vision`` among a model's
   ``capabilities``.

A route that answers 401 or 404, with a body that is not JSON, or with a
shape this module does not know, is not an error: it passes. A server that
cannot be reached ends the questions with no answer (``None``). With no
answer, ``auto`` stays off.
"""

from __future__ import annotations

from typing import Any, Final

import httpx

PROBE_TIMEOUT_S: Final = 2.0


def _names(item: Any, model: str) -> bool:
    return isinstance(item, dict) and model in (item.get("id"), item.get("model_alias"))


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
    items = payload.get("data") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if _names(item, model):
            return entry_verdict(item)
    return None


def omlx_verdict(payload: Any, model: str) -> bool | None:
    """What an oMLX status reply says about *model*: True for ``vlm``, False for ``llm``."""
    items = payload.get("models") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if _names(item, model):
            kind = item.get("model_type")
            return kind == "vlm" if kind in ("vlm", "llm") else None
    return None


def ollama_verdict(payload: Any) -> bool | None:
    """Whether an Ollama ``/api/show`` reply lists ``vision`` among its capabilities."""
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
    return "vision" in capabilities if isinstance(capabilities, list) else None


async def server_takes_images(
    http: httpx.AsyncClient,
    model: str,
    *,
    status: str | None,
    show: str,
    listing: str | None = None,
) -> bool | None:
    """Ask the *listing* route (if given), then oMLX's *status* route (if given), then Ollama's *show* route.

    The first route that says yes or no answers; None if none does or the server
    cannot be reached. Each question waits :data:`PROBE_TIMEOUT_S` at most.
    """
    if not model:
        return None
    routes = (
        ("GET", listing, lambda body: listing_verdict(body, model)),
        ("GET", status, lambda body: omlx_verdict(body, model)),
        ("POST", show, ollama_verdict),
    )
    for method, route, read in routes:
        if route is None:
            continue
        try:
            if method == "GET":
                reply = await http.get(route, timeout=PROBE_TIMEOUT_S)
            else:
                reply = await http.post(
                    route, json={"model": model}, timeout=PROBE_TIMEOUT_S
                )
        except httpx.HTTPError:  # unreachable or too slow: the next route would be too
            return None
        if reply.status_code != 200:
            continue
        try:
            verdict = read(reply.json())
        except ValueError:  # not JSON
            continue
        if verdict is not None:
            return verdict
    return None
