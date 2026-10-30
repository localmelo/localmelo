"""Ask a model server whether a model reads images (``tool_images = "auto"``).

oMLX's ``GET /v1/models/status`` gives each model a ``model_type`` (``vlm`` or
``llm``) under its folder name (``id``) and its alias (``model_alias``);
Ollama's ``POST /api/show`` lists ``vision`` among a model's ``capabilities``.
Other OpenAI-compatible servers (vLLM, SGLang, MLC) say nothing in their model
listing, so the answer is unknown there and ``auto`` stays off.
"""

from __future__ import annotations

from typing import Any, Final

import httpx

PROBE_TIMEOUT_S: Final = 2.0


def omlx_verdict(payload: Any, model: str) -> bool | None:
    """What an oMLX status reply says about *model*: True for ``vlm``, False for ``llm``."""
    items = payload.get("models") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and model in (
            item.get("id"),
            item.get("model_alias"),
        ):
            kind = item.get("model_type")
            return kind == "vlm" if kind in ("vlm", "llm") else None
    return None


async def server_takes_images(
    http: httpx.AsyncClient, model: str, *, status: str | None, show: str
) -> bool | None:
    """Ask oMLX's *status* route (if given), then Ollama's *show* route; None if neither says."""
    if not model:
        return None
    try:
        if status is not None:
            reply = await http.get(status, timeout=PROBE_TIMEOUT_S)
            if reply.status_code == 200:
                verdict = omlx_verdict(reply.json(), model)
                if verdict is not None:
                    return verdict
        reply = await http.post(show, json={"model": model}, timeout=PROBE_TIMEOUT_S)
        body = reply.json() if reply.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None
    capabilities = body.get("capabilities") if isinstance(body, dict) else None
    return "vision" in capabilities if isinstance(capabilities, list) else None
