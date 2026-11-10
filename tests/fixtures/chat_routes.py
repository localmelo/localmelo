"""A stub model server's handler that answers the chat routes only.

Before its first run an agent asks the server about the model (its context
length: ``GET /v1/models``, oMLX's status, Ollama's ``/api/ps``). A stub that
answers every request with a scripted chat reply would spend a reply on each of
those questions; :func:`chat_only` answers them 404, so the server says nothing
and the run is fitted as it was.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import httpx

CHAT_PATHS: Final = frozenset({"/v1/chat/completions", "/api/chat"})

Handler = Callable[[httpx.Request], httpx.Response]


def chat_only(handler: Handler) -> Handler:
    """*handler* for the chat routes; 404 for every other request."""

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path not in CHAT_PATHS:
            return httpx.Response(404)
        return handler(request)

    return route
