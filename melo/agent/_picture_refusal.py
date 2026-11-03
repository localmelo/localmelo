"""Telling a server that refuses a request's pictures from one that refuses it for another reason.

A model that cannot read pictures answers a request that carries them with an error, and
what the error says differs by server. The text may name the pictures (``image``,
``image_url``, ``vision``), or it may be a bare HTTP 400 that says nothing more. The agent
reads it only for a request that really carried pixels (see
:attr:`~localmelo.melo.agent._tool_images.ToolImageStore.sent_pixels`).
"""

from __future__ import annotations

import re
from typing import Final

from localmelo.melo.contracts.providers import ProviderError

# What a server's refusal of pictures usually names.
_NAMES_PICTURES: Final = re.compile(
    r"\b(images?|image_url|pictures?|vision|multi-?modal)\b", re.IGNORECASE
)
# The statuses a server uses for a request body it will not take.
REFUSING_STATUSES: Final = frozenset({400, 415, 422})


def names_pictures(exc: ProviderError) -> bool:
    """Whether *exc* is an error a request would not repeat, and its text names pictures."""
    return not exc.retryable and bool(_NAMES_PICTURES.search(str(exc)))


def is_body_refusal(exc: ProviderError) -> bool:
    """Whether *exc* is the server refusing the request body (not its size, not an outage)."""
    return exc.status_code in REFUSING_STATUSES and not exc.context_overflow
