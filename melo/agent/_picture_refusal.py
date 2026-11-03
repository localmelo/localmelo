"""Telling a server that refuses a request's pictures from one that refuses it for another reason.

A model that cannot read pictures answers a request that carries them with an error that
says so: a picture or image input "is not supported", an ``image_url`` is an "unknown
variant", the model "has no vision". Nothing else is taken for it. A bare HTTP 400, a
complaint about a parameter, a picture that could not be decoded, a request that is too
long for the model's window, an outage and a refused key all come with a request that
carried pictures too, and none of them says the model cannot read them. The agent reads
the error only for a request that really carried pixels (see
:attr:`~localmelo.melo.agent._tool_images.ToolImageStore.sent_pixels`).
"""

from __future__ import annotations

import re
from typing import Final

from localmelo.melo.contracts.providers import ProviderError

# The statuses a server uses for a request body it will not take.
_REFUSING_STATUSES: Final = frozenset({400, 415, 422})
# The words a refusal of pictures uses for what it refuses...
_PICTURES: Final = re.compile(
    r"\b(images?|image_url|pictures?|vision|multi-?modal|mmproj)\b", re.IGNORECASE
)
# ...and for the refusal itself.
_UNSUPPORTED: Final = re.compile(
    r"(?:\bnot|n't|\bno|\bun)[\s-]*support"
    r"|\bonly\s+supported\b"
    r"|\bunknown\s+variant\b"
    r"|\bat\s+most\s+0\b"
    r"|\btext[\s-]only\b"
    r"|\bno\s+(?:vision|multi-?modal)\b"
    r"|\b(?:cannot|can't|unable\s+to)\s+(?:read|see|process|accept|handle|take)\b"
    r"|\bdoes\s+not\s+accept\b"
    r"|\bnot\s+a\s+(?:vision|multi-?modal)\b"
    r"|\bmissing\s+(?:data|the\s+\w+)\s+required\b",
    re.IGNORECASE,
)
# What a request too long for the model's window says; a picture is part of its length.
_TOO_LONG: Final = re.compile(
    r"context|too\s+(?:long|large|many|big)|exceed|token\s+limit|max(?:imum)?\s+length",
    re.IGNORECASE,
)


def names_pictures(exc: ProviderError) -> bool:
    """Whether *exc* is a server saying that it cannot take the pictures of the request.

    A length error comes first and is never one: the pictures made the request long,
    and a shorter one would pass. Then only a refusal of the body counts (HTTP 400, 415
    or 422, or no status at all from a provider that runs in the process), one that a
    new request would not undo, and its words must name pictures *and* say they are not
    taken. ``detail`` is read with the message: a streamed error keeps the server's
    reason there.
    """
    if exc.context_overflow or exc.retryable:
        return False
    if exc.status_code is not None and exc.status_code not in _REFUSING_STATUSES:
        return False
    text = f"{exc} {exc.detail}"
    if _TOO_LONG.search(text):
        return False
    return bool(_PICTURES.search(text) and _UNSUPPORTED.search(text))
