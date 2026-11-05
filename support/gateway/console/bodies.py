"""The request bodies API v2 accepts: unknown keys refused.

Every string of a body is valid UTF-8 by the time a route sees it:
:class:`~.errors.ApiMiddleware` refuses a lone surrogate (``"\\ud800"``, which a
JSON body can carry and Python accepts, but SQLite cannot store) before routing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Body(BaseModel):
    """The base of every write route's request body: see the module's rules."""

    model_config = ConfigDict(extra="forbid")
