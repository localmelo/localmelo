"""Reading public web pages without a browser: the ``web_fetch`` tool.

``[tools] web_fetch`` (on unless the config turns it off) gives the chats and
the console's tasks a :class:`WebFetchProvider`; direct runs never get it.
:mod:`.guard` decides which addresses may be reached, :mod:`.fetch` downloads
a page under fixed limits, :mod:`.extract` turns HTML into text, and
:mod:`.tool` frames the result as data for the model.
"""

from __future__ import annotations

from localmelo.support.web.tool import NAME, NOTE, WebFetchProvider

__all__ = ["NAME", "NOTE", "WebFetchProvider"]
