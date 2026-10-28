"""Incremental UTF-8 server-sent events, independent of transport chunks."""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterator

from localmelo.melo.contracts.providers import ProviderError

# Bound an unterminated line or event from a broken server.
_MAX_EVENT_CHARS = 1_048_576


async def sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    buffer = ""
    data: list[str] = []
    size = 0
    try:
        async for chunk in chunks:
            buffer += decoder.decode(chunk)
            while True:
                # A trailing CR may be the first half of a CRLF.
                positions = [i for c in ("\n", "\r") if (i := buffer.find(c)) >= 0]
                if not positions:
                    break
                pos = min(positions)
                if buffer[pos] == "\r" and pos + 1 == len(buffer):
                    break
                line = buffer[:pos]
                end = pos + (2 if buffer[pos : pos + 2] == "\r\n" else 1)
                buffer = buffer[end:]
                if not line:
                    if data:
                        yield "\n".join(data)
                    data, size = [], 0
                elif not line.startswith(":"):
                    field, sep, value = line.partition(":")
                    if field == "data":
                        value = value.removeprefix(" ") if sep else ""
                        size += len(value) + 1
                        if size > _MAX_EVENT_CHARS:
                            raise ProviderError("SSE event exceeds size limit")
                        data.append(value)
            if len(buffer) + size > _MAX_EVENT_CHARS:
                raise ProviderError("SSE event exceeds size limit")
        buffer += decoder.decode(b"", final=True)
        if buffer == "\r" and data:
            yield "\n".join(data)
    except UnicodeDecodeError as exc:
        raise ProviderError("SSE response contains invalid UTF-8") from exc
    # SSE dispatches only on a blank line, not on an interrupted frame.
