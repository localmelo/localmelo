"""Convert MCP ``tools/call`` results into executor ToolOutput."""

from __future__ import annotations

import json
from typing import Any

from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import ImagePart


def tool_result_to_output(result: Any) -> ToolOutput:
    """Map a ``CallToolResult`` to text plus images.

    Text parts are joined with newlines. Image parts (and image blobs in
    embedded resources) become :class:`ImagePart`; other binary content is
    replaced by a short text note. ``structuredContent`` is used as text
    only when there is no other text. ``isError`` is kept.
    """
    if not isinstance(result, dict):
        return ToolOutput(
            text=f"[invalid MCP tool result: {type(result).__name__}]", is_error=True
        )
    texts: list[str] = []
    images: list[ImagePart] = []
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            _convert_item(item, texts, images)
    structured = result.get("structuredContent")
    if not texts and structured is not None:
        texts.append(json.dumps(structured, ensure_ascii=False))
    return ToolOutput(
        text="\n".join(texts), images=images, is_error=result.get("isError") is True
    )


def _convert_item(item: Any, texts: list[str], images: list[ImagePart]) -> None:
    if not isinstance(item, dict):
        return
    kind = item.get("type")
    if kind == "text":
        texts.append(str(item.get("text", "")))
    elif kind == "image":
        data, mime = item.get("data"), item.get("mimeType")
        if isinstance(data, str) and isinstance(mime, str):
            images.append(ImagePart(mime_type=mime, data=data))
        else:
            texts.append("[malformed image content]")
    elif kind == "resource":
        resource = item.get("resource")
        resource = resource if isinstance(resource, dict) else {}
        uri = resource.get("uri", "")
        mime = resource.get("mimeType")
        if isinstance(resource.get("text"), str):
            texts.append(resource["text"])
        elif (
            isinstance(resource.get("blob"), str)
            and isinstance(mime, str)
            and mime.startswith("image/")
        ):
            images.append(ImagePart(mime_type=mime, data=resource["blob"]))
        else:
            texts.append(f"[binary resource {uri} omitted]")
    elif kind == "resource_link":
        name = item.get("name") or ""
        texts.append(
            f"[resource link: {name} {item.get('uri', '')}]".replace("  ", " ")
        )
    elif kind == "audio":
        texts.append(
            f"[audio content omitted ({item.get('mimeType', 'unknown type')})]"
        )
    else:
        texts.append(f"[unsupported MCP content type: {kind!r}]")
